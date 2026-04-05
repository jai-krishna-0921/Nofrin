"""
agents/worker.py

Worker node: Exa search + LLM evidence extraction.

Receives: WorkerInput (via Send() dispatch from supervisor dispatcher)
Returns:  {"worker_results": [WorkerResult(...)]}

The worker_results field uses operator.add reducer, so each parallel
worker's single-item list accumulates into the full list automatically.

AsyncExa is used directly — no asyncio.to_thread() wrapper needed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, cast

import httpx
from exa_py import AsyncExa
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.runtime import Runtime
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graph.context import NofrinContext
from graph.progress import (
    worker_done,
    worker_exa_results,
    worker_exa_search,
    worker_url_fail,
    worker_url_ok,
)
from graph.state import Evidence, SourceType, WorkerInput, WorkerResult
from graph.utils import AgentParseError, parse_agent_json


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is a ChatAnthropic instance (supports cache_control)."""
    try:
        from langchain_anthropic import ChatAnthropic

        return isinstance(llm, ChatAnthropic)
    except ImportError:
        return False


logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "worker_v1.txt"

# Limit concurrent LLM calls to avoid Groq / OpenRouter rate-limit 429s.
# All evidence extractions across all parallel workers share this semaphore.
_LLM_SEMAPHORE = asyncio.Semaphore(3)


# ---------------------------------------------------------------------------
# Two-level deserialization
# ---------------------------------------------------------------------------
# parse_agent_json() calls schema_class(**data) and cannot recursively
# convert nested structures. _RawEvidenceExtraction is a flat TypedDict
# that parse_agent_json() can instantiate directly.


class _RawEvidenceExtraction(TypedDict):
    """Flat TypedDict parse_agent_json() can instantiate directly."""

    claim: str
    supporting_chunks: list[str]
    confidence: float
    contradiction_score: float


# ---------------------------------------------------------------------------
# Exa parameter builders
# ---------------------------------------------------------------------------


def _build_exa_params_web(sub_query: str) -> dict[str, object]:  # noqa: ARG001
    """Build Exa params for web: neural search, text=True, no date filter."""
    return {
        "num_results": 10,
        "type": "neural",
        "text": True,
    }


def _build_exa_params_academic(sub_query: str) -> dict[str, object]:  # noqa: ARG001
    """Build Exa params for academic: category='research paper', highlights enabled."""
    return {
        "num_results": 10,
        "type": "neural",
        "category": "research paper",
        "highlights": {"max_characters": 2000},
        "text": True,
    }


def _build_exa_params_news(sub_query: str) -> dict[str, object]:  # noqa: ARG001
    """Build Exa params for news: neural, start_published_date=today-90d."""
    start_date = (date.today() - timedelta(days=90)).isoformat()
    return {
        "num_results": 10,
        "type": "neural",
        "start_published_date": start_date,
        "text": True,
    }


def _get_exa_params(sub_query: str, source_type: SourceType) -> dict[str, object]:
    """Route to the correct parameter builder.

    Args:
        sub_query: The search query string.
        source_type: One of 'web', 'academic', 'news'.

    Returns:
        Dict of keyword arguments for exa.search_and_contents() (excluding query).

    Raises:
        ValueError: If source_type is not a valid SourceType.
    """
    if source_type == "web":
        return _build_exa_params_web(sub_query)
    if source_type == "academic":
        return _build_exa_params_academic(sub_query)
    if source_type == "news":
        return _build_exa_params_news(sub_query)
    raise ValueError(
        f"Invalid source_type '{source_type}'. Must be one of: academic, news, web"
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    """Load worker prompt from prompts/worker_v1.txt.

    Returns:
        Prompt template string containing placeholder keys.

    Raises:
        FileNotFoundError: If the prompt file does not exist at PROMPT_PATH.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Worker prompt not found at {PROMPT_PATH}. "
            "Ensure prompts/worker_v1.txt exists in the project root."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_extraction_messages(
    sub_query: str,
    source_url: str,
    source_title: str,
    result_text: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build the message list for an evidence extraction LLM call.

    Args:
        sub_query: The sub-query this result was retrieved for.
        source_url: URL of the Exa result.
        source_title: Title of the Exa result.
        result_text: Text or joined highlights from the Exa result.
        prompt_template: Loaded prompt template string.
        use_cache_control: When True (Anthropic only), the static prompt
            instructions are placed in a cached block and the dynamic
            per-result content (source info + text) in a separate uncached
            block. False returns a plain string SystemMessage.

    Returns:
        List containing a single SystemMessage.
    """
    if use_cache_control:
        # Static instructions cached; per-result dynamic context not cached.
        dynamic = (
            f"Sub-query: {sub_query}\n"
            f"Source title: {source_title}\n"
            f"Source URL: {source_url}\n\n"
            f"Text to analyze:\n{result_text}"
        )
        content_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_template,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic,
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = (
        prompt_template.replace("{{sub_query}}", sub_query)
        .replace("{{source_url}}", source_url)
        .replace("{{source_title}}", source_title)
        .replace("{{result_text}}", result_text)
    )
    return [SystemMessage(content=filled)]


# ---------------------------------------------------------------------------
# Exa search (with retry)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    reraise=True,
)
async def _search_exa(
    exa_client: AsyncExa,
    sub_query: str,
    source_type: SourceType,
) -> list[object]:
    """Execute Exa search with source-type-specific params.

    AsyncExa.search_and_contents() is natively async — awaited directly.

    Args:
        exa_client: Pre-configured AsyncExa client from NofrinContext.
        sub_query: The query string to search.
        source_type: Determines which parameter builder is used.

    Returns:
        List of Exa result objects (may be empty — not an error).

    Retries on HTTP errors and timeouts (3×, exponential backoff 1–10s).
    """
    params = _get_exa_params(sub_query, source_type)
    response = await exa_client.search_and_contents(sub_query, **params)
    results = list(response.results)
    return results


# ---------------------------------------------------------------------------
# LLM extraction (with retry)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (AgentParseError, httpx.HTTPStatusError, httpx.TimeoutException)
    ),
    reraise=True,
)
async def _extract_evidence_from_result(
    exa_result: object,
    sub_query: str,
    llm: BaseChatModel,
    prompt_template: str,
) -> tuple[Evidence, int]:
    """Extract Evidence from one Exa result using LLM.

    Returns: (Evidence dataclass, tokens_used int).

    Metadata (source_url, source_title, published_date) is taken directly
    from exa_result attributes — never hallucinated by the LLM.

    Text priority: highlights (joined) → text[:4000].
    None guard: if both highlights and text are None/empty, returns an
    Evidence with claim="" so _extract_all_evidence can filter it out.

    Confidence and contradiction_score are clamped to [0.0, 1.0].

    Retries on AgentParseError, httpx errors (3×, exponential backoff).
    """
    source_url: str = str(getattr(exa_result, "url", "") or "")
    source_title: str = str(getattr(exa_result, "title", "") or "")
    published_date: str | None = getattr(exa_result, "published_date", None)

    highlights_raw = getattr(exa_result, "highlights", None)
    text_raw = getattr(exa_result, "text", None)

    if isinstance(highlights_raw, list) and highlights_raw:
        result_text = " ".join(str(h) for h in highlights_raw)
    elif isinstance(text_raw, str) and text_raw:
        result_text = text_raw[:4000]
    else:
        # No content available — return empty claim to be filtered downstream
        return (
            Evidence(
                claim="",
                supporting_chunks=[],
                source_url=source_url,
                source_title=source_title,
                published_date=published_date,
                confidence=0.0,
                contradiction_score=0.0,
            ),
            0,
        )

    messages = _build_extraction_messages(
        sub_query=sub_query,
        source_url=source_url,
        source_title=source_title,
        result_text=result_text,
        prompt_template=prompt_template,
        use_cache_control=_is_anthropic(llm),
    )

    async with _LLM_SEMAPHORE:
        response = await llm.ainvoke(messages)
    raw: str = str(response.content)

    tokens_used: int = 0
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        tokens_used = int(usage.get("total_tokens", 0))

    raw_extraction: _RawEvidenceExtraction = parse_agent_json(
        raw, _RawEvidenceExtraction
    )

    confidence = min(1.0, max(0.0, float(raw_extraction["confidence"])))
    contradiction_score = min(
        1.0, max(0.0, float(raw_extraction["contradiction_score"]))
    )

    return (
        Evidence(
            claim=raw_extraction["claim"],
            supporting_chunks=list(raw_extraction["supporting_chunks"]),
            source_url=source_url,
            source_title=source_title,
            published_date=published_date,
            confidence=confidence,
            contradiction_score=contradiction_score,
        ),
        tokens_used,
    )


async def _extract_all_evidence(
    exa_results: list[object],
    sub_query: str,
    llm: BaseChatModel,
    worker_id: str = "",
) -> tuple[list[Evidence], int]:
    """Process all Exa results in parallel via asyncio.gather().

    Uses return_exceptions=True so a single extraction failure does not
    cancel other concurrent tasks. Failed tasks are caught and logged;
    the corresponding result is skipped.

    Filters out: failed extractions and empty claims (claim == "").

    Args:
        exa_results: List of Exa result objects from _search_exa.
        sub_query: The sub-query these results were retrieved for.
        llm: Pre-configured LLM client from NofrinContext.
        worker_id: Worker identifier for progress logging.

    Returns:
        Tuple of (list[Evidence], total_tokens_used).
    """
    if not exa_results:
        return [], 0

    worker_exa_results(worker_id, len(exa_results))
    prompt_template = _load_prompt()

    coros = [
        _extract_evidence_from_result(result, sub_query, llm, prompt_template)
        for result in exa_results
    ]

    raw_outcomes = await asyncio.gather(*coros, return_exceptions=True)
    outcomes = cast(list[tuple[Evidence, int] | BaseException], list(raw_outcomes))

    evidence_items: list[Evidence] = []
    total_tokens = 0

    for i, outcome in enumerate(outcomes):
        url = str(getattr(exa_results[i], "url", "") or "")
        if isinstance(outcome, BaseException):
            logger.warning(
                "Evidence extraction failed for result %d of sub-query '%s': %s",
                i,
                sub_query,
                outcome,
            )
            worker_url_fail(worker_id, url, str(outcome))
            continue
        evidence, tokens = outcome
        if evidence.claim:  # filter empty-claim sentinel
            worker_url_ok(worker_id, url, evidence.claim)
            evidence_items.append(evidence)
        total_tokens += tokens

    return evidence_items, total_tokens


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _exa_result_to_dict(result: object) -> dict[str, object]:
    """Convert an Exa result to a plain dict for raw_search_results storage."""
    return {
        "url": getattr(result, "url", None),
        "title": getattr(result, "title", None),
        "id": getattr(result, "id", None),
        "score": getattr(result, "score", None),
        "published_date": getattr(result, "published_date", None),
    }


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def worker_node(
    state: WorkerInput,
    runtime: Runtime[NofrinContext],
) -> dict[str, list[WorkerResult]]:
    """LangGraph node: Exa search + LLM evidence extraction.

    Receives WorkerInput via Send() — not the full ResearchAgentState.
    Returns {"worker_results": [single WorkerResult]}.
    The operator.add reducer accumulates results from all parallel workers.

    Args:
        state: WorkerInput (worker_id, sub_query, source_type).
        runtime: Injected context containing llm_worker and exa_client.

    Returns:
        {"worker_results": [WorkerResult(...)]}

    Raises:
        ValueError: If state["source_type"] is not a valid SourceType.
    """
    worker_id: str = state["worker_id"]
    sub_query: str = state["sub_query"]
    source_type: SourceType = state["source_type"]

    # Validate source_type before any API call
    _get_exa_params(sub_query, source_type)

    exa_client: AsyncExa = runtime.context.exa_client
    llm: BaseChatModel = runtime.context.llm_worker

    worker_exa_search(worker_id, sub_query, str(source_type))
    exa_results = await _search_exa(exa_client, sub_query, source_type)
    raw_search_results = [_exa_result_to_dict(r) for r in exa_results]

    evidence_items, tokens_used = await _extract_all_evidence(
        exa_results, sub_query, llm, worker_id
    )
    worker_done(worker_id, len(evidence_items), tokens_used)

    return {
        "worker_results": [
            WorkerResult(
                worker_id=worker_id,
                sub_query=sub_query,
                source_type=source_type,
                evidence_items=evidence_items,
                raw_search_results=raw_search_results,
                tokens_used=tokens_used,
            )
        ]
    }


__all__ = ["worker_node"]

"""
agents/supervisor.py

Supervisor node: intent classification and query decomposition.

Reads:  state["user_query"]
Writes: state["intent_type"], state["sub_queries"], state["source_routing"]

The LLM is injected via Runtime[NofrinContext] — not instantiated here.
This keeps the node fully testable and provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import httpx
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
from graph.progress import supervisor_done, supervisor_start
from graph.state import IntentType, ResearchAgentState, SourceType
from graph.utils import AgentParseError, parse_agent_json


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is a ChatAnthropic instance (supports cache_control).

    Used to decide whether to add cache_control content blocks to messages.
    Wraps the import so non-Anthropic deployments don't need langchain_anthropic
    installed (though it is always present in our requirements.txt).
    """
    try:
        from langchain_anthropic import ChatAnthropic

        return isinstance(llm, ChatAnthropic)
    except ImportError:
        return False


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor_v1.txt"

_VALID_INTENT_TYPES: frozenset[str] = frozenset(
    {"exploratory", "comparative", "adversarial", "factual"}
)
_VALID_SOURCE_TYPES: frozenset[str] = frozenset({"web", "academic", "news"})

# ---------------------------------------------------------------------------
# Two-level deserialization
# ---------------------------------------------------------------------------
# parse_agent_json() calls schema_class(**data) and cannot recursively
# convert nested dicts to dataclasses. We parse into flat TypedDicts first,
# then convert to typed dataclasses manually.


class _RawSubQuery(TypedDict):
    """Raw dict shape of one sub-query as returned by the LLM."""

    query: str
    source_type: str
    rationale: str


class _RawSupervisorOutput(TypedDict):
    """Flat TypedDict that parse_agent_json() can instantiate directly."""

    intent_type: str
    sub_queries: list[_RawSubQuery]


@dataclass
class SubQueryItem:
    """One sub-query from the supervisor LLM output — fully typed."""

    query: str
    source_type: SourceType
    rationale: str


@dataclass
class SupervisorOutput:
    """Typed supervisor output after nested conversion.

    Internal schema only — not stored in state directly.
    """

    intent_type: IntentType
    sub_queries: list[SubQueryItem]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    """Load supervisor prompt from prompts/supervisor_v1.txt.

    Returns:
        Prompt template string containing {{user_query}} placeholder.

    Raises:
        FileNotFoundError: If the prompt file does not exist at PROMPT_PATH.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Supervisor prompt not found at {PROMPT_PATH}. "
            "Ensure prompts/supervisor_v1.txt exists in the project root."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def _build_messages(
    user_query: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build the message list for the supervisor LLM call.

    Args:
        user_query: The raw user research question.
        prompt_template: The loaded prompt template string.
        use_cache_control: When True (Anthropic provider only), the static
            prompt template is placed in a cached content block and the
            dynamic user query in a separate uncached block. For all other
            providers pass False (default) — a plain string SystemMessage
            is returned instead.

    Returns:
        List containing a single SystemMessage.
    """
    if use_cache_control:
        # Anthropic prompt caching: static instructions cached, dynamic query not.
        # The template is sent as-is (with {{user_query}} placeholder visible);
        # the actual query follows in the next uncached block.
        content_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_template,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": user_query,
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = prompt_template.replace("{{user_query}}", user_query)
    return [SystemMessage(content=filled)]


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_output(output: SupervisorOutput, research_mode: str = "research") -> None:
    """Validate parsed supervisor output for structural correctness.

    Checks:
    - fast mode: exactly 3 sub-queries
    - research mode: 3 <= len(sub_queries) <= 5
    - All source_type values are valid SourceType literals
    - No duplicate query strings
    - No empty query strings

    Args:
        output: The parsed SupervisorOutput to validate.
        research_mode: "fast" or "research" — controls query count rule.

    Raises:
        AgentParseError: If any validation rule is violated.
    """
    count = len(output.sub_queries)
    if research_mode == "fast":
        if count != 3:
            raise AgentParseError(
                f"Fast mode requires exactly 3 sub-queries; got {count}."
            )
    else:
        if count < 3 or count > 5:
            raise AgentParseError(
                f"Supervisor returned {count} sub-queries; expected 3–5."
            )

    queries: list[str] = []
    for sq in output.sub_queries:
        if not sq.query.strip():
            raise AgentParseError("Supervisor returned an empty query string.")
        if sq.source_type not in _VALID_SOURCE_TYPES:
            raise AgentParseError(
                f"Invalid source_type '{sq.source_type}'. "
                f"Must be one of: {sorted(_VALID_SOURCE_TYPES)}"
            )
        queries.append(sq.query)

    if len(queries) != len(set(queries)):
        raise AgentParseError(
            "Supervisor returned duplicate sub-queries. All queries must be unique."
        )


# ---------------------------------------------------------------------------
# LLM call (with retry)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (AgentParseError, httpx.HTTPStatusError, httpx.TimeoutException)
    ),
    reraise=True,
)
async def _call_llm(
    user_query: str,
    llm: BaseChatModel,
    research_mode: str = "research",
) -> SupervisorOutput:
    """Call the LLM and parse the response into a typed SupervisorOutput.

    Retries up to 3 times (exponential backoff 1–10s) on:
    - AgentParseError  (bad JSON or validation failure)
    - httpx.HTTPStatusError  (4xx/5xx from the provider)
    - httpx.TimeoutException  (network timeout)

    Parse strategy:
      1. parse_agent_json(raw, _RawSupervisorOutput) → flat TypedDict
      2. Convert each sub_queries[i] dict → SubQueryItem dataclass
      3. _validate_output() checks counts, literals, and duplicates

    Args:
        user_query: The user's research question.
        llm: Pre-configured LLM client injected from NofrinContext.

    Returns:
        Parsed and validated SupervisorOutput.

    Raises:
        AgentParseError: After 3 failed parse/validation attempts.
    """
    prompt_template = _load_prompt()
    messages = _build_messages(user_query, prompt_template, _is_anthropic(llm))

    response = await llm.ainvoke(messages)
    raw: str = str(response.content)

    raw_output: _RawSupervisorOutput = parse_agent_json(raw, _RawSupervisorOutput)

    if raw_output["intent_type"] not in _VALID_INTENT_TYPES:
        raise AgentParseError(
            f"Invalid intent_type '{raw_output['intent_type']}'. "
            f"Must be one of: {sorted(_VALID_INTENT_TYPES)}"
        )

    sub_queries = [
        SubQueryItem(
            query=sq["query"],
            source_type=sq["source_type"],  # type: ignore[arg-type]
            rationale=sq["rationale"],
        )
        for sq in raw_output["sub_queries"]
    ]

    output = SupervisorOutput(
        intent_type=raw_output["intent_type"],  # type: ignore[arg-type]
        sub_queries=sub_queries,
    )
    _validate_output(output, research_mode)
    return output


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def supervisor_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, IntentType | list[str] | dict[str, SourceType]]:
    """LangGraph node: classify intent and decompose query into sub-queries.

    Entry point of the research pipeline. Reads state["user_query"],
    calls the LLM, and returns a partial state update dict.
    LangGraph merges this dict into the full ResearchAgentState automatically.

    The LLM is taken from runtime.context.llm_supervisor — configured once
    at graph invocation time, not instantiated per-call.

    Args:
        state: Current ResearchAgentState (only user_query is read).
        runtime: Injected runtime context containing the pre-built LLM client.

    Returns:
        Partial state update:
            {
                "intent_type": IntentType,
                "sub_queries": list[str],
                "source_routing": dict[str, SourceType],
            }

    Raises:
        ValueError: If state["user_query"] is empty.
        AgentParseError: If LLM response cannot be parsed after 3 retries.
    """
    user_query = state["user_query"].strip()
    if not user_query:
        raise ValueError("state['user_query'] must not be empty.")

    research_mode: str = str(state.get("research_mode", "research"))
    supervisor_start(user_query)
    output = await _call_llm(user_query, runtime.context.llm_supervisor, research_mode)
    sub_queries = [sq.query for sq in output.sub_queries]
    supervisor_done(str(output.intent_type), sub_queries)

    return {
        "intent_type": output.intent_type,
        "sub_queries": sub_queries,
        "source_routing": {sq.query: sq.source_type for sq in output.sub_queries},
    }

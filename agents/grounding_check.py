"""
agents/grounding_check.py

Grounding check node: validates synthesis findings against worker evidence.

Reads:  state["synthesis"], state["compressed_worker_results"]
Writes: state["grounding_issues"] (always written, empty list if no issues)

Must run BEFORE the critic node. The critic reads grounding_issues as
additional context for its adversarial review.
"""

from __future__ import annotations

import logging
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
from graph.progress import grounding_done, grounding_start
from graph.state import (
    Evidence,
    Finding,
    ResearchAgentState,
    SynthesisOutput,
    WorkerResult,
)
from graph.utils import AgentParseError, parse_agent_json

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "grounding_check_v1.txt"

_EVIDENCE_TEXT_CAP = 500  # chars per supporting_chunk snippet in the block
_TOTAL_BLOCK_CAP = 5000  # total chars for the full findings+evidence block

_VALID_ISSUE_TYPES: frozenset[str] = frozenset(
    {"UNSUPPORTED", "HALLUCINATED_CITATION", "MISSING_CITATION"}
)


# ---------------------------------------------------------------------------
# Two-level deserialization TypedDicts
# ---------------------------------------------------------------------------


class _RawGroundingIssue(TypedDict):
    """Raw grounding issue dict as returned by the LLM."""

    type: str  # "UNSUPPORTED" | "HALLUCINATED_CITATION" | "MISSING_CITATION"
    finding_heading: str
    description: str


class _RawGroundingOutput(TypedDict):
    """Flat TypedDict that parse_agent_json() can instantiate directly."""

    issues: list[_RawGroundingIssue]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is a ChatAnthropic instance (supports cache_control).

    Same pattern as coordinator._is_anthropic.
    """
    try:
        from langchain_anthropic import ChatAnthropic

        return isinstance(llm, ChatAnthropic)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt(path: Path) -> str:
    """Load prompt template from path.

    Args:
        path: Absolute path to the prompt .txt file.

    Returns:
        Prompt template string with {{placeholder}} tokens.

    Raises:
        FileNotFoundError: If the prompt file does not exist at path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Grounding check prompt not found at {path}. "
            "Ensure the prompts/ directory is present and the file exists."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Evidence serialization
# ---------------------------------------------------------------------------


def _build_findings_with_evidence_block(
    findings: list[Finding],
    evidence_by_url: dict[str, Evidence],
) -> str:
    """Serialize synthesis findings with their referenced evidence for the LLM.

    For each finding:
      FINDING {n}: {heading}
      Body: {body}
      Evidence:
        [URL] {url}
        Claim: {evidence.claim}
        Supporting: {first_chunk[:_EVIDENCE_TEXT_CAP]}

    For evidence_ref URLs not in evidence_by_url:
      [URL] {url}
      Note: Evidence not found — URL only

    Caps total output at ~_TOTAL_BLOCK_CAP chars (per-item boundary, same pattern
    as coordinator._serialize_evidence — no mid-item truncation).

    Args:
        findings: List of Finding dataclasses from SynthesisOutput.
        evidence_by_url: Map from source_url → Evidence for lookup.

    Returns:
        Serialized string ready for {{findings_with_evidence_block}} substitution.
    """
    lines: list[str] = []
    total_chars = 0
    cap_reached = False

    for n, finding in enumerate(findings, start=1):
        if cap_reached:
            break

        finding_header = f"FINDING {n}: {finding.heading}"
        finding_body = f"Body: {finding.body}"
        finding_block_lines = [finding_header, finding_body, "Evidence:"]

        for url in finding.evidence_refs:
            url_line = f"  [URL] {url}"
            ev = evidence_by_url.get(url)
            if ev is None:
                logger.warning(
                    "Grounding check: evidence_ref URL '%s' not in compressed_worker_results",
                    url,
                )
                ev_lines = [url_line, "  Note: Evidence not found — URL only"]
            else:
                chunk_text = ""
                if ev.supporting_chunks:
                    chunk_text = ev.supporting_chunks[0][:_EVIDENCE_TEXT_CAP]
                ev_lines = [
                    url_line,
                    f"  Claim: {ev.claim}",
                    f"  Supporting: {chunk_text}",
                ]
            finding_block_lines.extend(ev_lines)

        block_str = "\n".join(finding_block_lines)
        if total_chars + len(block_str) > _TOTAL_BLOCK_CAP:
            lines.append("[cap reached — remaining findings omitted]")
            cap_reached = True
            break

        lines.append(block_str)
        total_chars += len(block_str)

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Issue string formatting
# ---------------------------------------------------------------------------


def _format_issue_string(raw: _RawGroundingIssue) -> str:
    """Convert a raw grounding issue dict to a formatted string for state.

    Format: "[{type}] Finding: '{finding_heading}' — {description}"

    Args:
        raw: Raw grounding issue dict (TypedDict at runtime = plain dict).

    Returns:
        Formatted issue string for grounding_issues list.
    """
    # _RawGroundingIssue is a TypedDict; at runtime it is a plain dict.
    issue_type: str = str(raw["type"])
    heading: str = str(raw["finding_heading"])
    description: str = str(raw["description"])
    return f"[{issue_type}] Finding: '{heading}' — {description}"


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def _build_messages(
    findings_with_evidence_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for the grounding check LLM call.

    use_cache_control=True (Anthropic): static prompt instructions in a cached
      content block; findings_with_evidence_block in a separate uncached block.
    use_cache_control=False (other providers): plain string SystemMessage with
      {{findings_with_evidence_block}} placeholder substituted.

    Args:
        findings_with_evidence_block: Serialized findings+evidence string.
        prompt_template: Loaded grounding_check_v1.txt template.
        use_cache_control: Pass True only for Anthropic providers.

    Returns:
        List containing a single SystemMessage.
    """
    if use_cache_control:
        content_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_template,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"SYNTHESIS FINDINGS WITH THEIR CITED EVIDENCE:\n{findings_with_evidence_block}",
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = prompt_template.replace(
        "{{findings_with_evidence_block}}", findings_with_evidence_block
    )
    return [SystemMessage(content=filled)]


# ---------------------------------------------------------------------------
# Parse and validate
# ---------------------------------------------------------------------------


def _parse_grounding_output(raw: str) -> list[str]:
    """Parse LLM JSON and convert to list[str] for state["grounding_issues"].

    Steps:
      1. parse_agent_json(raw, _RawGroundingOutput)
      2. Validate each issue: type in _VALID_ISSUE_TYPES
      3. Validate: finding_heading and description not empty
      4. _format_issue_string() each valid issue → str
      5. Return list of formatted strings (empty list if issues=[])

    Args:
        raw: Raw LLM response string.

    Returns:
        List of formatted grounding issue strings.

    Raises:
        AgentParseError: If JSON is malformed or any issue has invalid type/empty fields.
    """
    parsed: _RawGroundingOutput = parse_agent_json(raw, _RawGroundingOutput)

    result: list[str] = []
    raw_issues = parsed["issues"]

    for i, raw_issue in enumerate(raw_issues):
        # At runtime TypedDict nesting arrives as plain dicts from json.loads().
        if isinstance(raw_issue, dict):
            issue_type = str(raw_issue.get("type", ""))
            heading = str(raw_issue.get("finding_heading", ""))
            description = str(raw_issue.get("description", ""))
        else:
            issue_type = str(getattr(raw_issue, "type", ""))
            heading = str(getattr(raw_issue, "finding_heading", ""))
            description = str(getattr(raw_issue, "description", ""))

        if issue_type not in _VALID_ISSUE_TYPES:
            raise AgentParseError(
                f"Issue {i} has invalid type '{issue_type}'. "
                f"Must be one of: {sorted(_VALID_ISSUE_TYPES)}"
            )
        if not heading.strip():
            raise AgentParseError(
                f"Issue {i} has an empty finding_heading — required field."
            )
        if not description.strip():
            raise AgentParseError(
                f"Issue {i} has an empty description — required field."
            )

        formatted_raw: _RawGroundingIssue = {
            "type": issue_type,
            "finding_heading": heading,
            "description": description,
        }
        result.append(_format_issue_string(formatted_raw))

    return result


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
    messages: list[BaseMessage],
    llm: BaseChatModel,
) -> list[str]:
    """Call the LLM and parse the grounding check response.

    Retries up to 3 times (exponential backoff 1–10s) on:
    - AgentParseError  (bad JSON or validation failure)
    - httpx.HTTPStatusError  (4xx/5xx from the provider)
    - httpx.TimeoutException  (network timeout)

    Args:
        messages: Pre-built message list for the LLM.
        llm: Pre-configured LLM client from NofrinContext.

    Returns:
        List of formatted grounding issue strings (empty if no issues).

    Raises:
        AgentParseError: After 3 failed parse/validation attempts.
    """
    response = await llm.ainvoke(messages)
    raw: str = str(response.content)
    return _parse_grounding_output(raw)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def grounding_check_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, list[str]]:
    """LangGraph node: validate synthesis findings against worker evidence.

    Reads state["synthesis"] and state["compressed_worker_results"].
    Calls the LLM once to detect UNSUPPORTED, HALLUCINATED_CITATION, and
    MISSING_CITATION issues across all findings simultaneously.
    Always writes state["grounding_issues"] — empty list if no issues found.

    Fast path: if synthesis.findings is empty, returns {"grounding_issues": []}
    without making any LLM call.

    Args:
        state: ResearchAgentState. Reads: synthesis, compressed_worker_results.
        runtime: Injected NofrinContext. Uses llm_coordinator.

    Returns:
        {"grounding_issues": list[str]}

    Raises:
        ValueError: If synthesis is None or compressed_worker_results is empty.
        AgentParseError: After 3 failed LLM/parse attempts.
    """
    synthesis: SynthesisOutput | None = state["synthesis"]
    if synthesis is None:
        raise ValueError(
            "grounding_check_node: state['synthesis'] is None — "
            "coordinator must run before grounding_check."
        )

    worker_results: list[WorkerResult] = state["compressed_worker_results"]
    if not worker_results:
        raise ValueError(
            "grounding_check_node: compressed_worker_results is empty — "
            "no evidence available for grounding check."
        )

    # Fast path: nothing to check.
    if not synthesis.findings:
        grounding_done(0)
        return {"grounding_issues": []}

    grounding_start(len(synthesis.findings))

    # Build URL → Evidence lookup from compressed worker results.
    evidence_by_url: dict[str, Evidence] = {}
    for wr in worker_results:
        for ev in wr.evidence_items:
            if ev.source_url not in evidence_by_url:
                evidence_by_url[ev.source_url] = ev

    llm: BaseChatModel = runtime.context.llm_coordinator
    prompt_template = _load_prompt(PROMPT_PATH)
    findings_block = _build_findings_with_evidence_block(
        synthesis.findings, evidence_by_url
    )
    messages = _build_messages(findings_block, prompt_template, _is_anthropic(llm))

    issues = await _call_llm(messages, llm)
    grounding_done(len(issues))
    return {"grounding_issues": issues}


__all__ = ["grounding_check_node"]

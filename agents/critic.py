"""
agents/critic.py

Critic node: adversarial 5-dimension evaluation of the synthesis brief.

Reads:  state["synthesis"], state["grounding_issues"]
Writes: state["critic_output"], state["total_tokens_used"]

Must run AFTER grounding_check (CLAUDE.md rule).
Does NOT modify synthesis, revision_count, or grounding_issues.
Routing to delivery or coordinator is handled by route_after_critic in graph/router.py.
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
from graph.progress import critic_done, critic_start
from graph.state import (
    CriticIssue,
    CriticOutput,
    CriticSuggestion,
    ResearchAgentState,
    SynthesisOutput,
)
from graph.utils import AgentParseError, parse_agent_json

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic_v1.txt"
_SYNTHESIS_BLOCK_CAP = 6000  # chars — per-item boundary, no mid-item truncation

_VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "major", "minor"})


# ---------------------------------------------------------------------------
# Two-level deserialization TypedDicts
# ---------------------------------------------------------------------------


class _RawCriticIssue(TypedDict):
    """Raw critic issue dict as returned by the LLM."""

    issue_text: str
    quote_from_synthesis: str
    severity: str  # validated against _VALID_SEVERITIES before use


class _RawCriticSuggestion(TypedDict):
    """Raw critic suggestion dict as returned by the LLM."""

    action: str
    target_section: str
    new_evidence_needed: bool


class _RawCriticOutput(TypedDict):
    """Flat TypedDict that parse_agent_json() can instantiate directly.

    Intentionally omits final_quality_score and passed — LLM does NOT return
    these fields; they are computed in code via _compute_final_score() and
    CriticOutput.__post_init__.
    """

    factuality_score: float
    citation_alignment_score: float
    reasoning_score: float
    completeness_score: float
    bias_score: float
    issues: list[_RawCriticIssue]
    suggestions: list[_RawCriticSuggestion]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is a ChatAnthropic instance (supports cache_control).

    Same pattern as coordinator._is_anthropic and grounding_check._is_anthropic.
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
            f"Critic prompt not found at {path}. "
            "Ensure the prompts/ directory is present and the file exists."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthesis serialization
# ---------------------------------------------------------------------------


def _serialize_synthesis(synthesis: SynthesisOutput) -> str:
    """Serialize SynthesisOutput for the critic prompt.

    Includes: topic, executive_summary, all findings (heading + body +
    evidence_refs), risks, gaps. Caps at ~_SYNTHESIS_BLOCK_CAP chars
    at per-item boundaries — no mid-item truncation.

    Args:
        synthesis: Coordinator's SynthesisOutput to be reviewed.

    Returns:
        Serialized string ready for {{synthesis_block}} substitution.
    """
    lines: list[str] = []
    total_chars = 0
    cap_reached = False

    header = (
        f"TOPIC: {synthesis.topic}\n\nEXECUTIVE SUMMARY:\n{synthesis.executive_summary}"
    )
    lines.append(header)
    total_chars += len(header)

    for n, finding in enumerate(synthesis.findings, start=1):
        if cap_reached:
            break
        refs_str = (
            ", ".join(finding.evidence_refs) if finding.evidence_refs else "(none)"
        )
        finding_block = (
            f"\nFINDING {n}: {finding.heading}\n"
            f"Body: {finding.body}\n"
            f"Evidence refs: {refs_str}"
        )
        if total_chars + len(finding_block) > _SYNTHESIS_BLOCK_CAP:
            lines.append("\n[cap reached — remaining findings omitted]")
            cap_reached = True
            break
        lines.append(finding_block)
        total_chars += len(finding_block)

    if synthesis.risks:
        risks_block = "\nRISKS:\n" + "\n".join(f"- {r}" for r in synthesis.risks)
        if not cap_reached and total_chars + len(risks_block) <= _SYNTHESIS_BLOCK_CAP:
            lines.append(risks_block)
            total_chars += len(risks_block)

    if synthesis.gaps:
        gaps_block = "\nGAPS:\n" + "\n".join(f"- {g}" for g in synthesis.gaps)
        if not cap_reached and total_chars + len(gaps_block) <= _SYNTHESIS_BLOCK_CAP:
            lines.append(gaps_block)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grounding issues serialization
# ---------------------------------------------------------------------------


def _serialize_grounding_issues(issues: list[str]) -> str:
    """Return grounding issues as a numbered list, or the clean-pass message.

    Args:
        issues: list[str] from state["grounding_issues"].

    Returns:
        Formatted string ready for {{grounding_issues_block}} substitution.
    """
    if not issues:
        return "(none — grounding check passed clean)"
    return "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def _build_messages(
    synthesis_block: str,
    grounding_issues_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for the critic LLM call.

    use_cache_control=True (Anthropic): static prompt + rubric cached in a
      cache_control block; synthesis_block + grounding_issues_block are in a
      separate uncached dynamic block.
    use_cache_control=False (other providers): plain string SystemMessage
      with {{placeholders}} substituted.

    Args:
        synthesis_block: Serialized SynthesisOutput string.
        grounding_issues_block: Serialized grounding issues string.
        prompt_template: Loaded critic_v1.txt template.
        use_cache_control: Pass True only for Anthropic providers.

    Returns:
        List containing a single SystemMessage.
    """
    dynamic_text = (
        f"SYNTHESIS TO REVIEW:\n{synthesis_block}\n\n"
        f"KNOWN GROUNDING ISSUES (pre-checked by automated fact-checker):\n"
        f"{grounding_issues_block}"
    )
    if use_cache_control:
        content_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_template,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic_text,
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = prompt_template.replace("{{synthesis_block}}", synthesis_block).replace(
        "{{grounding_issues_block}}", grounding_issues_block
    )
    return [SystemMessage(content=filled)]


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_final_score(raw: _RawCriticOutput) -> float:
    """Compute weighted final_quality_score from 5 raw dimension scores.

    Formula: 0.30*F + 0.25*C + 0.20*R + 0.15*Co + 0.10*B

    Each dimension is clamped to [0.0, 5.0] before weighting to guard
    against LLM hallucinating out-of-range values.

    Does NOT use the LLM's own final_quality_score even if present in raw.

    Args:
        raw: _RawCriticOutput containing the 5 dimension scores.

    Returns:
        Weighted float in [0.0, 5.0].
    """
    f = min(5.0, max(0.0, float(raw["factuality_score"])))
    c = min(5.0, max(0.0, float(raw["citation_alignment_score"])))
    r = min(5.0, max(0.0, float(raw["reasoning_score"])))
    co = min(5.0, max(0.0, float(raw["completeness_score"])))
    b = min(5.0, max(0.0, float(raw["bias_score"])))
    return 0.30 * f + 0.25 * c + 0.20 * r + 0.15 * co + 0.10 * b


# ---------------------------------------------------------------------------
# Parse and validate
# ---------------------------------------------------------------------------


def _parse_critic_output(raw: str) -> CriticOutput:
    """Parse LLM JSON and build a typed CriticOutput.

    Steps:
      1. parse_agent_json(raw, _RawCriticOutput)
      2. Clamp all 5 dimension scores to [0.0, 5.0]
      3. Validate each issue.severity against _VALID_SEVERITIES
      4. Validate issues: issue_text and quote_from_synthesis not empty
      5. Convert _RawCriticIssue → CriticIssue dataclass
      6. Convert _RawCriticSuggestion → CriticSuggestion dataclass
      7. Compute final_quality_score via _compute_final_score()
      8. Build and return CriticOutput (passed set by __post_init__)

    Args:
        raw: Raw LLM response string.

    Returns:
        Fully validated CriticOutput with code-computed scores.

    Raises:
        AgentParseError: If JSON is malformed or validation fails.
    """
    parsed: _RawCriticOutput = parse_agent_json(raw, _RawCriticOutput)

    # Clamp dimension scores (guards against LLM returning out-of-range values).
    factuality = min(5.0, max(0.0, float(parsed["factuality_score"])))
    citation = min(5.0, max(0.0, float(parsed["citation_alignment_score"])))
    reasoning = min(5.0, max(0.0, float(parsed["reasoning_score"])))
    completeness = min(5.0, max(0.0, float(parsed["completeness_score"])))
    bias = min(5.0, max(0.0, float(parsed["bias_score"])))

    # Validate and convert issues.
    issues: list[CriticIssue] = []
    raw_issues = parsed["issues"]
    for i, raw_issue in enumerate(raw_issues):
        if isinstance(raw_issue, dict):
            severity_str = str(raw_issue.get("severity", ""))
            issue_text = str(raw_issue.get("issue_text", ""))
            quote = str(raw_issue.get("quote_from_synthesis", ""))
        else:
            severity_str = str(getattr(raw_issue, "severity", ""))
            issue_text = str(getattr(raw_issue, "issue_text", ""))
            quote = str(getattr(raw_issue, "quote_from_synthesis", ""))

        if severity_str not in _VALID_SEVERITIES:
            raise AgentParseError(
                f"Issue {i} has invalid severity '{severity_str}'. "
                f"Must be one of: {sorted(_VALID_SEVERITIES)}"
            )
        if not issue_text.strip():
            raise AgentParseError(
                f"Issue {i} has an empty issue_text — required field."
            )
        # quote_from_synthesis is required but the LLM occasionally omits it.
        # Rather than crashing the pipeline, substitute a placeholder and log.
        if not quote.strip():
            logger.warning(
                "Issue %d has an empty quote_from_synthesis — using placeholder.", i
            )
            quote = "(no quote provided)"

        issues.append(
            CriticIssue(
                issue_text=issue_text,
                quote_from_synthesis=quote,
                severity=severity_str,  # type: ignore[arg-type]
            )
        )

    # Convert suggestions (no strict validation — empty fields are acceptable
    # for suggestions; only issues carry mandatory evidence).
    suggestions: list[CriticSuggestion] = []
    raw_suggestions = parsed["suggestions"]
    for raw_sug in raw_suggestions:
        if isinstance(raw_sug, dict):
            action = str(raw_sug.get("action", ""))
            target = str(raw_sug.get("target_section", ""))
            new_ev = bool(raw_sug.get("new_evidence_needed", False))
        else:
            action = str(getattr(raw_sug, "action", ""))
            target = str(getattr(raw_sug, "target_section", ""))
            new_ev = bool(getattr(raw_sug, "new_evidence_needed", False))

        suggestions.append(
            CriticSuggestion(
                action=action,
                target_section=target,
                new_evidence_needed=new_ev,
            )
        )

    # Compute final score in code — never trust the LLM's own value.
    # We pass a clamped copy so _compute_final_score sees the sanitized values.
    clamped: _RawCriticOutput = {
        "factuality_score": factuality,
        "citation_alignment_score": citation,
        "reasoning_score": reasoning,
        "completeness_score": completeness,
        "bias_score": bias,
        "issues": raw_issues,
        "suggestions": raw_suggestions,
    }
    final_score = _compute_final_score(clamped)

    return CriticOutput(
        factuality_score=factuality,
        citation_alignment_score=citation,
        reasoning_score=reasoning,
        completeness_score=completeness,
        bias_score=bias,
        final_quality_score=final_score,
        issues=issues,
        suggestions=suggestions,
        passed=False,  # overridden by __post_init__
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
    messages: list[BaseMessage],
    llm: BaseChatModel,
) -> tuple[CriticOutput, int]:
    """Call the LLM and parse/validate the critic response.

    Retries up to 3 times (exponential backoff 1–10s) on:
    - AgentParseError  (bad JSON or validation failure)
    - httpx.HTTPStatusError  (4xx/5xx from the provider)
    - httpx.TimeoutException  (network timeout)

    Args:
        messages: Pre-built message list for the LLM.
        llm: Pre-configured LLM client from NofrinContext.

    Returns:
        (CriticOutput, tokens_used).

    Raises:
        AgentParseError: After 3 failed parse/validation attempts.
    """
    response = await llm.ainvoke(messages)
    raw: str = str(response.content)
    tokens: int = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        tokens = int(response.usage_metadata.get("total_tokens", 0))
    critic_output = _parse_critic_output(raw)
    return critic_output, tokens


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def critic_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, object]:
    """LangGraph node: adversarial 5-dimension review of the synthesis.

    Reads state["synthesis"] and state["grounding_issues"].
    Calls llm_critic (most capable model) for scoring and issue identification.
    Computes final_quality_score in code — never trusts LLM's own computation.
    Always writes state["critic_output"].

    Args:
        state: ResearchAgentState. Reads: synthesis, grounding_issues,
               total_tokens_used.
        runtime: Injected NofrinContext. Uses llm_critic.

    Returns:
        {"critic_output": CriticOutput, "total_tokens_used": int}

    Raises:
        ValueError: If synthesis is None.
        AgentParseError: After 3 failed LLM/parse attempts.
    """
    synthesis: SynthesisOutput | None = state["synthesis"]
    if synthesis is None:
        raise ValueError(
            "critic_node: state['synthesis'] is None — "
            "coordinator must run before critic."
        )

    grounding_issues: list[str] = state["grounding_issues"]
    llm: BaseChatModel = runtime.context.llm_critic
    prompt_template = _load_prompt(PROMPT_PATH)

    critic_start(state["revision_count"])
    synthesis_block = _serialize_synthesis(synthesis)
    grounding_issues_block = _serialize_grounding_issues(grounding_issues)
    messages = _build_messages(
        synthesis_block,
        grounding_issues_block,
        prompt_template,
        use_cache_control=_is_anthropic(llm),
    )

    critic_output, tokens_used = await _call_llm(messages, llm)
    critic_done(
        critic_output.final_quality_score,
        critic_output.passed,
        len(critic_output.issues),
    )

    current_tokens: int = state["total_tokens_used"]
    return {
        "critic_output": critic_output,
        "total_tokens_used": current_tokens + tokens_used,
    }


__all__ = ["critic_node"]

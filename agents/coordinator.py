"""
agents/coordinator.py

Coordinator node: synthesizes compressed worker evidence into SynthesisOutput.

Two execution paths:
  PATH 1 (revision_count == 0): first-pass synthesis from evidence only.
  PATH 2 (revision_count > 0):  targeted revision guided by critic issues.

Reads:  state["compressed_worker_results"], state["revision_count"],
        state["prior_syntheses"], state["critic_output"]
Writes: state["synthesis"], state["prior_syntheses"] (PATH 2 only),
        state["revision_count"] (PATH 2 only), state["total_tokens_used"]
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
from graph.progress import coordinator_done, coordinator_start
from graph.state import (
    CriticIssue,
    Evidence,
    Finding,
    ResearchAgentState,
    SynthesisOutput,
    WorkerResult,
)
from graph.utils import AgentParseError, parse_agent_json

FIRST_PASS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "coordinator_v1.txt"
REVISION_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "coordinator_revision_v2.txt"
)

logger = logging.getLogger(__name__)

# Total character cap across all serialized evidence items.
# Each item is either fully included or fully excluded (no mid-item truncation).
_EVIDENCE_CHAR_CAP = 6000


# ---------------------------------------------------------------------------
# Two-level deserialization TypedDicts
# ---------------------------------------------------------------------------
# parse_agent_json() calls schema_class(**data) and cannot recursively
# convert nested dicts to dataclasses.  We parse into flat TypedDicts first,
# then convert to typed dataclasses manually in _parse_and_validate().


class _RawFinding(TypedDict):
    """Raw finding dict as returned by the LLM."""

    heading: str
    body: str
    evidence_refs: list[str]


class _RawSynthesisOutput(TypedDict):
    """Flat TypedDict that parse_agent_json() can instantiate directly.

    findings is kept as list[dict] at runtime (TypedDict nesting is not
    enforced); conversion to Finding dataclasses happens in _parse_and_validate.
    """

    topic: str
    executive_summary: str
    findings: list[_RawFinding]
    risks: list[str]
    gaps: list[str]
    citation_urls: list[str]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is a ChatAnthropic instance (supports cache_control).

    Same pattern as supervisor._is_anthropic — wraps the import so
    non-Anthropic deployments don't require langchain_anthropic installed.
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
    """Load a prompt template from path.

    Args:
        path: Absolute path to the prompt .txt file.

    Returns:
        Prompt template string with {{placeholder}} tokens.

    Raises:
        FileNotFoundError: If the prompt file does not exist at path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Coordinator prompt not found at {path}. "
            "Ensure the prompts/ directory is present and the file exists."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Evidence serialization
# ---------------------------------------------------------------------------


def _serialize_evidence(worker_results: list[WorkerResult]) -> str:
    """Serialize compressed_worker_results into a readable text block for the LLM.

    Groups evidence by worker sub_query.  Format per item:
      [E{n}] CLAIM: {claim} | CONFIDENCE: {confidence:.2f} | URL: {source_url}

    Cap is enforced per-evidence-item: each item is either fully included or
    fully excluded — never truncated mid-sentence.  Total output is capped at
    ~_EVIDENCE_CHAR_CAP characters to prevent context overflow.

    Args:
        worker_results: List of WorkerResult from state["compressed_worker_results"].

    Returns:
        Serialized string ready to be substituted into {{evidence_block}}.
    """
    lines: list[str] = []
    total_chars = 0
    evidence_num = 0
    cap_reached = False

    for wr in worker_results:
        if cap_reached:
            break
        lines.append(f"--- Sub-query: {wr.sub_query} ({wr.source_type}) ---")
        for ev in wr.evidence_items:
            evidence_num += 1
            line = (
                f"[E{evidence_num}] CLAIM: {ev.claim} | "
                f"CONFIDENCE: {ev.confidence:.2f} | "
                f"URL: {ev.source_url}"
            )
            if total_chars + len(line) > _EVIDENCE_CHAR_CAP:
                lines.append("[cap reached — remaining evidence omitted]")
                cap_reached = True
                break
            lines.append(line)
            total_chars += len(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Critic issues serialization
# ---------------------------------------------------------------------------


def _serialize_critic_issues(issues: list[CriticIssue]) -> str:
    """Serialize a CriticIssue list into a numbered block for the revision prompt.

    Format per issue:
      {n}. [{severity}] {issue_text}
         Quote: "{quote_from_synthesis}"

    Args:
        issues: List of CriticIssue dataclasses from CriticOutput.issues.

    Returns:
        Serialized string ready for {{critic_issues_block}} substitution.
    """
    parts: list[str] = []
    for n, issue in enumerate(issues, start=1):
        parts.append(
            f"{n}. [{issue.severity}] {issue.issue_text}\n"
            f'   Quote: "{issue.quote_from_synthesis}"'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def _build_first_pass_messages(
    evidence_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for PATH 1 (first-pass synthesis).

    use_cache_control=True (Anthropic): static prompt instructions in a
      cached content block; evidence_block in a separate uncached block.
    use_cache_control=False (other providers): plain string SystemMessage
      with {{evidence_block}} placeholder substituted.

    Args:
        evidence_block: Serialized evidence from _serialize_evidence().
        prompt_template: Loaded coordinator_v1.txt template.
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
                "text": f"RESEARCH EVIDENCE:\n{evidence_block}",
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = prompt_template.replace("{{evidence_block}}", evidence_block)
    return [SystemMessage(content=filled)]


def _build_revision_messages(
    evidence_block: str,
    prior_synthesis_block: str,
    critic_issues_block: str,
    revision_count: int,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for PATH 2 (revision pass).

    use_cache_control=True (Anthropic): static prompt instructions cached;
      all dynamic content (evidence, prior synthesis, critic issues) in
      a single uncached block.
    use_cache_control=False: plain string SystemMessage with all three
      {{placeholder}} tokens substituted.

    Args:
        evidence_block: Serialized evidence from _serialize_evidence().
        prior_synthesis_block: Serialized text of the prior SynthesisOutput.
        critic_issues_block: Serialized issues from _serialize_critic_issues().
        revision_count: Current revision number (1-based for display).
        prompt_template: Loaded coordinator_revision_v2.txt template.
        use_cache_control: Pass True only for Anthropic providers.

    Returns:
        List containing a single SystemMessage.
    """
    if use_cache_control:
        dynamic = (
            f"RESEARCH EVIDENCE:\n{evidence_block}\n\n"
            f"PRIOR SYNTHESIS:\n{prior_synthesis_block}\n\n"
            f"CRITIC ISSUES TO ADDRESS:\n{critic_issues_block}"
        )
        content_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_template.replace(
                    "{{revision_count}}", str(revision_count)
                ),
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic,
            },
        ]
        return [SystemMessage(content=content_blocks)]  # type: ignore[arg-type]
    filled = (
        prompt_template.replace("{{evidence_block}}", evidence_block)
        .replace("{{prior_synthesis_block}}", prior_synthesis_block)
        .replace("{{critic_issues_block}}", critic_issues_block)
        .replace("{{revision_count}}", str(revision_count))
    )
    return [SystemMessage(content=filled)]


# ---------------------------------------------------------------------------
# Parse and validate
# ---------------------------------------------------------------------------


def _parse_and_validate(
    raw: str,
    available_urls: set[str],
    evidence_by_url: dict[str, Evidence],
    synthesis_version: int,
    prior_attempt_summary: str | None,
) -> SynthesisOutput:
    """Parse LLM JSON response and validate against available evidence URLs.

    Steps:
      1. parse_agent_json(raw, _RawSynthesisOutput) — strips fences, parses JSON
      2. Validate: executive_summary not empty
      3. Validate: at least 1 finding
      4. For each finding: at least 1 evidence_ref; ALL refs must be in
         available_urls — raises AgentParseError if any ref is not found
         (never silently drops ungrounded refs)
      5. Convert each raw finding dict → Finding dataclass
      6. Map citation_urls → real Evidence objects from evidence_by_url
         (URLs not in available_urls are silently excluded — they are a
         convenience summary, not the primary grounding mechanism)
      7. Build and return SynthesisOutput

    Args:
        raw: Raw LLM response string.
        available_urls: Set of all source_url strings in compressed_worker_results.
        evidence_by_url: Map from source_url → Evidence for citation lookup.
        synthesis_version: Version number to stamp on the output.
        prior_attempt_summary: Short prior synthesis summary (revision only).

    Returns:
        Validated SynthesisOutput dataclass.

    Raises:
        AgentParseError: If any validation rule fails (triggers tenacity retry).
    """
    raw_output: _RawSynthesisOutput = parse_agent_json(raw, _RawSynthesisOutput)

    if not str(raw_output["executive_summary"]).strip():
        raise AgentParseError("Coordinator returned an empty executive_summary.")

    raw_findings = raw_output["findings"]
    if not raw_findings:
        raise AgentParseError("Coordinator returned 0 findings — at least 1 required.")

    findings: list[Finding] = []
    for i, raw_finding in enumerate(raw_findings):
        # TypedDict nesting is not enforced at runtime — each finding arrives
        # as a plain dict from json.loads().
        if isinstance(raw_finding, dict):
            heading: str = str(raw_finding.get("heading", ""))
            body: str = str(raw_finding.get("body", ""))
            refs: list[str] = [str(r) for r in raw_finding.get("evidence_refs", [])]
        else:
            # Defensive path — should not occur with well-formed JSON.
            heading = str(getattr(raw_finding, "heading", ""))
            body = str(getattr(raw_finding, "body", ""))
            refs = [str(r) for r in getattr(raw_finding, "evidence_refs", [])]

        if not refs:
            raise AgentParseError(
                f"Finding {i} ('{heading[:40]}') has no evidence_refs — "
                "at least 1 required."
            )

        valid_refs: list[str] = []
        for ref in refs:
            if ref in available_urls:
                valid_refs.append(ref)
            else:
                logger.warning(
                    "Finding %d: URL '%s' not in available evidence — dropping ref.",
                    i,
                    ref,
                )
        if not valid_refs:
            raise AgentParseError(
                f"Finding {i} ('{heading[:40]}') has no valid evidence_refs after "
                "dropping hallucinated citations — retrying."
            )

        findings.append(Finding(heading=heading, body=body, evidence_refs=valid_refs))

    # Map citation_urls → real Evidence objects from compressed_worker_results.
    # Unknown URLs (not in available_urls) are silently excluded — they are a
    # convenience flat list, not the per-finding grounding mechanism.
    citations: list[Evidence] = []
    seen_urls: set[str] = set()
    for url in raw_output["citation_urls"]:
        if url in evidence_by_url and url not in seen_urls:
            citations.append(evidence_by_url[url])
            seen_urls.add(url)

    return SynthesisOutput(
        topic=str(raw_output["topic"]),
        executive_summary=str(raw_output["executive_summary"]),
        findings=findings,
        risks=[str(r) for r in (raw_output["risks"] or [])],
        gaps=[str(g) for g in (raw_output["gaps"] or [])],
        citations=citations,
        synthesis_version=synthesis_version,
        prior_attempt_summary=prior_attempt_summary,
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
    available_urls: set[str],
    evidence_by_url: dict[str, Evidence],
    synthesis_version: int,
    prior_attempt_summary: str | None,
) -> tuple[SynthesisOutput, int]:
    """Call the LLM and parse/validate the response into a SynthesisOutput.

    Retries up to 3 times (exponential backoff 1–10s) on:
    - AgentParseError  (bad JSON or validation failure in _parse_and_validate)
    - httpx.HTTPStatusError  (4xx/5xx from the provider)
    - httpx.TimeoutException  (network timeout)

    Args:
        messages: Pre-built message list (SystemMessage with evidence/prompt).
        llm: Pre-configured LLM client from NofrinContext.
        available_urls: Set of valid evidence source_urls for citation validation.
        evidence_by_url: Map from URL → Evidence for building citations list.
        synthesis_version: Version number for the resulting SynthesisOutput.
        prior_attempt_summary: Short summary of prior synthesis (revision only).

    Returns:
        Tuple of (SynthesisOutput, tokens_used_int).

    Raises:
        AgentParseError: After 3 failed parse/validation attempts.
        httpx.HTTPStatusError / httpx.TimeoutException: After 3 network failures.
    """
    response = await llm.ainvoke(messages)
    raw: str = str(response.content)

    tokens_used: int = 0
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        tokens_used = int(usage.get("total_tokens", 0))

    synthesis = _parse_and_validate(
        raw,
        available_urls,
        evidence_by_url,
        synthesis_version,
        prior_attempt_summary,
    )
    return synthesis, tokens_used


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def coordinator_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, object]:
    """LangGraph node: synthesize worker evidence into a SynthesisOutput.

    PATH 1 (revision_count == 0):
      - Reads compressed_worker_results only.
      - Uses prompts/coordinator_v1.txt.
      - Does NOT modify prior_syntheses or revision_count.
      - Sets synthesis_version = 1.
      - Returns: {"synthesis": SynthesisOutput, "total_tokens_used": int}

    PATH 2 (revision_count > 0):
      - Reads compressed_worker_results + current synthesis + critic issues.
      - Uses prompts/coordinator_revision_v1.txt.
      - Appends current state["synthesis"] → prior_syntheses BEFORE overwriting.
      - Increments revision_count by 1.
      - Sets synthesis_version = prior.synthesis_version + 1.
      - Returns: {"synthesis": SynthesisOutput, "prior_syntheses": list,
                  "revision_count": int, "total_tokens_used": int}

    Args:
        state: ResearchAgentState. Reads: compressed_worker_results,
               revision_count, synthesis, prior_syntheses, critic_output,
               total_tokens_used.
        runtime: Injected NofrinContext. Uses llm_coordinator.

    Returns:
        Partial state update dict. Keys depend on path (see above).

    Raises:
        ValueError: If compressed_worker_results is empty.
        AgentParseError: After 3 failed LLM/parse attempts.
    """
    worker_results: list[WorkerResult] = state["compressed_worker_results"]
    if not worker_results:
        raise ValueError(
            "coordinator_node: compressed_worker_results is empty — "
            "no evidence to synthesize. Check boundary_compressor output."
        )

    llm: BaseChatModel = runtime.context.llm_coordinator
    revision_count: int = state["revision_count"]

    # Build lookup structures used by _parse_and_validate.
    available_urls: set[str] = {
        ev.source_url for wr in worker_results for ev in wr.evidence_items
    }
    evidence_by_url: dict[str, Evidence] = {}
    for wr in worker_results:
        for ev in wr.evidence_items:
            if ev.source_url not in evidence_by_url:
                evidence_by_url[ev.source_url] = ev

    evidence_block = _serialize_evidence(worker_results)
    use_cache = _is_anthropic(llm)

    coordinator_start(revision_count, len(worker_results))

    if revision_count == 0:
        # ── PATH 1: first-pass synthesis ─────────────────────────────────────
        prompt_template = _load_prompt(FIRST_PASS_PROMPT_PATH)
        messages = _build_first_pass_messages(
            evidence_block, prompt_template, use_cache
        )
        synthesis, tokens = await _call_llm(
            messages,
            llm,
            available_urls,
            evidence_by_url,
            synthesis_version=1,
            prior_attempt_summary=None,
        )
        coordinator_done(len(synthesis.findings), tokens)
        return {
            "synthesis": synthesis,
            "revision_count": 1,  # marks first-pass as done; PATH 2 triggers on next call
            "total_tokens_used": state["total_tokens_used"] + tokens,
        }

    # ── PATH 2: revision pass ─────────────────────────────────────────────────
    # The current synthesis must be moved to prior_syntheses BEFORE we
    # overwrite state["synthesis"] — this preserves the full revision history.
    prior_synthesis: SynthesisOutput | None = state["synthesis"]
    old_prior: list[SynthesisOutput] = list(state["prior_syntheses"])

    if prior_synthesis is not None:
        updated_prior = old_prior + [prior_synthesis]
        synthesis_version = prior_synthesis.synthesis_version + 1
        prior_attempt_summary: str | None = prior_synthesis.executive_summary[:120]
    else:
        # Guard: revision_count > 0 but no prior synthesis — treat as first pass.
        updated_prior = old_prior
        synthesis_version = revision_count + 1
        prior_attempt_summary = None

    # Serialize prior synthesis text for the revision prompt.
    if prior_synthesis is not None:
        prior_synthesis_block = (
            f"Topic: {prior_synthesis.topic}\n"
            f"Executive Summary: {prior_synthesis.executive_summary}\n"
            f"Findings:\n"
            + "\n".join(
                f"  - {f.heading}: {f.body[:200]}" for f in prior_synthesis.findings
            )
        )
    else:
        prior_synthesis_block = "(no prior synthesis available)"

    # Serialize critic issues.
    critic_output = state["critic_output"]
    issues = critic_output.issues if critic_output is not None else []
    critic_issues_block = _serialize_critic_issues(issues)

    prompt_template = _load_prompt(REVISION_PROMPT_PATH)
    messages = _build_revision_messages(
        evidence_block,
        prior_synthesis_block,
        critic_issues_block,
        revision_count,
        prompt_template,
        use_cache,
    )

    synthesis, tokens = await _call_llm(
        messages,
        llm,
        available_urls,
        evidence_by_url,
        synthesis_version=synthesis_version,
        prior_attempt_summary=prior_attempt_summary,
    )
    coordinator_done(len(synthesis.findings), tokens)

    return {
        "synthesis": synthesis,
        "prior_syntheses": updated_prior,
        "revision_count": revision_count + 1,
        "total_tokens_used": state["total_tokens_used"] + tokens,
    }


__all__ = ["coordinator_node"]

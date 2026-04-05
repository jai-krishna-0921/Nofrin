"""
agents/delivery.py

Delivery node: format SynthesisOutput into the requested output format.

Reads:  state["synthesis"], state["critic_output"], state["output_format"],
        state["revision_count"]
Writes: state["final_output"]

Pure formatting logic — no LLM calls, no token cost.
No prompt file needed. No runtime context needed.
"""

from __future__ import annotations

from graph.progress import delivery_done, delivery_start
from graph.state import (
    CriticOutput,
    Evidence,
    Finding,
    ResearchAgentState,
    SynthesisOutput,
)

_QUALITY_CAVEAT_TEMPLATE: str = (
    "> **QUALITY NOTICE:** This research brief was delivered after reaching "
    "the maximum revision limit (2 revisions) without meeting the quality "
    "threshold (score: {score:.2f}/5.0). Review with appropriate scrutiny.\n"
)

# Must match graph/router.py._MAX_REVISIONS — declared locally to avoid
# circular import risk (router imports from state, state imports nothing from agents).
_MAX_REVISIONS: int = 2


# ---------------------------------------------------------------------------
# Force-delivery detection
# ---------------------------------------------------------------------------


def _is_force_delivered(state: ResearchAgentState) -> bool:
    """Return True if revision_count >= _MAX_REVISIONS and critic did not pass.

    Args:
        state: Current ResearchAgentState.

    Returns:
        True when max revisions exhausted without meeting quality threshold.
    """
    critic_output: CriticOutput | None = state["critic_output"]
    if critic_output is None:
        return False
    return state["revision_count"] >= _MAX_REVISIONS and not critic_output.passed


# ---------------------------------------------------------------------------
# Citation map
# ---------------------------------------------------------------------------


def _build_citation_map(citations: list[Evidence]) -> dict[str, int]:
    """Build URL → 1-based footnote index mapping from synthesis.citations.

    Args:
        citations: List of Evidence objects from SynthesisOutput.citations.

    Returns:
        Dict mapping source_url to 1-based footnote index.
    """
    return {ev.source_url: i for i, ev in enumerate(citations, start=1)}


# ---------------------------------------------------------------------------
# Finding renderer
# ---------------------------------------------------------------------------


def _render_finding(finding: Finding, citation_map: dict[str, int]) -> str:
    """Render a single finding as markdown.

    Includes: ### heading, body paragraph, [^N] footnote refs for each
    evidence_ref URL present in citation_map. Unknown URLs are silently
    skipped (no footnote appended).

    Args:
        finding: Finding dataclass to render.
        citation_map: URL → footnote index mapping.

    Returns:
        Markdown string for the finding block.
    """
    refs = "".join(
        f"[^{citation_map[url]}]"
        for url in finding.evidence_refs
        if url in citation_map
    )
    return f"### {finding.heading}\n\n{finding.body}{refs}"


# ---------------------------------------------------------------------------
# Citations section renderer
# ---------------------------------------------------------------------------


def _render_citations_section(citations: list[Evidence]) -> str:
    """Render the References section with numbered footnote definitions.

    Format per entry:
        [^1]: [Source Title](URL) - Published: YYYY-MM-DD

    Args:
        citations: List of Evidence objects from SynthesisOutput.citations.

    Returns:
        Full References section string, or empty string if citations is empty.
    """
    if not citations:
        return ""
    lines: list[str] = ["## References", ""]
    for i, ev in enumerate(citations, start=1):
        pub = f" - Published: {ev.published_date}" if ev.published_date else ""
        lines.append(f"[^{i}]: [{ev.source_title}]({ev.source_url}){pub}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quality badge
# ---------------------------------------------------------------------------


def _render_quality_badge(
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """Render the quality badge line.

    Args:
        critic_output: CriticOutput from state, or None if critic never ran.
        is_force_delivered: True when max revisions reached without passing.

    Returns:
        Badge string, e.g. "**Quality: PASS** (4.25/5.0)"
    """
    if critic_output is None:
        return "**Quality: N/A**"
    score = critic_output.final_quality_score
    if is_force_delivered:
        return f"**Quality: REVIEW RECOMMENDED** ({score:.2f}/5.0)"
    return f"**Quality: PASS** ({score:.2f}/5.0)"


# ---------------------------------------------------------------------------
# Markdown renderer (exported for testing and reuse)
# ---------------------------------------------------------------------------


def render_markdown(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """Render SynthesisOutput as a complete markdown document.

    Pure function: no side effects, no LLM calls, deterministic output.

    Section order:
        1. Quality caveat (ONLY if force-delivered)
        2. # {topic}
        3. Quality badge line
        4. ## Executive Summary
        5. ## Key Findings (### per finding, [^N] footnote refs)
        6. ## Risks (omit if empty)
        7. ## Known Gaps (omit if empty)
        8. ## References (footnote definitions, omit if no citations)

    Args:
        synthesis: SynthesisOutput from the coordinator node.
        critic_output: CriticOutput from the critic node, or None.
        is_force_delivered: True when max revisions exhausted without passing.

    Returns:
        Complete markdown document as a single string.
    """
    citation_map = _build_citation_map(synthesis.citations)
    parts: list[str] = []

    # 1. Quality caveat (force-delivery only)
    if is_force_delivered and critic_output is not None:
        caveat = _QUALITY_CAVEAT_TEMPLATE.format(
            score=critic_output.final_quality_score
        )
        parts.append(caveat)

    # 2. Title
    parts.append(f"# {synthesis.topic}")

    # 3. Quality badge
    parts.append(_render_quality_badge(critic_output, is_force_delivered))

    # 4. Executive summary
    parts.append(f"## Executive Summary\n\n{synthesis.executive_summary}")

    # 5. Key findings
    if synthesis.findings:
        finding_blocks = "\n\n".join(
            _render_finding(f, citation_map) for f in synthesis.findings
        )
        parts.append(f"## Key Findings\n\n{finding_blocks}")
    else:
        parts.append("## Key Findings")

    # 6. Risks (omit if empty)
    if synthesis.risks:
        risk_lines = "\n".join(f"- {r}" for r in synthesis.risks)
        parts.append(f"## Risks\n\n{risk_lines}")

    # 7. Known Gaps (omit if empty)
    if synthesis.gaps:
        gap_lines = "\n".join(f"- {g}" for g in synthesis.gaps)
        parts.append(f"## Known Gaps\n\n{gap_lines}")

    # 8. References (omit if no citations)
    citations_section = _render_citations_section(synthesis.citations)
    if citations_section:
        parts.append(citations_section)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stub renderers for unimplemented formats
# ---------------------------------------------------------------------------


def _render_docx(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """DOCX renderer — not yet implemented.

    Raises:
        NotImplementedError: DOCX rendering requires python-docx integration.
    """
    raise NotImplementedError(
        "DOCX rendering not yet implemented. Use output_format='markdown'."
    )


def _render_pdf(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """PDF renderer — not yet implemented.

    Raises:
        NotImplementedError: PDF rendering requires WeasyPrint integration.
    """
    raise NotImplementedError(
        "PDF rendering not yet implemented. Use output_format='markdown'."
    )


def _render_pptx(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """PPTX renderer — not yet implemented.

    Raises:
        NotImplementedError: PPTX rendering requires python-pptx integration.
    """
    raise NotImplementedError(
        "PPTX rendering not yet implemented. Use output_format='markdown'."
    )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def delivery_node(
    state: ResearchAgentState,
) -> dict[str, str]:
    """LangGraph node: format synthesis into the requested output format.

    Pure formatting logic — no LLM call, no token cost.

    Reads state["synthesis"], state["critic_output"], state["output_format"],
    and state["revision_count"].

    Detects force-delivery case (revision_count >= 2 AND critic_output.passed
    == False) and injects a quality caveat at the top of the output.

    Args:
        state: ResearchAgentState. No runtime parameter — no LLM client needed.

    Returns:
        {"final_output": str} — partial state update with rendered document.

    Raises:
        ValueError: If synthesis is None (coordinator must run first).
        NotImplementedError: If output_format is "docx", "pdf", or "pptx".
    """
    synthesis: SynthesisOutput | None = state["synthesis"]
    if synthesis is None:
        raise ValueError(
            "delivery_node: state['synthesis'] is None — "
            "coordinator must run before delivery."
        )

    critic_output: CriticOutput | None = state["critic_output"]
    output_format = state["output_format"]
    force_delivered = _is_force_delivered(state)

    delivery_start(str(output_format))
    if output_format == "markdown":
        final_output = render_markdown(synthesis, critic_output, force_delivered)
    elif output_format == "docx":
        final_output = _render_docx(synthesis, critic_output, force_delivered)
    elif output_format == "pdf":
        final_output = _render_pdf(synthesis, critic_output, force_delivered)
    elif output_format == "pptx":
        final_output = _render_pptx(synthesis, critic_output, force_delivered)
    else:
        raise NotImplementedError(f"Unknown output_format: {output_format!r}")

    delivery_done(str(output_format), len(final_output))
    return {"final_output": final_output}


__all__ = ["delivery_node", "render_markdown"]

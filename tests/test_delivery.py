"""
tests/test_delivery.py

Unit tests for agents/delivery.py.

Tests markdown rendering, force-delivery caveat injection, quality badge
formatting, section inclusion/omission logic, citation formatting, and
NotImplementedError for unimplemented formats.
"""

from __future__ import annotations

import pytest

from agents.delivery import delivery_node, render_markdown
from graph.state import (
    CriticIssue,
    CriticOutput,
    CriticSuggestion,
    Evidence,
    Finding,
    ResearchAgentState,
    SynthesisOutput,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def build_evidence(
    source_url: str = "https://example.com",
    source_title: str = "Example",
    claim: str = "Test claim",
) -> Evidence:
    """Create a minimal Evidence dataclass."""
    return Evidence(
        claim=claim,
        supporting_chunks=["chunk"],
        source_url=source_url,
        source_title=source_title,
        published_date="2024-01-01",
        confidence=0.9,
        contradiction_score=0.1,
    )


def build_synthesis(
    topic: str = "Test Topic",
    executive_summary: str = "Test summary",
    findings: list[Finding] | None = None,
    risks: list[str] | None = None,
    gaps: list[str] | None = None,
    citations: list[Evidence] | None = None,
) -> SynthesisOutput:
    """Create a SynthesisOutput with configurable content."""
    if findings is None:
        findings = [
            Finding(
                heading="Finding 1",
                body="Body text",
                evidence_refs=["https://example.com"],
            )
        ]
    if risks is None:
        risks = ["Risk 1"]
    if gaps is None:
        gaps = ["Gap 1"]
    if citations is None:
        citations = [build_evidence()]
    return SynthesisOutput(
        topic=topic,
        executive_summary=executive_summary,
        findings=findings,
        risks=risks,
        gaps=gaps,
        citations=citations,
        synthesis_version=1,
        prior_attempt_summary=None,
    )


def build_critic(passed: bool = True, score: float = 4.2) -> CriticOutput:
    """Create a CriticOutput with specified passed/score values.

    IMPORTANT: __post_init__ overrides passed from final_quality_score,
    so set score < 4.0 if you want passed=False.
    """
    return CriticOutput(
        factuality_score=score,
        citation_alignment_score=score,
        reasoning_score=score,
        completeness_score=score,
        bias_score=score,
        final_quality_score=score,
        issues=[],
        suggestions=[],
        passed=passed,  # ignored — computed in __post_init__
    )


def build_state(
    synthesis: SynthesisOutput | None,
    critic_output: CriticOutput | None = None,
    output_format: str = "markdown",
    revision_count: int = 0,
) -> ResearchAgentState:
    """Create a minimal ResearchAgentState for delivery tests."""
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format=output_format,  # type: ignore[typeddict-item]
        sub_queries=["sub query 1"],
        source_routing={"sub query 1": "web"},
        worker_results=[],
        compressed_worker_results=[],
        synthesis=synthesis,
        grounding_issues=[],
        critic_output=critic_output,
        revision_count=revision_count,
        prior_syntheses=[],
        session_id="test-session",
        total_tokens_used=0,
        cost_usd=0.0,
        final_output=None,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delivery_node_returns_final_output_key() -> None:
    """Test #1: delivery_node returns dict with 'final_output' key."""
    state = build_state(synthesis=build_synthesis())
    result = await delivery_node(state)
    assert "final_output" in result


@pytest.mark.asyncio
async def test_final_output_is_string() -> None:
    """Test #2: final_output value is a string."""
    state = build_state(synthesis=build_synthesis())
    result = await delivery_node(state)
    assert isinstance(result["final_output"], str)


@pytest.mark.asyncio
async def test_markdown_contains_topic_as_title() -> None:
    """Test #3: markdown output contains topic as # title."""
    synthesis = build_synthesis(topic="Test Topic")
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "# Test Topic" in result["final_output"]


@pytest.mark.asyncio
async def test_markdown_contains_executive_summary_heading() -> None:
    """Test #4: markdown output contains ## Executive Summary heading."""
    state = build_state(synthesis=build_synthesis())
    result = await delivery_node(state)
    assert "## Executive Summary" in result["final_output"]


@pytest.mark.asyncio
async def test_markdown_contains_all_finding_headings() -> None:
    """Test #5: all finding headings appear as ### in output."""
    findings = [
        Finding(
            heading="Finding Alpha",
            body="Body Alpha",
            evidence_refs=["https://example.com/1"],
        ),
        Finding(
            heading="Finding Beta",
            body="Body Beta",
            evidence_refs=["https://example.com/2"],
        ),
    ]
    citations = [
        build_evidence(
            source_url="https://example.com/1", source_title="Source 1"
        ),
        build_evidence(
            source_url="https://example.com/2", source_title="Source 2"
        ),
    ]
    synthesis = build_synthesis(findings=findings, citations=citations)
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    output = result["final_output"]
    assert "### Finding Alpha" in output
    assert "### Finding Beta" in output


@pytest.mark.asyncio
async def test_findings_have_footnote_refs() -> None:
    """Test #6: [^1] footnote refs appear in finding body."""
    findings = [
        Finding(
            heading="Finding 1",
            body="Body text",
            evidence_refs=["https://example.com"],
        )
    ]
    citations = [build_evidence(source_url="https://example.com")]
    synthesis = build_synthesis(findings=findings, citations=citations)
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "[^1]" in result["final_output"]


@pytest.mark.asyncio
async def test_risks_section_present_when_nonempty() -> None:
    """Test #7: ## Risks section present when risks is non-empty."""
    synthesis = build_synthesis(risks=["Risk A", "Risk B"])
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "## Risks" in result["final_output"]


@pytest.mark.asyncio
async def test_gaps_section_present_when_nonempty() -> None:
    """Test #8: ## Known Gaps section present when gaps is non-empty."""
    synthesis = build_synthesis(gaps=["Gap X", "Gap Y"])
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "## Known Gaps" in result["final_output"]


@pytest.mark.asyncio
async def test_citations_section_present() -> None:
    """Test #9: ## References and [^1]: appear in citations section."""
    state = build_state(synthesis=build_synthesis())
    result = await delivery_node(state)
    output = result["final_output"]
    assert "## References" in output
    assert "[^1]:" in output


@pytest.mark.asyncio
async def test_quality_badge_pass() -> None:
    """Test #10: 'Quality: PASS' when critic_output.passed=True."""
    critic = build_critic(passed=True, score=4.2)
    state = build_state(synthesis=build_synthesis(), critic_output=critic)
    result = await delivery_node(state)
    assert "Quality: PASS" in result["final_output"]


@pytest.mark.asyncio
async def test_quality_badge_review_recommended() -> None:
    """Test #11: 'Quality: REVIEW RECOMMENDED' when force-delivered."""
    critic = build_critic(passed=False, score=3.5)
    state = build_state(
        synthesis=build_synthesis(),
        critic_output=critic,
        revision_count=2,  # max revisions reached
    )
    result = await delivery_node(state)
    assert "Quality: REVIEW RECOMMENDED" in result["final_output"]


@pytest.mark.asyncio
async def test_force_delivery_caveat_injected_at_top() -> None:
    """Test #12: 'QUALITY NOTICE:' in first non-empty line when force-delivered."""
    critic = build_critic(passed=False, score=3.5)
    state = build_state(
        synthesis=build_synthesis(),
        critic_output=critic,
        revision_count=2,
    )
    result = await delivery_node(state)
    output = result["final_output"]
    # Caveat should be at the very top
    assert "QUALITY NOTICE:" in output
    # Verify it comes before the title
    caveat_pos = output.index("QUALITY NOTICE:")
    title_pos = output.index("# Test Topic")
    assert caveat_pos < title_pos


@pytest.mark.asyncio
async def test_force_delivery_caveat_contains_score() -> None:
    """Test #13: caveat string contains score '3.50/5.0'."""
    critic = build_critic(passed=False, score=3.50)
    state = build_state(
        synthesis=build_synthesis(),
        critic_output=critic,
        revision_count=2,
    )
    result = await delivery_node(state)
    assert "3.50/5.0" in result["final_output"]


@pytest.mark.asyncio
async def test_no_caveat_when_critic_passed() -> None:
    """Test #14: no 'QUALITY NOTICE' when critic passed."""
    critic = build_critic(passed=True, score=4.5)
    state = build_state(
        synthesis=build_synthesis(),
        critic_output=critic,
        revision_count=2,
    )
    result = await delivery_node(state)
    assert "QUALITY NOTICE" not in result["final_output"]


@pytest.mark.asyncio
async def test_no_caveat_when_revision_under_cap() -> None:
    """Test #15: no caveat when revision_count=1, passed=False (not force-delivered)."""
    critic = build_critic(passed=False, score=3.5)
    state = build_state(
        synthesis=build_synthesis(),
        critic_output=critic,
        revision_count=1,  # under cap (2)
    )
    result = await delivery_node(state)
    assert "QUALITY NOTICE" not in result["final_output"]


@pytest.mark.asyncio
async def test_synthesis_none_raises_value_error() -> None:
    """Test #16: ValueError raised when synthesis is None."""
    state = build_state(synthesis=None)
    with pytest.raises(ValueError, match="synthesis.*is None"):
        await delivery_node(state)


@pytest.mark.asyncio
async def test_output_format_markdown_produces_output() -> None:
    """Test #17: markdown format produces non-empty output."""
    state = build_state(synthesis=build_synthesis(), output_format="markdown")
    result = await delivery_node(state)
    assert len(result["final_output"]) > 0


@pytest.mark.asyncio
async def test_output_format_docx_raises_not_implemented() -> None:
    """Test #18: NotImplementedError raised for docx format."""
    state = build_state(synthesis=build_synthesis(), output_format="docx")
    with pytest.raises(NotImplementedError, match="DOCX"):
        await delivery_node(state)


@pytest.mark.asyncio
async def test_output_format_pdf_raises_not_implemented() -> None:
    """Test #19: NotImplementedError raised for pdf format."""
    state = build_state(synthesis=build_synthesis(), output_format="pdf")
    with pytest.raises(NotImplementedError, match="PDF"):
        await delivery_node(state)


@pytest.mark.asyncio
async def test_output_format_pptx_raises_not_implemented() -> None:
    """Test #20: NotImplementedError raised for pptx format."""
    state = build_state(synthesis=build_synthesis(), output_format="pptx")
    with pytest.raises(NotImplementedError, match="PPTX"):
        await delivery_node(state)


@pytest.mark.asyncio
async def test_render_markdown_is_pure() -> None:
    """Test #21: render_markdown is pure — same args produce identical output."""
    synthesis = build_synthesis()
    critic = build_critic(passed=True, score=4.5)
    is_force_delivered = False

    output1 = render_markdown(synthesis, critic, is_force_delivered)
    output2 = render_markdown(synthesis, critic, is_force_delivered)

    assert output1 == output2


@pytest.mark.asyncio
async def test_citations_format_url_and_title() -> None:
    """Test #22: citations in References use [Title](URL) pattern."""
    citations = [
        build_evidence(
            source_url="https://example.com/article",
            source_title="Example Article",
        )
    ]
    synthesis = build_synthesis(citations=citations)
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    output = result["final_output"]
    # Expected format: [^1]: [Example Article](https://example.com/article)
    assert "[Example Article](https://example.com/article)" in output


@pytest.mark.asyncio
async def test_empty_risks_omits_section() -> None:
    """Test #23: '## Risks' NOT in output when risks=[]."""
    synthesis = build_synthesis(risks=[])
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "## Risks" not in result["final_output"]


@pytest.mark.asyncio
async def test_empty_gaps_omits_section() -> None:
    """Test #24: '## Known Gaps' NOT in output when gaps=[]."""
    synthesis = build_synthesis(gaps=[])
    state = build_state(synthesis=synthesis)
    result = await delivery_node(state)
    assert "## Known Gaps" not in result["final_output"]

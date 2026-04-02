"""
tests/test_critic.py

Unit tests for agents/critic.py.

Tests the critic node's adversarial 5-dimension evaluation, weighted scoring,
issue/suggestion validation, retry behavior, and cache control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_none

from agents.critic import (
    _build_messages,
    _compute_final_score,
    _load_prompt,
    _parse_critic_output,
    _RawCriticOutput,
    critic_node,
)
from graph.state import (
    CriticIssue,
    CriticOutput,
    CriticSuggestion,
    Evidence,
    Finding,
    ResearchAgentState,
    SynthesisOutput,
    WorkerResult,
)
from graph.utils import AgentParseError


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_evidence(
    source_url: str = "http://example.com/1",
    claim: str = "Test claim about renewables.",
) -> Evidence:
    """Create a minimal Evidence dataclass."""
    return Evidence(
        claim=claim,
        supporting_chunks=["verbatim quote from source"],
        source_url=source_url,
        source_title="Example Article",
        published_date="2026-01-15",
        confidence=0.85,
        contradiction_score=0.0,
    )


def make_synthesis(findings: list[Finding] | None = None) -> SynthesisOutput:
    """Create a SynthesisOutput with specified findings."""
    if findings is None:
        findings = [
            Finding(
                heading="Test Finding",
                body="Test body",
                evidence_refs=["http://example.com/1"],
            )
        ]
    return SynthesisOutput(
        topic="Test Topic",
        executive_summary="Test summary.",
        findings=findings,
        risks=["Risk 1"],
        gaps=["Gap 1"],
        citations=[make_evidence()],
        synthesis_version=1,
        prior_attempt_summary=None,
    )


_UNSET = object()


def make_state(
    synthesis: SynthesisOutput | None | object = _UNSET,
    grounding_issues: list[str] | None = None,
    total_tokens_used: int = 0,
) -> ResearchAgentState:
    """Create a minimal ResearchAgentState for critic tests."""
    actual_synthesis: SynthesisOutput | None
    if synthesis is _UNSET:
        actual_synthesis = make_synthesis()
    else:
        actual_synthesis = synthesis  # type: ignore[assignment]
    if grounding_issues is None:
        grounding_issues = []
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format="markdown",
        sub_queries=[],
        source_routing={},
        worker_results=[],
        compressed_worker_results=[],
        synthesis=actual_synthesis,
        grounding_issues=grounding_issues,
        critic_output=None,
        revision_count=0,
        prior_syntheses=[],
        session_id="test-session",
        total_tokens_used=total_tokens_used,
        cost_usd=0.0,
        final_output=None,
    )


def make_runtime(llm: Any = None) -> MagicMock:
    """Create a mock Runtime[NofrinContext] with a mock LLM."""
    if llm is None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
    runtime = MagicMock()
    runtime.context.llm_critic = llm
    return runtime


def llm_response_json(data: dict[str, Any], tokens: int = 50) -> MagicMock:
    """Build a valid LLM response with usage_metadata."""
    response = MagicMock()
    response.content = json.dumps(data)
    response.usage_metadata = {"total_tokens": tokens}
    return response


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_node_returns_critic_output() -> None:
    """1. result contains "critic_output" key as CriticOutput."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            {
                "factuality_score": 4.5,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    assert "critic_output" in result
    assert isinstance(result["critic_output"], CriticOutput)


@pytest.mark.asyncio
async def test_final_score_computed_not_from_llm() -> None:
    """2. LLM JSON has no final_quality_score; code computes it from the 5 scores."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        # Intentionally omit final_quality_score in LLM response
        return llm_response_json(
            {
                "factuality_score": 5.0,
                "citation_alignment_score": 5.0,
                "reasoning_score": 5.0,
                "completeness_score": 5.0,
                "bias_score": 5.0,
                "issues": [],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    critic_output = result["critic_output"]
    assert isinstance(critic_output, CriticOutput)
    # Should be computed via _compute_final_score, not from LLM
    assert critic_output.final_quality_score == 5.0


def test_weighted_formula_correct() -> None:
    """3. scores (5,5,5,5,5) → final=5.0; scores (4,4,4,4,4) → final=4.0."""
    # Test case 1: All 5.0
    raw1: _RawCriticOutput = {
        "factuality_score": 5.0,
        "citation_alignment_score": 5.0,
        "reasoning_score": 5.0,
        "completeness_score": 5.0,
        "bias_score": 5.0,
        "issues": [],
        "suggestions": [],
    }
    final1 = _compute_final_score(raw1)
    assert final1 == 5.0

    # Test case 2: All 4.0
    raw2: _RawCriticOutput = {
        "factuality_score": 4.0,
        "citation_alignment_score": 4.0,
        "reasoning_score": 4.0,
        "completeness_score": 4.0,
        "bias_score": 4.0,
        "issues": [],
        "suggestions": [],
    }
    final2 = _compute_final_score(raw2)
    assert final2 == 4.0


@pytest.mark.asyncio
async def test_passed_true_when_score_gte_4() -> None:
    """4. final=4.075 → critic_output.passed == True."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        # Weighted: 0.30*4.5 + 0.25*4.0 + 0.20*4.0 + 0.15*4.0 + 0.10*3.0 = 4.15
        return llm_response_json(
            {
                "factuality_score": 4.5,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 3.0,
                "issues": [],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    critic_output = result["critic_output"]
    assert isinstance(critic_output, CriticOutput)
    assert critic_output.final_quality_score >= 4.0
    assert critic_output.passed is True


@pytest.mark.asyncio
async def test_passed_false_when_score_lt_4() -> None:
    """5. final=3.5 → critic_output.passed == False."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        # Weighted: 0.30*3.0 + 0.25*3.5 + 0.20*3.5 + 0.15*4.0 + 0.10*4.0 = 3.475
        return llm_response_json(
            {
                "factuality_score": 3.0,
                "citation_alignment_score": 3.5,
                "reasoning_score": 3.5,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    critic_output = result["critic_output"]
    assert isinstance(critic_output, CriticOutput)
    assert critic_output.final_quality_score < 4.0
    assert critic_output.passed is False


def test_dimension_scores_clamped_to_0_5() -> None:
    """6. LLM returns score=6.0 → clamped to 5.0; score=-1.0 → 0.0."""
    raw_json = json.dumps(
        {
            "factuality_score": 6.0,  # Should be clamped to 5.0
            "citation_alignment_score": -1.0,  # Should be clamped to 0.0
            "reasoning_score": 4.0,
            "completeness_score": 3.5,
            "bias_score": 10.0,  # Should be clamped to 5.0
            "issues": [],
            "suggestions": [],
        }
    )

    critic_output = _parse_critic_output(raw_json)

    assert critic_output.factuality_score == 5.0
    assert critic_output.citation_alignment_score == 0.0
    assert critic_output.bias_score == 5.0


@pytest.mark.asyncio
async def test_critic_issues_are_criticissue_dataclasses() -> None:
    """7. all(isinstance(i, CriticIssue) for i in critic_output.issues)."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            {
                "factuality_score": 4.0,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [
                    {
                        "issue_text": "Issue 1",
                        "quote_from_synthesis": "Quote 1",
                        "severity": "critical",
                    },
                    {
                        "issue_text": "Issue 2",
                        "quote_from_synthesis": "Quote 2",
                        "severity": "minor",
                    },
                ],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    critic_output = result["critic_output"]
    assert isinstance(critic_output, CriticOutput)
    assert len(critic_output.issues) == 2
    assert all(isinstance(i, CriticIssue) for i in critic_output.issues)


@pytest.mark.asyncio
async def test_critic_suggestions_are_criticsuggesion_dataclasses() -> None:
    """8. all(isinstance(s, CriticSuggestion) for s in critic_output.suggestions)."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            {
                "factuality_score": 4.0,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [],
                "suggestions": [
                    {
                        "action": "Add more evidence",
                        "target_section": "Finding 1",
                        "new_evidence_needed": True,
                    },
                    {
                        "action": "Clarify claim",
                        "target_section": "Finding 2",
                        "new_evidence_needed": False,
                    },
                ],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    critic_output = result["critic_output"]
    assert isinstance(critic_output, CriticOutput)
    assert len(critic_output.suggestions) == 2
    assert all(isinstance(s, CriticSuggestion) for s in critic_output.suggestions)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_severity_raises_agent_parse_error() -> None:
    """9. issue.severity="INVALID" → AgentParseError."""
    raw_json = json.dumps(
        {
            "factuality_score": 4.0,
            "citation_alignment_score": 4.0,
            "reasoning_score": 4.0,
            "completeness_score": 4.0,
            "bias_score": 4.0,
            "issues": [
                {
                    "issue_text": "Test issue",
                    "quote_from_synthesis": "Test quote",
                    "severity": "INVALID",
                }
            ],
            "suggestions": [],
        }
    )

    with pytest.raises(AgentParseError, match="invalid severity"):
        _parse_critic_output(raw_json)


def test_empty_issue_text_raises_agent_parse_error() -> None:
    """10. issue_text="" → AgentParseError."""
    raw_json = json.dumps(
        {
            "factuality_score": 4.0,
            "citation_alignment_score": 4.0,
            "reasoning_score": 4.0,
            "completeness_score": 4.0,
            "bias_score": 4.0,
            "issues": [
                {
                    "issue_text": "",
                    "quote_from_synthesis": "Test quote",
                    "severity": "critical",
                }
            ],
            "suggestions": [],
        }
    )

    with pytest.raises(AgentParseError, match="empty issue_text"):
        _parse_critic_output(raw_json)


def test_empty_quote_raises_agent_parse_error() -> None:
    """11. quote_from_synthesis="" → AgentParseError."""
    raw_json = json.dumps(
        {
            "factuality_score": 4.0,
            "citation_alignment_score": 4.0,
            "reasoning_score": 4.0,
            "completeness_score": 4.0,
            "bias_score": 4.0,
            "issues": [
                {
                    "issue_text": "Test issue",
                    "quote_from_synthesis": "",
                    "severity": "critical",
                }
            ],
            "suggestions": [],
        }
    )

    with pytest.raises(AgentParseError, match="empty quote_from_synthesis"):
        _parse_critic_output(raw_json)


@pytest.mark.asyncio
async def test_synthesis_none_raises_value_error() -> None:
    """12. synthesis=None → ValueError before LLM call."""
    state = make_state(synthesis=None)
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    runtime = make_runtime(llm)

    with pytest.raises(ValueError, match="synthesis"):
        await critic_node(state, runtime)

    assert llm.ainvoke.call_count == 0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


def test_grounding_issues_appear_in_serialized_context() -> None:
    """13. non-empty grounding_issues → block in message text."""
    from agents.critic import _serialize_grounding_issues

    issues = ["[UNSUPPORTED] Finding: 'X' — desc", "[MISSING_CITATION] Finding: 'Y' — desc"]
    result = _serialize_grounding_issues(issues)

    assert "[UNSUPPORTED]" in result
    assert "[MISSING_CITATION]" in result
    assert "1." in result
    assert "2." in result


def test_empty_grounding_issues_shows_clean_message() -> None:
    """14. grounding_issues=[] → "(none — grounding check passed clean)" in message."""
    from agents.critic import _serialize_grounding_issues

    result = _serialize_grounding_issues([])

    assert "(none — grounding check passed clean)" in result


def test_build_messages_anthropic_cache_control() -> None:
    """15. use_cache_control=True → list content with cache_control block."""
    messages = _build_messages(
        synthesis_block="test synthesis",
        grounding_issues_block="test issues",
        prompt_template="test prompt template",
        use_cache_control=True,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2

    static_block = content[0]
    dynamic_block = content[1]

    assert isinstance(static_block, dict)
    assert static_block.get("cache_control") == {"type": "ephemeral"}
    assert static_block.get("text") == "test prompt template"

    assert isinstance(dynamic_block, dict)
    assert "cache_control" not in dynamic_block
    assert "test synthesis" in str(dynamic_block.get("text", ""))


def test_build_messages_groq_no_cache_control() -> None:
    """16. use_cache_control=False → plain str SystemMessage."""
    messages = _build_messages(
        synthesis_block="test synthesis",
        grounding_issues_block="test issues",
        prompt_template="template with {{synthesis_block}} and {{grounding_issues_block}}",
        use_cache_control=False,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert isinstance(content, str)
    assert "test synthesis" in content
    assert "test issues" in content


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("agents.critic._call_llm.retry.wait", new=wait_none())
async def test_retry_on_parse_failure() -> None:
    """17. LLM returns bad JSON once → retried → valid on 2nd call (2 ainvoke calls)."""
    state = make_state()
    call_count = 0

    async def mock_ainvoke(messages: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            response = MagicMock()
            response.content = "{ invalid json"
            response.usage_metadata = {"total_tokens": 50}
            return response
        return llm_response_json(
            {
                "factuality_score": 4.0,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [],
                "suggestions": [],
            }
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    assert call_count == 2
    assert "critic_output" in result


@pytest.mark.asyncio
@patch("agents.critic._call_llm.retry.wait", new=wait_none())
async def test_raises_after_max_retries() -> None:
    """18. LLM always returns bad JSON → AgentParseError after 3 attempts."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        response = MagicMock()
        response.content = "{ invalid json"
        response.usage_metadata = {"total_tokens": 50}
        return response

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    with pytest.raises(AgentParseError):
        await critic_node(state, runtime)

    assert llm.ainvoke.call_count == 3


# ---------------------------------------------------------------------------
# Token tracking test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_tokens_accumulated() -> None:
    """19. state has 200 tokens; LLM response has 50 tokens → result is 250."""
    state = make_state(total_tokens_used=200)

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            {
                "factuality_score": 4.0,
                "citation_alignment_score": 4.0,
                "reasoning_score": 4.0,
                "completeness_score": 4.0,
                "bias_score": 4.0,
                "issues": [],
                "suggestions": [],
            },
            tokens=50,
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await critic_node(state, runtime)

    assert result["total_tokens_used"] == 250


# ---------------------------------------------------------------------------
# Load prompt test
# ---------------------------------------------------------------------------


def test_load_prompt_missing_file_raises() -> None:
    """20. _load_prompt(nonexistent_path) → FileNotFoundError."""
    nonexistent_path = Path("/nonexistent/path/critic_v1.txt")

    with pytest.raises(FileNotFoundError, match="Critic prompt not found"):
        _load_prompt(nonexistent_path)

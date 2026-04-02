"""
tests/test_grounding_check.py

Unit tests for agents/grounding_check.py.

Tests the grounding check node's validation of synthesis findings against
worker evidence, retry behavior, cache control, and issue detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.grounding_check import (
    _build_messages,
    _load_prompt,
    _parse_grounding_output,
    grounding_check_node,
)
from graph.state import (
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


def make_worker_result(evidences: list[Evidence] | None = None) -> WorkerResult:
    """Create a WorkerResult with specified evidence items."""
    if evidences is None:
        evidences = [make_evidence()]
    return WorkerResult(
        worker_id="worker-1",
        sub_query="test sub query",
        source_type="web",
        evidence_items=evidences,
        raw_search_results=[],
        tokens_used=100,
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
        risks=[],
        gaps=[],
        citations=[make_evidence()],
        synthesis_version=1,
        prior_attempt_summary=None,
    )


_UNSET = object()


def make_state(
    synthesis: SynthesisOutput | None | object = _UNSET,
    worker_results: list[WorkerResult] | None = None,
) -> ResearchAgentState:
    """Create a minimal ResearchAgentState for grounding check tests."""
    actual_synthesis: SynthesisOutput | None
    if synthesis is _UNSET:
        actual_synthesis = make_synthesis()
    else:
        actual_synthesis = synthesis  # type: ignore[assignment]
    if worker_results is None:
        worker_results = [make_worker_result()]
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format="markdown",
        sub_queries=[],
        source_routing={},
        worker_results=[],
        compressed_worker_results=worker_results,
        synthesis=actual_synthesis,
        grounding_issues=[],
        critic_output=None,
        revision_count=0,
        prior_syntheses=[],
        session_id="test-session",
        total_tokens_used=0,
        cost_usd=0.0,
        final_output=None,
    )


def make_runtime(llm: Any = None) -> MagicMock:
    """Create a mock Runtime[NofrinContext] with a mock LLM."""
    if llm is None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
    runtime = MagicMock()
    runtime.context.llm_coordinator = llm
    return runtime


def llm_response_json(issues: list[dict[str, str]]) -> MagicMock:
    """Build a valid LLM response with specified issues."""
    response = MagicMock()
    response.content = json.dumps({"issues": issues})
    return response


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_issues_returns_empty_list() -> None:
    """1. LLM returns {"issues":[]} → result["grounding_issues"] == []."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json([])

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert "grounding_issues" in result
    assert result["grounding_issues"] == []


@pytest.mark.asyncio
async def test_unsupported_issue_detected() -> None:
    """2. LLM returns UNSUPPORTED issue → "[UNSUPPORTED]" in the first grounding_issue string."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            [
                {
                    "type": "UNSUPPORTED",
                    "finding_heading": "Test Finding",
                    "description": "This claim lacks supporting evidence.",
                }
            ]
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert len(result["grounding_issues"]) == 1
    assert "[UNSUPPORTED]" in result["grounding_issues"][0]


@pytest.mark.asyncio
async def test_hallucinated_citation_detected() -> None:
    """3. LLM returns HALLUCINATED_CITATION → "[HALLUCINATED_CITATION]" in result."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            [
                {
                    "type": "HALLUCINATED_CITATION",
                    "finding_heading": "Test Finding",
                    "description": "Citation URL does not support the claim.",
                }
            ]
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert len(result["grounding_issues"]) == 1
    assert "[HALLUCINATED_CITATION]" in result["grounding_issues"][0]


@pytest.mark.asyncio
async def test_missing_citation_detected() -> None:
    """4. LLM returns MISSING_CITATION → "[MISSING_CITATION]" in result."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            [
                {
                    "type": "MISSING_CITATION",
                    "finding_heading": "Test Finding",
                    "description": "Claim needs a citation.",
                }
            ]
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert len(result["grounding_issues"]) == 1
    assert "[MISSING_CITATION]" in result["grounding_issues"][0]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_none_raises_value_error() -> None:
    """5. synthesis=None → ValueError before any LLM call (assert llm.ainvoke NOT called)."""
    state = make_state(synthesis=None)
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    runtime = make_runtime(llm)

    with pytest.raises(ValueError, match="synthesis"):
        await grounding_check_node(state, runtime)

    assert llm.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_empty_worker_results_raises_value_error() -> None:
    """6. compressed_worker_results=[] → ValueError before LLM call."""
    state = make_state(worker_results=[])
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    runtime = make_runtime(llm)

    with pytest.raises(ValueError, match="compressed_worker_results is empty"):
        await grounding_check_node(state, runtime)

    assert llm.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_evidence_ref_not_in_worker_results_treated_gracefully() -> None:
    """7. finding has a URL not in any evidence; LLM called and returns clean → no crash, grounding_issues == []."""
    ev = make_evidence(source_url="http://real.com")
    synthesis = make_synthesis(
        findings=[
            Finding(
                heading="Test",
                body="Body",
                evidence_refs=["http://not-in-worker-results.com"],
            )
        ]
    )
    state = make_state(synthesis=synthesis, worker_results=[make_worker_result([ev])])

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json([])

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert result["grounding_issues"] == []


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_triggers_retry() -> None:
    """8. LLM returns bad JSON once → retried → valid on 2nd call → 2 ainvoke calls total."""
    state = make_state()
    call_count = 0

    async def mock_ainvoke(messages: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            response = MagicMock()
            response.content = "{ invalid json"
            return response
        return llm_response_json([])

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert call_count == 2
    assert result["grounding_issues"] == []


@pytest.mark.asyncio
async def test_raises_after_max_retries() -> None:
    """9. LLM always returns bad JSON → AgentParseError after exactly 3 ainvoke calls."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        response = MagicMock()
        response.content = "{ invalid json"
        return response

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    with pytest.raises(AgentParseError):
        await grounding_check_node(state, runtime)

    assert llm.ainvoke.call_count == 3


# ---------------------------------------------------------------------------
# Cache control tests
# ---------------------------------------------------------------------------


def test_build_messages_anthropic_cache_control() -> None:
    """10. _build_messages(block, template, use_cache_control=True) → SystemMessage content is list, first block has cache_control."""
    messages = _build_messages(
        findings_with_evidence_block="test block",
        prompt_template="test prompt",
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
    assert static_block.get("text") == "test prompt"

    assert isinstance(dynamic_block, dict)
    assert "cache_control" not in dynamic_block
    assert "test block" in str(dynamic_block.get("text", ""))


def test_build_messages_groq_no_cache_control() -> None:
    """11. _build_messages(block, template, use_cache_control=False) → SystemMessage content is plain str."""
    messages = _build_messages(
        findings_with_evidence_block="test block",
        prompt_template="template with {{findings_with_evidence_block}}",
        use_cache_control=False,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert isinstance(content, str)
    assert "test block" in content
    assert "cache_control" not in content


# ---------------------------------------------------------------------------
# Multiple findings tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_findings_all_clean() -> None:
    """12. state has 3 findings; LLM returns no issues → grounding_issues == []."""
    findings = [
        Finding(heading="F1", body="B1", evidence_refs=["http://example.com/1"]),
        Finding(heading="F2", body="B2", evidence_refs=["http://example.com/2"]),
        Finding(heading="F3", body="B3", evidence_refs=["http://example.com/3"]),
    ]
    evidences = [
        make_evidence(source_url="http://example.com/1"),
        make_evidence(source_url="http://example.com/2"),
        make_evidence(source_url="http://example.com/3"),
    ]
    synthesis = make_synthesis(findings=findings)
    state = make_state(
        synthesis=synthesis, worker_results=[make_worker_result(evidences)]
    )

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json([])

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert result["grounding_issues"] == []


@pytest.mark.asyncio
async def test_multiple_findings_some_issues_accumulated() -> None:
    """13. LLM returns 2 issues → both strings in grounding_issues list."""
    state = make_state()

    async def mock_ainvoke(messages: Any) -> Any:
        return llm_response_json(
            [
                {
                    "type": "UNSUPPORTED",
                    "finding_heading": "F1",
                    "description": "Issue 1",
                },
                {
                    "type": "MISSING_CITATION",
                    "finding_heading": "F2",
                    "description": "Issue 2",
                },
            ]
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert len(result["grounding_issues"]) == 2
    assert "[UNSUPPORTED]" in result["grounding_issues"][0]
    assert "[MISSING_CITATION]" in result["grounding_issues"][1]


# ---------------------------------------------------------------------------
# Fast path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_findings_skips_llm_call() -> None:
    """14. synthesis.findings == [] → {"grounding_issues": []} returned, LLM NOT called."""
    synthesis = make_synthesis(findings=[])
    state = make_state(synthesis=synthesis)
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    runtime = make_runtime(llm)

    result = await grounding_check_node(state, runtime)

    assert result["grounding_issues"] == []
    assert llm.ainvoke.call_count == 0


# ---------------------------------------------------------------------------
# Load prompt test
# ---------------------------------------------------------------------------


def test_load_prompt_missing_file_raises() -> None:
    """15. _load_prompt(Path("/nonexistent/path/gc_v99.txt")) → FileNotFoundError with message containing "Grounding check prompt not found"."""
    nonexistent_path = Path("/nonexistent/path/gc_v99.txt")

    with pytest.raises(FileNotFoundError, match="Grounding check prompt not found"):
        _load_prompt(nonexistent_path)


# ---------------------------------------------------------------------------
# Parse validation tests
# ---------------------------------------------------------------------------


def test_invalid_issue_type_raises_agent_parse_error() -> None:
    """16. LLM returns {"issues":[{"type":"FABRICATED","finding_heading":"H","description":"D"}]} → AgentParseError (invalid type)."""
    raw = json.dumps(
        {
            "issues": [
                {
                    "type": "FABRICATED",
                    "finding_heading": "Heading",
                    "description": "Description",
                }
            ]
        }
    )

    with pytest.raises(AgentParseError, match="invalid type"):
        _parse_grounding_output(raw)

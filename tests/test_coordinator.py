"""
tests/test_coordinator.py

Unit tests for agents/coordinator.py.

Tests the coordinator node's synthesis generation, revision logic,
validation, retry behavior, citation mapping, and cache control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.coordinator import (
    FIRST_PASS_PROMPT_PATH,
    REVISION_PROMPT_PATH,
    _EVIDENCE_CHAR_CAP,
    _build_first_pass_messages,
    _build_revision_messages,
    _load_prompt,
    _parse_and_validate,
    _serialize_evidence,
    coordinator_node,
)
from graph.state import (
    CriticOutput,
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
    confidence: float = 0.85,
) -> Evidence:
    """Create a minimal Evidence dataclass."""
    return Evidence(
        claim=claim,
        supporting_chunks=["verbatim quote from source"],
        source_url=source_url,
        source_title="Example Article",
        published_date="2026-01-15",
        confidence=confidence,
        contradiction_score=0.0,
    )


def make_worker_result(
    sub_query: str = "test sub query",
    evidences: list[Evidence] | None = None,
) -> WorkerResult:
    """Create a WorkerResult with specified evidence items."""
    if evidences is None:
        evidences = [make_evidence()]
    return WorkerResult(
        worker_id="worker-1",
        sub_query=sub_query,
        source_type="web",
        evidence_items=evidences,
        raw_search_results=[],
        tokens_used=100,
    )


def make_state(
    revision_count: int = 0,
    compressed_worker_results: list[WorkerResult] | None = None,
    synthesis: SynthesisOutput | None = None,
    prior_syntheses: list[SynthesisOutput] | None = None,
    critic_output: CriticOutput | None = None,
    total_tokens_used: int = 0,
) -> ResearchAgentState:
    """Create a minimal ResearchAgentState for coordinator tests."""
    if compressed_worker_results is None:
        compressed_worker_results = [make_worker_result()]
    if prior_syntheses is None:
        prior_syntheses = []
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format="markdown",
        research_mode="research",
        sub_queries=[],
        source_routing={},
        worker_results=[],
        compressed_worker_results=compressed_worker_results,
        synthesis=synthesis,
        grounding_issues=[],
        critic_output=critic_output,
        revision_count=revision_count,
        prior_syntheses=prior_syntheses,
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
    runtime.context.llm_coordinator = llm
    return runtime


def valid_llm_response_json(
    worker_results: list[WorkerResult],
    topic: str = "Test Topic",
    executive_summary: str = "This is a test executive summary.",
) -> str:
    """Build valid JSON string referencing worker_results URLs."""
    available_urls = [
        ev.source_url for wr in worker_results for ev in wr.evidence_items
    ]
    if not available_urls:
        available_urls = ["http://example.com/1"]

    return json.dumps(
        {
            "topic": topic,
            "executive_summary": executive_summary,
            "findings": [
                {
                    "heading": "Finding One",
                    "body": "This is the body of finding one.",
                    "evidence_refs": [available_urls[0]],
                }
            ],
            "risks": ["Risk 1"],
            "gaps": ["Gap 1"],
            "citation_urls": available_urls,
        }
    )


def valid_llm_response(
    worker_results: list[WorkerResult],
    topic: str = "Test Topic",
    executive_summary: str = "This is a test executive summary.",
    tokens: int = 10,
) -> MagicMock:
    """Create a mock LLM response with valid JSON referencing worker_results URLs."""
    content = valid_llm_response_json(worker_results, topic, executive_summary)
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"total_tokens": tokens}
    return response


# ---------------------------------------------------------------------------
# PATH 1 tests (first pass, revision_count == 0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_pass_returns_synthesis() -> None:
    """1. result contains 'synthesis' key with a SynthesisOutput."""
    state = make_state(revision_count=0, total_tokens_used=50)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert "synthesis" in result
    assert isinstance(result["synthesis"], SynthesisOutput)


@pytest.mark.asyncio
async def test_first_pass_synthesis_version_is_1() -> None:
    """2. synthesis.synthesis_version == 1."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.synthesis_version == 1


@pytest.mark.asyncio
async def test_first_pass_does_not_modify_prior_syntheses() -> None:
    """3. 'prior_syntheses' NOT in result."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert "prior_syntheses" not in result


@pytest.mark.asyncio
async def test_first_pass_sets_revision_count_to_one() -> None:
    """4. First-pass coordinator sets revision_count=1 so the next call takes PATH 2."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert result["revision_count"] == 1


@pytest.mark.asyncio
async def test_first_pass_accumulates_total_tokens() -> None:
    """5. result['total_tokens_used'] == state_tokens + coordinator_tokens."""
    state = make_state(revision_count=0, total_tokens_used=100)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results, tokens=25)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert result["total_tokens_used"] == 125  # 100 + 25


@pytest.mark.asyncio
async def test_first_pass_uses_first_pass_prompt() -> None:
    """6. _load_prompt called with coordinator_v1.txt path (mock it)."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    with patch(
        "agents.coordinator._load_prompt", return_value="test prompt"
    ) as mock_load:
        await coordinator_node(state, runtime)
        mock_load.assert_called_once_with(FIRST_PASS_PROMPT_PATH)


@pytest.mark.asyncio
async def test_first_pass_prior_attempt_summary_is_none() -> None:
    """7. synthesis.prior_attempt_summary is None."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.prior_attempt_summary is None


# ---------------------------------------------------------------------------
# PATH 2 tests (revision pass, revision_count > 0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_pass_moves_synthesis_to_prior() -> None:
    """8. result['prior_syntheses'][-1] == old synthesis."""
    old_synthesis = SynthesisOutput(
        topic="Old Topic",
        executive_summary="Old summary.",
        findings=[
            Finding(heading="Old", body="Old body", evidence_refs=["http://old.com"])
        ],
        risks=[],
        gaps=[],
        citations=[],
        synthesis_version=1,
        prior_attempt_summary=None,
    )
    state = make_state(revision_count=1, synthesis=old_synthesis)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    prior = result["prior_syntheses"]
    assert isinstance(prior, list)
    assert len(prior) == 1
    assert prior[-1] == old_synthesis


@pytest.mark.asyncio
async def test_revision_pass_increments_revision_count() -> None:
    """9. result['revision_count'] == state revision_count + 1."""
    state = make_state(
        revision_count=1,
        synthesis=SynthesisOutput(
            topic="T",
            executive_summary="S",
            findings=[Finding(heading="H", body="B", evidence_refs=["http://e.com"])],
            risks=[],
            gaps=[],
            citations=[],
            synthesis_version=1,
            prior_attempt_summary=None,
        ),
    )
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert result["revision_count"] == 2


@pytest.mark.asyncio
async def test_revision_pass_synthesis_version_increments() -> None:
    """10. result['synthesis'].synthesis_version == prior.synthesis_version + 1."""
    prior_synthesis = SynthesisOutput(
        topic="Prior",
        executive_summary="Prior summary.",
        findings=[Finding(heading="P", body="P body", evidence_refs=["http://p.com"])],
        risks=[],
        gaps=[],
        citations=[],
        synthesis_version=1,
        prior_attempt_summary=None,
    )
    state = make_state(revision_count=1, synthesis=prior_synthesis)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.synthesis_version == 2


@pytest.mark.asyncio
async def test_revision_pass_uses_revision_prompt() -> None:
    """11. _load_prompt called with coordinator_revision_v1.txt path."""
    prior_synthesis = SynthesisOutput(
        topic="Prior",
        executive_summary="Prior summary.",
        findings=[Finding(heading="P", body="P body", evidence_refs=["http://p.com"])],
        risks=[],
        gaps=[],
        citations=[],
        synthesis_version=1,
        prior_attempt_summary=None,
    )
    state = make_state(revision_count=1, synthesis=prior_synthesis)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    with patch(
        "agents.coordinator._load_prompt", return_value="revision prompt"
    ) as mock_load:
        await coordinator_node(state, runtime)
        mock_load.assert_called_once_with(REVISION_PROMPT_PATH)


@pytest.mark.asyncio
async def test_revision_pass_prior_syntheses_preserves_history() -> None:
    """12. existing prior_syntheses + old synthesis = full updated list."""
    old_synthesis_1 = SynthesisOutput(
        topic="Old 1",
        executive_summary="Summary 1.",
        findings=[Finding(heading="F1", body="B1", evidence_refs=["http://1.com"])],
        risks=[],
        gaps=[],
        citations=[],
        synthesis_version=1,
        prior_attempt_summary=None,
    )
    old_synthesis_2 = SynthesisOutput(
        topic="Old 2",
        executive_summary="Summary 2.",
        findings=[Finding(heading="F2", body="B2", evidence_refs=["http://2.com"])],
        risks=[],
        gaps=[],
        citations=[],
        synthesis_version=2,
        prior_attempt_summary=None,
    )
    state = make_state(
        revision_count=2,
        synthesis=old_synthesis_2,
        prior_syntheses=[old_synthesis_1],
    )
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    prior = result["prior_syntheses"]
    assert isinstance(prior, list)
    assert len(prior) == 2
    assert prior[0] == old_synthesis_1
    assert prior[1] == old_synthesis_2


# ---------------------------------------------------------------------------
# Dataclass conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_findings_are_finding_dataclasses() -> None:
    """13. all(isinstance(f, Finding) for f in synthesis.findings)."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert len(synthesis.findings) > 0
    assert all(isinstance(f, Finding) for f in synthesis.findings)


@pytest.mark.asyncio
async def test_citations_are_evidence_dataclasses() -> None:
    """14. all(isinstance(e, Evidence) for e in synthesis.citations)."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert len(synthesis.citations) > 0
    assert all(isinstance(e, Evidence) for e in synthesis.citations)


@pytest.mark.asyncio
async def test_executive_summary_populated() -> None:
    """15. synthesis.executive_summary != ''."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(
            worker_results,
            executive_summary="This is a populated executive summary.",
        )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.executive_summary != ""
    assert synthesis.executive_summary == "This is a populated executive summary."


@pytest.mark.asyncio
async def test_topic_populated() -> None:
    """16. synthesis.topic != ''."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]

    async def mock_ainvoke(messages: Any) -> Any:
        return valid_llm_response(worker_results, topic="Populated Topic")

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    synthesis = result["synthesis"]
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.topic != ""
    assert synthesis.topic == "Populated Topic"


# ---------------------------------------------------------------------------
# Validation tests (AgentParseError)
# ---------------------------------------------------------------------------


def test_invalid_evidence_ref_raises_agent_parse_error() -> None:
    """17. All evidence_refs invalid -> AgentParseError raised (finding has zero valid refs).
    Mixed case (some valid, some not) should succeed and drop only the bad ones."""
    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [
                {
                    "heading": "Finding",
                    "body": "Body",
                    "evidence_refs": ["http://not-in-evidence.com"],
                }
            ],
            "risks": [],
            "gaps": [],
            "citation_urls": [],
        }
    )
    available_urls: set[str] = {"http://valid.com"}
    evidence_by_url: dict[str, Evidence] = {}

    # All refs are bad → finding ends up with zero valid refs → raises
    with pytest.raises(AgentParseError, match="no valid evidence_refs"):
        _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)


def test_mixed_evidence_refs_drops_bad_keeps_good() -> None:
    """17b. Finding with one valid and one hallucinated ref → bad ref dropped, no crash."""
    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [
                {
                    "heading": "Finding",
                    "body": "Body",
                    "evidence_refs": ["http://valid.com", "http://hallucinated.com"],
                }
            ],
            "risks": [],
            "gaps": [],
            "citation_urls": [],
        }
    )
    available_urls: set[str] = {"http://valid.com"}
    evidence_by_url: dict[str, Evidence] = {}

    result = _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)
    assert result.findings[0].evidence_refs == ["http://valid.com"]


def test_empty_executive_summary_raises_agent_parse_error() -> None:
    """18. executive_summary == '' -> AgentParseError."""
    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "",
            "findings": [
                {
                    "heading": "Finding",
                    "body": "Body",
                    "evidence_refs": ["http://valid.com"],
                }
            ],
            "risks": [],
            "gaps": [],
            "citation_urls": [],
        }
    )
    available_urls: set[str] = {"http://valid.com"}
    evidence_by_url: dict[str, Evidence] = {}

    with pytest.raises(AgentParseError, match="empty executive_summary"):
        _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)


def test_no_findings_raises_agent_parse_error() -> None:
    """19. findings == [] -> AgentParseError."""
    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [],
            "risks": [],
            "gaps": [],
            "citation_urls": [],
        }
    )
    available_urls: set[str] = {"http://valid.com"}
    evidence_by_url: dict[str, Evidence] = {}

    with pytest.raises(AgentParseError, match="0 findings"):
        _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)


def test_finding_with_no_evidence_refs_raises() -> None:
    """20. finding with evidence_refs == [] -> AgentParseError."""
    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [{"heading": "Finding", "body": "Body", "evidence_refs": []}],
            "risks": [],
            "gaps": [],
            "citation_urls": [],
        }
    )
    available_urls: set[str] = {"http://valid.com"}
    evidence_by_url: dict[str, Evidence] = {}

    with pytest.raises(AgentParseError, match="has no evidence_refs"):
        _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)


@pytest.mark.asyncio
async def test_empty_compressed_worker_results_raises_value_error() -> None:
    """21. compressed_worker_results == [] -> ValueError."""
    state = make_state(revision_count=0, compressed_worker_results=[])
    runtime = make_runtime()

    with pytest.raises(ValueError, match="compressed_worker_results is empty"):
        await coordinator_node(state, runtime)


# ---------------------------------------------------------------------------
# Citation mapping tests
# ---------------------------------------------------------------------------


def test_citation_urls_mapped_to_real_evidence_objects() -> None:
    """22. citation_url present in evidence -> Evidence with correct source_url in synthesis.citations."""
    evidence = make_evidence(source_url="http://citation.com", claim="Cited claim.")
    evidence_by_url: dict[str, Evidence] = {"http://citation.com": evidence}
    available_urls: set[str] = {"http://citation.com"}

    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [
                {
                    "heading": "Finding",
                    "body": "Body",
                    "evidence_refs": ["http://citation.com"],
                }
            ],
            "risks": [],
            "gaps": [],
            "citation_urls": ["http://citation.com"],
        }
    )

    synthesis = _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)

    assert len(synthesis.citations) == 1
    assert synthesis.citations[0].source_url == "http://citation.com"
    assert synthesis.citations[0].claim == "Cited claim."


def test_unknown_citation_url_silently_dropped() -> None:
    """23. citation URL not in any WorkerResult evidence -> excluded from citations (no error)."""
    evidence = make_evidence(source_url="http://known.com", claim="Known claim.")
    evidence_by_url: dict[str, Evidence] = {"http://known.com": evidence}
    available_urls: set[str] = {"http://known.com"}

    raw_json = json.dumps(
        {
            "topic": "Test",
            "executive_summary": "Summary.",
            "findings": [
                {
                    "heading": "Finding",
                    "body": "Body",
                    "evidence_refs": ["http://known.com"],
                }
            ],
            "risks": [],
            "gaps": [],
            "citation_urls": ["http://known.com", "http://unknown.com"],
        }
    )

    synthesis = _parse_and_validate(raw_json, available_urls, evidence_by_url, 1, None)

    assert len(synthesis.citations) == 1
    assert synthesis.citations[0].source_url == "http://known.com"


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_parse_failure() -> None:
    """24. LLM returns invalid JSON once -> retried -> succeeds on 2nd call."""
    state = make_state(revision_count=0)
    worker_results = state["compressed_worker_results"]
    call_count = 0

    async def mock_ainvoke(messages: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            response = MagicMock()
            response.content = "{ invalid json"
            response.usage_metadata = {"total_tokens": 10}
            return response
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert call_count == 2
    assert "synthesis" in result


@pytest.mark.asyncio
async def test_raises_after_max_retries() -> None:
    """25. LLM always returns invalid JSON -> AgentParseError after 3 attempts."""
    state = make_state(revision_count=0)

    async def mock_ainvoke(messages: Any) -> Any:
        response = MagicMock()
        response.content = "{ invalid json"
        response.usage_metadata = {"total_tokens": 10}
        return response

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    with pytest.raises(AgentParseError):
        await coordinator_node(state, runtime)

    assert llm.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_retry_on_validation_agentparseerror() -> None:
    """29. First call returns valid JSON but evidence_ref not in evidence (AgentParseError).
    Second call returns valid JSON with correct refs. Verify 2 ainvoke calls.
    """
    ev = make_evidence(source_url="http://real.com", claim="Real claim.")
    worker_results = [make_worker_result(evidences=[ev])]
    state = make_state(revision_count=0, compressed_worker_results=worker_results)
    call_count = 0

    async def mock_ainvoke(messages: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Valid JSON but evidence_ref is NOT in available_urls -> AgentParseError
            bad_json = json.dumps(
                {
                    "topic": "Test",
                    "executive_summary": "Summary.",
                    "findings": [
                        {
                            "heading": "Finding",
                            "body": "Body",
                            "evidence_refs": ["http://hallucinated.com"],
                        }
                    ],
                    "risks": [],
                    "gaps": [],
                    "citation_urls": [],
                }
            )
            response = MagicMock()
            response.content = bad_json
            response.usage_metadata = {"total_tokens": 10}
            return response
        # Second call: correct evidence_ref
        return valid_llm_response(worker_results)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    runtime = make_runtime(llm)

    result = await coordinator_node(state, runtime)

    assert call_count == 2
    assert isinstance(result["synthesis"], SynthesisOutput)


# ---------------------------------------------------------------------------
# Cache control tests
# ---------------------------------------------------------------------------


def test_build_first_pass_messages_anthropic_has_cache_control() -> None:
    """26. _build_first_pass_messages with use_cache_control=True -> content is list with cache_control block."""
    msgs = _build_first_pass_messages(
        evidence_block="evidence text here",
        prompt_template="static prompt template",
        use_cache_control=True,
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, list)
    assert len(content) == 2

    static_block = content[0]
    dynamic_block = content[1]

    assert isinstance(static_block, dict)
    assert static_block.get("cache_control") == {"type": "ephemeral"}
    assert static_block.get("text") == "static prompt template"

    assert isinstance(dynamic_block, dict)
    assert "cache_control" not in dynamic_block
    assert "evidence text here" in str(dynamic_block.get("text", ""))


def test_build_revision_messages_groq_no_cache_control() -> None:
    """27. _build_revision_messages with use_cache_control=False -> SystemMessage content is plain str."""
    msgs = _build_revision_messages(
        evidence_block="evidence",
        prior_synthesis_block="prior",
        critic_issues_block="issues",
        revision_count=1,
        prompt_template=(
            "prompt with {{evidence_block}} {{prior_synthesis_block}} {{critic_issues_block}}"
        ),
        use_cache_control=False,
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, str)
    assert "evidence" in content
    assert "prior" in content
    assert "issues" in content
    assert "cache_control" not in content


# ---------------------------------------------------------------------------
# Load prompt test
# ---------------------------------------------------------------------------


def test_load_prompt_missing_file_raises() -> None:
    """28. _load_prompt with nonexistent path -> FileNotFoundError."""
    nonexistent_path = Path("/nonexistent/path/coordinator_v99.txt")

    with pytest.raises(FileNotFoundError, match="Coordinator prompt not found"):
        _load_prompt(nonexistent_path)


# ---------------------------------------------------------------------------
# _serialize_evidence cap test
# ---------------------------------------------------------------------------


def test_serialize_evidence_caps_per_item_not_mid_sentence() -> None:
    """30. _serialize_evidence stops at whole-item boundaries, never mid-sentence.
    Items that would push total over _EVIDENCE_CHAR_CAP are excluded completely.
    The cap-reached sentinel is appended as a whole line.
    """
    # Build enough evidence items to exceed the cap.
    # Each item serializes to roughly 80-120 chars; _EVIDENCE_CHAR_CAP is 6000.
    long_claim = "A" * 200  # 200-char claim per item, ~240 chars per serialized line
    evidences = [
        make_evidence(
            source_url=f"http://example.com/{i}",
            claim=long_claim,
        )
        for i in range(30)  # 30 * 240 = 7200 chars, exceeds 6000
    ]
    wr = make_worker_result(evidences=evidences)
    output = _serialize_evidence([wr])

    # Cap reached message must be present
    assert "[cap reached" in output

    # The output must NOT exceed cap by more than one item's worth
    # (it stops before adding the item that would exceed, then adds the sentinel)
    lines = output.splitlines()

    # No line should be a mid-word truncation of an evidence item
    for line in lines:
        if line.startswith("[E"):
            # Evidence lines must contain all three parts: CLAIM, CONFIDENCE, URL
            assert "CLAIM:" in line
            assert "CONFIDENCE:" in line
            assert "URL:" in line

    # Total output length should be close to (but not exceed) cap + sentinel length
    assert len(output) < _EVIDENCE_CHAR_CAP + 200  # 200 chars slack for the sentinel

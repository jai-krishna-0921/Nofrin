"""
tests/test_worker.py

Unit tests for agents/worker.py.

Tests cover: node output structure, Exa parameter routing, evidence
extraction quality, partial/full failure handling, retry behavior,
param builder unit tests, invalid source_type, and gather resilience.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.worker import (
    _build_exa_params_academic,
    _build_exa_params_news,
    _build_exa_params_web,
    _build_extraction_messages,
    _extract_all_evidence,
    _get_exa_params,
    worker_node,
)
from graph.context import NofrinContext
from graph.state import Evidence, WorkerInput
from graph.utils import AgentParseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_exa_result(
    url: str = "http://example.com/article",
    title: str = "Test Article",
    text: str | None = "Relevant content about the research topic.",
    highlights: list[str] | None = None,
    published_date: str | None = "2026-01-15",
) -> MagicMock:
    """Create a mock Exa search result with the given attributes."""
    result = MagicMock()
    result.url = url
    result.title = title
    result.text = text
    result.highlights = highlights
    result.published_date = published_date
    result.score = 0.9
    result.id = "exa-result-123"
    return result


def make_llm_response(
    claim: str = "The research topic shows significant advances.",
    supporting_chunks: list[str] | None = None,
    confidence: float = 0.85,
    contradiction_score: float = 0.0,
    tokens: int = 100,
) -> MagicMock:
    """Create a mock LLM response that returns valid evidence extraction JSON."""
    if supporting_chunks is None:
        supporting_chunks = ["verbatim supporting quote from the text"]
    content = json.dumps(
        {
            "claim": claim,
            "supporting_chunks": supporting_chunks,
            "confidence": confidence,
            "contradiction_score": contradiction_score,
        }
    )
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"total_tokens": tokens}
    return response


def make_worker_state(source_type: str = "web") -> WorkerInput:
    """Create a minimal WorkerInput TypedDict."""
    return WorkerInput(
        worker_id="worker-123",
        sub_query="test query about renewables",
        source_type=source_type,  # type: ignore[typeddict-item]
    )


def make_evidence(claim: str = "Test claim.") -> Evidence:
    """Create a minimal Evidence dataclass for test assertions."""
    return Evidence(
        claim=claim,
        supporting_chunks=["supporting quote"],
        source_url="http://example.com",
        source_title="Example",
        published_date=None,
        confidence=0.8,
        contradiction_score=0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runtime() -> MagicMock:
    """Mock Runtime[NofrinContext] — Exa returns one result, LLM returns valid JSON."""
    runtime = MagicMock()
    runtime.context = MagicMock(spec=NofrinContext)

    search_response = MagicMock()
    search_response.results = [make_exa_result()]
    mock_exa = MagicMock()
    mock_exa.search_and_contents = AsyncMock(return_value=search_response)
    runtime.context.exa_client = mock_exa

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=make_llm_response())
    runtime.context.llm_worker = mock_llm

    return runtime


@pytest.fixture
def mock_runtime_no_results() -> MagicMock:
    """Mock Runtime[NofrinContext] — Exa returns zero results (no LLM calls needed)."""
    runtime = MagicMock()
    runtime.context = MagicMock(spec=NofrinContext)

    search_response = MagicMock()
    search_response.results = []
    mock_exa = MagicMock()
    mock_exa.search_and_contents = AsyncMock(return_value=search_response)
    runtime.context.exa_client = mock_exa

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=make_llm_response())
    runtime.context.llm_worker = mock_llm

    return runtime


# ---------------------------------------------------------------------------
# 1-3: Node output structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_node_returns_correct_worker_id(
    mock_runtime_no_results: MagicMock,
) -> None:
    """WorkerResult.worker_id matches WorkerInput.worker_id."""
    result = await worker_node(make_worker_state("web"), mock_runtime_no_results)
    assert result["worker_results"][0].worker_id == "worker-123"


@pytest.mark.asyncio
async def test_worker_node_returns_correct_sub_query(
    mock_runtime_no_results: MagicMock,
) -> None:
    """WorkerResult.sub_query matches WorkerInput.sub_query."""
    result = await worker_node(make_worker_state("web"), mock_runtime_no_results)
    assert result["worker_results"][0].sub_query == "test query about renewables"


@pytest.mark.asyncio
async def test_worker_node_returns_correct_source_type(
    mock_runtime_no_results: MagicMock,
) -> None:
    """WorkerResult.source_type matches WorkerInput.source_type."""
    result = await worker_node(make_worker_state("web"), mock_runtime_no_results)
    assert result["worker_results"][0].source_type == "web"


# ---------------------------------------------------------------------------
# 4-7: Exa parameter routing (via worker_node with zero results)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_node_web_no_date_filter(
    mock_runtime_no_results: MagicMock,
) -> None:
    """Web source_type: search_and_contents called without start_published_date."""
    await worker_node(make_worker_state("web"), mock_runtime_no_results)
    call_kwargs = (
        mock_runtime_no_results.context.exa_client.search_and_contents.call_args[1]
    )
    assert "start_published_date" not in call_kwargs


@pytest.mark.asyncio
async def test_worker_node_academic_category_research_paper(
    mock_runtime_no_results: MagicMock,
) -> None:
    """Academic source_type: search_and_contents called with category='research paper'."""
    await worker_node(make_worker_state("academic"), mock_runtime_no_results)
    call_kwargs = (
        mock_runtime_no_results.context.exa_client.search_and_contents.call_args[1]
    )
    assert call_kwargs["category"] == "research paper"


@pytest.mark.asyncio
async def test_worker_node_academic_highlights_enabled(
    mock_runtime_no_results: MagicMock,
) -> None:
    """Academic source_type: search_and_contents called with highlights dict."""
    await worker_node(make_worker_state("academic"), mock_runtime_no_results)
    call_kwargs = (
        mock_runtime_no_results.context.exa_client.search_and_contents.call_args[1]
    )
    assert call_kwargs["highlights"] == {"max_characters": 2000}


@pytest.mark.asyncio
async def test_worker_node_news_start_date_90_days_ago(
    mock_runtime_no_results: MagicMock,
) -> None:
    """News source_type: search_and_contents called with start_published_date = today-90d."""
    await worker_node(make_worker_state("news"), mock_runtime_no_results)
    call_kwargs = (
        mock_runtime_no_results.context.exa_client.search_and_contents.call_args[1]
    )
    expected = (date.today() - timedelta(days=90)).isoformat()
    assert call_kwargs["start_published_date"] == expected


# ---------------------------------------------------------------------------
# 8-9: Evidence extraction quality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evidence_items_are_evidence_dataclasses(
    mock_runtime: MagicMock,
) -> None:
    """All items in WorkerResult.evidence_items are Evidence instances."""
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) > 0
    assert all(isinstance(item, Evidence) for item in items)


@pytest.mark.asyncio
async def test_tokens_used_populated(mock_runtime: MagicMock) -> None:
    """WorkerResult.tokens_used > 0 when LLM returns usage_metadata."""
    result = await worker_node(make_worker_state("web"), mock_runtime)
    assert result["worker_results"][0].tokens_used > 0


# ---------------------------------------------------------------------------
# 10-12: Partial/full failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_failure_skips_one_returns_others() -> None:
    """3 results, middle extraction fails → 2 Evidence items returned."""
    good_evidence = make_evidence("Good claim from result.")
    call_count = 0

    async def mock_extract(
        exa_result: object,
        sub_query: str,
        llm: object,
        prompt_template: str,
    ) -> tuple[Evidence, int]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise AgentParseError("forced parse failure for middle result")
        return good_evidence, 50

    with patch("agents.worker._extract_evidence_from_result", side_effect=mock_extract):
        with patch("agents.worker._load_prompt", return_value="template"):
            items, tokens = await _extract_all_evidence(
                [MagicMock(), MagicMock(), MagicMock()], "query", MagicMock()
            )

    assert len(items) == 2
    assert tokens == 100  # 2 successful extractions × 50 tokens each


@pytest.mark.asyncio
async def test_all_results_fail_returns_empty_evidence() -> None:
    """All extractions fail → evidence_items == [], tokens == 0."""

    async def mock_extract(*args: object, **kwargs: object) -> tuple[Evidence, int]:
        raise AgentParseError("forced failure")

    with patch("agents.worker._extract_evidence_from_result", side_effect=mock_extract):
        with patch("agents.worker._load_prompt", return_value="template"):
            items, tokens = await _extract_all_evidence(
                [MagicMock(), MagicMock()], "query", MagicMock()
            )

    assert items == []
    assert tokens == 0


@pytest.mark.asyncio
async def test_zero_exa_results_returns_empty_evidence(
    mock_runtime_no_results: MagicMock,
) -> None:
    """Exa returns 0 results → evidence_items == [], tokens == 0, no exception."""
    result = await worker_node(make_worker_state("web"), mock_runtime_no_results)
    worker_result = result["worker_results"][0]
    assert worker_result.evidence_items == []
    assert worker_result.tokens_used == 0


# ---------------------------------------------------------------------------
# 13: Raw search results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_search_results_length_matches_exa_results(
    mock_runtime: MagicMock,
) -> None:
    """len(raw_search_results) matches the number of results Exa returned."""
    result = await worker_node(make_worker_state("web"), mock_runtime)
    worker_result = result["worker_results"][0]
    # Default fixture returns 1 result
    assert len(worker_result.raw_search_results) == 1


# ---------------------------------------------------------------------------
# 14-19: Evidence field values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confidence_parsed_correctly(mock_runtime: MagicMock) -> None:
    """LLM confidence=0.85 → evidence.confidence == 0.85."""
    mock_runtime.context.llm_worker.ainvoke.return_value = make_llm_response(
        confidence=0.85
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_contradiction_score_parsed_correctly(mock_runtime: MagicMock) -> None:
    """LLM contradiction_score=0.3 → evidence.contradiction_score == 0.3."""
    mock_runtime.context.llm_worker.ainvoke.return_value = make_llm_response(
        contradiction_score=0.3
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].contradiction_score == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_confidence_clamped_above_1(mock_runtime: MagicMock) -> None:
    """LLM returns confidence=1.5 → clamped to 1.0."""
    mock_runtime.context.llm_worker.ainvoke.return_value = make_llm_response(
        confidence=1.5
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].confidence == 1.0


@pytest.mark.asyncio
async def test_source_url_from_exa_not_llm(mock_runtime: MagicMock) -> None:
    """evidence.source_url is taken from the Exa result, not hallucinated by LLM."""
    exa_url = "http://specific-exa-url.example.com/research"
    search_response = MagicMock()
    search_response.results = [make_exa_result(url=exa_url)]
    mock_runtime.context.exa_client.search_and_contents = AsyncMock(
        return_value=search_response
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].source_url == exa_url


@pytest.mark.asyncio
async def test_source_title_from_exa_not_llm(mock_runtime: MagicMock) -> None:
    """evidence.source_title is taken from the Exa result, not hallucinated by LLM."""
    exa_title = "Definitive Title From Exa API"
    search_response = MagicMock()
    search_response.results = [make_exa_result(title=exa_title)]
    mock_runtime.context.exa_client.search_and_contents = AsyncMock(
        return_value=search_response
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].source_title == exa_title


@pytest.mark.asyncio
async def test_published_date_from_exa_not_llm(mock_runtime: MagicMock) -> None:
    """evidence.published_date is taken from the Exa result, not hallucinated by LLM."""
    exa_date = "2025-09-20"
    search_response = MagicMock()
    search_response.results = [make_exa_result(published_date=exa_date)]
    mock_runtime.context.exa_client.search_and_contents = AsyncMock(
        return_value=search_response
    )
    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert len(items) == 1
    assert items[0].published_date == exa_date


# ---------------------------------------------------------------------------
# 20-23: Param builder unit tests
# ---------------------------------------------------------------------------


def test_build_exa_params_web_no_date_filter() -> None:
    """Web params do not include start_published_date."""
    params = _build_exa_params_web("query")
    assert "start_published_date" not in params


def test_build_exa_params_academic_has_category() -> None:
    """Academic params include category='research paper'."""
    params = _build_exa_params_academic("query")
    assert params["category"] == "research paper"


def test_build_exa_params_academic_has_highlights() -> None:
    """Academic params include highlights={'max_characters': 2000}."""
    params = _build_exa_params_academic("query")
    assert params["highlights"] == {"max_characters": 2000}


def test_build_exa_params_news_date_is_90_days_ago() -> None:
    """News params include start_published_date == today - 90 days."""
    params = _build_exa_params_news("query")
    expected = (date.today() - timedelta(days=90)).isoformat()
    assert params["start_published_date"] == expected


# ---------------------------------------------------------------------------
# 24: Invalid source_type
# ---------------------------------------------------------------------------


def test_invalid_source_type_raises_value_error() -> None:
    """source_type='invalid' → ValueError before any API call is made."""
    with pytest.raises(ValueError, match="Invalid source_type"):
        _get_exa_params("test query", "invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 25: Exa retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exa_api_failure_retries() -> None:
    """Exa raises HTTPStatusError on first call → retried → second call succeeds."""
    import httpx

    from agents.worker import _search_exa

    mock_exa = MagicMock()
    error = httpx.HTTPStatusError(
        "429 Too Many Requests",
        request=MagicMock(),
        response=MagicMock(),
    )
    success_response = MagicMock()
    success_response.results = []
    mock_exa.search_and_contents = AsyncMock(side_effect=[error, success_response])

    # Patch asyncio.sleep to skip the exponential backoff wait in tests
    with patch("asyncio.sleep", new_callable=AsyncMock):
        results = await _search_exa(mock_exa, "test query", "web")

    assert mock_exa.search_and_contents.call_count == 2
    assert results == []


# ---------------------------------------------------------------------------
# 26-27: Highlights / text None handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_highlights_none_falls_back_to_text(mock_runtime: MagicMock) -> None:
    """highlights=None, text='...' → text used in prompt → evidence extracted."""
    search_response = MagicMock()
    search_response.results = [
        make_exa_result(
            text="Fallback text content. Important research finding here.",
            highlights=None,
        )
    ]
    mock_runtime.context.exa_client.search_and_contents = AsyncMock(
        return_value=search_response
    )

    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    # Text fallback worked — extraction succeeded
    assert len(items) == 1


@pytest.mark.asyncio
async def test_both_highlights_and_text_none_skips_result(
    mock_runtime: MagicMock,
) -> None:
    """highlights=None, text=None → empty claim sentinel → evidence_items == []."""
    search_response = MagicMock()
    search_response.results = [make_exa_result(text=None, highlights=None)]
    mock_runtime.context.exa_client.search_and_contents = AsyncMock(
        return_value=search_response
    )

    result = await worker_node(make_worker_state("web"), mock_runtime)
    items = result["worker_results"][0].evidence_items
    assert items == []  # empty claim filtered out by _extract_all_evidence


# ---------------------------------------------------------------------------
# 28: asyncio.gather resilience (return_exceptions=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_partial_exception_does_not_cancel_others() -> None:
    """Middle extraction raises; others complete — return_exceptions=True ensures no cancellation."""
    good_evidence = make_evidence("Completed extraction claim.")
    call_count = 0

    async def mock_extract(
        exa_result: object,
        sub_query: str,
        llm: object,
        prompt_template: str,
    ) -> tuple[Evidence, int]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("catastrophic failure in middle extraction")
        return good_evidence, 75

    with patch("agents.worker._extract_evidence_from_result", side_effect=mock_extract):
        with patch("agents.worker._load_prompt", return_value="template"):
            items, tokens = await _extract_all_evidence(
                [MagicMock(), MagicMock(), MagicMock()], "gather test", MagicMock()
            )

    # 1st and 3rd completed; 2nd failed but didn't cancel others
    assert len(items) == 2
    assert tokens == 150  # 2 × 75 tokens


# ---------------------------------------------------------------------------
# cache_control tests
# ---------------------------------------------------------------------------


def test_build_extraction_messages_anthropic_has_cache_control() -> None:
    """use_cache_control=True → SystemMessage content list with cached static block."""
    msgs = _build_extraction_messages(
        sub_query="test sub query",
        source_url="http://example.com",
        source_title="Example Title",
        result_text="Some result text here.",
        prompt_template="static instructions template",
        use_cache_control=True,
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, list), "Expected list content blocks for Anthropic"
    assert len(content) == 2

    static_block = content[0]
    dynamic_block = content[1]

    assert isinstance(static_block, dict)
    assert static_block.get("cache_control") == {"type": "ephemeral"}
    assert static_block.get("text") == "static instructions template"

    assert isinstance(dynamic_block, dict)
    assert "cache_control" not in dynamic_block
    dynamic_text = dynamic_block.get("text", "")
    assert isinstance(dynamic_text, str)
    assert "test sub query" in dynamic_text
    assert "http://example.com" in dynamic_text
    assert "Some result text here." in dynamic_text


def test_build_extraction_messages_groq_no_cache_control() -> None:
    """use_cache_control=False → plain string SystemMessage with all placeholders filled."""
    msgs = _build_extraction_messages(
        sub_query="my sub query",
        source_url="http://source.com",
        source_title="Source Title",
        result_text="Result text.",
        prompt_template="query={{sub_query}} url={{source_url}} title={{source_title}} text={{result_text}}",
        use_cache_control=False,
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, str), "Expected plain string for non-Anthropic provider"
    assert "my sub query" in content
    assert "http://source.com" in content
    assert "Source Title" in content
    assert "Result text." in content
    assert "cache_control" not in content

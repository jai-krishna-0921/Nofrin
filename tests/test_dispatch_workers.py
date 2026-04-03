"""
tests/test_dispatch_workers.py

Unit tests for graph/router.py:dispatch_workers()
"""

from __future__ import annotations

import pytest
from langgraph.types import Send

from graph.router import dispatch_workers
from graph.state import ResearchAgentState, SourceType, WorkerInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_state(
    sub_queries: list[str],
    source_routing: dict[str, SourceType],
) -> ResearchAgentState:
    """Build minimal ResearchAgentState for dispatch_workers tests."""
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format="markdown",
        sub_queries=sub_queries,
        source_routing=source_routing,
        worker_results=[],
        compressed_worker_results=[],
        synthesis=None,
        grounding_issues=[],
        critic_output=None,
        revision_count=0,
        prior_syntheses=[],
        session_id="test-session",
        total_tokens_used=0,
        cost_usd=0.0,
        final_output=None,
    )


# ---------------------------------------------------------------------------
# Test 1: Returns list of Send objects
# ---------------------------------------------------------------------------


def test_dispatch_workers_returns_list_of_send() -> None:
    """dispatch_workers returns a list where all items are Send instances."""
    state = build_state(
        sub_queries=["query 1", "query 2"],
        source_routing={"query 1": "web", "query 2": "academic"},
    )
    result = dispatch_workers(state)

    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


# ---------------------------------------------------------------------------
# Test 2: Count matches sub_queries
# ---------------------------------------------------------------------------


def test_dispatch_workers_count_matches_sub_queries() -> None:
    """Number of Send objects equals number of sub_queries."""
    sub_queries = ["query A", "query B", "query C"]
    state = build_state(
        sub_queries=sub_queries,
        source_routing={},
    )
    result = dispatch_workers(state)

    assert len(result) == len(sub_queries)


# ---------------------------------------------------------------------------
# Test 3: Target node is "worker"
# ---------------------------------------------------------------------------


def test_dispatch_workers_target_node_is_worker() -> None:
    """All Send objects target the 'worker' node."""
    state = build_state(
        sub_queries=["q1", "q2", "q3"],
        source_routing={},
    )
    result = dispatch_workers(state)

    assert all(s.node == "worker" for s in result)


# ---------------------------------------------------------------------------
# Test 4: Worker IDs are sequential
# ---------------------------------------------------------------------------


def test_worker_ids_are_sequential() -> None:
    """Worker IDs in Send.arg payloads are 'worker-0', 'worker-1', 'worker-2'."""
    state = build_state(
        sub_queries=["alpha", "beta", "gamma"],
        source_routing={},
    )
    result = dispatch_workers(state)

    worker_ids = [s.arg["worker_id"] for s in result]
    assert worker_ids == ["worker-0", "worker-1", "worker-2"]


# ---------------------------------------------------------------------------
# Test 5: Sub-queries preserved in payload
# ---------------------------------------------------------------------------


def test_sub_queries_preserved_in_payload() -> None:
    """Each Send.arg['sub_query'] matches the original sub_queries list."""
    sub_queries = ["What is X?", "How does Y work?", "Why Z?"]
    state = build_state(
        sub_queries=sub_queries,
        source_routing={},
    )
    result = dispatch_workers(state)

    for idx, send in enumerate(result):
        assert send.arg["sub_query"] == sub_queries[idx]


# ---------------------------------------------------------------------------
# Test 6: Source types from routing
# ---------------------------------------------------------------------------


def test_source_types_from_routing() -> None:
    """Send.arg['source_type'] matches source_routing dict."""
    state = build_state(
        sub_queries=["q1", "q2", "q3"],
        source_routing={
            "q1": "academic",
            "q2": "news",
            "q3": "web",
        },
    )
    result = dispatch_workers(state)

    assert result[0].arg["source_type"] == "academic"
    assert result[1].arg["source_type"] == "news"
    assert result[2].arg["source_type"] == "web"


# ---------------------------------------------------------------------------
# Test 7: Missing routing defaults to "web"
# ---------------------------------------------------------------------------


def test_missing_routing_defaults_to_web() -> None:
    """Sub-query absent from source_routing → source_type defaults to 'web'."""
    state = build_state(
        sub_queries=["mapped", "unmapped"],
        source_routing={"mapped": "academic"},
    )
    result = dispatch_workers(state)

    assert result[0].arg["source_type"] == "academic"  # mapped
    assert result[1].arg["source_type"] == "web"  # unmapped → default


# ---------------------------------------------------------------------------
# Test 8: Empty sub_queries raises ValueError
# ---------------------------------------------------------------------------


def test_empty_sub_queries_raises_value_error() -> None:
    """ValueError raised when sub_queries list is empty."""
    state = build_state(
        sub_queries=[],
        source_routing={},
    )

    with pytest.raises(ValueError, match="sub_queries is empty"):
        dispatch_workers(state)


# ---------------------------------------------------------------------------
# Test 9: Three sub-queries creates three Sends
# ---------------------------------------------------------------------------


def test_three_sub_queries_creates_three_sends() -> None:
    """3 sub-queries → exactly 3 Send objects."""
    state = build_state(
        sub_queries=["one", "two", "three"],
        source_routing={},
    )
    result = dispatch_workers(state)

    assert len(result) == 3
    assert all(isinstance(s, Send) for s in result)


# ---------------------------------------------------------------------------
# Test 10: Five sub-queries creates five Sends
# ---------------------------------------------------------------------------


def test_five_sub_queries_creates_five_sends() -> None:
    """5 sub-queries → exactly 5 Send objects."""
    state = build_state(
        sub_queries=["a", "b", "c", "d", "e"],
        source_routing={},
    )
    result = dispatch_workers(state)

    assert len(result) == 5
    assert all(isinstance(s, Send) for s in result)

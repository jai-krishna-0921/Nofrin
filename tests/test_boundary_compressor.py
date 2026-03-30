"""
tests/test_boundary_compressor.py

Unit tests for graph/boundary_compressor.py.
"""

from __future__ import annotations

from graph.boundary_compressor import boundary_compressor_node, compress_worker_results
from graph.state import Evidence, WorkerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evidence(confidence: float, claim: str = "Test claim.") -> Evidence:
    return Evidence(
        claim=claim,
        supporting_chunks=["quote"],
        source_url=f"http://example.com/{confidence}",
        source_title="Title",
        published_date=None,
        confidence=confidence,
        contradiction_score=0.0,
    )


def make_worker_result(
    evidence_confidences: list[float],
    worker_id: str = "worker-1",
    sub_query: str = "test query",
    source_type: str = "web",
    tokens_used: int = 200,
) -> WorkerResult:
    return WorkerResult(
        worker_id=worker_id,
        sub_query=sub_query,
        source_type=source_type,  # type: ignore[arg-type]
        evidence_items=[make_evidence(c) for c in evidence_confidences],
        raw_search_results=[
            {"url": f"http://example.com/{i}", "title": f"Result {i}"}
            for i in range(len(evidence_confidences))
        ],
        tokens_used=tokens_used,
    )


# ---------------------------------------------------------------------------
# Test 1: top 5 by confidence kept when >5 items
# ---------------------------------------------------------------------------


def test_compressor_keeps_top_5_by_confidence() -> None:
    """10 evidence items → only the 5 with highest confidence are kept."""
    confidences = [0.1, 0.9, 0.3, 0.8, 0.5, 0.7, 0.2, 0.6, 0.4, 1.0]
    wr = make_worker_result(confidences)
    [compressed] = compress_worker_results([wr])

    assert len(compressed.evidence_items) == 5
    kept_confidences = [e.confidence for e in compressed.evidence_items]
    assert kept_confidences == sorted(kept_confidences, reverse=True)
    assert kept_confidences == [1.0, 0.9, 0.8, 0.7, 0.6]


# ---------------------------------------------------------------------------
# Test 2: raw_search_results stripped
# ---------------------------------------------------------------------------


def test_compressor_strips_raw_search_results() -> None:
    """raw_search_results is emptied in every compressed WorkerResult."""
    wr = make_worker_result([0.9, 0.8])
    assert len(wr.raw_search_results) == 2  # pre-condition

    [compressed] = compress_worker_results([wr])
    assert compressed.raw_search_results == []


# ---------------------------------------------------------------------------
# Test 3: fewer than 5 items → all kept
# ---------------------------------------------------------------------------


def test_compressor_fewer_than_5_evidence_keeps_all() -> None:
    """3 evidence items → all 3 kept (no truncation below _TOP_N)."""
    wr = make_worker_result([0.9, 0.5, 0.3])
    [compressed] = compress_worker_results([wr])
    assert len(compressed.evidence_items) == 3


# ---------------------------------------------------------------------------
# Test 4: worker metadata preserved
# ---------------------------------------------------------------------------


def test_compressor_preserves_worker_metadata() -> None:
    """worker_id, sub_query, source_type, tokens_used survive compression."""
    wr = make_worker_result(
        [0.9],
        worker_id="worker-42",
        sub_query="specific sub query",
        source_type="academic",
        tokens_used=512,
    )
    [compressed] = compress_worker_results([wr])

    assert compressed.worker_id == "worker-42"
    assert compressed.sub_query == "specific sub query"
    assert compressed.source_type == "academic"
    assert compressed.tokens_used == 512


# ---------------------------------------------------------------------------
# Test 5: multiple workers processed independently
# ---------------------------------------------------------------------------


def test_compressor_multiple_workers() -> None:
    """3 worker results each compressed independently; lengths correct."""
    workers = [
        make_worker_result([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], worker_id="w1"),  # 6 → 5
        make_worker_result([0.9, 0.8], worker_id="w2"),                        # 2 → 2
        make_worker_result([], worker_id="w3"),                                 # 0 → 0
    ]
    compressed = compress_worker_results(workers)

    assert len(compressed) == 3
    assert len(compressed[0].evidence_items) == 5
    assert len(compressed[1].evidence_items) == 2
    assert len(compressed[2].evidence_items) == 0
    # raw_search_results stripped on all
    assert all(wr.raw_search_results == [] for wr in compressed)
    # worker_ids preserved
    assert [wr.worker_id for wr in compressed] == ["w1", "w2", "w3"]


# ---------------------------------------------------------------------------
# Test 6 (bonus): boundary_compressor_node integrates with state dict
# ---------------------------------------------------------------------------


def test_boundary_compressor_node_writes_compressed_field() -> None:
    """boundary_compressor_node returns compressed_worker_results key."""
    wr = make_worker_result([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])  # 7 items → 5
    # Minimal state dict — only worker_results is read by the node
    state = {"worker_results": [wr]}  # type: ignore[typeddict-item]
    result = boundary_compressor_node(state)  # type: ignore[arg-type]

    assert "compressed_worker_results" in result
    assert len(result["compressed_worker_results"]) == 1
    assert len(result["compressed_worker_results"][0].evidence_items) == 5

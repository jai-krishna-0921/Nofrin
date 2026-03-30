"""
graph/boundary_compressor.py

Pure-Python node that runs between the worker fan-in and the coordinator.

Responsibilities:
  1. Keep the top 5 evidence items per WorkerResult (by confidence, descending).
  2. Strip raw_search_results from every WorkerResult to reduce state size
     before the coordinator's context window.

Writes to `compressed_worker_results` (a plain list[WorkerResult] with no
reducer) rather than back to `worker_results` (which uses operator.add and
cannot be replaced from within a normal node).

No LLM calls, no async, no external I/O.
"""

from __future__ import annotations

from graph.state import Evidence, ResearchAgentState, WorkerResult

_TOP_N = 5  # maximum evidence items kept per WorkerResult


# ---------------------------------------------------------------------------
# Core compression logic
# ---------------------------------------------------------------------------


def _compress_single(wr: WorkerResult) -> WorkerResult:
    """Compress one WorkerResult: top-N evidence by confidence, no raw results.

    Args:
        wr: The WorkerResult to compress.

    Returns:
        New WorkerResult with evidence_items truncated to _TOP_N (highest
        confidence first) and raw_search_results cleared.
    """
    top_evidence: list[Evidence] = sorted(
        wr.evidence_items,
        key=lambda e: e.confidence,
        reverse=True,
    )[:_TOP_N]

    return WorkerResult(
        worker_id=wr.worker_id,
        sub_query=wr.sub_query,
        source_type=wr.source_type,
        evidence_items=top_evidence,
        raw_search_results=[],  # stripped — saves coordinator context tokens
        tokens_used=wr.tokens_used,
    )


def compress_worker_results(results: list[WorkerResult]) -> list[WorkerResult]:
    """Compress all WorkerResults.

    Args:
        results: Accumulated list from all parallel workers.

    Returns:
        New list where each WorkerResult has at most _TOP_N evidence items
        and an empty raw_search_results list.
    """
    return [_compress_single(wr) for wr in results]


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def boundary_compressor_node(
    state: ResearchAgentState,
) -> dict[str, list[WorkerResult]]:
    """LangGraph node: compress accumulated worker results.

    Reads:  state["worker_results"]  (operator.add-reduced list from workers)
    Writes: state["compressed_worker_results"]  (plain list, no reducer)

    Args:
        state: Current ResearchAgentState after all workers have joined.

    Returns:
        Partial state update: {"compressed_worker_results": [...]}.
    """
    compressed = compress_worker_results(state["worker_results"])
    return {"compressed_worker_results": compressed}


__all__ = ["boundary_compressor_node", "compress_worker_results"]

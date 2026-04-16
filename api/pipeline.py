"""
api/pipeline.py

run_pipeline() coroutine — reuses main.py helpers, calls build_graph() unchanged.
"""

from __future__ import annotations

import json
from typing import Callable

from graph.builder import build_graph
from graph.progress import set_sse_callback
from graph.state import OutputFormat, ResearchMode


async def run_pipeline(
    session_id: str,
    query: str,
    output_format: OutputFormat,
    research_mode: ResearchMode,
    emit: Callable[[str, str], None],
) -> None:
    """Execute the research pipeline and emit SSE events via emit callback.

    Args:
        session_id:     Unique session identifier (passed to _build_context).
        query:          User's research question.
        output_format:  Requested output format.
        research_mode:  "fast" or "research".
        emit:           Callback(event_type, json_payload) for SSE events.
    """
    from main import _build_context, _build_initial_state  # noqa: PLC0415

    set_sse_callback(emit)
    try:
        emit(
            "status",
            json.dumps(
                {
                    "phase": "pipeline",
                    "message": f"Starting {research_mode} mode pipeline…",
                }
            ),
        )
        graph = build_graph()
        context = _build_context(session_id)
        initial_state = _build_initial_state(
            query, output_format, session_id, research_mode
        )

        result: dict[str, object] = await graph.ainvoke(  # type: ignore[call-overload]
            initial_state,
            context=context,
        )

        final_output = result.get("final_output")
        cost_raw = result.get("cost_usd")
        tokens_raw = result.get("total_tokens_used")
        cost: float = float(cost_raw) if isinstance(cost_raw, (int, float)) else 0.0
        tokens: int = int(tokens_raw) if isinstance(tokens_raw, int) else 0

        if isinstance(final_output, str) and final_output:
            emit(
                "result",
                json.dumps(
                    {
                        "final_output": final_output,
                        "cost_usd": round(cost, 4),
                        "tokens": tokens,
                    }
                ),
            )
        else:
            emit("error", json.dumps({"message": "Pipeline produced no output"}))
    except Exception as exc:
        emit("error", json.dumps({"message": str(exc)}))
    finally:
        set_sse_callback(None)
        emit("done", "")


__all__ = ["run_pipeline"]

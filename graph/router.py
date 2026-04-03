"""
graph/router.py

Routing logic for the Deep Research Agent.

Nodes / functions defined here:

  dispatch_workers     — conditional-edge function; called after supervisor.
                         Returns list[Send] to fan-out one worker per sub-query.

  budget_gate_node     — regular node; appends a gap warning to synthesis if
                         cost_usd > 75 % of cost_ceiling_usd, then returns the
                         (possibly updated) synthesis as a state update.

  route_after_critic   — conditional-edge function; returns the name of the
                         next node: "delivery" or "coordinator".

Wiring (in graph/builder.py):
  supervisor → conditional(dispatch_workers) → worker × N (fan-out)
  worker → boundary_compressor (fan-in via operator.add reducer)
  ... → critic → budget_gate_node → conditional(route_after_critic)
                                         ├── "delivery"   (pass / budget / cap)
                                         └── "coordinator" (revise)
"""

from __future__ import annotations

import dataclasses

from langgraph.runtime import Runtime
from langgraph.types import Send

from graph.context import NofrinContext
from graph.state import ResearchAgentState, SourceType, SynthesisOutput, WorkerInput

# Fraction of the cost ceiling that triggers a forced delivery.
_BUDGET_THRESHOLD = 0.75

# Hard revision cap (also enforced in CLAUDE.md; keep in sync).
_MAX_REVISIONS = 2


# ---------------------------------------------------------------------------
# Fan-out dispatcher: supervisor → workers
# ---------------------------------------------------------------------------


def dispatch_workers(
    state: ResearchAgentState,
) -> list[Send]:
    """Fan-out dispatcher: create one Send() per sub-query targeting the worker node.

    Called via add_conditional_edges after supervisor_node completes.
    Reads state["sub_queries"] and state["source_routing"] (written by supervisor).

    Worker IDs are sequential: "worker-0", "worker-1", etc.
    If a sub_query is absent from source_routing, defaults to "web".

    Args:
        state: Current ResearchAgentState after supervisor node.

    Returns:
        list[Send] — one Send("worker", WorkerInput) per sub-query.

    Raises:
        ValueError: If sub_queries is empty.
    """
    sub_queries: list[str] = state["sub_queries"]
    source_routing: dict[str, SourceType] = state["source_routing"]

    if not sub_queries:
        raise ValueError(
            "dispatch_workers: sub_queries is empty — "
            "supervisor_node must populate sub_queries before dispatch."
        )

    sends: list[Send] = []
    for idx, sub_query in enumerate(sub_queries):
        source_type: SourceType = source_routing.get(sub_query, "web")
        worker_input = WorkerInput(
            worker_id=f"worker-{idx}",
            sub_query=sub_query,
            source_type=source_type,
        )
        sends.append(Send("worker", worker_input))

    return sends


# ---------------------------------------------------------------------------
# Budget gate node
# ---------------------------------------------------------------------------


def budget_gate_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, SynthesisOutput]:
    """Append a budget warning to synthesis.gaps if spending exceeds 75 %.

    This is the only place where the budget warning is injected into state so
    that downstream nodes (delivery, coordinator) can surface it in output.

    Args:
        state: Current ResearchAgentState. Reads cost_usd and synthesis.
        runtime: Injected context. Reads cost_ceiling_usd.

    Returns:
        Partial state update: {"synthesis": updated_synthesis} if the budget
        threshold is exceeded and synthesis is not None; empty dict otherwise.
    """
    cost_usd: float = state["cost_usd"]
    cost_ceiling: float = runtime.context.cost_ceiling_usd
    synthesis: SynthesisOutput | None = state["synthesis"]

    if synthesis is not None and cost_usd > _BUDGET_THRESHOLD * cost_ceiling:
        warning = (
            f"BUDGET WARNING: session cost ${cost_usd:.3f} exceeded "
            f"{int(_BUDGET_THRESHOLD * 100)}% of ${cost_ceiling:.2f} limit. "
            "Research halted early; output may be incomplete."
        )
        updated = dataclasses.replace(synthesis, gaps=list(synthesis.gaps) + [warning])
        return {"synthesis": updated}

    return {}


# ---------------------------------------------------------------------------
# Conditional edge: route after critic
# ---------------------------------------------------------------------------


def route_after_critic(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> str:
    """Decide whether to deliver or revise after the critic has scored.

    Priority order:
      1. Budget gate  — if cost > 75 % of ceiling, force deliver.
      2. Quality pass — if critic_output.passed is True, deliver.
      3. Revision cap — if revision_count >= 2, force deliver with caveat.
      4. Default      — route back to coordinator for another revision pass.

    Args:
        state: Current ResearchAgentState.
        runtime: Injected context. Reads cost_ceiling_usd.

    Returns:
        "delivery" or "coordinator".
    """
    cost_usd: float = state["cost_usd"]
    cost_ceiling: float = runtime.context.cost_ceiling_usd

    # 1. Budget gate (budget_gate_node already injected the warning into gaps).
    if cost_usd > _BUDGET_THRESHOLD * cost_ceiling:
        return "delivery"

    # 2. Quality passed.
    critic_output = state["critic_output"]
    if critic_output is not None and critic_output.passed:
        return "delivery"

    # 3. Hard revision cap (CLAUDE.md: max 2 revisions).
    if state["revision_count"] >= _MAX_REVISIONS:
        return "delivery"

    # 4. Request another revision.
    return "coordinator"


__all__ = ["budget_gate_node", "dispatch_workers", "route_after_critic"]

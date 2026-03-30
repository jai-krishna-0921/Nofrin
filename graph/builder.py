"""
graph/builder.py

LangGraph workflow assembly for the Deep Research Agent.

Current wiring (Phase 1 — nodes implemented so far):
  START → supervisor → boundary_compressor → END

Planned full wiring (Phase 1 complete):
  START → supervisor
        → [Send() fan-out: worker × N]
        → boundary_compressor
        → coordinator
        → grounding_check
        → critic
        → budget_gate
        → conditional(route_after_critic)
              ├── "delivery" → END
              └── "coordinator" (revision loop, max 2)

Phase 2 additions: CachePolicy(ttl=300) on worker nodes, InMemoryCache.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.context import NofrinContext
from graph.state import ResearchAgentState


def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Assemble and return the compiled Deep Research Agent graph.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.

    Usage::

        from graph.builder import build_graph
        from graph.context import NofrinContext
        from graph.llm import get_llm
        from exa_py import AsyncExa
        import os, uuid

        graph = build_graph()
        context = NofrinContext(
            llm_supervisor=get_llm(role="default"),
            llm_worker=get_llm(role="default"),
            llm_coordinator=get_llm(role="default"),
            llm_critic=get_llm(role="critic"),
            exa_client=AsyncExa(api_key=os.environ["EXA_API_KEY"]),
            session_id=str(uuid.uuid4()),
            cost_ceiling_usd=float(os.getenv("COST_CEILING_USD", "1.00")),
        )
        result = await graph.ainvoke(initial_state, context=context)
    """
    # Phase 2: compile with InMemoryCache to activate CachePolicy on worker nodes:
    #   from langgraph.cache.memory import InMemoryCache
    #   return builder.compile(cache=InMemoryCache())

    builder: StateGraph = StateGraph(  # type: ignore[type-arg]
        ResearchAgentState,
        context_schema=NofrinContext,  # type: ignore[arg-type]
    )

    # ── Nodes (import here to avoid circular imports as agent package grows) ─

    from agents.supervisor import supervisor_node
    from graph.boundary_compressor import boundary_compressor_node
    from graph.router import budget_gate_node

    builder.add_node("supervisor", supervisor_node)  # type: ignore[call-overload]
    builder.add_node("boundary_compressor", boundary_compressor_node)
    builder.add_node("budget_gate", budget_gate_node)  # type: ignore[arg-type]

    # TODO: add as implemented
    # from agents.worker import worker_node
    # from agents.coordinator import coordinator_node
    # from agents.grounding_check import grounding_check_node
    # from agents.critic import critic_node
    # from agents.delivery import delivery_node
    # from graph.router import route_after_critic
    #
    # builder.add_node("worker", worker_node, cache_policy=CachePolicy(ttl=300))
    # builder.add_node("coordinator", coordinator_node)
    # builder.add_node("grounding_check", grounding_check_node)
    # builder.add_node("critic", critic_node)
    # builder.add_node("delivery", delivery_node)

    # ── Edges (Phase 1 partial — extend as nodes are added) ──────────────────

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "boundary_compressor")
    builder.add_edge("boundary_compressor", END)

    # TODO: full linear wiring once all nodes exist
    # Phase 2: replace supervisor→worker edge with Send() fan-out dispatcher:
    #   builder.add_conditional_edges("supervisor", dispatch_workers, ["worker"])
    #   builder.add_edge("worker", "boundary_compressor")   # fan-in via operator.add
    #   builder.add_edge("boundary_compressor", "coordinator")
    #   builder.add_edge("coordinator", "grounding_check")
    #   builder.add_edge("grounding_check", "critic")
    #   builder.add_edge("critic", "budget_gate")
    #   builder.add_conditional_edges(
    #       "budget_gate",
    #       route_after_critic,
    #       {"delivery": "delivery", "coordinator": "coordinator"},
    #   )
    #   builder.add_edge("delivery", END)

    return builder.compile()


__all__ = ["build_graph"]

"""
graph/context.py

Runtime context for the Deep Research Agent graph.

NofrinContext is passed as `context_schema` to StateGraph and injected
into every node via `runtime: Runtime[NofrinContext]`. This keeps
ResearchAgentState focused on values that change during execution, and
makes nodes fully testable by injecting mock dependencies at invocation.

Usage — graph construction:
    from langgraph.graph import StateGraph
    builder = StateGraph(ResearchAgentState, context_schema=NofrinContext)

Usage — graph invocation:
    from graph.llm import get_llm
    context = NofrinContext(
        llm_supervisor=get_llm(role="default"),
        llm_worker=get_llm(role="default"),
        llm_critic=get_llm(role="critic"),
        session_id=str(uuid.uuid4()),
        cost_ceiling_usd=float(os.getenv("COST_CEILING_USD", "1.00")),
    )
    graph.invoke(initial_state, context=context)

Usage — inside a node:
    async def supervisor_node(
        state: ResearchAgentState,
        runtime: Runtime[NofrinContext],
    ) -> ...:
        llm = runtime.context.llm_supervisor
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exa_py import AsyncExa
from langchain_core.language_models import BaseChatModel


@dataclass
class NofrinContext:
    """Dependency container injected into all graph nodes at runtime.

    LLM clients are pre-configured so each node gets the right model
    without any get_llm(role=...) logic inside agent files.

    Attributes:
        llm_supervisor: LLM for the supervisor node (cheaper, fast).
        llm_worker:     LLM for worker nodes (cheaper, fast).
        llm_coordinator: LLM for the coordinator/synthesis node.
        llm_critic:     LLM for the critic node (most capable model).
        exa_client:     AsyncExa client for all worker Exa searches.
        session_id:     Unique ID for this research session.
        cost_ceiling_usd: Maximum spend allowed for this session.
    """

    llm_supervisor: BaseChatModel
    llm_worker: BaseChatModel
    llm_coordinator: BaseChatModel
    llm_critic: BaseChatModel
    exa_client: AsyncExa
    session_id: str
    cost_ceiling_usd: float = field(default=1.00)


__all__ = ["NofrinContext"]

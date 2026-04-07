"""
tests/test_router.py

Unit tests for graph/router.py — route_after_critic conditional edge.

Tests cover: fast mode always delivers, research mode routes to coordinator
when quality is low and revision cap not reached.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from graph.router import route_after_critic
from graph.state import CriticOutput, ResearchAgentState


def _make_critic_output(passed: bool) -> CriticOutput:
    """Create a minimal CriticOutput with the given passed value."""
    return CriticOutput(
        factuality_score=3.5 if not passed else 4.5,
        citation_alignment_score=3.5 if not passed else 4.5,
        reasoning_score=3.5 if not passed else 4.5,
        completeness_score=3.5 if not passed else 4.5,
        bias_score=3.5 if not passed else 4.5,
        final_quality_score=3.5 if not passed else 4.5,
        issues=[],
        suggestions=[],
        passed=passed,
    )


def _make_state(
    research_mode: str = "research",
    critic_passed: bool = True,
    revision_count: int = 0,
    cost_usd: float = 0.0,
) -> ResearchAgentState:
    """Build a minimal ResearchAgentState for router tests."""
    return ResearchAgentState(
        user_query="test query",
        intent_type="factual",
        output_format="markdown",
        research_mode=research_mode,  # type: ignore[typeddict-item]
        sub_queries=[],
        source_routing={},
        worker_results=[],
        compressed_worker_results=[],
        synthesis=None,
        grounding_issues=[],
        critic_output=_make_critic_output(critic_passed),
        revision_count=revision_count,
        prior_syntheses=[],
        session_id="test",
        total_tokens_used=0,
        cost_usd=cost_usd,
        final_output=None,
    )


def _make_runtime(cost_ceiling: float = 1.00) -> MagicMock:
    runtime = MagicMock()
    runtime.context.cost_ceiling_usd = cost_ceiling
    return runtime


# ---------------------------------------------------------------------------
# Feature 1: Fast mode routing tests
# ---------------------------------------------------------------------------


def test_route_after_critic_fast_mode_returns_delivery_when_critic_passes() -> None:
    """Fast mode always routes to delivery regardless of critic score."""
    state = _make_state(research_mode="fast", critic_passed=True)
    result = route_after_critic(state, _make_runtime())
    assert result == "delivery"


def test_route_after_critic_fast_mode_returns_delivery_when_critic_fails() -> None:
    """Fast mode routes to delivery even when critic did not pass."""
    state = _make_state(research_mode="fast", critic_passed=False)
    result = route_after_critic(state, _make_runtime())
    assert result == "delivery"


def test_route_after_critic_research_mode_routes_to_coordinator_when_quality_low() -> (
    None
):
    """Research mode routes to coordinator when critic fails and revision cap not reached."""
    state = _make_state(
        research_mode="research",
        critic_passed=False,
        revision_count=0,
        cost_usd=0.0,
    )
    result = route_after_critic(state, _make_runtime(cost_ceiling=1.00))
    assert result == "coordinator"

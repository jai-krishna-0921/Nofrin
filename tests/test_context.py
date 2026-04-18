"""
tests/test_context.py

Unit tests for graph/context.py — NofrinContext dataclass.

Tests cover: optional fields (tavily_client, brave_api_key) default to None
when not provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from graph.context import NofrinContext


def _make_minimal_context() -> NofrinContext:
    """Construct NofrinContext with only required fields."""
    return NofrinContext(
        llm_supervisor=MagicMock(),
        llm_worker=MagicMock(),
        llm_coordinator=MagicMock(),
        llm_critic=MagicMock(),
        exa_client=MagicMock(),
        session_id="test-session",
    )


# ---------------------------------------------------------------------------
# Feature 2: Optional context fields default to None
# ---------------------------------------------------------------------------


def test_nofrin_context_tavily_client_defaults_to_none() -> None:
    """NofrinContext.tavily_client defaults to None when not provided."""
    ctx = _make_minimal_context()
    assert ctx.tavily_client is None


def test_nofrin_context_brave_api_key_defaults_to_none() -> None:
    """NofrinContext.brave_api_key defaults to None when not provided."""
    ctx = _make_minimal_context()
    assert ctx.brave_api_key is None

"""
tests/test_main.py

Unit tests for main.py CLI entry point.

Tests cover: _build_initial_state propagates research_mode correctly for
both fast and research CLI flag values.
"""

from __future__ import annotations

from main import _build_initial_state


# ---------------------------------------------------------------------------
# Feature 1: CLI --mode flag propagates to initial state
# ---------------------------------------------------------------------------


def test_build_initial_state_fast_mode_sets_research_mode() -> None:
    """_build_initial_state with 'fast' sets research_mode='fast' in state."""
    state = _build_initial_state("test query", "markdown", "session-1", "fast")
    assert state["research_mode"] == "fast"


def test_build_initial_state_research_mode_sets_research_mode() -> None:
    """_build_initial_state with 'research' sets research_mode='research' in state."""
    state = _build_initial_state("test query", "markdown", "session-1", "research")
    assert state["research_mode"] == "research"

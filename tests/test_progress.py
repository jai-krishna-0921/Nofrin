"""
tests/test_progress.py

Unit tests for graph/progress.py — latency instrumentation (Feature 3).

Tests cover: _step_end records end time, pipeline_summary prints registered
nodes, flags slow nodes, skips nodes with missing start/end, _step_elapsed
returns a formatted string, and the zero-division guard.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import graph.progress as progress_mod


@pytest.fixture(autouse=True)
def reset_progress_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level timer dicts before each test to prevent cross-test pollution."""
    monkeypatch.setattr(progress_mod, "_STEP_TIMERS", {})
    monkeypatch.setattr(progress_mod, "_NODE_END_TIMES", {})


# ---------------------------------------------------------------------------
# Feature 3: Latency instrumentation tests
# ---------------------------------------------------------------------------


def test_step_end_records_end_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """_step_end stores the current monotonic time in _NODE_END_TIMES."""
    progress_mod._step_start("supervisor")
    progress_mod._step_end("supervisor")
    assert "supervisor" in progress_mod._NODE_END_TIMES
    assert isinstance(progress_mod._NODE_END_TIMES["supervisor"], float)


def test_step_elapsed_returns_seconds_string() -> None:
    """_step_elapsed returns a non-empty elapsed string after _step_start is called."""
    progress_mod._step_start("critic")
    elapsed = progress_mod._step_elapsed("critic")
    assert isinstance(elapsed, str)
    assert len(elapsed) > 0
    assert "s" in elapsed


def test_pipeline_summary_prints_registered_node(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """pipeline_summary prints a line for nodes that have both start and end times."""
    now = time.monotonic()
    progress_mod._STEP_TIMERS["supervisor"] = now - 3.0
    progress_mod._NODE_END_TIMES["supervisor"] = now

    progress_mod.pipeline_summary()

    captured = capsys.readouterr()
    assert "supervisor" in captured.err


def test_pipeline_summary_flags_slow_node(capsys: pytest.CaptureFixture[str]) -> None:
    """Nodes with elapsed > 10s are flagged as [SLOW] in pipeline_summary output."""
    now = time.monotonic()
    # elapsed = 15s, well above the 10s threshold
    progress_mod._STEP_TIMERS["coordinator"] = now - 15.0
    progress_mod._NODE_END_TIMES["coordinator"] = now

    progress_mod.pipeline_summary()

    captured = capsys.readouterr()
    assert "[SLOW]" in captured.err


def test_pipeline_summary_skips_node_with_missing_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nodes with a start time but no end time are skipped in pipeline_summary."""
    now = time.monotonic()
    progress_mod._STEP_TIMERS["grounding"] = now - 2.0
    # _NODE_END_TIMES["grounding"] intentionally not set

    progress_mod.pipeline_summary()

    captured = capsys.readouterr()
    assert "grounding" not in captured.err


def test_pipeline_summary_zero_division_guard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When total elapsed == 0, pct is 0.0 with no ZeroDivisionError."""
    fixed_time = 1000.0
    monkeypatch.setattr(progress_mod, "_START_TIME", fixed_time)
    progress_mod._STEP_TIMERS["supervisor"] = fixed_time - 2.0
    progress_mod._NODE_END_TIMES["supervisor"] = fixed_time - 0.5

    # Mock time.monotonic() to return _START_TIME so total = 0
    with patch("graph.progress.time") as mock_time:
        mock_time.monotonic.return_value = fixed_time
        progress_mod.pipeline_summary()

    captured = capsys.readouterr()
    assert "supervisor" in captured.err
    assert "0.0%" in captured.err

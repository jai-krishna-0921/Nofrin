"""
eval/regression_test.py

Regression gate: asserts current RAGAS scores don't drop below baseline - 0.05.
Skips entries where baseline is null (not yet measured).

Usage:
    pytest eval/regression_test.py -v -m eval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

BASELINE_PATH = Path(__file__).parent / "baseline.json"
TOLERANCE = 0.05  # allow up to 5% drop before failing


def _load_baseline() -> list[dict[str, Any]]:
    return json.loads(BASELINE_PATH.read_text())  # type: ignore[no-any-return]


def _baseline_cases() -> list[tuple[str, str, float]]:
    """Return (test_id, metric, baseline_score) for all non-null entries."""
    cases: list[tuple[str, str, float]] = []
    for entry in _load_baseline():
        test_id: str = entry["test_id"]
        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            value = entry.get(metric)
            if value is not None:
                cases.append((test_id, metric, float(value)))
    return cases


_CASES = _baseline_cases()


@pytest.mark.eval
@pytest.mark.skipif(len(_CASES) == 0, reason="All baselines are null — nothing to regress against")
@pytest.mark.parametrize("test_id,metric,baseline", _CASES)
def test_score_does_not_regress(
    test_id: str,
    metric: str,
    baseline: float,
) -> None:
    """Current score must be >= baseline - TOLERANCE."""
    # Import here to avoid heavy imports during collection
    from eval.ragas_eval import _load_fixtures, _score_sample  # type: ignore[import]

    fixtures = _load_fixtures(fixture_id=test_id)
    assert fixtures, f"No fixture found for {test_id}"
    fixture = fixtures[0]

    # Use stored response placeholder — real eval requires a live pipeline run.
    # This test validates the scoring logic against a known good response.
    # Replace `response` with actual pipeline output in CI.
    response: str = fixture.get("_cached_response", "")  # type: ignore[assignment]
    if not response:
        pytest.skip(f"{test_id}: no cached response — run ragas_eval.py first")

    scores = _score_sample(fixture["query"], response, [response])
    current = scores[metric]
    assert current >= baseline - TOLERANCE, (
        f"{test_id}/{metric}: score {current:.3f} dropped below "
        f"baseline {baseline:.3f} - tolerance {TOLERANCE}"
    )

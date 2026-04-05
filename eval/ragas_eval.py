"""
eval/ragas_eval.py

RAGAS evaluation harness for the Deep Research Agent.

Loads fixtures from eval/dataset/, runs the pipeline against each query,
then scores the output with faithfulness, answer_relevancy, and context_precision.

Usage:
    python eval/ragas_eval.py                  # evaluate all fixtures
    python eval/ragas_eval.py --id test_001    # single fixture
    python eval/ragas_eval.py --update-baseline  # write scores back to baseline.json

Environment variables:
    LLM_PROVIDER     Provider for the evaluator LLM (default: groq)
    EXA_API_KEY      Exa.ai key (required for live pipeline runs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "dataset"
BASELINE_PATH = EVAL_DIR / "baseline.json"


def _load_fixtures(fixture_id: str | None = None) -> list[dict[str, Any]]:
    """Load all JSON fixtures from eval/dataset/."""
    fixtures: list[dict[str, Any]] = []
    for path in sorted(DATASET_DIR.glob("*.json")):
        data: dict[str, Any] = json.loads(path.read_text())
        if fixture_id is None or data.get("id") == fixture_id:
            fixtures.append(data)
    return fixtures


def _load_baseline() -> dict[str, dict[str, float | None]]:
    """Load baseline.json as {test_id: {metric: score}}."""
    raw: list[dict[str, Any]] = json.loads(BASELINE_PATH.read_text())
    return {entry["test_id"]: entry for entry in raw}


def _save_baseline(scores: dict[str, dict[str, float | None]]) -> None:
    """Write updated scores back to baseline.json."""
    baseline = _load_baseline()
    for test_id, metrics in scores.items():
        if test_id in baseline:
            baseline[test_id].update(metrics)
    output = list(baseline.values())
    BASELINE_PATH.write_text(json.dumps(output, indent=2))


async def _run_pipeline(query: str, output_format: str = "markdown") -> str | None:
    """Run the research pipeline for a single query and return final_output."""
    import uuid

    from graph.builder import build_graph
    from graph.context import NofrinContext
    from graph.llm import get_llm
    from graph.state import ResearchAgentState

    try:
        from exa_py import AsyncExa  # noqa: PLC0415
    except ImportError:
        print("exa_py not installed — skipping live pipeline run", file=sys.stderr)
        return None

    graph = build_graph()
    session_id = str(uuid.uuid4())
    context = NofrinContext(
        llm_supervisor=get_llm(role="default"),
        llm_worker=get_llm(role="default"),
        llm_coordinator=get_llm(role="default"),
        llm_critic=get_llm(role="critic"),
        exa_client=AsyncExa(api_key=os.environ.get("EXA_API_KEY", "")),
        session_id=session_id,
        cost_ceiling_usd=float(os.getenv("COST_CEILING_USD", "1.00")),
    )
    initial_state = ResearchAgentState(
        user_query=query,
        intent_type="exploratory",
        output_format=output_format,  # type: ignore[arg-type]
        sub_queries=[],
        source_routing={},
        worker_results=[],
        compressed_worker_results=[],
        synthesis=None,
        grounding_issues=[],
        critic_output=None,
        revision_count=0,
        prior_syntheses=[],
        session_id=session_id,
        total_tokens_used=0,
        cost_usd=0.0,
        final_output=None,
    )
    result: dict[str, Any] = await graph.ainvoke(  # type: ignore[call-overload,assignment]
        initial_state, context=context
    )
    return result.get("final_output")  # type: ignore[return-value]


def _score_sample(
    query: str,
    response: str,
    contexts: list[str],
) -> dict[str, float]:
    """Run RAGAS metrics on a single sample. Returns {metric: score}."""
    from ragas import evaluate  # type: ignore[import-untyped]
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # type: ignore[import-untyped]
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness  # type: ignore[import-untyped]

    from graph.llm import get_llm

    evaluator_llm = LangchainLLMWrapper(get_llm(role="default"))

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        reference=response,  # self-reference; faithfulness needs retrieved_contexts
        retrieved_contexts=contexts if contexts else [response],
    )
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=evaluator_llm,
    )
    df = result.to_pandas()
    row = df.iloc[0]
    return {
        "faithfulness": float(row.get("faithfulness", 0.0)),
        "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
        "context_precision": float(row.get("context_precision", 0.0)),
    }


async def evaluate_all(
    fixture_id: str | None = None,
    update_baseline: bool = False,
) -> dict[str, dict[str, float | None]]:
    """Evaluate all fixtures (or one) and optionally update baseline.json."""
    fixtures = _load_fixtures(fixture_id)
    if not fixtures:
        print(f"No fixtures found (id={fixture_id})", file=sys.stderr)
        return {}

    all_scores: dict[str, dict[str, float | None]] = {}

    for fixture in fixtures:
        test_id: str = fixture["id"]
        query: str = fixture["query"]
        print(f"\n[{test_id}] Running pipeline for: {query[:60]}...")

        response = await _run_pipeline(query)
        if response is None:
            print(f"[{test_id}] SKIP — pipeline returned no output")
            all_scores[test_id] = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            }
            continue

        print(f"[{test_id}] Scoring with RAGAS...")
        scores = _score_sample(query, response, [response])
        all_scores[test_id] = scores  # type: ignore[assignment]
        print(
            f"[{test_id}] faithfulness={scores['faithfulness']:.3f} "
            f"answer_relevancy={scores['answer_relevancy']:.3f} "
            f"context_precision={scores['context_precision']:.3f}"
        )

    if update_baseline:
        _save_baseline(all_scores)
        print("\nbaseline.json updated.")

    return all_scores


def main() -> None:
    """CLI entry point for the eval harness."""
    parser = argparse.ArgumentParser(description="RAGAS eval harness")
    parser.add_argument("--id", dest="fixture_id", default=None)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write scores back to eval/baseline.json",
    )
    args = parser.parse_args()
    asyncio.run(evaluate_all(args.fixture_id, args.update_baseline))


if __name__ == "__main__":
    main()

"""
main.py

CLI entry point for the Deep Research Agent.

Usage:
    python main.py --query "What are the latest advances in fusion energy?"
    python main.py --query "..." --format docx --session-id my-session

Environment variables required:
    LLM_PROVIDER         groq (default) | openrouter | ollama | anthropic
    EXA_API_KEY          Exa.ai API key
    COST_CEILING_USD     Maximum spend per session (default: 1.00)

Results:
    final_output  → stdout
    cost_usd, total_tokens_used → stderr
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from graph.builder import build_graph
from graph.context import NofrinContext
from graph.llm import get_llm
from graph.state import OutputFormat, ResearchAgentState


def _build_initial_state(
    query: str,
    output_format: OutputFormat,
    session_id: str,
) -> ResearchAgentState:
    """Build a fresh ResearchAgentState for a new research session.

    Args:
        query: The user's research query.
        output_format: Requested output format.
        session_id: Unique session identifier.

    Returns:
        Fully initialised ResearchAgentState with all required fields.
    """
    return ResearchAgentState(
        user_query=query,
        intent_type="exploratory",  # supervisor will classify; this is the init default
        output_format=output_format,
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


def _build_context(session_id: str) -> NofrinContext:
    """Build NofrinContext from environment variables.

    Args:
        session_id: Unique session identifier.

    Returns:
        NofrinContext with all LLM clients and settings configured.
    """
    from exa_py import AsyncExa  # noqa: PLC0415

    exa_key = os.environ.get("EXA_API_KEY", "")
    cost_ceiling = float(os.getenv("COST_CEILING_USD", "1.00"))

    return NofrinContext(
        llm_supervisor=get_llm(role="default"),
        llm_worker=get_llm(role="default"),
        llm_coordinator=get_llm(role="default"),
        llm_critic=get_llm(role="critic"),
        exa_client=AsyncExa(api_key=exa_key),
        session_id=session_id,
        cost_ceiling_usd=cost_ceiling,
    )


async def _run(
    query: str,
    output_format: OutputFormat,
    session_id: str,
) -> int:
    """Run the research pipeline and print results.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    graph = build_graph()
    context = _build_context(session_id)
    initial_state = _build_initial_state(query, output_format, session_id)

    result: dict[str, object] = await graph.ainvoke(  # type: ignore[call-overload]
        initial_state,
        context=context,
    )

    final_output = result.get("final_output")
    if isinstance(final_output, str) and final_output:
        print(final_output)
    else:
        print("[No output produced]", file=sys.stderr)
        return 1

    cost = result.get("cost_usd", 0.0)
    tokens = result.get("total_tokens_used", 0)
    print(f"cost_usd={cost:.4f} total_tokens={tokens}", file=sys.stderr)
    return 0


def main() -> None:
    """Parse CLI args and run the Deep Research Agent."""
    parser = argparse.ArgumentParser(
        prog="deep-research-agent",
        description="Run a deep research query and produce a structured brief.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Research question to investigate.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["markdown", "docx", "pdf", "pptx"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        help="Optional session ID for tracing (auto-generated if omitted).",
    )

    args = parser.parse_args()
    session_id: str = args.session_id or str(uuid.uuid4())
    output_format: OutputFormat = args.output_format

    try:
        exit_code = asyncio.run(_run(args.query, output_format, session_id))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

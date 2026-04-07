"""
graph/progress.py

Lightweight progress logger for the Deep Research Agent.
Prints timestamped, human-readable lines to stderr so the terminal shows
live pipeline progress without interfering with stdout (the final report).

Colour codes (ANSI, stripped when stderr is not a TTY):
  cyan    — pipeline lifecycle (start / done)
  yellow  — agent steps (supervisor, coordinator, critic, grounding, delivery)
  blue    — Exa search calls
  green   — evidence extracted from a URL
  red     — warnings / errors
  dim     — sub-detail lines
"""

from __future__ import annotations

import sys
import time
from typing import TextIO


def _supports_colour(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


_USE_COLOUR = _supports_colour(sys.stderr)

# ANSI codes
_RESET = "\033[0m" if _USE_COLOUR else ""
_CYAN = "\033[36m" if _USE_COLOUR else ""
_YELLOW = "\033[33m" if _USE_COLOUR else ""
_BLUE = "\033[34m" if _USE_COLOUR else ""
_GREEN = "\033[32m" if _USE_COLOUR else ""
_RED = "\033[31m" if _USE_COLOUR else ""
_DIM = "\033[2m" if _USE_COLOUR else ""
_BOLD = "\033[1m" if _USE_COLOUR else ""

_START_TIME: float = time.monotonic()
_STEP_TIMERS: dict[str, float] = {}
_NODE_END_TIMES: dict[str, float] = {}
_SLOW_THRESHOLD_SECS: float = 10.0


def _ts() -> str:
    """Elapsed seconds since first import, formatted as [+HH:MM:SS]."""
    elapsed = int(time.monotonic() - _START_TIME)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"[+{h:02d}:{m:02d}:{s:02d}]"


def _step_start(name: str) -> None:
    _STEP_TIMERS[name] = time.monotonic()


def _step_end(name: str) -> None:
    """Record the wall-clock time when a step finishes."""
    _NODE_END_TIMES[name] = time.monotonic()


def _step_elapsed(name: str) -> str:
    """Return '12s' or '1m 34s' since _step_start(name) was called."""
    start = _STEP_TIMERS.get(name)
    if start is None:
        return ""
    secs = int(time.monotonic() - start)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


def _emit(colour: str, prefix: str, message: str) -> None:
    print(
        f"{_DIM}{_ts()}{_RESET} {colour}{_BOLD}{prefix}{_RESET} {message}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Public API — called from agent nodes
# ---------------------------------------------------------------------------


def pipeline_start(query: str) -> None:
    _step_start("pipeline")
    _emit(_CYAN, "▶ PIPELINE", f"query: {query!r}")


def pipeline_done(cost_usd: float, total_tokens: int) -> None:
    elapsed = _step_elapsed("pipeline")
    _emit(
        _CYAN,
        "✓ PIPELINE",
        f"done — cost=${cost_usd:.4f}  tokens={total_tokens:,}  total={elapsed}",
    )


def pipeline_summary() -> None:
    """Print per-node elapsed time table after pipeline completes. Flags nodes >10s as [SLOW]."""
    _emit(_CYAN, "PIPELINE SUMMARY", "per-node latency")
    total = time.monotonic() - _START_TIME
    nodes = ["supervisor", "coordinator", "grounding", "critic", "delivery"]
    for node in nodes:
        start = _STEP_TIMERS.get(node)
        end = _NODE_END_TIMES.get(node)
        if start is None or end is None:
            continue
        elapsed = end - start
        pct = (elapsed / total * 100) if total > 0 else 0.0
        slow = f" {_RED}[SLOW]{_RESET}" if elapsed > _SLOW_THRESHOLD_SECS else ""
        print(
            f"  {_DIM}{node:<20}{_RESET}  {elapsed:6.1f}s  ({pct:4.1f}%){slow}",
            file=sys.stderr,
        )
    print(f"  {_DIM}{'TOTAL':<20}{_RESET}  {total:6.1f}s", file=sys.stderr)


def supervisor_start(query: str) -> None:
    _step_start("supervisor")
    _emit(_YELLOW, "  SUPERVISOR", "classifying intent + decomposing query…")


def supervisor_done(intent: str, sub_queries: list[str]) -> None:
    _step_end("supervisor")
    _emit(
        _YELLOW,
        "  SUPERVISOR",
        f"intent={intent}  sub-queries={len(sub_queries)}  [{_step_elapsed('supervisor')}]",
    )
    for i, sq in enumerate(sub_queries, 1):
        _emit(_DIM, f"    [{i}]", sq)


def worker_exa_search(worker_id: str, sub_query: str, source_type: str) -> None:
    label = sub_query[:70] + ("…" if len(sub_query) > 70 else "")
    _emit(_BLUE, f"  EXA [{source_type}]", f"worker={worker_id}  query={label!r}")


def worker_exa_results(worker_id: str, n_results: int) -> None:
    _emit(_DIM, "    ↳ results", f"{n_results} URLs returned  (worker={worker_id})")


def worker_url_ok(worker_id: str, url: str, claim_preview: str) -> None:
    label = claim_preview[:80] + ("…" if len(claim_preview) > 80 else "")
    _emit(_GREEN, "    ✓ evidence", f"{url}  →  {label!r}")


def worker_url_fail(worker_id: str, url: str, reason: str) -> None:
    _emit(_RED, "    ✗ failed", f"worker={worker_id}  url={url}  reason={reason}")


def worker_done(worker_id: str, n_evidence: int, tokens: int) -> None:
    _emit(
        _DIM,
        "    ↳ worker done",
        f"worker={worker_id}  evidence={n_evidence}  tokens={tokens:,}",
    )


def coordinator_start(revision_count: int, n_workers: int) -> None:
    _step_start("coordinator")
    pass_label = (
        "first-pass synthesis"
        if revision_count == 0
        else f"revision {revision_count}/2"
    )
    _emit(_YELLOW, "  COORDINATOR", f"{pass_label}  ({n_workers} worker results)")


def coordinator_done(n_findings: int, tokens: int) -> None:
    _step_end("coordinator")
    _emit(
        _YELLOW,
        "  COORDINATOR",
        f"done — findings={n_findings}  tokens={tokens:,}  [{_step_elapsed('coordinator')}]",
    )


def grounding_start(n_findings: int) -> None:
    _step_start("grounding")
    _emit(_YELLOW, "  GROUNDING", f"checking {n_findings} findings against evidence…")


def grounding_done(n_issues: int) -> None:
    _step_end("grounding")
    status = "clean" if n_issues == 0 else f"{n_issues} issue(s) found"
    _emit(_YELLOW, "  GROUNDING", f"done — {status}  [{_step_elapsed('grounding')}]")


def critic_start(revision_count: int) -> None:
    _step_start("critic")
    _emit(_YELLOW, "  CRITIC", f"adversarial review (revision_count={revision_count})…")


def critic_done(score: float, passed: bool, n_issues: int) -> None:
    _step_end("critic")
    badge = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
    _emit(
        _YELLOW,
        "  CRITIC",
        f"score={score:.2f}  {badge}  issues={n_issues}  [{_step_elapsed('critic')}]",
    )


def delivery_start(output_format: str) -> None:
    _step_start("delivery")
    _emit(_YELLOW, "  DELIVERY", f"rendering output as {output_format}…")


def delivery_done(output_format: str, char_count: int) -> None:
    _step_end("delivery")
    _emit(
        _YELLOW,
        "  DELIVERY",
        f"done — {output_format}  ({char_count:,} chars)  [{_step_elapsed('delivery')}]",
    )


def warn(agent: str, message: str) -> None:
    _emit(_RED, f"  WARN [{agent}]", message)

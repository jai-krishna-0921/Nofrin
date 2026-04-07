# Deep Research Agent v2 Enhancements — Implementation Plan

## Problem Statement

The current pipeline operates in a single mode optimized for comprehensive research: 5 sub-queries, full Exa content retrieval, and up to 2 revision passes. This makes every query expensive (~$0.15–0.45) and slow (30–90s). Users need:

1. A **fast mode** for quick lookups that trades depth for speed and cost.
2. **Multi-provider search routing** to improve resilience (Tavily for fast/web, Exa for academic, Brave fallback).
3. **Per-node latency instrumentation** so bottlenecks are visible at pipeline end.

---

## Feature 1 — Dual Mode (`--mode fast|research`)

### Approach

Add a `ResearchMode` literal to state and a `--mode` CLI flag. Nodes that behave differently read `state["research_mode"]` directly — no changes to `NofrinContext`.

**Why state, not context?** Context holds injected dependencies (LLM clients, API clients). Mode is an execution parameter — it belongs in state alongside `revision_count` and `output_format`.

### Files to Modify

| File | Change |
|------|--------|
| `graph/state.py` | Add `ResearchMode = Literal["fast", "research"]`; add `research_mode: ResearchMode` field |
| `main.py` | Add `--mode` argparse arg, pass to initial state |
| `agents/supervisor.py` | Limit sub-queries to 3 in fast mode |
| `graph/router.py` | Skip revision loop in fast mode (`route_after_critic` returns `"delivery"` immediately) |

> `agents/worker.py` changes are in Feature 2.

### Code Snippets

**`graph/state.py`** — add after existing `Literal` imports:
```python
ResearchMode = Literal["fast", "research"]
```
Add field to `ResearchAgentState` (after `output_format`):
```python
research_mode: ResearchMode  # "fast" or "research"
```

**`main.py`** — add after `--format` arg:
```python
parser.add_argument(
    "--mode",
    dest="research_mode",
    choices=["fast", "research"],
    default="fast",
    help="'fast' (3 queries, no revision) or 'research' (5 queries, 2 revisions). Default: fast.",
)
```
Pass to initial state:
```python
initial_state: ResearchAgentState = {
    ...existing fields...,
    "research_mode": args.research_mode,
}
```

**`graph/router.py`** — inside `route_after_critic`, before all other checks:
```python
# Fast mode: skip revision loop entirely
if state.get("research_mode", "fast") == "fast":
    return "delivery"
```

**`agents/supervisor.py`** — update `_validate_output` to accept mode and enforce sub-query count:
```python
def _validate_output(output: SupervisorOutput, research_mode: str = "research") -> None:
    count = len(output.sub_queries)
    if research_mode == "fast":
        if count != 3:
            raise AgentParseError(f"Fast mode requires exactly 3 sub-queries; got {count}.")
    else:
        if count < 3 or count > 5:
            raise AgentParseError(f"Supervisor returned {count} sub-queries; expected 3–5.")
```
Read `state["research_mode"]` in `supervisor_node` and pass to `_validate_output`.

### Tests to Add

| File | Count | What |
|------|-------|------|
| `tests/test_state.py` | +2 | `ResearchMode` type alias present; `research_mode` field in state |
| `tests/test_supervisor.py` | +4 | Fast mode enforces 3 queries; research mode allows 3–5 |
| `tests/test_router.py` (new) | +3 | Fast mode routes directly to delivery regardless of score |
| `tests/test_main.py` (new) | +2 | CLI parses `--mode fast` and `--mode research` |

**Feature 1 total: ~11 new tests**

### Failure Modes

| Failure | Mitigation |
|---------|------------|
| LLM ignores 3-query limit in fast mode | `_validate_output` raises `AgentParseError` → retry |
| `research_mode` absent from old state | `.get("research_mode", "fast")` safe default |
| Invalid `--mode` value | `argparse choices=` rejects at CLI level |

### Token Cost Impact

| Mode | Sub-queries | Revision passes | Approx. cost |
|------|-------------|-----------------|--------------|
| `research` | 5 | 0–2 | $0.15–0.45 |
| `fast` | 3 | 0 | $0.06–0.09 |

**Fast mode is ~60–70% cheaper.**

---

## Feature 2 — Multi-Provider Worker Routing

### Approach

Add `tavily_client` (optional) and `brave_api_key` (optional) to `NofrinContext`. Worker selects provider based on `research_mode` and `source_type`:

| Mode | source_type | Primary | Fallback |
|------|-------------|---------|----------|
| `fast` | `web` | Tavily (basic) | Brave |
| `fast` | `news` | Tavily (basic) | Brave |
| `fast` | `academic` | Exa | Brave |
| `research` | all | Exa | Brave |

Brave is always fallback-only; it triggers only when the primary call fails.

### Files to Modify

| File | Change |
|------|--------|
| `graph/context.py` | Add `tavily_client: AsyncTavilyClient | None = None` and `brave_api_key: str | None = None` |
| `agents/worker.py` | Add provider selection, `_search_tavily()`, `_search_brave()`, unified result mapping |
| `main.py` | Initialise Tavily client and read `BRAVE_API_KEY` from env |

### Code Snippets

**`graph/context.py`**:
```python
from tavily import AsyncTavilyClient  # already in requirements or add it

@dataclass
class NofrinContext:
    # ...existing fields...
    tavily_client: AsyncTavilyClient | None = None
    brave_api_key: str | None = None
```

**`agents/worker.py`** — provider selection helper:
```python
def _select_provider(
    source_type: SourceType,
    research_mode: str,
    has_tavily: bool,
) -> str:
    if source_type == "academic":
        return "exa"
    if research_mode == "fast" and has_tavily:
        return "tavily"
    return "exa"
```

**`agents/worker.py`** — Tavily search:
```python
async def _search_tavily(
    client: AsyncTavilyClient,
    query: str,
    source_type: SourceType,
) -> list[ExaResult]:
    resp = await client.search(
        query,
        search_depth="basic",
        max_results=5,
        include_raw_content=True,
    )
    return [
        ExaResult(url=r["url"], title=r.get("title", ""), text=r.get("content", ""), highlights=[], published_date=r.get("published_date"))
        for r in resp.get("results", [])
    ]
```
(Reuse existing `ExaResult` dataclass for uniformity; avoids a new type just for this.)

**Brave fallback** — only triggered via `return_exceptions=True` path already in place:
```python
async def _search_brave(api_key: str, query: str) -> list[ExaResult]:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key},
            params={"q": query, "count": 5},
            timeout=10,
        )
        r.raise_for_status()
        return [
            ExaResult(url=h["url"], title=h.get("title", ""), text=h.get("description", ""), highlights=[], published_date=None)
            for h in r.json().get("web", {}).get("results", [])
        ]
```

### Tests to Add

| File | Count | What |
|------|-------|------|
| `tests/test_worker.py` | +6 | `_select_provider` for all mode × source_type combos |
| `tests/test_worker.py` | +3 | Tavily result mapped to `ExaResult` shape |
| `tests/test_worker.py` | +2 | Brave fallback triggers when primary raises |
| `tests/test_context.py` (new) | +2 | Optional client fields default to `None` |

**Feature 2 total: ~13 new tests**

### Failure Modes

| Failure | Mitigation |
|---------|------------|
| `TAVILY_API_KEY` missing | `tavily_client=None` → fall through to Exa |
| `BRAVE_API_KEY` missing | Brave fallback skipped; worker returns empty evidence |
| Tavily rate limit / 429 | Exception caught by `return_exceptions=True`; worker returns `[]` |

### Token / Cost Impact

No LLM token impact. Search costs:
- Tavily basic: ~$0.001/query, 0.5–1 s latency
- Exa: ~$0.001/query, 1–3 s latency
- Brave: ~$0.003/query, 0.5–1 s latency

---

## Feature 3 — Latency Instrumentation (Pipeline Summary)

### Approach

Extend `graph/progress.py` with a `pipeline_summary()` function. Called from `main.py` after `pipeline_done()`. Reads `_STEP_TIMERS` (already populated by every node's `*_start()` call) and prints a table. Flags nodes exceeding 10 s as `[SLOW]`.

### Files to Modify

| File | Change |
|------|--------|
| `graph/progress.py` | Add `_SLOW_THRESHOLD_SECS`, `_NODE_ELAPSED` dict (records when each `*_done` call happens), and `pipeline_summary()` |
| `main.py` | Call `pipeline_summary()` after `pipeline_done()` |

### Code Snippet

**`graph/progress.py`** additions:

```python
_SLOW_THRESHOLD_SECS: float = 10.0
_NODE_END_TIMES: dict[str, float] = {}  # populated by each *_done function


def _step_end(name: str) -> None:
    """Record the wall-clock time when a step finishes."""
    _NODE_END_TIMES[name] = time.monotonic()


def pipeline_summary() -> None:
    """Print per-node elapsed time table. Flag nodes >10 s as [SLOW]."""
    _emit(_CYAN, "PIPELINE SUMMARY", "per-node latency")
    total = time.monotonic() - _START_TIME
    nodes = [
        "supervisor", "coordinator", "grounding", "critic", "delivery",
    ]
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
    print(
        f"  {_DIM}{'TOTAL':<20}{_RESET}  {total:6.1f}s",
        file=sys.stderr,
    )
```

Wire `_step_end("coordinator")` into `coordinator_done()`, etc. (one line per existing `*_done` function).

### Tests to Add

| File | Count | What |
|------|-------|------|
| `tests/test_progress.py` (new) | +3 | Summary prints all nodes; flags slow node; skips missing nodes |
| `tests/test_progress.py` | +2 | `_step_end` records time; `_step_elapsed` calculation |
| `tests/test_progress.py` | +1 | Zero-division guard when total=0 |

**Feature 3 total: ~6 new tests**

### Failure Modes

| Failure | Mitigation |
|---------|------------|
| Node never ran (e.g., revision skipped) | `if start is None or end is None: continue` |
| `total_elapsed == 0` | `if total > 0` guard |

### Token / Cost Impact

**Zero.** Pure logging.

---

## Dependency Order & Recommended Sequence

```
Feature 1 (Dual Mode)           ← must land first (Feature 2 reads research_mode)
    └── Feature 2 (Multi-Provider)
Feature 3 (Latency)             ← independent, can land any time
```

Recommended: implement 1 → 3 → 2 (latency is a quick standalone win).

---

## Summary

| Feature | Files changed | New tests | Cost impact |
|---------|---------------|-----------|-------------|
| 1 — Dual Mode | 4 | ~11 | −60% in fast mode |
| 2 — Multi-Provider | 3 | ~13 | ~0 |
| 3 — Latency | 2 | ~6 | 0 |
| **Total** | **9** | **~30** | |

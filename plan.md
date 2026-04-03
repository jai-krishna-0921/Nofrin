# Plan: Send() Fan-out Parallel Worker Dispatch

> **Context:** All nodes done (166/166 tests). This wires the supervisor → workers fan-out that was stubbed as `add_edge("supervisor", "boundary_compressor")`.

---

## Problem Statement

`supervisor_node` decomposes the user query into 3–5 sub-queries with source-type routing (web/academic/news) and writes them to `state["sub_queries"]` and `state["source_routing"]`. The worker node (`agents/worker.py`) is fully implemented and accepts `WorkerInput` via Send(). The fan-out dispatch function and its graph wiring are the only missing pieces. `worker_results` already uses `Annotated[list[WorkerResult], operator.add]`, so parallel results accumulate automatically.

---

## LangGraph Send() API (from context7)

```python
from langgraph.types import Send

Send(node: str, arg: Any)
```

- `node` — target node name string
- `arg` — state payload passed to that node invocation

**Fan-out pattern:**
```python
def dispatch_function(state: StateType) -> list[Send]:
    return [Send("worker", payload) for payload in payloads]

builder.add_conditional_edges("supervisor", dispatch_function, ["worker"])
builder.add_edge("worker", "boundary_compressor")  # fan-in
```

- When the conditional function returns `list[Send]`, LangGraph spawns parallel invocations.
- The `["worker"]` in the third arg is a list of reachable node names (not a dict — this is different from string-routing).
- Fan-in: `add_edge("worker", "boundary_compressor")` ensures all workers complete before boundary_compressor runs.
- `operator.add` reducer on `worker_results` automatically concatenates each worker's `{"worker_results": [one_result]}`.

---

## Answers to Planning Questions

### Q1: Which node dispatches the Send() calls?

A `dispatch_workers` function in **`graph/router.py`** (keeps all routing logic together alongside `route_after_critic`). It reads `state["sub_queries"]` and `state["source_routing"]` after supervisor completes and returns `list[Send]`. No LLM calls — pure state transformation.

### Q2: What does each Send() carry?

```python
Send("worker", WorkerInput(
    worker_id="worker-0",
    sub_query="...",
    source_type="academic",
))
```

`worker_node` signature (confirmed from `agents/worker.py`):
```python
async def worker_node(state: WorkerInput, runtime: Runtime[NofrinContext]) -> dict[str, list[WorkerResult]]:
```
Worker reads only `worker_id`, `sub_query`, `source_type` — the three `WorkerInput` fields. Send() must pass `WorkerInput`, not `ResearchAgentState`.

### Q3: How do worker results merge back?

`worker_results: Annotated[list[WorkerResult], operator.add]` — LangGraph applies the reducer automatically. Each worker returns `{"worker_results": [single_result]}`. After all N workers complete, `worker_results` contains all N items. No merge code needed.

### Q4: Where does boundary_compressor fit?

After all workers complete (fan-in via `add_edge("worker", "boundary_compressor")`). LangGraph guarantees all parallel Send() invocations finish before the next node runs.

### Q5: Changes in graph/builder.py

```diff
- builder.add_edge("supervisor", "boundary_compressor")
+ builder.add_conditional_edges("supervisor", dispatch_workers, ["worker"])
+ builder.add_edge("worker", "boundary_compressor")
```

Add imports inside `build_graph()`:
```python
from agents.worker import worker_node
from graph.router import budget_gate_node, dispatch_workers, route_after_critic
```

Add node:
```python
builder.add_node("worker", worker_node)  # type: ignore[arg-type]
```

Remove the TODO comment block that described this change as future work.

### Q6: Changes in graph/router.py

Add `dispatch_workers` function and update `__all__`. No changes to existing functions.

New imports needed at top of router.py:
```python
from langgraph.types import Send
```

(Already imports `ResearchAgentState`, `SourceType`, `WorkerInput` — verify and add any missing.)

### Q7: New agent files?

**None.** `worker_node` is already implemented. This is purely graph wiring.

### Q8: Tests

10 test cases in `tests/test_dispatch_workers.py` — see table below.

### Q9: WorkerInput vs ResearchAgentState

**`WorkerInput`** — confirmed from `agents/worker.py`. Worker reads only `worker_id`, `sub_query`, `source_type`.

### Q10: worker_id assignment

Sequential: `"worker-0"`, `"worker-1"`, … — deterministic, index-correlated, no UUID overhead.

---

## Files to Modify

| Path | Action |
|---|---|
| `graph/router.py` | **Update** — add `dispatch_workers` function, update `__all__` |
| `graph/builder.py` | **Update** — add worker node, replace supervisor→boundary_compressor edge with fan-out |
| `tests/test_dispatch_workers.py` | **Create** — 10 tests |

---

## dispatch_workers Signature (full)

```python
def dispatch_workers(
    state: ResearchAgentState,
) -> list[Send]:
    """Fan-out dispatcher: create one Send() per sub-query targeting the worker node.

    Called via add_conditional_edges after supervisor_node completes.
    Reads state["sub_queries"] and state["source_routing"] (written by supervisor).

    Worker IDs are sequential: "worker-0", "worker-1", etc.
    If a sub_query is not in source_routing, defaults to "web".

    Args:
        state: Current ResearchAgentState after supervisor node.

    Returns:
        list[Send] — one Send("worker", WorkerInput) per sub-query.

    Raises:
        ValueError: If sub_queries is empty.
    """
```

---

## graph/router.py changes (exact additions)

**New import** (add to existing imports at top):
```python
from langgraph.types import Send
```

**New function** (add before `__all__`):
```python
# ---------------------------------------------------------------------------
# Fan-out dispatcher: supervisor → workers
# ---------------------------------------------------------------------------


def dispatch_workers(
    state: ResearchAgentState,
) -> list[Send]:
    """Fan-out dispatcher: create one Send() per sub-query..."""
    sub_queries: list[str] = state["sub_queries"]
    source_routing: dict[str, SourceType] = state["source_routing"]

    if not sub_queries:
        raise ValueError(
            "dispatch_workers: sub_queries is empty — "
            "supervisor_node must populate sub_queries before dispatch."
        )

    sends: list[Send] = []
    for idx, sub_query in enumerate(sub_queries):
        source_type: SourceType = source_routing.get(sub_query, "web")
        worker_input = WorkerInput(
            worker_id=f"worker-{idx}",
            sub_query=sub_query,
            source_type=source_type,
        )
        sends.append(Send("worker", worker_input))

    return sends
```

**Update `__all__`:**
```python
__all__ = ["budget_gate_node", "dispatch_workers", "route_after_critic"]
```

---

## graph/builder.py changes (exact diff)

```diff
 # ── Nodes ──────────────────────────────────────────────────────────────────

     from agents.coordinator import coordinator_node
     from agents.critic import critic_node
     from agents.delivery import delivery_node
     from agents.grounding_check import grounding_check_node
     from agents.supervisor import supervisor_node
+    from agents.worker import worker_node
     from graph.boundary_compressor import boundary_compressor_node
-    from graph.router import budget_gate_node, route_after_critic
+    from graph.router import budget_gate_node, dispatch_workers, route_after_critic

     builder.add_node("supervisor", supervisor_node)  # type: ignore[call-overload]
+    builder.add_node("worker", worker_node)  # type: ignore[arg-type]
     builder.add_node("boundary_compressor", boundary_compressor_node)

 # ── Edges ──────────────────────────────────────────────────────────────────

     builder.add_edge(START, "supervisor")
-    builder.add_edge("supervisor", "boundary_compressor")
+    builder.add_conditional_edges("supervisor", dispatch_workers, ["worker"])
+    builder.add_edge("worker", "boundary_compressor")
     builder.add_edge("boundary_compressor", "coordinator")
     ...
-    # TODO: Phase 2 — replace supervisor→boundary_compressor with Send() fan-out:
-    #   builder.add_conditional_edges("supervisor", dispatch_workers, ["worker"])
-    #   builder.add_edge("worker", "boundary_compressor")   # fan-in via operator.add
```

---

## Test Cases (10)

| # | Test | Assertion |
|---|---|---|
| 1 | `test_dispatch_workers_returns_list_of_send` | `all(isinstance(s, Send) for s in result)` |
| 2 | `test_dispatch_workers_count_matches_sub_queries` | `len(result) == len(sub_queries)` |
| 3 | `test_dispatch_workers_target_node_is_worker` | `all(s.node == "worker" for s in result)` |
| 4 | `test_worker_ids_are_sequential` | worker_ids in payloads are `["worker-0", "worker-1", "worker-2"]` |
| 5 | `test_sub_queries_preserved_in_payload` | Each `Send.arg["sub_query"]` matches `sub_queries[i]` |
| 6 | `test_source_types_from_routing` | `Send.arg["source_type"]` matches `source_routing[sub_query]` |
| 7 | `test_missing_routing_defaults_to_web` | Sub-query absent from routing dict → `source_type == "web"` |
| 8 | `test_empty_sub_queries_raises_value_error` | `ValueError` when `sub_queries == []` |
| 9 | `test_three_sub_queries_creates_three_sends` | 3 queries → 3 Sends |
| 10 | `test_five_sub_queries_creates_five_sends` | 5 queries → 5 Sends |

---

## Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| Empty `sub_queries` | Supervisor validation failure | `ValueError` with clear message |
| Sub-query not in `source_routing` | Supervisor bug | Default to `"web"` |
| Worker exceeds rate limit | Too many concurrent Exa calls | Existing tenacity retry in `worker.py` |
| One worker fails | Exception propagates | `worker.py` uses `asyncio.gather(return_exceptions=True)` internally |
| `Send` import missing | LangGraph version | `from langgraph.types import Send` (LangGraph 1.0+) |
| Type error on worker node | Wrong state type passed | Worker expects `WorkerInput` — Send() carries exactly that |

---

## Token Cost Impact

`dispatch_workers` is pure Python — zero LLM calls, zero token cost. Worker token costs are unchanged (already implemented in `worker.py`, just not wired).

---

## Architect Notes

1. **`dispatch_workers` is not async** — it's a plain sync function. LangGraph conditional edge functions are synchronous.
2. **`["worker"]` not `{"worker": "worker"}`** — when returning `list[Send]`, the third arg to `add_conditional_edges` is a `list[str]` of reachable node names, NOT a dict path map.
3. **No `runtime` param on `dispatch_workers`** — routing functions only receive state.
4. **mypy `--strict`**: `Send` from `langgraph.types` — verify no `# type: ignore` needed.
5. **`source_routing.get(sub_query, "web")`** — `TypedDict` `dict` access returns `SourceType | None`; `.get()` with default makes mypy happy without a cast.

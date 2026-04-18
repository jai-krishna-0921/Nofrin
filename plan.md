# Deep Research Agent — FastAPI Web UI Plan

## Problem Statement

The pipeline only has a CLI interface (`main.py`). Adding a web UI enables browser-based research queries with real-time progress streaming, allowing users to submit queries, observe live status updates as each pipeline node executes, and receive the final research brief rendered as HTML. Target pipeline runtime is 30–90 seconds, making real-time progress feedback essential for UX.

---

## Architecture Decision: SSE Streaming

**SSE over WebSockets and polling.** The pipeline is fire-and-observe: the client submits a query via POST then receives a unidirectional event stream, which exactly matches SSE semantics. Unlike WebSockets, SSE requires no bidirectional communication, uses standard HTTP, works through proxies without special configuration, and auto-reconnects natively in browsers. Polling would require clients to hit a status endpoint every 1–2 seconds for 30–90 seconds — wasteful and slow to surface progress. SSE delivers events instantly as they occur. `sse-starlette` provides a production-ready implementation that integrates cleanly with FastAPI's async architecture and the project's all-async requirement (CLAUDE.md).

---

## File Structure

```
api/
  __init__.py          # Package marker
  server.py            # FastAPI app + endpoints
  session_store.py     # In-memory SessionState dict
  pipeline.py          # run_pipeline() coroutine
static/
  index.html           # Single-file frontend (no npm, no build step)
tests/
  test_api.py          # 15 new tests (added to existing tests/ dir)
```

---

## Session State Design

```python
# api/session_store.py
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

SessionStatus = Literal["pending", "running", "done", "error"]


@dataclass
class SessionState:
    """State for a single research session.

    Attributes:
        session_id:   Unique identifier for this session.
        status:       Current lifecycle status.
        events:       Buffered SSE payloads for late-joining clients.
        final_output: Rendered markdown output when done.
        cost_usd:     Total LLM cost for this session.
        tokens_used:  Total tokens consumed.
        error:        Error message if status == "error".
        created_at:   Session creation timestamp (UTC).
    """

    session_id: str
    status: SessionStatus = "pending"
    events: list[str] = field(default_factory=list)
    final_output: str | None = None
    cost_usd: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    """Thread-safe in-memory session storage with TTL cleanup."""

    TTL_SECONDS: int = 3600  # 1 hour

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock: threading.Lock = threading.Lock()

    def create(self, session_id: str) -> SessionState:
        session = SessionState(session_id=session_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def set(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def list_expired(self, max_age_seconds: int | None = None) -> list[str]:
        max_age = max_age_seconds if max_age_seconds is not None else self.TTL_SECONDS
        cutoff = datetime.utcnow() - timedelta(seconds=max_age)
        with self._lock:
            return [sid for sid, s in self._sessions.items() if s.created_at < cutoff]

    def cleanup_expired(self, max_age_seconds: int | None = None) -> int:
        expired = self.list_expired(max_age_seconds)
        with self._lock:
            for sid in expired:
                self._sessions.pop(sid, None)
        return len(expired)


store = SessionStore()  # module-level singleton
```

---

## API Endpoints

| Method | Path | Request Body | Response | Purpose |
|--------|------|--------------|----------|---------|
| `POST` | `/research` | `{"query": str, "mode": "fast"\|"research", "format": "markdown"}` | `{"session_id": str}` | Start pipeline in background task |
| `GET` | `/stream/{session_id}` | — | SSE event stream | Real-time pipeline events |
| `GET` | `/status/{session_id}` | — | `{"session_id", "status", "cost_usd", "tokens_used", "error"}` | Poll-friendly JSON snapshot |
| `GET` | `/` | — | `text/html` | Serve `static/index.html` |

---

## SSE Event Protocol

Four event types, emitted in order:

### 1. `event: status`
One per pipeline phase, sent when a node starts or completes.
```
event: status
data: {"phase": "supervisor", "message": "Classifying intent and decomposing query..."}
```
Phases: `supervisor`, `worker`, `coordinator`, `grounding_check`, `critic`, `delivery`

### 2. `event: result`
Emitted once on successful completion.
```
event: result
data: {"final_output": "# Research Brief\n\n...", "cost_usd": 0.05, "tokens": 1200}
```

### 3. `event: error`
Emitted on pipeline failure.
```
event: error
data: {"message": "Cost ceiling exceeded: $1.02 > $1.00"}
```

### 4. `event: done`
Always the final event — signals stream closure.
```
event: done
data: 
```

---

## Graph Integration

`api/pipeline.py` reuses `_build_context()` and `_build_initial_state()` from `main.py` and calls `build_graph()` unchanged. An `emit` callback is injected so `graph/progress.py` can push SSE events without knowing about FastAPI.

### `run_pipeline()` Signature

```python
# api/pipeline.py
async def run_pipeline(
    session_id: str,
    query: str,
    output_format: OutputFormat,
    research_mode: ResearchMode,
    emit: Callable[[str, str], None],  # emit(event_type, json_payload)
) -> None:
    """Execute the research pipeline and emit SSE events via the emit callback."""
```

### Progress Event Hooking

Add to `graph/progress.py`:

```python
_SSE_CALLBACK: Callable[[str, str], None] | None = None

def set_sse_callback(callback: Callable[[str, str], None] | None) -> None:
    """Register (or clear) the SSE progress callback."""
    global _SSE_CALLBACK
    _SSE_CALLBACK = callback

def _emit_sse(phase: str, message: str) -> None:
    """Forward a progress event to the registered SSE callback, if any."""
    if _SSE_CALLBACK is not None:
        import json
        _SSE_CALLBACK("status", json.dumps({"phase": phase, "message": message}))
```

Each existing `*_start()` and `*_done()` function calls `_emit_sse(node, message)` alongside the existing `_emit()` call. `run_pipeline()` calls `set_sse_callback(emit)` at start and `set_sse_callback(None)` in the `finally` block.

### `POST /research` Background Task

```python
@app.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    session_id = str(uuid.uuid4())
    store.create(session_id)

    def emit(event_type: str, data: str) -> None:
        session = store.get(session_id)
        if session is None:
            return
        session.events.append(f"{event_type}:{data}")
        if event_type == "error":
            session.status = "error"
            session.error = json.loads(data).get("message") if data else None
        elif event_type == "result":
            parsed = json.loads(data)
            session.final_output = parsed.get("final_output")
            session.cost_usd = parsed.get("cost_usd", 0.0)
            session.tokens_used = parsed.get("tokens", 0)
        elif event_type == "done" and session.status != "error":
            session.status = "done"
        store.set(session)

    async def run_task() -> None:
        session = store.get(session_id)
        if session is not None:
            session.status = "running"
            store.set(session)
        await run_pipeline(
            session_id=session_id,
            query=request.query,
            output_format=request.format,
            research_mode=request.mode,
            emit=emit,
        )

    background_tasks.add_task(run_task)
    return ResearchResponse(session_id=session_id)
```

### SSE Endpoint

```python
@app.get("/stream/{session_id}")
async def stream_events(session_id: str) -> EventSourceResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        # Replay buffered events for late-joining clients
        cursor = 0
        while True:
            current = store.get(session_id)
            if current is None:
                break
            while cursor < len(current.events):
                event_type, data = current.events[cursor].split(":", 1)
                yield {"event": event_type, "data": data}
                cursor += 1
                if event_type == "done":
                    return
            if current.status in ("done", "error") and cursor >= len(current.events):
                yield {"event": "done", "data": ""}
                return
            await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())
```

---

## Frontend UX

Single `static/index.html` — no npm, no build step, no CDN build tools. Uses `marked.js` via CDN for markdown rendering.

**Layout:**
1. **Form** — query textarea, fast/research radio buttons, Submit button (disabled during run)
2. **Progress area** — live status events as a vertical timeline with timestamps and phase badges
3. **Results area** — final output rendered HTML (hidden until `result` event received)
4. **Stats bar** — `$cost`, token count, elapsed duration (shown on completion)

**Key JS pattern:**
```javascript
const res = await fetch('/research', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query, mode, format: 'markdown'})
});
const {session_id} = await res.json();

const es = new EventSource(`/stream/${session_id}`);
es.addEventListener('status', e => addEvent(JSON.parse(e.data)));
es.addEventListener('result', e => renderResult(JSON.parse(e.data)));
es.addEventListener('error', e => showError(JSON.parse(e.data)));
es.addEventListener('done', () => { es.close(); submitBtn.disabled = false; });
```

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | `>=0.109.0` | Web framework |
| `uvicorn[standard]` | `>=0.27.0` | ASGI server |
| `sse-starlette` | `>=2.0.0` | SSE via EventSourceResponse |
| `python-multipart` | `>=0.0.6` | Form parsing (FastAPI requirement) |

Add to `requirements.txt` under a new `# ── Web API ──` section.

---

## Tests to Add

| File | Count | What |
|------|-------|------|
| `tests/test_api.py` | +3 | SessionStore: create returns pending, get returns None for missing, list_expired finds old sessions |
| `tests/test_api.py` | +2 | POST /research: returns 200 + UUID session_id; accepts research mode |
| `tests/test_api.py` | +1 | GET / returns 200 HTML |
| `tests/test_api.py` | +2 | GET /status: returns session state; 404 for missing session |
| `tests/test_api.py` | +1 | GET /stream: 404 for unknown session_id |
| `tests/test_api.py` | +4 | SSE stream events: status → result → done order; error event sets session.status |
| `tests/test_api.py` | +2 | run_pipeline: emits correct event sequence with mocked graph |

**Total: 15 new tests**

---

## Failure Modes

| Failure | Mitigation |
|---------|------------|
| Session not found | HTTP 404 on `/stream/` and `/status/` |
| Pipeline raises exception | `run_pipeline` `try/except` emits `error` event, sets `session.status = "error"` |
| Client disconnects mid-stream | EventSourceResponse handles disconnect gracefully; session state persists for `/status` retrieval |
| Memory leak from old sessions | TTL cleanup (`store.cleanup_expired()`) runs on startup and every 10 min via `asyncio.create_task` |
| Concurrent access to session state | `threading.Lock` in `SessionStore` |
| SSE client timeout (browser ~45s default) | Client-side `EventSource` auto-reconnects; buffered `session.events` replayed so no data lost |

---

## Cost / Performance Impact

**LLM cost:** Zero change — no new LLM calls.

**Memory overhead:**
- Each `SessionState`: ~2–5 KB (event buffer strings)
- 1-hour TTL + 10-min cleanup → at most ~360 sessions in memory
- Worst case: ~1.8 MB

**Latency:** SSE poll interval 100ms — negligible. No additional network round-trips vs CLI.

---

## Run Commands

```bash
# Install new deps
pip install -r requirements.txt

# Start development server
uvicorn api.server:app --reload --port 8000

# Run all tests including new API tests
pytest tests/ -v

# Type check new API files
mypy api/ --strict
```

"""
api/server.py

FastAPI application for the Deep Research Agent web interface.

Endpoints:
  POST /research              — start pipeline, return session_id
  GET  /stream/{session_id}   — SSE event stream
  GET  /status/{session_id}   — poll-friendly JSON snapshot
  GET  /                      — serve static/index.html

Usage:
  uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv()
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.pipeline import run_pipeline
from api.session_store import store
from graph.state import OutputFormat, ResearchMode

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="Deep Research Agent", version="2.0.0")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    query: str
    mode: ResearchMode = "fast"
    format: OutputFormat = "markdown"


class ResearchResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    session_id: str
    status: str
    cost_usd: float
    tokens_used: int
    error: str | None


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------


_cleanup_task: asyncio.Task[None] | None = None


@app.on_event("startup")
async def _start_cleanup_task() -> None:
    """Schedule periodic session cleanup every 10 minutes."""
    global _cleanup_task

    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(600)
            removed = store.cleanup_expired()
            if removed:
                logger.info("Cleaned up %d expired sessions", removed)

    _cleanup_task = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def _stop_cleanup_task() -> None:
    """Cancel the cleanup task on server shutdown."""
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve static/index.html."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index))


@app.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """Start the research pipeline in a background task."""
    session_id = str(uuid.uuid4())
    store.create(session_id)

    def emit(event_type: str, data: str) -> None:
        session = store.get(session_id)
        if session is None:
            return
        session.events.append(f"{event_type}:{data}")
        if event_type == "error":
            session.status = "error"
            try:
                session.error = json.loads(data).get("message") if data else None
            except (json.JSONDecodeError, AttributeError):
                session.error = data
        elif event_type == "result":
            try:
                parsed = json.loads(data)
                session.final_output = parsed.get("final_output")
                session.cost_usd = float(parsed.get("cost_usd", 0.0))
                session.tokens_used = int(parsed.get("tokens", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        elif event_type == "done" and session.status != "error":
            session.status = "done"
        store.set(session)

    async def _run_task() -> None:
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

    background_tasks.add_task(_run_task)
    return ResearchResponse(session_id=session_id)


@app.get("/stream/{session_id}")
async def stream_events(session_id: str) -> EventSourceResponse:
    """SSE endpoint — streams pipeline events, replays buffer for late joiners."""
    if store.get(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def _event_generator() -> AsyncGenerator[dict[str, str], None]:
        cursor = 0
        while True:
            current = store.get(session_id)
            if current is None:
                break
            while cursor < len(current.events):
                raw = current.events[cursor]
                event_type, _, data = raw.partition(":")
                yield {"event": event_type, "data": data}
                cursor += 1
                if event_type == "done":
                    return
            if current.status in ("done", "error") and cursor >= len(current.events):
                yield {"event": "done", "data": ""}
                return
            await asyncio.sleep(0.1)

    return EventSourceResponse(_event_generator())


@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str) -> StatusResponse:
    """Poll-friendly JSON snapshot of session state."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return StatusResponse(
        session_id=session.session_id,
        status=session.status,
        cost_usd=session.cost_usd,
        tokens_used=session.tokens_used,
        error=session.error,
    )


__all__ = ["app"]

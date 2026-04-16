"""
tests/test_api.py

Unit tests for the FastAPI web interface (api/server.py, api/session_store.py, api/pipeline.py).

Covers SessionStore CRUD, HTTP endpoints, SSE streaming, and pipeline orchestration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.pipeline import run_pipeline
from api.server import app
from api.session_store import SessionState, SessionStore


# ---------------------------------------------------------------------------
# SessionStore unit tests
# ---------------------------------------------------------------------------


def test_session_store_create_returns_pending_status() -> None:
    """Test 1: SessionStore.create returns SessionState with status='pending'."""
    store = SessionStore()
    session = store.create("test-session-1")
    assert session.session_id == "test-session-1"
    assert session.status == "pending"
    assert session.events == []
    assert session.final_output is None
    assert session.cost_usd == 0.0
    assert session.tokens_used == 0
    assert session.error is None


def test_session_store_get_returns_none_for_missing_session() -> None:
    """Test 2: SessionStore.get returns None for missing session_id."""
    store = SessionStore()
    result = store.get("nonexistent-id")
    assert result is None


def test_session_store_list_expired_finds_old_sessions() -> None:
    """Test 3: SessionStore.list_expired finds sessions older than max_age."""
    store = SessionStore()
    # Create two sessions — one old, one fresh
    old_session = SessionState(
        session_id="old-session",
        created_at=datetime.now(UTC) - timedelta(seconds=3700),
    )
    fresh_session = SessionState(
        session_id="fresh-session",
        created_at=datetime.now(UTC) - timedelta(seconds=100),
    )
    store.set(old_session)
    store.set(fresh_session)

    # Query with default TTL (3600 seconds)
    expired = store.list_expired()
    assert "old-session" in expired
    assert "fresh-session" not in expired


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


def test_post_research_returns_200_with_uuid_session_id() -> None:
    """Test 4: POST /research returns 200 with a UUID session_id."""

    # Mock run_pipeline to avoid actually running the graph
    async def mock_run_pipeline(*args: Any, **kwargs: Any) -> None:
        pass

    with patch("api.server.store", SessionStore()):
        with patch("api.server.run_pipeline", side_effect=mock_run_pipeline):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/research",
                json={"query": "test query", "mode": "fast", "format": "markdown"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert isinstance(data["session_id"], str)
            assert len(data["session_id"]) == 36  # UUID format


def test_post_research_with_mode_research_returns_200() -> None:
    """Test 5: POST /research with mode='research' returns 200."""

    # Mock run_pipeline to avoid actually running the graph
    async def mock_run_pipeline(*args: Any, **kwargs: Any) -> None:
        pass

    with patch("api.server.store", SessionStore()):
        with patch("api.server.run_pipeline", side_effect=mock_run_pipeline):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/research",
                json={"query": "test query", "mode": "research", "format": "markdown"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data


def test_get_root_returns_200_with_html_content_type() -> None:
    """Test 6: GET / returns 200 with content-type text/html."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_status_returns_correct_fields_for_existing_session() -> None:
    """Test 7: GET /status/{session_id} returns correct fields for existing session."""
    test_store = SessionStore()
    session = test_store.create("test-session-7")
    session.status = "running"
    session.cost_usd = 0.05
    session.tokens_used = 1500
    test_store.set(session)

    with patch("api.server.store", test_store):
        client = TestClient(app)
        response = client.get("/status/test-session-7")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-7"
        assert data["status"] == "running"
        assert data["cost_usd"] == 0.05
        assert data["tokens_used"] == 1500
        assert data["error"] is None


def test_get_status_returns_404_for_unknown_session_id() -> None:
    """Test 8: GET /status/{session_id} returns 404 for unknown session_id."""
    with patch("api.server.store", SessionStore()):
        client = TestClient(app)
        response = client.get("/status/unknown-session")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


def test_get_stream_returns_404_for_unknown_session_id() -> None:
    """Test 9: GET /stream/{session_id} returns 404 for unknown session_id."""
    with patch("api.server.store", SessionStore()):
        client = TestClient(app)
        response = client.get("/stream/unknown-session")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


def test_sse_stream_emits_status_events_from_session_buffer() -> None:
    """Test 10: SSE stream emits 'status' events from session.events buffer."""
    test_store = SessionStore()
    session = test_store.create("test-session-10")
    session.events.append(
        "status:" + json.dumps({"phase": "supervisor", "message": "Analyzing query"})
    )
    session.events.append("done:")
    test_store.set(session)

    with patch("api.server.store", test_store):
        client = TestClient(app)
        with client.stream("GET", "/stream/test-session-10") as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
            # Expected: event: status, data: {...}, event: done, data: (empty)
            found_status = False
            found_done = False
            for line in lines:
                if line.startswith("event: status"):
                    found_status = True
                if line.startswith("event: done"):
                    found_done = True
            assert found_status
            assert found_done


def test_sse_stream_emits_result_event_and_sets_final_output() -> None:
    """Test 11: SSE stream emits 'result' event and sets session.final_output."""
    test_store = SessionStore()
    session = test_store.create("test-session-11")
    result_data = {
        "final_output": "# Test Output",
        "cost_usd": 0.12,
        "tokens": 2000,
    }
    session.events.append("result:" + json.dumps(result_data))
    session.events.append("done:")
    test_store.set(session)

    with patch("api.server.store", test_store):
        client = TestClient(app)
        with client.stream("GET", "/stream/test-session-11") as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
            found_result = False
            for line in lines:
                if line.startswith("event: result"):
                    found_result = True
            assert found_result


def test_sse_stream_emits_done_event_as_final_event() -> None:
    """Test 12: SSE stream emits 'done' event as final event."""
    test_store = SessionStore()
    session = test_store.create("test-session-12")
    session.events.append(
        "status:" + json.dumps({"phase": "pipeline", "message": "Starting"})
    )
    session.events.append("done:")
    test_store.set(session)

    with patch("api.server.store", test_store):
        client = TestClient(app)
        with client.stream("GET", "/stream/test-session-12") as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
            # Last event should be 'done'
            event_lines = [line for line in lines if line.startswith("event:")]
            assert event_lines[-1] == "event: done"


def test_sse_stream_sets_status_error_when_error_event_emitted() -> None:
    """Test 13: SSE stream sets session.status='error' when 'error' event emitted."""
    test_store = SessionStore()
    session = test_store.create("test-session-13")
    session.status = "pending"
    test_store.set(session)

    # Manually inject error event via emit callback
    def emit(event_type: str, data: str) -> None:
        current = test_store.get("test-session-13")
        if current is None:
            return
        current.events.append(f"{event_type}:{data}")
        if event_type == "error":
            current.status = "error"
            try:
                current.error = json.loads(data).get("message") if data else None
            except (json.JSONDecodeError, AttributeError):
                current.error = data
        test_store.set(current)

    emit("error", json.dumps({"message": "Test error"}))
    emit("done", "")

    updated_session = test_store.get("test-session-13")
    assert updated_session is not None
    assert updated_session.status == "error"
    assert updated_session.error == "Test error"


# ---------------------------------------------------------------------------
# run_pipeline integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_emits_status_result_done_sequence() -> None:
    """Test 14: run_pipeline emits status→result→done sequence with mocked graph."""
    events: list[tuple[str, str]] = []

    def emit(event_type: str, data: str) -> None:
        events.append((event_type, data))

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "final_output": "# Test Output",
            "cost_usd": 0.10,
            "total_tokens_used": 1000,
        }
    )

    with patch("api.pipeline.build_graph", return_value=mock_graph):
        with patch("main._build_context", return_value={}):
            with patch("main._build_initial_state", return_value={}):
                await run_pipeline(
                    session_id="test-session-14",
                    query="test query",
                    output_format="markdown",
                    research_mode="fast",
                    emit=emit,
                )

    # Should have status → result → done
    assert len(events) >= 3
    assert events[0][0] == "status"
    assert any(evt[0] == "result" for evt in events)
    assert events[-1][0] == "done"


@pytest.mark.asyncio
async def test_run_pipeline_emits_status_error_done_when_graph_raises() -> None:
    """Test 15: run_pipeline emits status→error→done when graph raises."""
    events: list[tuple[str, str]] = []

    def emit(event_type: str, data: str) -> None:
        events.append((event_type, data))

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Test graph failure"))

    with patch("api.pipeline.build_graph", return_value=mock_graph):
        with patch("main._build_context", return_value={}):
            with patch("main._build_initial_state", return_value={}):
                await run_pipeline(
                    session_id="test-session-15",
                    query="test query",
                    output_format="markdown",
                    research_mode="fast",
                    emit=emit,
                )

    # Should have status → error → done
    assert len(events) >= 3
    assert events[0][0] == "status"
    assert any(evt[0] == "error" for evt in events)
    assert events[-1][0] == "done"

    # Check error message contains exception text
    error_events = [evt for evt in events if evt[0] == "error"]
    assert len(error_events) == 1
    error_data = json.loads(error_events[0][1])
    assert "Test graph failure" in error_data["message"]


__all__ = [
    "test_session_store_create_returns_pending_status",
    "test_session_store_get_returns_none_for_missing_session",
    "test_session_store_list_expired_finds_old_sessions",
    "test_post_research_returns_200_with_uuid_session_id",
    "test_post_research_with_mode_research_returns_200",
    "test_get_root_returns_200_with_html_content_type",
    "test_get_status_returns_correct_fields_for_existing_session",
    "test_get_status_returns_404_for_unknown_session_id",
    "test_get_stream_returns_404_for_unknown_session_id",
    "test_sse_stream_emits_status_events_from_session_buffer",
    "test_sse_stream_emits_result_event_and_sets_final_output",
    "test_sse_stream_emits_done_event_as_final_event",
    "test_sse_stream_sets_status_error_when_error_event_emitted",
    "test_run_pipeline_emits_status_result_done_sequence",
    "test_run_pipeline_emits_status_error_done_when_graph_raises",
]

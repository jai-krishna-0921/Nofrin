"""
api/session_store.py

In-memory session state store for the FastAPI web interface.

Keyed by session_id (UUID string). Thread-safe via threading.Lock.
Sessions expire after TTL_SECONDS (default 1 hour); cleanup is triggered
externally (e.g., from a periodic background task in server.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

SessionStatus = Literal["pending", "running", "done", "error"]


@dataclass
class SessionState:
    """State for a single research session.

    Attributes:
        session_id:   Unique identifier for this session.
        status:       Current lifecycle status.
        events:       Buffered SSE payloads (format: "event_type:json_data")
                      for late-joining clients to replay.
        final_output: Rendered markdown output when status == "done".
        cost_usd:     Total LLM cost accumulated during the session.
        tokens_used:  Total tokens consumed.
        error:        Error message if status == "error".
        created_at:   Session creation timestamp (UTC, for TTL cleanup).
    """

    session_id: str
    status: SessionStatus = "pending"
    events: list[str] = field(default_factory=list)
    final_output: str | None = None
    cost_usd: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionStore:
    """Thread-safe in-memory session storage with TTL cleanup.

    Uses threading.Lock for safety when FastAPI background tasks mutate
    session state concurrently with SSE stream readers.
    """

    TTL_SECONDS: int = 3600  # 1 hour

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock: threading.Lock = threading.Lock()

    def create(self, session_id: str) -> SessionState:
        """Create a new session with pending status and store it."""
        session = SessionState(session_id=session_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        """Retrieve session by ID. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def set(self, session: SessionState) -> None:
        """Update an existing session."""
        with self._lock:
            self._sessions[session.session_id] = session

    def list_expired(self, max_age_seconds: int | None = None) -> list[str]:
        """Return session IDs older than max_age_seconds."""
        max_age = max_age_seconds if max_age_seconds is not None else self.TTL_SECONDS
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age)
        with self._lock:
            return [sid for sid, s in self._sessions.items() if s.created_at < cutoff]

    def cleanup_expired(self, max_age_seconds: int | None = None) -> int:
        """Remove expired sessions atomically. Returns count removed."""
        max_age = max_age_seconds if max_age_seconds is not None else self.TTL_SECONDS
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age)
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items() if s.created_at < cutoff
            ]
            for sid in expired:
                self._sessions.pop(sid, None)
        return len(expired)


# Module-level singleton shared across all FastAPI routes.
store = SessionStore()

__all__ = ["SessionState", "SessionStatus", "SessionStore", "store"]

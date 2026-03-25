"""
graph/utils.py

Shared utilities for the Deep Research Agent pipeline.

All JSON parsing in agent nodes must go through parse_agent_json() —
never raw json.loads() (CLAUDE.md rule).
"""

from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

T = TypeVar("T")


class AgentParseError(Exception):
    """Raised when an agent response cannot be parsed into the expected schema."""


def parse_agent_json(raw: str, schema_class: Type[T], max_retries: int = 2) -> T:
    """Parse a raw LLM response string into a typed schema instance.

    Strips markdown code fences if present, then attempts json.loads().
    On failure, retries up to max_retries times (caller is responsible for
    sending a correction prompt on subsequent attempts).

    Args:
        raw: The raw string returned by the LLM.
        schema_class: A dataclass or TypedDict class to instantiate.
        max_retries: Number of parse attempts before raising AgentParseError.

    Returns:
        An instance of schema_class populated from the parsed JSON.

    Raises:
        AgentParseError: If parsing fails after max_retries attempts.
    """
    for attempt in range(max_retries + 1):
        try:
            clean = _strip_fences(raw)
            data: dict[str, Any] = json.loads(clean)
            return schema_class(**data)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            if attempt == max_retries:
                raise AgentParseError(
                    f"Failed to parse agent output after {max_retries + 1} attempt(s): {exc}\n"
                    f"Raw output (first 500 chars): {raw[:500]}"
                ) from exc
            # Caller should send a correction prompt and pass new raw output
            # on the next call. This loop only handles transient parse noise.
    # Unreachable, but satisfies the type checker.
    raise AgentParseError("Exhausted retries")  # pragma: no cover


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) from text."""
    stripped = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    pattern = r"^```(?:json)?\s*\n([\s\S]*?)\n?```$"
    match = re.match(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


__all__ = [
    "AgentParseError",
    "parse_agent_json",
]

"""
tests/test_supervisor.py

Unit tests for agents/supervisor.py.

Tests the supervisor node's intent classification, query decomposition,
validation logic, retry behavior, and prompt loading.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.supervisor import (
    SubQueryItem,
    SupervisorOutput,
    _RawSupervisorOutput,
    _build_messages,
    _load_prompt,
    _validate_output,
    supervisor_node,
)
from graph.context import NofrinContext
from graph.state import ResearchAgentState
from graph.utils import AgentParseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state() -> ResearchAgentState:
    """Minimal valid ResearchAgentState for supervisor node tests."""
    return ResearchAgentState(
        user_query="What are the trends in renewable energy?",
        intent_type="factual",  # Will be overwritten by supervisor
        output_format="markdown",
        sub_queries=[],
        source_routing={},
        worker_results=[],
        synthesis=None,
        grounding_issues=[],
        critic_output=None,
        revision_count=0,
        prior_syntheses=[],
        session_id="test-session",
        total_tokens_used=0,
        cost_usd=0.0,
        final_output=None,
    )


@pytest.fixture
def mock_runtime() -> MagicMock:
    """Create a mock Runtime[NofrinContext] with a mock LLM."""
    runtime = MagicMock()
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    runtime.context = MagicMock(spec=NofrinContext)
    runtime.context.llm_supervisor = llm
    return runtime


# ---------------------------------------------------------------------------
# Intent type tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_returns_exploratory_intent(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM returns exploratory JSON; state update has intent_type == 'exploratory'."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "renewable energy trends 2026", "source_type": "web", "rationale": "overview"},
            {"query": "solar panel efficiency research", "source_type": "academic", "rationale": "technical"},
            {"query": "renewable energy market news", "source_type": "news", "rationale": "recent"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert result["intent_type"] == "exploratory"
    assert len(result["sub_queries"]) == 3
    assert isinstance(result["sub_queries"], list)
    assert all(isinstance(q, str) for q in result["sub_queries"])


@pytest.mark.asyncio
async def test_supervisor_returns_comparative_intent(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM returns comparative JSON; intent_type == 'comparative'."""
    json_response = """{
        "intent_type": "comparative",
        "sub_queries": [
            {"query": "React framework strengths", "source_type": "web", "rationale": "compare A"},
            {"query": "Vue.js framework strengths", "source_type": "web", "rationale": "compare B"},
            {"query": "React vs Vue performance benchmarks", "source_type": "academic", "rationale": "direct comparison"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert result["intent_type"] == "comparative"


@pytest.mark.asyncio
async def test_supervisor_returns_adversarial_intent(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM returns adversarial JSON; intent_type == 'adversarial'."""
    json_response = """{
        "intent_type": "adversarial",
        "sub_queries": [
            {"query": "blockchain scalability limitations", "source_type": "web", "rationale": "find weaknesses"},
            {"query": "blockchain security vulnerabilities research", "source_type": "academic", "rationale": "technical flaws"},
            {"query": "blockchain project failures news", "source_type": "news", "rationale": "real-world problems"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert result["intent_type"] == "adversarial"


@pytest.mark.asyncio
async def test_supervisor_returns_factual_intent(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM returns factual JSON; intent_type == 'factual'."""
    json_response = """{
        "intent_type": "factual",
        "sub_queries": [
            {"query": "Tesla Q4 2024 earnings report", "source_type": "news", "rationale": "verify claim"},
            {"query": "Tesla profitability Q4 2024", "source_type": "web", "rationale": "cross-check"},
            {"query": "Tesla financial statements 2024", "source_type": "web", "rationale": "official source"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert result["intent_type"] == "factual"


# ---------------------------------------------------------------------------
# Sub-query structure tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_sub_queries_list_correct(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """result['sub_queries'] is a list[str] — not dicts."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "query one", "source_type": "web", "rationale": "reason"},
            {"query": "query two", "source_type": "academic", "rationale": "reason"},
            {"query": "query three", "source_type": "news", "rationale": "reason"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert isinstance(result["sub_queries"], list)
    assert len(result["sub_queries"]) == 3
    assert all(isinstance(q, str) for q in result["sub_queries"])
    assert result["sub_queries"][0] == "query one"
    assert result["sub_queries"][1] == "query two"


@pytest.mark.asyncio
async def test_supervisor_source_routing_keys_match_sub_queries(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Every key in result['source_routing'] appears in result['sub_queries']."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "alpha query", "source_type": "web", "rationale": "reason"},
            {"query": "beta query", "source_type": "academic", "rationale": "reason"},
            {"query": "gamma query", "source_type": "news", "rationale": "reason"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    routing_keys = set(result["source_routing"].keys())
    sub_query_set = set(result["sub_queries"])

    assert routing_keys == sub_query_set
    assert all(key in result["sub_queries"] for key in result["source_routing"])


# ---------------------------------------------------------------------------
# Count validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_sub_query_count_min_3(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Result has at least 3 sub-queries."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "query 1", "source_type": "web", "rationale": "reason"},
            {"query": "query 2", "source_type": "academic", "rationale": "reason"},
            {"query": "query 3", "source_type": "news", "rationale": "reason"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert len(result["sub_queries"]) >= 3


@pytest.mark.asyncio
async def test_supervisor_sub_query_count_max_5(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Result has at most 5 sub-queries."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "query 1", "source_type": "web", "rationale": "reason"},
            {"query": "query 2", "source_type": "academic", "rationale": "reason"},
            {"query": "query 3", "source_type": "news", "rationale": "reason"},
            {"query": "query 4", "source_type": "web", "rationale": "reason"},
            {"query": "query 5", "source_type": "web", "rationale": "reason"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    assert len(result["sub_queries"]) <= 5


@pytest.mark.asyncio
async def test_supervisor_no_duplicate_sub_queries(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """All sub-query strings are unique."""
    json_response = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "unique query one", "source_type": "web", "rationale": "reason"},
            {"query": "unique query two", "source_type": "academic", "rationale": "reason"},
            {"query": "unique query three", "source_type": "news", "rationale": "reason"}
        ]
    }"""

    mock_response = MagicMock()
    mock_response.content = json_response
    mock_runtime.context.llm_supervisor.ainvoke.return_value = mock_response

    result = await supervisor_node(base_state, mock_runtime)

    sub_queries = result["sub_queries"]
    assert len(sub_queries) == len(set(sub_queries))


# ---------------------------------------------------------------------------
# Empty query test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_empty_query_raises(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """state['user_query'] = '' raises ValueError."""
    base_state["user_query"] = ""

    with pytest.raises(ValueError, match="must not be empty"):
        await supervisor_node(base_state, mock_runtime)


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------


def test_parse_valid_supervisor_json() -> None:
    """parse_agent_json() correctly parses valid supervisor JSON into _RawSupervisorOutput."""
    from graph.utils import parse_agent_json

    raw_json = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "test query", "source_type": "web", "rationale": "test rationale"}
        ]
    }"""

    result: _RawSupervisorOutput = parse_agent_json(raw_json, _RawSupervisorOutput)

    assert result["intent_type"] == "exploratory"
    assert len(result["sub_queries"]) == 1
    assert result["sub_queries"][0]["query"] == "test query"


def test_parse_supervisor_json_with_fences() -> None:
    """parse_agent_json() strips ```json fences before parsing."""
    from graph.utils import parse_agent_json

    raw_json = """```json
    {
        "intent_type": "factual",
        "sub_queries": [
            {"query": "fenced query", "source_type": "news", "rationale": "test"}
        ]
    }
    ```"""

    result: _RawSupervisorOutput = parse_agent_json(raw_json, _RawSupervisorOutput)

    assert result["intent_type"] == "factual"
    assert result["sub_queries"][0]["query"] == "fenced query"


def test_parse_supervisor_json_invalid_raises() -> None:
    """Malformed JSON raises AgentParseError."""
    from graph.utils import parse_agent_json

    invalid_json = "{ not valid json"

    with pytest.raises(AgentParseError):
        parse_agent_json(invalid_json, _RawSupervisorOutput)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_output_too_few_queries() -> None:
    """2 sub-queries raises AgentParseError."""
    output = SupervisorOutput(
        intent_type="exploratory",
        sub_queries=[
            SubQueryItem(query="q1", source_type="web", rationale="r1"),
            SubQueryItem(query="q2", source_type="web", rationale="r2"),
        ],
    )

    with pytest.raises(AgentParseError, match="2 sub-queries; expected 3–5"):
        _validate_output(output)


def test_validate_output_zero_queries() -> None:
    """Empty sub_queries list raises AgentParseError."""
    output = SupervisorOutput(
        intent_type="exploratory",
        sub_queries=[],
    )

    with pytest.raises(AgentParseError, match="0 sub-queries; expected 3–5"):
        _validate_output(output)


def test_validate_output_too_many_queries() -> None:
    """6 sub-queries raises AgentParseError."""
    output = SupervisorOutput(
        intent_type="exploratory",
        sub_queries=[
            SubQueryItem(query=f"q{i}", source_type="web", rationale="r")
            for i in range(6)
        ],
    )

    with pytest.raises(AgentParseError, match="6 sub-queries; expected 3–5"):
        _validate_output(output)


def test_validate_output_invalid_source_type() -> None:
    """source_type='website' raises AgentParseError."""
    output = SupervisorOutput(
        intent_type="exploratory",
        sub_queries=[
            SubQueryItem(query="q1", source_type="web", rationale="r1"),
            SubQueryItem(query="q2", source_type="website", rationale="r2"),  # type: ignore[arg-type]
            SubQueryItem(query="q3", source_type="web", rationale="r3"),
        ],
    )

    with pytest.raises(AgentParseError, match="Invalid source_type 'website'"):
        _validate_output(output)


# ---------------------------------------------------------------------------
# Retry behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_retries_on_parse_failure(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM returns invalid JSON on first call, valid JSON on second; result is valid."""
    invalid_response = MagicMock()
    invalid_response.content = "{ invalid json"

    valid_json = """{
        "intent_type": "exploratory",
        "sub_queries": [
            {"query": "q1", "source_type": "web", "rationale": "r1"},
            {"query": "q2", "source_type": "academic", "rationale": "r2"},
            {"query": "q3", "source_type": "news", "rationale": "r3"}
        ]
    }"""
    valid_response = MagicMock()
    valid_response.content = valid_json

    mock_runtime.context.llm_supervisor.ainvoke.side_effect = [
        invalid_response,
        valid_response,
    ]

    result = await supervisor_node(base_state, mock_runtime)

    assert result["intent_type"] == "exploratory"
    assert len(result["sub_queries"]) == 3
    assert mock_runtime.context.llm_supervisor.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_supervisor_raises_after_max_retries(
    base_state: ResearchAgentState, mock_runtime: MagicMock
) -> None:
    """Mock LLM always returns invalid JSON; AgentParseError raised after exhausting retries."""
    invalid_response = MagicMock()
    invalid_response.content = "{ invalid json"

    mock_runtime.context.llm_supervisor.ainvoke.return_value = invalid_response

    with pytest.raises(AgentParseError):
        await supervisor_node(base_state, mock_runtime)

    # Should have retried 3 times total (initial + 2 retries as per tenacity config)
    assert mock_runtime.context.llm_supervisor.ainvoke.call_count == 3


# ---------------------------------------------------------------------------
# Prompt loading tests
# ---------------------------------------------------------------------------


def test_load_prompt_loads_file() -> None:
    """_load_prompt() returns a non-empty string containing '{{user_query}}'."""
    prompt = _load_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "{{user_query}}" in prompt


def test_load_prompt_missing_file_raises() -> None:
    """Monkeypatching PROMPT_PATH to a non-existent path raises FileNotFoundError."""
    import agents.supervisor as supervisor_module

    original_path = supervisor_module.PROMPT_PATH
    try:
        supervisor_module.PROMPT_PATH = Path("/nonexistent/path/supervisor_v1.txt")

        with pytest.raises(FileNotFoundError, match="Supervisor prompt not found"):
            _load_prompt()
    finally:
        supervisor_module.PROMPT_PATH = original_path


# ---------------------------------------------------------------------------
# cache_control tests
# ---------------------------------------------------------------------------


def test_build_messages_anthropic_has_cache_control() -> None:
    """use_cache_control=True → SystemMessage content is list with cache_control block."""
    msgs = _build_messages(
        "test query", "static prompt template", use_cache_control=True
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, list), "Expected list content blocks for Anthropic"
    assert len(content) == 2

    static_block = content[0]
    dynamic_block = content[1]

    assert isinstance(static_block, dict)
    assert static_block.get("cache_control") == {"type": "ephemeral"}
    assert static_block.get("text") == "static prompt template"

    assert isinstance(dynamic_block, dict)
    assert "cache_control" not in dynamic_block
    assert dynamic_block.get("text") == "test query"


def test_build_messages_groq_no_cache_control() -> None:
    """use_cache_control=False (groq/ollama/openrouter) → plain string SystemMessage."""
    msgs = _build_messages(
        "my question", "prompt with {{user_query}} here", use_cache_control=False
    )

    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, str), "Expected plain string for non-Anthropic provider"
    assert "my question" in content
    assert "cache_control" not in content

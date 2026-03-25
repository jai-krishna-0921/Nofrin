"""
tests/test_state.py

Unit tests for graph/state.py data types.

Tests all dataclasses, TypedDicts, and type aliases exported by the state module.
No mocking required — these are pure data types.
"""

import operator
from typing import get_type_hints

import pytest

from graph.state import (
    Evidence,
    Finding,
    CriticIssue,
    CriticSuggestion,
    WorkerResult,
    SynthesisOutput,
    CriticOutput,
    WorkerInput,
    ResearchAgentState,
    SourceType,
    IntentType,
    OutputFormat,
    IssueSeverity,
)


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


def test_evidence_construction():
    """Construct an Evidence with all required fields, assert field values are accessible."""
    evidence = Evidence(
        claim="Python was created by Guido van Rossum",
        supporting_chunks=["Guido van Rossum began working on Python in 1989"],
        source_url="https://example.com/python-history",
        source_title="The History of Python",
        published_date="2023-01-15",
        confidence=0.95,
        contradiction_score=0.0,
    )

    assert evidence.claim == "Python was created by Guido van Rossum"
    assert len(evidence.supporting_chunks) == 1
    assert evidence.supporting_chunks[0] == "Guido van Rossum began working on Python in 1989"
    assert evidence.source_url == "https://example.com/python-history"
    assert evidence.source_title == "The History of Python"
    assert evidence.published_date == "2023-01-15"
    assert evidence.confidence == 0.95
    assert evidence.contradiction_score == 0.0


def test_finding_construction():
    """Construct a Finding, verify heading/body/evidence_refs."""
    finding = Finding(
        heading="Python Origins",
        body="Python was developed in the late 1980s by Guido van Rossum as a successor to ABC.",
        evidence_refs=[
            "https://example.com/python-history",
            "https://example.com/abc-language",
        ],
    )

    assert finding.heading == "Python Origins"
    assert "Guido van Rossum" in finding.body
    assert len(finding.evidence_refs) == 2
    assert finding.evidence_refs[0] == "https://example.com/python-history"


def test_critic_issue_construction():
    """Construct a CriticIssue with severity='critical'."""
    issue = CriticIssue(
        issue_text="Unsupported claim without citation",
        quote_from_synthesis="Python is the most popular language in the world",
        severity="critical",
    )

    assert issue.issue_text == "Unsupported claim without citation"
    assert "Python is the most popular language" in issue.quote_from_synthesis
    assert issue.severity == "critical"


def test_critic_suggestion_construction():
    """Construct a CriticSuggestion."""
    suggestion = CriticSuggestion(
        action="Add citation from TIOBE index or Stack Overflow survey",
        target_section="Executive Summary",
        new_evidence_needed=True,
    )

    assert "TIOBE" in suggestion.action
    assert suggestion.target_section == "Executive Summary"
    assert suggestion.new_evidence_needed is True


def test_worker_result_construction():
    """Construct a WorkerResult containing two Evidence items."""
    evidence1 = Evidence(
        claim="Python 3.0 was released in 2008",
        supporting_chunks=["Python 3.0 was released on December 3, 2008"],
        source_url="https://example.com/py3-release",
        source_title="Python 3.0 Release Notes",
        published_date="2008-12-03",
        confidence=0.99,
        contradiction_score=0.0,
    )
    evidence2 = Evidence(
        claim="Python 2 reached end-of-life in 2020",
        supporting_chunks=["Python 2.7 support ended January 1, 2020"],
        source_url="https://example.com/py2-eol",
        source_title="Python 2 EOL Announcement",
        published_date="2020-01-01",
        confidence=0.98,
        contradiction_score=0.0,
    )

    worker_result = WorkerResult(
        worker_id="worker_001",
        sub_query="When was Python 3 released?",
        source_type="web",
        evidence_items=[evidence1, evidence2],
        raw_search_results=[
            {"url": "https://example.com/py3-release", "rank": 1},
            {"url": "https://example.com/py2-eol", "rank": 2},
        ],
        tokens_used=1500,
    )

    assert worker_result.worker_id == "worker_001"
    assert worker_result.sub_query == "When was Python 3 released?"
    assert worker_result.source_type == "web"
    assert len(worker_result.evidence_items) == 2
    assert worker_result.evidence_items[0].claim == "Python 3.0 was released in 2008"
    assert worker_result.evidence_items[1].claim == "Python 2 reached end-of-life in 2020"
    assert len(worker_result.raw_search_results) == 2
    assert worker_result.tokens_used == 1500


def test_synthesis_output_construction():
    """Construct a SynthesisOutput with findings, risks, gaps, citations.
    
    Verify synthesis_version and prior_attempt_summary=None.
    """
    finding = Finding(
        heading="Python Release Timeline",
        body="Python 3.0 was released in 2008, marking a major milestone.",
        evidence_refs=["https://example.com/py3-release"],
    )

    evidence = Evidence(
        claim="Python 3.0 was released in 2008",
        supporting_chunks=["Python 3.0 was released on December 3, 2008"],
        source_url="https://example.com/py3-release",
        source_title="Python 3.0 Release Notes",
        published_date="2008-12-03",
        confidence=0.99,
        contradiction_score=0.0,
    )

    synthesis = SynthesisOutput(
        topic="Python Programming Language History",
        executive_summary="This research brief covers the major milestones in Python history.",
        findings=[finding],
        risks=["Information limited to publicly available sources"],
        gaps=["Detailed internal development decisions not publicly documented"],
        citations=[evidence],
        synthesis_version=1,
        prior_attempt_summary=None,
    )

    assert synthesis.topic == "Python Programming Language History"
    assert "major milestones" in synthesis.executive_summary
    assert len(synthesis.findings) == 1
    assert synthesis.findings[0].heading == "Python Release Timeline"
    assert len(synthesis.risks) == 1
    assert len(synthesis.gaps) == 1
    assert len(synthesis.citations) == 1
    assert synthesis.synthesis_version == 1
    assert synthesis.prior_attempt_summary is None


def test_critic_output_passed_true():
    """Construct CriticOutput with final_quality_score=4.5; assert passed is True."""
    critic = CriticOutput(
        factuality_score=4.8,
        citation_alignment_score=4.5,
        reasoning_score=4.3,
        completeness_score=4.2,
        bias_score=4.6,
        final_quality_score=4.5,
        issues=[],
        suggestions=[],
        passed=False,  # Should be overridden by __post_init__
    )

    assert critic.final_quality_score == 4.5
    assert critic.passed is True  # __post_init__ sets this based on score >= 4.0


def test_critic_output_passed_false():
    """Construct CriticOutput with final_quality_score=3.9; assert passed is False."""
    critic = CriticOutput(
        factuality_score=3.8,
        citation_alignment_score=4.0,
        reasoning_score=3.9,
        completeness_score=3.7,
        bias_score=4.1,
        final_quality_score=3.9,
        issues=[
            CriticIssue(
                issue_text="Some claims lack citations",
                quote_from_synthesis="Python is widely used",
                severity="major",
            )
        ],
        suggestions=[
            CriticSuggestion(
                action="Add usage statistics",
                target_section="Introduction",
                new_evidence_needed=True,
            )
        ],
        passed=True,  # Should be overridden to False by __post_init__
    )

    assert critic.final_quality_score == 3.9
    assert critic.passed is False  # __post_init__ sets this based on score < 4.0


def test_critic_output_passed_boundary():
    """Score=4.0 exactly; assert passed is True."""
    critic = CriticOutput(
        factuality_score=4.0,
        citation_alignment_score=4.0,
        reasoning_score=4.0,
        completeness_score=4.0,
        bias_score=4.0,
        final_quality_score=4.0,
        issues=[],
        suggestions=[],
        passed=False,  # Should be overridden to True
    )

    assert critic.final_quality_score == 4.0
    assert critic.passed is True  # Exactly 4.0 should pass


def test_critic_output_passed_overrides_llm_value():
    """Construct with final_quality_score=4.5 but manually set passed=False.
    
    Assert the resulting object has passed=True (demonstrating __post_init__ override).
    """
    # This test verifies that __post_init__ overrides whatever value is passed
    critic = CriticOutput(
        factuality_score=4.5,
        citation_alignment_score=4.6,
        reasoning_score=4.4,
        completeness_score=4.7,
        bias_score=4.3,
        final_quality_score=4.5,
        issues=[],
        suggestions=[],
        passed=False,  # LLM might incorrectly return False
    )

    # __post_init__ must override to True because score >= 4.0
    assert critic.passed is True


# ---------------------------------------------------------------------------
# Reducer test (most important for LangGraph Send pattern)
# ---------------------------------------------------------------------------


def test_worker_results_reducer():
    """Simulate the LangGraph operator.add reducer for worker_results.
    
    This confirms the reducer logic is correct for the Send fan-out pattern.
    """
    evidence1 = Evidence(
        claim="First claim",
        supporting_chunks=["chunk1"],
        source_url="https://example.com/1",
        source_title="Source 1",
        published_date=None,
        confidence=0.9,
        contradiction_score=0.0,
    )

    evidence2 = Evidence(
        claim="Second claim",
        supporting_chunks=["chunk2"],
        source_url="https://example.com/2",
        source_title="Source 2",
        published_date=None,
        confidence=0.85,
        contradiction_score=0.1,
    )

    worker1 = WorkerResult(
        worker_id="worker_001",
        sub_query="query1",
        source_type="web",
        evidence_items=[evidence1],
        raw_search_results=[],
        tokens_used=500,
    )

    worker2 = WorkerResult(
        worker_id="worker_002",
        sub_query="query2",
        source_type="academic",
        evidence_items=[evidence2],
        raw_search_results=[],
        tokens_used=600,
    )

    # Simulate what LangGraph does with operator.add reducer
    list1 = [worker1]
    list2 = [worker2]
    merged = operator.add(list1, list2)

    assert len(merged) == 2
    assert merged[0].worker_id == "worker_001"
    assert merged[1].worker_id == "worker_002"
    assert merged[0].source_type == "web"
    assert merged[1].source_type == "academic"


# ---------------------------------------------------------------------------
# TypedDict tests
# ---------------------------------------------------------------------------


def test_research_agent_state_is_typeddict():
    """Verify ResearchAgentState is a TypedDict subclass using get_type_hints."""
    # TypedDict creates a dict-like class at runtime, not a real class
    # We can verify the structure by checking if we can instantiate it as a dict
    state: ResearchAgentState = {
        "user_query": "test query",
        "intent_type": "factual",
        "output_format": "markdown",
        "sub_queries": [],
        "source_routing": {},
        "worker_results": [],
        "synthesis": None,
        "grounding_issues": [],
        "critic_output": None,
        "revision_count": 0,
        "prior_syntheses": [],
        "session_id": "test-session",
        "total_tokens_used": 0,
        "cost_usd": 0.0,
        "final_output": None,
    }

    # Verify it's a dict
    assert isinstance(state, dict)
    
    # Verify key fields are present
    assert state["user_query"] == "test query"
    assert state["intent_type"] == "factual"
    assert state["output_format"] == "markdown"
    assert state["revision_count"] == 0
    
    # Verify we can get type hints
    hints = get_type_hints(ResearchAgentState)
    assert "user_query" in hints
    assert "worker_results" in hints
    assert "synthesis" in hints


def test_worker_input_typeddict():
    """Construct a WorkerInput dict with all required keys, verify values."""
    worker_input: WorkerInput = {
        "worker_id": "worker_003",
        "sub_query": "What are the key features of Python 3.11?",
        "source_type": "web",
    }

    # Verify it's a dict
    assert isinstance(worker_input, dict)
    
    # Verify all required keys are present and accessible
    assert worker_input["worker_id"] == "worker_003"
    assert worker_input["sub_query"] == "What are the key features of Python 3.11?"
    assert worker_input["source_type"] == "web"
    
    # Verify we can get type hints
    hints = get_type_hints(WorkerInput)
    assert "worker_id" in hints
    assert "sub_query" in hints
    assert "source_type" in hints


# ---------------------------------------------------------------------------
# Type alias tests (basic verification)
# ---------------------------------------------------------------------------


def test_source_type_literals():
    """Verify SourceType literal values are valid."""
    valid_sources: list[SourceType] = ["web", "academic", "news"]
    
    # These should all be valid SourceType values
    for source in valid_sources:
        worker_input: WorkerInput = {
            "worker_id": "test",
            "sub_query": "test",
            "source_type": source,
        }
        assert worker_input["source_type"] == source


def test_intent_type_literals():
    """Verify IntentType literal values are valid."""
    valid_intents: list[IntentType] = ["exploratory", "comparative", "adversarial", "factual"]
    
    # These should all be valid IntentType values
    for intent in valid_intents:
        state: dict = {"intent_type": intent}
        assert state["intent_type"] == intent


def test_output_format_literals():
    """Verify OutputFormat literal values are valid."""
    valid_formats: list[OutputFormat] = ["markdown", "docx", "pdf", "pptx"]
    
    # These should all be valid OutputFormat values
    for fmt in valid_formats:
        state: dict = {"output_format": fmt}
        assert state["output_format"] == fmt


def test_issue_severity_literals():
    """Verify IssueSeverity literal values are valid."""
    valid_severities: list[IssueSeverity] = ["critical", "major", "minor"]
    
    # These should all be valid IssueSeverity values
    for severity in valid_severities:
        issue = CriticIssue(
            issue_text="test",
            quote_from_synthesis="test",
            severity=severity,
        )
        assert issue.severity == severity

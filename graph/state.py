"""
graph/state.py

Typed state schema for the Deep Research Agent LangGraph pipeline.

All agent nodes read from and write into ResearchAgentState.
The `worker_results` field uses an operator.add reducer so that
parallel workers dispatched via Send() accumulate results rather
than overwriting each other.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SourceType = Literal["web", "academic", "news"]
IntentType = Literal["exploratory", "comparative", "adversarial", "factual"]
OutputFormat = Literal["markdown", "docx", "pdf", "pptx"]
IssueSeverity = Literal["critical", "major", "minor"]

# ---------------------------------------------------------------------------
# Supporting dataclasses — in dependency order
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Atomic sourced claim extracted by a worker node."""

    claim: str
    supporting_chunks: list[str]
    source_url: str
    source_title: str
    published_date: Optional[str]
    confidence: float  # 0.0–1.0
    contradiction_score: float  # 0.0 = no contradiction, 1.0 = directly contradicted


@dataclass
class Finding:
    """A single finding in the synthesis brief.

    Replaces bare dict in SynthesisOutput.findings to satisfy the
    no-bare-dict rule (CLAUDE.md).
    """

    heading: str
    body: str
    evidence_refs: list[str]  # list of source_url strings


@dataclass
class CriticIssue:
    """A specific problem identified by the critic node.

    Replaces bare dict in CriticOutput.issues (CLAUDE.md no-bare-dict rule).
    """

    issue_text: str
    quote_from_synthesis: str
    severity: IssueSeverity


@dataclass
class CriticSuggestion:
    """An actionable suggestion from the critic node.

    Replaces bare dict in CriticOutput.suggestions (CLAUDE.md no-bare-dict rule).
    """

    action: str
    target_section: str
    new_evidence_needed: bool


@dataclass
class WorkerResult:
    """Full output of one parallel research worker."""

    worker_id: str
    sub_query: str
    source_type: SourceType
    evidence_items: list[Evidence]
    # dict[str, object] instead of dict[str, Any] — no Any per CLAUDE.md.
    # These are raw external API payloads; object is the correct strict type.
    raw_search_results: list[dict[str, object]]
    tokens_used: int


@dataclass
class SynthesisOutput:
    """Coordinator node's structured research brief."""

    topic: str
    executive_summary: str
    findings: list[Finding]
    risks: list[str]
    gaps: list[str]  # explicitly acknowledged unknowns
    citations: list[Evidence]
    synthesis_version: int  # increments on each coordinator revision
    prior_attempt_summary: Optional[str]  # set on revision pass only


@dataclass
class CriticOutput:
    """Adversarial review output from the critic node.

    `passed` is always derived from final_quality_score in __post_init__
    so the field is authoritative regardless of what the LLM returned.
    """

    factuality_score: float  # weight: 0.30
    citation_alignment_score: float  # weight: 0.25
    reasoning_score: float  # weight: 0.20
    completeness_score: float  # weight: 0.15
    bias_score: float  # weight: 0.10
    final_quality_score: float  # weighted average of the five dimensions
    issues: list[CriticIssue]
    suggestions: list[CriticSuggestion]
    passed: bool  # computed — do not trust raw LLM value

    def __post_init__(self) -> None:
        # Override whatever the LLM returned to guarantee consistency.
        # Router logic in graph/router.py depends on this field being correct.
        self.passed = self.final_quality_score >= 4.0


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

from typing import TypedDict  # noqa: E402 — after dataclasses for readability


class WorkerInput(TypedDict):
    """Payload sent to each worker node via LangGraph Send().

    The dispatch function in graph/builder.py constructs one WorkerInput
    per sub-query and fans out with Send("worker_node", worker_input).
    """

    worker_id: str
    sub_query: str
    source_type: SourceType


class ResearchAgentState(TypedDict):
    """Master state flowing through all LangGraph nodes.

    worker_results uses Annotated[..., operator.add] so that parallel
    workers dispatched via Send() each append one WorkerResult and
    LangGraph accumulates them all rather than the last write winning.
    """

    # --- Input ---
    user_query: str
    intent_type: IntentType
    output_format: OutputFormat

    # --- Decomposition (set by supervisor node) ---
    sub_queries: list[str]
    source_routing: dict[str, SourceType]  # {sub_query: source_type}

    # --- Research (reducer required for Send fan-out) ---
    worker_results: Annotated[list[WorkerResult], operator.add]

    # --- Post-compression (set by boundary_compressor, read by coordinator) ---
    # operator.add cannot be "replaced", so boundary_compressor writes here instead.
    compressed_worker_results: list[WorkerResult]

    # --- Synthesis ---
    synthesis: Optional[SynthesisOutput]
    grounding_issues: list[str]  # populated by grounding_check node

    # --- Critique ---
    critic_output: Optional[CriticOutput]
    revision_count: int  # starts at 0, hard cap at 2

    # --- Memory ---
    prior_syntheses: list[SynthesisOutput]  # history for coordinator revision context

    # --- Meta ---
    session_id: str
    total_tokens_used: int
    cost_usd: float
    final_output: Optional[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Type aliases
    "SourceType",
    "IntentType",
    "OutputFormat",
    "IssueSeverity",
    # Dataclasses
    "Evidence",
    "Finding",
    "CriticIssue",
    "CriticSuggestion",
    "WorkerResult",
    "SynthesisOutput",
    "CriticOutput",
    # TypedDicts
    "WorkerInput",
    "ResearchAgentState",
]

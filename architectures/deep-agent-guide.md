# Deep Research Agent — Full Build Guide

> **Status:** Ready to build  
> **Feature scope:** Deep Research Agent (one module of the larger multi-agent MCP-layered application)  
> **Primary framework:** LangGraph (Python)  
> **Primary model:** Claude Sonnet 4.6 (Anthropic API)  
> **Observability:** LangSmith (from Day 1, not an afterthought)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Non-Negotiable Principles](#2-the-non-negotiable-principles)
3. [Agent State Schema](#3-agent-state-schema)
4. [Step-by-Step Node Definitions](#4-step-by-step-node-definitions)
5. [Full Tech Stack](#5-full-tech-stack)
6. [Phased Build Plan](#6-phased-build-plan)
7. [Prompt Engineering Guide (Per Agent)](#7-prompt-engineering-guide-per-agent)
8. [Critical Implementation Details](#8-critical-implementation-details)
9. [Evaluation & Grounding](#9-evaluation--grounding)
10. [Output Delivery via SKILL.md](#10-output-delivery-via-skillmd)
11. [Cost & Observability](#11-cost--observability)
12. [Common Failure Modes & How to Avoid Them](#12-common-failure-modes--how-to-avoid-them)
13. [File & Folder Structure](#13-file--folder-structure)
14. [Environment Setup](#14-environment-setup)

---

## 1. Architecture Overview

```
INPUT: Research question
        │
        ▼
┌─────────────────────────────────────────────────┐
│  STEP 1 — SUPERVISOR (intent + decomposition)   │
│  Claude Sonnet 4.6                               │
│  • Classifies intent type                        │
│  • Generates 3-5 targeted sub-queries            │
│  • Routes each query to the right source type    │
└────────────────────┬────────────────────────────┘
                     │  (fan-out)
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  WORKER A    │ │  WORKER B    │ │  WORKER C    │
│  Web search  │ │  Academic /  │ │  News /      │
│  (Exa/Tavily)│ │  arXiv       │ │  recency     │
│              │ │              │ │              │
│  Returns:    │ │  Returns:    │ │  Returns:    │
│  {claim,     │ │  {claim,     │ │  {claim,     │
│   evidence,  │ │   evidence,  │ │   evidence,  │
│   source,    │ │   source,    │ │   source,    │
│   confidence}│ │   confidence}│ │   confidence}│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │  (fan-in)
                        ▼
┌─────────────────────────────────────────────────┐
│  STEP 3 — COORDINATOR (synthesis)               │
│  Claude Sonnet 4.6                               │
│  • Deduplicates overlapping findings             │
│  • Resolves contradictions explicitly            │
│  • Produces structured JSON output               │
│  • Reads prior attempt from memory on revision   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 3b — GROUNDING CHECK (NEW — non-optional) │
│  Lightweight verification pass                   │
│  • Every claim must trace to a specific source   │
│  • Ungrounded claims flagged before critic sees  │
│  • Dead/unreachable URLs caught here             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 4 — CRITIC (adversarial review)           │
│  Claude Sonnet 4.6 with strict adversarial prompt│
│  Evaluates:                                      │
│  • factuality_score (0-5)                        │
│  • citation_alignment_score (0-5)                │
│  • reasoning_score (0-5)                         │
│  • completeness_score (0-5)                      │
│  • bias_score (0-5)                              │
│  • final_quality_score (0-5, weighted average)   │
│  • issues[] with specific quotes from synthesis  │
│  • suggestions[] with actionable instructions    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 5 — CONDITIONAL ROUTER                    │
│  score >= 4 ──────────────────────► DELIVER     │
│  score < 4 ──► back to COORDINATOR              │
│               (max 2 revisions, then force-deliver│
│                with quality caveat appended)     │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 6 — STRUCTURED OUTPUT DELIVERY            │
│  Reads final JSON → routes to format:           │
│  • Markdown brief (default)                      │
│  • DOCX via SKILL.md                             │
│  • PDF via SKILL.md                              │
│  • PPTX via SKILL.md                             │
└─────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HORIZONTAL: LangSmith observability wraps ALL nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 2. The Non-Negotiable Principles

These are not best practices — they are load-bearing walls. Violate them and the system degrades in ways that are hard to diagnose.

### 2.1 Grounding is a first-class citizen

Every claim in the final brief must be traceable to a specific source with a verifiable URL. This is not an "advanced feature" — it is the minimum bar that separates a research *system* from a language model *guessing*. Build the grounding check in Phase 1.

### 2.2 The revision loop must actually change the synthesis

When the critic scores below 4 and triggers a revision, the coordinator must receive:
- The original synthesis (verbatim)
- The critic's issues with specific quotes
- An explicit instruction: **"Do not restate in different words. Address each flagged issue with new evidence, or explicitly mark it as a known gap."**

Without this, the revision loop is an expensive paraphrase machine.

### 2.3 The critic prompt is adversarial, not polite

The critic's system prompt must explicitly instruct it to distrust the synthesis it receives. It must look for what is missing, not just what is wrong. It must assume the coordinator is overconfident. This is a persona decision — get it right before you tune the scoring thresholds.

### 2.4 Observability before features

Wire LangSmith tracing in Phase 1 before the critic, before parallelism, before output formatting. You cannot debug a multi-agent system you cannot observe.

### 2.5 Version your prompts like code

Every agent's system prompt is software. It needs a version, a changelog, and a test set. Use LangSmith's prompt versioning hub. Treat any prompt change as a deployment with a regression test.

### 2.6 Cost ceiling per session

Set a maximum token budget per research session before you deploy. A session that triggers 2 revision loops with 5 workers can cost 10x a clean single-pass run. Track this from day one.

---

## 3. Agent State Schema

This is the single most important design decision. Everything in LangGraph flows through a typed state object. Get this right before writing any agent logic.

```python
from typing import TypedDict, Optional
from dataclasses import dataclass, field

@dataclass
class Evidence:
    claim: str
    supporting_chunks: list[str]
    source_url: str
    source_title: str
    published_date: Optional[str]
    confidence: float          # 0.0 - 1.0
    contradiction_score: float # 0.0 = no contradiction, 1.0 = directly contradicted

@dataclass
class WorkerResult:
    worker_id: str
    sub_query: str
    source_type: str           # "web" | "academic" | "news"
    evidence_items: list[Evidence]
    raw_search_results: list[dict]
    tokens_used: int

@dataclass
class SynthesisOutput:
    topic: str
    executive_summary: str
    findings: list[dict]       # [{heading, body, evidence_refs[]}]
    risks: list[str]
    gaps: list[str]            # explicitly acknowledged unknowns
    citations: list[Evidence]
    synthesis_version: int     # increments on each revision
    prior_attempt_summary: Optional[str]  # filled on revision pass

@dataclass
class CriticOutput:
    factuality_score: float
    citation_alignment_score: float
    reasoning_score: float
    completeness_score: float
    bias_score: float
    final_quality_score: float  # weighted: factuality 30%, citation 25%, reasoning 20%, completeness 15%, bias 10%
    issues: list[dict]          # [{issue_text, quote_from_synthesis, severity}]
    suggestions: list[dict]     # [{action, target_section, new_evidence_needed}]
    passed: bool                # final_quality_score >= 4.0

class ResearchAgentState(TypedDict):
    # Input
    user_query: str
    intent_type: str            # "exploratory" | "comparative" | "adversarial" | "factual"
    output_format: str          # "markdown" | "docx" | "pdf" | "pptx"

    # Decomposition
    sub_queries: list[str]
    source_routing: dict        # {sub_query: source_type}

    # Research
    worker_results: list[WorkerResult]

    # Synthesis
    synthesis: Optional[SynthesisOutput]
    grounding_issues: list[str]  # populated by grounding check

    # Critique
    critic_output: Optional[CriticOutput]
    revision_count: int

    # Memory
    prior_syntheses: list[SynthesisOutput]  # history for revision context

    # Meta
    session_id: str
    total_tokens_used: int
    cost_usd: float
    final_output: Optional[str]
```

---

## 4. Step-by-Step Node Definitions

### Node 1: Supervisor

**Responsibility:** Understand intent, decompose into sub-queries, route each to a source type.

```python
async def supervisor_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    Input:  state.user_query
    Output: state.intent_type, state.sub_queries, state.source_routing
    """
    # Classify intent first (separate call, cheap)
    # Then decompose with routing instructions
    # Return structured JSON: {intent_type, sub_queries[], source_routing{}}
```

**Intent types and their query shapes:**

| Intent | Description | Query shape |
|--------|-------------|-------------|
| `exploratory` | Broad landscape overview | Wide, diverse sub-queries |
| `comparative` | A vs B analysis | Parallel queries per subject |
| `adversarial` | Find weaknesses / risks | Explicitly negative framing |
| `factual` | Specific claim verification | Precise, targeted queries |

---

### Node 2: Parallel Workers

**Responsibility:** Execute web/academic/news searches, extract structured evidence.

```python
async def run_workers_parallel(state: ResearchAgentState) -> ResearchAgentState:
    """
    Fan-out: one coroutine per sub-query
    Fan-in: gather all results into state.worker_results
    Uses asyncio.gather() — LangGraph's send() pattern
    """
    tasks = [
        worker_node(query, source_type, worker_id)
        for worker_id, (query, source_type)
        in enumerate(state.source_routing.items())
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Handle partial failures gracefully — if one worker fails,
    # continue with remaining results rather than aborting
```

**Per-worker output contract:**
```json
{
  "worker_id": "worker_0",
  "sub_query": "...",
  "source_type": "web",
  "evidence_items": [
    {
      "claim": "...",
      "supporting_chunks": ["...", "..."],
      "source_url": "https://...",
      "source_title": "...",
      "published_date": "2025-03-01",
      "confidence": 0.85,
      "contradiction_score": 0.1
    }
  ]
}
```

---

### Node 3: Coordinator (Synthesis)

**Responsibility:** Merge all worker results into a coherent, structured brief.

```python
async def coordinator_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    On first pass: synthesize from worker_results
    On revision pass: receive prior synthesis + critic issues + explicit revision instruction
    """
    if state.revision_count > 0:
        # Include prior synthesis and critic feedback in prompt
        # Explicitly instruct: do not restate, address each issue
        pass
```

**On revision pass, the coordinator prompt must include:**
1. The original synthesis (full text)
2. Each critic issue with the specific quote it references
3. Instruction: *"Do not rephrase. Either provide new evidence to counter each issue, or explicitly add it to the `gaps[]` section as an acknowledged limitation."*

---

### Node 3b: Grounding Check

**Responsibility:** Verify every claim has a traceable, reachable source before the critic sees it.

```python
async def grounding_check_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    For each finding in synthesis:
      1. Check it references at least one evidence item
      2. Verify the source URL is reachable (HEAD request, not full fetch)
      3. Check the claim text is supported by the cited chunk (semantic match)
    Ungrounded findings → state.grounding_issues[]
    """
```

**Do not skip this.** A critic that scores a hallucinated synthesis as "4/5 faithful" — because it doesn't have access to ground truth — is a real failure mode. The grounding check catches it before critique.

---

### Node 4: Critic

**Responsibility:** Adversarial multi-dimensional evaluation. This node must be genuinely skeptical.

```python
async def critic_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    Evaluates synthesis on 5 dimensions.
    Must receive grounding_issues from Node 3b.
    Returns structured CriticOutput with specific quotes.
    """
```

**Scoring weights:**
```
final_quality_score = (
    factuality_score       * 0.30  +
    citation_alignment     * 0.25  +
    reasoning_score        * 0.20  +
    completeness_score     * 0.15  +
    bias_score             * 0.10
)
```

**Critic output must include specific quotes.** Not "the reasoning is weak" — but "the claim 'X causes Y' in Section 2 has no supporting evidence and contradicts the source cited in finding 3."

---

### Node 5: Conditional Router

```python
def router_node(state: ResearchAgentState) -> str:
    """
    Returns the name of the next node.
    """
    if state.critic_output.passed:
        return "deliver"
    elif state.revision_count >= 2:
        # Force delivery with quality caveat
        state.synthesis.findings.append({
            "heading": "Quality Notice",
            "body": "This brief did not reach the target quality threshold after 2 revision passes. "
                    "The following issues remain unresolved: " + 
                    str([i["issue_text"] for i in state.critic_output.issues])
        })
        return "deliver"
    else:
        state.revision_count += 1
        state.prior_syntheses.append(state.synthesis)
        return "coordinator"
```

---

### Node 6: Output Delivery

**Responsibility:** Format the final JSON into the requested output format.

```python
async def delivery_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    Reads state.output_format and routes:
    - "markdown" → render markdown string
    - "docx"     → call SKILL.md docx pipeline
    - "pdf"      → call SKILL.md pdf pipeline
    - "pptx"     → call SKILL.md pptx pipeline
    """
```

---

## 5. Full Tech Stack

### Core

| Layer | Choice | Reason |
|-------|--------|--------|
| Agent framework | **LangGraph 1.0** | Cyclical graphs, checkpointing, deferred nodes for map-reduce, stable API |
| LLM | **Claude Sonnet 4.6** | Best-in-class for long-context synthesis, already in your MCP stack |
| Python version | **3.11+** | Required for LangGraph async patterns |

### Search & Retrieval

| Source type | Tool | Notes |
|-------------|------|-------|
| Web (primary) | **Exa.ai** | Returns structured results with highlights, published_date, relevance scores |
| Web (fallback) | **Tavily** | Good fallback, simpler API |
| Academic | **Semantic Scholar API** | Free, returns abstracts + citations. arXiv API for preprints |
| News / recency | **Exa with date filter** | Filter to last 30/90/365 days per query |

### Storage & State

| Purpose | Tool | Notes |
|---------|------|-------|
| LangGraph checkpointing | **SQLite** (dev) / **PostgreSQL** (prod) | Built into LangGraph |
| Vector store (future) | **Qdrant** | For caching past research sessions |
| Metadata | **PostgreSQL** | Session records, cost tracking |
| File output | **Local / S3** | Final DOCX/PDF/PPTX files |

### Observability & Evaluation

| Purpose | Tool | Notes |
|---------|------|-------|
| Tracing | **LangSmith** | Framework-agnostic, wire it first |
| Eval metrics | **RAGAS** | Faithfulness, answer relevancy, context precision |
| Prompt versioning | **LangSmith Hub** | Version every agent prompt |
| Cost tracking | **Custom middleware** | Token count × model price, per session |

### Output Generation

| Format | Library | Activated by |
|--------|---------|-------------|
| DOCX | python-docx + SKILL.md | `output_format = "docx"` |
| PDF | WeasyPrint or ReportLab + SKILL.md | `output_format = "pdf"` |
| PPTX | python-pptx + SKILL.md | `output_format = "pptx"` |
| Markdown | Native string render | `output_format = "markdown"` (default) |

### Infrastructure (when scaling)

| Purpose | Tool |
|---------|------|
| Async task queue | **Celery + Redis** |
| API layer | **FastAPI** |
| Containerization | **Docker** |
| Deployment | **Railway / Render** (dev) → **AWS ECS / GCP Cloud Run** (prod) |

---

## 6. Phased Build Plan

### Phase 1 — Linear Pipeline + Observability (Week 1-2)

**Goal:** Get a working end-to-end pipeline with one worker (no parallelism) and basic output. Wire LangSmith tracing on every node.

**Deliverables:**
- [ ] `ResearchAgentState` fully typed and tested
- [ ] Supervisor node with intent classification + sub-query generation
- [ ] Single worker with Exa.ai search integration
- [ ] Coordinator with structured JSON synthesis
- [ ] Grounding check (basic URL reachability + reference check)
- [ ] Delivery node (markdown output only)
- [ ] LangSmith tracing on all nodes
- [ ] Cost tracking middleware

**What you are NOT building yet:** parallelism, critic, revision loop, multi-format output.

**Test it with 5-10 research questions and inspect the traces. Fix the coordinator's JSON schema compliance before moving on.**

---

### Phase 2 — Parallel Workers + Source Routing (Week 2-3)

**Goal:** True parallel execution across 3 workers with different source types.

**Deliverables:**
- [ ] `asyncio.gather()` worker fan-out
- [ ] LangGraph deferred coordinator node (waits for all workers)
- [ ] Source routing logic (web / academic / news per sub-query type)
- [ ] Semantic Scholar API integration for Worker B
- [ ] Partial failure handling (if one worker fails, continue with others)
- [ ] Worker result deduplication in coordinator

**Key check:** Verify that when Worker A and Worker C return contradicting claims, the coordinator represents both rather than silently preferring one.

---

### Phase 3 — Critic + Revision Loop (Week 3-4)

**Goal:** Full critique and conditional revision cycle with hard cap.

**Deliverables:**
- [ ] Critic node with 5-dimension scoring
- [ ] Critic prompt with adversarial persona (see Section 7)
- [ ] Conditional router (score >= 4 → deliver, else → revise)
- [ ] Revision instruction injection into coordinator on loop-back
- [ ] Revision counter in state with max=2 cap
- [ ] Force-deliver with quality caveat on cap hit
- [ ] Memory: prior synthesis stored in state for coordinator context

**Test it by deliberately giving the system bad source material and verifying the critic catches it.**

---

### Phase 4 — Grounding Hardening + Evaluation (Week 4-5)

**Goal:** Make the grounding check production-strength. Build an eval dataset.

**Deliverables:**
- [ ] Semantic similarity check: claim text vs cited chunk (use embeddings)
- [ ] URL reachability verification on all citations
- [ ] RAGAS integration for faithfulness and context precision scoring
- [ ] Eval dataset of 20-30 research questions with human-annotated expected outputs
- [ ] Regression test pipeline: run eval on any prompt change

---

### Phase 5 — Multi-Format Output (Week 5-6)

**Goal:** DOCX, PDF, and PPTX generation from final JSON.

**Deliverables:**
- [ ] DOCX delivery via SKILL.md pipeline
- [ ] PDF delivery via SKILL.md pipeline
- [ ] PPTX delivery (for executive summary format) via SKILL.md pipeline
- [ ] Output format selector in the API request
- [ ] Citation appendix in all long-form formats

---

### Phase 6 — Production Hardening (Week 6-7)

**Goal:** Retry logic, cost controls, API layer, deployment.

**Deliverables:**
- [ ] Exponential backoff on all external API calls
- [ ] Per-session cost ceiling enforcement (configurable, default $1.00)
- [ ] FastAPI wrapper with `/research` endpoint
- [ ] Session logging to PostgreSQL
- [ ] Docker containerization
- [ ] Rate limiting middleware

---

## 7. Prompt Engineering Guide (Per Agent)

### Supervisor Prompt

```
You are a research planning expert. Your job is to understand the user's research question and produce a structured decomposition plan.

First, classify the intent:
- "exploratory": the user wants a broad landscape overview
- "comparative": the user wants A vs B analysis
- "adversarial": the user wants to find weaknesses, risks, or counterarguments
- "factual": the user wants a specific claim verified

Then, generate 3-5 sub-queries that together cover the research question completely.
For each sub-query, specify the source type: "web", "academic", or "news".

Rules:
- Sub-queries must be specific, not generic.
- No two sub-queries should be semantically redundant.
- The "academic" type is for scientific/technical claims needing peer-reviewed evidence.
- The "news" type is for recency-sensitive topics (events, market conditions, policy changes).

Return ONLY valid JSON in this exact format:
{
  "intent_type": "...",
  "sub_queries": [
    {"query": "...", "source_type": "web|academic|news", "rationale": "..."},
    ...
  ]
}
```

---

### Worker Prompt

```
You are a research extraction specialist. You have been given a search query and a set of search results.

Your job is to extract structured evidence from these results.

For each distinct claim you find:
1. State the claim precisely (one sentence)
2. Quote the supporting text chunk verbatim
3. Record the source URL and title
4. Assign a confidence score (0.0-1.0) based on source quality and specificity
5. Assign a contradiction_score (0.0-1.0) if you find conflicting evidence in other results

Do not infer. Do not extrapolate. Only extract what is directly stated in the provided results.
Flag claims from sources older than 2 years with low confidence unless they describe stable facts.

Return ONLY valid JSON matching the Evidence schema.
```

---

### Coordinator Prompt (First Pass)

```
You are a research synthesis specialist. You have received evidence from multiple research workers.

Your job is to produce a unified, grounded research brief.

Rules:
1. Every finding must cite at least one evidence item by source_url.
2. If two workers report contradicting claims, represent BOTH — do not resolve by choosing one.
3. Explicitly list what is unknown or not covered in a "gaps" section.
4. Do not add any claim that is not present in the provided evidence.
5. Do not use phrases like "it is widely known" or "experts agree" without citation.

Return ONLY valid JSON matching the SynthesisOutput schema.
```

---

### Coordinator Prompt (Revision Pass — CRITICAL)

```
You are a research synthesis specialist receiving a revision request.

Below is your previous synthesis, followed by the critic's specific issues.

IMPORTANT RULES FOR THIS REVISION:
1. Do NOT restate the same content in different words.
2. For each issue in the critic's list:
   - If you can address it with evidence already in the worker results: do so explicitly.
   - If you cannot address it: add it to the "gaps" section with a note: "Unable to resolve: [critic's issue]"
3. Do not add new claims that were not in the original worker results.
4. Increase the synthesis_version by 1.
5. In prior_attempt_summary, briefly describe what changed and why.

Previous synthesis: {{prior_synthesis}}
Critic issues: {{critic_issues}}
Worker results: {{worker_results}}

Return ONLY valid JSON matching the SynthesisOutput schema.
```

---

### Critic Prompt (MOST IMPORTANT — tune this hardest)

```
You are an adversarial research critic. Your job is to find problems in the synthesis you receive.

You are skeptical by default. You do not give benefit of the doubt. You assume the synthesis is overconfident until proven otherwise.

Evaluate on five dimensions:

1. FACTUALITY (0-5): Are claims accurate and supported by cited sources?
   - 5: Every claim is directly supported by a cited source
   - 3: Most claims are supported; some are inferred
   - 1: Claims are present with no supporting evidence

2. CITATION ALIGNMENT (0-5): Does each citation actually support the claim it's attached to?
   - 5: Every citation directly supports its claim
   - 3: Most citations are relevant; some are tangential
   - 1: Citations are present but don't support the claims they're attached to

3. REASONING (0-5): Is the logical structure sound?
   - 5: Conclusions follow directly from evidence; no logical leaps
   - 3: Mostly sound with minor inferential gaps
   - 1: Conclusions are not supported by the evidence presented

4. COMPLETENESS (0-5): Are important perspectives or counterarguments missing?
   - 5: All major perspectives are represented
   - 3: Some perspectives are present; notable ones are missing
   - 1: Significant perspectives are absent

5. BIAS (0-5): Is the synthesis neutral?
   - 5: No detectable framing bias
   - 3: Minor framing bias present
   - 1: Significant framing or selection bias

For each issue you identify:
- Quote the EXACT text from the synthesis that is problematic
- Explain why it is a problem
- Rate severity: "critical" | "major" | "minor"

For each suggestion:
- Name the specific action required
- Name the section it applies to
- Indicate whether new evidence is needed or existing evidence can address it

Return ONLY valid JSON matching the CriticOutput schema.
```

---

## 8. Critical Implementation Details

### 8.1 Handling Worker Failures Gracefully

```python
results = await asyncio.gather(*tasks, return_exceptions=True)

successful_results = []
failed_workers = []

for i, result in enumerate(results):
    if isinstance(result, Exception):
        failed_workers.append({"worker_id": i, "error": str(result)})
    else:
        successful_results.append(result)

if len(successful_results) == 0:
    raise ResearchAgentError("All workers failed. Cannot proceed.")

# Continue with partial results — note failed workers in gaps section
```

### 8.2 The Deferred Coordinator Pattern in LangGraph

```python
from langgraph.graph import StateGraph, Send

def dispatch_workers(state: ResearchAgentState):
    """Fan-out: send one message per sub-query"""
    return [
        Send("worker_node", {"query": q, "source_type": t, "worker_id": i})
        for i, (q, t) in enumerate(state.source_routing.items())
    ]

workflow.add_conditional_edges(
    "supervisor",
    dispatch_workers,
    ["worker_node"]
)
# LangGraph automatically defers the coordinator until all worker_node calls complete
```

### 8.3 Cost Tracking Middleware

```python
import anthropic

def track_cost(response, state: ResearchAgentState, model: str) -> ResearchAgentState:
    # claude-sonnet-4-6: $3/M input tokens, $15/M output tokens (verify current pricing)
    input_cost = (response.usage.input_tokens / 1_000_000) * 3.00
    output_cost = (response.usage.output_tokens / 1_000_000) * 15.00
    
    state["total_tokens_used"] += response.usage.input_tokens + response.usage.output_tokens
    state["cost_usd"] += input_cost + output_cost
    
    # Enforce ceiling
    if state["cost_usd"] > COST_CEILING_USD:
        raise CostCeilingExceeded(f"Session cost ${state['cost_usd']:.4f} exceeded ceiling ${COST_CEILING_USD}")
    
    return state
```

### 8.4 JSON Schema Compliance

All agents return JSON. Claude is highly reliable at structured JSON output when you:

1. Provide the exact schema in the prompt
2. Use `response_format` if available, otherwise end the prompt with: *"Return ONLY valid JSON. No preamble, no markdown fences, no explanation."*
3. Wrap all JSON parsing in try/except with a retry mechanism:

```python
import json

def parse_agent_json(raw: str, schema_class, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            # Strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return schema_class(**json.loads(clean))
        except (json.JSONDecodeError, TypeError) as e:
            if attempt == max_retries:
                raise AgentParseError(f"Failed to parse agent output after {max_retries} retries: {e}")
            # Retry with correction prompt
```

---

## 9. Evaluation & Grounding

### 9.1 RAGAS Metrics to Track

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Faithfulness | Are all claims supported by retrieved context? | > 0.85 |
| Answer Relevancy | Does the answer address the question? | > 0.80 |
| Context Precision | Is retrieved context relevant? | > 0.75 |
| Context Recall | Does context cover the answer? | > 0.70 |

### 9.2 Building Your Eval Dataset

Start with 20-30 questions spanning all 4 intent types. For each question, manually annotate:
- Expected key findings (must appear in output)
- Expected citations (at least one from a credible source)
- Forbidden claims (known hallucinations on this topic)
- Expected gaps (things the system should acknowledge as unknown)

Run this eval on every prompt change. Gate merges on no regression against the baseline.

### 9.3 Citation Verification

```python
import httpx

async def verify_citation(url: str, timeout: float = 5.0) -> bool:
    """Check if a URL is reachable (HEAD request, not full fetch)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(url, timeout=timeout, follow_redirects=True)
            return response.status_code < 400
    except Exception:
        return False
```

---

## 10. Output Delivery via SKILL.md

### Format Selection Logic

```python
OUTPUT_FORMAT_MAP = {
    "markdown": render_markdown_brief,
    "docx":     generate_docx_via_skill,
    "pdf":      generate_pdf_via_skill,
    "pptx":     generate_pptx_via_skill,
}

async def delivery_node(state: ResearchAgentState) -> ResearchAgentState:
    formatter = OUTPUT_FORMAT_MAP.get(state["output_format"], render_markdown_brief)
    state["final_output"] = await formatter(state["synthesis"])
    return state
```

### Standard Brief Structure

All output formats should follow this structure:

```
1. Executive Summary (150-200 words)
2. Key Findings (N sections, each with heading + body + inline citations)
3. Risk / Counterarguments section (if intent_type is adversarial or completeness score flagged)
4. Known Gaps / Limitations (explicitly acknowledged unknowns)
5. Quality Score (critic's final_quality_score, shown to user for transparency)
6. Citations / References (numbered, with URL and publication date)
```

### Quality Score Transparency

Always show the critic's `final_quality_score` in the output. This builds trust and signals to the user when a brief hit the quality ceiling with unresolved issues.

---

## 11. Cost & Observability

### LangSmith Setup (Do This First)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "deep-research-agent"

# That's it. LangGraph automatically emits traces to LangSmith.
# Every node call, every LLM call, every tool call — all traced.
```

### What to Monitor in LangSmith

| Metric | Dashboard | Alert threshold |
|--------|-----------|----------------|
| Revision rate | % of sessions needing > 0 revisions | > 40% → tune critic |
| Force-deliver rate | % of sessions hitting revision cap | > 10% → tune coordinator |
| Avg cost per session | Cost dashboard | > $0.80 → optimize prompts |
| Critic score distribution | Custom eval | Median < 3.5 → redesign synthesis |
| Worker failure rate | Error tracking | > 5% → add retry logic |
| P95 latency | Performance | > 120s → optimize parallelism |

### Per-Session Cost Estimate (Approximate)

| Component | Tokens (est.) | Cost (est.) |
|-----------|--------------|-------------|
| Supervisor | ~1K input + 0.5K output | $0.01 |
| 3 Workers × search+extract | ~3K input + 1K output each | $0.07 |
| Coordinator (first pass) | ~8K input + 2K output | $0.05 |
| Grounding check | ~2K input + 0.5K output | $0.01 |
| Critic | ~6K input + 1K output | $0.03 |
| Coordinator (revision, if needed) | ~10K input + 2K output | $0.06 |
| **Total (single pass)** | | **~$0.17** |
| **Total (with 1 revision)** | | **~$0.26** |
| **Total (with 2 revisions)** | | **~$0.35** |

*Based on approximate Claude Sonnet 4.6 pricing. Verify current pricing at console.anthropic.com.*

---

## 12. Common Failure Modes & How to Avoid Them

| Failure | Symptom | Prevention |
|---------|---------|-----------|
| Coordinator hallucination | Claims with no source | Grounding check node |
| Revision loop stall | Coordinator rewords, doesn't fix | Explicit revision instruction (see Section 7) |
| Critic too lenient | Scores 4/5 on bad synthesis | Adversarial critic prompt (see Section 7) |
| Critic too harsh | Always scores < 4, infinite loops | Hard revision cap of 2 |
| Worker coordination failure | One failed worker aborts session | `return_exceptions=True` in gather |
| JSON parse failure | Agent returns prose instead of JSON | Retry with correction prompt |
| Cost overrun | Session costs $5+ | Cost ceiling middleware |
| Stale citations | URL 404s in the brief | URL reachability check in grounding node |
| Intent misclassification | Comparative query treated as factual | Explicit few-shot examples in supervisor prompt |
| Prompt drift | New prompt breaks old eval cases | Prompt versioning + regression eval on every change |

---

## 13. File & Folder Structure

```
deep_research_agent/
│
├── agents/
│   ├── __init__.py
│   ├── supervisor.py          # Intent classification + query decomposition
│   ├── worker.py              # Search + evidence extraction
│   ├── coordinator.py         # Synthesis (first pass + revision pass)
│   ├── grounding_check.py     # Citation verification + claim tracing
│   ├── critic.py              # Adversarial multi-dimensional evaluation
│   └── delivery.py            # Output formatting + SKILL.md routing
│
├── graph/
│   ├── __init__.py
│   ├── state.py               # ResearchAgentState + all dataclasses
│   ├── builder.py             # LangGraph workflow assembly
│   └── router.py              # Conditional edge logic
│
├── tools/
│   ├── __init__.py
│   ├── exa_search.py          # Exa.ai integration
│   ├── academic_search.py     # Semantic Scholar + arXiv
│   ├── url_verifier.py        # Citation reachability checks
│   └── cost_tracker.py        # Token cost middleware
│
├── output/
│   ├── __init__.py
│   ├── markdown_renderer.py
│   ├── docx_generator.py      # Uses SKILL.md pipeline
│   ├── pdf_generator.py       # Uses SKILL.md pipeline
│   └── pptx_generator.py      # Uses SKILL.md pipeline
│
├── eval/
│   ├── dataset/               # 20-30 annotated test questions
│   ├── ragas_eval.py          # Faithfulness, relevancy metrics
│   └── regression_test.py     # Run on every prompt change
│
├── prompts/
│   ├── supervisor_v1.txt
│   ├── worker_v1.txt
│   ├── coordinator_v1.txt
│   ├── coordinator_revision_v1.txt
│   └── critic_v1.txt
│
├── api/
│   ├── __init__.py
│   └── routes.py              # FastAPI /research endpoint
│
├── config.py                  # COST_CEILING_USD, model names, timeouts
├── main.py                    # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 14. Environment Setup

### Requirements

```txt
# requirements.txt

# Core
langgraph>=1.0.0
langchain-anthropic>=0.3.0
anthropic>=0.40.0

# Search
exa-py>=1.0.0
tavily-python>=0.3.0

# Observability
langsmith>=0.1.0

# Evaluation
ragas>=0.2.0

# Output generation
python-docx>=1.1.0
python-pptx>=1.0.0
reportlab>=4.0.0
weasyprint>=62.0

# API
fastapi>=0.115.0
uvicorn>=0.32.0

# Async + infra
httpx>=0.27.0
redis>=5.0.0
celery>=5.4.0

# Data
pydantic>=2.8.0
psycopg2-binary>=2.9.0

# Dev
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

### Environment Variables

```bash
# .env.example

# Anthropic
ANTHROPIC_API_KEY=

# LangSmith (set up FIRST)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=deep-research-agent

# Search
EXA_API_KEY=
TAVILY_API_KEY=

# Database
DATABASE_URL=postgresql://localhost:5432/research_agent

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Cost control
COST_CEILING_USD=1.00

# Output
OUTPUT_DIR=./outputs
```

### Quick Start

```bash
# 1. Clone / create your project
mkdir deep_research_agent && cd deep_research_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in your API keys

# 5. Run Phase 1 baseline (linear pipeline, one worker)
python main.py --query "What are the current risks of enterprise AI adoption?" --format markdown

# 6. Open LangSmith and inspect the trace
# https://smith.langchain.com → your project → latest run
```

---

## Quick Reference: The 10 Rules

1. **Grounding check is not optional.** Every claim needs a traceable source before the critic sees it.
2. **The revision instruction must be explicit.** Not "improve this" — but "address each of these specific issues or acknowledge them as gaps."
3. **The critic is adversarial, not helpful.** Its job is to distrust the synthesis, not validate it.
4. **LangSmith before everything else.** You cannot debug what you cannot observe.
5. **Version your prompts like code.** Every change needs a test run against your eval dataset.
6. **Set a cost ceiling from day one.** Revision loops compound token usage fast.
7. **Handle worker failures gracefully.** One failed search should not abort the session.
8. **Build linear first, then parallel.** Don't introduce concurrency until the single-worker pipeline is solid.
9. **Show the quality score in the output.** Transparency builds trust.
10. **Don't add cross-agent debate before you have a working critic loop.** Sequence matters.

---

*Last updated: March 2026 | Architecture version: 1.0 | Framework: LangGraph 1.0 | Model: Claude Sonnet 4.6*
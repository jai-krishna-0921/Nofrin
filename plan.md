# Plan: agents/coordinator.py

## Problem

The coordinator node synthesizes all compressed worker evidence into a structured `SynthesisOutput`
research brief. It has two distinct execution paths: a **first pass** (initial synthesis from
evidence) and a **revision pass** (targeted rewrite guided by specific critic issues). The two
paths use different prompts, different inputs, and different state mutations — but share the same
LLM call infrastructure, two-level JSON deserialization, and cache_control pattern.

Key challenges:
1. Two-level deserialization: `SynthesisOutput` contains nested `Finding` objects and `Evidence`
   citations — `parse_agent_json()` cannot recurse, so flat `TypedDict` intermediates are required.
2. Citation mapping: the LLM returns `citation_urls` (a flat list of strings); coordinator must
   map these back to actual `Evidence` objects from `compressed_worker_results`.
3. State mutation asymmetry: revision pass must move current `synthesis` → `prior_syntheses`
   BEFORE overwriting `synthesis`; first pass does not touch `prior_syntheses`.
4. Evidence-ref validation at parse time: every `finding.evidence_refs` URL must exist in
   `compressed_worker_results` — this is a pre-grounding-check catch.

---

## Sequential-Thinking Analysis

**Path routing**: `revision_count == 0` → PATH 1; `revision_count > 0` → PATH 2. This is read
from state at the top of `coordinator_node` to select prompt template and message builder.

**Two-level parse (same as supervisor.py)**:
- LLM returns JSON with `findings: list[dict]` and `citation_urls: list[str]`
- `_RawSynthesisOutput` TypedDict captures the raw shape (no dataclasses)
- `_RawFinding` TypedDict: `{heading: str, body: str, evidence_refs: list[str]}`
- After parse: convert each `_RawFinding` → `Finding`; map `citation_urls` to `Evidence` objects

**Evidence serialization**: serialize `compressed_worker_results` into a readable text block
grouped by worker/sub_query. Cap total at ~6000 chars to prevent context overflow.

**State writes (per path)**:
- PATH 1: `{"synthesis": new, "total_tokens_used": accumulated}`
- PATH 2: `{"synthesis": new, "prior_syntheses": updated_list, "revision_count": incremented,
           "total_tokens_used": accumulated}`

**`synthesis_version`**: 1 on first pass; `prior_synthesis.synthesis_version + 1` on revision.

**`prior_attempt_summary`**: None on first pass; brief one-sentence summary of what the prior
synthesis attempted (set to `prior_synthesis.executive_summary[:120]`) on revision.

**Alternatives rejected**:
- Single prompt for both paths: revision prompt must forbid restating prior synthesis — injecting
  this into a shared prompt creates ambiguity about when the constraint applies.
- Storing raw evidence in state: already stripped by boundary_compressor; use only what's there.

---

## Files to create / modify

| Path | Action |
|---|---|
| `agents/coordinator.py` | **Create** — coordinator node, both paths |
| `prompts/coordinator_v1.txt` | **Create** — first-pass synthesis prompt |
| `prompts/coordinator_revision_v1.txt` | **Create** — revision prompt with verbatim constraint |
| `prompts/CHANGELOG.md` | **Update** — add both prompt entries |
| `agents/__init__.py` | **Update** — export `coordinator_node` |
| `graph/builder.py` | **Update** — wire `boundary_compressor → coordinator` |
| `tests/test_coordinator.py` | **Create** — 27 unit tests |

---

## Prompt files (full text)

### `prompts/coordinator_v1.txt`

```
You are a research synthesis specialist. Your task is to synthesize evidence from multiple web, academic, and news sources into a structured research brief.

RESEARCH EVIDENCE:
{{evidence_block}}

TASK:
Synthesize the evidence above into a comprehensive research brief. Follow these rules strictly:

1. CONFLICT RULE: If two or more sources make conflicting claims about the same topic, you MUST represent BOTH positions in findings[]. Never silently discard a minority view. Label the finding heading to indicate the conflict, e.g. "Conflicting Views on X".

2. CITATION RULE: Every finding body must be supported by at least one source URL from the evidence. List the supporting URLs in evidence_refs[].

3. GAPS RULE: If the evidence does not cover an aspect that would be needed to fully answer the research question, list it explicitly in gaps[]. Do not fabricate answers.

4. SCOPE RULE: Do not introduce claims not present in the evidence above.

Return ONLY valid JSON — no preamble, no markdown fences, no explanation:

{
  "topic": "Concise topic label (5–10 words)",
  "executive_summary": "2–3 sentence overview of the key findings",
  "findings": [
    {
      "heading": "Finding heading (concise)",
      "body": "Detailed finding text citing the evidence",
      "evidence_refs": ["https://source-url-1.com", "https://source-url-2.com"]
    }
  ],
  "risks": ["Risk or caveat 1", "Risk or caveat 2"],
  "gaps": ["Gap or unknown 1", "Gap or unknown 2"],
  "citation_urls": ["https://source-url-1.com", "https://source-url-2.com"]
}

citation_urls must be a flat list of ALL source URLs you cited across all findings.
```

### `prompts/coordinator_revision_v1.txt`

```
You are a research synthesis specialist performing a targeted revision of a prior research brief.

RESEARCH EVIDENCE:
{{evidence_block}}

PRIOR SYNTHESIS (do not restate — address the issues below):
{{prior_synthesis_block}}

CRITIC ISSUES TO ADDRESS:
{{critic_issues_block}}

TASK:
Revise the research brief to address the critic's issues. Follow these rules strictly:

Do not restate the prior synthesis in different words. For each issue in the critic's list: either provide new evidence from the worker results to address it, or add it to gaps[] with the note "Unresolved: [issue text]". Do not add claims not present in worker results.

Additional rules:
1. CONFLICT RULE: If two or more sources make conflicting claims, represent BOTH in findings[].
2. CITATION RULE: Every finding must cite at least one source URL in evidence_refs[].
3. SCOPE RULE: Do not introduce claims not present in the evidence.
4. VERSION RULE: This is a revision — only change what the critic identified. Do not remove valid findings from the prior synthesis unless the critic explicitly flagged them.

Return ONLY valid JSON — no preamble, no markdown fences, no explanation:

{
  "topic": "Concise topic label (5–10 words)",
  "executive_summary": "2–3 sentence overview of the revised key findings",
  "findings": [
    {
      "heading": "Finding heading (concise)",
      "body": "Detailed finding text citing the evidence",
      "evidence_refs": ["https://source-url-1.com"]
    }
  ],
  "risks": ["Risk or caveat 1"],
  "gaps": ["Gap 1", "Unresolved: [critic issue text if unresolvable]"],
  "citation_urls": ["https://source-url-1.com"]
}

citation_urls must be a flat list of ALL source URLs cited across all findings.
```

---

## TypedDicts for two-level deserialization

```python
class _RawFinding(TypedDict):
    """Raw finding dict as returned by the LLM."""
    heading: str
    body: str
    evidence_refs: list[str]


class _RawSynthesisOutput(TypedDict):
    """Flat TypedDict parse_agent_json() can instantiate directly.

    findings and citation_urls are kept as raw dicts/strings here;
    conversion to Finding dataclasses and Evidence objects happens
    in _parse_and_validate().
    """
    topic: str
    executive_summary: str
    findings: list[_RawFinding]
    risks: list[str]
    gaps: list[str]
    citation_urls: list[str]
```

---

## Code outline (signatures + docstrings only)

```python
"""
agents/coordinator.py

Coordinator node: synthesizes compressed worker evidence into SynthesisOutput.

Two execution paths:
  PATH 1 (revision_count == 0): first-pass synthesis from evidence only.
  PATH 2 (revision_count > 0):  targeted revision guided by critic issues.

Reads:  state["compressed_worker_results"], state["revision_count"],
        state["prior_syntheses"], state["critic_output"]
Writes: state["synthesis"], state["prior_syntheses"] (PATH 2 only),
        state["revision_count"] (PATH 2 only), state["total_tokens_used"]
"""

FIRST_PASS_PROMPT_PATH = Path(...) / "prompts" / "coordinator_v1.txt"
REVISION_PROMPT_PATH   = Path(...) / "prompts" / "coordinator_revision_v1.txt"


def _is_anthropic(llm: BaseChatModel) -> bool:
    """Return True if llm is ChatAnthropic — same pattern as supervisor._is_anthropic."""
    ...


def _load_prompt(path: Path) -> str:
    """Load prompt template from path.
    Raises FileNotFoundError with clear message if missing.
    """
    ...


def _serialize_evidence(worker_results: list[WorkerResult]) -> str:
    """Serialize compressed_worker_results into a readable text block for the LLM.

    Groups evidence by worker sub_query. Format per item:
      [E{n}] CLAIM: {claim} | CONFIDENCE: {confidence:.2f} | URL: {source_url}

    Caps total output at ~6000 characters to prevent context overflow.
    Returns the serialized string.
    """
    ...


def _serialize_critic_issues(issues: list[CriticIssue]) -> str:
    """Serialize CriticIssue list into a numbered list for the revision prompt.

    Format: "{n}. [{severity}] {issue_text}\n   Quote: \"{quote_from_synthesis}\""
    Returns the serialized string.
    """
    ...


def _build_first_pass_messages(
    evidence_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for PATH 1 (first-pass synthesis).

    use_cache_control=True (Anthropic): static prompt in cached block,
      evidence_block in uncached block.
    use_cache_control=False (others): plain string SystemMessage with
      {{evidence_block}} placeholder substituted.
    """
    ...


def _build_revision_messages(
    evidence_block: str,
    prior_synthesis_block: str,
    critic_issues_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for PATH 2 (revision pass).

    use_cache_control=True: static prompt cached; dynamic content
      (evidence + prior synthesis + critic issues) in uncached block.
    use_cache_control=False: plain string SystemMessage with all
      placeholders substituted.
    """
    ...


def _parse_and_validate(
    raw: str,
    available_urls: set[str],
    synthesis_version: int,
    prior_attempt_summary: str | None,
) -> SynthesisOutput:
    """Parse LLM JSON response and validate against available evidence URLs.

    Steps:
      1. parse_agent_json(raw, _RawSynthesisOutput)
      2. Validate: executive_summary not empty
      3. Validate: at least 1 finding
      4. For each finding: at least 1 evidence_ref, all refs in available_urls
      5. Convert _RawFinding → Finding dataclass
      6. Map citation_urls → Evidence objects (only URLs in available_urls kept)
      7. Build and return SynthesisOutput(synthesis_version=synthesis_version, ...)

    Raises:
        AgentParseError: if any validation rule fails (triggers retry).
    """
    ...


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (AgentParseError, httpx.HTTPStatusError, httpx.TimeoutException)
    ),
    reraise=True,
)
async def _call_llm(
    messages: list[BaseMessage],
    llm: BaseChatModel,
    available_urls: set[str],
    synthesis_version: int,
    prior_attempt_summary: str | None,
) -> tuple[SynthesisOutput, int]:
    """Call the LLM and parse/validate the response.

    Returns: (SynthesisOutput, tokens_used).
    Retries on AgentParseError, HTTP errors, timeouts (3×, exp backoff 1–10s).
    """
    ...


async def coordinator_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, object]:
    """LangGraph node: synthesize worker evidence into SynthesisOutput.

    PATH 1 (revision_count == 0):
      - Builds first-pass messages from compressed_worker_results only.
      - Does NOT modify prior_syntheses or revision_count.
      - Sets synthesis_version = 1.

    PATH 2 (revision_count > 0):
      - Moves state["synthesis"] → prior_syntheses before overwriting.
      - Builds revision messages with evidence + prior synthesis + critic issues.
      - Increments revision_count by 1.
      - Sets synthesis_version = prior.synthesis_version + 1.

    Args:
        state: ResearchAgentState. Reads compressed_worker_results, revision_count,
               synthesis, prior_syntheses, critic_output, total_tokens_used.
        runtime: Injected NofrinContext. Uses llm_coordinator.

    Returns:
        Partial state update dict. Keys depend on path (see above).

    Raises:
        AgentParseError: after 3 failed LLM/parse attempts.
        ValueError: if compressed_worker_results is empty (no evidence to synthesize).
    """
    ...
```

---

## JSON contract

**LLM must return:**
```json
{
  "topic": "Impact of AI on Healthcare Diagnostics",
  "executive_summary": "AI diagnostic tools have achieved parity with radiologists in specific imaging tasks, though broad clinical adoption faces regulatory and trust barriers.",
  "findings": [
    {
      "heading": "AI Matches Radiologist Accuracy in Chest X-Rays",
      "body": "Multiple peer-reviewed studies show AI models achieve AUC >0.95 for pneumonia detection, matching specialist performance.",
      "evidence_refs": ["https://nejm.org/ai-radiology-2024", "https://arxiv.org/abs/2401.12345"]
    },
    {
      "heading": "Conflicting Views on Regulatory Readiness",
      "body": "FDA officials argue existing frameworks suffice; independent researchers contend AI bias in training data requires new validation protocols.",
      "evidence_refs": ["https://fda.gov/ai-guidance", "https://bmj.com/ai-bias-study"]
    }
  ],
  "risks": ["Training data bias may not generalise to non-Western populations"],
  "gaps": ["Long-term patient outcome data beyond 2-year follow-up not available in sources"],
  "citation_urls": [
    "https://nejm.org/ai-radiology-2024",
    "https://arxiv.org/abs/2401.12345",
    "https://fda.gov/ai-guidance",
    "https://bmj.com/ai-bias-study"
  ]
}
```

---

## State fields written

| Field | Type | PATH 1 | PATH 2 |
|---|---|---|---|
| `synthesis` | `SynthesisOutput` | ✅ written | ✅ written (new version) |
| `prior_syntheses` | `list[SynthesisOutput]` | ❌ not touched | ✅ append old synthesis first |
| `revision_count` | `int` | ❌ not touched | ✅ `+= 1` |
| `total_tokens_used` | `int` | ✅ `state value + tokens` | ✅ `state value + tokens` |

---

## Tests — `tests/test_coordinator.py` (27 tests)

| # | Test | Assertion |
|---|---|---|
| 1 | `test_first_pass_returns_synthesis` | result contains `synthesis` key |
| 2 | `test_first_pass_synthesis_version_is_1` | `synthesis.synthesis_version == 1` |
| 3 | `test_first_pass_does_not_modify_prior_syntheses` | `"prior_syntheses"` not in result |
| 4 | `test_first_pass_does_not_increment_revision_count` | `"revision_count"` not in result |
| 5 | `test_first_pass_accumulates_total_tokens` | `result["total_tokens_used"] == state_tokens + coordinator_tokens` |
| 6 | `test_revision_pass_moves_synthesis_to_prior` | `result["prior_syntheses"][-1] == old_synthesis` |
| 7 | `test_revision_pass_increments_revision_count` | `result["revision_count"] == state["revision_count"] + 1` |
| 8 | `test_revision_pass_synthesis_version_increments` | `result["synthesis"].synthesis_version == prior.synthesis_version + 1` |
| 9 | `test_revision_pass_uses_revision_prompt` | `_load_prompt` called with `coordinator_revision_v1.txt` path |
| 10 | `test_first_pass_uses_first_pass_prompt` | `_load_prompt` called with `coordinator_v1.txt` path |
| 11 | `test_findings_are_finding_dataclasses` | `all(isinstance(f, Finding) for f in synthesis.findings)` |
| 12 | `test_citations_are_evidence_dataclasses` | `all(isinstance(e, Evidence) for e in synthesis.citations)` |
| 13 | `test_executive_summary_populated` | `synthesis.executive_summary != ""` |
| 14 | `test_topic_populated` | `synthesis.topic != ""` |
| 15 | `test_conflicting_claims_both_in_findings` | LLM returns 2 conflicting findings → both present in result |
| 16 | `test_invalid_evidence_ref_raises_agent_parse_error` | evidence_ref URL not in available_urls → `AgentParseError` |
| 17 | `test_empty_executive_summary_raises_agent_parse_error` | `executive_summary: ""` → `AgentParseError` |
| 18 | `test_no_findings_raises_agent_parse_error` | `findings: []` → `AgentParseError` |
| 19 | `test_finding_with_no_evidence_refs_raises` | finding with `evidence_refs: []` → `AgentParseError` |
| 20 | `test_citation_urls_mapped_to_evidence_objects` | `citation_urls` URL present in evidence → `Evidence` in `synthesis.citations` |
| 21 | `test_unknown_citation_url_silently_dropped` | citation URL not in any WorkerResult → excluded from citations |
| 22 | `test_retry_on_parse_failure` | LLM returns invalid JSON once → retried → valid on 2nd call |
| 23 | `test_raises_after_max_retries` | LLM always returns invalid JSON → `AgentParseError` after 3 attempts |
| 24 | `test_empty_compressed_worker_results_raises` | `compressed_worker_results == []` → `ValueError` |
| 25 | `test_build_messages_anthropic_has_cache_control` | `use_cache_control=True` → content blocks with `cache_control` |
| 26 | `test_build_messages_groq_no_cache_control` | `use_cache_control=False` → plain string `SystemMessage` |
| 27 | `test_load_prompt_missing_file_raises` | monkeypatch path → `FileNotFoundError` |

---

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| LLM returns invalid JSON | Hallucination, truncation | `parse_agent_json()` + tenacity retry (3×) |
| LLM cites URL not in evidence | Hallucination | `_parse_and_validate` raises `AgentParseError` → retry |
| LLM returns empty executive_summary | Bad instruction following | Validation in `_parse_and_validate` → retry |
| LLM returns 0 findings | Under-generation | Validation in `_parse_and_validate` → retry |
| LLM silently drops conflicting claim | Alignment / bias | Prompt instruction + reviewer (grounding_check + critic) catch this |
| All 3 retries fail | Systematic LLM failure | `reraise=True` propagates `AgentParseError` to caller |
| Prompt file missing | Deployment error | `FileNotFoundError` with clear path in message |
| `compressed_worker_results` empty | Bug in boundary_compressor or no evidence | `ValueError` before any LLM call |
| Evidence serialization overflow | Many workers, verbose evidence | `_serialize_evidence` hard cap at ~6000 chars |
| `prior_syntheses[-1]` missing on PATH 2 | Bug: revision_count > 0 but no prior synthesis | Guard: if `not state["prior_syntheses"]`, use empty prior_attempt_summary |

---

## Cost impact

**Per coordinator call (1 LLM call, ~3000–8000 input tokens):**

| Provider | Tokens (est.) | Cost |
|---|---|---|
| Groq (llama-3.1-8b) | ~5000 | **$0.00** (free tier) |
| Anthropic (claude-sonnet-4-6) | ~5000 input, ~800 output | **~$0.018** |
| OpenRouter (varies) | ~5000 | **~$0.005–0.020** |

**Per session (1–3 coordinator calls due to revision loop):**

| Provider | Max cost (3 calls) |
|---|---|
| Groq | ~$0.00 |
| Anthropic | ~$0.054 (~5.4% of $1.00 ceiling) |

Anthropic prompt caching saves ~40% on repeated first-pass blocks (cache hit on
`coordinator_v1.txt` template which never changes between calls).

---

## Critic scoring weights (for future `agents/critic.py` reference)

When implementing the critic node, use these weights for `final_quality_score`:

| Dimension | Field | Weight |
|---|---|---|
| Factuality | `factuality_score` | 30% |
| Citation alignment | `citation_alignment_score` | 25% |
| Reasoning | `reasoning_score` | 20% |
| Completeness | `completeness_score` | 15% |
| Bias | `bias_score` | 10% |

Formula: `final_quality_score = 0.30*F + 0.25*C + 0.20*R + 0.15*Co + 0.10*B`

Pass threshold: `final_quality_score >= 4.0` (enforced in `CriticOutput.__post_init__`).

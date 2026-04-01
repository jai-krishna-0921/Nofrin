# Plan: agents/critic.py

> **Context:** coordinator + grounding_check done (122/122 tests). Next: critic → delivery.

## Problem

The critic node is the adversarial reviewer in the revision loop. It reads the
`SynthesisOutput` produced by coordinator, uses `grounding_issues` from grounding_check
as pre-computed context, and scores the synthesis across 5 dimensions. The score drives
the routing decision: pass to delivery or route back to coordinator for revision.

Key design constraints:
1. `CriticOutput` is already fully typed in `graph/state.py` — implement to that schema exactly.
2. `final_quality_score` and `passed` are computed in code — NOT trusted from LLM output.
3. `route_after_critic` in `graph/router.py` is already complete — critic node only writes
   `critic_output` to state; routing happens downstream.
4. CLAUDE.md: revision cap = 2. After 2 revisions, router force-delivers; critic does NOT
   inject the caveat (that is `budget_gate_node`'s pattern — handled by routing layer).
5. Uses `llm_critic` (already in NofrinContext — the most capable model).

---

## Answers to planning questions

### Q1: Five evaluation dimensions, weights, range, threshold

| Dimension | Field | Weight | Score range |
|---|---|---|---|
| Factuality | `factuality_score` | 30% | 0.0–5.0 |
| Citation alignment | `citation_alignment_score` | 25% | 0.0–5.0 |
| Reasoning quality | `reasoning_score` | 20% | 0.0–5.0 |
| Completeness | `completeness_score` | 15% | 0.0–5.0 |
| Bias / balance | `bias_score` | 10% | 0.0–5.0 |

Formula: `final_quality_score = 0.30*F + 0.25*C + 0.20*R + 0.15*Co + 0.10*B`

Pass threshold: `final_quality_score >= 4.0` (enforced in `CriticOutput.__post_init__`).

The LLM is instructed to return the 5 raw dimension scores only. Code computes
`final_quality_score` and sets `passed` via `__post_init__` — LLM values for these
fields are overridden and never trusted.

### Q2: grounding_issues consumption

- Critic reads `state["grounding_issues"]` (list[str]) as input context.
- Presented to the LLM as a dedicated block: `KNOWN GROUNDING ISSUES:\n{issues}`
- The critic does NOT re-verify grounding — it uses these as pre-computed facts that
  inform its scoring (especially factuality and citation_alignment dimensions).
- If `grounding_issues` is empty, the block is replaced with "(none — grounding check passed clean)".

### Q3: CriticOutput structure (from state.py)

```python
@dataclass
class CriticOutput:
    factuality_score: float          # LLM-provided, 0.0–5.0
    citation_alignment_score: float  # LLM-provided, 0.0–5.0
    reasoning_score: float           # LLM-provided, 0.0–5.0
    completeness_score: float        # LLM-provided, 0.0–5.0
    bias_score: float                # LLM-provided, 0.0–5.0
    final_quality_score: float       # CODE-COMPUTED — not trusted from LLM
    issues: list[CriticIssue]        # specific problems identified
    suggestions: list[CriticSuggestion]  # actionable improvements
    passed: bool                     # __post_init__ sets this from final_quality_score >= 4.0
```

`CriticIssue`: `{issue_text: str, quote_from_synthesis: str, severity: IssueSeverity}`
`CriticSuggestion`: `{action: str, target_section: str, new_evidence_needed: bool}`
`IssueSeverity`: `Literal["critical", "major", "minor"]`

### Q4: State fields written

| Field | Written by critic | Notes |
|---|---|---|
| `critic_output` | ✅ always | Full CriticOutput with computed scores |
| `total_tokens_used` | ✅ accumulated | `state value + tokens_from_llm` |
| `grounding_issues` | ❌ not touched | Already written by grounding_check |
| `revision_count` | ❌ not touched | Incremented by coordinator on PATH 2 |
| `synthesis` | ❌ not touched | Modified by coordinator only |

### Q5: Routing decision

`route_after_critic` in `graph/router.py` is **already fully implemented** — no changes needed.

Priority order (already coded):
1. `cost_usd > 0.75 * cost_ceiling_usd` → `"delivery"` (budget exhausted)
2. `critic_output.passed == True` → `"delivery"` (quality threshold met)
3. `revision_count >= 2` → `"delivery"` (hard cap; router forces delivery, quality caveat
   already injected via `budget_gate_node` pattern if applicable)
4. else → `"coordinator"` (trigger revision pass)

### Q6: Revision count cap

The cap is enforced entirely by `route_after_critic` checking `revision_count >= _MAX_REVISIONS`.
The critic node itself does NOT modify `revision_count` or inject quality caveats.
Coordinator (PATH 2) increments `revision_count` by 1.
If `revision_count == 2` and critic fails again, `route_after_critic` routes to "delivery"
without a third coordinator call. No force-delivery caveat is written by the critic — that
would require a separate node (out of scope for this plan).

### Q7: route_after_critic changes needed

**None.** The existing implementation is complete and correct. The new builder wiring must
use `add_conditional_edges("budget_gate", route_after_critic, ...)` — that is a builder
change, not a router change.

### Q8: Token cost

Typical call (synthesis with 3 findings, 2 grounding issues):

| Token category | Estimate |
|---|---|
| Prompt template + rubric | ~600 tokens |
| Synthesis: exec summary + 3 findings (body ~150 words each) | ~500 tokens |
| Grounding issues (2 strings) | ~80 tokens |
| Input total | **~1180 tokens** |
| Output (5 scores + 2 issues + 2 suggestions) | **~350 tokens** |
| **Total per call** | **~1530 tokens** |

| Provider | Cost per call |
|---|---|
| Groq (llama-3.1-8b) | **~$0.00** (free tier) |
| Anthropic (claude-sonnet-4-6) | **~$0.005** |
| OpenRouter (varies) | **~$0.001–$0.005** |

### Q9: Updated graph wiring

```
grounding_check → critic → budget_gate → conditional(route_after_critic)
                                              ├── "delivery" → END (stub until delivery built)
                                              └── "coordinator" (revision loop)
```

New `add_node` calls needed:
```python
builder.add_node("critic", critic_node)  # type: ignore[arg-type]
```

New edges:
```python
builder.add_edge("grounding_check", "critic")
builder.add_edge("critic", "budget_gate")
builder.add_conditional_edges(
    "budget_gate",
    route_after_critic,
    {"delivery": END, "coordinator": "coordinator"},
)
```

Remove: `builder.add_edge("grounding_check", END)` (current stub).
The "delivery" branch routes to END until delivery node is built.

---

## Files to create / modify

| Path | Action |
|---|---|
| `agents/critic.py` | **Create** |
| `prompts/critic_v1.txt` | **Create** — full text below |
| `prompts/CHANGELOG.md` | **Update** |
| `agents/__init__.py` | **Update** — export `critic_node` |
| `graph/builder.py` | **Update** — wire grounding_check → critic → budget_gate → conditional |
| `graph/router.py` | **No change** — already complete |
| `tests/test_critic.py` | **Create** — 20+ tests |

---

## Prompt file (full text)

### `prompts/critic_v1.txt`

```
You are an adversarial research reviewer. Your task is to critically evaluate a research brief for factual accuracy, citation quality, reasoning soundness, completeness, and bias.

SYNTHESIS TO REVIEW:
{{synthesis_block}}

KNOWN GROUNDING ISSUES (pre-checked by automated fact-checker):
{{grounding_issues_block}}

SCORING RUBRIC:
Score each dimension from 0.0 to 5.0 (not an integer — use decimals):

1. FACTUALITY (weight 30%): Are all factual claims accurate and traceable to the cited evidence? Deduct for claims that go beyond what the evidence supports. 5.0 = all claims verified; 0.0 = pervasive fabrication.

2. CITATION_ALIGNMENT (weight 25%): Do the cited URLs match the claims they support? Deduct for citations that exist but don't support the specific claim made. Consider the KNOWN GROUNDING ISSUES above. 5.0 = perfect alignment; 0.0 = systematic hallucination.

3. REASONING (weight 20%): Are the logical inferences from the evidence sound? Deduct for non-sequiturs, overgeneralizations, and unsupported causal claims. 5.0 = rigorous reasoning; 0.0 = fundamental logical errors.

4. COMPLETENESS (weight 15%): Does the synthesis address the apparent research scope adequately? Deduct for significant omissions or gaps not acknowledged. 5.0 = comprehensive; 0.0 = critically incomplete.

5. BIAS (weight 10%): Is the synthesis balanced? Deduct for selective use of evidence, framing that favors one position without acknowledging counterevidence, or missing "CONFLICT RULE" violations. 5.0 = balanced; 0.0 = highly biased.

TASK:
1. Score each of the 5 dimensions honestly. Be strict — a score of 4.0+ means the brief is ready for delivery.
2. List specific issues (things that MUST be fixed in a revision). Only list issues worth a revision — trivial wording is not an issue.
3. List actionable suggestions tied to specific sections.

Return ONLY valid JSON — no preamble, no markdown fences, no explanation:

{
  "factuality_score": 4.2,
  "citation_alignment_score": 3.8,
  "reasoning_score": 4.5,
  "completeness_score": 3.5,
  "bias_score": 4.0,
  "issues": [
    {
      "issue_text": "The claim that AUC >0.97 is not present in any cited source",
      "quote_from_synthesis": "AI models achieve AUC >0.97 for pneumonia detection",
      "severity": "critical"
    }
  ],
  "suggestions": [
    {
      "action": "Replace AUC claim with the correct threshold from the NEJM source",
      "target_section": "AI Matches Radiologist Accuracy in Chest X-Rays",
      "new_evidence_needed": false
    }
  ]
}

severity must be exactly "critical", "major", or "minor".
new_evidence_needed must be a boolean (true/false).
```

---

## TypedDicts for two-level deserialization

```python
class _RawCriticIssue(TypedDict):
    issue_text: str
    quote_from_synthesis: str
    severity: str  # validated against IssueSeverity literals

class _RawCriticSuggestion(TypedDict):
    action: str
    target_section: str
    new_evidence_needed: bool

class _RawCriticOutput(TypedDict):
    factuality_score: float
    citation_alignment_score: float
    reasoning_score: float
    completeness_score: float
    bias_score: float
    issues: list[_RawCriticIssue]
    suggestions: list[_RawCriticSuggestion]
```

---

## Code outline (signatures + docstrings only)

```python
"""
agents/critic.py

Critic node: adversarial 5-dimension evaluation of the synthesis brief.

Reads:  state["synthesis"], state["grounding_issues"]
Writes: state["critic_output"], state["total_tokens_used"]

Must run AFTER grounding_check (CLAUDE.md rule).
Does NOT modify synthesis, revision_count, or grounding_issues.
Routing to delivery or coordinator is handled by route_after_critic in graph/router.py.
"""

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic_v1.txt"
_SYNTHESIS_BLOCK_CAP = 6000  # chars
_VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "major", "minor"})


def _is_anthropic(llm: BaseChatModel) -> bool: ...

def _load_prompt(path: Path) -> str: ...

def _serialize_synthesis(synthesis: SynthesisOutput) -> str:
    """Serialize SynthesisOutput for the critic prompt.

    Includes: topic, executive_summary, all findings (heading + body + evidence_refs),
    risks, gaps. Caps at ~_SYNTHESIS_BLOCK_CAP chars per-item boundary.
    """
    ...

def _serialize_grounding_issues(issues: list[str]) -> str:
    """Return grounding issues as a numbered list, or
    '(none — grounding check passed clean)' if empty.
    """
    ...

def _build_messages(
    synthesis_block: str,
    grounding_issues_block: str,
    prompt_template: str,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    """Build messages for the critic LLM call.

    use_cache_control=True (Anthropic): static prompt + rubric cached;
      synthesis_block + grounding_issues_block in uncached dynamic block.
    use_cache_control=False: plain string SystemMessage.
    """
    ...

def _compute_final_score(raw: _RawCriticOutput) -> float:
    """Compute weighted final_quality_score from 5 raw dimension scores.

    Formula: 0.30*F + 0.25*C + 0.20*R + 0.15*Co + 0.10*B
    Clamps each dimension to [0.0, 5.0] before weighting.
    Does NOT use the LLM's own final_quality_score if present.
    """
    ...

def _parse_critic_output(
    raw: str,
    computed_final_score: float | None = None,  # pass None to compute internally
) -> CriticOutput:
    """Parse LLM JSON and build a typed CriticOutput.

    Steps:
      1. parse_agent_json(raw, _RawCriticOutput)
      2. Clamp all 5 dimension scores to [0.0, 5.0]
      3. Validate each issue.severity against _VALID_SEVERITIES
      4. Validate issues: issue_text and quote_from_synthesis not empty
      5. Convert _RawCriticIssue → CriticIssue dataclass
      6. Convert _RawCriticSuggestion → CriticSuggestion dataclass
      7. Compute final_quality_score via _compute_final_score()
      8. Build and return CriticOutput(... final_quality_score=computed, passed=<from __post_init__>)

    Raises:
        AgentParseError: If JSON malformed or validation fails.
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
) -> tuple[CriticOutput, int]:
    """Call the LLM and parse/validate the critic response.

    Returns: (CriticOutput, tokens_used).
    """
    ...

async def critic_node(
    state: ResearchAgentState,
    runtime: Runtime[NofrinContext],
) -> dict[str, object]:
    """LangGraph node: adversarial 5-dimension review of the synthesis.

    Reads state["synthesis"] and state["grounding_issues"].
    Calls llm_critic (most capable model) for scoring and issue identification.
    Computes final_quality_score in code — never trusts LLM's own computation.
    Always writes state["critic_output"].

    Args:
        state: ResearchAgentState. Reads: synthesis, grounding_issues, total_tokens_used.
        runtime: Injected NofrinContext. Uses llm_critic.

    Returns:
        {"critic_output": CriticOutput, "total_tokens_used": int}

    Raises:
        ValueError: If synthesis is None.
        AgentParseError: After 3 failed LLM/parse attempts.
    """
    ...
```

---

## JSON contract

**LLM must return (5 scores, issues, suggestions — no final_quality_score, no passed):**
```json
{
  "factuality_score": 4.2,
  "citation_alignment_score": 3.8,
  "reasoning_score": 4.5,
  "completeness_score": 3.5,
  "bias_score": 4.0,
  "issues": [
    {
      "issue_text": "The claim that AUC >0.97 is not present in any cited source",
      "quote_from_synthesis": "AI models achieve AUC >0.97 for pneumonia detection",
      "severity": "critical"
    }
  ],
  "suggestions": [
    {
      "action": "Replace AUC claim with the correct threshold from the NEJM source",
      "target_section": "AI Matches Radiologist Accuracy in Chest X-Rays",
      "new_evidence_needed": false
    }
  ]
}
```

**Code computes:**
`final_quality_score = 0.30*4.2 + 0.25*3.8 + 0.20*4.5 + 0.15*3.5 + 0.10*4.0 = 4.075`
`passed = (4.075 >= 4.0) = True`

---

## State fields written

| Field | Type | Always written |
|---|---|---|
| `critic_output` | `CriticOutput` | ✅ yes |
| `total_tokens_used` | `int` | ✅ accumulated |

---

## Tests — `tests/test_critic.py` (20 cases)

| # | Test | Assertion |
|---|---|---|
| 1 | `test_critic_node_returns_critic_output` | result contains "critic_output" key as CriticOutput |
| 2 | `test_final_score_computed_not_from_llm` | LLM JSON has no `final_quality_score`; code computes it from the 5 scores |
| 3 | `test_weighted_formula_correct` | scores (5,5,5,5,5) → final=5.0; scores (4,4,4,4,4) → final=4.0 |
| 4 | `test_passed_true_when_score_gte_4` | final=4.075 → critic_output.passed == True |
| 5 | `test_passed_false_when_score_lt_4` | final=3.5 → critic_output.passed == False |
| 6 | `test_dimension_scores_clamped_to_0_5` | LLM returns score=6.0 → clamped to 5.0; score=-1.0 → 0.0 |
| 7 | `test_critic_issues_are_criticissue_dataclasses` | all(isinstance(i, CriticIssue) for i in critic_output.issues) |
| 8 | `test_critic_suggestions_are_criticsuggesion_dataclasses` | all(isinstance(s, CriticSuggestion) for s in critic_output.suggestions) |
| 9 | `test_invalid_severity_raises_agent_parse_error` | issue.severity="INVALID" → AgentParseError |
| 10 | `test_empty_issue_text_raises_agent_parse_error` | issue_text="" → AgentParseError |
| 11 | `test_empty_quote_raises_agent_parse_error` | quote_from_synthesis="" → AgentParseError |
| 12 | `test_synthesis_none_raises_value_error` | synthesis=None → ValueError before LLM call |
| 13 | `test_grounding_issues_appear_in_serialized_context` | non-empty grounding_issues → block in message text |
| 14 | `test_empty_grounding_issues_shows_clean_message` | grounding_issues=[] → "(none — grounding check passed clean)" in message |
| 15 | `test_build_messages_anthropic_cache_control` | use_cache_control=True → list content with cache_control block |
| 16 | `test_build_messages_groq_no_cache_control` | use_cache_control=False → plain str SystemMessage |
| 17 | `test_retry_on_parse_failure` | LLM returns bad JSON once → retried → valid on 2nd call (2 ainvoke calls) |
| 18 | `test_raises_after_max_retries` | LLM always returns bad JSON → AgentParseError after 3 attempts |
| 19 | `test_total_tokens_accumulated` | state has 200 tokens; LLM response has 50 tokens → result is 250 |
| 20 | `test_load_prompt_missing_file_raises` | _load_prompt(nonexistent_path) → FileNotFoundError |

---

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| `synthesis is None` | Bug: critic called before coordinator | `ValueError` before any LLM call |
| LLM returns final_quality_score / passed | LLM follows prompt poorly | Both fields computed in code; LLM values silently overridden by `__post_init__` |
| LLM returns score > 5.0 or < 0.0 | Hallucination | `_compute_final_score` clamps each dimension to [0.0, 5.0] |
| Invalid severity string | LLM invents severity level | Whitelist check in `_parse_critic_output` → AgentParseError → retry |
| Empty issue_text or quote | Malformed partial response | Validated in `_parse_critic_output` → AgentParseError → retry |
| Synthesis too long for context | Many large findings | `_serialize_synthesis` caps at _SYNTHESIS_BLOCK_CAP chars per-item |
| All 3 retries fail | Systematic LLM failure | `reraise=True` propagates `AgentParseError` |
| Prompt file missing | Deployment error | `FileNotFoundError` with clear message |
| issues=[] when there are real problems | LLM too lenient | Downstream revision loop limited to 2 cycles; acceptable risk |

---

## Architect notes

1. **No route_after_critic changes needed.** The function already handles all 4 priority cases.

2. **builder.py conditional edges** require the "delivery" stub to route to END until the
   delivery node is implemented. Use `{"delivery": END, "coordinator": "coordinator"}` as
   the path map in `add_conditional_edges`.

3. **`_RawCriticOutput` intentionally omits `final_quality_score` and `passed`** — the LLM
   does not return these, and even if it did, they'd be overridden by `__post_init__`.
   If the LLM includes extra keys, `TypedDict(**data)` silently accepts them (they're just dicts).

4. **Score computation separation**: `_compute_final_score` is a pure function (no LLM call,
   no side effects) — easy to unit test independently without mocking.

5. **grounding_issues informs scoring**: the prompt explicitly tells the LLM to use grounding
   issues when scoring citation_alignment. This is the primary linkage between grounding_check
   and critic — no direct code dependency, only via state.

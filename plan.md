# Plan: agents/delivery.py

> **Context:** coordinator + grounding_check + critic done (142/142 tests). Next: delivery (final node before END).

## Problem Statement

The delivery node is the terminal formatting node in the Deep Research Agent pipeline. It transforms the structured `SynthesisOutput` (and optional `CriticOutput`) into a human-readable document in the requested output format. Unlike other agent nodes, delivery performs **pure formatting logic with zero LLM calls**. It must detect the force-delivery case (`revision_count >= 2`, `critic_output.passed == False`) and inject a quality caveat at the top of the output. The node reads `output_format` from state and dispatches to the appropriate renderer (markdown only for now; docx/pdf/pptx are TODO stubs that raise `NotImplementedError`).

---

## Answers to Planning Questions

### Q1: What does delivery read from state?

| Field | Read | Why |
|---|---|---|
| `synthesis` | ✅ | Primary content: topic, executive_summary, findings, risks, gaps, citations |
| `critic_output` | ✅ | Quality badge: `critic_output.passed` and `final_quality_score` |
| `output_format` | ✅ | Dispatches to correct renderer |
| `revision_count` | ✅ | Detects force-delivery case |
| `cost_usd` | ❌ | Not user-facing |
| `final_output` | ❌ | This is what we WRITE, not read |
| `grounding_issues` | ❌ | Already incorporated into critic scoring |

### Q2: What format(s) does it produce?

- `markdown`: Fully implemented via `render_markdown()` pure function.
- `docx`, `pdf`, `pptx`: Raise `NotImplementedError("Format not yet implemented")` stubs.

**Why markdown only:** Other formats require external dependencies (python-docx, WeasyPrint, python-pptx) not yet integrated. Markdown is the dev-default format and serves as the reference renderer.

### Q3: State write

Returns `{"final_output": str}` — LangGraph merges this partial update into state.

### Q4: Force-delivery case

**Detection:**
```python
is_force_delivered = (
    state["revision_count"] >= _MAX_REVISIONS
    and state["critic_output"] is not None
    and not state["critic_output"].passed
)
```

**Caveat text (exact):**
```
> **QUALITY NOTICE:** This research brief was delivered after reaching the maximum revision limit (2 revisions) without meeting the quality threshold (score: {score:.2f}/5.0). Review with appropriate scrutiny.
```

Injected at the very top of the markdown output, before the title.

### Q5: LLM call?

**None.** Zero LLM calls, zero token cost. No prompt file needed. No runtime context needed.

### Q6: Graph wiring change in builder.py

```python
# Add import:
from agents.delivery import delivery_node

# Add node:
builder.add_node("delivery", delivery_node)

# Update conditional edges (replace END stub):
builder.add_conditional_edges(
    "budget_gate",
    route_after_critic,
    {"delivery": "delivery", "coordinator": "coordinator"},
)

# Add terminal edge:
builder.add_edge("delivery", END)
```

Remove: the `{"delivery": END, ...}` stub in the current conditional edges map.

### Q7: Estimated token cost

**$0.00** — pure formatting.

### Q8: Tests

24 test cases — see table below.

---

## Files to Create / Modify

| Path | Action |
|---|---|
| `agents/delivery.py` | **Create** |
| `agents/__init__.py` | **Update** — export `delivery_node` |
| `graph/builder.py` | **Update** — wire delivery node, update conditional edges |
| `tests/test_delivery.py` | **Create** — 24 tests |

---

## Function Signatures (no implementation bodies)

```python
"""
agents/delivery.py

Delivery node: format SynthesisOutput into the requested output format.

Reads:  state["synthesis"], state["critic_output"], state["output_format"],
        state["revision_count"]
Writes: state["final_output"]

Pure formatting logic — no LLM calls, no token cost.
No prompt file needed.
"""

_QUALITY_CAVEAT_TEMPLATE: str = (
    "> **QUALITY NOTICE:** This research brief was delivered after reaching "
    "the maximum revision limit (2 revisions) without meeting the quality "
    "threshold (score: {score:.2f}/5.0). Review with appropriate scrutiny.\n"
)

_MAX_REVISIONS: int = 2  # must match graph/router.py._MAX_REVISIONS


def _is_force_delivered(state: ResearchAgentState) -> bool:
    """Return True if revision_count >= 2 AND critic_output.passed == False."""
    ...


def _build_citation_map(citations: list[Evidence]) -> dict[str, int]:
    """Build URL → 1-based footnote index mapping from synthesis.citations."""
    ...


def _render_finding(finding: Finding, citation_map: dict[str, int]) -> str:
    """Render a single finding as markdown (### heading, body, [^N] refs)."""
    ...


def _render_citations_section(citations: list[Evidence]) -> str:
    """Render the References section with footnote definitions.

    Format: [^1]: [Source Title](URL) - Published: YYYY-MM-DD
    Returns empty string if citations is empty.
    """
    ...


def _render_quality_badge(
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """Render the quality badge line.

    Outputs:
      "**Quality: PASS** (4.25/5.0)"    — when critic_output.passed
      "**Quality: REVIEW RECOMMENDED** (3.50/5.0)"  — when force-delivered
      "**Quality: N/A**"                 — when critic_output is None
    """
    ...


def render_markdown(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    """Render SynthesisOutput as a complete markdown document.

    Pure function: no side effects, no LLM calls, deterministic output.

    Section order:
        1. Quality caveat (ONLY if force-delivered)
        2. # {topic}
        3. Quality badge line
        4. ## Executive Summary
        5. ## Key Findings (### per finding, [^N] footnote refs)
        6. ## Risks  (omit if empty)
        7. ## Known Gaps  (omit if empty)
        8. ## References (footnote definitions, omit if no citations)
    """
    ...


async def delivery_node(
    state: ResearchAgentState,
) -> dict[str, str]:
    """LangGraph node: format synthesis into the requested output format.

    Note: no Runtime parameter — no LLM client needed.

    Raises:
        ValueError: If synthesis is None.
        NotImplementedError: If output_format is "docx", "pdf", or "pptx".
    """
    ...
```

---

## render_markdown() Output Specification

### Section order (exact)

```
1. [Quality caveat — only if force-delivered]
2. # {topic}
3. {quality_badge}
4. ## Executive Summary
   {executive_summary}
5. ## Key Findings
   ### {finding.heading}
   {finding.body}[^i][^j]
6. ## Risks         [omit if empty]
   - {risk}
7. ## Known Gaps    [omit if empty]
   - {gap}
8. ## References    [omit if no citations]
   [^1]: [Title](URL) - Published: date
```

### Example (force-delivered, score 3.50)

```markdown
> **QUALITY NOTICE:** This research brief was delivered after reaching the maximum revision limit (2 revisions) without meeting the quality threshold (score: 3.50/5.0). Review with appropriate scrutiny.

# Renewable Energy Investment Trends 2025

**Quality: REVIEW RECOMMENDED** (3.50/5.0)

## Executive Summary

Global renewable energy investments reached $500 billion in 2024...

## Key Findings

### Solar Costs Continue to Decline

The levelized cost of solar energy dropped 12% year-over-year.[^1][^2]

## Risks

- Supply chain constraints may slow deployment in 2025

## Known Gaps

- Limited data on emerging markets

## References

[^1]: [IEA World Energy Outlook 2024](https://iea.org/reports/weo-2024) - Published: 2024-10-15
[^2]: [BloombergNEF Solar Report](https://about.bnef.com/solar) - Published: 2024-09-20
```

### Example (passed, score 4.25)

```markdown
# Renewable Energy Investment Trends 2025

**Quality: PASS** (4.25/5.0)

## Executive Summary
...
```

---

## Test Cases (24)

| # | Test | Assertion |
|---|---|---|
| 1 | `test_delivery_node_returns_final_output_key` | `"final_output" in result` |
| 2 | `test_final_output_is_string` | `isinstance(result["final_output"], str)` |
| 3 | `test_markdown_contains_topic_as_title` | `"# Test Topic"` in output |
| 4 | `test_markdown_contains_executive_summary_heading` | `"## Executive Summary"` in output |
| 5 | `test_markdown_contains_all_finding_headings` | All finding headings as `###` in output |
| 6 | `test_findings_have_footnote_refs` | `[^1]` appears in finding body |
| 7 | `test_risks_section_present_when_nonempty` | `"## Risks"` in output |
| 8 | `test_gaps_section_present_when_nonempty` | `"## Known Gaps"` in output |
| 9 | `test_citations_section_present` | `"## References"` and `[^1]:` in output |
| 10 | `test_quality_badge_pass` | `"Quality: PASS"` in output when `critic_output.passed=True` |
| 11 | `test_quality_badge_review_recommended` | `"Quality: REVIEW RECOMMENDED"` when force-delivered |
| 12 | `test_force_delivery_caveat_injected_at_top` | `"QUALITY NOTICE:"` is first non-empty line |
| 13 | `test_force_delivery_caveat_contains_score` | `"3.50/5.0"` in caveat string |
| 14 | `test_no_caveat_when_critic_passed` | `"QUALITY NOTICE"` not in output |
| 15 | `test_no_caveat_when_revision_under_cap` | `revision_count=1, passed=False` → no caveat |
| 16 | `test_synthesis_none_raises_value_error` | `ValueError` before any rendering |
| 17 | `test_output_format_markdown_produces_output` | `len(result["final_output"]) > 0` |
| 18 | `test_output_format_docx_raises_not_implemented` | `NotImplementedError` |
| 19 | `test_output_format_pdf_raises_not_implemented` | `NotImplementedError` |
| 20 | `test_output_format_pptx_raises_not_implemented` | `NotImplementedError` |
| 21 | `test_render_markdown_is_pure` | Same args → identical output on two calls |
| 22 | `test_citations_format_url_and_title` | `[Title](URL)` pattern in References |
| 23 | `test_empty_risks_omits_section` | `"## Risks"` NOT in output when `risks=[]` |
| 24 | `test_empty_gaps_omits_section` | `"## Known Gaps"` NOT in output when `gaps=[]` |

---

## Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| `synthesis is None` | delivery called before coordinator | `ValueError` |
| `critic_output is None` | edge case — critic skipped | Badge shows `N/A` |
| Unsupported `output_format` | docx/pdf/pptx request | `NotImplementedError` |
| Empty `findings` | coordinator edge case | `## Key Findings` section renders empty |
| URL not in `citation_map` | evidence_ref not in citations | Append URL inline without footnote |
| `revision_count > _MAX_REVISIONS` | router bug | `>= 2` check still fires correctly |

---

## Architect Notes

1. **No `runtime` parameter** — same pattern as `boundary_compressor_node`. Signature is `async def delivery_node(state: ResearchAgentState) -> dict[str, str]`.

2. **`_MAX_REVISIONS = 2`** — must be kept in sync with `graph/router.py._MAX_REVISIONS`. Do not import from router (circular import risk); declare as a local constant.

3. **`render_markdown` exported** — for direct testing and future reuse by docx/pptx renderers as a reference.

4. **Section blank lines** — separate each section with a double newline (`\n\n`) for valid markdown rendering.

5. **Footnote refs in findings** — only include `[^N]` for URLs present in `citation_map`; silently skip unknown refs.

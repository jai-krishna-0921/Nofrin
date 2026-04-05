# Prompt Changelog

> Every prompt change is a deployment. Log it here with: version, date, what changed, why, eval delta.

---

## supervisor_v1.txt
- **Date**: 2026-03-29
- **What**: Initial intent classification + query decomposition. Two-step prompt: classify intent (4 types with examples), then generate 3-5 sub-queries with source routing (web/academic/news).
- **Why**: Baseline — first implementation
- **Eval delta**: N/A (baseline)

## worker_v1.txt
- **Date**: 2026-03-29
- **What**: Evidence extraction prompt. LLM extracts one claim + supporting chunks per Exa result. Returns confidence (0–1) and contradiction_score (0–1). Empty claim sentinel for "no relevant evidence" case.
- **Why**: Baseline — first implementation
- **Eval delta**: N/A (baseline)

## coordinator_v1.txt
- **Date**: 2026-03-30
- **What**: First-pass synthesis. Conflict rule (both positions), citation rule (evidence_refs required per finding), gaps rule (acknowledge unknowns), scope rule (no hallucination). Returns structured JSON: topic, executive_summary, findings[], risks[], gaps[], citation_urls[].
- **Why**: Baseline — first implementation
- **Eval delta**: N/A (baseline)

## coordinator_revision_v1.txt
- **Date**: 2026-03-30
- **What**: Revision-pass synthesis. Adds PRIOR SYNTHESIS and CRITIC ISSUES blocks. Verbatim "do not restate" constraint: each critic issue must be addressed with new evidence or added to gaps[] as "Unresolved: [issue]". VERSION RULE: only change what critic flagged.
- **Why**: Without explicit "do not restate" instruction, revision loop produces expensive paraphrases instead of targeted fixes
- **Eval delta**: N/A (baseline)

## critic_v1.txt
- **Date**: 2026-04-01
- **What**: Adversarial 5-dimension evaluation prompt. Scores factuality (30%), citation alignment (25%), reasoning (20%), completeness (15%), bias (10%) on a 0.0–5.0 scale. Returns issues[] with mandatory quote_from_synthesis + severity, and suggestions[] with new_evidence_needed flag. final_quality_score and passed are computed in code — LLM returns raw scores only.
- **Why**: Baseline — first implementation
- **Eval delta**: N/A (baseline)

---

## grounding_check_v1.txt
- **Date**: 2026-03-31
- **What**: Fact-checker prompt validating synthesis findings against cited evidence. Detects three issue types: UNSUPPORTED (claim absent from evidence), HALLUCINATED_CITATION (URL present but unrelated), MISSING_CITATION (claim exists in evidence but no ref). Returns structured JSON `{issues:[{type,finding_heading,description}]}`. Empty array if clean.
- **Why**: Baseline — first implementation
- **Eval delta**: N/A (baseline)

## coordinator_revision_v2.txt
- **Date**: 2026-04-05
- **What**: Added `Revision {{revision_count}} of 2` counter + final-pass directive at top: "If this is revision 2, resolve ALL remaining issues — do not defer to gaps[] unless the evidence genuinely does not support resolution." `coordinator.py` updated: `_build_revision_messages` gains `revision_count: int` parameter; `.replace("{{revision_count}}", str(revision_count))` added to filled chain and cache_control branch; call site updated; `REVISION_PROMPT_PATH` points to v2.
- **Why**: LLM had no visibility into revision budget; on the final pass it was deferring resolvable issues to gaps[] unnecessarily because it didn't know it was the last chance to fix them.
- **Eval delta**: TBD (run `pytest eval/` after baseline is measured)

---

## Template for new entries

```
## <agent>_v<N>.txt
- **Date**: YYYY-MM-DD
- **What**: <what changed>
- **Why**: <what failure mode or eval gap triggered this>
- **Eval delta**: faithfulness +0.0x, relevancy +0.0x (run pytest eval/ to fill in)
```

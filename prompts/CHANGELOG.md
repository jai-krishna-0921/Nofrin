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
- **Date**: project start
- **What**: First-pass synthesis with citation requirements
- **Why**: Baseline
- **Eval delta**: N/A (baseline)

## coordinator_revision_v1.txt
- **Date**: project start
- **What**: Revision-pass synthesis with explicit "do not restate" instruction
- **Why**: Without this, revision loop produces expensive paraphrases not fixes
- **Eval delta**: N/A (baseline)

## critic_v1.txt
- **Date**: project start
- **What**: Adversarial 5-dimension evaluation with specific quote requirements
- **Why**: Baseline
- **Eval delta**: N/A (baseline)

---

## Template for new entries

```
## <agent>_v<N>.txt
- **Date**: YYYY-MM-DD
- **What**: <what changed>
- **Why**: <what failure mode or eval gap triggered this>
- **Eval delta**: faithfulness +0.0x, relevancy +0.0x (run pytest eval/ to fill in)
```

# Prompt Changelog

> Every prompt change is a deployment. Log it here with: version, date, what changed, why, eval delta.

---

## supervisor_v1.txt
- **Date**: project start
- **What**: Initial intent classification + query decomposition
- **Why**: Baseline
- **Eval delta**: N/A (baseline)

## worker_v1.txt
- **Date**: project start
- **What**: Evidence extraction with structured JSON output
- **Why**: Baseline
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

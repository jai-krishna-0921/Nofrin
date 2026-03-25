---
description: Review and improve an agent prompt. Usage: /review-prompt <agent-name> e.g. /review-prompt critic
---

Review and improve the prompt for agent: $ARGUMENTS

Steps:
1. Read `prompts/$ARGUMENTS_v*.txt` (latest version)
2. Read the corresponding agent node in `agents/$ARGUMENTS.py` 
3. Read `eval/dataset/` — understand what good output looks like
4. Run `pytest eval/ -v` and check the current eval scores in `eval/baseline.json`

Critique the prompt on:
- **Specificity**: Are instructions concrete or vague?
- **Schema compliance**: Does it tell the model exactly what JSON to return?
- **Adversarial gaps**: For the critic prompt — is it genuinely adversarial or too polite?
- **Revision instruction**: For the coordinator revision prompt — does it say "do not restate"?
- **Token efficiency**: Are there redundant instructions that could be removed?

Propose the improved prompt. Show a diff vs the current version.
After I approve: create `prompts/$ARGUMENTS_v<N+1>.txt` and update `prompts/CHANGELOG.md`.
Then run `pytest eval/` to confirm no regression before finalising.

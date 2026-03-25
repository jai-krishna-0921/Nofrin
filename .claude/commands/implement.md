---
description: Implement from an approved plan.md. Usage: /implement — run after /plan and approval
---

Implement the plan in `plan.md`. Follow these steps exactly:

1. **Re-read plan.md** — confirm you have the latest version
2. **Re-read CLAUDE.md** — confirm all rules are loaded
3. **Implement** in this order:
   a. State changes in graph/state.py first (if any)
   b. New utility functions in graph/utils.py (if any)
   c. Agent node in agents/ 
   d. Prompt file in prompts/ (versioned .txt)
   e. Graph wiring in graph/builder.py
   f. Router logic in graph/router.py (if any)
4. **After each file**, run: `mypy <file> --strict` and fix any errors before moving on
5. **After all files**: run `ruff check . && ruff format .`
6. **Invoke test-writer subagent** to write tests for the new node
7. **Run tests**: `pytest tests/ -v`
8. **Run eval**: `pytest eval/ -v` — confirm no regression
9. **Update memory.md** with what changed and why
10. **Invoke critic-reviewer subagent** on all modified files

Do not commit until steps 7-10 all pass.

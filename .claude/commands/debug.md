---
description: Debug a failing run using LangSmith traces. Usage: /debug <session-id or error description>
---

Debug issue: $ARGUMENTS

Steps:
1. If a session ID is provided, look for trace files in `.langsmith_traces/` or ask me to paste the relevant trace
2. Read the full error message and traceback
3. Use sequential-thinking MCP to reason through the failure systematically:
   - Which node failed?
   - What was the input state to that node?
   - What was the expected output schema?
   - What did the model actually return?
   - Is this a JSON parse failure, a schema mismatch, a cost ceiling breach, or a worker failure?

Common failure modes for this project:
- **JSON parse failure**: model returned prose — fix via parse_agent_json() retry
- **Coordinator restatement**: revision loop not actually changing synthesis — check coordinator_revision prompt
- **Critic too lenient**: quality score 4+ on bad synthesis — check critic prompt adversarial persona
- **Worker coordination failure**: asyncio.gather() not using return_exceptions=True
- **Schema mismatch**: agent returned extra/missing fields — check state.py dataclass
- **Cost ceiling breach**: session hit $1.00 — check which revision caused blowup

After identifying root cause:
- Propose minimal fix
- Identify if CLAUDE.md should be updated to prevent recurrence
- If yes, suggest the exact rule to add

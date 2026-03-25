---
description: Plan a new feature or node before writing any code. Usage: /plan <what you want to build>
---

Use the `planner` subagent to create a plan.md for: $ARGUMENTS

Instructions for planner:
1. Read CLAUDE.md, graph/state.py, and relevant existing agent files first
2. Use context7 MCP for any library-specific implementation questions
3. Use sequential-thinking MCP to structure the approach for complex multi-node changes
4. Output plan.md with: problem, approach, files, code snippets, tests, failure modes, cost impact
5. Do NOT write any implementation code yet — plan only

After plan.md is created, review it yourself as the critic-reviewer subagent.
Present me with the plan and wait for my approval before proceeding.

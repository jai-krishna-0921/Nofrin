# Deep Research Agent — Project Context

## Why
Multi-agent MCP-layered application. This module is the Deep Research Agent: a
LangGraph pipeline that decomposes queries, runs parallel web/academic/news workers,
synthesizes findings with citations, critiques output adversarially, and delivers
grounded research briefs in Markdown / DOCX / PDF / PPTX.

## What (project map)
```
deep_research_agent/
├── agents/          # supervisor, worker, coordinator, grounding_check, critic, delivery
├── graph/           # state.py, builder.py, router.py (LangGraph wiring)
├── tools/           # exa_search, academic_search, url_verifier, cost_tracker
├── output/          # markdown_renderer, docx_generator, pdf_generator, pptx_generator
├── eval/            # ragas_eval.py, regression_test.py, dataset/
├── prompts/         # versioned .txt files — one per agent role
├── .claude/         # agents/, commands/ (subagent defs and slash commands)
└── CLAUDE.md        # this file
```

Read `docs/architecture.md` for full design. Read `docs/state_schema.md` for the
typed `ResearchAgentState` contract before touching any agent file.

## How (build commands)
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run (single query, markdown output)
python main.py --query "..." --format markdown

# Run eval suite
pytest eval/ -v

# Type check
mypy agents/ graph/ tools/ --strict

# Lint
ruff check . && ruff format .

# LangSmith traces (open in browser after any run)
# https://smith.langchain.com → project: deep-research-agent
```

## Stack
- **Framework**: LangGraph 1.0 (Python 3.11+)
- **LLM**: Provider abstracted via `graph/llm.py` and `LLM_PROVIDER` env var. Supported: `groq` (dev default), `openrouter`, `ollama`, `anthropic`. Do not add providers beyond these four. Do not call any LLM SDK directly in agent files — always go through `get_llm()`.
- **Search**: Exa.ai primary, Tavily fallback
- **Observability**: LangSmith — tracing enabled from Day 1
- **Output**: python-docx / python-pptx / WeasyPrint via SKILL.md pipeline

## Non-negotiable rules

**Architecture**
- All agent outputs must be valid JSON matching the schema in `graph/state.py`
- Every claim in synthesis must reference a `source_url` from worker evidence
- The grounding check node runs BEFORE the critic — never skip it
- Revision loop max = 2. After 2 revisions, force-deliver with quality caveat

**Code style**
- Type annotations on every function signature — no bare `dict` or `Any`
- All async — use `asyncio.gather()` for parallel workers, never `threading`
- Wrap all Anthropic API calls in `try/except` with exponential backoff
- All JSON parsing through `parse_agent_json()` in `graph/utils.py` — never raw `json.loads()`
- No secrets in code — all keys via environment variables in `.env`

**Prompts**
- Prompts live in `prompts/` as versioned `.txt` files — never hardcoded in Python
- After correcting a prompt: bump the version suffix (`_v1.txt` → `_v2.txt`), update the changelog in `prompts/CHANGELOG.md`
- Run `pytest eval/` after any prompt change before committing

**Testing**
- Every new agent node needs a unit test in `tests/`
- Eval dataset lives in `eval/dataset/` — never delete or modify existing cases, only add
- CI gate: eval regression must not drop below baseline scores in `eval/baseline.json`

**Cost**
- `COST_CEILING_USD` in `config.py` defaults to `1.00` — do not raise without explicit instruction
- Log token cost per session to LangSmith custom metadata

## MCP servers in use
- **context7**: Up-to-date docs for LangGraph, Anthropic SDK, Exa, RAGAS — use when implementing any external library
- **sequential-thinking**: Use for planning complex multi-node LangGraph wiring
- **filesystem**: Direct file ops within project root only

## What NOT to do
- Do not add LLM providers beyond groq / openrouter / ollama / anthropic
- Do not call any LLM SDK directly in agent files — always use `get_llm()` from `graph/llm.py`
- Do not implement cross-agent debate between workers — build critic loop correctly first
- Do not skip the grounding check to "save tokens"
- Do not raise the revision cap above 2
- Do not use `json.loads()` directly — always use `parse_agent_json()`
- Do not commit `.env` or any file containing API keys

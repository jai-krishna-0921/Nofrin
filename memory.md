# Project Memory

> Updated after every significant session. Paste this into Claude when starting a new session
> on a complex multi-session task.

## Current phase
Phase 1 — Linear pipeline + LangSmith observability

## What exists right now
- [x] `graph/state.py` — ResearchAgentState fully typed + all supporting dataclasses
- [x] `graph/llm.py` — provider-agnostic LLM factory (Groq default, OpenRouter/Ollama/Anthropic switchable via LLM_PROVIDER env var)
- [x] `graph/utils.py` — parse_agent_json() utility (required by all agent nodes)
- [x] `tests/test_state.py` — 17 unit tests, all passing
- [x] `.env` — LLM_PROVIDER=groq, LangSmith, Exa keys scaffolded
- [ ] Supervisor node
- [ ] Single worker (Exa search)
- [ ] Coordinator (first pass)
- [ ] Grounding check (basic)
- [ ] Delivery node (markdown only)
- [ ] LangSmith tracing
- [ ] Cost tracking middleware

## What was just completed
Session 2026-03-25:
- Implemented graph/state.py — 7 dataclasses (Evidence, Finding, CriticIssue, CriticSuggestion, WorkerResult, SynthesisOutput, CriticOutput) + 2 TypedDicts (WorkerInput, ResearchAgentState) + 4 Literal type aliases. Key: worker_results uses Annotated[list[WorkerResult], operator.add] reducer for Send fan-out; CriticOutput.__post_init__ enforces passed from score.
- Implemented graph/llm.py — provider-agnostic factory. Model name corrected to claude-sonnet-4-6 for Anthropic branch.
- Implemented graph/utils.py — parse_agent_json() with fence-stripping and retry; AgentParseError exception.
- mypy --strict passes on all three files. ruff clean.

## Active decisions and why
- **LangGraph chosen over CrewAI**: revision loop is a first-class requirement; CrewAI lacks
  native cyclical graph support
- **LLM abstraction via get_llm(role)**: user decision overrides original single-provider rule. Groq (free tier) as default for dev velocity; swap to Anthropic for production via LLM_PROVIDER=anthropic. Critic gets llama-3.3-70b-versatile, all others get llama-3.1-8b-instant.
- **Grounding check before critic**: prevents critic from scoring hallucinated synthesis as faithful
- **Revision cap = 2**: without cap, cost compounds; force-deliver with quality caveat on hit
- **prompts/ as versioned .txt files**: enables regression testing on prompt changes
- **Finding/CriticIssue/CriticSuggestion dataclasses**: replace bare dict in guide schema — required by CLAUDE.md no-bare-dict rule
- **CriticOutput.__post_init__ enforces passed**: LLM can return wrong passed value; __post_init__ guarantees consistency with final_quality_score >= 4.0

## Known issues / TODOs
_(add as discovered, remove when resolved)_

## Eval baseline scores
_(populated after Phase 4 — fill in after first eval run)_
- faithfulness: 
- answer_relevancy: 
- context_precision:

## Last 3 CLAUDE.md updates
_(log rule additions here so we don't lose context on why they were added)_

---
*Update this file at the end of every working session.*
*Point Claude at it at the start of every new session on multi-session tasks.*

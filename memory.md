# Project Memory

> Updated after every significant session. Paste this into Claude when starting a new session
> on a complex multi-session task.

## Current phase
Phase 1 — Linear pipeline + LangSmith observability

## What exists right now
- [x] `graph/state.py` — ResearchAgentState fully typed + all supporting dataclasses
- [x] `graph/llm.py` — provider-agnostic LLM factory (Groq default, OpenRouter/Ollama/Anthropic switchable via LLM_PROVIDER env var)
- [x] `graph/utils.py` — parse_agent_json() utility (required by all agent nodes)
- [x] `graph/context.py` — NofrinContext dataclass for Runtime dependency injection
- [x] `graph/builder.py` — graph assembly skeleton with CachePolicy notes for Phase 2
- [x] `agents/supervisor.py` — intent classification + query decomposition, Runtime[NofrinContext] injection
- [x] `prompts/supervisor_v1.txt` — versioned supervisor prompt
- [x] `tests/test_state.py` — 17 unit tests, all passing
- [x] `tests/test_supervisor.py` — 21 unit tests, all passing
- [x] `.env` — LLM_PROVIDER=ollama (gpt-oss:120b-cloud), LangSmith wired
- [ ] Single worker (Exa search)
- [ ] Single worker (Exa search)
- [ ] Coordinator (first pass)
- [ ] Grounding check (basic)
- [ ] Delivery node (markdown only)
- [ ] LangSmith tracing
- [ ] Cost tracking middleware

## What was just completed
Session 2026-03-29:
- Implemented agents/supervisor.py — Runtime[NofrinContext] injection, two-level parse (_RawSupervisorOutput TypedDict → SupervisorOutput dataclass), tenacity retry (AgentParseError + httpx errors), _validate_output() enforces 3-5 queries/valid source_types/no duplicates.
- Implemented graph/context.py — NofrinContext dataclass (llm_supervisor, llm_worker, llm_coordinator, llm_critic, session_id, cost_ceiling_usd).
- Implemented graph/builder.py — skeleton with CachePolicy(ttl=300) notes for Phase 2 workers.
- Implemented prompts/supervisor_v1.txt — two-step prompt (classify intent, decompose to sub-queries).
- 38/38 tests passing. mypy --strict clean on all files.

Session 2026-03-25:
- Implemented graph/state.py — 7 dataclasses (Evidence, Finding, CriticIssue, CriticSuggestion, WorkerResult, SynthesisOutput, CriticOutput) + 2 TypedDicts (WorkerInput, ResearchAgentState) + 4 Literal type aliases. Key: worker_results uses Annotated[list[WorkerResult], operator.add] reducer for Send fan-out; CriticOutput.__post_init__ enforces passed from score.
- Implemented graph/llm.py — provider-agnostic factory. Model name corrected to claude-sonnet-4-6 for Anthropic branch.
- Implemented graph/utils.py — parse_agent_json() with fence-stripping and retry; AgentParseError exception.
- mypy --strict passes on all three files. ruff clean.

## Active decisions and why
- **LangGraph chosen over CrewAI**: revision loop is a first-class requirement; CrewAI lacks
  native cyclical graph support
- **LLM abstraction via get_llm(role)**: user decision overrides original single-provider rule. Groq (free tier) as default for dev velocity; swap to Anthropic for production via LLM_PROVIDER=anthropic. Critic gets llama-3.3-70b-versatile, all others get llama-3.1-8b-instant.
- **Runtime[NofrinContext] dependency injection**: LLMs are pre-configured in NofrinContext and injected into nodes — never instantiated inside node functions. Makes nodes fully testable and prevents get_llm(role=...) logic scattering across files.
- **CachePolicy(ttl=300) on worker nodes (Phase 2)**: skip re-searching during revision loop. Workers return cached results; only coordinator + critic re-run. InMemoryCache passed at builder.compile(cache=...).
- **Ephemeral worker subgraphs (Phase 2)**: each worker runs as an isolated compiled subgraph to prevent raw search payloads polluting coordinator context window.
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

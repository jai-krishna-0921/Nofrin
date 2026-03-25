# Build Phases

> Each phase is gated — do not start the next phase until all checkboxes are complete
> and eval scores meet the minimum bar.

---

## Phase 1 — Linear Pipeline + Observability
**Goal**: End-to-end working pipeline with one worker, markdown output, LangSmith tracing.

### Gate criteria
- [ ] `python main.py --query "..." --format markdown` completes without error
- [ ] LangSmith trace shows all nodes (supervisor → worker → coordinator → grounding → delivery)
- [ ] `pytest tests/ -v` passes
- [ ] Per-session cost logs visible in LangSmith custom metadata
- [ ] 5 manual test runs reviewed in LangSmith — traces look correct

### Checklist
- [ ] `graph/state.py` — ResearchAgentState fully typed
- [ ] `agents/supervisor.py` — intent classification + sub-query generation
- [ ] `agents/worker.py` — single worker, Exa.ai integration
- [ ] `agents/coordinator.py` — synthesis, first pass only
- [ ] `agents/grounding_check.py` — URL reachability + reference check
- [ ] `agents/delivery.py` — markdown output
- [ ] `graph/builder.py` — linear graph wired
- [ ] `tools/cost_tracker.py` — token cost middleware
- [ ] LangSmith env vars configured, tracing verified
- [ ] `tests/` — unit tests for all nodes above

---

## Phase 2 — Parallel Workers + Source Routing
**Gate**: Phase 1 gate criteria met.

### Gate criteria
- [ ] 3 workers run truly in parallel (verify in LangSmith trace timing)
- [ ] Worker source routing: web / academic / news confirmed in traces
- [ ] One worker failure does not abort session (test with mocked failure)
- [ ] Contradicting claims from different workers both appear in synthesis

### Checklist
- [ ] `asyncio.gather(..., return_exceptions=True)` fan-out
- [ ] LangGraph deferred coordinator node (waits for all workers)
- [ ] `tools/academic_search.py` — Semantic Scholar API
- [ ] Source routing logic in supervisor
- [ ] Coordinator deduplication logic
- [ ] Partial failure handling tested

---

## Phase 3 — Critic + Revision Loop
**Gate**: Phase 2 gate criteria met.

### Gate criteria
- [ ] Critic scores bad synthesis < 4 (test with deliberately poor worker output)
- [ ] Revision loop changes synthesis structurally (not just rephrasing)
- [ ] Revision counter caps at 2, force-deliver with quality caveat on hit
- [ ] Memory: prior synthesis available in state on revision pass

### Checklist
- [ ] `agents/critic.py` — 5-dimension adversarial scoring
- [ ] `prompts/critic_v1.txt` — adversarial persona
- [ ] `prompts/coordinator_revision_v1.txt` — explicit "do not restate" instruction
- [ ] `graph/router.py` — conditional edge logic
- [ ] Revision counter in state + cap enforcement
- [ ] prior_syntheses populated in state on loop-back
- [ ] Critic output includes specific quotes from synthesis

---

## Phase 4 — Grounding Hardening + Evaluation
**Gate**: Phase 3 gate criteria met.

### Gate criteria
- [ ] Faithfulness score ≥ 0.85 on eval dataset
- [ ] Answer relevancy ≥ 0.80
- [ ] All citations in final output are URL-reachable
- [ ] Regression test baseline committed to `eval/baseline.json`

### Checklist
- [ ] Semantic similarity check in grounding_check (claim vs cited chunk)
- [ ] URL reachability on all citations before delivery
- [ ] RAGAS integration
- [ ] Eval dataset: 20+ annotated questions (5 per intent type)
- [ ] `eval/regression_test.py` — gate on baseline.json
- [ ] CI: run eval on PR (GitHub Action)

---

## Phase 5 — Multi-Format Output
**Gate**: Phase 4 gate criteria met.

### Gate criteria
- [ ] DOCX output opens without errors in Microsoft Word
- [ ] PDF output renders citations correctly
- [ ] PPTX executive summary format usable
- [ ] All formats include citation appendix

### Checklist
- [ ] `output/docx_generator.py` via SKILL.md pipeline
- [ ] `output/pdf_generator.py` via SKILL.md pipeline
- [ ] `output/pptx_generator.py` via SKILL.md pipeline
- [ ] Output format selector in API
- [ ] Standard brief structure (executive summary → findings → risks → gaps → quality score → citations)

---

## Phase 6 — Production Hardening
**Gate**: Phase 5 gate criteria met.

### Gate criteria
- [ ] P95 latency < 120s on test suite
- [ ] Cost per session < $0.40 on average (single pass)
- [ ] 0 unhandled exceptions in 50 test runs
- [ ] Docker image builds and runs

### Checklist
- [ ] Exponential backoff on all external API calls
- [ ] Per-session cost ceiling enforcement
- [ ] FastAPI `/research` endpoint
- [ ] Session logging to PostgreSQL
- [ ] Docker containerisation
- [ ] Rate limiting middleware

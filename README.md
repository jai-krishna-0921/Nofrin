# Nofrin

**Nofrin** is a multi-agent deep research system built on LangGraph. It plans, reasons, executes, and self-corrects to produce grounded research briefs with citations.

## Overview

Nofrin decomposes complex research queries into parallel search tasks, synthesizes findings with proper attribution, and iteratively refines output through adversarial critique — all within a configurable cost ceiling.

```
Query → Supervisor → [Workers × N] → Compressor → Coordinator → Grounding → Critic → Delivery
                          ↑                                              │
                          └────────────── Revision Loop ──────────────────┘
```

## Pipeline Architecture

| Node | Role |
|------|------|
| **Supervisor** | Classifies intent (exploratory/comparative/adversarial/factual) and decomposes query into 3–5 sub-queries with source routing (web/academic/news) |
| **Worker** | Parallel Exa.ai search + LLM evidence extraction per sub-query |
| **Boundary Compressor** | Deduplicates and compresses worker results |
| **Coordinator** | Synthesizes evidence into structured brief with citations |
| **Grounding Check** | Validates claims against source evidence |
| **Critic** | 5-dimension adversarial evaluation (factuality, citation alignment, reasoning, completeness, bias) |
| **Delivery** | Renders final output in requested format |

### Revision Loop

The critic scores the synthesis on five dimensions (weights in parentheses):
- **Factuality** (0.30)
- **Citation Alignment** (0.25)
- **Reasoning** (0.20)
- **Completeness** (0.15)
- **Bias** (0.10)

If the weighted score < 4.0 and revisions < 2, the coordinator revises the synthesis guided by critic feedback. Hard cap: 2 revisions.

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export LLM_PROVIDER=groq        # groq (default) | openrouter | ollama | anthropic
export EXA_API_KEY=your_key
export COST_CEILING_USD=1.00    # optional, default $1.00

# Run
python main.py --query "What are the latest advances in fusion energy?" --format markdown
```

## Output Formats

| Format | Flag |
|--------|------|
| Markdown | `--format markdown` (default) |
| Word | `--format docx` |
| PDF | `--format pdf` |
| PowerPoint | `--format pptx` |

## Project Structure

```
nofrin/
├── agents/
│   ├── supervisor.py      # Intent classification, query decomposition
│   ├── worker.py         # Exa search, evidence extraction
│   ├── coordinator.py    # Synthesis brief generation
│   ├── grounding_check.py # Claim validation
│   ├── critic.py         # 5-dimension adversarial evaluation
│   └── delivery.py       # Format rendering
├── graph/
│   ├── state.py          # ResearchAgentState schema
│   ├── builder.py        # LangGraph assembly
│   ├── router.py         # Conditional edges, dispatch, budget gate
│   ├── context.py        # Runtime context injection
│   ├── llm.py            # Provider abstraction
│   └── utils.py          # JSON parsing, error handling
├── prompts/              # Versioned LLM prompts (.txt)
├── eval/
│   ├── ragas_eval.py     # RAGAS evaluation harness
│   ├── regression_test.py # CI gate
│   └── dataset/          # Eval cases
├── tests/                # Unit tests per agent
├── main.py               # CLI entry point
└── CLAUDE.md             # Project rules for AI assistants
```

## LLM Providers

Nofrin supports four providers via `LLM_PROVIDER` env var:

| Provider | Value | Notes |
|----------|-------|-------|
| Groq | `groq` | Default for development |
| OpenRouter | `openrouter` | Multi-model access |
| Ollama | `ollama` | Local inference |
| Anthropic | `anthropic` | Claude models, prompt caching |

All LLM calls go through `get_llm()` in `graph/llm.py` — never call provider SDKs directly.

## Observability

LangSmith tracing is enabled by default. View traces at:
```
https://smith.langchain.com → project: deep-research-agent
```

Cost tracking per session is logged to stderr:
```
cost_usd=0.42 total_tokens=15234
```

## Development

```bash
# Run tests
pytest tests/ -v

# Type check
mypy agents/ graph/ --strict

# Lint
ruff check . && ruff format .

# Run eval suite
pytest eval/ -v
```

## Cost Ceiling

The `COST_CEILING_USD` env var (default: $1.00) enforces a hard budget. When spending exceeds 75% of the ceiling:
1. A warning is appended to `synthesis.gaps`
2. Research halts and delivers early

## State Schema

The core `ResearchAgentState` is a `TypedDict` with operator.add reducer on `worker_results` for parallel fan-out accumulation:

```python
class ResearchAgentState(TypedDict):
    user_query: str
    intent_type: IntentType
    output_format: OutputFormat
    sub_queries: list[str]
    source_routing: dict[str, SourceType]
    worker_results: Annotated[list[WorkerResult], operator.add]
    compressed_worker_results: list[WorkerResult]
    synthesis: Optional[SynthesisOutput]
    grounding_issues: list[str]
    critic_output: Optional[CriticOutput]
    revision_count: int  # max 2
    prior_syntheses: list[SynthesisOutput]
    session_id: str
    total_tokens_used: int
    cost_usd: float
    final_output: Optional[str]
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `groq` | LLM provider |
| `EXA_API_KEY` | Yes | — | Exa.ai search API key |
| `COST_CEILING_USD` | No | `1.00` | Max spend per session |
| `OPENAI_API_KEY` | Conditional | — | Required if using OpenRouter |
| `ANTHROPIC_API_KEY` | Conditional | — | Required if using Anthropic |
| `GROQ_API_KEY` | Conditional | — | Required if using Groq |

## License

MIT
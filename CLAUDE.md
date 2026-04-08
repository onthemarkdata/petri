# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Petri — a colony-based research orchestration framework that decomposes claims into DAGs of logical units and validates them bottom-up through a multi-agent adversarial review pipeline.

## Build & Install

```bash
# Prerequisite: Claude Code CLI (authenticated)
# Install (everything included — no extras needed)
uv pip install -e "."
```

## Test Commands

```bash
# Run all tests
uv run pytest tests/

# Run with short traceback
uv run pytest tests/ --tb=short -q

# Run specific test file
uv run pytest tests/unit/test_models.py

# Run specific test class
uv run pytest tests/unit/test_queue.py::TestStateTransitions
```

## CLI Commands

```bash
petri init              # Initialize .petri/ directory
petri seed <claim>      # Decompose a claim into a colony DAG
petri check             # Show node statuses
petri grow --all        # Run validation pipeline
petri stop              # Gracefully halt processing
petri feed <source>     # Feed new evidence
petri graph             # Visualize colony structure
petri launch            # Launch REST+SSE dashboard (port 8090)
petri scan              # Run contradiction scanner
petri connect <a> <b>   # Inspect or create a dependency edge
petri inspect           # Check prerequisites
```

## Architecture

```
petri/
├── models.py              # Pydantic models: Node, Colony, Event, QueueEntry, AgentRole
├── config.py              # Centralized config loader (petri.yaml)
├── cli.py                 # Typer CLI (8 commands)
│
├── engine/                # Core pipeline orchestration
│   ├── processor.py       # Pipeline processor (queue-driven, concurrent)
│   ├── propagation.py     # Evidence re-entry, dependency propagation
│   ├── load_balancer.py   # Adaptive concurrency control
│   └── preflight.py       # Prerequisite checks (Python, Claude Code)
│
├── storage/               # File-based persistence
│   ├── event_log.py       # Append-only JSONL per node
│   └── queue.py           # 13-state machine with fcntl file locking
│
├── analysis/              # Validation, convergence, scanning
│   ├── convergence.py     # Verdict matrix, blocking check, circuit breaker
│   ├── validators.py      # Source hierarchy enforcement
│   └── scanner.py         # Contradiction scanner (10 categories, 6-level authority)
│
├── reasoning/             # LLM-driven logic
│   ├── decomposer.py      # Claim → colony decomposition
│   ├── debate.py          # 4 structured debate pairings
│   ├── ingest.py          # Content ingestion (URL, file, text)
│   └── claude_code_provider.py  # InferenceProvider via Claude Code CLI
│
├── graph/                 # DAG structure
│   └── colony.py          # DAG operations, cycle detection, level computation
│
├── defaults/              # Opinionated config (13 agents, 4 debates, constitution)
├── templates/             # Plain-text templates for adapter config generation
├── dashboard/             # FastAPI REST+SSE, SQLite migration
└── adapters/              # Harness adapters (Claude Code adapter)
```

## Active Technologies

- Python 3.11+ / Core: stdlib + Pydantic / CLI: Typer / Dashboard: FastAPI + SSE
- Storage: JSONL (event logs, append-only), JSON (queue, file-locked), SQLite (disposable dashboard index)
- Testing: pytest
- Config generation: stdlib string.Template (no Jinja2)

## Key Design Decisions

- Composite key identity: `{dish}-{colony}-{level}-{seq}` for nodes
- Two-store separation: event log (JSONL) + queue (JSON) — no data duplication
- 13 agents: 3 leads (non-blocking orchestrators) + 10 specialists (6 blocking)
- Convergence = all 6 blocking verdicts in pass set (mechanical, no LLM)
- Default LLM: claude-sonnet-4-6 via Claude Code CLI

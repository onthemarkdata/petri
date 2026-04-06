# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Petri — a colony-based research orchestration framework that decomposes claims into DAGs of logical units and validates them bottom-up through a 13-agent adversarial review pipeline.

## Build & Install

```bash
# Prerequisite: Ollama with default model
ollama pull gemma3:4b-it

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
petri analyze --graph   # Visualize colony structure
petri analyze --dashboard  # Launch REST+SSE dashboard (port 8090)
petri analyze --scan    # Run contradiction scanner
```

## Architecture

```
petri/
├── models.py          # Pydantic models: Node, Colony, Event, QueueEntry, AgentRole
├── cli.py             # Typer CLI (7 commands)
├── colony.py          # DAG operations, cycle detection, level computation
├── decomposer.py      # Claim → colony decomposition
├── event_log.py       # Append-only JSONL per node
├── queue.py           # 13-state machine with fcntl file locking
├── processor.py       # Pipeline processor (queue-driven, concurrent)
├── convergence.py     # Verdict matrix, blocking check, circuit breaker
├── debate.py          # 4 structured debate pairings
├── propagation.py     # Evidence re-entry, dependency propagation
├── validators.py      # Source hierarchy enforcement
├── scanner.py         # Contradiction scanner (10 categories, 6-level authority)
├── defaults/          # Opinionated config (13 agents, 4 debates, constitution)
├── templates/         # Plain-text templates for adapter config generation
├── dashboard/         # FastAPI REST+SSE, SQLite migration
└── adapters/          # Harness adapters (Claude Code adapter)
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
- Default LLM: gemma-3-4b-it (local, free) — paid models opt-in via petri.yaml

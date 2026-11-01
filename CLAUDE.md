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
petri check             # Show cell statuses
petri grow              # Run validation pipeline (all eligible cells)
petri stop              # Gracefully halt processing
petri feed <source>     # Feed new evidence
petri graph             # Visualize colony structure
petri launch            # Launch REST+SSE dashboard (port 8090)
petri scan              # Run contradiction scanner
petri connect <a> <b>   # Inspect or create a dependency edge
petri inspect           # Check prerequisites
petri export            # Regenerate derived Markdown/JSONL from SQLite
petri backup --with-export  # Snapshot domain data and its derived text
```

## Architecture

```
petri/
├── models.py        # Cells, colonies, events, and typed results
├── config.py        # Uncached validated settings
├── cli/             # Lifecycle and inspection commands
├── execution/       # DBOS SQLite runtime and typed pipeline
├── harness/         # First-party inference models and transports
├── storage/         # Per-dish SQLite data, backups, and derived exports
├── lifecycle/       # Revalidation, dependent flags, and proposals
├── analysis/        # Convergence, source validation, and scanning
├── reasoning/       # Typed seed decomposition and evidence ingestion
├── graph/           # Dependency graphs and integrity checks
├── telemetry/       # Local spans and usage, with optional OTLP
├── query/           # Read-only SQLite queries
└── dashboard/       # Local REST and SSE interface
```

## Active Technologies

- Python 3.11+ / Core: stdlib + Pydantic / CLI: Typer / Dashboard: FastAPI + SSE
- Storage: per-dish SQLite for domain data, events, spans, and usage; separate DBOS SQLite checkpoints.
- Execution: DBOS is the only backend. Existing v1 dishes remain untouched; use v0.3.7 to run them.
- Testing: pytest
- Config generation: stdlib string.Template (no Jinja2)

## Key Design Decisions

- Composite key identity: `{dish}-{colony}-{level}-{seq}` for cells
- Two-store separation: event log (JSONL) + queue (JSON) — no data duplication
- 13 agents: 3 leads (non-blocking orchestrators) + 10 specialists (6 blocking)
- Convergence = all 6 blocking verdicts in pass set (mechanical, no LLM)
- Default LLM: claude-sonnet-4-6 via Claude Code CLI

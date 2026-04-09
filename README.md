![](petri/assets/petri-title-screen.png)

# Petri

[![PyPI version](https://img.shields.io/pypi/v/petri-grow)](https://pypi.org/project/petri-grow/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/petri-grow)](https://pypi.org/project/petri-grow/)
[![License](https://img.shields.io/pypi/l/petri-grow)](https://github.com/onthemarkdata/petri/blob/main/LICENSE)

An agent orchestration framework to grow your AI's context via Claude Code. Decomposes claims into DAGs of logical units and validates them bottom-up through a multi-agent adversarial review pipeline.

What results are a repository of curated information, citations, and URLs, organized by concepts, how they relate to each other, and the nuances of the claims. This is quite useful for AI agents to have pre-loaded context assets available for reference.

Petri includes both a CLI tool intended for AI agents and an interactive UI mode to help keep track of all active agents within Petri and to review the context, reasoning, and citations curated by the AI agents.

## Demo
https://github.com/user-attachments/assets/ef47dc07-741d-4ac5-9f35-2afb12fc9046

## ⚠️ Cost Warning ⚠️

**Petri uses Claude via Claude Code, which costs money.** Each cell goes through 13 agents across multiple iterations, generating significant token usage. A single colony with 10+ cells can produce **thousands of LLM calls** across Socratic analysis, research, critique, debate, red team, and evaluation phases.

The default model is `claude-sonnet-4-6`. You can switch models in `petri.yaml` or the setup wizard:

```yaml
model:
  name: claude-opus-4-6  # most capable, higher cost
```

**Monitor your usage.** Start with small claims to understand the cost profile before running large colonies.

Alternatively, you can [use an open-source model in Claude Code via Ollama, by following this documentation](https://docs.ollama.com/integrations/claude-code). But this requires a bit of setup on your side, and you need hardware capable of running a model locally.

## Setup

Petri needs **Python 3.11+** and the **Claude Code CLI** (which provides the LLM inference). The cleanest path is to install both into a fresh [uv](https://docs.astral.sh/uv/) environment.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate a Python 3.11+ virtual environment

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

Every command from here on assumes this venv is active. If you open a new shell, re-run the `source .venv/bin/activate` line.

### 3. Install petri-grow into the venv

```bash
uv pip install petri-grow
```

### 4. Install and authenticate Claude Code

Petri uses [Claude Code](https://claude.com/claude-code) as its agent harness. Follow the install + login steps at https://docs.anthropic.com/en/docs/claude-code, then verify:

```bash
claude --version
```

### 5. Verify the full prerequisite chain

```bash
petri inspect
```

This reports any missing pieces (Python version, Claude Code login, PATH issues) without modifying your system.

## Quickstart

Petri has two modes. Start with **Interactive Mode** — it's the fastest path from zero to a running colony.

### Interactive Mode (recommended)

```bash
mkdir my-research && cd my-research
petri launch
```

`petri launch` auto-creates `.petri/` from defaults, opens the Petri Lab dashboard at http://127.0.0.1:8090, and walks you through dish setup, seeding, growing, and inspection in the browser. No other commands required.

### CLI Mode (AI-agent friendly)

Petri is designed to be AI agent first for UX. It's highly recommended to have a Claude Code session already started and pass this README file link (https://github.com/onthemarkdata/petri/blob/main/README.md) directly to Claude Code to set up.

```bash
# 1. Initialize a petri dish
mkdir my-research && cd my-research
petri init
# → Initialized petri dish 'my-research' at /path/to/my-research
#   Model: claude-sonnet-4-6

# 2. Seed a colony from a claim
petri seed "Open source models will catch up to current frontier models in the next 6 months."
# → Colony 'open-source-models' created with 6 nodes across 3 levels

# 3. Check status
petri check
# → Shows a table of all cells with status PENDING

# 4. Grow cells through the validation pipeline
petri grow
# → Processes every eligible cell bottom-up: Socratic → Research → Critique → Red Team → Evaluation

# 5. Feed new evidence (requires cells that have completed validation)
petri feed https://arxiv.org/abs/2026.12345
# → Ingests content, matches to relevant cells, flags for re-validation

# 6. Analyze
petri graph                 # text tree / DOT export
petri launch                # Live web dashboard on port 8090
petri scan --fix            # contradiction scanner
# → `graph` shows the colony DAG as text;
#   `launch` opens the full Petri Lab dashboard (Computer tab,
#   Lab overview, Colony DAG, Logs, Cell detail)

# 7. Stop
petri stop
# → Gracefully halts any active processing
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline and state machine details.

## CLI Reference

```
petri --help
```

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `petri init` | Create `.petri/` directory with defaults (interactive setup wizard) | `--name`, `--no-questions` |
| `petri seed <claim>` | Decompose a claim into a colony DAG | `--no-questions`, `--colony` |
| `petri check` | Show cell statuses across colonies | `--colony`, `--cell`, `--json` |
| `petri grow` | Run cells through the validation pipeline (defaults to all eligible) | `--cell`, `--colony`, `--dry-run`, `--max-concurrent` |
| `petri feed <source>` | Ingest new evidence and flag affected cells | `--colony`, `--auto-reopen` |
| `petri graph` | Render the colony DAG as text tree or DOT | `--format`, `--colony` |
| `petri scan` | Run the contradiction scanner | `--fix`, `--loop` |
| `petri connect <a> <b>` | Inspect or create a dependency edge between two cells | |
| `petri launch` | Open the live web dashboard | `--port`, `--host` |
| `petri stop` | Gracefully halt active processing | `--force` |
| `petri inspect` | Check that all prerequisites are installed | |

**Typical workflow:**

1. `petri init` -- one-time setup
2. `petri seed "your claim"` -- decompose into a colony
3. `petri grow` -- validate bottom-up (leaf cells first, then parents)
4. `petri check` -- inspect progress
5. `petri feed <url>` -- add evidence, re-open affected cells
6. `petri grow` -- re-validate impacted cells
7. `petri graph` -- view the final colony structure

> **Note:** `petri grow` with no flags processes every currently eligible cell. For multi-level colonies, run it multiple times until all levels are resolved — leaf cells validate first, unlocking their parents. Scope to a subset with `--cell <id>` (repeatable) or `--colony <name>`.

## How It Works

Each cell in the colony goes through:

1. **Socratic questioning** -- Clarify terms, challenge assumptions, identify what evidence is needed
2. **Research phase** -- Investigator gathers evidence with URL-cited sources, Freshness Checker verifies currency, Dependency Auditor checks prerequisites
3. **Critique phase** -- Specialist agents assess in parallel, Cell Lead mediates structured debates
4. **Convergence check** -- All blocking verdicts must pass (mechanical check, no LLM)
5. **Circuit breaker** -- Max 3 iterations per cycle; if not converged, flags for human guidance
6. **Red Team** -- Dedicated adversarial phase builds the strongest case against the cell
7. **Evidence Evaluation** -- Neutral weighing of all evidence: VALIDATED, DISPROVEN, or DEFER

**Citation-first evidence model:** Every agent must back claims with URL-linked sources ranked by a 6-level hierarchy (direct measurement → community report). Summaries are kept terse to prevent context rot across iterations.

Every action is logged as an immutable event in the cell's JSONL file, identified by a composite key (`{dish}-{colony}-{level}-{seq}-{8hex}`).

## Architecture

- **Multi-agent pipeline**: lead orchestrators + specialists (blocking and advisory)
- **Event sourcing**: append-only JSONL per cell, rolled up to SQLite for the dashboard
- **Queue state machine**: enforced transitions, file-locked for concurrency
- **Claude Code native**: Petri runs on top of the Claude Code CLI as its agent harness — Claude Code is a hard dependency, not a pluggable option. The `adapters/` layer exists for internal separation of concerns, not to enable other harnesses.
- **Live dashboard**: single-file SPA with PTY-backed terminal, interactive colony DAG, and per-cell detail pages

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, state machine diagram, and agent details.

## Development

```bash
uv pip install -e ".[all]"
uv run pytest tests/
```

## Credit
- Image Credits: Gemini Pro, April 2026
- Music by [Karl Casey @ White Bat Audio](https://youtu.be/o9C-4Gln9Tc?si=7Hi3_ouxqvgd28eR)

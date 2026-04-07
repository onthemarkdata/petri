![](petri/assets/petri-title-screen.png)
*Image Credit: Gemini Pro, April 2026*

# Petri

[![PyPI version](https://img.shields.io/pypi/v/petri-grow)](https://pypi.org/project/petri-grow/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/petri-grow)](https://pypi.org/project/petri-grow/)
[![License](https://img.shields.io/pypi/l/petri-grow)](https://github.com/onthemarkdata/petri/blob/main/LICENSE)

An agent orchestration framework to grow your AI's context. Decomposes claims into DAGs of logical units and validates them bottom-up through a multi-agent adversarial review pipeline.

## Cost Warning

**Petri uses Claude via Claude Code, which costs money.** Each node goes through 13 agents across multiple iterations, generating significant token usage. A single colony with 10+ nodes can produce **thousands of LLM calls** across Socratic analysis, research, critique, debate, red team, and evaluation phases.

The default model is `claude-sonnet-4-6`. You can switch models in `petri.yaml` or the setup wizard:

```yaml
model:
  name: claude-opus-4-6  # most capable, higher cost
```

**Monitor your usage.** Start with small claims to understand the cost profile before running large colonies.

## Prerequisites

### 1. Python 3.11+

Petri requires Python 3.11 or later. Check your version:

```bash
python3 --version
```

If you need a newer version, [uv](https://docs.astral.sh/uv/) can install one for you:

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
```

### 2. Claude Code

Petri uses [Claude Code](https://claude.com/claude-code) as its agentic harness for inference.

Install: https://docs.anthropic.com/en/docs/claude-code

Verify it's working:

```bash
claude --version
```

### Verify everything

```bash
petri inspect
```

This checks all prerequisites and reports what's missing.

## Install

```bash
# Recommended (with uv)
uv pip install petri-grow

# Or with pip
pip install petri-grow
```

This installs the CLI and core library. Claude Code must be authenticated (see above).

## Quickstart

```bash
# 1. Initialize a petri dish
mkdir my-research && cd my-research
petri init
# → Initialized petri dish 'my-research' at /path/to/my-research
#   Model: claude-sonnet-4-6

# 2. Seed a colony from a claim
petri seed "A hotdog is a sandwich" --no-questions
# → Colony 'hotdog-sandwich' created with 6 nodes across 3 levels

# 3. Check status
petri check
# → Shows a table of all nodes with status PENDING

# 4. Grow nodes through the validation pipeline
petri grow --all
# → Processes nodes bottom-up: Socratic → Research → Critique → Red Team → Evaluation

# 5. Feed new evidence (requires nodes that have completed validation)
petri feed https://arxiv.org/abs/2026.12345
# → Ingests content, matches to relevant nodes, flags for re-validation

# 6. Analyze
petri analyze --graph       # text tree / DOT export
petri analyze --dashboard   # REST + SSE API on port 8090
petri analyze --scan --fix  # contradiction scanner
# → --graph shows the colony DAG; --dashboard opens a live web UI

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
| `petri init` | Create `.petri/` directory with defaults | `--name` |
| `petri seed <claim>` | Decompose a claim into a colony DAG | `--no-questions`, `--colony` |
| `petri check` | Show node statuses across colonies | `--colony`, `--node`, `--json` |
| `petri grow` | Run nodes through the validation pipeline | `--all`, `--colony`, `--dry-run`, `--max-concurrent` |
| `petri feed <source>` | Ingest new evidence and flag affected nodes | `--colony`, `--auto-reopen` |
| `petri analyze` | Visualization and diagnostics | `--graph`, `--dashboard`, `--scan`, `--fix` |
| `petri stop` | Gracefully halt active processing | `--force` |
| `petri inspect` | Check that all prerequisites are installed | |

**Typical workflow:**

1. `petri init` -- one-time setup
2. `petri seed "your claim"` -- decompose into a colony
3. `petri grow --all` -- validate bottom-up (cells first, then parents)
4. `petri check` -- inspect progress
5. `petri feed <url>` -- add evidence, re-open affected nodes
6. `petri grow --all` -- re-validate impacted nodes
7. `petri analyze --graph` -- view the final colony structure

> **Note:** `petri grow --all` processes all currently eligible nodes. For multi-level colonies, run it multiple times until all levels are resolved — cells validate first, unlocking their parents.

## How It Works

Each node in the colony goes through:

1. **Socratic questioning** -- Clarify terms, challenge assumptions, identify what evidence is needed
2. **Research phase** -- Investigator gathers evidence with URL-cited sources, Freshness Checker verifies currency, Dependency Auditor checks prerequisites
3. **Critique phase** -- Specialist agents assess in parallel, Node Lead mediates structured debates
4. **Convergence check** -- All blocking verdicts must pass (mechanical check, no LLM)
5. **Circuit breaker** -- Max 3 iterations per cycle; if not converged, flags for human guidance
6. **Red Team** -- Dedicated adversarial phase builds the strongest case against the node
7. **Evidence Evaluation** -- Neutral weighing of all evidence: VALIDATED, DISPROVEN, or DEFER

**Citation-first evidence model:** Every agent must back claims with URL-linked sources ranked by a 6-level hierarchy (direct measurement → community report). Summaries are kept terse to prevent context rot across iterations.

Every action is logged as an immutable event in the node's JSONL file, identified by a composite key (`{dish}-{colony}-{level}-{seq}-{8hex}`).

## Architecture

- **Multi-agent pipeline**: lead orchestrators + specialists (blocking and advisory)
- **Event sourcing**: append-only JSONL per node, rolled up to SQLite for the dashboard
- **Queue state machine**: enforced transitions, file-locked for concurrency
- **Harness-agnostic**: core uses only stdlib + Pydantic; adapters bridge to Claude Code and future harnesses

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, state machine diagram, and agent details.

## Development

```bash
uv pip install -e ".[all]"
uv run pytest tests/
```

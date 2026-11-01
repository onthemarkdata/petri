# Architecture

Petri decomposes research claims into dependency graphs and validates eligible
cells through typed agents. DBOS is the only execution backend. It uses local
SQLite checkpoints; no external database service is required.

## Colony DAG

A **colony** is a DAG of **cells**, each representing a sub-claim derived from a root claim. The hierarchy is **petri dish → colony(s) → cell(s)**. Cells are identified by composite keys: `{dish}-{colony}-{level}-{seq}`.

- **Level 0** is the center (root claim)
- Levels increase outward as claims decompose into more fundamental sub-claims
- **Leaf cells** (with no dependencies) are validated first
- Parent cells unlock for validation only after all their dependencies pass

Cycle detection (Kahn's algorithm) ensures the graph remains a valid DAG. Cross-colony edges are supported for shared dependencies.

## Agent System

Petri uses 13 agents organized into leads and specialists:

### Leads (non-blocking orchestrators)

| Agent | Role |
|-------|------|
| `decomposition_lead` | Manages claim decomposition into DAG |
| `cell_lead` | Orchestrates cell through validation phases |
| `red_team_lead` | Leads adversarial disproval attempt |

### Specialists

| Agent | Phase | Blocking | Pass Verdicts | Block Verdicts |
|-------|-------|----------|---------------|----------------|
| `investigator` | Research | Yes | EVIDENCE_SUFFICIENT, EVIDENCE_CONTRADICTS | NEEDS_MORE_EVIDENCE, CANNOT_DETERMINE |
| `freshness_checker` | Research | Yes | EVIDENCE_CURRENT, STALE_BUT_HOLDS | MATERIALLY_STALE |
| `dependency_auditor` | Research | Yes | DEPENDENCIES_CLEAN | UNVALIDATED_DEPS, CIRCULAR_REASONING |
| `skeptic` | Critique | Yes | ARGUMENT_WITHSTANDS_CRITIQUE | CRITICAL_FLAW_FOUND, UNADDRESSED_COUNTERARGUMENT |
| `champion` | Critique | Yes | STRONG_CASE, DEFENSIBLE_WITH_CAVEATS | CANNOT_DEFEND |
| `pragmatist` | Critique | Yes | PRODUCTION_READY, DIRECTIONALLY_CORRECT | WOULD_NOT_SHIP_THIS |
| `simplifier` | Critique | No | APPROPRIATELY_SCOPED, COULD_BE_SIMPLER | OVERCOMPLICATED |
| `triage` | Critique | Conditional | HIGH_VALUE, MODERATE_VALUE | LOW_VALUE_DEFER (redirects to DEFER_OPEN) |
| `impact_assessor` | Critique | No | CRITICAL_PATH, SUPPORTING_NOT_BLOCKING, ISOLATED_LOW_IMPACT | -- |
| `evidence_evaluator` | Evaluation | No | EVIDENCE_CONFIRMS, EVIDENCE_REFUTES, EVIDENCE_INCONCLUSIVE | -- |

**6 blocking agents** must all pass for convergence. The `triage` agent is conditionally blocking -- a LOW_VALUE_DEFER verdict short-circuits the cell to deferred status. Advisory agents (`simplifier`, `impact_assessor`) inform but do not gate.

## Durable pipeline

The public typed pipeline runs Socratic questioning, research, critique and
debates, convergence, red team, and evaluation. Its graph checks declared
transitions and advances the iteration clock on retry edges. Terminal results
include validated, disproven, deferred, needs-human, and sync-conflict outcomes.

Each cell round is a DBOS workflow with small phase checkpoints. A separate
step commits terminal domain status. A colony parent reads the SQLite graph,
enqueues eligible children, and waits before checking eligibility again.
Every dependency must be validated before a cell becomes eligible.

The execution owner lock permits one active process per dish. Queue settings
and application admission enforce concurrency and rolling start-rate limits.
Read-only dashboard clients can coexist with that owner. Completed checkpoints
are reused after restart. A crash between an external response and its durable
commit can repeat the external call; provider idempotency is required to close
that gap.

`grow` starts or resumes durable work. `stop` records cancellation at the next
committed step boundary; `stop --force` can interrupt an active inference call.
Revalidation allocates a new cell round and flags dependents without reopening
them automatically. See [durable execution](docs/v2/DURABLE_EXECUTION.md) and
[seed recovery](docs/v2/SEED_RECOVERY.md) for identities and recovery checks.

## Citation-First Evidence Model

All agent output follows a citation-first pattern: every claim must be backed by at least one URL-linked source. Agents produce numbered sources with hierarchy levels and 1-2 sentence findings rather than verbose free-text analysis. Summaries are capped at 1-3 sentences to prevent context rot as evidence accumulates across iterations.

Evidence files (`evidence.md`) follow this structure per phase:

```markdown
### Iteration 1 — Phase 1 Research

**Source 1 (Level 2 — Authoritative Documentation):** Title (Year) — URL — Finding. **Supports claim.**
**Source 2 (Level 2 — Peer-Reviewed):** Title (Year) — URL — Finding. **Supports claim.**

**Summary:** Terse assessment with source counts and dimensions.
```

## Source Hierarchy

Evidence is ranked by a six-level credibility hierarchy:

| Level | Type |
|-------|------|
| 1 | Direct Measurement |
| 2 | Authoritative Documentation |
| 3 | Derived Calculation |
| 4 | Corroborated Expert Consensus |
| 5 | Single Expert Opinion |
| 6 | Community Report |

Terminal verdicts (VALIDATED, DISPROVEN) require at least one source at Level 4 or higher. Cells with only Level 5-6 evidence cannot reach terminal status.

## Storage and configuration

Domain rows, append-only events, graph edges, evidence, spans, and usage live in
`.petri/petri-dishes/<dish>/petri.sqlite`. DBOS control state lives separately in
`.petri/dbos.sqlite`. Markdown and JSONL exports are derived artifacts.

Settings are validated against the v2 schema whenever they are loaded. Agent
structure lives in code; instructions and supported vocabulary overrides live
in YAML. Configuration has no import-time constants or shared loader cache.

Fresh dishes use DBOS and create no queue file. Existing v0.x dishes remain
untouched and are not migrated; use v0.3.7 to run their original runtime. The
[historical architecture](docs/history/architecture-v1.md) remains available
for interpreting old data. The removed Unix queue lock is no longer an import
requirement.

## Inference and observability

First-party Pydantic AI models support pi, Claude Code, and configured providers.
Typed decomposition journals validated responses before materializing graph
changes. Evidence and agent outputs remain in the domain database rather than
large workflow checkpoints.

`petri launch` reads SQLite through the query and execution seams. Local traces,
usage, graph intelligence, and lifecycle proposals are available without an
external observability service. Optional OTLP configuration is documented in
[observability](docs/observability.md). The
[v2 design record](docs/ARCHITECTURE-V2.md) explains the migration decisions.

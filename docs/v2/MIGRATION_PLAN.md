# Petri v2 Migration Plan

> **Status:** Planning complete — ready to cut GitHub issues. No implementation has started.
> **Date:** 2026-07-12 · **Amended same day (D4Δ):** storage simplified to a single Petri-owned
> SQLite file per dish; DuckDB removed; text becomes a derived export. Diagrams and issues reflect
> the amendment.
> **Companion documents:** full issue bodies in `docs/v2/issues/M*.md` (81 issues, GitHub-ready;
> machine-readable source: `docs/v2/issues/backlog.json`); decision record in `docs/ARCHITECTURE-V2.md`;
> field evidence index in `docs/field-reports.md`.

## How to use this document

This is the roadmap and the "why." Each milestone below has a one-page summary here and a full
issue backlog file under `docs/v2/issues/` whose entries are written to be pasted into GitHub
issues verbatim (context, scope, out-of-scope, touched files, checkbox acceptance criteria,
labels, dependencies). `docs/ARCHITECTURE-V2.md` was seeded from §2–§4 of this plan and is the
permanent in-repo home for the decision record; M1's first issue audits and finalizes it.

---

## 1. Executive summary

Petri v0.3.4 works — it validated real research — but its dogfooding exposed two systemic
weaknesses, each documented in the maintainer's own field notes:

1. **The breakdown logic is template-driven.** 12 colonies produced 983 cells against a
   hand-crafted baseline of 101 claims + 128 edges (9.7× inflation, zero cross-links); ~25% of
   cells were trivially-true axioms that would have burned ~3,159 agent calls in validation.
2. **The processing layer re-implements distributed-systems infrastructure by hand.** A
   14-state queue machine in `fcntl`-locked JSON, an adaptive load balancer, subprocess
   stream-parsing with hand-rolled retries — plus a 418-line external bash wrapper
   (`migrate-to-petri.sh`) doing rate-limit parsing and exponential backoff, because the engine
   couldn't. An 88-cell colony (hours of paid compute) was destroyed by a re-seed after a
   rate-limit interrupt.

v2 fixes both by re-platforming — **incrementally, never breaking the CLI surface** — onto:

- **Pydantic AI** (`Agent` + typed structured outputs) for the 13-agent roster, debates, and the
  decomposer — replacing ~900 lines of subprocess/JSON-extraction code and the entire
  "silent-fallback verdict" bug class.
- **pydantic-graph** for the per-cell validation pipeline (typed nodes replace the 14-state queue
  machine). The colony DAG itself stays runtime *data* — it is never modeled as code edges.
- **DBOS Transact** (`dbos-transact-py`, MIT, **SQLite system database by default — no Docker, no
  Postgres, ever**) for durable execution: crash/interrupt resume, exactly-once step recording,
  rate-limit-aware queues with concurrency caps. DBOS sits behind an internal `ExecutionBackend`
  seam and must pass a Phase-0 validation spike before anything depends on it.
- **A harness abstraction with pi as the default backend** (`pi --mode rpc`, LF-delimited JSONL),
  the Claude Code CLI kept as a supported adapter, and any pydantic-ai provider string accepted —
  restoring v0.1.0's original "Harness-Agnostic Core" constitution principle.
- **OpenTelemetry end to end**: DBOS workflow→step spans + pydantic-ai run/tool/token spans +
  Petri domain attributes, exported over OTLP *and* persisted to a local spans table so
  `petri launch` shows per-cell traces with zero external infrastructure.
- **One Petri-owned SQLite file per dish (`petri.sqlite`) as the domain source of truth** (D4Δ):
  an append-only `events` table with schema-enforced UNIQUE event ids, `cells` + `edges` tables
  for the colony DAG, `spans`/`usage` for observability, and SQL views for all reads — stdlib
  `sqlite3`, zero new storage dependencies. Text (JSONL/markdown per cell) becomes a **derived
  artifact via `petri export`** for git commits and PR review. The JSONL→SQLite dashboard
  migration, the combined-log rollup, and the planned DuckDB layer all retire.
- **A `pydantic_evals` regression suite** turns the maintainer's manual v1–v4 prompt experiments
  into automated decomposition-quality gates.

**Scale:** 7 milestones, 81 issues (35 S / 45 M / 1 L), 15 genuine good-first-issues, every
milestone shipping a working release. **No backward compatibility** — v2 starts fresh; old dishes
remain on disk as readable JSONL.

---

## 2. Locked decision record (D1–D10)

Settled 2026-07-12 by the maintainer; issues must not re-litigate these.

| # | Decision | Key constraint |
|---|----------|----------------|
| D1 | Incremental strangler — CLI surface and `.petri/` layout stay; internals replaced subsystem-by-subsystem, shipping at every step | **Never require Docker or Postgres.** DBOS behind a swappable seam + Phase-0 spike |
| D2 | Durability: crash/interrupt resume, rate-limit-aware queueing, exactly-once LLM step recording, workflow observability | **OpenTelemetry is a first-class migration goal**, not an add-on |
| D3 | Inference: first-party pydantic-ai `Model` subclasses + any provider string in `petri.yaml` | Zero-API-key subscription auth stays possible |
| D4Δ | Storage (amended): one Petri-owned `petri.sqlite` per dish = domain truth (append-only `events` + `cells`/`edges` + `spans`/`usage` + views, stdlib sqlite3); DBOS SQLite = execution state only (separate file); `petri export` emits derived text for git/PR review | Events never edited or deleted — now schema-enforced; DuckDB dropped; `dashboard/migrate.py` and the combined-log rollup retire |
| D5 | Harness layer with **pi as default** (`--mode rpc`), Claude Code kept as adapter | pi is an optional runtime dependency, never a pip dependency |
| D6 | Agentic decomposer (search_cells tool, typed outputs, bedrock stop, counterarguments, per-parent caps) + `pydantic_evals` regression suite | Verdict-driven re-decomposition (#9) is a follow-on (M7) |
| D7 | OTLP export + trace view in the `petri launch` dashboard backed by the local `spans` table | Zero-infra default; DBOS Conductor/Console (proprietary) never bundled |
| D8 | No backward compatibility — v2 starts fresh | Old dishes stay on disk; reads tolerate their presence |
| D9 | Agent structure (enums, blocking semantics, debate protocol, pipeline) in typed code; content (instructions, vocabularies, pairings, models) in `petri.yaml` | "Configured, not hardcoded" survives with type safety |
| D10 | Milestones + small independently-landable issues + tracking epics + in-repo ARCHITECTURE-V2 doc | Label semantics defined once, in the doc |

## 3. What v2 preserves exactly (identity invariants)

- **Mechanical convergence** — all 6 blocking specialists' latest verdicts in their pass sets; a
  boolean check, no LLM judge — including weakest-link directives, short-circuits, and the
  3-iteration circuit breaker.
- **Append-only domain event sourcing** as the audit trail (the blog part-1 thesis) — now a
  schema-enforced `events` table (D4Δ), exportable as diffable text via `petri export`.
- **Composite cell keys** `{dish}-{colony}-{level}-{seq}` — which now double as deterministic DBOS
  workflow IDs, giving exactly-once cell processing for free.
- **Two-store separation** — domain truth (`petri.sqlite`) vs execution state (was `queue.json`,
  becomes the DBOS system DB in its own file); nothing crosses the boundary.
- **The 13-agent roster** (3 non-blocking leads + 10 specialists, 6 blocking) and the 4 debate
  pairings. (The shipped `petri.yaml` has a 14th entry, `socratic_questioner` — disposition is
  settled: it becomes a pre-pipeline utility outside the roster enum.)
- **The colony DAG as runtime data** (`graph/colony.py` — cycle detection, Kahn validation, BFS
  levels, bottom-up eligibility). pydantic-graph models only the fixed per-cell pipeline shape.
- **The constitution** as governance, with leads re-reading it per iteration (the documented
  drift-prevention pattern).
- **The CLI surface** (all 11 commands) and zero-infrastructure single-machine operation.
- **The micro-orchestrator identity**: v2 adopts substrate (typed model calls, checkpointed
  execution), not someone else's workflow abstraction. The workflow remains Petri's own.

## 4. Target architecture

### New packages

| Package | Purpose |
|---|---|
| `petri/harness/` | Harness abstraction (D5): pydantic-ai `Model` subclasses speaking CLI/RPC protocols — `PiModel` (default, `--mode rpc`), `ClaudeCodeModel`, provider-string passthrough; typed error taxonomy (`RateLimitedError.retry_after_seconds`, `AuthExpiredError`), shared retry policy (disable-able to avoid double-retry under DBOS) |
| `petri/execution/` | The durability seam (D1): a thin `ExecutionBackend` interface over DBOS-on-SQLite — durable workflows/steps, queues (concurrency, limiter), cancellation, fork, recovery. Swappable if the spike fails |
| `petri/pipeline/` | The per-cell validation pipeline as pydantic-graph typed nodes (Socratic → Research → Critique/Debates → Convergence → RedTeam → Evaluate), replacing the 14-state queue machine |
| `petri/agents/` | The agent factory (D9): builds 13 pydantic-ai Agents from typed structure + YAML content; debates as message-history hand-offs with shared usage + `UsageLimits`; eager config validation at startup |
| `petri/observability/` | OTel throughout (D2/D7): OTLP export + local `spans`/`usage` tables in petri.sqlite (age-pruned); domain attributes (dish/colony/cell, agent role, verdict, tokens/cost) |
| `petri/query/` | The read path (D4Δ): SQL views + typed query functions over petri.sqlite — convergence reads, validator reads, dashboard, analytics. Stdlib sqlite3; no DuckDB |
| `petri/evals/` | `pydantic_evals` decomposition-quality suite (D6): quota detection, bedrock stops, counterargument presence, restatement-children detection, cross-colony edge precision/recall |

### Module fates (current → v2)

| Current module | Fate | Target / rationale |
|---|---|---|
| `models.py` | Rewritten | Domain half (Cell, Event, typed payloads, composite keys) ports; `QueueState`/`QueueEntry` retire with the queue; `is_atomic` + `bedrock_reason` + typed claim-relation added to cells |
| `config.py` | Rewritten | Validated config object passed explicitly; import-time constants, `lru_cache` loader, and silent packaged-default fallback all retire (root cause of field issue #2) |
| `cli/*`, `cli_ui.py` | Kept | Same Typer commands — the strangler seam; internals rewired to the new engine |
| `engine/processor.py` | Replaced | → `petri/pipeline/` (pydantic-graph nodes) executed under `petri/execution/` |
| `engine/grow_loop.py` | Replaced | → DBOS queue draining + a colony orchestrator workflow that loops levels until no eligible cells remain (single-command full-colony growth) |
| `engine/propagation.py` | Wrapped | Pure DFS logic kept; flag-don't-auto-requeue human gate preserved |
| `engine/load_balancer.py` | Replaced | → DBOS queue flow control (`worker_concurrency`, `limiter`) |
| `engine/preflight.py` | Kept | Extended: pi/Node detection, version-range check |
| `storage/event_log.py` | Rewritten | Remains the single write seam, now writing rows to the `events` table (append-only, UNIQUE deterministic IDs — idempotent by constraint); `rollup_to_combined` retires |
| `storage/queue.py` | Replaced | → DBOS system DB + pipeline graph. The fcntl lock, `VALID_TRANSITIONS`, and the Windows blocker all retire |
| `storage/paths.py` | Rewritten | Shrinks dramatically under D4Δ: the tree becomes `petri-dishes/<dish_id>/petri.sqlite` + `exports/` (#14); lossy `parse_cell_id` consolidated away |
| `analysis/convergence.py` | Kept | Pure functions, ported verbatim onto typed verdicts — identity feature |
| `analysis/validators.py` | Rewritten | Same source-hierarchy rules as pure policy + `@output_validator` (ModelRetry) |
| `analysis/scanner.py` | Rewritten (M7) | Most drift categories disappear when structure is typed; shrinks to config validation + event-log integrity |
| `reasoning/decomposer.py` | Rewritten | → agentic decomposer (M3): search_cells, typed results, bedrock stop, per-parent caps in code |
| `reasoning/debate.py` | Rewritten | v1 "debates" are format-only stubs; v2 runs real multi-turn exchanges (M2) |
| `reasoning/ingest.py` | Wrapped | Kept; network I/O becomes DBOS steps; failure sentinels become typed errors (M7) |
| `reasoning/claude_code_provider.py` | Replaced | → `petri/harness/` Model subclasses; prompt builders extracted first (M1) |
| `graph/colony.py` | Kept | Runtime-data DAG (in-memory ops unchanged); persistence moves to the `cells`/`edges` tables, `dependents` becomes a view; gains integrity hardening in M6 |
| `dashboard/api.py` | Rewritten | FastAPI+SSE shell kept; reads → petri.sqlite via `petri/query/`; `/api/queue` → ExecutionBackend; `/api/proc` PTY bridge → workflow start/cancel/status |
| `dashboard/migrate.py` | Retired | The dashboard reads the domain store directly — no index rebuild |
| `dashboard/frontend.py` + `templates/frontend.html` | Kept | Field-hardened UI; gains the trace waterfall (M5) |
| `adapters/*` | Replaced | → `petri/harness/` backends; the generated-markdown-config approach retires |
| `templates/*.txt` | Retired | Prompt content moves to YAML instructions + typed agents |
| `defaults/` (petri.yaml, constitution.md) | Kept | The user-facing contract; schema updated to the D9 split |

## 5. Milestone roadmap

Dependency DAG (every milestone ships a working release):

```
M1-harness ──┬── M2-agents ──┬── M4-dbos ──┬── M5-otel ──┐
             └── M3-decomposer ──┘         └── M6-storage ─┴── M7-lifecycle
```

- **M1 — Harness & inference layer** (ships 0.4.0). pi RPC spike; `petri/harness/`
  (typed errors incl. rate-limit reset parsing, retry policy); pi transport + `PiModel`;
  `ClaudeCodeModel` port; provider-string passthrough; opt-in bridge so the *v1 engine* runs on
  any backend; offline TestModel/FunctionModel testing; ARCHITECTURE-V2 + field-reports docs
  bootstrap. The DBOS spike (M4.1) is explicitly schedulable in parallel with M1.
- **M2 — Agents on Pydantic AI**. Typed agent contract (13-agent roster settled);
  validated YAML loader (closes #2); agent factory with eager startup validation; real debates
  (message-history hand-offs, shared usage, `UsageLimits`); source-hierarchy as output validators;
  convergence ported exactly (EXECUTION_ERROR never passes); `petri agents list/check/run`.
- **M3 — Agentic decomposer + evals**. The `petri.sqlite` schema + migration mechanism lands
  first (D4Δ: events/cells/edges tables, UNIQUE event ids, `PRAGMA user_version`); the dish
  storage root flips to `petri-dishes/<dish_id>/petri.sqlite` (#14) with ID-parser consolidation;
  per-parent caps in code covering the root expansion (#3); bedrock stop persisted as required
  `is_atomic`/`bedrock_reason` (#10) + typed counterargument relations (#13) + negative examples
  (#12); `search_cells` tool building on the never-wired `find_shared_premises`, with cross-colony
  edges as typed rows (#11); seed overwrite guard (#5) + interim resumable seed; decomposition
  edit/adjust approval path (spec'd in v0, never built); the `pydantic_evals` suite with the
  archive corpus as ground truth.
- **M4 — Durable execution** (the L-sized `DBOSExecutionBackend` lives here). Phase-0
  DBOS-on-SQLite spike with written pass/fail criteria + golden grow fixture committed as
  exported text (no real grow data exists anywhere); `ExecutionBackend` seam; pipeline-as-graph
  consuming M2's real debates and convergence; idempotent event writes (deterministic IDs +
  UNIQUE constraint — `INSERT OR IGNORE`); queues with the reconciled throttling
  principle (max concurrency, limiter + parsed reset hints do *all* throttling); seed re-platformed
  onto DBOS (closes #8); `/api/queue` repointed before the v1 engine is deleted; sync-check
  reconciliation; human parking states (needs-human-guidance, sync-conflict); full config.py
  retirement; docs + conformance-test newcomer issues.
- **M5 — OpenTelemetry**. Mechanism spike (pure-OTel vs logfire-SDK-no-send); DBOS +
  pydantic-ai instrumentation; domain attributes on every command path (seed/grow/scan/feed/
  re-decompose); OTLP config; `spans`/`usage` tables in petri.sqlite as the local sink (own
  schema migration); dashboard trace waterfall (closes #15);
  per-cell/per-phase token cost accounting (restores the prototype's lost capability — field data
  shows self-reported cost logging simply doesn't happen: 1 event in ~2,700).
- **M6 — Storage & dashboard reads**. `petri/query/` as SQL views + typed query functions over
  petri.sqlite (retires `migrate.py` *and* the `combined.jsonl` rollup — no DuckDB anywhere);
  the new **`petri export`** command (derived JSONL/markdown for git & PR review, incremental,
  hookable into `grow`/CI); REST endpoints onto the query module; SSE from the event stream;
  `/api/proc` PTY bridge → workflow endpoints; `petri backup` (#6) via `VACUUM INTO` snapshots;
  graph integrity hardening; starter analytics views.
- **M7 — Iterative lifecycle** (some RFC-style by design). Verdict-driven
  re-decomposition (#9) split into proposal workflow + human approval gate; `petri feed`
  re-validation via `fork_workflow` with typed ingest steps; scanner re-scope; scheduled
  contradiction scans; convergence-point prioritization (explicitly replacing M4's level
  ordering); edge intelligence surfaced (edge count = importance, chains = fragility); the
  resurrected **Analyst** (read-only research-health monitoring over the analytics views +
  `spans`/`usage` tables); GitHub human-feedback re-entry RFC; worktree housekeeping.

### Issue index

Full bodies with acceptance criteria: `docs/v2/issues/<milestone>.md`.

### M1-harness

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M1-harness.1 | Bootstrap docs/ARCHITECTURE-V2.md and per-milestone tracking epics | S |  | relates:#2, relates:#3, relates:#4, relates:#5, relates:#6, relates:#7, relates:#8, relates:#9, relates:#10, relates:#11, relates:#12, relates:#13, relates:#14, relates:#15 |
| M1-harness.2 | SPIKE: Validate pi --mode rpc end-to-end and document the Node.js dependency story | M |  |  |
| M1-harness.3 | Create petri/harness package with typed error taxonomy and shared retry/backoff policy | S |  |  |
| M1-harness.4 | Build TestModel/FunctionModel offline test harness for harness-backed code | S |  |  |
| M1-harness.5 | Add fake_pi test stub and tests/README.md three-tier testing pattern doc | S | ✓ |  |
| M1-harness.6 | Implement pi RPC transport (LF-delimited JSONL over stdin/stdout) | M |  |  |
| M1-harness.7 | Implement PiModel: pydantic-ai Model subclass backed by the pi RPC transport | M |  |  |
| M1-harness.8 | Port Claude Code CLI backend to a pydantic-ai Model subclass (ClaudeCodeModel) | M |  |  |
| M1-harness.9 | Extract ClaudeCodeProvider prompt builders into petri/reasoning/prompts.py (behavior-preserving) | S | ✓ | relates:#3, relates:#12, relates:#13 |
| M1-harness.10 | Add harness resolution and pydantic-ai provider-string passthrough from petri.yaml | M |  | relates:#2 |
| M1-harness.11 | Add HarnessInferenceProvider bridge (part 1 of 2): decomposition-path methods on pydantic-ai Agents | M |  |  |
| M1-harness.12 | Add HarnessInferenceProvider bridge (part 2 of 2): assess_cell verdict discipline, match_evidence, and CLI wiring | M |  |  |

### M2-agents

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M2-agents.1 | Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster | M |  |  |
| M2-agents.2 | Add verdict-enum ↔ petri.yaml conformance tests for the 13-agent roster | S | ✓ |  |
| M2-agents.3 | Ship v2 petri.yaml agent-content schema with a validated loader (no import-time constants, no silent fallback) | M |  | #2 |
| M2-agents.4 | Build the 13-agent pydantic-ai roster via an agent factory (structure from code, content from YAML) | M |  |  |
| M2-agents.5 | Implement debates as programmatic agent hand-off with message_history, shared usage, and UsageLimits caps | M |  |  |
| M2-agents.6 | Port source-hierarchy enforcement to a pure policy core plus @output_validator raising ModelRetry | S |  |  |
| M2-agents.7 | Port mechanical convergence to the typed contract with semantics preserved exactly | S |  |  |
| M2-agents.8 | Add `petri agents list` and `petri agents check` CLI commands | S | ✓ | relates:#2 |
| M2-agents.9 | Add `petri agents run <agent>` one-shot execution with real usage and cost caps | M |  |  |

### M3-decomposer

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M3-decomposer.1 | Thread dish config through the decomposition path and retire import-time config constants | S |  | relates:#2 |
| M3-decomposer.2 | Consolidate dish-id resolution and cell-ID parsing into single canonical helpers | S |  | relates:#14 |
| M3-decomposer.3 | Define the petri.sqlite schema and migration mechanism | M |  |  |
| M3-decomposer.4 | Adopt the dish-scoped petri-dishes/<dish_id>/ layout with petri.sqlite as the domain store | M |  | #14 |
| M3-decomposer.5 | Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry | M |  |  |
| M3-decomposer.6 | Add bedrock stop condition, counterargument sub-claims, and negative examples to the decomposition agents | M |  | #10, #12, #13 |
| M3-decomposer.7 | Stop the decomposer treating layer limits as quotas: per-parent caps in code, no counts in prompts | S |  | #3 |
| M3-decomposer.8 | Add search_cells tool so decomposition creates cross-colony reference edges instead of duplicate cells | M |  | #11, relates:#14 |
| M3-decomposer.9 | Refuse to overwrite an existing colony in petri seed | S | ✓ | #5 |
| M3-decomposer.10 | Make seed resumable: checkpoint decomposition state and add petri seed --resume | M |  | relates:#8 |
| M3-decomposer.11 | Add an edit path to decomposition approval in petri seed | S |  |  |
| M3-decomposer.12 | Add a pydantic_evals regression suite for decomposition quality (organic counts, bedrock stops, counterarguments) | M |  | relates:#3, relates:#10, relates:#11, relates:#12, relates:#13 |

### M4-dbos

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M4-dbos.1 | Spike: validate DBOS-on-SQLite for crash-resume, queue concurrency, rate limiting, fork, cancel, and polling latency | M |  |  |
| M4-dbos.2 | Add ExecutionBackend seam and typed execution-state models; route grow/check/stop through it | M |  |  |
| M4-dbos.3 | Harden append_event: deterministic event IDs with skip-if-present idempotent appends | S |  |  |
| M4-dbos.4 | Model the per-cell validation pipeline as a pydantic-graph with typed transitions | M |  |  |
| M4-dbos.5 | Document the 14-state queue → pipeline-graph node mapping table with a generated mermaid diagram | S | ✓ |  |
| M4-dbos.6 | Port tests/unit/test_queue.py transition cases as pipeline-graph conformance tests | S | ✓ |  |
| M4-dbos.7 | Port Socratic, Research, and Critique+Debates phase runners into pipeline nodes with idempotent domain-event writes | M |  |  |
| M4-dbos.8 | Port ConvergenceCheck, RedTeam, and Evaluate phases plus the iterate loop and circuit breaker into pipeline nodes | M |  | relates:#9 |
| M4-dbos.9 | Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity | L |  |  |
| M4-dbos.10 | Add colony-level parent workflow: bottom-up eligibility scheduling and validation rounds | M |  |  |
| M4-dbos.11 | Re-platform petri seed onto the DBOS backend | M |  | #8 |
| M4-dbos.12 | Port sync-check as a startup/periodic reconciliation step | S |  |  |
| M4-dbos.13 | Repoint dashboard /api/queue onto the ExecutionBackend seam | S |  |  |
| M4-dbos.14 | Rewire petri grow/stop/feed lifecycle onto the durable backend (recovery, cancel, re-entry) | M |  | relates:#8 |
| M4-dbos.15 | Retire the v1 engine: delete processor, grow_loop, propagation, load_balancer, and the fcntl queue | M |  |  |

### M5-otel

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M5-otel.1 | Add telemetry bootstrap and petri.yaml `telemetry:` config section (petri/telemetry package) | M |  | relates:#2 |
| M5-otel.2 | Spike: choose the pydantic-ai OTel instrumentation mechanism (pure OTel vs logfire SDK in no-send mode) | S |  |  |
| M5-otel.3 | Enable DBOS OTLP tracing with hierarchical workflow→step spans behind the durability seam | S |  |  |
| M5-otel.4 | Instrument pydantic-ai agent runs with OpenTelemetry (no Logfire service required) | M |  |  |
| M5-otel.5 | Attach Petri domain span attributes: dish/colony/cell, agent role, verdict, iteration, tokens/cost | M |  |  |
| M5-otel.6 | Persist spans to the `spans` and `usage` tables in petri.sqlite (zero-infra local span sink) | M |  |  |
| M5-otel.7 | Add dashboard trace API endpoints reading the spans table in petri.sqlite | S | ✓ |  |
| M5-otel.8 | Add per-cell trace waterfall view to the petri launch dashboard | M |  | #15 |
| M5-otel.9 | Surface per-cell and per-phase token/cost accounting in the petri launch dashboard | S | ✓ |  |
| M5-otel.10 | Write observability docs: zero-infra default plus self-hosted Jaeger/SigNoz/Langfuse wiring | S | ✓ |  |

### M6-storage

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M6-storage.1 | Add petri/query/ read layer: typed SQL query functions over petri.sqlite | M |  | relates:#11 |
| M6-storage.2 | Move dashboard REST endpoints to the petri.sqlite query layer and add /api/edges | M |  | relates:#15 |
| M6-storage.3 | Rebuild /api/stream SSE on events-table tailing plus pydantic-ai event_stream_handler live progress | M |  | relates:#15 |
| M6-storage.4 | Retire dashboard/migrate.py and the disposable SQLite dashboard index | S | ✓ |  |
| M6-storage.5 | Add petri backup command with --list and a pre-destructive snapshot hook | S | ✓ | #6, relates:#5 |
| M6-storage.6 | Add petri backup --restore with overwrite guard and execution-state desync warning | S |  | relates:#6 |
| M6-storage.7 | Add petri export: derived JSONL/markdown artifacts for git and PR review | M | ✓ |  |
| M6-storage.8 | Replace the /api/proc PTY bridge and synchronous /api/seed with ExecutionBackend-driven runs | M |  | relates:#8 |
| M6-storage.9 | Graph integrity hardening: one authoritative colony topology with validated edges and lossless round-trips | M |  | relates:#11 |
| M6-storage.10 | Ship starter SQL analytics views in petri.sqlite over the event log | S | ✓ |  |

### M7-lifecycle

| # | Issue | Size | GFI | Field issues |
|---|-------|------|-----|--------------|
| M7-lifecycle.1 | RFC: Define the verdict-driven re-decomposition lifecycle (triggers, context payload, guards, human gate) | S |  | relates:#9 |
| M7-lifecycle.2 | Add re-decomposition trigger detection over cell verdict history | S |  | relates:#9 |
| M7-lifecycle.3 | Implement the re-decomposition proposal workflow: decompose_why with verdict context and colony-graph insertion | M |  | relates:#9, relates:#13 |
| M7-lifecycle.4 | Add the human approval gate for re-decomposition proposals to petri grow (--redecompose-on, approve/reject, enqueue) | S |  | #9 |
| M7-lifecycle.5 | Re-validate fed evidence via DBOS fork_workflow(start_step) instead of full cell reopen | M |  |  |
| M7-lifecycle.6 | Re-scope the contradiction scanner for v2 and run it as a DBOS scheduled workflow | M |  |  |
| M7-lifecycle.7 | Extend the dish edge registry with DishGraph queries, cross-colony cycle detection, and tombstones | M |  | relates:#11 |
| M7-lifecycle.8 | Add edge-intelligence metrics: importance, fragility, and convergence points | S | ✓ | relates:#11 |
| M7-lifecycle.9 | Surface edge intelligence in petri check, petri graph, and the dashboard | M |  | relates:#11, relates:#15 |
| M7-lifecycle.10 | Prioritize grow scheduling by convergence-point rank (opt-in) | S |  | relates:#11 |
| M7-lifecycle.11 | Resurrect the Analyst: read-only research-health monitoring over petri.sqlite analytics views and the spans/usage tables | M |  |  |
| M7-lifecycle.12 | RFC: Human-feedback re-entry via GitHub (review comments as focused directives) | S |  |  |
| M7-lifecycle.13 | Housekeeping: archive stale development worktrees and record canonical field-evidence corpora | S |  |  |

## 6. Cross-cutting design principles

1. **Throttling is reconciled** (the prototype said "never voluntarily throttle"; the dish runs
   used `max_concurrent: 2` to survive): v2 maxes out worker concurrency and lets the DBOS
   limiter + typed rate-limit classification (parsed reset time → durable sleep) do *all*
   throttling. The orchestrator never voluntarily idles and never burns retries against a wall.
2. **Fail loud, never fall back silently.** The v0.2.x "silent PASS" bug (unparseable output
   became the first valid verdict) is the canonical anti-pattern. Malformed harness frames raise
   typed errors; exhausted retries write `EXECUTION_ERROR` to the event log as a first-class,
   never-passing verdict.
3. **Exactly-once means exactly-once *recording*, not spend.** One LLM call per DBOS step bounds
   the re-pay blast radius to a single agent invocation; the plan and blog must not oversell this.
4. **Checkpoint-boundary contract**: only primitives, small `model_dump()` dicts, and path/ID
   references cross step boundaries (pickle, ~2MB guidance). Evidence stays on disk.
5. **Deterministic event IDs** derived from `(workflow_id, step, iteration)`, enforced by a
   UNIQUE constraint (`INSERT OR IGNORE`) — at-least-once retries physically cannot duplicate
   the audit trail (field logs show v1 did).
6. **The engine owns iteration numbers**, never agents (field logs show agent-assigned iterations
   violated monotonicity).
7. **Typed payloads before analytics**: SQL views are only built over pydantic-validated event
   schemas (archive logs show list↔dict drift within a single file).
8. **Capability manifests per harness backend**: search-requiring agent roles refuse to bind to
   backends without web-search/fetch (citation integrity is the product).

## 7. Top risks & mitigations (full set in the M-file risk sections)

| Risk | Mitigation baked into the plan |
|---|---|
| DBOS fails evaluation (SQLite mode immature; workload mismatch) | Phase-0 spike with written pass/fail criteria before anything is ported; `ExecutionBackend` seam keeps it swappable; fallback = pydantic-graph `FileStatePersistence` + custom SQLite queue |
| pi is a v0.x Node dependency for a Python tool | Optional runtime dep, never pip dep; `petri inspect` preflights node+pi with a tested version range; Claude Code adapter + API providers are first-class fallbacks |
| pi capability gap on web search → fabricated citations | Capability manifest per backend; search-requiring roles refuse unsupported backends; spike issue answers the tool-surface question first |
| DBOS code-checksum versioning strands in-flight workflows on `pip install -U` | Pin `application_version` to the Petri release string; signature-diff CI check on workflow shape; documented fork/resume recovery path |
| Pickle/2MB checkpoint ceiling | Checkpoint-boundary contract (above), enforced in review + tests |
| Two-store drift (JSONL vs DBOS state) under retries | Deterministic event IDs + ported sync-check reconciliation with a human-resolved sync-conflict state |
| Dashboard SSE across process boundaries (`run_stream` unsupported in workflows) | One event-bus design decided up front: stream events → OTel span events → spans table → SSE |
| Petri now owns a domain schema + migrations (new in D4Δ) | Defined once in M3; `PRAGMA user_version` forward-only migrations; migration tests in CI; spans/usage added as their own M5 migration |
| Cost story underwhelms if durability ships without decomposition fixes | M3 (decomposer) is sequenced before M4 completes; the eval suite + token attributes produce the before/after numbers |
| Rewrite sheds field-earned safety behavior | #5 guard + #6 backup are explicit v2 acceptance criteria; #5 and #8 ship together (the guard alone made failures permanent) |
| Contributor stack breadth (pydantic-ai + DBOS + OTel + TS harness) | D10 discipline: the ARCHITECTURE-V2 doc carries the why; 14 genuine good-first-issues; every issue independently landable |

## 8. Open questions (to resolve in ARCHITECTURE-V2 / early spikes)

The full list (16) travels with the plan; the load-bearing ones:

1. DBOS spike pass/fail thresholds (resume latency after `kill -9`, dispatch latency at
   concurrency 8 on SQLite, checkpoint DB growth per 1k cells).
2. pi's per-provider tool surface (does web search exist outside Anthropic backends?) and
   zero-API-key auth modes under pi — both gate the "pi as default" UX.
3. How config-driven `redirect_on_block` routes are typed under pydantic-graph's static edges
   (generic redirect node vs generated unions vs narrowed config).
4. `petri stop` semantics under DBOS (cancel cell vs colony workflows; resume vs re-enqueue).
5. Human-in-the-loop mechanism: DBOS `recv`/`set_event` vs pydantic-graph interrupt+persistence.
6. pydantic-graph API choice: classic `BaseNode` vs the newer `GraphBuilder` (v2-era) API.
7. Windows support declaration once `fcntl` is gone (core yes, PTY dashboard terminal deferred?).
8. The cost-benchmark protocol backing the migration's numbers (baseline, metric, publication).
9. `petri export` ergonomics (auto-hook on grow completion vs manual; diff-stable deterministic
   output) and spans/usage retention defaults (D4Δ follow-ons).

## 9. Field-issue traceability

| Field issue | Closed by |
|---|---|
| #2 config split-brain | M2 — validated loader (root cause: import-time constants) |
| #3 budget-as-quota | M3 — per-parent caps in code (incl. root expansion) |
| #5 seed destroys colony | M3 — overwrite guard |
| #6 backup | M6 — `petri backup` + pre-destructive snapshots |
| #8 resume interrupted seed | M4 — seed on DBOS (interim file checkpoint in M3) |
| #9 verdict-driven re-decomposition | M7 — proposal workflow + approval gate |
| #10 bedrock/triviality | M3 — persisted `is_atomic` + evaluator |
| #11 cross-colony search | M3 — `search_cells` + edge registry |
| #12 negative examples | M3 — prompt + eval Cases |
| #13 counterarguments | M3 — typed claim relations |
| #14 dish-scoped layout | M3 — storage-root flip to `petri-dishes/<dish_id>/petri.sqlite` (early, before dependents) |
| #15 dashboard | M5 — trace waterfall (+ related M6/M7 work) |

## 10. Evidence base & method

This plan was produced from: a full read of every source file in the repo; every file in the four
`.claude/worktrees` (1,376 files, coverage-audited — duplicates byte-verified and read once);
verified research on pydantic-ai 2.9, pydantic-graph, DBOS (SQLite default confirmed), and pi
v0.80; and the maintainer's field documents — chiefly `orchestration-improvements.md` (the
hand-designed durable-execution architecture of March 2026) and `petri_improvements.md` (the
decomposition field report with 15 locally-validated patches). Drafts were adversarially verified
on three axes (API accuracy against live docs, coverage against field issues and code-review pain
points, contributor-readiness) — 41 findings, all dispositioned; a final referee validated
cross-milestone consistency (dependency DAG, single-closer traceability, label discipline).

Calibration note from the maintainer: the dish datasets are behavioral evidence ("noise to
understand behavior"), not a quality target — the hand-crafted archive corpus (101 claims,
128 edges) is the calibration standard, and the eval suite treats it that way.

Amendment note (D4Δ, same day): during diagram review the maintainer challenged the three-layer
storage design. The review chain — agents + the dashboard mediate all access (dissolving the
"humans grep files" argument), SQLite makes idempotency/atomicity/snapshot-reads by-construction
rather than by-convention, and DuckDB's only job was querying JSONL — collapsed storage to one
Petri-owned SQLite file per dish with text as a derived export. The decision record, issues,
diagrams, and this document were re-propagated accordingly.

## 11. Next steps

1. Review this plan + the M1 backlog (`docs/v2/issues/M1-harness.md`).
2. Cut the issues on GitHub via `scripts/create_v2_issues.py` (labels, milestones, epics,
   sub-issues, and dependency links; `docs/ARCHITECTURE-V2.md` and `docs/field-reports.md` are
   committed so every issue link resolves).
3. Start the two parallel spikes: pi RPC (M1.2) and DBOS-on-SQLite (M4.1) — they gate everything
   and neither depends on the other.
4. Publish blog part 2 and point readers at the good-first-issue set.

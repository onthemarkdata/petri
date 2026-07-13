# M5-otel — Issue Backlog

> Tracking epic for milestone **M5-otel**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Make OpenTelemetry a first-class capability across the entire Petri process (D2 addendum + D7): DBOS workflow→step spans, pydantic-ai agent/model/tool spans, and Petri domain attributes (dish/colony/cell, agent role, verdict, tokens/cost) flow through one TracerProvider. Users get a zero-infra default — per-cell trace waterfalls inside `petri launch` backed by the `spans` and `usage` tables in the dish's petri.sqlite (amended D4) — plus configurable OTLP export to self-hosted Jaeger/SigNoz/Langfuse via petri.yaml. Two field-proven failure modes are what this milestone retires: (1) agent self-reported usage does not happen — in ~2,700 real events from the prototype Petri was extracted from, exactly ONE token_usage event was ever self-reported despite full schema, API, and dashboard support, so token/cost capture MUST happen automatically at the Model/harness layer; and (2) the prototype's only workflow observability was a bash loop (monitor-agents.sh) polling file mtimes and auto-committing every five minutes — ps-based liveness guessing, now replaced by real spans. The local span sink is the `spans`/`usage` tables in petri.sqlite (age-pruned), added by their own small schema migration (PRAGMA user_version bump via the mechanism from M3-decomposer) and written/read with stdlib sqlite3 only — fully independent of M6-storage's petri/query/ read layer and `petri export` work (no ordering dependency on M6-storage in either direction). This milestone also owns the package-version drift fix (petri/__init__.py + the dashboard frontend's hardcoded version); M6-storage does not claim it.

**Shippable release.** A v2 pre-release (next alpha/minor) where every `petri seed`/`petri grow` run emits end-to-end OTel traces (CLI root span → DBOS workflow/step spans → agent run/model-request/tool spans, all carrying petri.* domain attributes including verdict and token/cost), `petri launch` shows a per-cell trace waterfall and per-cell/per-phase token/cost totals with nothing external running, and a documented `telemetry:` section in petri.yaml points OTLP at any self-hosted backend.

**Depends on milestones:** M1-harness (harness abstraction + pydantic-ai Model port per D3/D5 — typed rate-limit/errors and Model-layer usage reporting that span attributes consume), M2-agents (agent factory per D9 — the Agents that get instrumented), M3-decomposer (petri.sqlite schema + PRAGMA user_version migration mechanism per amended D4 — the spans/usage span-sink tables land as their own small follow-on migration), M4-dbos (durability seam + per-cell pipeline graph per D1/D2 — the workflow/step spans and pipeline nodes that carry domain attributes)

**Milestone risks:**
- pydantic-ai 2.9.0's OTel-native instrumentation path is unverified — the standalone mechanism spike (dependency-free, schedulable immediately) must run before the instrumentation issue; if the logfire SDK turns out to be required in no-send mode, that adds a dependency the maintainer may not want and reshapes the implementation issue.
- DBOS OTLP span behavior on the SQLite system database (Petri's mandated default) is documented but example-verified mostly on Postgres; span parenting between DBOS step spans and pydantic-ai agent spans across the seam may not nest automatically and could require explicit context propagation or span links.
- petri.sqlite is shared across concurrent processes: `petri grow` writes spans/usage rows (and domain events) while `petri launch` reads traces and age-prunes on startup. WAL mode (from the M3-decomposer schema mechanism) gives concurrent readers during writes, but writers contend — a naive exporter holding long transactions would stall the pipeline or surface 'database is locked' errors; the span-store issue makes the transaction/busy_timeout strategy an explicit reviewed decision.
- Span volume: 13 agents × up to 6 phases × 3 iterations per cell, times hundreds of cells, can make the spans/usage tables large and the waterfall slow; sample_ratio and retention_days (age-pruning) defaults may need tuning after the first dogfood run.
- The age-pruned spans/usage tables share petri.sqlite with the never-edited, never-deleted events table; a prune or migration bug that touched domain tables would violate the event-sourcing invariant (amended D4) — the span-store ACs pin pruning and the spans/usage migration to telemetry tables only, and review must hold that line.
- Trace context does not propagate into harness subprocesses (pi in rpc mode, Claude Code CLI): their internal activity is a single opaque span, and token/cost attributes may be unavailable on subscription-auth CLI backends — the attribute table and the cost dashboard must be honest about omissions (n/a, never fabricated zeros), echoing the field lesson that agent-side token tracking never materializes in practice.
- Cross-milestone coupling: issues here reference the durability seam and pipeline nodes (M4-dbos), the agent factory (M2-agents), the harness adapters (M1-harness), and the petri.sqlite schema/migration mechanism (M3-decomposer) by role rather than final path; file paths in those issue bodies must be pinned when those milestones land. M7-lifecycle's scan/feed/re-decomposition issues must carry the matching root-span ACs defined in the domain-attributes issue.
- Automated DOM/JSON-level frontend assertions require the waterfall tree-builder to be factored for headless execution (e.g. node-run extracted JS); if the single-file template makes extraction awkward, the test harness cost grows — acceptable this milestone, but reinforces that the 3415-line frontend.html may need splitting when a later milestone adds more views.

---

## M5-otel.1 — Add telemetry bootstrap and petri.yaml `telemetry:` config section (petri/telemetry package)

**Size:** M · **Labels:** migration-v2, observability · **Field issues:** relates:#2

**Context.** v1 has zero tracing; the only cross-cutting visibility is the hand-rolled CellProgressEvent plumbing (petri/engine/processor.py:57, fire() closure at processor.py:1536-1552). M5-otel needs one shared, idempotent OTel bootstrap that everything else (DBOS, pydantic-ai instrumentation, domain attributes, the local span sink writing the `spans`/`usage` tables in petri.sqlite) plugs into, driven by a new `telemetry:` section in petri.yaml.

**Scope.**
- New package `petri/telemetry/` with `setup.py`: `init_telemetry(config) -> TracerProvider` — builds a Resource (`service.name=petri`, `service.version`, `petri.dish_id`), registers an OTLP span exporter only when an endpoint is configured, is idempotent (second call is a no-op), and never raises into caller code.
- New Pydantic `TelemetryConfig` model (enabled: bool = true, `otlp: {traces_endpoint, logs_endpoint, headers, protocol}`, `local: {enabled: bool = true, retention_days}`, `sample_ratio: float = 1.0`, `service_name`). Validated at load — no raw-dict `.get()` accessors (the config accessors in petri/config.py:140-240 bypass the Pydantic models today; do not repeat that).
- `get_telemetry_config()` accessor in petri/config.py. It must NOT become a module-level import-time constant like LLM_INFERENCE_MODEL/MAX_ITERATIONS/MAX_CONCURRENT/AGENT_TOOLS (config.py:247-250) — that import-time freeze caused a real config split-brain in the field: modules froze package defaults at import time, so dish-level petri.yaml overrides were silently ignored (field issue #2; see docs/field-reports.md). It must honor dish-level config (load_dish_config, config.py:30-41) and env-var overrides (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`).
- Add `telemetry:` block with commented defaults to petri/defaults/petri.yaml (324 lines; sits naturally after `max_concurrent` at line 38).
- Add `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http` to core dependencies in pyproject.toml:22-30 (core, not an extra — matches the precedent of commit 28586fe moving dashboard deps to core). Default protocol http/protobuf to avoid the grpc binary dependency.

**Out of scope.** DBOS wiring (next issue), pydantic-ai instrumentation, the local span sink (span-store issue), dashboard, docs.

**Touched files.** petri/config.py (:18-27 lru_cache loader, :30-41 load_dish_config, :247-250 constants to avoid emulating), petri/defaults/petri.yaml, pyproject.toml:22-30. **New:** petri/telemetry/__init__.py, petri/telemetry/setup.py, tests/unit/test_telemetry_setup.py.

**Acceptance criteria:**
- [ ] pytest: TelemetryConfig parses a dish petri.yaml `telemetry:` block and overrides package defaults; a missing section yields defaults (enabled=true, no OTLP endpoint, local.enabled=true)
- [ ] pytest: init_telemetry with no OTLP endpoint registers a TracerProvider with zero OTLP exporters and makes no network calls
- [ ] pytest: init_telemetry with `otlp.traces_endpoint` set registers an OTLP exporter pointed at that endpoint; OTEL_EXPORTER_OTLP_ENDPOINT env var overrides the yaml value
- [ ] pytest: calling init_telemetry twice does not create duplicate providers/exporters
- [ ] pytest: `telemetry.enabled: false` results in a no-op provider (spans created via the petri tracer are non-recording)
- [ ] pytest: importing petri.telemetry in a directory with no .petri/ raises nothing and reads no config (no import-time side effects)
- [ ] pytest: malformed telemetry section (e.g. sample_ratio: 'lots') fails loudly at config load with a field-level Pydantic error

---

## M5-otel.2 — Spike: choose the pydantic-ai OTel instrumentation mechanism (pure OTel vs logfire SDK in no-send mode)

**Size:** S · **Labels:** migration-v2, observability, spike

**Context.** pydantic-ai's commonly documented tracing path is `logfire.instrument_pydantic_ai()` ("traces runs, tool calls, and token usage"), but D7 forbids requiring a Logfire account/service — users must be able to point OTLP at self-hosted Jaeger/SigNoz/Langfuse or use the zero-infra local store. What is NOT settled: does pydantic-ai 2.9.0 offer OTel-native instrumentation that emits to a plain global TracerProvider (e.g. an agent-level instrument switch / InstrumentationSettings), or is the logfire SDK required in a local/no-send configuration? Verify against the installed package — do not invent APIs. The implementation issue in this milestone depends on this spike's outcome.

**Scope (timeboxed, ~1 day).**
- With pydantic-ai 2.9.0 installed, build a toy Agent using TestModel/FunctionModel and evaluate both candidate paths: (a) any first-party instrumentation switch that emits to a vanilla OTel TracerProvider with no logfire import; (b) the logfire SDK configured with send_to_logfire disabled, exporting only to our provider.
- For each path record: which spans are produced (agent run / model request / tool call), whether token-usage attributes appear (gen_ai.* or equivalent) and under what names, what extra dependencies get pulled in, and any network traffic (must be zero traffic to logfire.dev).
- **Deliverable:** an ADR note in docs/ARCHITECTURE-V2.md naming the chosen mechanism, stating whether the logfire SDK becomes a dependency (and the pinning policy if so), and listing the attribute names token usage arrives under.

**Out of scope.** Production wiring (the implementation issue), DBOS span-parenting behavior (asserted in the implementation issue), domain attributes.

**Touched files.** docs/ARCHITECTURE-V2.md (ADR note). No production code changes.

**Acceptance criteria:**
- [ ] ADR note merged in docs/ARCHITECTURE-V2.md naming the chosen mechanism and stating whether the logfire SDK is required; if required, the no-send configuration is spelled out with proof (network capture or exporter inspection) that no logfire.dev endpoint is contacted
- [ ] A minimal runnable snippet (committed alongside the ADR note or as a scratch test) shows agent-run and model-request spans arriving at an in-memory exporter via the chosen mechanism
- [ ] Token-usage attribute names documented in the ADR note, or their absence recorded with a follow-up issue filed
- [ ] Spike stayed timeboxed: findings recorded even if inconclusive, with the open question narrowed

---

## M5-otel.3 — Enable DBOS OTLP tracing with hierarchical workflow→step spans behind the durability seam

**Size:** S · **Labels:** migration-v2, observability, durable-execution · **Depends on:** Add telemetry bootstrap and petri.yaml `telemetry:` config section (petri/telemetry package)

**Context.** DBOS ships OpenTelemetry support: install the `dbos[otel]` extra, set `enable_otlp=True` plus `otel_attribute_format='semconv'` in DBOSConfig, and DBOS emits hierarchical spans (step spans as children of workflow spans) to the global tracer, with `otlp_traces_endpoints`/`otlp_logs_endpoints` for self-hosted backends. D1 requires DBOS to stay swappable behind an internal seam, so all of this wiring must live inside the seam's DBOS backend module (created in M4-dbos), not leak into pipeline code. This delivers D2 requirement 4 (workflow observability) and supersedes tracing-by-progress-callback (petri/engine/processor.py:57, :1536-1552 — those callbacks stay for the CLI spinner UX only).

**Scope.**
- Add `dbos[otel]` extra to pyproject.toml:22-30.
- In the v2 DBOS runtime module behind the durability seam: build DBOSConfig with `enable_otlp=True`, `otel_attribute_format='semconv'`, and thread `otlp_traces_endpoints`/`otlp_logs_endpoints` from `get_telemetry_config()` (previous issue) — petri.yaml is the single config source, no separate DBOS config file.
- Enforce init order: `init_telemetry()` runs before `DBOS.launch()` so DBOS emits into Petri's TracerProvider (DBOS emits to the global tracer).
- `telemetry.enabled: false` → DBOS launched without OTLP enabled.
- Extend the seam interface with an optional telemetry hook so no module outside the seam imports dbos for tracing purposes.
- **Open question (verify, do not assume):** DBOS documents enable_otlp for both SQLite and Postgres system databases, but published examples skew Postgres; confirm identical span emission on the SQLite default (DBOS's own dbos.sqlite system DB — never shared with the dish's petri.sqlite per amended D4; D1: SQLite only, no Postgres/Docker ever) and record the finding in docs/ARCHITECTURE-V2.md (D10).

**Out of scope.** Domain attributes (separate issue), pydantic-ai spans, exporters beyond what the bootstrap issue registered.

**Touched files.** pyproject.toml:22-30; the v2 runtime bootstrap module behind the durability seam (name per M4-dbos — pin the path in the PR); petri/telemetry/setup.py (init-order assertion). **New:** tests/integration/test_dbos_otel_spans.py.

**Acceptance criteria:**
- [ ] pytest (integration, in-memory span exporter, SQLite system DB): a minimal @DBOS.workflow with two @DBOS.step calls run through the petri runtime produces one workflow span with two child step spans (verified via parent_span_id)
- [ ] pytest: setting telemetry.otlp.traces_endpoint in petri.yaml results in that endpoint appearing in the DBOSConfig dict passed through the seam (asserted on the seam boundary, no live export needed)
- [ ] pytest: telemetry.enabled=false → DBOS is launched with OTLP disabled and the workflow run emits no spans
- [ ] pytest: init_telemetry is invoked before DBOS.launch() in the runtime bootstrap (order asserted via a recording fake)
- [ ] grep-level check in review: no module outside the seam backend imports dbos for telemetry
- [ ] ADR note added to docs/ARCHITECTURE-V2.md recording the SQLite-vs-Postgres span-emission verification result

---

## M5-otel.4 — Instrument pydantic-ai agent runs with OpenTelemetry (no Logfire service required)

**Size:** M · **Labels:** migration-v2, observability, agents · **Depends on:** Add telemetry bootstrap and petri.yaml `telemetry:` config section (petri/telemetry package); Spike: choose the pydantic-ai OTel instrumentation mechanism (pure OTel vs logfire SDK in no-send mode)

**Context.** pydantic-ai agent/model/tool spans are the middle layer of the trace, between DBOS workflow/step spans (previous issue) and Petri domain attributes (next issue). D7 requires that users can point at self-hosted Jaeger/SigNoz/Langfuse — a Logfire account/service must never be required. The instrumentation-mechanism spike (this milestone) settles HOW: a pure-OTel instrumentation switch vs the logfire SDK in no-send mode; this issue implements whichever the ADR note chose, wiring agent instrumentation into the shared TracerProvider for every Agent built by the M2-agents factory (13-agent roster) and — once M3-decomposer lands — the decomposer agent. It replaces the hand-threaded on_progress streaming plumbing as the observability channel (reasoning/claude_code_provider.py:84-167 stream parsing, decomposer.py:74 callback threading; the CLI spinner keeps its callbacks until dashboard SSE moves to event_stream_handler per the locked constraints — no run_stream() inside DBOS workflows).

**Scope.**
- New `petri/telemetry/instrument_agents.py`: single `instrument_agents()` entry point called from runtime bootstrap after init_telemetry; applies the ADR-chosen mechanism so every Agent run emits run/model-request/tool-call spans and token-usage data into the shared provider.
- If the ADR chose the logfire SDK path: pin it, configure it to export only to our TracerProvider/OTLP (no logfire.dev traffic), and document why in the ADR note.
- Confirm span parenting when agent runs execute inside DBOS workflow steps (spans should nest under the step span from the previous issue); if parenting does not hold automatically, document the observed linking behavior and file a follow-up.
- Instrumentation must not depend on streaming APIs (locked constraint: no run_stream() inside DBOS workflows).

**Out of scope.** Petri domain attributes (next issue), pi-harness subprocess-internal spans (the pi process is a black box this milestone — its call is one span, not a subtree), re-running the mechanism investigation (done in the spike).

**Touched files.** the v2 agent factory module (from M2-agents) and runtime bootstrap; petri/telemetry/setup.py. Current-code references for what this supersedes: reasoning/claude_code_provider.py:84-167 (_process_stream_lines/_extract_text_delta), reasoning/decomposer.py:74. **New:** petri/telemetry/instrument_agents.py, tests/unit/test_agent_instrumentation.py.

**Acceptance criteria:**
- [ ] Implementation matches the mechanism recorded in the spike's ADR note in docs/ARCHITECTURE-V2.md; any deviation updates the ADR in the same PR
- [ ] pytest (TestModel/FunctionModel + in-memory exporter): agent.run() produces an agent-run span and at least one model-request span
- [ ] pytest: an agent with a registered tool produces a tool-call child span when the tool is invoked
- [ ] pytest: token usage appears on span attributes (gen_ai.* or the ADR-documented equivalent) when the model reports usage
- [ ] pytest: no span exporter in the test run is configured with a logfire.dev/logfire cloud endpoint (applies whether or not the logfire SDK is a dependency)
- [ ] pytest (integration with the DBOS tracing issue): an agent run inside a @DBOS.step nests under (or is explicitly linked to) the step span; observed behavior asserted and documented

---

## M5-otel.5 — Attach Petri domain span attributes: dish/colony/cell, agent role, verdict, iteration, tokens/cost

**Size:** M · **Labels:** migration-v2, observability, agents, lifecycle · **Depends on:** Enable DBOS OTLP tracing with hierarchical workflow→step spans behind the durability seam; Instrument pydantic-ai agent runs with OpenTelemetry (no Logfire service required)

**Context.** DBOS and pydantic-ai spans describe execution, not the domain. D7 mandates Petri attributes (dish/colony/cell id, agent role, verdict, cost) on spans so traces answer "which cell, which agent, which verdict, what did it cost". This also lands the field-proven lesson that token measurement must happen at the harness level, automatically: in ~2,700 real events from the prototype Petri was extracted from, exactly one token_usage event was ever self-reported by an agent despite full schema, API, and dashboard support (see docs/field-reports.md). Token counts now come from pydantic-ai RunUsage instead of agent self-estimates.

**Scope.**
- New `petri/telemetry/attributes.py`: a typed, documented `petri.*` attribute namespace — petri.command, petri.dish_id, petri.colony_id, petri.cell_id (composite key, build_cell_key at petri/models.py:512-514), petri.level, petri.seq, petri.phase, petri.agent_role, petri.agent_blocking, petri.verdict, petri.iteration, petri.weakest_link, petri.tokens_input, petri.tokens_output, petri.cost_usd — with pure helper functions (set_cell_attributes, set_agent_attributes, set_verdict_attributes, root_command_span) and no global state.
- Apply helpers in the v2 per-cell pipeline graph nodes (the nodes replacing the v1 phase runners: _run_socratic_phase processor.py:823, _run_phase1 processor.py:1077, _run_phase2 processor.py:1152-1250, _run_convergence processor.py:1253, _run_red_team processor.py:1361, _run_evaluation processor.py:1422; verdict shape per _verdict_data processor.py:257-273) and on the cell workflow span via the seam (DBOS exposes custom span attributes via DBOS.span / SetWorkflowAttributes — use whichever the seam exposes).
- Root command spans for ALL entry points, not just seed/grow: wrap `petri seed`, `petri grow`, `petri scan`, and `petri feed`/re-validation invocations in a root span carrying petri.command and petri.dish_id, so a whole run is one trace (D2: OTel across the WHOLE process). Touch the command closures at petri/cli/grow.py:25-47 and petri/cli/seed.py:175-187 plus the scan and feed entry points in petri/cli/ (thin wrappers only). Re-decomposition has no v1 command: the root_command_span helper and its attribute-table entry ship HERE, and M7-lifecycle's scan, feed/fork-revalidation, and re-decomposition issues carry matching ACs that apply these helpers to their reworked paths (coordinated cross-milestone).
- Token/cost: read pydantic-ai RunUsage per agent run; cost_usd computed from a per-model price table in petri.yaml if present, else attribute omitted. When the backend cannot report usage, attributes are OMITTED, never zero/fabricated. **Open question:** whether pi's `--mode rpc` reports token usage in its JSONL protocol — verify against the pi harness adapter from M1-harness; if absent, document the gap in the attribute table.

**Out of scope.** Persisting spans (next issue), dashboard rendering, metrics/logs signals (traces only this milestone), the M7-lifecycle rework of scan/feed/re-decomposition themselves (only their span hooks and attribute names are defined here).

**Touched files.** v2 cell-pipeline graph node modules and durability seam (paths per M4-dbos), petri/cli/grow.py:25-47, petri/cli/seed.py:175-187, petri/cli/ scan and feed entry points. Current-code references: petri/models.py:512-514, petri/engine/processor.py:257-273/:823/:1077/:1152-1250/:1253/:1361/:1422. **New:** petri/telemetry/attributes.py, tests/unit/test_span_attributes.py.

**Acceptance criteria:**
- [ ] Attribute table (name, type, when present) committed to docs/ARCHITECTURE-V2.md, including petri.command values for seed, grow, scan, feed, and the planned re-decomposition path (docs issue later copies it into user docs)
- [ ] pytest (FunctionModel + in-memory exporter): one cell run through the v2 pipeline yields phase spans carrying petri.cell_id, petri.agent_role and petri.verdict, and a convergence span carrying petri.iteration and outcome
- [ ] pytest: petri.cell_id equals build_cell_key output for the cell under test
- [ ] pytest: token attributes present when the test model reports usage; absent (key not set) when it does not
- [ ] pytest: attribute helpers are pure functions — same inputs produce same attribute dicts, no module-level mutable state
- [ ] pytest: `petri grow --dry-run` in a seeded fixture dish produces a root span named for the command with petri.command and petri.dish_id set (in-memory exporter)
- [ ] pytest: `petri scan` and `petri feed <file>` on a fixture dish each produce a root span named for the command with petri.command and petri.dish_id set (in-memory exporter)
- [ ] Matching root-span/attribute ACs recorded on M7-lifecycle's scan, feed/fork-revalidation, and re-decomposition issues (cross-milestone coordination note in the tracking epics)

---

## M5-otel.6 — Persist spans to the `spans` and `usage` tables in petri.sqlite (zero-infra local span sink)

**Size:** M · **Labels:** migration-v2, observability, storage · **Depends on:** Add telemetry bootstrap and petri.yaml `telemetry:` config section (petri/telemetry package)

**Context.** D7's zero-infra requirement: traces visible in `petri launch` with nothing external running. Spans need a local sink the dashboard can query. Amended D4 (see docs/ARCHITECTURE-V2.md) sets the pattern: `petri.sqlite` — one Petri-owned SQLite file per dish — holds the domain tables (`events`, `cells`, `edges`) plus the execution-telemetry sink tables this issue adds: `spans` and `usage`, written by an OTel exporter via stdlib sqlite3 and age-pruned. Note on independence: this issue owns the spans/usage tables, their schema migration, and their reader API, fully independent of M6-storage's petri/query/ read layer; there is NO ordering dependency on M6-storage in either direction (M6-storage separately retires the disposable SQLite dashboard index built by petri/dashboard/migrate.py — rebuild_sqlite migrate.py:55, incremental_sync migrate.py:103, invoked from petri/cli/launch.py:171-173). What this issue DOES build on is M3-decomposer's petri.sqlite schema/migration mechanism (PRAGMA user_version forward-only migrations, WAL mode, write-seam module).

**Scope.**
- Schema migration: add `spans` and `usage` tables to petri.sqlite as their own small forward-only migration bumping PRAGMA user_version (mechanism from M3-decomposer). `spans` columns: trace_id, span_id, parent_span_id, name, kind, start_time_ns, end_time_ns, status_code, status_message, attributes (JSON), resource (JSON). `usage` columns: one row per usage-bearing span — span_id, trace_id, cell_id, phase, agent_role, tokens_input, tokens_output, cost_usd (nullable) — so token/cost accounting becomes a plain SQL aggregate (the costs issue in this milestone reads it).
- New `petri/telemetry/span_store.py`: an OTel SpanExporter registered by init_telemetry when `telemetry.local.enabled` (default true), wrapped in a BatchSpanProcessor, writing span rows in short transactions via stdlib sqlite3 and extracting usage rows from the petri.tokens_input/petri.tokens_output/petri.cost_usd span attributes (canonical names per the domain-attributes issue's attribute table; synthetic spans carrying those names are sufficient for development and tests — no ordering constraint beyond coordinating the names). No zero-filled usage rows: spans without usage attributes produce no usage row (omitted-never-fabricated policy).
- Concurrency: WAL mode (established by the M3 schema issue) gives concurrent readers during writes; set a busy_timeout and keep exporter transactions short so `petri grow` (writing spans) and `petri launch` (reading) coexist. Document the transaction/busy_timeout strategy in the PR. The former open question about single-writer analytic stores is closed by construction — spans are rows in petri.sqlite.
- Reader API: `query_spans(petri_dir, cell_id=None, trace_id=None, limit=...)` returning typed rows via stdlib sqlite3 — the single entry point the dashboard trace API will use.
- Retention: age-prune — delete spans/usage rows older than `telemetry.local.retention_days` on `petri launch` startup (hook near the current rebuild call at launch.py:171-173). spans and usage are the ONLY tables ever pruned; the `events` table is append-only, never edited, never deleted (amended D4 invariant — prune must be structurally unable to touch domain tables).
- Exporter must never raise into pipeline code (locked/unwritable DB → drop + warn once); failure to persist spans must not fail a cell run.
- Zero new dependencies: stdlib sqlite3 only — amended D4 removes DuckDB from the stack entirely; no third-party storage dependency anywhere in this issue.

**Out of scope.** Dashboard endpoints/UI (following issues), remote OTLP export (bootstrap issue), the domain tables (spans are execution telemetry, NOT domain events — the append-only `events` table remains the domain source of truth per amended D4 and this issue never writes to it), `petri export` (M6-storage).

**Touched files.** petri/telemetry/setup.py (register exporter), petri/cli/launch.py:171-173 (prune hook), the petri.sqlite migration module from M3-decomposer (pin the path in the PR). Current-code references: petri/dashboard/migrate.py:55/:103 (the SQLite-index precedent M6-storage retires). **New:** petri/telemetry/span_store.py, the spans/usage schema migration, tests/unit/test_span_store.py.

**Acceptance criteria:**
- [ ] pytest: exporting 100 synthetic spans then querying via query_spans returns 100 rows with trace_id/span_id/parent_span_id intact and petri.* attributes JSON round-tripped
- [ ] pytest: query_spans(cell_id=...) filters correctly using the petri.cell_id attribute
- [ ] pytest: spans carrying petri.tokens_input/petri.tokens_output attributes produce matching usage rows; spans without usage attributes produce none (no zero-filled rows)
- [ ] pytest: the spans/usage migration applies cleanly to a petri.sqlite at the prior user_version, bumps user_version, and is a no-op on re-open; events/cells/edges table contents are identical before and after
- [ ] pytest: one process exporting spans while another reads via query_spans (multiprocessing test, WAL mode) — all committed spans read back without corruption or unhandled 'database is locked' errors
- [ ] pytest: prune() removes spans/usage rows older than retention_days, leaves newer ones, and never deletes from events/cells/edges (row counts asserted)
- [ ] pytest: exporter against a read-only or locked database logs a warning once and does not raise; the calling code path completes
- [ ] pytest: everything written lands in the dish's petri.sqlite — no side files, no writes outside the dish directory, no network; no third-party storage dependency is added (storage uses stdlib sqlite3)

---

## M5-otel.7 — Add dashboard trace API endpoints reading the spans table in petri.sqlite

**Size:** S · **Labels:** migration-v2, observability, dashboard, good-first-issue · **Good first issue** · **Depends on:** Persist spans to the `spans` and `usage` tables in petri.sqlite (zero-infra local span sink)

**Context.** The dashboard (FastAPI app, create_app at petri/dashboard/api.py:148, lifespan :174-178, routes through :827) has no notion of traces. Expose the local span sink (the `spans` table in the dish's petri.sqlite, amended D4) read-only so the frontend waterfall (next issue) and curious users (curl) can query per-cell traces. Newcomer-friendly: this issue is fully self-contained behind the `query_spans()` API — synthetic-span fixtures mean no live pipeline, no DBOS, and no LLM calls are needed to develop or test it.

**Scope.**
- New `petri/dashboard/traces_api.py` with an APIRouter registered in create_app (keep api.py from growing further):
  - `GET /api/traces?cell_id=&colony=&limit=` → trace summaries (trace_id, root span name, start time, duration, petri.cell_id, petri.verdict if present), newest first.
  - `GET /api/traces/{trace_id}` → the full span list for one trace (id, parent, name, kind, timing, status, attributes) — enough for the client to build a tree.
- Backed exclusively by `query_spans()` from the span-sink issue — no direct SQL against petri.sqlite inside the endpoints. Do NOT follow the get_cells/get_cell_detail anti-pattern of re-deserializing every colony from disk per request (api.py:581-600 and :619-636, a documented O(colonies×cells)-per-request pain point).
- Missing span data → empty list with 200, never 500: a petri.sqlite that predates the spans/usage migration, or a dish with no petri.sqlite at all (pre-v2 file-tree dishes are ignored per D8), must not break the dashboard.
- Read-only endpoints; no change to the CORS setup (api.py:181-186) or the localhost-bind default (launch.py:114-124).

**Out of scope.** Frontend rendering, SSE push of spans (frontend refetches on the existing /api/stream `event_inserted` signal for now, api.py:760-807), mutation endpoints, auth.

**Touched files.** petri/dashboard/api.py:148 (router registration inside create_app). **New:** petri/dashboard/traces_api.py, tests/unit/test_traces_api.py (httpx TestClient against the app factory, seeded with synthetic spans).

**Acceptance criteria:**
- [ ] pytest: with a store seeded with one 3-span trace, GET /api/traces returns one summary with correct duration and petri.cell_id; GET /api/traces/{trace_id} returns 3 spans whose parent_span_id links form a tree
- [ ] pytest: ?cell_id= filter returns only traces whose root/any span carries that petri.cell_id
- [ ] pytest: unknown trace_id → 404 with a JSON error body
- [ ] pytest: dish with no span data (petri.sqlite predating the spans/usage migration, or no petri.sqlite at all) → GET /api/traces returns 200 and []
- [ ] pytest: limit parameter caps the summary count and results are newest-first
- [ ] Code review check: no colony deserialization, no reads of the domain `events` table, and no direct SQL inside the new endpoints — all span reads go through query_spans()

---

## M5-otel.8 — Add per-cell trace waterfall view to the petri launch dashboard

**Size:** M · **Labels:** migration-v2, observability, dashboard · **Depends on:** Add dashboard trace API endpoints reading the spans table in petri.sqlite · **Field issues:** #15

**Context.** D7: "the Petri dashboard grows a lightweight trace view … `petri launch` shows per-cell traces with zero external infrastructure." The frontend is a single self-contained template (petri/templates/frontend.html, 3415 lines) rendered by build_frontend_html (petri/dashboard/frontend.py:87). Field issue #15 (see docs/field-reports.md) documented the dashboard's missing drill-down navigation and observability bugs; the waterfall is the v2 answer to "what is this cell actually doing and what did it cost". This issue also owns the package-version drift fix — frontend.py:87 has a stale hardcoded version default '0.3.0', and petri/__init__.py's version is itself stale at 0.3.0 vs released 0.3.4; both fixes land HERE (M6-storage explicitly does not claim them).

**Scope.**
- New "Traces" panel in frontend.html: list of recent traces (from GET /api/traces), and a per-trace waterfall — bars proportional to span duration, indentation/nesting by parent_span_id, color-coded by span kind (DBOS workflow / DBOS step / agent run / model request / tool call), click or hover reveals petri.* attributes (agent role, verdict, iteration, tokens, cost).
- Factor the span-list→tree waterfall-building logic into a separately testable unit (extracted JS function runnable headlessly, or an equivalent testable transform) so the rendered tree can be asserted automatically against fixture traces.
- Drill-down: from the existing cell detail view, a "View trace" link filters /api/traces by that cell's id (absorbs the drill-down-navigation intent of field issue #15).
- Refresh: refetch trace list when the existing /api/stream SSE poll (api.py:760-807) emits, plus a manual refresh control. No new SSE channel this milestone.
- Empty state: a dish with no spans shows an explanatory message (how to enable telemetry), not an error.
- Stay self-contained: no CDN scripts/fonts; plain JS/SVG or CSS bars in the single template, consistent with the current approach.
- Fix build_frontend_html's version default to source from petri.__version__ (frontend.py:87) and align petri/__init__.py with the released version.
- Add a manual-verification PR checklist to the PR description (visual waterfall inspection on a real traced run, span click/hover attribute reveal, drill-down link, devtools network check) — manual checks live in the PR checklist, not in the acceptance criteria.

**Out of scope.** Cross-cell/dish-wide trace analytics, span search, comparing runs, replacing the colony DAG visualization, token/cost accounting rollups (separate issue in this milestone).

**Touched files.** petri/templates/frontend.html, petri/dashboard/frontend.py:87 (new template variables + version fix), petri/__init__.py:1 (version drift). **New:** tests/unit/test_frontend_traces.py, a committed fixture trace (JSON matching the /api/traces/{trace_id} contract).

**Acceptance criteria:**
- [ ] pytest: the extracted waterfall tree-builder, fed the committed fixture trace, produces the expected nested structure — root command span with DBOS step, agent-run, and model-request descendants at the correct depths (JSON-level assertion; run the extracted JS headlessly via node, or an equivalent automated harness)
- [ ] pytest: rendered waterfall markup for the fixture trace contains one bar element per span, nesting/indentation derived from parent_span_id, and bar widths proportional to span duration (DOM-level assertions on the built output)
- [ ] pytest: build_frontend_html output contains the traces panel container, the drill-down 'View trace' link in the cell detail markup, and references /api/traces (string smoke test)
- [ ] pytest: rendered HTML embeds the real package version, not '0.3.0'; petri.__version__ matches the pyproject version
- [ ] pytest: empty-state — a dish with no spans renders the explanatory enable-telemetry message and no error markup
- [ ] pytest: the built HTML contains no non-localhost URL references (regex self-containment guard: no CDN scripts, fonts, or remote images)
- [ ] PR description contains the completed manual-verification checklist (waterfall renders on a real traced run, span click shows petri.* attributes including verdict and tokens, drill-down works, devtools shows zero non-localhost requests)

---

## M5-otel.9 — Surface per-cell and per-phase token/cost accounting in the petri launch dashboard

**Size:** S · **Labels:** migration-v2, observability, dashboard, good-first-issue · **Good first issue** · **Depends on:** Attach Petri domain span attributes: dish/colony/cell, agent role, verdict, iteration, tokens/cost; Add dashboard trace API endpoints reading the spans table in petri.sqlite

**Context.** The multi-agent prototype Petri was extracted from exposed token/cost accounting through its dashboard API (a /api/tokens endpoint with total_cost_usd rollups); Petri lost that capability in the extraction. Field evidence shows why self-reporting can never bring it back: in ~2,700 real prototype events, exactly one token_usage event was ever self-reported by an agent despite full schema, API, and dashboard support (see docs/field-reports.md). v2 captures usage automatically at the Model/harness layer as span attributes (petri.tokens_input / petri.tokens_output / petri.cost_usd, from pydantic-ai RunUsage — the domain-attributes issue), and the span exporter extracts them into the `usage` table in petri.sqlite (amended D4; the span-sink issue), so accounting is a plain SQL aggregate over trustworthy data instead of trusting agents. This restores the lost capability. Newcomer-friendly: the usage table schema and span-store reader are stable by the time this is workable, and synthetic span/usage fixtures make it fully self-contained (no LLM calls).

**Scope.**
- Aggregation helper in petri/telemetry/span_store.py: `aggregate_usage(petri_dir, group_by=...)` — SQL aggregation (stdlib sqlite3) over the `usage` table in petri.sqlite, grouped per cell and per phase (the usage table's cell_id and phase columns).
- New `GET /api/costs?cell_id=&colony=` endpoint (register alongside the traces router from the trace-API issue): per-cell and per-phase rollups (tokens in/out, cost_usd where present), plus a dish-level total.
- Dashboard panel: dish-level token/cost total, per-cell totals in the cell detail view, phases broken out. Same self-contained frontend constraints as the waterfall (no CDN, plain JS).
- Omission honesty: cells whose backend reported no usage show "n/a", never a fabricated zero (subscription-auth CLI backends may not report usage — mirrors the attribute-table policy; the usage table contains no zero-filled rows by construction).

**Out of scope.** Budgets/alerts, historical trend charts, per-model price-table editing UI, remote export of cost data.

**Touched files.** petri/telemetry/span_store.py (aggregation helper), petri/dashboard/traces_api.py or a sibling costs router registered in create_app (petri/dashboard/api.py:148), petri/templates/frontend.html. **New:** tests/unit/test_costs_api.py.

**Acceptance criteria:**
- [ ] pytest: a petri.sqlite seeded with synthetic usage rows → GET /api/costs returns per-cell rollups matching hand-computed sums, with per-phase grouping correct
- [ ] pytest: cells with zero usage rows report null tokens/cost (not 0); spans without usage attributes contribute nothing (the exporter writes no zero-filled usage rows)
- [ ] pytest: dish with no usage data (petri.sqlite predating the spans/usage migration, or no petri.sqlite at all) → GET /api/costs returns 200 with an empty rollup, never 500
- [ ] pytest: build_frontend_html output contains the cost panel container and references /api/costs (string smoke test)
- [ ] pytest: ?cell_id= and ?colony= filters scope the rollup correctly
- [ ] Code review check: aggregation happens in SQL over the usage table via the span-store helper — no per-request Python loops over raw span rows, no colony deserialization

---

## M5-otel.10 — Write observability docs: zero-infra default plus self-hosted Jaeger/SigNoz/Langfuse wiring

**Size:** S · **Labels:** migration-v2, observability, docs, good-first-issue · **Good first issue** · **Depends on:** Enable DBOS OTLP tracing with hierarchical workflow→step spans behind the durability seam; Instrument pydantic-ai agent runs with OpenTelemetry (no Logfire service required); Attach Petri domain span attributes: dish/colony/cell, agent role, verdict, iteration, tokens/cost

**Context.** D7 names self-hosted Jaeger/SigNoz/Langfuse (and Logfire) as export targets users wire up themselves. The maintainer's cost-sensitivity (prominent README cost warnings) makes the token/cost span attributes a headline: traces now show what each cell actually cost. Docs must also preserve the D1 framing — Petri itself never requires Docker; Docker appears only in the user's optional backend recipes.

**Scope.**
- New `docs/observability.md`: (1) zero-infra default — traces appear in `petri launch` out of the box; (2) full `telemetry:` petri.yaml reference matching the implemented TelemetryConfig schema, including env-var overrides; (3) copy-paste recipes for Jaeger (all-in-one), SigNoz, and Langfuse — exact OTLP endpoint URLs, auth headers where needed, and an explicit note that these backends are optional and external to Petri; (4) the `petri.*` span attribute table (copied from docs/ARCHITECTURE-V2.md, kept in sync); (5) troubleshooting checklist (enabled flag, endpoint reachability, sampler, retention).
- README: short "Observability" section linking the doc; mention token/cost visibility next to the existing cost warnings.
- CLAUDE.md: add petri/telemetry/ to the architecture tree and note the telemetry config section.
- pytest doc-parity check: extract the yaml snippet from docs/observability.md and validate it against TelemetryConfig so the reference cannot drift from code.

**Out of scope.** Hosted/proprietary backends requiring accounts as the documented default (Logfire cloud may get a one-line mention as "also works via OTLP"), metrics/logs signals, DBOS Conductor/Console (proprietary — must not be documented as part of Petri per the locked decisions).

**Touched files.** README.md, CLAUDE.md (architecture block). **New:** docs/observability.md, tests/unit/test_docs_telemetry_snippet.py.

**Acceptance criteria:**
- [ ] docs/observability.md exists and covers: zero-infra default, complete telemetry: yaml reference, three backend recipes with exact endpoint values, attribute table, troubleshooting
- [ ] At least the Jaeger recipe manually verified end-to-end (spans from a real `petri grow --dry-run`-scale run visible in Jaeger UI; screenshot attached to the PR); SigNoz/Langfuse recipes verified or explicitly marked community-verification-wanted
- [ ] pytest: the yaml snippet embedded in the doc parses and validates against TelemetryConfig (doc-parity test green)
- [ ] README links to docs/observability.md; CLAUDE.md architecture tree includes petri/telemetry/
- [ ] Doc states that Docker/Postgres are never required by Petri itself (recipes only describe the user's optional backend)
- [ ] No mention of DBOS Conductor/Console as a Petri feature

---
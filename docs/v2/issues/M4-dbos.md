# M4-dbos — Issue Backlog

> Tracking epic for milestone **M4-dbos**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Replace Petri's hand-rolled durability layer — the fcntl-locked 14-state queue.json machine, the ThreadPoolExecutor busy-wait scheduler, the adaptive load balancer, the .stop sentinel, and the external shell backoff wrapper v1 users had to run to survive provider rate limits — with a durable execution engine: the per-cell validation pipeline becomes a typed pydantic-graph (Socratic→Research→Critique→Debates→ConvergenceCheck→RedTeam→Evaluate→End) executed as parameterized DBOS workflows on DBOS queues over DBOS's own SQLite system DB (dbos.sqlite — kept strictly separate from the dish's petri.sqlite domain DB per the amended D4), behind an internal ExecutionBackend seam so DBOS stays swappable per D1. Users get what D2 demands: a killed or rate-limited `petri grow` — and, via the in-milestone seed re-platform, `petri seed` (closing field issue #8) — resumes exactly where it left off with no re-paid LLM calls; concurrency/rate limits become declarative queue config instead of a shell wrapper; and `petri stop` becomes a durable cancel instead of a racy sentinel file. Imported into this milestone from M6-storage (which no longer owns them): the append_event idempotency hardening and the dashboard /api/queue repoint onto the ExecutionBackend seam. Scanner rework stays wholly with M7-lifecycle — this milestone only disables the queue-dependent scanner categories with a pointer.

**Shippable release.** A minor release (e.g. v0.6.0) where `petri grow` runs every cell as an exactly-once durable workflow keyed by its composite cell key + round, and `petri seed` runs as a durable workflow keyed by colony id: kill -9 either mid-phase, hit a rate limit, or `petri stop` a grow, and re-running recovers PENDING workflows from the last completed step without duplicate LLM spend or duplicate rows in the events table; `petri.yaml` gains worker_concurrency/rate-limit settings enforced by DBOS queues, with rate-limit hits handled by durable sleep until the provider's advertised reset; the dashboard reads execution state through the ExecutionBackend seam so `petri launch` never breaks across the cutover; engine/processor.py, grow_loop.py, propagation.py, load_balancer.py, and storage/queue.py are deleted (D8: no back-compat), removing the fcntl Windows blocker. The `events` table in the dish's petri.sqlite remains the append-only domain source of truth (D4 as amended), DBOS execution state stays confined to its own separate dbos.sqlite, and human-readable text remains a derived artifact via `petri export` (M6-storage).

**Depends on milestones:** M1-harness, M2-agents, M3-decomposer

**Milestone risks:**
- DBOS-on-SQLite may fail the spike (polling latency with use_listen_notify=False, `database is locked` contention under concurrent workers, or cancel/drain gaps). Mitigation is structural: the ExecutionBackend seam and the pydantic-graph pipeline + phase-port issues are DBOS-free, so a fallback backend on pydantic-graph FileStatePersistence can implement the same seam; only the DBOS backend, colony-workflow, seed re-platform, sync-check, and lifecycle-rewire issues are DBOS-specific.
- DBOS's determinism contract is unforgiving: any filesystem read, clock, or config access in workflow code (not steps) causes silent replay divergence, not an error. The colony parent workflow reads colony DAG state from petri.sqlite — every such read must be a step, and review must police this since tests won't reliably catch it.
- Domain-event duplication on step retry: steps are at-least-once, and a crash between an event append and its checkpoint re-runs the append. The deterministic-event-ID derivation plus INSERT OR IGNORE against the events table's UNIQUE constraint (schema from M3-decomposer) is the load-bearing mitigation for the amended D4's 'append-only, never edited' guarantee — the constraint makes duplicates unrepresentable at the schema level rather than merely checked in application code, but the ID derivation must stay total (every emittable event has exactly one ID) or distinct-looking duplicates slip past the constraint. The archived double-spend shapes (~2.5x over-logging under retries; a duplicated iteration re-logged minutes apart) are the regression fixtures.
- Two SQLite files under concurrent write load: worker processes append domain events to the dish's petri.sqlite while DBOS checkpoints to its separate dbos.sqlite. WAL mode and busy-timeout handling (established by M3-decomposer's schema issue) mitigate lock contention on the domain DB, but the spike's volume-envelope numbers (10-500 events/cell, 10-50 concurrent queue entries) must be read against both files, and all event appends must go through the single write seam so contention handling lives in one place.
- Cancellation only preempts at step boundaries, so `petri stop` cannot interrupt an in-flight LLM call (potentially minutes on the CLI harness). This is a UX regression vs the (racy) v1 expectation and must be documented; truly immediate abort would require harness-level process kill, which is out of M4 scope.
- Pickle checkpoints are Python-version- and class-layout-sensitive, and DBOS's code-checksum versioning strands in-flight workflows across code upgrades unless application_version is pinned. A user upgrading petri mid-grow may find PENDING workflows unrecoverable — needs an operational note and possibly a pinning strategy in the DBOS backend issue.
- Grow-phase field evidence is thin: the v1 grow pipeline was never truly exercised in the dogfood runs (real dish queue.json stayed empty; the sample dish has only decomposition events), so M4's pipeline design leans on code archaeology, the prototype's concurrency lessons, and D2 rather than direct field data — the spike's committed golden grow-run fixture (committed as exported text + loader, not a binary .sqlite, so it reviews and diffs in git) is the mitigation and becomes the milestone's reference dataset. Field issue #8 (seed resume) is closed in-milestone by the seed re-platform issue; the unnumbered 'no built-in rate-limit retry' gap is absorbed by the DBOS backend's queue limiter + durable rate-limit sleep.
- Cross-milestone coupling: the dashboard /api/queue repoint is now in-milestone and a hard dependency of the retire issue, closing the strangler blocker (dashboard api.py:33 imports storage/queue.py); but generated harness config (skill_queue_update.txt, adapters/generators.py:262-269) still embeds VALID_TRANSITIONS prose — coordination with M1-harness must be enforced at the epic level. M3-decomposer's interim SeedCheckpointStore is deliberately designed against the spike's findings; if the spike lands late, M3 carries interim-design risk. NEW under D4Δ: the append_event hardening now consumes M3-decomposer's petri.sqlite schema issue (UNIQUE event-id constraint) — if that schema issue slips, this milestone's phase-port issues are blocked on their idempotency dependency.
- Exact API drift risk: research documented both DBOS.register_queue()/Queue and classic-vs-GraphBuilder pydantic-graph APIs from docs, not source; the spike, pipeline-graph, and DBOS backend issues carry explicit verify-at-implementation open questions rather than assumed signatures.

---

## M4-dbos.1 — Spike: validate DBOS-on-SQLite for crash-resume, queue concurrency, rate limiting, fork, cancel, and polling latency

**Size:** M · **Labels:** migration-v2, durable-execution, spike

D1 locks a hard constraint: Petri must never require Docker or Postgres, and DBOS is under evaluation — it must be validated on its SQLite-default system database BEFORE the backend is built. This spike de-risks every DBOS behavior M4 depends on, using only APIs documented in our research record (DBOSConfig with system_database_url / use_listen_notify=False, @DBOS.workflow/@DBOS.step, SetWorkflowID, Queue with concurrency/worker_concurrency/limiter/priority/deduplication_id, DBOS.cancel_workflow/resume_workflow/fork_workflow, workflows registered before DBOS.launch()).

Scheduling: this spike is schedulable in parallel with M1-harness — it depends on nothing else in the plan. Its findings gate the design of M3-decomposer's interim SeedCheckpointStore (which is explicitly built to be swapped) — the swap itself is performed by this milestone's 'Re-platform petri seed onto the DBOS backend' issue.

Scope:
- New throwaway package `spikes/dbos-sqlite/` (excluded from the wheel) with a toy 3-step "cell pipeline" workflow (each step: sleep + write a marker file, simulating an LLM call).
- Crash-resume: start N workflows, kill -9 the process mid-step, relaunch; assert completed steps are NOT re-executed (marker files written exactly once) and PENDING workflows run to completion.
- Queue concurrency: one SQLite system DB, two worker processes, Queue(worker_concurrency=K, concurrency=G); verify both caps hold and record any `database is locked`/busy errors under concurrent enqueue+dequeue.
- Rate limiter: limiter={'limit': N, 'period': s}; measure actual start spacing. This is the replacement for the external shell backoff wrapper v1 users ran around `petri grow` to parse provider reset messages and retry — v1 had no built-in rate-limit retry (see docs/field-reports.md).
- Double-spend reproduction: rebuild the two real double-spend failure shapes observed in archived v1 runs (summarized in docs/field-reports.md) as committed fixtures: (a) a research pass that logged ~2.5x its expected events under provider retries; (b) a fully duplicated iteration whose events were re-logged 3 minutes apart. Demonstrate that DBOS step checkpointing plus deterministic event IDs prevent both shapes.
- Volume envelope: run the toy pipeline at the spec envelope — 10–500 events per cell and 10–50 concurrent queue entries — on SQLite, recording lock/latency behavior at the extremes.
- Golden grow-run fixture: produce and COMMIT a golden end-to-end grow-run fixture as EXPORTED TEXT — JSONL event logs + colony metadata for a small colony run with fake models, committed as plain text so the fixture diffs in git (the same derived-text format `petri export` in M6-storage later standardizes) — plus a small loader that materializes the fixture into a petri.sqlite `events` table for tests. Never commit a binary .sqlite file. NO real grow-phase data exists anywhere — real dishes have an empty queue.json and the sample dish contains only decomposition events — so the v2 pipeline is designed from code, and this fixture becomes the reference dataset for the rest of the milestone.
- SetWorkflowID idempotency: same ID twice executes once; deduplication_id raises DBOSQueueDeduplicatedError.
- fork_workflow(workflow_id, start_step): fork a completed workflow from step 2, verify checkpointed steps 0-1 are copied and execution resumes forward (candidate mechanism for `petri feed` re-validation).
- cancel_workflow: verify preemption happens at the next step boundary (not mid-step) — this defines what 'graceful stop' can mean in the lifecycle-rewire issue.
- Polling latency: with use_listen_notify=False (mandatory on SQLite), measure enqueue→start latency distribution; document whether it is acceptable for interactive `petri grow`.
- Pickle ceiling: checkpoint payloads at ~100KB/1MB/2MB+; confirm the paths-on-disk pattern is required and document observed behavior.
- Record dbos-transact-py version pin and its Python floor (research flagged the floor as unverified).

Out of scope:
- Any change to `petri/` production code.
- Postgres, Conductor, or Console (proprietary — must not be bundled per research caveats).

Open questions to answer in the report (do not assume): exact registration API name (research mentions both `DBOS.register_queue()` and a `Queue` class); whether a Python-level drain/deactivation call exists for a single-process CLI or only the Admin API endpoint.

Touched files: NEW `spikes/dbos-sqlite/` (scripts + README + fixtures), NEW `docs/adr/ADR-M4-dbos-sqlite-spike.md` (results + go/no-go), NEW committed golden grow-run fixture (exported text + loader) under tests/fixtures/.

**Acceptance criteria:**
- [ ] spikes/dbos-sqlite/ contains runnable scripts (documented invocation) reproducing every scenario in scope
- [ ] docs/adr/ADR-M4-dbos-sqlite-spike.md records a pass/fail + measured numbers for each of: crash-resume, dual-process queue concurrency, limiter spacing, SetWorkflowID dedup, fork_workflow, cancel boundary behavior, and enqueue→start polling latency (p50/p95) with use_listen_notify=False
- [ ] Both double-spend fixtures (~2.5x over-logging under retries; a full iteration re-logged minutes apart) are reproduced against the toy pipeline and shown prevented under step checkpointing + deterministic event IDs
- [ ] Volume envelope validated: 10-500 events/cell and 10-50 concurrent queue entries run on SQLite with lock/latency numbers recorded in the ADR
- [ ] A golden end-to-end grow-run fixture is committed as exported text (JSONL event logs + colony metadata for a small colony, fake models) together with a loader that materializes it into a petri.sqlite events table for tests — no binary .sqlite file is committed — and it is named in the ADR as the v2 reference dataset
- [ ] Pickle-size findings documented with an explicit statement of the payload pattern the workflow issues must use (paths in checkpoints, evidence on disk)
- [ ] An explicit GO / NO-GO / GO-WITH-CONSTRAINTS recommendation for DBOS as the first ExecutionBackend implementation, with the fallback (pydantic-graph persistence-based backend behind the same seam) named if NO-GO
- [ ] dbos-transact-py version and Python floor recorded and compatible with Petri's 3.11+ floor

---

## M4-dbos.2 — Add ExecutionBackend seam and typed execution-state models; route grow/check/stop through it

**Size:** M · **Labels:** migration-v2, durable-execution

D1 requires DBOS to sit behind an internal seam so it stays swappable. Today the CLI talks directly to the v1 engine: `petri/cli/grow.py:141-227` drives `grow_loop` (petri/engine/grow_loop.py:42) around `process_queue` (petri/engine/processor.py:1784); `petri/cli/check.py:100-110` and `petri/cli/stop.py:48-101` read/poke the fcntl-locked queue (`storage/queue.py`: list_queue:267, get_state_summary:404, update_state:158-188). This issue introduces the seam and re-points those call sites at it, with the v1 engine as the first (and initially only) implementation — pure strangler move, zero behavior change.

Scope:
- NEW `petri/execution/__init__.py` and `petri/execution/backend.py`: an `ExecutionBackend` Protocol/ABC with methods approximately: `enqueue_cell(cell_id, round)`, `run_colony(colony_id)`, `cancel(scope)`, `get_status_summary()`, `list_cell_runs(...)`, `reopen_cell(cell_id, trigger)` — final surface to be shaped by what grow/check/stop/feed actually need.
- NEW `petri/execution/state.py`: typed `CellPipelineState` Pydantic model lifting QueueEntry's mutable fields (iteration, weakest_link, focused_directive, cycle_start_iteration — petri/models.py:306-317 and storage/queue.py:191-249) out of queue.json semantics; typed outcome enums replacing the bare strings in ConvergenceOutcome.outcome ('converged'/'iterate'/'circuit_breaker'/'short_circuit', petri/models.py:400-405).
- NEW `petri/execution/v1_engine.py`: adapter implementing ExecutionBackend by delegating to the existing process_queue/grow_loop/queue.py functions unchanged.
- Re-point `petri/cli/grow.py`, `petri/cli/check.py`, `petri/cli/stop.py` to call the backend instead of importing engine/storage functions directly. Backend selection via petri.yaml (`execution.backend: v1`, the only valid value for now) loaded through the existing config layer.
- Unit tests: a fake in-memory backend proving the CLI paths only depend on the protocol; existing grow/check/stop tests still pass against the v1 adapter.

Out of scope:
- Any DBOS import or dependency.
- Changing queue.json format, state machine semantics, or the pipeline itself.
- Dashboard read paths — this milestone's 'Repoint dashboard /api/queue onto the ExecutionBackend seam' issue moves the dashboard onto this seam.

Touched files: NEW petri/execution/{__init__,backend,state,v1_engine}.py; MODIFIED petri/cli/grow.py, petri/cli/check.py, petri/cli/stop.py, petri/config.py (backend key), tests.

**Acceptance criteria:**
- [ ] `uv run pytest tests/` passes with no changes to test expectations for grow/check/stop behavior
- [ ] petri/cli/{grow,check,stop}.py no longer import petri.engine.processor, petri.engine.grow_loop, or petri.storage.queue directly (verified by a grep-based test or import-linter rule)
- [ ] A fake ExecutionBackend in tests can drive `petri check` and `petri stop` code paths end-to-end without any queue.json on disk
- [ ] CellPipelineState round-trips through model_dump/model_validate and carries iteration/weakest_link/focused_directive/cycle_start_iteration with correct defaults
- [ ] `execution.backend` config key defaults to `v1`; an unknown value fails fast with a clear error

---

## M4-dbos.3 — Harden append_event: deterministic event IDs with skip-if-present idempotent appends

**Size:** S · **Labels:** migration-v2, storage

(Imported from the M6-storage draft — M4 owns it because the durable pipeline's at-least-once steps are what make it load-bearing. Scope has SHRUNK under the amended D4: the petri.sqlite `events` table schema with its deterministic UNIQUE event-id constraint ships in M3-decomposer's 'Define the petri.sqlite schema and migration mechanism' issue, so this issue is a derivation scheme plus INSERT OR IGNORE — not a hand-rolled presence index.)

Today `append_event` assigns random `secrets.token_hex(4)` IDs and does a full-file collision scan on every append (petri/storage/event_log.py:27-73, scan at :45-48). Under an at-least-once step execution model, a crash between an event append and its step checkpoint re-runs the append on retry and duplicates domain events — corrupting the append-only domain source of truth (the `events` table in the dish's petri.sqlite, per D4 as amended; see docs/ARCHITECTURE-V2.md). This is not hypothetical: archived v1 runs (summarized in docs/field-reports.md) show a research pass that over-logged ~2.5x its expected events under provider retries, and one full iteration whose events were re-logged 3 minutes apart. The event envelope is uniform across all real logs — {id, cell_id, timestamp, type, agent, iteration, data} — which is exactly what makes deterministic identity possible.

Scope:
- Deterministic event IDs derived from (cell_id, round, node, agent, iteration), plus a type/sequence discriminator for nodes that legitimately emit multiple events of the same type in one iteration — the derivation must be documented and total (every event a phase can emit has exactly one ID).
- Skip-if-present semantics via the schema: appends go through INSERT OR IGNORE against the events table's UNIQUE event-id constraint (delivered by M3-decomposer's schema/migration issue) — appending an event whose ID already exists is a silent no-op that returns the existing event. No application-level presence index, tail scan, or full-log rescan: the database constraint is the idempotency mechanism.
- The event-log write seam (petri/storage/event_log.py or its v2 successor) remains the SINGLE place domain events are written — it now writes rows, not lines; events stay append-only (never UPDATE, never DELETE), schema-enforced.
- Envelope shape {id, cell_id, timestamp, type, agent, iteration, data} unchanged as the row/column mapping; validate_event_data (petri/models.py:229-235) still passes.
- No migration tooling: v2 starts fresh per D8; pre-v2 JSONL file-tree dishes on disk are ignored.

Out of scope:
- The events/cells/edges schema itself and the PRAGMA user_version migration mechanism (owned by M3-decomposer's schema issue).
- The phase runners that consume this (the two phase-port issues in this milestone).
- DBOS itself — this is pure storage-layer hardening, testable in isolation.

Touched files: MODIFIED petri/storage/event_log.py (the single write seam over the events table); tests/unit/test_event_log.py.

**Acceptance criteria:**
- [ ] Appending the same logical event twice (identical cell_id/round/node/agent/iteration + discriminator) yields exactly one row in the events table; the second call is a no-op returning the existing event
- [ ] Event IDs are deterministic: two independent processes computing the ID for the same inputs produce the same ID with no coordination
- [ ] A retry-storm test (the same event appended 50 times, including concurrently) produces one row and an uncorrupted, queryable events table
- [ ] The envelope {id, cell_id, timestamp, type, agent, iteration, data} is preserved as the row/column mapping and validate_event_data passes on all written events
- [ ] Idempotency is enforced by INSERT OR IGNORE on the schema's UNIQUE event-id constraint — no application-level presence scan or in-memory ID index exists (asserted by test or grep-verified)
- [ ] No third-party storage dependency is added; all writes use stdlib sqlite3 through the single write seam

---

## M4-dbos.4 — Model the per-cell validation pipeline as a pydantic-graph with typed transitions

**Size:** M · **Labels:** migration-v2, durable-execution · **Depends on:** Add ExecutionBackend seam and typed execution-state models; route grow/check/stop through it

The v1 per-cell lifecycle is a stringly-typed dispatch loop (process_cell's while-loop over queue_state, petri/engine/processor.py:1604-1697) guarded by a hand-rolled transition table (VALID_TRANSITIONS, petri/storage/queue.py:31-46 — 14 states, docs say 13) that sync_check bypasses (queue.py:339,354,373). Per the locked decisions, the per-cell pipeline is fixed-shape and becomes pydantic-graph code, while the colony DAG stays runtime data in graph/colony.py — it must NOT be modeled as graph edges. This issue lands the graph skeleton: nodes, typed state, and return-type unions that make illegal transitions unrepresentable, with node bodies injected as callables so it is testable before the phase ports land.

Scope:
- NEW `petri/execution/pipeline.py`: pydantic-graph BaseNode classes `SocraticNode → ResearchNode → CritiqueNode → DebatesNode → ConvergenceCheckNode → RedTeamNode → EvaluateNode → End` (classic BaseNode[StateT, DepsT, RunEndT] API per research; edges inferred from return-type annotations). ConvergenceCheckNode's return union encodes the three-way branch (RedTeamNode | ResearchNode | End) matching processor.py:1291-1358; EvaluateNode returns End variants for VALIDATED/DISPROVEN/DEFER_OPEN/DEFER_CLOSED.
- Explicit parking states: needs_human_guidance and sync_conflict are typed End variants (End(NeedsHuman), End(SyncConflict)) — parked awaiting human input, distinct from failure. These are real states, not legacy: archived v1 runs hit sync-conflict 3 separate times, and this milestone's 'Port sync-check as a startup/periodic reconciliation step' issue deliberately parks divergent cells there.
- Engine-owned iteration numbers: ctx.state.iteration is assigned and incremented only by the graph engine — never taken from agent output. Archived v1 logs show agent-assigned iteration numbers violating monotonicity, so this is an invariant, not a style choice.
- Graph state = CellPipelineState from the ExecutionBackend-seam issue (iteration/weakest_link/focused_directive live in ctx.state, replacing set_iteration/set_weakest_link/set_focused_directive file writes, storage/queue.py:191-249).
- Deps object carries injected phase callables (stub/fake in this issue) plus cell identity and dish paths — no LLM calls here.
- A terse state-disposition mapping in the module docstring: each of the 14 v1 queue states → its node/End/obsolete disposition (e.g. mediating→ConvergenceCheckNode; sync_conflict→End(SyncConflict) parking variant; done→queued and deferred_open→queued re-entry edges → new-round workflows in the colony-workflow issue, not in-graph edges). The polished contributor-facing table + mermaid diagram ship separately in the good-first-issue 'Document the 14-state queue → pipeline-graph node mapping table with a generated mermaid diagram'; the full transition-matrix test port ships in 'Port tests/unit/test_queue.py transition cases as pipeline-graph conformance tests'.

Out of scope:
- Real phase logic, LLM calls, event writes (the two phase-port issues).
- DBOS (the DBOSExecutionBackend issue). The graph runs in-process here.
- Any change to graph/colony.py — the colony DAG stays data.
- The docs mapping table + mermaid diagram and the full conformance-test port (split out as the two good-first-issues named above).

Touched files: NEW petri/execution/pipeline.py, NEW tests/unit/test_pipeline_graph.py, MODIFIED petri/execution/state.py.

Open question (flag in PR, don't decide silently): classic BaseNode API vs the v2 GraphBuilder/@g.step API — research documents both; default to classic BaseNode for the typed-transition guarantee unless implementation reveals a blocker.

**Acceptance criteria:**
- [ ] pytest runs the full graph end-to-end with stubbed phase callables and reaches each End variant (validated, disproven, defer_open, defer_closed, needs_human, sync_conflict) via appropriate stub outputs
- [ ] The iterate loop (ConvergenceCheckNode → ResearchNode) increments ctx.state.iteration and records weakest_link/focused_directive, asserted by test
- [ ] ctx.state.iteration is engine-assigned and strictly monotonic across iterations — no node body or injected callable can set it directly (asserted by test)
- [ ] needs_human and sync_conflict are typed End variants distinct from failure Ends, and tests demonstrate both are reachable and carry a human-readable reason
- [ ] Illegal v1 transitions (e.g. research_active→mediating skipping critique) have no corresponding node-return type — asserted by inspecting node return annotations
- [ ] Module docstring contains the 14-state disposition mapping with no v1 state unaccounted for

---

## M4-dbos.5 — Document the 14-state queue → pipeline-graph node mapping table with a generated mermaid diagram

**Size:** S · **Labels:** migration-v2, docs, good-first-issue · **Good first issue** · **Depends on:** Model the per-cell validation pipeline as a pydantic-graph with typed transitions

v1's execution lifecycle lives in a hand-rolled transition table (VALID_TRANSITIONS, petri/storage/queue.py:31-46 — 14 states). The pipeline-graph issue replaces it with typed pydantic-graph nodes and End variants and records a terse disposition mapping in the module docstring. This issue turns that into proper contributor-facing documentation so anyone reviewing the migration can audit that no state was lost.

Scope:
- NEW docs/pipeline-graph.md: a table mapping every one of the 14 v1 states to its v2 disposition — a graph node, a typed End variant (including the two parking variants needs_human and sync_conflict), or 'obsolete' — with a one-line rationale per row (e.g. why done→queued re-entry became a new-round workflow instead of an in-graph edge).
- A mermaid diagram of the pipeline generated by pydantic-graph's built-in mermaid generator (never hand-drawn), committed alongside the doc, plus a tiny script or test that regenerates it so the committed copy cannot silently drift from the code.
- Brief prose explaining the two-store split the diagram implies: DBOS (dbos.sqlite) = execution state, petri.sqlite (events + cells tables) = domain state.

Out of scope:
- Changing the graph itself (pipeline-graph issue).
- Deleting storage/queue.py (retire-v1 issue).

Good first issue: everything needed is readable in petri/execution/pipeline.py and petri/storage/queue.py; no LLM calls, no DBOS setup, clear done-criteria.

Touched files: NEW docs/pipeline-graph.md, NEW regen script or test under tests/.

**Acceptance criteria:**
- [ ] docs/pipeline-graph.md contains a table covering all 14 VALID_TRANSITIONS states (storage/queue.py:31-46) with the v2 disposition and a one-line rationale for each — no state unaccounted for
- [ ] The mermaid diagram is generated from the graph by code and committed; a test regenerates it and fails if the committed copy is stale
- [ ] The doc explains the parking End variants (needs_human, sync_conflict) and where done→queued re-entry happens in v2 (new-round workflows, not in-graph edges)

---

## M4-dbos.6 — Port tests/unit/test_queue.py transition cases as pipeline-graph conformance tests

**Size:** S · **Labels:** migration-v2, durable-execution, good-first-issue · **Good first issue** · **Depends on:** Model the per-cell validation pipeline as a pydantic-graph with typed transitions

tests/unit/test_queue.py::TestStateTransitions (:57-116) encodes v1's legal/illegal state-transition matrix — the most complete specification of the cell lifecycle Petri has. The retire-v1 issue will delete that file along with storage/queue.py, so before that happens every case must be ported as a conformance test over the new pydantic-graph pipeline. This preserves the lifecycle contract across the migration and is the safety net that lets the deletion land confidently.

Scope:
- NEW tests/unit/test_pipeline_conformance.py: a parametrized suite where every legal v1 transition maps to a graph path (node-return route), a typed End variant, or a documented 'obsolete' disposition — and every illegal v1 transition is asserted unrepresentable by inspecting node return annotations / pydantic-graph edge introspection.
- Each original TestStateTransitions case is traceable to its ported case (matching test name or a comment referencing the original), so the retire-v1 deletion can be audited for lost coverage.
- Runs entirely on the stubbed graph from the pipeline-graph issue: no LLM, no DBOS, no filesystem beyond tmp_path.

Out of scope:
- Deleting the original test_queue.py (retire-v1 issue does that, depending on this issue).
- Testing phase logic or event writes (phase-port issues).

Good first issue: mechanical, tightly scoped, and an excellent guided tour of the new pipeline architecture.

Touched files: NEW tests/unit/test_pipeline_conformance.py.

**Acceptance criteria:**
- [ ] A parametrized test maps all 14 VALID_TRANSITIONS entries (storage/queue.py:31-46) to a graph path, an End variant, or a documented 'obsolete' disposition — no state unaccounted for
- [ ] Every illegal v1 transition (e.g. research_active→mediating skipping critique) is asserted unrepresentable via node return-annotation / graph-edge introspection
- [ ] The suite runs with stubbed phase callables only — no LLM, no DBOS, no filesystem beyond tmp_path
- [ ] Every original TestStateTransitions case is traceable to a ported case by name or comment, so the retire-v1 issue can delete tests/unit/test_queue.py without losing transition coverage

---

## M4-dbos.7 — Port Socratic, Research, and Critique+Debates phase runners into pipeline nodes with idempotent domain-event writes

**Size:** M · **Labels:** migration-v2, durable-execution, agents, storage · **Depends on:** Model the per-cell validation pipeline as a pydantic-graph with typed transitions; Harden append_event: deterministic event IDs with skip-if-present idempotent appends

Fill in the research-side node bodies by extracting the v1 phase runners: _run_socratic_phase (petri/engine/processor.py:823; its ad-hoc idempotency guard at 846-855 is superseded by step checkpointing), _run_phase1 research (processor.py:1077), and _run_phase2 critique (processor.py:1152-1250). Node bodies call the pydantic-ai agent registry from M2-agents — the _get/_to_str/_source_to_dict dict-or-model coercion layer (processor.py:230-354) must NOT be ported since outputs are validated Pydantic instances. Per the amended D4 (docs/ARCHITECTURE-V2.md), domain events are rows appended to the `events` table in the dish's petri.sqlite — the domain source of truth — from within these step functions via the storage write seam: verdict_issued via _verdict_data (processor.py:257-273), source_reviewed via _log_sources_from_result (processor.py:371). v1's direct evidence.md appends (_append_evidence, processor.py:417, with the existing formatters, processor.py:529-745) are NOT ported as file writes: the phase-formatted evidence markdown is captured in event data so the per-cell evidence document becomes a derived artifact regenerable by `petri export` (M6-storage); large raw evidence stays on disk with paths recorded in events.

Scope:
- Extract each phase into a plain, DBOS-step-shaped function (`petri/execution/phases.py`): explicit inputs/outputs, no module-global state, no reads of clocks/config outside arguments — so the DBOSExecutionBackend issue can wrap them with @DBOS.step unmodified.
- DebatesNode runs REAL debates: it consumes `run_debate` from petri/agents/debates.py (delivered by M2-agents) — real multi-turn agent exchanges over the 4 debate pairings, with shared usage tracking and UsageLimits enforced per cell run. v1's static round-1/1.5 template formatting (petri/reasoning/debate.py:61-113) is NOT ported. Fix the v1 KeyError when a pairing references a missing agent (processor.py:1227-1228) by validating all pairings at graph construction.
- Wire SocraticNode/ResearchNode/CritiqueNode/DebatesNode bodies (petri/execution/pipeline.py) to these functions via deps.
- Idempotent event appends: consume the deterministic-ID, skip-if-present append_event (INSERT OR IGNORE against the events table's UNIQUE event-id constraint) from 'Harden append_event: deterministic event IDs with skip-if-present idempotent appends' — steps are at-least-once, and a crash between append and checkpoint must not duplicate events on retry. This issue adds no ID logic of its own; it passes the (cell_id, round, node, agent, iteration) identity through.
- Pickle discipline (locked constraint): node/step return values and graph state carry cell identity, the dish's petri.sqlite path, and paths to large on-disk evidence plus small Pydantic results only — never evidence content or event payloads; add a test asserting serialized state stays far under 2MB.
- Streaming: no run_stream inside future workflows — expose a progress-callback hook on the phase functions (fed later by pydantic-ai event_stream_handler) replacing the fire()/CellProgressEvent closure threading (processor.py:57, 1536-1552).

Out of scope:
- Convergence/RedTeam/Evaluate nodes (the decision-side phase-port issue).
- append_event internals (owned by the hardening issue this depends on).
- The events table schema (M3-decomposer) and the `petri export` command (M6-storage).
- DBOS wrapping (the DBOSExecutionBackend issue).

Touched files: NEW petri/execution/phases.py; MODIFIED petri/execution/pipeline.py, tests. engine/processor.py is left untouched (deleted wholesale in the retire-v1 issue).

**Acceptance criteria:**
- [ ] Graph run over the conftest diamond fixture with FakeProvider/TestModel executes Socratic→Research→Critique→Debates and appends verdict_issued + source_reviewed rows to the events table matching the v1 event envelope (validated by validate_event_data, petri/models.py:229-235)
- [ ] debate_mediated events come from real multi-turn run_debate exchanges (petri/agents/debates.py) under shared UsageLimits — the recorded transcript contains turns from both debaters, not static template text (asserted on the fixture run)
- [ ] Executing the same phase function twice with identical inputs appends no duplicate event rows — the events table contents are identical after the second run (idempotency consumption test)
- [ ] A debate pairing referencing an unknown agent fails at graph construction with a clear error, not a KeyError mid-cell
- [ ] Serialized graph state after each node contains identifiers and paths (petri.sqlite location, large-evidence file paths), never evidence or event content; a test asserts pickled state size < 64KB for the fixture run
- [ ] The phase-formatted evidence markdown blocks equivalent to v1 output for the same fixture inputs are captured in event data such that the per-cell evidence document is regenerable from the events table (export-parity assertion on the fixture run)

---

## M4-dbos.8 — Port ConvergenceCheck, RedTeam, and Evaluate phases plus the iterate loop and circuit breaker into pipeline nodes

**Size:** M · **Labels:** migration-v2, durable-execution · **Depends on:** Model the per-cell validation pipeline as a pydantic-graph with typed transitions; Harden append_event: deterministic event IDs with skip-if-present idempotent appends · **Field issues:** relates:#9

Fill in the decision-side node bodies. Mechanical convergence is a core identity feature and must be preserved exactly: ConvergenceCheckNode calls the mechanical convergence functions as ported to petri/agents/convergence.py by M2-agents (check_convergence, identify_weakest_link, evaluate_short_circuits, compute_circuit_breaker) — pure verdict-set logic, no LLM in the verdict-set check, semantics identical to v1's petri/analysis/convergence.py:51-211. The v1 orchestration sources being ported: _run_convergence and its converged/iterate/circuit-breaker branching (petri/engine/processor.py:1253, 1291-1358), _run_red_team (:1361), _run_evaluation with its verdict→VALIDATED/DISPROVEN/DEFER_OPEN mapping (:1422, status writes :1493-1501), and _run_decomposition_audit on circuit-breaker trip (:1009).

Scope:
- ConvergenceCheckNode: converged → RedTeamNode; iterate → ResearchNode with ctx.state.iteration bumped (engine-assigned) and weakest_link/focused_directive set (replacing queue.json field pokes, storage/queue.py:191-249); circuit breaker ((iteration - cycle_start_iteration) >= max_iterations) → run decomposition audit step → End(NeedsHuman); triage short-circuit (blocking:'conditional' + redirect_on_block) → End(DeferOpen). Outcomes use the typed enums from the ExecutionBackend-seam issue, killing the string dispatch on ConvergenceOutcome.outcome (petri/models.py:400-405).
- RedTeamNode and EvaluateNode as step-shaped functions in petri/execution/phases.py, same event-write (idempotent append_event into the events table) and pickle rules as the research-side phase-port issue.
- Terminal-source gate: EvaluateNode enforces the source hierarchy via petri/agents/source_policy.py (delivered by M2-agents), which already exposes a typed pure-core over an events list — do NOT re-refactor petri/analysis/validators.py here; that duplicate refactor is dropped. EvaluateNode refuses VALIDATED/DISPROVEN when the policy fails.
- Cell status (CellStatus, the durable domain status, petri/models.py:24-34) written to the cell's row in the `cells` table of petri.sqlite (which replaces v1 metadata.json mutations per the amended D4) from a step at terminal Ends — status truth converges on: DBOS (dbos.sqlite) = execution state, petri.sqlite (events + cells tables) = domain state.

Out of scope:
- Research-side phases (the other phase-port issue).
- Changing convergence semantics, pass sets, short-circuit rules, or max_iterations defaults.
- Re-decomposition triggered by verdicts — field issue #9 (verdict outcomes should be able to trigger re-decomposition of a claim; see docs/field-reports.md) is owned by M7-lifecycle per D6.

Touched files: MODIFIED petri/execution/pipeline.py, petri/execution/phases.py, tests (new decision-side node tests alongside tests/unit/test_convergence.py).

**Acceptance criteria:**
- [ ] For identical Verdict inputs, ConvergenceCheckNode's decision equals v1 check_convergence/evaluate_short_circuits output (parametrized parity test over the convergence matrix cases in tests/unit/test_convergence.py)
- [ ] Iterate path test: a failing blocking verdict routes back to ResearchNode with iteration+1 (engine-assigned) and a focused_directive derived from the weakest link
- [ ] Circuit-breaker test: after max_iterations non-converging cycles the graph ends in End(NeedsHuman) and a decomposition-audit event is appended to the events table
- [ ] Triage LOW_VALUE_DEFER with all other blocking agents passing short-circuits to End(DeferOpen) without running RedTeam
- [ ] EvaluateNode refuses a terminal verdict when no source at or below minimum_terminal_level exists (source-policy parity test via petri/agents/source_policy.py) and returns a typed result model, not a dict

---

## M4-dbos.9 — Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity

**Size:** L · **Labels:** migration-v2, durable-execution, storage · **Depends on:** Spike: validate DBOS-on-SQLite for crash-resume, queue concurrency, rate limiting, fork, cancel, and polling latency; Add ExecutionBackend seam and typed execution-state models; route grow/check/stop through it; Port Socratic, Research, and Critique+Debates phase runners into pipeline nodes with idempotent domain-event writes; Port ConvergenceCheck, RedTeam, and Evaluate phases plus the iterate loop and circuit breaker into pipeline nodes

The DBOS implementation of the ExecutionBackend seam — gated on the spike's GO. One parameterized workflow `validate_cell(cell_id: str, round: int)` (registered at import time, before DBOS.launch(); NO dynamic per-colony workflow definitions) runs the pydantic-graph pipeline: graph traversal is deterministic workflow code, every phase function from petri/execution/phases.py is wrapped @DBOS.step (at-least-once, checkpointed, never re-run once complete). SetWorkflowID(f"{cell_key}:r{round}") — the composite key {dish}-{colony}-{level}-{seq} (build_cell_key, petri/models.py:512-514) plus round suffix — gives exactly-once per validation round (D2.3): re-enqueueing the same cell+round is a no-op, replacing the add_to_queue race handling (petri/engine/processor.py:1836-1843). This retires, without porting: the fcntl queue machine's claim/transition machinery (storage/queue.py:66-121, 158-188, TOCTOU get_next :252-264), the AdaptiveLoadBalancer (petri/engine/load_balancer.py:81, macOS-only vm_stat probe :38-78), the ThreadPoolExecutor slot-pool busy-wait (processor.py:1863-1944), and the intent of the external shell backoff wrapper v1 users ran to parse provider reset messages and retry (docs/field-reports.md documents v1's 'no built-in rate-limit retry' gap) — DBOS queue `limiter` + step retry policies + durable rate-limit sleep replace all of it.

Scope:
- NEW `petri/execution/dbos_backend.py` + `petri/execution/dbos_app.py`: DBOSConfig (name, system_database_url defaulting to a dedicated `dbos.sqlite` file under .petri/ — DBOS's own system database, which never shares tables with and is never queried alongside the dish's petri.sqlite domain DB, per the amended D4 — use_listen_notify=False), single cell Queue configured from petri.yaml: `worker_concurrency` (replaces max_concurrent), `limiter` ({'limit','period'}), `priority` derived from cell level so deeper cells run first (priority_enabled=True as the default). Note for future readers: M7-lifecycle's convergence-point prioritization explicitly REPLACES this level-derived priority — its baseline is M4's level ordering, not FIFO.
- validate_cell workflow: load cell context (step), run graph nodes with phase functions as steps, write terminal domain events/status to petri.sqlite (steps). Uncaught exceptions are terminal in DBOS — wrap phase steps with retry policies and convert expected failures into typed End outcomes; per research, disable provider-level HTTP retries to avoid double-retry.
- Rate-limit-aware durable sleep: consume M1-harness's RateLimitedError.retry_after_seconds — when a phase step classifies a provider rate limit, the workflow durable-sleeps until the advertised reset instead of burning step retries (the v1 provider burned all 3 retries in ~5 seconds against an hours-long limit; see docs/field-reports.md).
- Throttling principle (encode in code + docs): max out worker_concurrency and let the queue limiter plus rate-limit classification do ALL throttling — the orchestrator never voluntarily idles (the v1 adaptive load balancer under-utilized workers) and never burns retries against a rate-limit wall.
- Startup resource report: adopt the resource-advisor contract as UX — `petri grow` reports the binding constraint at startup (CPU / memory / provider RPM → the concurrency it chose).
- Pickle rule enforced at the workflow boundary: inputs are (cell_id, round) strings; step outputs are small Pydantic models/paths (<2MB).
- `execution.backend: dbos` opt-in flag (v1 remains default until the retire-v1 issue flips it).
- Crash-resume integration test (marked slow/local): kill -9 a worker mid-cell, relaunch, assert resume from last completed step with no duplicate rows in the events table (leverages the deterministic-event-ID INSERT OR IGNORE append).

Out of scope:
- Colony-level orchestration (colony-workflow issue), stop/resume/feed CLI (lifecycle-rewire issue), seed workflow (seed re-platform issue), deletion of old modules (retire-v1 issue).
- Postgres support, Conductor/Console (must not be bundled), OTLP export config (M5-otel — leave the hook).

Size note: L because bootstrap/queue-config/workflow are not independently landable — a backend without idempotent identity or flow control is not reviewable as working software; no natural seam.

Touched files: NEW petri/execution/{dbos_backend,dbos_app}.py; MODIFIED petri/config.py (dbos + queue settings), pyproject.toml (dbos dependency), petri/execution/backend.py (factory), tests/integration/test_dbos_backend.py (NEW).

**Acceptance criteria:**
- [ ] With execution.backend=dbos, a single cell validates end-to-end on the diamond fixture with FakeProvider/TestModel, creating DBOS's own dbos.sqlite system DB under .petri/ (separate from the dish's petri.sqlite) and requiring no Docker/Postgres
- [ ] Enqueueing the same (cell_id, round) twice executes the workflow once (asserted via workflow listing and a single set of rows in the events table)
- [ ] kill -9 mid-cell then relaunch: workflow completes, previously-checkpointed steps are not re-executed (step-side-effect markers written once), and the events table has no duplicate rows
- [ ] Queue caps hold: with worker_concurrency=2 and 5 eligible cells, at most 2 cells are in-flight (asserted via DBOS workflow status polling); limiter config demonstrably delays starts
- [ ] A step raising RateLimitedError with retry_after_seconds triggers a durable sleep until the advertised reset — the workflow suspends without consuming step retries (asserted with a fake provider advertising a reset time)
- [ ] `petri grow` startup output reports the binding constraint and chosen concurrency (resource-advisor report), asserted in a CLI test
- [ ] All workflows/queues are registered before DBOS.launch(); a test asserts backend construction performs no per-colony dynamic workflow definition
- [ ] No pickled checkpoint payload exceeds the documented bound for the fixture run (evidence stays on disk, paths in checkpoints)

---

## M4-dbos.10 — Add colony-level parent workflow: bottom-up eligibility scheduling and validation rounds

**Size:** M · **Labels:** migration-v2, durable-execution · **Depends on:** Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity

DBOS has no DAG engine, and the colony DAG stays runtime data — so bottom-up scheduling remains Petri logic hosted in a durable parent workflow. v1 does this via repeated filesystem passes: process_queue discovers eligible cells (find_eligible_cells, petri/engine/processor.py:1724) and grow_loop re-runs passes until all-terminal/no-progress (petri/engine/grow_loop.py:42-113, no-progress signature check :98-113). The replacement: a parameterized `grow_colony(dish_id, colony_id)` @DBOS.workflow that loads the ColonyGraph from the colony DAG rows (`cells`/`edges` tables in petri.sqlite, per the amended D4), computes eligible cells via the existing pure predicate get_eligible_for_validation (petri/graph/colony.py:212-238 — kept as-is, operating on the in-memory graph), enqueues validate_cell child workflows on the DBOS queue, awaits handles, recomputes eligibility as dependencies reach VALIDATED, and completes when no cell is eligible or pending. grow_loop's no-progress heuristic becomes obsolete: enqueued workflows simply run to completion.

Scope:
- NEW `petri/execution/colony_workflow.py`: grow_colony workflow, registered before DBOS.launch(); SetWorkflowID(f"grow:{dish}-{colony}:r{round}") so re-running `petri grow` attaches to / recovers the in-flight run instead of starting a duplicate.
- Single-command full-colony growth: grow_colony loops levels until NO eligible cells remain — one `petri grow` invocation grows the entire colony across all levels (v0.x `petri grow` had to be re-run once per level).
- Determinism discipline: colony DAG reads (cells/edges tables in petri.sqlite) and eligibility computation happen inside steps (I/O is non-deterministic); the workflow orchestrates on checkpointed step outputs and child-workflow handles (handle.get_result()).
- Rounds/re-entry primitive: backend method to start a cell's next round (max existing round + 1, read via a step) — the exactly-once key for deliberate re-validation after `petri feed` (consumed by the lifecycle-rewire issue). Default mechanism is a fresh round workflow; note fork_workflow(workflow_id, start_step) as a spike-validated open question for evidence re-entry that skips unaffected steps — decide in review, do not assume.
- Dish-level entry point: enqueue grow_colony per colony honoring `petri grow --colony <name>` filtering (petri/cli/grow.py flag) — CLI wiring itself lands in the lifecycle-rewire issue.
- Keep `petri grow --dry-run` (grow.py:98-122) working from the pure eligibility predicate without starting workflows.

Out of scope:
- CLI rewiring, stop/cancel, feed (lifecycle-rewire issue).
- Auto-reopening dependents on evidence re-entry — v1 propagate_upward flags dependents but never auto-reopens (petri/engine/propagation.py:167); that human-gated semantic is preserved and ported in the lifecycle-rewire issue.
- Cross-colony scheduling policy changes (future milestone).

Touched files: NEW petri/execution/colony_workflow.py; MODIFIED petri/execution/dbos_backend.py (run_colony/enqueue-round), tests/integration/test_colony_workflow.py (NEW).

**Acceptance criteria:**
- [ ] On the 5-cell diamond fixture, grow_colony validates cells in dependency order: no cell workflow starts before all its dependencies are VALIDATED (asserted from workflow start timestamps + the events table)
- [ ] A single grow_colony invocation on a 3-level fixture validates all levels with no re-invocation — the workflow loops until no cell is eligible or pending (single-command full-colony growth)
- [ ] A cell ending DISPROVEN/DEFER_OPEN blocks its dependents from being enqueued, and grow_colony completes reporting them as not-eligible (bottom-up gate parity with get_eligible_for_validation)
- [ ] Killing the process mid-colony and relaunching resumes the colony run: already-VALIDATED cells are not re-enqueued (same round IDs dedupe), remaining eligible cells complete
- [ ] Starting grow_colony twice for the same colony+round results in one execution (SetWorkflowID idempotency test)
- [ ] Round primitive: requesting re-validation of a done cell enqueues round r+1 with a distinct workflow ID and appends a cell_reopened-compatible event
- [ ] `petri grow --dry-run` output for the fixture matches v1's eligible-cell list

---

## M4-dbos.11 — Re-platform petri seed onto the DBOS backend

**Size:** M · **Labels:** migration-v2, durable-execution · **Depends on:** Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity · **Field issues:** #8

Field issue #8 (docs/field-reports.md): an interrupted or rate-limited `petri seed` lost ALL decomposition progress — every LLM call was re-paid on retry, and a partially-written colony could be left on disk. M3-decomposer ships an interim SeedCheckpointStore that checkpoints decomposition results into a seed_checkpoints table in petri.sqlite between steps; that store was explicitly designed to be swapped for the durable backend once the DBOS-on-SQLite spike validated the platform (this milestone's spike gates that design). This issue performs the swap, making `petri seed` a first-class durable workflow and closing field issue #8 fully.

Scope:
- NEW `petri/execution/seed_workflow.py`: a `seed_colony` @DBOS.workflow with SetWorkflowID = colony id — re-running `petri seed` for the same colony attaches to / recovers the in-flight decomposition instead of restarting it. Each decomposer agent call and each write of cells/edges rows (and colony-level domain events) to petri.sqlite is a @DBOS.step: checkpointed, exactly-once per D2.3, never re-paid.
- Remove M3-decomposer's interim SeedCheckpointStore: its checkpoint semantics (which decomposition steps completed, partial colony state) map 1:1 onto DBOS step checkpoints; delete the module and stop writing the seed_checkpoints table — retire the table via a small forward-only user_version migration or leave it inert (decide in review; never touching the events/cells/edges tables).
- Preserve the seed-overwrite guard: seeding an already-seeded colony id remains guarded (a deliberate v2 safety feature per the decision record — not compat tooling).
- Pickle discipline: decomposition step outputs are typed DecompositionResult models and file paths, staying well under the ~2MB checkpoint ceiling; large evidence stays on disk.
- Route petri/cli/seed.py through the ExecutionBackend seam so seed and grow share backend selection and startup reporting.

Out of scope:
- Decomposer prompts, agent logic, search_cells, and evals — all M3-decomposer.
- Grow-side workflows (colony-workflow and lifecycle-rewire issues).

Touched files: NEW petri/execution/seed_workflow.py; MODIFIED petri/cli/seed.py, petri/execution/{backend,dbos_backend}.py; DELETED M3's interim seed checkpoint module; NEW tests/integration/test_seed_resume.py.

**Acceptance criteria:**
- [ ] kill -9 during `petri seed` mid-decomposition, then re-run: decomposition resumes from the last completed step, and the fake-model call count across both runs equals a single uninterrupted run (no re-paid LLM calls)
- [ ] A provider rate limit mid-seed suspends the workflow (durable sleep / retry policy) and the seed completes after the limit clears without duplicated cells or events
- [ ] SetWorkflowID = colony id: two concurrent `petri seed` invocations for the same colony execute the decomposition once
- [ ] Re-seeding an already-seeded colony id does not silently overwrite it (overwrite guard preserved, asserted in a CLI test)
- [ ] The interim SeedCheckpointStore is deleted, the seed_checkpoints table is no longer written, and no references to the store remain in petri/ or tests/ (grep-verified)
- [ ] An integration test named for field issue #8 covers the original failure mode end-to-end (interrupted seed loses no progress)

---

## M4-dbos.12 — Port sync-check as a startup/periodic reconciliation step

**Size:** S · **Labels:** migration-v2, durable-execution · **Depends on:** Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity

Even with DBOS as the single execution-state authority, petri.sqlite domain truth and DBOS execution state (dbos.sqlite) can diverge: a crash in the window between an event append and its step checkpoint, or a human replacing/restoring the dish's petri.sqlite (e.g. from a backup snapshot) mid-run. v1 handled this with sync_check inside the fcntl queue (storage/queue.py:285-383), which reconciled queue.json against on-disk cell state and parked divergent cells in a sync_conflict state for human resolution. This is not theoretical — archived v1 runs hit real sync conflicts 3 separate times. The retire-v1 issue deletes that implementation; this issue re-creates the semantic on the new stack so the capability survives the migration.

Scope:
- A reconciliation step in the DBOS backend that runs at `petri grow` startup (and optionally on a periodic schedule): for each cell, compare petri.sqlite domain state (terminal status in the cells table, latest rows in the events table) against DBOS workflow status for that cell's workflows — reconciling petri.sqlite vs DBOS execution state.
- On divergence, transition the cell to the pipeline's explicit sync_conflict parking state (the End(SyncConflict) variant from the pipeline-graph issue) for HUMAN resolution — reconciliation never auto-repairs, never rewrites domain data, never deletes events (append-only per D4 as amended, schema-enforced).
- Surface parked conflicts in `petri check` with the detected divergence reason (e.g. 'workflow SUCCESS but no terminal status in the cells table').

Out of scope:
- Automatic conflict resolution or merge tooling.
- Dashboard conflict UI (dashboard surfaces come later; `petri check` is the v2 surface here).

Touched files: MODIFIED petri/execution/dbos_backend.py (reconciliation step), petri/cli/check.py (conflict surfacing), tests/integration/test_reconciliation.py (NEW).

**Acceptance criteria:**
- [ ] Replacing the dish's petri.sqlite with an older snapshot mid-run (simulating the crash window or a manual restore) causes the next `petri grow` startup reconciliation to park affected cells in sync_conflict rather than crash or silently re-run them
- [ ] A cell whose DBOS workflow reports SUCCESS but whose cells-table row lacks a terminal status is detected as divergent and parked
- [ ] Reconciliation never modifies the events table (row count and content hash before/after asserted in tests — append-only preserved)
- [ ] `petri check` lists parked sync-conflict cells with a human-readable divergence reason

---

## M4-dbos.13 — Repoint dashboard /api/queue onto the ExecutionBackend seam

**Size:** S · **Labels:** migration-v2, dashboard · **Depends on:** Add ExecutionBackend seam and typed execution-state models; route grow/check/stop through it; Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity

(Imported from the M6-storage draft — M4 owns it because it is a strangler blocker for this milestone's retire-v1 issue.)

The dashboard imports the v1 queue directly: petri/dashboard/api.py:33 imports storage/queue.py, and /api/queue serves queue.json-derived state. When the retire-v1 issue deletes storage/queue.py, `petri launch` would break outright. This issue moves the dashboard's execution-state reads onto the ExecutionBackend seam so the dashboard is backend-agnostic BEFORE the deletion lands — it must land before or together with the retire-v1 issue.

Scope:
- /api/queue (and any other dashboard endpoint reading execution state) served from the seam: ExecutionBackend.get_status_summary() / list_cell_runs() instead of storage/queue.py functions.
- Response shape kept compatible with the current frontend, or the frontend updated in the same PR — `petri launch` must render the queue view identically either way.
- A seam contract test: the endpoint returns equivalent data for the same fixture dish on both execution.backend=v1 and execution.backend=dbos.

Out of scope:
- The /api/proc PTY bridge and synchronous /api/seed replacement (owned by M6-storage, which depends on this milestone's backend seam).
- The petri/query/ read layer (SQL views in petri.sqlite) and the disposable dashboard-index retirement (M6-storage).
- Trace endpoints (M5-otel).

Touched files: MODIFIED petri/dashboard/api.py, petri/execution/backend.py (if the read surface needs an added method), dashboard tests.

**Acceptance criteria:**
- [ ] petri/dashboard/ contains no imports of petri.storage.queue (grep-verified test)
- [ ] /api/queue returns equivalent data on execution.backend=v1 and execution.backend=dbos for the same fixture dish (seam contract test)
- [ ] `petri launch` works on a dish with no queue.json when the dbos backend is active
- [ ] The dashboard queue view renders correctly against the new endpoint (frontend unchanged or updated in this PR)

---

## M4-dbos.14 — Rewire petri grow/stop/feed lifecycle onto the durable backend (recovery, cancel, re-entry)

**Size:** M · **Labels:** migration-v2, durable-execution, lifecycle · **Depends on:** Implement DBOSExecutionBackend: durable cell workflows on DBOS queues with exactly-once identity; Add colony-level parent workflow: bottom-up eligibility scheduling and validation rounds · **Field issues:** relates:#8

Replace the racy cooperative lifecycle with DBOS workflow management. Today: `petri stop` sets an in-process flag plus a cross-process .stop sentinel (petri/engine/processor.py:80-124) and force-transitions active queue entries to stalled by brute-force walking colony dirs (petri/cli/stop.py:48-101; --force and graceful branches are byte-identical, stop.py:54-58); `petri grow` clears the sentinel at startup AND in its finally block (petri/cli/grow.py:95, 232) — erasing stops aimed at concurrent runs — and conflates user-stop with failure in exit codes (grow.py:268-270); `petri feed` reopens cells via engine/propagation.py (reopen_cell :98, propagate_upward :167, get_impact_report :218, plus its divergent _cell_dir_for/_get_dish_id copies :32/:84).

Scope:
- `petri grow`: start/attach to grow_colony workflows via the backend; on startup DBOS recovery resumes PENDING workflows (D2.1 — interrupted grow resumes exactly where it left off, no re-paid LLM calls); delete stop-file plumbing from grow.py; distinct exit codes for all-terminal (0), user-stop (documented non-error code), and failure.
- `petri stop`: backend.cancel → DBOS.cancel_workflow on active colony + cell workflows (preempts at next step boundary — in-flight LLM step completes first; document this). Give --force real meaning (immediate cancel of all cells) vs graceful (drain: finish in-flight cells, enqueue nothing new). OPEN QUESTION from the spike: whether drain uses a Python-level deactivation call or the Admin API endpoint — implement what the spike verified, otherwise graceful = cancel-pending + await-in-flight.
- `petri feed`: re-point petri/cli/feed.py:90-143 to backend.reopen_cell (the colony-workflow issue's round primitive: cell_reopened event + round r+1 available to the next grow) and a ported pure propagate_upward/get_impact_report (moved out of engine/propagation.py into petri/execution/ or graph/colony.py; propagation_triggered events preserved; dependents flagged for the human, never auto-reopened — v1 semantics).
- Progress UI: feed the existing MultiSpinner (petri/cli/grow.py _on_event :148-190) from backend status polling + polling new rows in the events table instead of the fire()/CellProgressEvent closures; agent-token streaming arrives via the pydantic-ai event_stream_handler hook from the research-side phase-port issue (run_stream is not allowed inside workflows).
- Test-fixture migration (moved here from the retire-v1 issue): rewrite tests/conftest.py's seeded_petri_dir, which hand-writes queue.json (tests/conftest.py:434-450), to be backend-agnostic; the diamond-DAG fixture (conftest.py:278-357) is already storage-agnostic and stays verbatim. Update stop/grow tests that pin stop-file behavior (test_cli_stop_file.py, test_processor_stop_file.py) to cancel-semantics tests.

Out of scope:
- Deleting propagation.py/processor.py/grow_loop.py (retire-v1 issue — the v1 backend still exists until then).
- Dashboard PTY /api/proc bridge replacement (M6-storage owns it; the backend API this issue ships is its prerequisite).
- Seed-side durability — owned by this milestone's 'Re-platform petri seed onto the DBOS backend' issue, which closes field issue #8 (interrupted seed losing all progress; see docs/field-reports.md).

Touched files: MODIFIED petri/cli/grow.py, petri/cli/stop.py, petri/cli/feed.py, petri/execution/{backend,dbos_backend}.py, tests/conftest.py; MOVED propagation pure logic; tests.

**Acceptance criteria:**
- [ ] Ctrl+C (or kill) during `petri grow` on the dbos backend, then re-run `petri grow`: the run resumes, completed steps are not re-executed, and total step executions across both runs equal a single uninterrupted run (integration test)
- [ ] `petri stop` during an active grow cancels workflows at the next step boundary: no queue entries are force-poked, workflows report CANCELLED, and a subsequent `petri check` shows a consistent halted state
- [ ] `petri stop` issued when no grow is running is a clean no-op (no sentinel side effects to erase)
- [ ] `petri grow` exit code distinguishes all-terminal, user-stop, and failure (asserted in CLI tests)
- [ ] `petri feed` on a VALIDATED cell appends cell_reopened + propagation_triggered events, flags (not reopens) transitive dependents, and the next `petri grow` runs that cell as round r+1 exactly once
- [ ] tests/conftest.py's seeded_petri_dir no longer writes queue.json; the diamond-DAG fixture is byte-identical to before; stop-file tests are replaced with cancel-semantics tests
- [ ] No references to the .stop sentinel remain in petri/cli/ (grep-verified)

---

## M4-dbos.15 — Retire the v1 engine: delete processor, grow_loop, propagation, load_balancer, and the fcntl queue

**Size:** M · **Labels:** migration-v2, durable-execution, storage, lifecycle, breaking-change, docs · **Depends on:** Port tests/unit/test_queue.py transition cases as pipeline-graph conformance tests; Add colony-level parent workflow: bottom-up eligibility scheduling and validation rounds; Re-platform petri seed onto the DBOS backend; Port sync-check as a startup/periodic reconciliation step; Repoint dashboard /api/queue onto the ExecutionBackend seam; Rewire petri grow/stop/feed lifecycle onto the durable backend (recovery, cancel, re-entry)

End-of-milestone strangler payoff (D8: no back-compat, old dishes stay greppable on disk). With the seam, pipeline, backend, colony, seed, reconciliation, dashboard-repoint, and lifecycle issues landed, the v1 engine is dead code behind the seam: delete it, flip the default backend to dbos, and clean every reader. Deletions: petri/engine/processor.py (1954 lines), petri/engine/grow_loop.py, petri/engine/propagation.py (pure parts already moved in the lifecycle-rewire issue), petri/engine/load_balancer.py, petri/storage/queue.py (VALID_TRANSITIONS :31-46, fcntl _queue_lock :66-81, sync_check :285-383 — the fcntl implementation dies here; its reconciliation semantic already survives as this milestone's 'Port sync-check as a startup/periodic reconciliation step' issue), and petri/execution/v1_engine.py. This also removes the hard fcntl ImportError on Windows (queue.py:21-24) — a portability win worth calling out in the changelog.

Scope:
- Delete the modules above; `execution.backend` defaults to `dbos` and `v1` becomes an error with a pointer to the last v1 release.
- petri/cli/init.py: stop scaffolding queue.json ({version:1, entries:{}} creation, init.py:159-165); `petri init` instead ensures the DBOS system-DB (dbos.sqlite) location/config — the dish's petri.sqlite domain DB is created by M3-decomposer's schema/migration mechanism, not here.
- Full config.py retirement: delete the import-time constants (config.py:247-250), the lru_cache global loader, and the ~15 raw-dict accessor functions; convert all call sites to the validated explicit config object.
- Tests: delete tests/unit/test_queue.py — superseded case-for-case by 'Port tests/unit/test_queue.py transition cases as pipeline-graph conformance tests' (a hard dependency of this issue) — and test_engine_grow_loop.py. (The conftest seeded_petri_dir rewrite already landed with the lifecycle-rewire issue.)
- analysis/scanner.py: DISABLE the queue-dependent categories — _check_state_machine (scanner.py:339) and _check_queue_schema (:387) — with an in-code pointer to M7-lifecycle's scanner re-scope issue. M7-lifecycle is the SOLE owner of scanner rework; do not redesign categories here. Coordinate with M1-harness's changes to generated skill_queue_update.txt content (adapters/generators.py:262-269 embeds VALID_TRANSITIONS prose).
- Remove two fossils, if the adapters modules still exist by then: the 'analyze' entry in adapters/claude_code.py _COMMAND_NAMES and the gemma-3-4b-it defaults in adapters/generators.py.
- Remove QueueState/QueueEntry from petri/models.py (:37-51, :306-317) or mark deprecated re-exports, per what still imports them.
- Docs: CLAUDE.md architecture block (13-state machine, engine/ listing), README, ARCHITECTURE — replace queue-machine descriptions with the pipeline-graph + DBOS-backend description; changelog entry marked breaking.

Out of scope:
- Dashboard changes — /api/queue is already on the seam via this milestone's repoint issue (a hard dependency of this one); the disposable dashboard-index retirement belongs to M6-storage.
- Any new feature work.

Touched files: DELETED petri/engine/{processor,grow_loop,propagation,load_balancer}.py, petri/storage/queue.py, petri/execution/v1_engine.py, tests/unit/test_queue.py, tests/unit/test_engine_grow_loop.py; MODIFIED petri/cli/init.py, petri/config.py, petri/analysis/scanner.py, petri/models.py, adapters/ (fossils), CLAUDE.md, README.md, docs/.

**Acceptance criteria:**
- [ ] `grep -r 'storage.queue\|engine.processor\|engine.grow_loop\|engine.propagation\|load_balancer' petri/ tests/` returns no live imports (allowed only in changelog/docs history)
- [ ] `uv run pytest tests/` passes with queue.json absent; no fixture writes queue.json
- [ ] Fresh-directory smoke test: `petri init && petri seed <claim> && petri grow && petri check && petri stop` all function on the dbos backend with no queue.json created
- [ ] `petri init` on a dish containing a stale v0.x queue.json leaves it untouched (old dishes stay on disk per D8) and does not attempt to read it
- [ ] config.py exposes no import-time constants, no lru_cache global loader, and no raw-dict accessors; all call sites consume the validated config object (grep-verified)
- [ ] Scanner runs with the two queue-dependent categories disabled, emitting a notice pointing at the M7-lifecycle scanner re-scope; no scanner import of the deleted queue module remains
- [ ] `python -c 'import petri'` succeeds on a platform without fcntl semantics being required at import (no fcntl import outside deleted code)
- [ ] CLAUDE.md/README no longer reference the 13/14-state queue machine; changelog documents the breaking change and the no-migration policy

---
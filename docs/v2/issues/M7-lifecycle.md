# M7-lifecycle — Issue Backlog

> Tracking epic for milestone **M7-lifecycle**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Close Petri's biggest conceptual gap: the pipeline is currently strictly one-way — decompose, then validate (field issue #9; see docs/field-reports.md) — so inconclusive, caveated, flawed, or stalled cells are dead ends at their current granularity. This milestone makes the lifecycle iterative: terminal verdicts feed back into decomposition through a human-gated proposal + approval flow in `petri grow`; fed evidence re-validates only the affected workflow steps via DBOS fork (with typed ingest errors instead of sentinel evidence); the contradiction scanner is re-scoped and runs on a durable schedule (this milestone is the SOLE owner of scanner rework — M4-dbos only disables the queue-dependent categories with a pointer here); the cross-colony edge structure is surfaced as intelligence (importance, fragility, validation targets) that can replace M4-dbos's level-derived queue priority; and a new read-only Analyst monitors research health over M6-storage's SQL analytics views in petri.sqlite and the spans/usage tables written by M5-otel. Seed-order guidance is documented at the epic level: seed foundation colonies FIRST — cross-colony search (M3-decomposer's search_cells) can only reference cells that already exist, so later colonies benefit from earlier ones; this guidance is echoed in the search_cells user docs (coordinate with M3-decomposer). This milestone extends M3-decomposer's canonical petri/storage/edge_registry.py, which writes rows to the `edges` table in petri.sqlite (created there; read by M6-storage's petri/query/ SQL views), rather than introducing any new edge store. NOTE: this milestone is explicitly 'design informed by grow experience' — the dogfood runs never exercised the grow pipeline (the real dish's queue.json was empty; all field evidence comes from seeding), so several issues are RFC-style and expected to be refined against real v2 grow runs before or during implementation.

**Shippable release.** A v2.x release where `petri grow` closes the loop: EVIDENCE_INCONCLUSIVE / DEFENSIBLE_WITH_CAVEATS / CRITICAL_FLAW_FOUND / PIPELINE_STALLED verdicts produce re-decomposition proposals that a human approves via `petri grow --redecompose-on` before new sub-claims are queued for the next grow pass; `petri feed` re-validates completed cells by forking their workflow from the research step (no re-paid Socratic/LLM steps) and raises typed ingest errors instead of writing sentinel text as evidence; `petri scan` can run as a cron-scheduled DBOS workflow; `petri check`/`petri graph`/dashboard rank cells by cross-colony edge intelligence, with an opt-in scheduler priority that replaces M4-dbos's default level-derived ordering; and the read-only Analyst surfaces research-health flags (systemic agent blocks, stalled velocity, staleness risk) in the dashboard.

**Depends on milestones:** M3-decomposer, M4-dbos, M5-otel, M6-storage

**Milestone risks:**
- Evidence gap: the dogfood runs never exercised the grow pipeline (the real dish's queue.json was empty; all field data comes from seeding), so every lifecycle design here extrapolates — including the Analyst's flag thresholds, which are pre-grow guesses and therefore configurable. Mitigation: RFC-first issues, opt-in flags, and an explicit re-validation checkpoint against the first real v2 grow runs before implementing the proposal-workflow, approval-gate, and priority-scheduling issues.
- Re-decomposition loop explosion: the field precedent where decomposition inflated 101 claims into 983 cells (9.7x) at 13 agents/cell makes an unguarded decompose→validate→re-decompose loop a runaway-cost machine. Hard caps, bedrock stops, and a human gate are mandatory in the RFC and enforced in tests — but even guarded loops multiply spend, so the auto-approve mode stays off by default in the release.
- DBOS API uncertainty (per D1, DBOS is still under evaluation and must stay behind a seam): fork_workflow's start_step indexing and behavior on the SQLite system DB are not covered by DBOS's public docs; forked runs get NEW workflow ids, breaking the workflow_id==cell-key invariant and forcing a generation/lineage design; create_schedule registration must happen before DBOS.launch, so schedule toggling requires restarts. The feed/fork issue carries an explicit fallback path if fork proves unreliable on SQLite.
- Scanner scope is a moving target: earlier milestones delete most of the 10 scan categories (type-safe verdicts in M2-agents, pydantic-graph transitions and DBOS state in M4-dbos), so the scheduled-scan issue may shrink to a small integrity check — re-inventory before implementation to avoid porting dead checks. M7 is the sole owner of the rework; M4-dbos only disables queue-dependent categories with a pointer here.
- Cross-colony cycle detection and edge-intelligence metrics run over dish-scale graphs for the first time (v1 structurally could not); complexity is fine at 148 cells but unmeasured at the 983-cell scale — full-recompute-per-pass may need caching if the priority-scheduling issue computes the report every grow pass.
- Priority replacement coupling: M4-dbos ships the grow queue with priority_enabled=True and level-derived priority by default, so no queue redeclaration is needed — but the swap to convergence-point rank must remain a pure enqueue-time change, because DBOS code-checksum workflow versioning can strand in-flight workflows if workflow code changes ride along.
- Human-gate UX spans CLI and dashboard: this milestone ships a CLI-only approval flow for re-decomposition proposals; if users primarily live in the dashboard, the gate may feel hidden — acceptable for a follow-on (the GitHub feedback RFC sketches one richer channel), but flag it in release notes.

---

## M7-lifecycle.1 — RFC: Define the verdict-driven re-decomposition lifecycle (triggers, context payload, guards, human gate)

**Size:** S · **Labels:** migration-v2, lifecycle, decomposer, docs, rfc · **Field issues:** relates:#9

Context: Petri's pipeline is strictly sequential — decompose then validate — with no mechanism to feed validation results back into decomposition (field issue #9, see docs/field-reports.md; still unimplemented in v1). The four signal verdicts already exist in config but are dead ends today: EVIDENCE_INCONCLUSIVE (evidence_evaluator, petri/defaults/petri.yaml:295) maps to DEFER_OPEN and stops (v1: petri/engine/processor.py:1483-1503, the `# EVIDENCE_INCONCLUSIVE or unknown` branch at 1487-1488); DEFENSIBLE_WITH_CAVEATS (champion, petri.yaml:219) is a *pass* verdict whose caveats are discarded; CRITICAL_FLAW_FOUND (skeptic, petri.yaml:203) blocks, iterates via weakest_link (analysis/convergence.py:130-147), and eventually trips the circuit breaker (convergence.py:201-211) into `_run_decomposition_audit` (processor.py:1009, called at 1338) — which audits but never re-decomposes; PIPELINE_STALLED (cell_lead, petri.yaml:120) is also written by `petri stop` (petri/cli/stop.py:63-97), so user-initiated stops must be distinguishable from genuine stalls. This RFC is the design anchor for the whole milestone (D10: ADR anchors the 'why').

Scope:
- ADR section 'Iterative lifecycle' in docs/ARCHITECTURE-V2.md (bootstrapped in M1-harness; extend it, do not create a parallel doc)
- Trigger table: for each of the 4 verdicts, when it fires (post-evaluation vs post-convergence vs circuit-breaker), what context is captured, and what re-decomposition behavior follows (e.g. caveats become sub-claims; CRITICAL_FLAW_FOUND spawns counterargument sub-claims — the pattern from field issue #13, docs/field-reports.md, whose implementation lives in M3-decomposer's agent)
- Typed `RedecompositionTrigger` context payload schema: cell id, trigger verdict + agent, weakest_link, focused_directive, caveat/flaw text extracted from the triggering assessment
- Loop guards: max re-decomposition rounds per cell, per-parent sub-claim cap (D6 / field issue #3, docs/field-reports.md — caps are ceilings, never quotas), bedrock/triviality stop (field issue #10, docs/field-reports.md), dish-level cell budget. The cell-inflation precedent — a field run where decomposition inflated 101 claims into 983 cells (9.7x; docs/field-reports.md) — makes unguarded loops an unacceptable cost risk at 13 agents/cell
- Human approval gate as default, surfaced through `petri grow` (--redecompose-on flags plus an approve/reject flow — implemented in this milestone's 'Add the human approval gate…' issue), consistent with existing propagation semantics (propagate_upward flags dependents but never auto-reopens, petri/engine/propagation.py:167 — dependents are flagged for the human to decide); auto-approve as opt-in petri.yaml key
- Invariant statement: mechanical convergence semantics (all 6 blocking verdicts in pass set) are NOT modified — re-decomposition is downstream of convergence, never a replacement for it (locked decision)
- Open questions section, including: DBOS identity for re-decomposed generations (forked/new runs get new workflow_ids — the workflow_id==cell-key invariant needs a generation suffix or lineage attribute), and which trigger context survives the <2MB pickle checkpoint ceiling (large evidence stays on disk by path)

Out of scope:
- Any implementation (issues 'Add re-decomposition trigger detection over cell verdict history', 'Implement the re-decomposition proposal workflow…', and 'Add the human approval gate…')
- Changes to verdict vocabularies or convergence logic

Touched: docs only. References for the design: petri/defaults/petri.yaml:120,203,219,295; petri/engine/processor.py:1009,1338,1483-1503; petri/analysis/convergence.py:130-147,201-211; petri/engine/propagation.py:167; petri/cli/stop.py:63-97.

**Acceptance criteria:**
- [ ] ADR merged containing a trigger table covering all four verdicts (EVIDENCE_INCONCLUSIVE, DEFENSIBLE_WITH_CAVEATS, CRITICAL_FLAW_FOUND, PIPELINE_STALLED) with fire-point, captured context, and resulting behavior for each
- [ ] RedecompositionTrigger payload schema specified as a Pydantic model definition (fields + types) in the ADR
- [ ] Guard limits (max rounds per cell, per-parent cap, bedrock stop, dish budget) specified with concrete default values and their petri.yaml keys
- [ ] Human-gate default and auto-approve opt-in documented, including the --redecompose-on flag design and how PIPELINE_STALLED from `petri stop` is excluded from triggering
- [ ] Open-questions list includes DBOS workflow-identity/generation design and pickle payload limits
- [ ] ADR explicitly labels this design as informed-by-seeding-only evidence, to be re-validated after first real v2 grow runs

---

## M7-lifecycle.2 — Add re-decomposition trigger detection over cell verdict history

**Size:** S · **Labels:** migration-v2, lifecycle · **Depends on:** RFC: Define the verdict-driven re-decomposition lifecycle (triggers, context payload, guards, human gate) · **Field issues:** relates:#9

Context: Implements the detection half of the lifecycle RFC as a pure, LLM-free module. Today the only consumers of verdict history are check_convergence (petri/analysis/convergence.py:51-124, last-write-wins per agent) and get_verdicts (petri/storage/event_log.py:135-161, reads verdict_issued events from events.jsonl — event envelopes are uniform {id, cell_id, timestamp, type, agent, iteration, data}). Nothing classifies a settled cell as 're-decomposition candidate'. The four trigger verdicts live in petri/defaults/petri.yaml (:120 cell_lead PIPELINE_STALLED, :203 skeptic CRITICAL_FLAW_FOUND, :219 champion DEFENSIBLE_WITH_CAVEATS, :295 evidence_evaluator EVIDENCE_INCONCLUSIVE). PIPELINE_STALLED is also appended by `petri stop` (petri/cli/stop.py:63-97) — a user stop must NOT count as a trigger; only circuit-breaker stalls (compute_circuit_breaker, convergence.py:201-211) do.

Scope:
- New package petri/lifecycle/ with petri/lifecycle/triggers.py: `detect_triggers(verdicts, convergence_outcome, agent_roles, config) -> list[RedecompositionTrigger]` — pure function over typed Verdict models (same input shape as check_convergence)
- RedecompositionTrigger Pydantic model per the RFC schema (cell id, trigger verdict, agent, weakest_link, focused_directive, extracted caveat/flaw text, trigger generation counter)
- Distinguish stall provenance: circuit-breaker stall triggers; user-initiated stop does not (use the stall reason recorded in the v2 pipeline state / event payload)
- Guard pre-checks: respect max-rounds-per-cell from config so detection returns nothing once the budget is exhausted
- New test file tests/unit/test_lifecycle_triggers.py using the existing verdict fixture patterns from tests/unit/test_convergence.py

Out of scope:
- Calling the decomposer or mutating the colony (the proposal-workflow issue)
- Reading storage directly — the function takes verdict lists; keep a thin data-access wrapper separate from the pure core (the same pure-core/impure-shell split used across the v2 analysis layer)

Touched files: petri/models.py (register RedecompositionTrigger; note EVENT_DATA_MODELS registry at models.py:199-226 if a corresponding event payload model is added), petri/analysis/convergence.py (read-only reference — do not modify). New: petri/lifecycle/__init__.py, petri/lifecycle/triggers.py, tests/unit/test_lifecycle_triggers.py.

**Acceptance criteria:**
- [ ] pytest case per trigger verdict: each of the four verdicts produces exactly one RedecompositionTrigger with the correct agent and fire-point classification
- [ ] A stall recorded as user-initiated (petri stop provenance) produces zero triggers; a circuit-breaker stall produces one
- [ ] DEFENSIBLE_WITH_CAVEATS trigger carries the caveat text from the champion's assessment; CRITICAL_FLAW_FOUND carries weakest_link and focused_directive
- [ ] Detection returns an empty list once the per-cell max-rounds guard is exhausted
- [ ] Module imports no LLM/provider/DBOS code (pure function, verifiable by import graph in test)

---

## M7-lifecycle.3 — Implement the re-decomposition proposal workflow: decompose_why with verdict context and colony-graph insertion

**Size:** M · **Labels:** migration-v2, lifecycle, decomposer, agents, durable-execution · **Depends on:** RFC: Define the verdict-driven re-decomposition lifecycle (triggers, context payload, guards, human gate); Add re-decomposition trigger detection over cell verdict history · **Field issues:** relates:#9, relates:#13

Context: The execution half of field issue #9 (docs/field-reports.md — v1 is strictly decompose-then-validate; validation results never feed back into decomposition). In v1, decompose_why is only reachable during seeding (petri/reasoning/decomposer.py:345-410 worklist; call at decomposer.py:363-369 behind a hasattr duck-type gate; provider prompt at petri/reasoning/claude_code_provider.py:642-688, whose `[]` return conflates 'atomic' with 'parse failure' — M3-decomposer's typed decomposer agent eliminates that). The circuit-breaker path runs _run_decomposition_audit (petri/engine/processor.py:1009, called at 1338), which only audits; this workflow supersedes it with actual re-decomposition. This issue produces PROPOSALS only — nothing is enqueued; the approval gate and enqueue path are the next issue. RFC-style caveat: the exact prompt framing of 'verdict context' should be treated as provisional until real grow-phase evidence exists.

Scope:
- New petri/lifecycle/redecompose.py: a DBOS workflow that consumes a RedecompositionTrigger and (1) calls the M3-decomposer agent's decompose_why with a verdict-context block (trigger verdict, weakest_link, focused_directive, caveats/flaw text; for CRITICAL_FLAW_FOUND request at least one counterargument sub-claim — the pattern from field issue #13, docs/field-reports.md, whose primary implementation is M3-decomposer's agent), (2) validates the typed DecompositionResult (automatic ModelRetry per D6), (3) inserts new cells and dependency edges into the colony graph and recomputes levels (compute_levels, petri/graph/colony.py:165-188) — endpoint-existence validation and dedupe on add_edge are owned by M6-storage's graph-integrity hardening issue and are consumed here, not reimplemented, (4) persists cells through the v2 storage path and appends redecomposition_proposed domain events (register the payload model in EVENT_DATA_MODELS, petri/models.py:199-226)
- Enforce all RFC guards: max re-decomposition rounds per cell; per-parent sub-claim cap (ceilings, never quotas — field issue #3, docs/field-reports.md); bedrock/triviality stop (field issue #10, docs/field-reports.md); dish-level cell budget
- Respect the pickle <2MB constraint: trigger context carries evidence by path, not content
- Workflow spans carry petri.* attributes (dish/colony/cell id, trigger verdict, generation) per M5-otel span conventions

Out of scope:
- The human approval gate, --redecompose-on flags, and enqueueing approved sub-claims (next issue: 'Add the human approval gate for re-decomposition proposals to petri grow')
- Auto-reopening or re-validating dependents of the triggering cell (propagation semantics unchanged — flag only)
- Dashboard approval UI
- search_cells / cross-colony reuse inside re-decomposition (inherited from M3-decomposer's agent as-is)

Touched files: petri/lifecycle/triggers.py (consume), petri/graph/colony.py:165-188 (level recompute), petri/models.py:199-226 (event payload registration), M3-decomposer agent module (decompose_why entry point). New: petri/lifecycle/redecompose.py, tests/unit/test_redecompose.py, integration test with pydantic-ai TestModel/FunctionModel.

**Acceptance criteria:**
- [ ] Integration test: a cell driven to EVIDENCE_INCONCLUSIVE via FunctionModel yields a redecomposition_proposed event whose proposed sub-claims and dependency edges point at the triggering cell
- [ ] CRITICAL_FLAW_FOUND trigger produces at least one counterargument-framed sub-claim in the proposal (asserted on the typed DecompositionResult)
- [ ] Guards enforced: a cell at max re-decomposition rounds produces no new proposal; per-parent cap is never exceeded even when the model returns more sub-claims (hard truncation test)
- [ ] Killing the process mid-re-decomposition and restarting resumes without duplicate proposals or double-paid decomposer calls (DBOS step checkpointing + deterministic workflow id, asserted via list_workflow_steps)
- [ ] The workflow produces a proposal only — no cell-validation workflow is enqueued by any code path in this issue
- [ ] Mechanical convergence code paths (analysis/convergence.py) are untouched by the diff
- [ ] Re-decomposition workflow spans carry petri.* attributes (dish/colony/cell id, trigger verdict, generation) per M5-otel conventions

---

## M7-lifecycle.4 — Add the human approval gate for re-decomposition proposals to petri grow (--redecompose-on, approve/reject, enqueue)

**Size:** S · **Labels:** migration-v2, lifecycle, durable-execution · **Depends on:** Implement the re-decomposition proposal workflow: decompose_why with verdict context and colony-graph insertion · **Field issues:** #9

Context: Completes field issue #9 (docs/field-reports.md): the proposal workflow (previous issue) writes redecomposition_proposed events but enqueues nothing. Per the lifecycle RFC, human approval is the default gate — consistent with existing propagation semantics where dependents are flagged but never auto-reopened (petri/engine/propagation.py:167). A field run where decomposition inflated 101 claims into 983 cells (9.7x; docs/field-reports.md) at 13 agents/cell makes ungated loops a runaway-cost machine; the gate is a cost control, not just UX.

Scope:
- `petri grow` gains `--redecompose-on=<verdicts>` (comma list over inconclusive|caveats|flaw|stalled; default: none, i.e. trigger detection is disabled unless the user opts in per run or via petri.yaml) selecting which trigger verdicts are active for the run
- Proposal surfacing: pending redecomposition_proposed events listed with proposed sub-claims, trigger verdict, and guard status; interactive approve/reject per proposal, plus --approve-all/--reject-all for non-interactive sessions
- On approval: enqueue new cell-validation workflows with deterministic workflow ids (cell key + generation) so enqueueing is idempotent; append redecomposition_approved / redecomposition_rejected domain events (register payload models in EVENT_DATA_MODELS, petri/models.py:199-226)
- Auto-approve opt-in petri.yaml key per the RFC; ships off by default
- Approval decisions recorded as attributes on the grow root command span per M5-otel conventions (coordinates with M5-otel's domain-span-attributes issue)

Out of scope:
- Dashboard approval UI (CLI-only gate in this milestone; flagged in release notes)
- Changes to trigger detection or the proposal workflow
- Propagation semantics (flag-don't-requeue unchanged)

Touched files: petri/cli/grow.py (or the v2 grow CLI module), the v2 grow orchestration enqueue point (successor of the fan-out predicate get_eligible_for_validation, petri/graph/colony.py:212-238), petri/models.py:199-226. New: tests/integration/test_redecompose_gate.py.

**Acceptance criteria:**
- [ ] Default run (no --redecompose-on, no auto-approve key) detects nothing, surfaces nothing, and enqueues nothing (regression test)
- [ ] With --redecompose-on=inconclusive and a pending proposal, approval enqueues the proposed sub-claim cells and the next grow pass processes them; rejection enqueues nothing and appends redecomposition_rejected
- [ ] Approving the same proposal twice enqueues exactly once (deterministic workflow-id idempotency test)
- [ ] With the auto-approve petri.yaml key set, proposals enqueue without prompting; without it, a non-interactive session exits with a pending-proposals notice instead of hanging
- [ ] redecomposition_approved/redecomposition_rejected events appear in the domain event log (the events table in petri.sqlite) with proposal id and generation counter
- [ ] The grow root span records approval-decision attributes (petri.* conventions per M5-otel)

---

## M7-lifecycle.5 — Re-validate fed evidence via DBOS fork_workflow(start_step) instead of full cell reopen

**Size:** M · **Labels:** migration-v2, lifecycle, durable-execution

Context: v1 `petri feed` reopens settled cells and re-runs the entire pipeline from scratch: petri/cli/feed.py filters cells in {VALIDATED, DISPROVEN, DEFER_OPEN} (feed.py:60-71, with a redundant enum-or-raw-string double check), then calls reopen_cell (petri/engine/propagation.py:98 — resets metadata to NEW, logs cell_reopened, re-enqueues) and propagate_upward (propagation.py:167 — flags dependents, never auto-reopens) per cell (feed.py:90-143); the queue re-entry edges are done→queued / deferred_open→queued (petri/storage/queue.py:42-43). That re-pays every LLM step including the Socratic phase, which new evidence cannot change. Per the public DBOS documentation, DBOS.fork_workflow(workflow_id, start_step) is the primitive for this: it copies the original's inputs and checkpointed steps up to the chosen step, then re-executes forward on a new workflow ID. A second v1 defect fixed here: petri/reasoning/ingest.py writes '[Failed to fetch: ...]' sentinel strings as evidence content when a source fails, silently poisoning the evidence base.

Scope:
- New petri/lifecycle/revalidate.py: map a cell to its latest completed validation workflow (via DBOS.list_workflows / workflow attributes carrying dish/colony/cell metadata), determine the fork point (the first research-phase step — the earliest step whose output new evidence can change), and call DBOS.fork_workflow(workflow_id, start_step=...)
- Record lineage as domain events: evidence_fed + cell_reopened events appended to the domain event log (the `events` table in petri.sqlite — the domain source of truth per D4), written through the same write seam as pipeline events (storage/event_log.py's v2 successor), including original workflow id, forked workflow id, and generation counter — this is where the workflow_id==cell-key invariant gains its generation/lineage attribute per the lifecycle RFC
- Fed evidence content written to disk (evidence dir), path-only in the checkpointed step inputs (pickle <2MB constraint)
- Ingest hardening: petri/reasoning/ingest.py raises typed errors (IngestError with source kind and cause) instead of writing '[Failed to fetch: ...]' sentinel text as evidence; ingestion runs as a checkpointed DBOS step so a fetch failure is a retryable step failure, never silent poisoned evidence
- Rework petri/cli/feed.py to show the existing impact report (get_impact_report, propagation.py:218) and fork instead of reopen; keep propagate_upward's flag-don't-requeue semantics untouched; use M3-decomposer's consolidated resolve_dish_id (petri/storage/dish.py) instead of the duplicated dish-id resolution at propagation.py:84
- `petri feed` emits a root command span with petri.* attributes (dish/colony/cell ids, generation, original + forked workflow ids) per M5-otel conventions (coordinates with M5-otel's domain-span-attributes issue)

Open questions (do not resolve by inventing APIs — verify against the installed dbos-transact-py):
- Exact start_step indexing semantics of fork_workflow (step numbering vs step name) and its behavior on the SQLite system DB — public DBOS docs describe the primitive but not SQLite-specific behavior; validate in a throwaway script before wiring, and fall back to full re-enqueue (new workflow, same cell key + generation) if fork proves unreliable on SQLite
- Whether forked workflows appear in list_workflows with an explicit parent reference we can use for lineage, or whether lineage must live entirely in our domain events

Out of scope:
- LLM-based evidence→cell matching (v1 has none — the user picks cells manually; keeping that)
- Auto-requeueing flagged dependents (human decides, unchanged)
- Re-decomposition on feed (that path only triggers via verdicts)

Touched files: petri/cli/feed.py:35-43,60-71,90-143; petri/engine/propagation.py:84,98,167,218 (reopen_cell retired on this path; propagate_upward/get_impact_report kept); petri/storage/queue.py:42-43 (re-entry edges retired with the v2 queue by M4-dbos); petri/reasoning/ingest.py (typed errors, DBOS step). New: petri/lifecycle/revalidate.py, tests/integration/test_feed_fork.py.

**Acceptance criteria:**
- [ ] Feeding evidence to a VALIDATED cell creates a forked run whose Socratic-phase step output is copied, not re-executed, while research-phase steps re-execute (asserted via DBOS.list_workflow_steps on the forked run, using TestModel so no real LLM calls occur)
- [ ] The domain event log (events table in petri.sqlite) gains evidence_fed and cell_reopened events carrying original workflow id, forked workflow id, and generation counter
- [ ] Dependents of the fed cell are flagged (propagation_triggered events) but not enqueued
- [ ] Feeding the same evidence twice does not create a duplicate concurrent fork for the same cell generation (idempotency test)
- [ ] A failing URL/file fetch raises a typed IngestError and no evidence file containing '[Failed to fetch' is ever written (regression test); ingestion executes as a checkpointed DBOS step
- [ ] `petri feed` emits a root command span with petri.* attributes (dish/colony/cell ids, generation, original and forked workflow ids) per M5-otel conventions
- [ ] If the fork-on-SQLite open question resolves negative, the documented fallback (full re-enqueue with generation suffix) passes the same acceptance tests except the step-copy assertion, and the issue records which path shipped

---

## M7-lifecycle.6 — Re-scope the contradiction scanner for v2 and run it as a DBOS scheduled workflow

**Size:** M · **Labels:** migration-v2, lifecycle, durable-execution, observability

Context: This issue is the SOLE owner of the scanner rework — M4-dbos's retire-v1 issue only disables the queue-dependent categories and points here. The v1 scanner (petri/analysis/scanner.py scan:66-149) cross-checks 10 categories under a 6-level authority hierarchy (AUTHORITY_LEVELS scanner.py:23-30, stored but never actually consulted for fix direction). In v2, most categories die: categories 3-5 (_check_state_machine :339, _check_event_types :364, _check_queue_schema :387) validate queue.json and VALID_TRANSITIONS, which M4-dbos deletes; verdict-vocabulary drift (categories 1-2, 6 — _check_convergence_logic :435) becomes a construction-time type error once agents are pydantic-ai Agents with Literal verdict outputs (M2-agents). What survives is domain-data integrity: malformed/orphaned event rows in the `events` table of petri.sqlite (a malformed event = a row that violates the uniform envelope {id, cell_id, timestamp, type, agent, iteration, data} where the schema cannot enforce it, or whose data payload fails its registered payload model), colony round-trip drift, phantom edges (add_edge accepts nonexistent cell ids, petri/graph/colony.py:90-105), cross-colony edge integrity over the `edges` table written by M3-decomposer's petri/storage/edge_registry.py, and config-vs-generated-harness drift only when the kept Claude Code adapter (D5) still generates .claude/ output. RFC-style caveat: the surviving check list should be finalized after earlier milestones land — treat the check inventory in this issue as provisional.

Scope:
- Inventory pass: enumerate surviving v2 checks in the issue/PR description, mapping each retired v1 category to the mechanism that obsoletes it (type safety, pydantic-graph, DBOS)
- Rewrite surviving checks in petri/analysis/scanner.py operating on v2 storage (petri.sqlite via M6-storage's SQL views and typed query functions in petri/query/, per the amended D4); replace the ad-hoc ScanIssue class (scanner.py:36-57) with a Pydantic model
- Register a scheduled scan via DBOS.create_schedule (croniter cron string from a new petri.yaml key; the deprecated @DBOS.scheduled decorator must not be used). Registration must happen before DBOS.launch, so the schedule is gated by config read at process startup — document that toggling requires a restart. Off by default; exactly-once-per-firing semantics come from DBOS (idempotency key = schedule name + fire time)
- Scan results appended as scan_completed domain events through the same domain-event write seam as the pipeline (the events table via storage/event_log.py's v2 successor; register payload in EVENT_DATA_MODELS, petri/models.py:199-226) so the dashboard and CLI can surface findings
- `petri scan` (petri/cli/scan.py) stays on-demand and invokes the same workflow synchronously, keeping its exit-code-1-on-findings contract; both paths emit a root span with petri.* attributes (dish id, findings count, scheduled vs on-demand) per M5-otel conventions
- auto_fix (scanner.py:152-167, blind whole-file string replace) and scan_loop (scanner.py:170-187) are NEVER run on a schedule; keep them manual-only behind an explicit flag, or retire them if the surviving checks have no auto-fixable class — decide in the PR

Out of scope:
- Scanning generated Claude Code adapter config beyond what the kept adapter still generates in v2
- Dashboard UI for scan findings beyond exposing the events (events are enough for this milestone)

Touched files: petri/analysis/scanner.py:23-30,36-57,66-149,152-187,339,364,387,435; petri/cli/scan.py; petri/models.py:199-226. New: schedule registration in the v2 app-bootstrap module (wherever DBOS config/launch lives after M4-dbos), tests/unit/test_scanner_v2.py, tests/integration/test_scan_schedule.py.

**Acceptance criteria:**
- [ ] Retired-check inventory documented in the PR: every v1 category is either rewritten for v2 or mapped to the mechanism that makes it impossible
- [ ] With the schedule enabled (test cron of * * * * *) and the process running, a scan fires within the poll window and appends exactly one scan_completed event per firing (no duplicates across a restart mid-interval)
- [ ] With the config key absent, no schedule is registered (asserted via DBOS schedule inspection or absence of scan_completed events)
- [ ] `petri scan` still runs on demand, reports the same findings as the scheduled run on identical state, and exits 1 when findings exist
- [ ] Scheduled runs never invoke auto_fix (asserted by test seam or removal)
- [ ] A seeded phantom edge (edge to nonexistent cell id) and a malformed event row (a data payload failing its registered payload model under the uniform envelope {id, cell_id, timestamp, type, agent, iteration, data}) are both detected by the surviving checks in unit tests
- [ ] Both on-demand and scheduled scans emit a root span with petri.* attributes (dish id, findings count, trigger source) per M5-otel conventions

---

## M7-lifecycle.7 — Extend the dish edge registry with DishGraph queries, cross-colony cycle detection, and tombstones

**Size:** M · **Labels:** migration-v2, storage, lifecycle · **Field issues:** relates:#11

Context: M3-decomposer establishes the canonical edge store: petri/storage/edge_registry.py writing typed edge rows to the `edges` table in petri.sqlite (`.petri/petri-dishes/<dish_id>/petri.sqlite`; created there for search_cells reference edges — search_cells queries cells/edges via SQL; M6-storage's petri/query/ SQL views read it). This issue EXTENDS that module — no new storage module, no new edge store. What no milestone yet provides is graph-level intelligence over those edges: v1 structurally cannot even detect a cycle spanning two colonies, because deserialize_colony drops any edge whose target cell is absent from the colony (petri/graph/colony.py:384-386), so cross-colony edges live only in Cell.dependencies lists (petri/models.py:271-272) and never reach any graph object. Cross-colony edges are the structural backbone of field issue #11 (docs/field-reports.md: 36 typed edges across 12 colonies in the field-validated patch, never upstreamed). `petri connect` also still writes Edge(edge_type='cross_colony') onto the source colony's graph only (petri/cli/connect.py:81-95), and connect.py:37 tests the always-truthy 2-tuple returned by detect_interactive_mode(), making its non-interactive usage-error branch dead code.

Scope:
- DishGraph query API added to petri/storage/edge_registry.py: compose all colony graphs in a dish plus the cross-colony edges read from the `edges` table into one queryable structure (adjacency + reverse adjacency across colony boundaries), with duplicate edge rows deduped at load time
- Cross-colony cycle detection: extend the has_cycle_with_edge DFS approach (petri/graph/colony.py:107-134) across colony boundaries; adding a cross-colony edge that closes a dish-level cycle is rejected with a clear error
- Tombstone rows (M3-decomposer ships the typed edge row schema; tombstones are deferred to here if not already present): edge removal appends a typed tombstone row to the `edges` table, DishGraph composition honors tombstones, and existing edge rows are never updated or deleted in place (edge history stays auditable, consistent with D4's append-only discipline)
- Rewire petri/cli/connect.py to write cross-colony edges through the registry instead of the source colony's graph (connect.py:81-95), and fix the connect.py:37 tuple-truthiness bug in passing

Out of scope:
- Registry creation, the typed edge row schema, and the `edges` table definition in the petri.sqlite schema (owned by M3-decomposer)
- add_edge endpoint-existence validation/dedupe on colony graphs and deserialize_colony round-trip preservation of cross-colony edges (owned by M6-storage's graph-integrity hardening issue — this issue consumes those guarantees, it does not reimplement them)
- GET /api/edges and any dashboard/API surfacing (owned solely by this milestone's 'Surface edge intelligence…' issue)
- The decomposer's search_cells tool and reference-node creation during seeding (M3-decomposer owns edge creation; this issue owns dish-level querying)
- Edge-intelligence metrics (next issue)

Touched files: petri/storage/edge_registry.py (extend — no new module), petri/graph/colony.py:107-134 (DFS pattern reference, read-only), petri/cli/connect.py:37,81-95. New tests: tests/unit/test_dish_graph.py plus extensions to tests/unit/test_edge_registry.py (ported from M3).

**Acceptance criteria:**
- [ ] DishGraph over a fixture dish with two colonies and registry edges answers dependency/dependent queries across colony boundaries (cells from colony B reachable from colony A)
- [ ] A cross-colony edge that would create a cycle spanning two colonies is rejected with a clear error (structurally impossible to detect in v1 — regression-guarded)
- [ ] Removing an edge appends a typed tombstone row to the edges table; DishGraph composition honors tombstones; existing edge rows are never updated or deleted in place (asserted by comparing pre-existing edge rows before and after removal)
- [ ] Duplicate edge rows in the edges table are deduped when composing DishGraph
- [ ] `petri connect a b` writes the edge through the registry, and `petri connect` in a non-TTY session with missing args prints usage and exits nonzero (connect.py:37 bug fixed)

---

## M7-lifecycle.8 — Add edge-intelligence metrics: importance, fragility, and convergence points

**Size:** S · **Labels:** migration-v2, lifecycle, good-first-issue · **Good first issue** · **Depends on:** Extend the dish edge registry with DishGraph queries, cross-colony cycle detection, and tombstones · **Field issues:** relates:#11

Context: Field issue #11 (docs/field-reports.md) identifies three emergent signals from cross-colony edges that Petri has never computed: 'edge count = importance signal', 'edge chains = fragility detection', and 'convergence points = highest-value validation targets'. With the DishGraph query API landed on the edge registry (previous issue), these become pure graph algorithms. Existing prior art to reuse: get_impact_report (petri/engine/propagation.py:218) already computes transitive dependents via iterative DFS; the dual adjacency structure is _adj/_rev (petri/graph/colony.py:26-31); the canonical 5-cell diamond DAG test fixture lives at tests/conftest.py:278-357 and should be preserved and extended, not replaced.

Scope:
- New petri/analysis/edge_intelligence.py, pure functions over the DishGraph query API (petri/storage/edge_registry.py — no I/O, no LLM):
  - importance(cell) = total in-degree counting cross-colony edges, with the cross-colony count broken out
  - fragility(cell) = longest dependency chain passing through the cell + count of transitive dependents (a validated-cell flip invalidates everything downstream — reuse the propagation DFS pattern from propagation.py:218)
  - convergence_points(dish) = cells depended on by cells from >= 2 distinct colonies, ranked (these are the highest-value validation targets)
- Typed EdgeIntelligenceReport Pydantic model (per-cell metrics + dish-level ranked lists) suitable as a DBOS step return value and dashboard JSON payload
- Document each formula in the module docstring with the field-issue #11 rationale (link docs/field-reports.md)
- Tests on the diamond fixture extended with a second colony and 2-3 cross-colony edges

Out of scope:
- Any CLI/dashboard surfacing (next issue)
- Queue prioritization (separate issue)
- Incremental/cached computation — full recompute per call is fine at current scale (148-983 cells); note complexity in the docstring

Touched files: read-only use of the DishGraph API in petri/storage/edge_registry.py and petri/graph/colony.py:26-31; pattern reference petri/engine/propagation.py:218; fixture tests/conftest.py:278-357 (extend). New: petri/analysis/edge_intelligence.py, tests/unit/test_edge_intelligence.py.

**Acceptance criteria:**
- [ ] On the extended diamond fixture, importance/fragility/convergence-point values match hand-computed expected values in pytest
- [ ] A cell referenced by two colonies appears in convergence_points; removing one cross-colony edge drops it (fixture variant test)
- [ ] fragility of a leaf cell with many transitive dependents exceeds that of an equally-connected cell with none (directionality test)
- [ ] EdgeIntelligenceReport serializes to JSON via model_dump and round-trips
- [ ] Module has no imports from engine/, reasoning/, dashboard/, or DBOS (pure analysis layer, asserted in a test)

---

## M7-lifecycle.9 — Surface edge intelligence in petri check, petri graph, and the dashboard

**Size:** M · **Labels:** migration-v2, dashboard, lifecycle, observability · **Depends on:** Extend the dish edge registry with DishGraph queries, cross-colony cycle detection, and tombstones; Add edge-intelligence metrics: importance, fragility, and convergence points · **Field issues:** relates:#11, relates:#15

Context: The shipped dashboard has no edge awareness at all: get_cells (petri/dashboard/api.py:581) and get_cell_detail (api.py:619) re-deserialize colonies per request and return no edge metrics; the frontend is one template (petri/templates/frontend.html rendered by build_frontend_html, petri/dashboard/frontend.py:87, with hand-maintained constant blobs at frontend.py:46-78). Field issue #15 (docs/field-reports.md) documents a locally-patched dashboard variant — dish-wide DAG, cross-colony edge rendering by type, a /api/edges endpoint — that was never upstreamed; this issue upstreams its intent onto the v2 read path and closes #15. CLI-side, `petri check` joins graph cells with queue entries (petri/cli/check.py:100-110) and `petri graph` renders via render_text_tree/render_dot (petri/cli_ui.py:455-498) — neither shows importance, fragility, or targets.

Scope:
- `petri check`: add a validation-targets section — top-N convergence points with importance/fragility figures (N configurable, default 5)
- `petri graph`: annotate convergence points and cross-colony edges in both text-tree and DOT output (cli_ui.py:455-498)
- Dashboard API: GET /api/edges returning typed edges from the `edges` table in petri.sqlite via the M3-decomposer edge registry (sole owner of this endpoint in the plan), and GET /api/intelligence returning EdgeIntelligenceReport JSON; reads go through the v2 read path (SQL views and typed query functions in petri/query/ over petri.sqlite per the amended D4/M6-storage), not per-request colony re-deserialization (avoid extending the api.py:581/619 anti-pattern)
- Frontend: render cross-colony edges visually distinct by edge type and add a validation-targets panel; source colors/labels from one place instead of adding to the duplicated constants at frontend.py:55-78
- Emit OTel span attributes (dish/colony/cell id) on the new endpoints per D7/M5-otel conventions

Out of scope:
- Queue prioritization (next issue)
- Golden-angle spiral layout / pan-zoom work from the #15 local patch (not scheduled in the v2 migration plan; candidate follow-on)
- Re-decomposition approval UI

Touched files: petri/cli/check.py:100-110; petri/cli_ui.py:455-498; petri/dashboard/api.py:581,619 (+ new routes near them); petri/dashboard/frontend.py:46-78,87; petri/templates/frontend.html. New: tests/unit/test_api_edges.py (FastAPI TestClient), manual-verification checklist in the PR description.

**Acceptance criteria:**
- [ ] `petri check` on a dish with cross-colony edges prints a validation-targets section listing convergence points with importance and fragility values
- [ ] `petri graph --format dot` marks convergence points and styles cross-colony edges differently from intra edges (string-level assertions in tests)
- [ ] GET /api/edges returns all registry edges with edge_type; GET /api/intelligence returns a schema-valid EdgeIntelligenceReport (TestClient tests, no LLM, no live grow needed)
- [ ] Dashboard renders cross-colony edges visually distinct and shows the targets panel (manual verification steps documented and executed with a fixture dish)
- [ ] New endpoints do not deserialize colonies per request (verified by reading through the petri/query/ read-path layer in code review; no new deserialize_colony call sites in api.py)

---

## M7-lifecycle.10 — Prioritize grow scheduling by convergence-point rank (opt-in)

**Size:** S · **Labels:** migration-v2, lifecycle, durable-execution · **Depends on:** Add edge-intelligence metrics: importance, fragility, and convergence points · **Field issues:** relates:#11

Context: M4-dbos ships the grow queue declared with priority_enabled=True and a LEVEL-DERIVED priority as the default scheduling policy (cells scheduled bottom-up by DAG level — itself a replacement for v1's no-policy behavior, where get_next returned the first resumable entry in dict insertion order, petri/storage/queue.py:252-264). This issue explicitly REPLACES M4-dbos's level-derived priority when enabled: the baseline it must beat — and fall back to when disabled — is M4's level ordering, NOT FIFO. Field issue #11 (docs/field-reports.md) identifies convergence points as the highest-value validation targets: validating a cell that three colonies depend on settles more of the dish per LLM dollar than validating a leaf. Per public DBOS documentation, queues accept a per-enqueue priority (lower = higher priority) when declared with priority_enabled=True — which M4-dbos already sets, so no queue redeclaration is needed here. RFC-style caveat: whether priority meaningfully changes throughput under rate-limit pressure is exactly the kind of question only real grow runs answer — ship it opt-in and instrumented.

Scope:
- At the grow fan-out point (the v2 successor of get_eligible_for_validation, petri/graph/colony.py:212-238 — the predicate deciding which cell workflows start next), compute EdgeIntelligenceReport once per pass and derive a priority per eligible cell: convergence points first (by rank), then by importance, ties broken by level so bottom-up order is preserved — priority must never violate dependency eligibility, which is enforced by the fan-out predicate, not by priority
- Opt-in via a new petri.yaml key (e.g. scheduling.prioritize_targets: false default); when off, M4-dbos's level-derived priority applies unchanged
- Implement as a pure enqueue-time priority computation — no changes to workflow code, so DBOS code-checksum workflow versioning is not perturbed and in-flight workflows are unaffected
- Record the assigned priority as an OTel span attribute and in the cell's queued domain event for post-hoc analysis (D7)
- Open question to verify against the installed dbos version: exact priority parameter name/range and whether priority interacts with worker_concurrency ordering guarantees on SQLite — public DBOS docs cover priority + priority_enabled but not SQLite-specific ordering semantics

Out of scope:
- Adaptive/rate-limit-aware priority adjustment (D2's rate-limit queueing is M4-dbos's concern)
- Starvation-protection policies beyond the tie-break (revisit with grow experience)
- Any change to dependency eligibility semantics
- Queue declaration changes (M4-dbos ships priority_enabled=True)

Touched files: v2 grow fan-out module (successor of petri/engine/processor.py:1724 find_eligible_cells / colony.py:212-238), v2 config module (new key). New: tests/integration/test_priority_scheduling.py.

**Acceptance criteria:**
- [ ] With the flag on and queue concurrency=1 (strict priority order per DBOS docs), a convergence-point cell enqueued alongside a same-level leaf cell executes first (integration test with TestModel)
- [ ] With the flag off (default), assigned priorities are identical to M4-dbos's level-derived ordering (regression test asserting the level-derived priority values, not FIFO insertion order)
- [ ] Priority never schedules a cell whose dependencies are unvalidated (eligibility predicate test with priority enabled)
- [ ] Assigned priority appears in the cell's queued domain event and as a span attribute
- [ ] README/config docs document the key, its default, and state explicitly that enabling it replaces M4-dbos's level-derived priority
- [ ] The diff contains no workflow-code changes (enqueue-time only, protecting in-flight workflows from DBOS version-checksum stranding)

---

## M7-lifecycle.11 — Resurrect the Analyst: read-only research-health monitoring over petri.sqlite analytics views and the spans/usage tables

**Size:** M · **Labels:** migration-v2, lifecycle, observability, dashboard · **Depends on:** Add edge-intelligence metrics: importance, fragility, and convergence points

Context: Petri has no component watching the health of the research process itself — nothing notices that a single agent is blocking a large fraction of the dish, that verdict velocity has stalled, or that a heavily-depended-on validated cell carries staleness risk for everything downstream. After M6-storage and M5-otel land, the raw signals exist: M6-storage ships starter SQL analytics views in petri.sqlite over the events table (blocking-verdict patterns, cell velocity, stalled-cell detection, per-colony convergence rates — designed explicitly as the foundation for this issue), and M5-otel persists spans and token/cost usage to the `spans` and `usage` tables in petri.sqlite (age-pruned). This issue adds the Analyst: a read-only, LLM-free monitoring role that periodically evaluates those sources and raises proactive flags. It never blocks convergence, never mutates cells, and never spends tokens — it is pure diagnosis over data Petri already has.

Scope:
- New petri/analysis/analyst.py: pure evaluation functions over the M6-storage SQL analytics views (via petri/query/) and the M5-otel spans/usage tables in petri.sqlite, producing a typed ResearchHealthReport (Pydantic): per-flag records with severity, affected cells/agents, and the query evidence behind each flag
- Initial flag rules (thresholds configurable in petri.yaml, defaults given here): (1) systemic-agent flag — the same agent blocking >= 5 cells across the dish signals a problem with that agent's instructions or the claim set, not the individual cells; (2) stalled-velocity flag — no verdict_issued events dish-wide within a configurable window while eligible cells exist; (3) transitive-staleness flag — a validated cell whose transitive-dependent count (from edge-intelligence metrics) exceeds a threshold while its newest evidence event is older than a threshold. Evidence age is computed from event timestamps in the events table, NOT from Cell.created_at — created_at is an empty string in all real v1 cell metadata (known bug; fixed in M3-decomposer)
- Periodic execution as a DBOS scheduled workflow via DBOS.create_schedule (same registration pattern, config gating, and restart caveat as this milestone's scheduled-scanner issue); off by default
- Reports persisted append-only to a new `analyst_reports` table in petri.sqlite (its own small forward-only migration bumping PRAGMA user_version, following the M5-otel spans/usage migration precedent; queryable via petri/query/, consistent with the amended D4; the `events` table stays the domain event log and is never written by the Analyst)
- Surfacing: GET /api/health returning the latest ResearchHealthReport; a compact health panel in the dashboard; `petri check` prints active flags
- Strictly read-only: the Analyst never mutates cells, colony graphs, or execution state, and never triggers LLM calls

Out of scope:
- Remediation or auto-fix actions (flags are informational; humans act on them)
- LLM-based diagnosis of why an agent is blocking (possible follow-on)
- Alerting integrations (webhooks, email)

Touched files: petri/cli/check.py (flags section), dashboard API route module, schedule registration in the v2 app-bootstrap module (wherever DBOS config/launch lives after M4-dbos). New: petri/analysis/analyst.py, tests/unit/test_analyst.py (fixture views/spans), tests/integration/test_analyst_schedule.py.

**Acceptance criteria:**
- [ ] On a fixture dish where one agent blocks 5 cells, the report contains a systemic-agent flag naming the agent and the affected cells; at 4 cells no flag fires (threshold boundary test)
- [ ] Stalled-velocity flag fires on a fixture with eligible cells and no verdict_issued events inside the window, and does not fire when recent verdicts exist
- [ ] Transitive-staleness flag uses edge-intelligence transitive-dependent counts and event-timestamp-derived evidence age (never Cell.created_at)
- [ ] GET /api/health returns a schema-valid ResearchHealthReport and `petri check` prints active flags (TestClient + CLI tests on fixtures, no LLM)
- [ ] Analyst code imports no LLM/provider modules and performs no writes outside the analyst_reports table in petri.sqlite — the events, cells, and edges tables are never written by the Analyst (asserted in tests)
- [ ] With the schedule config key absent, no schedule is registered; with a test cron enabled, exactly one report row is appended per firing

---

## M7-lifecycle.12 — RFC: Human-feedback re-entry via GitHub (review comments as focused directives)

**Size:** S · **Labels:** migration-v2, lifecycle, docs, rfc

Context: Petri's human touchpoints are CLI prompts today (seed approval, plus this milestone's re-decomposition approval gate). For teams that live in GitHub, there is a natural richer channel: publish a cell's research output as a pull request, and treat code-review mechanics as structured re-entry into the validation loop — a changes-requested review's comments become a focused_directive that steers the cell's next validation iteration (the same focused-directive mechanism the weakest_link/debate path already uses), and merging the PR records human approval and unblocks dependents that were flagged pending a human decision. This is explicitly RFC/design-first: no implementation in this milestone. The mapping raises real design questions that deserve a document before code: cell↔PR identity mapping, auth/token handling for a local-first CLI tool, polling vs webhooks, how merge-approval interacts with mechanical convergence, and misfire handling for multi-reviewer threads.

Scope (RFC document only):
- An RFC in docs/ linked from docs/ARCHITECTURE-V2.md's ADR index covering: motivation and non-goals; cell↔PR mapping options (branch-per-cell vs PR-per-colony vs labels); the event model (e.g. github_feedback_received domain events carrying the focused-directive payload, registered in EVENT_DATA_MODELS); directive-extraction rules (which review-comment content becomes the directive, and how multi-reviewer disagreement is handled); merge-unblock semantics relative to existing propagation flags (flag-don't-requeue stays the default; merge is an explicit human 'requeue' signal); auth model (gh CLI vs token vs GitHub App); poll-vs-webhook tradeoff for a tool with no always-on server
- Explicit invariant: GitHub approval NEVER substitutes for mechanical convergence (all 6 blocking verdicts in pass set — locked identity feature); it gates human-decision points only
- A minimal end-to-end sequence diagram and a proposed split of follow-on implementation issues

Out of scope:
- Any code, GitHub App registration, or dashboard integration
- Scheduling the implementation (a future milestone decides based on the RFC)

Touched: docs only.

**Acceptance criteria:**
- [ ] RFC merged in docs/ and linked from the ARCHITECTURE-V2 ADR index
- [ ] Covers cell↔PR mapping, directive extraction, merge-unblock semantics, auth model, and the poll-vs-webhook tradeoff, each with a recommended option and rationale
- [ ] States the invariant that GitHub approval never bypasses mechanical convergence
- [ ] Proposes the follow-on implementation issue split with rough sizes
- [ ] Explicitly labeled design-first: no implementation is scheduled by this RFC

---

## M7-lifecycle.13 — Housekeeping: archive stale development worktrees and record canonical field-evidence corpora

**Size:** S · **Labels:** migration-v2, docs, housekeeping

Context: The repository has accumulated development worktrees/branches from earlier v2 planning work. Two of them — petri2 and agent-a60105df — have been verified as fully merged into (or strictly behind) main with no unique commits; keeping them around invites confusion about where v2 work happens. Separately, docs/field-reports.md (the committed index of field issues #2–#15 that v2 issue bodies cite) should record its provenance: the ai-eval and ai-factory worktrees hold the canonical field-evidence corpus (real dish data, patch history, run logs) from which the index summaries were written, so future editors know where the underlying data lives.

Scope:
- Re-verify with git (merge-base + log) that petri2 and agent-a60105df contain no unique commits, recording the verification output in the PR/issue
- Optionally tag archive/petri2 and archive/agent-a60105df for a paper trail, then remove the worktrees and delete the branches
- Add a provenance section to docs/field-reports.md naming the ai-eval and ai-factory worktrees as the canonical field-evidence corpora behind the summaries

Out of scope:
- Deleting or modifying the ai-eval/ai-factory corpora themselves
- Any source-code changes

Touched: docs/field-reports.md; repository worktrees/branches only (no source files). Requires maintainer branch permissions, so not flagged good-first-issue.

**Acceptance criteria:**
- [ ] petri2 and agent-a60105df no longer exist as worktrees or branches (or exist only as archive/ tags)
- [ ] The merge-base/log verification is recorded in the PR or issue before deletion
- [ ] docs/field-reports.md gains a provenance section naming the ai-eval and ai-factory worktrees as the canonical field-evidence corpora

---
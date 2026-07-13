# M6-storage — Issue Backlog

> Tracking epic for milestone **M6-storage**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Make each dish's petri.sqlite database — the domain source of truth per amended D4 — the thing everything reads directly: a petri/query/ layer of typed SQL query functions and views replaces the JSONL-to-SQLite copy pipeline (dashboard/migrate.py and the combined.jsonl rollup), the dashboard's REST and SSE endpoints move onto it, the dashboard stops owning child processes (the /api/proc PTY bridge and synchronous /api/seed are replaced by ExecutionBackend-driven durable runs), colony-graph persistence over the cells/edges tables becomes validated and lossless, a new `petri export` command makes JSONL/markdown a regenerable derived artifact for git commits and PR review, `petri backup` finally protects hours of paid LLM compute with consistent VACUUM INTO snapshots, and starter SQL analytics views in petri.sqlite lay the foundation for M7-lifecycle's Analyst. Moved out of this milestone: the petri.sqlite schema and migration mechanism, the dish-scoped directory layout, and dish-id/cell-id consolidation land in M3-decomposer; append_event idempotency hardening and the /api/queue ExecutionBackend repoint land in M4-dbos; the stale package/dashboard version-string fix lands in M5-otel.

**Shippable release.** A v2-alpha release where `petri launch` boots with zero "Building event index..." step and creates neither a root-level .petri/petri.sqlite dashboard index nor combined.jsonl; all dashboard reads (events, cells, stats, edges, SSE) come straight from each dish's petri.sqlite domain database (the events/cells/edges tables and dish-scoped layout shipped by M3-decomposer) through petri/query/ using stdlib sqlite3; seeding from the dashboard enqueues a durable workflow instead of running inside an HTTP request; `petri export` regenerates deterministic JSONL/markdown artifacts for git and PR review; and `petri backup`, `petri backup --list`, and `petri backup --restore` work end to end on VACUUM INTO snapshots.

**Depends on milestones:** M2-agents, M3-decomposer, M4-dbos

**Milestone risks:**
- The events-table integer watermark (SSE tailing, export --since) is safe only because the events table is append-only and never deleted (schema-enforced); any future pruning must target the spans/usage tables (age-pruned by design), never events — document the invariant next to the schema and the watermark consumers.
- SSE now polls the live domain database: keep poll reads as short-lived read-only connections (open, query, close) so long-held WAL read snapshots do not delay checkpointing while writers append at full rate.
- The events/cells/edges schema, write seam, dish layout, and PRAGMA user_version migration mechanism are all owned by M3-decomposer's petri.sqlite schema issue; the entire M6 query layer builds on them — if M3 slips, M6 cannot start against the real schema. /api/edges shipping against an empty edges table is harmless, but coordinate sequencing in the tracking epics (docs/ARCHITECTURE-V2.md).
- Cross-milestone blocking: the live half of the SSE issue needs M2-agents' event_stream_handler wiring, and the /api/proc replacement needs M4-dbos' ExecutionBackend seam plus durable seed workflow — sequence those two issues last and keep the rest of M6 landable against the v1 engine (strangler discipline).
- Cross-process live streaming is an unresolved design point (open question in the SSE issue): detached CLI grow runs won't feed the in-process broker; if the maintainer wants live tokens for detached runs, a local-socket bridge becomes a follow-on issue.
- petri export's deterministic output is load-bearing beyond M6: M4-dbos commits its golden grow fixture as exported text, so any change to export ordering/format churns that fixture — version the export format in the manifest and coordinate format changes across milestones.
- VACUUM INTO needs free disk roughly equal to the database size and fails cleanly rather than corrupting — petri backup must surface that error clearly (AC'd); restore must also remove stale -wal/-shm sidecar files or the restored database can appear corrupt (AC'd).
- Schema-migration version collisions: M6's analytics-views migration and M5-otel's spans/usage migration both bump PRAGMA user_version — agree numbering/order in the tracking epics to keep the forward-only migration chain linear.
- Filename collision hazard: the retired root-level .petri/petri.sqlite disposable index shares its basename with the new per-dish petri.sqlite domain database — docs, tests, and error messages must always say which one they mean; the retire issue's ACs pin that the root-level file never reappears and stale copies are left untouched (D8).

---

## M6-storage.1 — Add petri/query/ read layer: typed SQL query functions over petri.sqlite

**Size:** M · **Labels:** storage, dashboard, migration-v2 · **Field issues:** relates:#11

D4 as amended (docs/ARCHITECTURE-V2.md) locks the read path: SQL views and typed query functions in `petri/query/` reading each dish's petri.sqlite domain database directly via stdlib `sqlite3` — no copy step, no third-party storage engine. Today reads are either full-file Python scans (`petri/storage/event_log.py:79-181`: load_events/query_events/get_verdicts/get_sources) or the disposable SQLite index (`petri/dashboard/migrate.py`). This issue builds the replacement package and retires the `combined.jsonl` rollup intermediate; endpoint repointing is a separate issue.

Scope:
- Zero new dependencies: all storage access uses stdlib `sqlite3` with short-lived read-only connections (`mode=ro` URI); `pyproject.toml` is untouched. The events/cells/edges tables, the `dependents` SQL view, WAL mode, and the `PRAGMA user_version` migration mechanism are owned by M3-decomposer's petri.sqlite schema issue — this issue only reads them, over the dish layout `.petri/petri-dishes/<dish_id>/petri.sqlite` (landed in M3-decomposer before this issue).
- New package `petri/query/` exposing typed query functions:
  - `query_events(petri_dir, dish_id, *, cell_id=None, iteration=None, event_type=None, agent=None, since=None, limit=...)` returning dicts shaped exactly like the current GET /api/events rows (`petri/dashboard/api.py:559-570`). Real-data fact (field-dish sweep): the v1 event envelope was uniform — every event carries exactly {id, cell_id, timestamp, type, agent, iteration, data} — and the M3 events-table columns mirror it, so queries can rely on those columns.
  - `events_since(petri_dir, dish_id, watermark)` where the watermark is the events table's monotonically-increasing integer append-order key (SQLite rowid — safe because the events table is append-only, never edited, never deleted, schema-enforced). This restores O(1) `rowid > ?` tailing (`api.py:782-787`) — but against the domain source of truth instead of a copy. The SSE issue consumes this.
  - `event_stats(petri_dir, dish_id)` covering the aggregates in GET /api/stats (`api.py:713-728`: total events, distinct cells, counts by type, top cells) as SQL aggregates.
  - `get_verdicts` / `get_sources` equivalents so `analysis/validators.py:51-115` and convergence input can later migrate off event_log.py scans (keep the old readers working in parallel — strangler).
  - `list_cells(petri_dir, dish_id)` reading the `cells` table joined with the edges table and `dependents` view, replacing the O(colonies x cells) per-request `deserialize_colony` loop in `api.py:581-615`. Real-data fact: v1 persisted `created_at` as an EMPTY STRING in all real cell metadata (known writer bug); the M3 schema makes `created_at` NOT NULL, so list_cells can rely on it — pin that reliance with a fixture rather than re-adding tolerance code.
  - `get_edges(petri_dir, dish_id, *, edge_type=None)` reading the `edges` table in petri.sqlite (typed rows written by M3-decomposer's `petri/storage/edge_registry.py` — cross-colony reference edges, field issue #11, see docs/field-reports.md). Empty table => empty list. Reuse the row/record model from edge_registry — do NOT invent a parallel schema.
- Retire the rollup intermediate: delete `rollup_to_combined` (`petri/storage/event_log.py:187-205`), its docstring reference (`event_log.py:8-10`), its dashboard tail-loop call site and import in `petri/dashboard/api.py`, and its tests. SQL over the events table makes dish-level concatenation redundant. (The rest of the old index machinery is deleted in the retirement issue.)
- Concurrent-read correctness: reads must be correct while a writer is appending. With WAL mode + read-only connections this is closed by construction (snapshot reads — amended D4 explicitly closes the old live-tail open question); pin it with a test rather than assuming it.
- Pre-v2 data tolerance (D8, docs/ARCHITECTURE-V2.md: old v0.3.x dishes stay on disk, unloaded): dish enumeration recognizes only directories containing petri.sqlite; pre-v2 flat file-tree dishes are skipped without error, so `petri launch` never crashes on a project containing old data.
- Exclude `.petri/backups/` (snapshot databases) and `exports/` (derived text artifacts) from dish enumeration and all query results.

Out of scope:
- Changing any api.py endpoint (next issue).
- Deleting migrate.py / the disposable SQLite index (retirement issue).
- Defining or migrating the petri.sqlite schema (M3-decomposer).
- Persisting OTel spans (D7 — M5-otel; its spans/usage tables in petri.sqlite land as their own small migration and have no ordering dependency on this query layer).

New files: `petri/query/__init__.py`, `petri/query/reads.py`, `tests/unit/test_query_reads.py`. Touched: `petri/storage/event_log.py` (rollup deletion), `petri/dashboard/api.py` (rollup call/import removal only).

**Acceptance criteria:**
- [ ] pytest: query_events over a seeded multi-colony fixture dish returns the same events (same ids, same field shapes) as v0.3.4's GET /api/events returned for identical filters (fixture-pinned expected rows)
- [ ] pytest: events_since with an integer watermark returns exactly the strictly-newer events in append order — including events sharing a timestamp (ordering comes from the append-order key, not the timestamp) — with no duplicates and no gaps across a reconnect
- [ ] pytest: a query issued while a writer connection holds an uncommitted transaction returns only committed rows and never raises (WAL snapshot read over a read-only connection)
- [ ] pytest: get_edges returns [] on an empty edges table and returns typed cross-colony edge rows from a fixture using the edge_registry row model
- [ ] pytest: list_cells returns cell_id/colony_id/claim_text/level/status/dependencies/dependents for every row in the cells table, matching /api/cells' current shape (api.py:602-613), with dependencies/dependents sourced from the edges table and the dependents view
- [ ] pytest: snapshot databases under .petri/backups/ and derived files under exports/ never appear in dish enumeration or any query result
- [ ] grep/CI: no third-party storage dependency is added — 'duckdb' appears nowhere in petri/ or pyproject.toml, and all storage access uses stdlib sqlite3
- [ ] grep/pytest: rollup_to_combined is gone from petri/ and a full launch-and-query cycle produces no combined.jsonl
- [ ] pytest: a pre-v2 flat file-tree dish placed beside v2 dishes is ignored by dish enumeration and every query function (no crash, no rows) — `petri launch` works on a project containing old v0.3.x data

---

## M6-storage.2 — Move dashboard REST endpoints to the petri.sqlite query layer and add /api/edges

**Size:** M · **Labels:** dashboard, storage, migration-v2 · **Depends on:** Add petri/query/ read layer: typed SQL query functions over petri.sqlite · **Field issues:** relates:#15

Repoint every endpoint in `petri/dashboard/api.py` that reads the disposable SQLite index or re-deserializes colonies per request onto the new `petri/query/` layer over each dish's petri.sqlite, keeping response shapes frontend-compatible (`petri/templates/frontend.html` calls /api/cells:1694, /api/queue:1695, /api/stats:1696, /api/events:2236, /api/cell:2292, /api/dishes:3250).

What currently reads the disposable index: GET /api/events (`api.py:521-570` via get_db `api.py:136-142`), GET /api/cell/{cell_id} events block (`api.py:639-661`), GET /api/stats event aggregates (`api.py:709-728`). What re-deserializes every colony per request: GET /api/cells (`api.py:581-615`) and GET /api/cell/{cell_id} (`api.py:619-705`). There is no /api/edges endpoint — field issue #15 (see docs/field-reports.md) patched one in locally for cross-colony edge rendering; this issue ships it properly.

Scope:
- `create_app(petri_dir, db_path)` (`api.py:148`) drops the `db_path` parameter (it pointed at the old root-level `.petri/petri.sqlite` disposable index — not to be confused with the per-dish petri.sqlite domain database); update the call site `petri/cli/launch.py:175` and remove `db_path = petri_dir / "petri.sqlite"` (`launch.py:142`). Keep the index-dependent code paths deletable but do NOT delete migrate.py here (retirement issue).
- /api/events, /api/stats (event half), and the events block of /api/cell/{id} call petri/query functions; JSON response shapes byte-compatible with today (pinned by tests).
- /api/cells uses query.list_cells; /api/cell/{id} serves claim/evidence/summary content from the cells table — per amended D4 (docs/ARCHITECTURE-V2.md) the cells table replaces the v1 evidence.md/summary.md stubs, so the colony.json cell_paths file resolution (`api.py:670-674`) goes away; the response keeps its evidence_md/summary_md keys, now sourced from petri.sqlite.
- New GET /api/edges returning the live rows of the `edges` table (typed cross-colony reference edges) via query.get_edges; empty list when the table is empty. The row schema is owned by M3-decomposer's edge_registry (which writes rows) — coordinate field names in the PR, do not fork the schema.
- Old-dish tolerance (D8, docs/ARCHITECTURE-V2.md): /api/dishes lists only v2 dishes (directories containing petri.sqlite); `petri launch` on a project that also contains pre-v2 file-tree dishes ignores them gracefully — no crash, no phantom entries.
- Minimal frontend change: fetch /api/edges alongside /api/cells (frontend.html:1694) and include edges in the DAG data structure with their edge_type. The full field-issue-#15 visual overhaul (golden-angle spiral, collapsible tree sidebar) stays out of scope.

Out of scope:
- /api/stream SSE (separate issue).
- /api/queue and queue-derived stats keys — that repoint moved to M4-dbos (dashboard reads execution state via the ExecutionBackend seam); until it lands, the queue half of /api/stats (`api.py:731-742`) keeps reading queue.json (strangler).
- The stale version string (`petri/dashboard/frontend.py:87` hardcoded 0.3.0 and the stale `petri/__init__.py` version) — owned by M5-otel's dashboard waterfall issue, not here.
- Deleting migrate.py/_tail_loop.

Touched files: `petri/dashboard/api.py`, `petri/templates/frontend.html`, `petri/cli/launch.py`, `tests/` (new test_dashboard_api tests).

**Acceptance criteria:**
- [ ] pytest (httpx TestClient): GET /api/events with each filter combination returns the same JSON shape as v0.3.4 for a seeded fixture dish, with no disposable index database present (no root-level .petri/petri.sqlite)
- [ ] pytest: GET /api/cells and GET /api/cell/{id} return current field sets (cell_id, colony_id, claim_text, level, status, dependencies, dependents, evidence_md, summary_md, events) sourced entirely from the per-dish petri.sqlite — no per-cell files are read
- [ ] pytest: GET /api/edges returns [] on a dish with an empty edges table and returns the typed edges from fixture rows
- [ ] pytest: GET /api/stats event aggregates (total_events, cells_with_events, events_by_type, top_cells) match petri/query's event_stats for the fixture
- [ ] pytest: /api/dishes on a project containing a pre-v2 file-tree dish alongside v2 dishes lists only the v2 dishes, and `petri launch` serves all endpoints without error (pre-v2 dishes ignored gracefully)
- [ ] Manual: `petri launch` on a seeded dish renders the DAG with cross-colony edges visible when edges rows exist

---

## M6-storage.3 — Rebuild /api/stream SSE on events-table tailing plus pydantic-ai event_stream_handler live progress

**Size:** M · **Labels:** dashboard, observability, agents, migration-v2 · **Depends on:** Add petri/query/ read layer: typed SQL query functions over petri.sqlite · **Field issues:** relates:#15

The current SSE endpoint (`petri/dashboard/api.py:760-807`) polls the disposable SQLite index every 2s for `rowid > last_rowid` — pseudo-push over a copy of a copy, with `except Exception: pass` at `api.py:804-805`. In v2 the disposable index is gone, but the domain `events` table in petri.sqlite is append-only with a monotonic integer key, so the same cheap `rowid > ?` tailing works directly against the source of truth via `petri/query`'s events_since. Additionally, the locked v2 constraint (docs/ARCHITECTURE-V2.md) is that `run_stream()` must NOT be used inside DBOS workflows — live agent output comes from an `event_stream_handler` configured on the agent instead.

Scope:
- Domain-event channel: replace index polling with `query.events_since(integer watermark)` against the dish's petri.sqlite, using a short-lived read-only connection per poll (WAL snapshot reads make concurrent writer/reader correctness a non-issue by construction — amended D4, docs/ARCHITECTURE-V2.md). Keep the emitted `event_inserted` payload shape (`api.py:792-803`: id, cell_id, type, agent, iteration — the uniform event envelope, now table columns) so the existing frontend handler keeps working; verify the handler is not the stale one field issue #15 found (see docs/field-reports.md: its SSE handler was calling stale loadLabData()).
- Live-progress channel: new in-process pub/sub broker (`petri/dashboard/broker.py`: `publish(progress_event)` / `subscribe() -> async iterator`) plus an adapter (`petri/engine/streaming.py`) that converts pydantic-ai agent stream events (as delivered to an `event_stream_handler`) into small JSON payloads {cell_id, agent, kind, text_delta}. This replaces the hand-threaded CellProgressEvent/on_progress plumbing (`petri/engine/processor.py:57`, `:1536-1552`) as the dashboard-facing surface.
- SSE emits two event types: `event_inserted` (domain) and `agent_progress` (live). Frontend: minimal handling of agent_progress (append to the logs panel, newest-first per field issue #15's fix).
- Replace the two `except Exception: pass` blocks (`api.py:804-805` and the tail loop's, if still present when this lands) with logged errors.
- OPEN QUESTION (flag in PR, decide with maintainer): transport for runs NOT started inside the dashboard process. If v2 runs DBOS workflows in-process under `petri launch`, the broker is fed directly; for detached CLI `petri grow` runs, live token streaming is unavailable and the domain-event tail (<=2s latency, same as today) is the coverage — do we accept that, or add a local-socket bridge later? The broker interface must not preclude either answer.

Out of scope:
- Wiring event_stream_handler onto the real production agents (M2-agents owns agent construction; M4-dbos owns workflow wiring; this issue ships the adapter + broker, testable with fake stream events).
- OTel span persistence / trace view (D7 — M5-otel; its spans/usage tables in petri.sqlite land as their own small migration, independent of this milestone).
- Deleting migrate.py.

Touched files: `petri/dashboard/api.py:760-807`, `petri/templates/frontend.html` (SSE handler around :1651). New files: `petri/dashboard/broker.py`, `petri/engine/streaming.py`, `tests/unit/test_broker.py`, `tests/unit/test_sse_stream.py`.

**Acceptance criteria:**
- [ ] pytest (httpx TestClient + asyncio): appending an event row through the domain write seam while /api/stream is connected yields an `event_inserted` SSE message with the current payload shape within one poll interval
- [ ] pytest: watermark resumes correctly — reconnecting after N new events replays exactly those N (no duplicates, no gaps), including events sharing a timestamp (the integer append-order watermark, not timestamps, defines order)
- [ ] pytest: publishing a fake pydantic-ai stream event through the adapter yields an `agent_progress` SSE message with cell_id/agent/text delta; multiple subscribers each receive it
- [ ] pytest: a subscriber that disconnects does not block or crash the broker (publish continues for remaining subscribers)
- [ ] No `except Exception: pass` remains in the SSE code path (grep-pinned)
- [ ] Manual: `petri launch` + a dish receiving new events shows live log updates without page refresh

---

## M6-storage.4 — Retire dashboard/migrate.py and the disposable SQLite dashboard index

**Size:** S · **Labels:** storage, dashboard, migration-v2, breaking-change, good-first-issue · **Good first issue** · **Depends on:** Move dashboard REST endpoints to the petri.sqlite query layer and add /api/edges; Rebuild /api/stream SSE on events-table tailing plus pydantic-ai event_stream_handler live progress

Once the REST endpoints and SSE stream read petri.sqlite via petri/query/, the JSONL-to-SQLite copy pipeline is dead code. Per amended D4 (docs/ARCHITECTURE-V2.md) this step is explicitly retired. Note the name collision: the file being retired is the old root-level `.petri/petri.sqlite` disposable dashboard index — NOT the per-dish `.petri/petri-dishes/<dish_id>/petri.sqlite` domain database, which is the v2 source of truth. Old root-level index / `combined.jsonl` files on users' disks are left untouched (D8: no migration tooling). Note: `rollup_to_combined` itself is already deleted by the query-layer issue — this issue removes the index machinery and sweeps any remaining references.

What to delete/remove:
- `petri/dashboard/migrate.py` entirely (init_db :45-52, rebuild_sqlite :55-100, incremental_sync :103-165) and its tests.
- In `petri/dashboard/api.py`: the remaining migrate imports, the `file_offsets` seeding block (:158-167), the `_tail_loop` background task (:827-841) and its lifespan wiring (:173-178), and the `get_db` helper (:136-142) once no endpoint uses it — domain reads live in petri/query/, which owns its own short-lived read-only connections.
- In `petri/cli/launch.py`: the `init_db, rebuild_sqlite` import (:134), the 'Building event index...' block (:171-173). Update the now-inaccurate comments justifying SIGKILL of stale dashboards via 'the sqlite index is disposable' (:66-67, :136-137) — the kill remains safe because the dashboard holds only short-lived read-only connections to the per-dish domain database (WAL) and owns no writable state.
- Docs: CLAUDE.md architecture block line 'dashboard/ FastAPI REST+SSE, SQLite migration' and the 'SQLite (disposable dashboard index)' entry under Active Technologies; any ARCHITECTURE.md mention of the disposable index/5-second tail.

Out of scope:
- storage/queue.py (retired by M4-dbos, not here).
- Any endpoint behavior change (must already be on petri/query/).

Touched files: `petri/dashboard/migrate.py` (delete), `petri/dashboard/api.py`, `petri/cli/launch.py`, `CLAUDE.md`, tests referencing migrate.

**Acceptance criteria:**
- [ ] `grep -rn 'migrate\|rebuild_sqlite\|incremental_sync\|combined.jsonl\|rollup_to_combined' petri/` returns no hits (pinned by a test or CI grep step)
- [ ] Manual + integration test: `petri launch` on a fresh seeded dish starts, serves /api/events and /api/stream, and creates neither a root-level .petri/petri.sqlite index nor .petri/combined.jsonl — the only databases present are the per-dish petri.sqlite domain DBs and DBOS's own system DB
- [ ] A pre-existing stale root-level petri.sqlite index / combined.jsonl in the fixture dir is ignored and unmodified after a full launch-and-query cycle (mtime/bytes unchanged)
- [ ] Full test suite passes with migrate tests removed
- [ ] CLAUDE.md no longer mentions the SQLite dashboard index

---

## M6-storage.5 — Add petri backup command with --list and a pre-destructive snapshot hook

**Size:** S · **Labels:** lifecycle, storage, migration-v2, good-first-issue · **Good first issue** · **Field issues:** #6, relates:#5

Field issue #6 (see docs/field-reports.md): colony data represents hours of paid LLM compute — a single 88-cell colony took ~13 minutes of Opus time — and there is no backup mechanism. The seed-overwrite bug (field issue #5, same index) destroyed those 88 cells with no recovery path. Per D8 (docs/ARCHITECTURE-V2.md), backup is a v2-forward safety feature: no v0.3.x salvage tooling.

Per amended D4 (docs/ARCHITECTURE-V2.md), backup simplifies to a consistent snapshot of the dish's petri.sqlite domain database: SQLite's `VACUUM INTO` produces a single transactionally-consistent copy even while writers are active (WAL) — one file captures the domain event log (events table), the colony DAG (cells/edges tables), and all views. Execution state (the DBOS system DB, dbos.sqlite) is deliberately excluded — it is recoverable/re-runnable and never shares the domain database.

Scope:
- New `petri/cli/backup.py` following the existing `register(app)` pattern (`petri/cli/grow.py:25-47` as reference); register in `petri/cli/__init__.py`.
- `petri backup [--dish <dish_id>]` snapshots the dish database via `VACUUM INTO` to `.petri/backups/<dish>-<UTC-timestamp>/petri.sqlite`, plus a copy of `.petri/defaults/` config. The database is the unit of consistency, so snapshots are whole-dish; per-colony text views are `petri export --colony`'s job, and the `petri backup --with-export` flag that bundles a text export into the snapshot directory is wired by the export issue (which owns export machinery).
- `petri backup --list`: snapshots newest-first with dish, timestamp, cell count (queried from the snapshot database), size.
- Reusable `snapshot_before_destructive(petri_dir, dish_id) -> Path` helper, exported for programmatic use.
- Hook wiring (single owner: this issue): the seed command's regenerate/overwrite path and the dashboard's POST /api/seed call `snapshot_before_destructive` before overwriting an existing colony. (The broader interactive seed-overwrite guard UX belongs to M7-lifecycle; the snapshot safety net ships here.)
- `.petri/backups/` must be invisible to readers: it lives outside `petri-dishes/`, and the petri/query dish-enumeration exclusion is pinned by test (coordinates with the query-layer issue's exclusion AC).
- Surface `VACUUM INTO` failures clearly: it needs free disk roughly equal to the database size and fails cleanly rather than corrupting — report the error and the required space hint.

Out of scope:
- `--restore` (split into the follow-up issue in this milestone).
- `--with-export` wiring (owned by the `petri export` issue).
- Automatic scheduled backups; compression/tar; cloud targets.
- Backing up or restoring DBOS execution state.

New files: `petri/cli/backup.py`, `tests/unit/test_cli_backup.py`. Touched: `petri/cli/__init__.py`, `petri/cli/seed.py` (regenerate-path hook call), `petri/dashboard/api.py` (POST /api/seed hook call), README CLI table, CLAUDE.md CLI Commands block.

**Acceptance criteria:**
- [ ] pytest: backing up a seeded fixture dish creates a snapshot petri.sqlite whose events/cells/edges row sets are identical to the source database's (opened read-only with stdlib sqlite3; VACUUM INTO rewrites pages, so compare rows, not bytes)
- [ ] pytest: a snapshot taken while a writer connection holds an uncommitted transaction contains only committed rows and passes PRAGMA integrity_check
- [ ] pytest: --list output is newest-first and includes every snapshot created in the test with dish, timestamp, cell count, and size
- [ ] pytest: snapshot_before_destructive returns the created snapshot path and is importable without CLI context
- [ ] pytest: re-running seed over an existing colony (regenerate/overwrite path) creates a snapshot before any row is modified
- [ ] pytest: dashboard POST /api/seed targeting an existing colony creates a snapshot before overwriting
- [ ] pytest: snapshot databases under .petri/backups/ never appear in petri/query dish enumeration or query results
- [ ] Manual: `petri backup && petri backup --list` works on a real dish

---

## M6-storage.6 — Add petri backup --restore with overwrite guard and execution-state desync warning

**Size:** S · **Labels:** lifecycle, storage, migration-v2 · **Depends on:** Add petri backup command with --list and a pre-destructive snapshot hook · **Field issues:** relates:#6

Follow-up to "Add petri backup command with --list and a pre-destructive snapshot hook" (split out to keep that issue small). Restores a snapshot created by `petri backup` back into the dish tree.

Context: snapshots are transactionally-consistent `VACUUM INTO` copies of the per-dish petri.sqlite domain database under `.petri/backups/<dish>-<UTC-timestamp>/` (plus defaults/ and, when `--with-export` was used, an exports/ text tree). Execution state is never snapshotted — DBOS keeps its own separate system DB file, never shared, never queried directly (amended D4 two-store separation, docs/ARCHITECTURE-V2.md) — so restoring the domain database while durable workflows still reference the pre-restore state can desync the two stores. Restore must warn, not silently proceed.

Scope:
- `petri backup --restore <snapshot-name>` in `petri/cli/backup.py`: copies the snapshot petri.sqlite back over the target dish's database, removing stale `-wal`/`-shm` sidecar files so the restored database opens cleanly; restores defaults/ if snapshotted.
- Refuses to overwrite an existing dish database without `--force` (exit non-zero with a clear message naming the target and the flag).
- When a DBOS system-DB file (dbos.sqlite) is present under `.petri/`, print a prominent desync warning (execution state may reference pre-restore domain state; recommend halting runs first) before proceeding.
- `--list` remains the discovery surface for snapshot names.

Out of scope:
- Partial/selective restore (single colony or cell — the database is the unit of consistency; `petri export` provides read-only per-colony views); restoring execution state; automatic run-halting.

Touched files: `petri/cli/backup.py`, `tests/unit/test_cli_backup.py`, README CLI table.

**Acceptance criteria:**
- [ ] pytest round-trip: back up a seeded fixture dish, append further events through the write seam, restore, and assert the restored database's events/cells/edges row sets exactly match the snapshot's (post-snapshot appends gone)
- [ ] pytest: restore onto a dish with an existing database exits non-zero without --force and succeeds with it
- [ ] pytest: restoring one dish's snapshot touches only that dish's petri.sqlite — sibling dishes' databases are unchanged (mtime/bytes)
- [ ] pytest: with a fake DBOS system-DB file present, restore prints the desync warning (captured output asserted)
- [ ] pytest: after restore, the database opens cleanly and passes PRAGMA integrity_check (stale -wal/-shm sidecars removed)
- [ ] Manual: `petri backup && petri backup --restore <name> --force` works on a real dish

---

## M6-storage.7 — Add petri export: derived JSONL/markdown artifacts for git and PR review

**Size:** M · **Labels:** storage, lifecycle, migration-v2, good-first-issue · **Good first issue** · **Depends on:** Add petri/query/ read layer: typed SQL query functions over petri.sqlite; Add petri backup command with --list and a pre-destructive snapshot hook

Per amended D4 (docs/ARCHITECTURE-V2.md), each dish's petri.sqlite database is the domain source of truth and text is a derived artifact: `petri export` regenerates JSONL and markdown from the database at any time, so dishes can be committed to git, reviewed in PRs, and grepped — without text files ever becoming a second source of truth. Determinism is load-bearing beyond this milestone: M4-dbos commits its golden grow fixture as exported text, so identical database state must always produce an identical export tree.

Scope:
- New `petri/cli/export.py` following the existing `register(app)` pattern (`petri/cli/grow.py:25-47` as reference); register in `petri/cli/__init__.py`.
- `petri export` (whole dish), `--colony <slug>`, and `--cell <cell_id>` scoping. Output under `.petri/petri-dishes/<dish_id>/exports/` (the dish layout from M3-decomposer): per-cell `events.jsonl` (one event per line, append order) and per-cell markdown (claim text, status, level, dependencies, verdict summary, and the evidence/summary content from the cells table), plus a dish-level `edges.jsonl` derived from the edges table so cross-colony structure is reviewable.
- Deterministic output: stable event ordering (the events table's append-order key), stable JSON key order, stable file iteration order, trailing newlines — the same database state always produces a byte-identical export tree (git-diff friendly).
- `--since <watermark>` incremental mode: rewrites only cells that gained events since the recorded watermark; an export manifest inside `exports/` records the last watermark and the export format version.
- Optional auto-export hook: a petri.yaml flag (and/or `petri grow --export`) runs an incremental export on grow completion — usable from CI.
- Wire `petri backup --with-export`: when passed, the backup command (previous issue) also writes an export tree inside the snapshot directory.
- All reads go through petri/query/ typed functions (read-only). Exports are never read back by Petri (derived artifacts only — no ingest/round-trip) and `exports/` is excluded from dish enumeration (coordinates with the query-layer issue's exclusion AC).

Out of scope:
- Importing exports back into petri.sqlite (text is derived; no salvage/ingest tooling — consistent with D8's no-compat stance).
- Exporting DBOS execution state or the spans/usage tables (M5-otel).
- Dashboard export UI.

Approachable and self-contained: pure read-only derivation over the query layer with no concurrency concerns — a good first issue once petri/query/ exists.

New files: `petri/cli/export.py`, `tests/unit/test_cli_export.py`. Touched: `petri/cli/__init__.py`, `petri/cli/backup.py` (--with-export), `petri/cli/grow.py` (completion hook), README CLI table, CLAUDE.md CLI Commands block.

**Acceptance criteria:**
- [ ] pytest: exporting a seeded fixture dish twice with no database changes produces byte-identical export trees (deterministic ordering, key order, newlines)
- [ ] pytest: each per-cell events.jsonl line matches the corresponding events-table row (same envelope fields: id, cell_id, timestamp, type, agent, iteration, data) in append order
- [ ] pytest: the per-cell markdown for a fixture converged cell contains its claim text, status, and verdict summary
- [ ] pytest: --since with a recorded watermark rewrites only cells that gained events (all other exported files byte-identical and mtime-unchanged) and updates the manifest watermark
- [ ] pytest: with the auto-export flag enabled, grow completion triggers an incremental export (pinned with a faked grow completion)
- [ ] pytest: `petri backup --with-export` writes an export tree inside the snapshot directory
- [ ] pytest + grep: files under exports/ are excluded from petri/query dish enumeration and no Petri code path reads exports/ back
- [ ] Manual: `petri export` on a real dish, then `git add` + `git diff --staged`, shows reviewable per-cell text

---

## M6-storage.8 — Replace the /api/proc PTY bridge and synchronous /api/seed with ExecutionBackend-driven runs

**Size:** M · **Labels:** dashboard, durable-execution, migration-v2, breaking-change · **Depends on:** Add petri backup command with --list and a pre-destructive snapshot hook · **Field issues:** relates:#8

The dashboard currently owns child processes in two ways, both of which violate the v2 architecture (D1/D4, docs/ARCHITECTURE-V2.md) where durable workflows own execution:

1. The /api/proc PTY bridge spawns interactive subprocesses from the dashboard and holds them in an in-process `_proc` session table; sessions are never reaped, and a dashboard restart orphans them.
2. POST /api/seed (`petri/dashboard/api.py:490`) runs the whole decomposition synchronously inside the HTTP request: a dropped connection, timeout, or dashboard restart mid-seed leaves a half-written colony on disk with no resume path.

M4-dbos ships the ExecutionBackend seam (execution state behind a swappable internal interface) and the durable seed workflow ("Re-platform petri seed onto the DBOS backend", which closes the interrupted-seed data-loss story — field issue #8, see docs/field-reports.md). This issue moves the dashboard onto that seam so it never owns long-running child processes.

Scope:
- Remove the /api/proc PTY bridge endpoints and the `_proc` session table; reap any live session state cleanly on shutdown during the transition.
- POST /api/seed becomes enqueue-only: validate input, call the ExecutionBackend to start the durable seed workflow (workflow id = colony id, matching M4's convention), return 202 with the workflow id and a status URL. No domain writes (no petri.sqlite rows) happen inside the request.
- New run-control endpoints delegating to the seam: POST /api/runs (start), POST /api/runs/{id}/cancel, GET /api/runs/{id} (status). Do not query DBOS system tables directly — only the ExecutionBackend protocol.
- Preserve the pre-destructive snapshot behavior: re-seeding an existing colony must still trigger `snapshot_before_destructive` (wired by the backup issue, which owns that hook) before any overwrite — in the new flow this happens before/at enqueue or as the workflow's first step; pin it by test either way.
- Tighten CORS: replace the wildcard `allow_origins` with the local dashboard origin(s) (http://127.0.0.1:8090 and http://localhost:8090 by default, configurable).
- Frontend: the seed form switches to enqueue-then-poll (or SSE) for status instead of blocking on the response.

Out of scope:
- Implementing the ExecutionBackend or the durable seed workflow (both M4-dbos).
- The domain-event SSE stream (separate issue).
- Authentication.

Touched files: `petri/dashboard/api.py`, `petri/templates/frontend.html`. New files: `tests/unit/test_dashboard_runs.py`.

**Acceptance criteria:**
- [ ] pytest: POST /api/seed with a faked ExecutionBackend returns 202 with a workflow id and performs zero domain writes (no petri.sqlite rows, no files) during the request
- [ ] pytest: POST /api/runs/{id}/cancel and GET /api/runs/{id} delegate to the faked backend; unknown run id returns 404
- [ ] grep-pinned: no PTY/_proc session code remains under petri/dashboard/
- [ ] pytest: re-seeding an existing colony via POST /api/seed triggers snapshot_before_destructive before any overwrite (faked backend)
- [ ] pytest: CORS configuration no longer allows all origins; a request from a disallowed origin receives no CORS allow headers
- [ ] Manual: seeding from the dashboard on a real dish completes via the durable workflow and survives a dashboard restart mid-run

---

## M6-storage.9 — Graph integrity hardening: one authoritative colony topology with validated edges and lossless round-trips

**Size:** M · **Labels:** storage, migration-v2 · **Field issues:** relates:#11

Colony topology is currently held twice in memory with nothing forcing agreement: each Cell carries `dependencies`/`dependents` lists (`petri/models.py`) while `ColonyGraph` maintains its own `_adj`/`_rev` adjacency maps (`petri/graph/colony.py`). Per the locked design (docs/ARCHITECTURE-V2.md), the colony DAG is runtime DATA — these in-memory structures stay in v2 — while its persistence is the `cells` + `edges` tables in petri.sqlite (amended D4; the schema and write seam are owned by M3-decomposer, with `dependents` a SQL view). v1's colony.json serialization goes away with the file tree, but its defects must not be carried into the new save/load path:

- `add_edge` does not validate that both endpoints exist and does not dedupe repeated edges.
- v1's serialize/deserialize round-trips DROPPED cross-colony edges: an edge to a cell in another colony vanished on reload (directly undermining the cross-colony reference-edge design — field issue #11, see docs/field-reports.md). The v2 save/load over the cells/edges tables must preserve them.
- `load_colonies` (`petri/cli/_bootstrap.py:57-72`) silently skips colonies that fail to load — users see cells disappear with no error.
- v1 `serialize_colony` output ordering was nondeterministic, defeating round-trip tests; v2 loads must use stable ORDER BY so round-trips are exact (text-diff determinism is `petri export`'s job, not this issue's).

Scope:
- Pick ONE authoritative in-memory topology representation (recommended: ColonyGraph adjacency) and derive the other (Cell.dependencies/dependents become computed views or are rebuilt on load); document the choice and invariant in the module docstring.
- `add_edge` validates both endpoints exist and dedupes; cross-colony endpoints (cells not present locally) are validated against the edge-registry row shape from M3-decomposer's `petri/storage/edge_registry.py` (which writes edges-table rows) rather than local cells, and are preserved as typed references. Dedupe is belt-and-braces: in-memory check plus the edges table's UNIQUE constraint.
- Colony save/load to/from the cells+edges tables (through M3's write seam) is lossless, including cross-colony reference edges; save-load-save is a fixpoint.
- `remove_cell` keeps both in-memory views consistent (no dangling ids) and keeps the cells/edges rows consistent with them (the `dependents` view reflects the removal).
- The loader surfaces invalid data (e.g., an edge row referencing a missing cell, a row failing model validation) with the dish/colony identified — while still loading healthy colonies — no silent skip.
- Regression tests: save-load-save fixpoint; remove_cell followed by round-trip; cross-colony edge survival.

Out of scope:
- The petri/query read layer (separate issue).
- The petri.sqlite schema and edge registry themselves (M3-decomposer) and cross-colony DishGraph queries/cycle detection (M7-lifecycle).

Touched files: `petri/graph/colony.py`, `petri/models.py`, `petri/cli/_bootstrap.py`, `tests/unit/` (colony/graph tests plus new round-trip regression tests).

**Acceptance criteria:**
- [ ] pytest: save-load-save over the cells/edges tables is a fixpoint for a fixture colony that includes a cross-colony edge (stable ordering, edge preserved — regression for v1's drop)
- [ ] pytest: add_edge with a nonexistent local endpoint raises a typed error; adding the same edge twice yields exactly one in-memory edge and exactly one edges-table row
- [ ] pytest: after remove_cell, neither Cell.dependencies/dependents nor ColonyGraph adjacency contains a dangling reference, and the cells/edges tables agree (dependents view reflects the removal)
- [ ] pytest: loading a dish containing one colony with an invalid edge row (missing endpoint) reports that colony by id and still loads the healthy colonies (no silent skip)
- [ ] pytest: Cell.dependencies/dependents agree with ColonyGraph adjacency after an arbitrary mutation sequence (add_edge/remove_cell property-style test)

---

## M6-storage.10 — Ship starter SQL analytics views in petri.sqlite over the event log

**Size:** S · **Labels:** storage, migration-v2, good-first-issue · **Good first issue** · **Depends on:** Add petri/query/ read layer: typed SQL query functions over petri.sqlite

With petri.sqlite as the domain source of truth and the petri/query layer in place, the append-only events table becomes queryable analytically with zero extra infrastructure — amended D4 (docs/ARCHITECTURE-V2.md) explicitly includes SQL views (convergence reads, analytics) as part of the database. Ship a small set of ready-made views and typed wrapper functions that answer the questions researchers actually ask about a running dish — and that M7-lifecycle's Analyst (read-only research-health monitoring) will later consume as its foundation.

The events table's columns mirror the uniform v1 envelope — every event carries exactly {id, cell_id, timestamp, type, agent, iteration, data}, with timestamps written as UTC ISO-8601 — so views can rely on those columns.

Views to ship:
- Blocking-verdict patterns: which agents issue non-passing verdicts most often, per colony and per level (from verdict-issued events: agent + verdict payload).
- Cell velocity: elapsed time from a cell's first event to its convergence event.
- Stalled-cell detection: cells with no new events for a configurable window.
- Per-colony convergence rates: converged/total cells per colony.

Scope:
- SQL view definitions shipped as a small forward-only schema migration using M3-decomposer's `PRAGMA user_version` mechanism (coordinate version numbering with M5-otel's spans/usage migration so the linear migration chain stays intact), plus typed wrapper functions in `petri/query/analytics.py` (one per view) returning plain dicts.
- A short docs section (README or docs/) documenting each view and showing one ad-hoc query example with the stock `sqlite3` CLI against a dish's petri.sqlite, so users can go beyond the shipped views.

Out of scope:
- Dashboard UI for analytics; the Analyst agent itself (M7-lifecycle).
- Span/token/cost analytics (M5-otel owns the spans/usage tables and cost accounting).

New files: `petri/query/analytics.py`, `tests/unit/test_analytics_views.py`. Touched: README or docs/, the schema migration module.

**Acceptance criteria:**
- [ ] pytest: each view returns the expected rows over a fixture dish with a known verdict/convergence history
- [ ] pytest: the stalled-cell view flags a fixture cell with only old timestamps and does not flag a freshly-active one
- [ ] pytest: all views return empty results (no errors) on a dish with zero events
- [ ] pytest: applying the views migration bumps PRAGMA user_version exactly once and re-opening the database does not re-apply it (forward-only, idempotent on open)
- [ ] Docs: an Analytics section documents each shipped view and one ad-hoc sqlite3 CLI query example

---
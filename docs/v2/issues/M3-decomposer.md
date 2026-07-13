# M3-decomposer — Issue Backlog

> Tracking epic for milestone **M3-decomposer**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Rebuild claim decomposition as a pydantic-ai Agent subsystem (D6): typed outputs with automatic retry (is_atomic required with bedrock_reason, typed supports/limits/rebuts relations, real created_at timestamps enforced NOT NULL by schema), a search_cells tool so colonies reference existing cells instead of duplicating them, bedrock/triviality stops, counterargument sub-claims, per-parent caps enforced in code with no counts visible in prompts, dish-config threading, an interactive edit path at seed approval, and seed safety (overwrite guard + resume). A pydantic_evals regression suite codifies the maintainer's v1-v4 prompt experiments so decomposition quality is measured, not eyeballed. This attacks the dominant field-report theme (see docs/field-reports.md): 983 template-driven cells vs a hand-crafted 101-claim/128-edge equivalent (9.7x inflation), ~25% trivial axioms, and a seed command that destroyed 88 cells. Imported from M6-storage: dish-id/cell-ID-parser consolidation and the dish-scoped petri-dishes/<dish_id>/ layout flip (field issue #14) — simplified by the amended D4 (see docs/ARCHITECTURE-V2.md) to a single per-dish petri.sqlite domain store plus an exports/ directory — sequenced BEFORE search_cells so cross-colony edges have a dish-level store to live in. The petri.sqlite schema itself is born here (events/cells/edges tables, UNIQUE deterministic event ids, PRAGMA user_version forward-only migrations, WAL mode, the single row-writing seam; stdlib sqlite3 only), and so is the canonical cross-colony edge store: petri/storage/edge_registry.py writing append-only rows to the edges table — M6-storage's query layer (petri/query/ SQL views) reads it, M7-lifecycle extends it. Moved out: the durable re-platform of seed onto DBOS (full closure of field issue #8) is a dedicated M4-dbos issue; verdict-driven re-decomposition (field issue #9) is M7-lifecycle; derived text artifacts of the sqlite store (`petri export` JSONL/markdown) are M6-storage.

**Shippable release.** A strangler release (e.g. 0.5.0 / v2-alpha) where `petri seed` — same CLI surface — runs the agentic decomposer end-to-end: organic claim-driven trees (no quota-fill at any level, including L1/root), the dish-scoped petri-dishes/<dish_id>/ layout with petri.sqlite as the single domain store, cross-colony reference edges persisted as append-only rows in the dish's edges table, bedrock stops on axioms with persisted bedrock_reason, typed counterargument sub-claims (relation=limits/rebuts), a Y/n/edit approval prompt, refusal to overwrite existing colonies, `petri seed --resume` after interruption (interim checkpoint in a seed_checkpoints table in petri.sqlite behind the SeedCheckpointStore seam, swapped to DBOS in M4-dbos), and `pytest`-runnable decomposition evals gating regressions. This alpha's claimed surface on v2 dishes is seed/check/graph/dashboard listing; `petri grow` on v2 dishes arrives with the M4-dbos re-platform, and derived text artifacts (`petri export` JSONL/markdown for git commits and PR review) arrive with M6-storage.

**Depends on milestones:** M1-harness

**Milestone risks:**
- pydantic-ai 2.9.0 API details (per-agent retries config, output_validator, event_stream_handler, pydantic-graph persistence idiom) must be verified against the installed package source — each dependent issue carries an explicit verify-before-coding open question; signatures may differ from secondhand summaries
- Seed resume ships as an interim checkpoint (seed_checkpoints table in petri.sqlite behind the SeedCheckpointStore seam) and must be re-platformed onto DBOS by the dedicated M4-dbos issue (SetWorkflowID = colony id, full closure of field issue #8); the M4-dbos DBOS-on-SQLite spike gates the design assumptions — if the spike fails, the interim path becomes permanent, though SQLite transactional writes already give it the atomicity the old side-file design would have needed hardening for. The SeedCheckpointStore seam is the mitigation
- The dish-scoped petri.sqlite flip touches CLI, engine, dashboard, storage, and every test fixture mid-milestone; it is deliberately sequenced early (before search_cells), the amended D4 removes most per-cell path handling, and domain I/O is confined to the schema issue's single write seam to contain the blast radius, but it will conflict with any in-flight branch
- Until the M4-dbos re-platform lands, `petri grow` cannot run against v2 petri.sqlite dishes (v1 grow code and its agent-facing templates read the retired file tree); the M3 release deliberately claims seed/check/graph/dashboard-listing only — release notes must say so explicitly, and the seed overwrite guard plus --resume keep v2 dishes safe in the interim
- cell_search reads the cells/edges tables via stdlib SQL from day one; M6-storage's petri/query/ SQL views may later back the candidate fetch — both read paths must stay behind one function signature or they drift
- Cross-colony edges live in the dish-level edges table in petri.sqlite (petri/storage/edge_registry.py is the sole writer) because v1 deserialize_colony drops out-of-colony edges (colony.py:384-386); until M6-storage's query layer reads them everywhere and M7-lifecycle extends the registry (DishGraph queries, cycle detection, tombstones), reference edges are invisible in some views and cross-colony cycle detection remains impossible at the graph layer
- Decomposition quality is statistical: offline FunctionModel evals validate mechanics, not model behavior — a live eval smoke on the real default model is required before tagging the milestone release, and it costs real tokens
- Prompt-content changes (bedrock, counterarguments, neutral count phrasing) interact — v3 showed over-correction causes binary-split tunneling; land them in the given issue order and re-run live evals after each, not only at the end
- The counterargument and parent-child near-duplication output_validator retry loops could burn tokens on genuinely one-sided or tightly-scoped claims; both are bounded to one retry by design, but thresholds need live-eval tuning
- Milestone assumes M1-harness's Model seam (pi default / Claude Code adapter) is stable enough for tool-calling agents (search_cells requires tool support through the harness protocol) — if pi's RPC tool bridge slips, agents can fall back to direct pydantic-ai providers (D3) at the cost of subscription-auth parity

---

## M3-decomposer.1 — Thread dish config through the decomposition path and retire import-time config constants

**Size:** S · **Labels:** migration-v2, decomposer · **Field issues:** relates:#2

Context: field issue #2, config split-brain (see docs/field-reports.md) — `decompose_claim()` reads the package-installed defaults, not the dish's `.petri/defaults/petri.yaml`. Root cause: `load_config` is `@functools.lru_cache(maxsize=1)` with a silent fallback to packaged defaults when the given path does not exist (petri/config.py:18-27, fallback at line 21), and four module-level constants are frozen at import time (petri/config.py:247-250). `_provider_decompose` lazily imports `get_max_decomposition_depth`/`get_max_nodes_per_layer` (petri/reasoning/decomposer.py:247-250) so a dish config of depth 10 / 13 nodes-per-layer still produced 16-cell depth-3 trees; 10 colonies were deleted because of this. `load_dish_config` already exists (petri/config.py:30-41) but the decomposition call chain never uses it.

Scope:
- New typed settings model `DecompositionSettings` (pydantic BaseModel: `max_decomposition_depth`, `max_nodes_per_layer`, `per_parent_cap` placeholder for the caps issue, model string) in a new module `petri/reasoning/decomposition_settings.py`, built from `load_dish_config(petri_dir)` with packaged defaults as explicit fallback values — structure in code, content in YAML (D9).
- Thread the settings object explicitly: `petri/cli/seed.py` builds it from the resolved dish and passes it to `decompose_claim(...)`; `_provider_decompose` (decomposer.py:219-417) takes it as a parameter instead of importing config getters.
- Make `load_config(<explicit nonexistent path>)` raise instead of silently falling back (config.py:21); the no-argument default-load behaviour stays.
- Remove decomposition-path reads of the import-time constants; leave `LLM_INFERENCE_MODEL`/`MAX_ITERATIONS`/`MAX_CONCURRENT`/`AGENT_TOOLS` (config.py:247-250) in place for the grow pipeline — retiring those fully belongs to M4-dbos (its retire-v1 issue owns full config.py retirement).

Out of scope: grow/processor config threading; pydantic-settings adoption codebase-wide; changing petri.yaml schema.

Touches: petri/config.py:18-27, 30-41, 75-80, 114-126, 247-250; petri/reasoning/decomposer.py:247-250, 67-118, 219-417; petri/cli/seed.py (provider/settings resolution around lines 203-217); new petri/reasoning/decomposition_settings.py; tests/unit/test_decomposition_settings.py (new).

**Acceptance criteria:**
- [ ] pytest: two temp dishes with different `max_decomposition_depth`/`max_nodes_per_layer` in their `.petri/defaults/petri.yaml` yield different DecompositionSettings in the same process (no lru_cache poisoning), asserted without process restart
- [ ] pytest: `load_config(Path('/nonexistent/petri.yaml'))` raises (FileNotFoundError or ValueError) instead of silently returning packaged defaults
- [ ] pytest: `_provider_decompose` driven with an injected fake provider respects the settings object's depth/per-layer values, not the packaged defaults
- [ ] grep-based test or lint assertion: `petri/reasoning/decomposer.py` no longer imports `get_max_decomposition_depth`/`get_max_nodes_per_layer`
- [ ] Existing test suite (`uv run pytest tests/`) passes unchanged

---

## M3-decomposer.2 — Consolidate dish-id resolution and cell-ID parsing into single canonical helpers

**Size:** S · **Labels:** migration-v2, storage · **Field issues:** relates:#14

Context (imported into M3-decomposer from M6-storage so it lands before the dish-scoped layout flip and the search_cells edge registry): dish identity and cell-ID parsing are each implemented several times with diverging semantics.

Dish-id resolution exists in FOUR divergent copies:
- petri/cli/_bootstrap.py:51 `get_dish_id` — `load_dish_config(petri_dir)` then `config.get('name', petri_dir.parent.name)`
- petri/engine/processor.py:138 `_get_dish_id` — reads `petri_dir/defaults/petri.yaml` via yaml.safe_load (a DIFFERENT file from the others)
- petri/engine/propagation.py:84 `_get_dish_id` — line-parses `petri_dir/petri.yaml` without a YAML dependency
- petri/dashboard/api.py:815 `_get_dish_id` — line-parses `petri_dir/petri.yaml` with a subtly different guard
The divergence is real: the grow engine can compute a different dish id than the CLI/dashboard for the same dish, because it reads a different file. Cell IDs embed the dish (`{dish}-{colony}-{level}-{seq}`), so a divergent dish id silently mis-addresses cells.

Cell-ID parsing has THREE implementations plus ad-hoc splits: petri/models.py:522 `parse_key` (accepts an optional dish_id, which resolves the multi-hyphen dish/colony ambiguity), petri/storage/paths.py:47 `parse_cell_id` (self-documented as a lossy heuristic), petri/storage/queue.py:310 raw `rsplit('-', 2)`; plus ad-hoc rsplits at petri/cli_ui.py:567 and petri/engine/processor.py:153-160 (`_colony_slug`).

Scope:
- New `petri/storage/dish.py` exposing `resolve_dish_id(petri_dir) -> str` as the ONLY dish-id resolver: dish config via `load_dish_config` (petri/config.py:30-41), fallback to `petri_dir.parent.name`. Delete all four existing copies and convert their call sites. Document in the docstring which config file is canonical (the one `load_dish_config` reads) — this settles the petri.yaml vs defaults/petri.yaml divergence.
- Make `petri.models.parse_key` the single cell-ID parser. `paths.parse_cell_id` becomes a thin deprecated wrapper delegating to `parse_key` (emits DeprecationWarning); convert its direct call sites (petri/cli/check.py:107, petri/cli/stop.py:64, petri/cli/seed.py:99). Convert the raw rsplit sites (queue.py:310, cli_ui.py:567, processor.py:153-160) to `parse_key` with the resolved dish id.

Out of scope: the layout flip itself (later issue); multi-dish support; changing the composite-key schema.

Touches: new petri/storage/dish.py; petri/cli/_bootstrap.py:51-55; petri/engine/processor.py:138-160; petri/engine/propagation.py:84-92; petri/dashboard/api.py:815-824; petri/storage/paths.py:47-102; petri/storage/queue.py:310; petri/cli_ui.py:567; petri/cli/check.py, petri/cli/stop.py, petri/cli/seed.py (imports); tests/unit/test_dish_resolution.py (new).

**Acceptance criteria:**
- [ ] pytest: `resolve_dish_id` returns the configured dish name when set and the parent-directory name when absent, and returns the SAME value for the contexts that previously used the CLI, engine, and dashboard copies (regression for the defaults/petri.yaml vs petri.yaml divergence)
- [ ] grep test: no definition of `get_dish_id`/`_get_dish_id` remains outside petri/storage/dish.py
- [ ] pytest: `parse_key` with a dish_id correctly parses a cell id whose dish and colony slugs both contain hyphens; `paths.parse_cell_id` emits DeprecationWarning and returns the same tuple
- [ ] grep test: no raw `rsplit("-", 2)`-style cell-id parsing outside petri/models.py
- [ ] Existing test suite (`uv run pytest tests/`) passes unchanged

---

## M3-decomposer.3 — Define the petri.sqlite schema and migration mechanism

**Size:** M · **Labels:** migration-v2, storage

Context: D4 as amended (see docs/ARCHITECTURE-V2.md) makes a single Petri-owned SQLite file per dish — `petri.sqlite` — the domain source of truth, written with stdlib `sqlite3` only: the `events` table replaces per-cell JSONL logs as the append-only domain event log, and `cells` + `edges` put the colony DAG on disk (replacing metadata.json mutations and evidence.md stubs), with `dependents` derived as a SQL view. DBOS keeps its own separate system DB (`dbos.sqlite`, M4-dbos) — never shared tables, never queried directly; execution state only. The event-sourcing invariant is preserved and becomes schema-enforced: events are append-only, never edited, never deleted. Text becomes a derived artifact: M6-storage's `petri export` regenerates JSONL/markdown from this file at any time. This issue creates the schema, the migration mechanism, and the write seam that the dish-layout issue and every persistence issue in this milestone (decomposer cell persistence, edge registry, seed checkpoints) build on.

Scope:
- New `petri/storage/database.py` (the v2 successor to the event_log write seam): `connect(db_path)` applying `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, and a busy_timeout; forward-only migrations keyed by `PRAGMA user_version` (ordered SQL scripts, each applied in one transaction; opening a DB whose user_version is NEWER than the code knows fails loudly with a clear message).
- Schema v1: `events` (deterministic TEXT event id with a UNIQUE constraint, cell_id, event_type, JSON payload, created_at NOT NULL) — idempotent appends via `INSERT OR IGNORE`; `cells` (cell id PK, colony, level, seq, claim_text, status, relation, is_atomic, bedrock_reason, created_at TEXT NOT NULL CHECK(created_at <> '')); `edges` (from_cell, to_cell, edge_type, reason, created_at NOT NULL); a `dependents` VIEW derived from edges.
- Append-only enforcement in schema: SQL triggers raise on UPDATE/DELETE of `events` rows; the write seam exposes `append_event(...)` and read functions only — no update/delete API for events.
- Single write seam: all domain writes (events, cells, edges) go through this module's functions. `storage/event_log.py`'s public append/read surface is re-pointed here (or kept as thin delegating wrappers) so existing callers converge on rows, not lines — full call-site conversion happens in the layout issue and M4-dbos.
- Deterministic event ids: document the derivation contract now — a stable function of the event's coordinates (for decomposition events: cell_id + event_type + counter; the grow-pipeline scheme (cell_id, round, node, agent, iteration) is exercised by M4-dbos's idempotent-append work) — stored in the UNIQUE column.
- Zero new dependencies: stdlib sqlite3 only; nothing added to pyproject.

Out of scope: `spans`/`usage` tables (M5-otel adds them as its own small migration bumping user_version); the `seed_checkpoints` table (the seed-resume issue adds it via the migration mechanism defined here); petri/query/ SQL views for dashboard/analytics and `petri export` (M6-storage); the dish-scoped on-disk location of the file (the layout issue); the DBOS system DB (M4-dbos, separate file); porting grow-pipeline writers (M4-dbos).

Touches: new petri/storage/database.py; new petri/storage/migrations/ (or an in-module ordered migration list); petri/storage/event_log.py (seam re-pointing/wrappers); tests/unit/test_sqlite_schema.py (new).

**Acceptance criteria:**
- [ ] pytest: `connect()` on a fresh path creates petri.sqlite with journal_mode=WAL and user_version at the current schema version; reopening applies no migrations (idempotent); a DB stamped with a FUTURE user_version fails loudly with a clear error
- [ ] pytest: appending the same deterministic event id twice through the seam yields exactly one row (INSERT OR IGNORE observed) and the second append is reported as a duplicate
- [ ] pytest: direct UPDATE/DELETE on the events table raises via the schema triggers, and the write-seam module exposes no update/delete API for events (API-surface assertion)
- [ ] pytest: inserting a cell row with NULL or '' created_at fails the schema constraint (the v1 empty-string bug is unrepresentable at the storage layer)
- [ ] pytest: the `dependents` view returns the correct dependents for a small fixture DAG inserted via the seam
- [ ] no third-party storage dependency is added; storage uses stdlib sqlite3 only (pyproject + import assertion: no duckdb, no sqlalchemy, no aiosqlite)
- [ ] Existing test suite (`uv run pytest tests/`) passes unchanged — this issue only introduces the module; nothing is converted yet

---

## M3-decomposer.4 — Adopt the dish-scoped petri-dishes/<dish_id>/ layout with petri.sqlite as the domain store

**Size:** M · **Labels:** migration-v2, storage, breaking-change · **Depends on:** Consolidate dish-id resolution and cell-ID parsing into single canonical helpers; Define the petri.sqlite schema and migration mechanism · **Field issues:** #14

Context: field issue #14 (see docs/field-reports.md). Cell IDs are composite `{dish}-{colony}-{level}-{seq}`, but on disk there is NO dish level: colonies live directly at `.petri/petri-dishes/<colony_slug>/`, and petri/storage/paths.py:105-134 (`colony_dir`) explicitly documents that no intermediate `<dish_id>/` subdirectory exists. Consequences: the dish exists in identifiers but not in the filesystem, and dish-level artifacts have nowhere to live — specifically the cross-colony `edges` table added later in this milestone needs a dish-level store. This issue was moved into M3-decomposer from M6-storage by the plan verifiers precisely so it lands BEFORE the search_cells issue. Under D4 as amended (see docs/ARCHITECTURE-V2.md) the issue SIMPLIFIES relative to the original file-tree flip: instead of moving a per-cell directory tree one level down, the dish directory contains a single `petri.sqlite` domain store — most per-cell path handling disappears outright.

Scope:
- New on-disk tree: `.petri/petri-dishes/<dish_id>/` containing `petri.sqlite` (schema + write seam from the schema issue) and `exports/` (created eagerly, empty until M6-storage's `petri export` populates it). Keyed by `resolve_dish_id` (consolidation issue).
- `petri init` creates `petri-dishes/<dish_id>/` eagerly and initializes petri.sqlite (migrations run to current user_version).
- Single choke point: paths.py exposes `dish_dir(petri_dir)` and `dish_db_path(petri_dir)`; the per-colony/per-cell helpers (`colony_dir` at paths.py:105-134, `iter_events_files` at paths.py:170-192, per-cell dirs) are retired from the v2 seed path — v2 domain reads/writes go through the petri.sqlite seam instead.
- Convert surfaces that constructed file-tree paths or counted metadata.json files to seam-backed equivalents: petri/cli/seed.py:291 (colony existence check and the seed overwrite guard's cell count become cells-table queries), petri/cli/connect.py:94 (edge creation writes an edges row via the seam), petri/cli/stop.py:69, petri/cli/_bootstrap.py:62, petri/cli/init.py:154, petri/cli/launch.py:164 (repair list), petri/dashboard/api.py:374, 490, 586, 624 (colony listing and POST /api/seed read/write petri.sqlite — the minimum so the existing dashboard listing and seed endpoint keep working; the full REST conversion is M6-storage), petri/engine/propagation.py:45.
- Minimal row-backed colony load: a single `load_colony(db, colony)` helper reading cells/edges rows so `petri check`/`petri graph` and the dashboard listing keep working ahead of M6-storage's petri/query/ views; keep it small and behind one function.
- D8 (no compatibility): pre-v2 colonies sitting directly under `petri-dishes/` are left on disk and IGNORED by v2 code — no migration command, no dual-format reader. (M6-storage separately ensures `petri launch` ignores pre-v2 file-tree dishes gracefully.)
- Update all tests/fixtures: fixtures become temp dishes seeded through the write seam into petri.sqlite (shared fixture helpers), not directory trees.

Out of scope: multi-dish support within one `.petri` (single dish remains the model; the layout just makes the dish explicit); the edge registry itself (search_cells issue); migrating or reading v0.3.x dishes (D8); `petri export` and the petri/query/ read layer (M6-storage); re-platforming `petri grow` and the agent-facing skill templates (petri/templates/skill_read_cell.txt, skill_event_log_read.txt, skill_event_log_write.txt) onto the row-backed store — v1 grow code reads the retired file tree, and grow on v2 dishes lands with M4-dbos (D1 strangler: subsystem-by-subsystem).

Touches: petri/storage/paths.py:105-192 (retire/replace helpers); petri/storage/event_log.py:4, 185-200 (module docs + dish scan, now seam-backed); petri/cli/seed.py:291, connect.py:94, stop.py:69, _bootstrap.py:62, init.py:154, launch.py:164; petri/dashboard/api.py:374, 490, 586, 624; petri/engine/propagation.py:45; tests/ fixtures throughout.

**Acceptance criteria:**
- [ ] pytest: a temp-dish seed -> check -> graph round-trip persists cells and edges as rows in `.petri/petri-dishes/<dish_id>/petri.sqlite` and every converted CLI command resolves them from the database
- [ ] grep test: no `petri_dir / "petri-dishes" / <colony>`-style path construction outside petri/storage/paths.py — path building has exactly one choke point, and v2 code constructs no per-cell directories
- [ ] pytest: `petri init` creates `petri-dishes/<dish_id>/` containing an initialized petri.sqlite (schema at current user_version) and an empty exports/ directory
- [ ] pytest: a leftover pre-v2 colony directory placed directly under `petri-dishes/` is ignored (not loaded, no crash) by `petri check` and the dashboard colony listing (D8)
- [ ] pytest: dashboard colony listing and POST /api/seed operate on the dish-scoped petri.sqlite — the listing reflects seeded rows and the seed endpoint writes through the same seam as the CLI
- [ ] Existing test suite passes with fixtures converted to seam-backed temp petri.sqlite dishes

---

## M3-decomposer.5 — Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry

**Size:** M · **Labels:** migration-v2, agents, decomposer, harness · **Depends on:** Thread dish config through the decomposition path and retire import-time config constants; Define the petri.sqlite schema and migration mechanism

Context: decomposition currently shells out via `ClaudeCodeProvider.decompose_claim` (petri/reasoning/claude_code_provider.py:589-640) and `decompose_why` (642-688), parsing free text with the three-tier regex `_extract_json` (170-192). Failures silently become `{"nodes": [], "edges": []}` (line 640) or `[]` (682-688), making a parse failure indistinguishable from a genuinely atomic premise — the tree gets silently truncated. `_provider_decompose` (petri/reasoning/decomposer.py:219-417) then probes raw dicts (301-336, 379-408) and duck-types the provider with `hasattr(provider, 'decompose_why')` (363). Two facts from the on-disk field data motivate the schema design: no atomicity marker is persisted in ANY real v1 cell metadata today (real metadata.json files carry exactly 8 keys, none atomicity-related), and `created_at` is an empty string in every real v1 cell metadata file — a live bug.

Scope:
- New `petri/reasoning/decomposition_schemas.py`: `PremiseCandidate` (claim_text, plus a `relation` placeholder field typed fully in the quality-prompts issue), `InitialDecomposition` (list of premises), `WhyExpansion` (sub_premises, `is_atomic: bool` as a REQUIRED field, `bedrock_reason: str | None` with a model validator requiring it to be non-None whenever `is_atomic=True`). Atomicity becomes an explicit, always-present typed field — never inferred from emptiness; empty-on-failure disappears — exhausted retries raise.
- New `petri/reasoning/decomposition_agents.py`: two module-level pydantic-ai Agents (`initial_decomposition_agent`, `five_whys_agent`) with `output_type` set to the schemas above and unique `name`s (required later for DBOS durable wrapping — D2; the wrapping itself lands in M4-dbos). Model comes from the pydantic-ai model string in petri.yaml via the M1-harness seam (pi default, Claude Code adapter, or direct `anthropic:`/`openai:` strings — D3/D5).
- Port the 4-step BRAINSTORM/PRIORITIZE/SELECT/EMIT prompts (claude_code_provider.py:609-635, 656-681) verbatim as agent instructions — they are domain IP; quality changes land in follow-on issues.
- Rewire `_provider_decompose` to call the agents; delete dict probing and the `hasattr` gate; on validation failure pydantic-ai reflects the error to the model (ModelRetry) automatically; after retries exhaust, raise a typed DecompositionError so `petri seed` fails loudly (matches decomposer.py:332-336 fail-loud precedent).
- Persistence: decomposer-created cells and edges are written through the petri.sqlite write seam as cells/edges rows (schema issue) — not metadata.json mutations. Cells table rows carry the typed fields (is_atomic/bedrock_reason/relation columns populated by this and the quality-prompts issue).
- Fix the created_at bug: cells created by the decomposer get a real UTC ISO-8601 `created_at` at creation time; the cells table's NOT NULL + non-empty CHECK constraint (schema issue) makes the v1 empty-string bug unrepresentable at the storage layer.
- Streaming: preserve the live-spinner UX (`on_progress`, decomposer.py:74, 226) using pydantic-ai stream events; note constraint: no `run_stream()` once wrapped in DBOS workflows (M4-dbos) — use `event_stream_handler`, so build the progress plumbing on the event-handler path now.
- Offline tests with TestModel/FunctionModel scripting well-formed and malformed responses.

Open questions (do not invent APIs): exact per-agent retry configuration shape (`retries` dict form) and the `event_stream_handler` signature must be verified against the installed pydantic-ai 2.9.0 source before coding — do not code from memory or secondhand summaries.

Out of scope: prompt-content changes (bedrock/counterargument/negative examples), per-parent caps, search_cells tool, `assess_cell`/other InferenceProvider methods (M4-dbos grow pipeline), deleting claude_code_provider.py (still used by grow).

Touches: petri/reasoning/decomposer.py:219-417 (esp. 267-273, 301-336, 349-410), petri/reasoning/claude_code_provider.py:170-192, 589-688 (decomposition methods become unused by seed), petri/models.py:484-490 (DecompositionResult stays as the aggregate return) and the Cell created_at path, petri/cli/seed.py decompose loop (390-472), cell persistence via the petri/storage/database.py seam (cells/edges rows); new petri/reasoning/decomposition_agents.py, petri/reasoning/decomposition_schemas.py, tests/unit/test_decomposition_agents.py.

**Acceptance criteria:**
- [ ] pytest with FunctionModel: a scripted malformed first response followed by a valid response completes decomposition (auto-retry observed via call count), with zero subprocess/CLI invocations
- [ ] pytest: schema-invalid output after retries exhaust raises DecompositionError — no silent empty colony; the `decompose_why`-equivalent path never returns [] for a failure
- [ ] pytest: `WhyExpansion` requires `is_atomic`; `is_atomic=True` with `bedrock_reason=None` fails validation (drives ModelRetry); `is_atomic=True` with a reason stops expansion of that premise and is distinguishable in the result from a failed call (assert on typed fields, not emptiness)
- [ ] pytest: cells created by the decomposer carry a non-empty ISO-8601 `created_at` that round-trips through the cells table in petri.sqlite, and the schema rejects NULL/empty created_at (regression test for the v1 empty-string bug)
- [ ] pytest: `on_progress`/stream-event plumbing receives text chunks during a FunctionModel-driven run (spinner contract preserved)
- [ ] `petri seed` manual smoke against a live model produces a colony whose cells and edges are persisted as rows in the dish's petri.sqlite (D1 strangler constraint: same CLI surface)
- [ ] grep test: `_extract_json` is not referenced from the seed/decomposition path

---

## M3-decomposer.6 — Add bedrock stop condition, counterargument sub-claims, and negative examples to the decomposition agents

**Size:** M · **Labels:** migration-v2, decomposer, agents · **Depends on:** Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry · **Field issues:** #10, #12, #13

Context: three field-validated prompt/schema fixes (see docs/field-reports.md). Field issue #10: ~25% of cells (243/983) were undisputed axioms ('OpenAI exists as a legal entity') because the only stop conditions are max depth and an atomicity flag the model 'rarely returns' (escape hatch buried at claude_code_provider.py:674-676); each trivial cell would burn the full 13-agent pipeline (~3,159 wasted calls). Field issue #13: `decompose_why` produces only supporting children — adversarial pressure is deferred to Red Team 'by which point the tree structure is fixed'. Field issue #12: without negative examples the model produces safe, unfalsifiable claims. All three were patched locally and validated (148-cell re-seed). A fourth observed failure mode: restatement children — a child that merely rephrases its parent (real field failures: healthcare-customer colony cell 002-001 restated 001-001; swarm-build colony 002-001 restated its parent) — wastes a full validation pipeline on a duplicate.

Scope (builds on the agents from the pydantic-ai rebuild):
- Bedrock test as an explicit FIRST step in the five_whys agent instructions: 'Would a knowledgeable domain expert accept this claim immediately without needing to verify anything? If YES, return is_atomic: true with the reason' (maintainer's validated wording). `is_atomic` and `bedrock_reason` are already required/typed in `WhyExpansion` (rebuild issue).
- Typed claim relations: `relation: Literal['supports', 'limits', 'rebuts']` on `PremiseCandidate` in petri/reasoning/decomposition_schemas.py (replacing the placeholder), so counterarguments are typed, not just present; five_whys instructions require at least one sub-claim be 'the strongest COUNTERARGUMENT to the parent claim'; an `@agent.output_validator` raises ModelRetry when a non-atomic expansion returns >=2 sub-premises with zero children whose relation is 'limits' or 'rebuts' (bounded: one retry, then accept — do not deadlock on genuinely one-sided claims; document this choice in code).
- Parent-child near-duplication guard: the same output_validator raises ModelRetry when a sub-premise near-duplicates its parent claim (normalized token-overlap similarity above a documented threshold — stdlib only). Bounded to one retry; if the retry still duplicates, the duplicate child is dropped with a warning event rather than persisted as a cell. The evals suite scores this same property (ParentChildDistinctness evaluator, separate issue).
- Negative examples: the three BAD CLAIM examples from the maintainer's validated patch (an obvious fact, a logical prerequisite, a too-vague claim) added to the initial-decomposition agent instructions, plus the quality bar from the maintainer's field notes: good claims carry quantified thresholds and explicit exclusions.
- Persist `relation` onto cells: field on `Cell` (petri/models.py:263-274, default 'supports') written by the decomposer loop (decomposer.py:379-408) and stored as a typed column on the cells table in petri.sqlite (column defined by the schema issue), so grow/dashboard can later distinguish counterargument cells; bedrock cells persist `bedrock_reason` as a cells-table column via the same cell-creation write path (note: no atomicity marker is persisted in any real v1 cell metadata today — this issue makes it real, as a typed column).

Out of scope: pre-grow triage skip of trivial cells (M4-dbos); verdict-driven re-decomposition (field issue #9 — M7-lifecycle, per D6); dashboard rendering of relation.

Touches: petri/reasoning/decomposition_agents.py (instructions + output_validator; new from prior issue), petri/reasoning/decomposition_schemas.py, petri/reasoning/decomposer.py:379-410 (carry relation/bedrock_reason onto the Cell row), petri/models.py:263-274; tests/unit/test_decomposition_quality_prompts.py (new).

**Acceptance criteria:**
- [ ] pytest with FunctionModel scripting an axiomatic reply: expansion of that premise stops, the cell's cells-table row carries bedrock_reason, and no further model call is made for it (call-count assertion)
- [ ] pytest: output_validator raises ModelRetry exactly once when a scripted expansion returns 3 supports/0 limits-or-rebuts sub-premises, and accepts the retry response; a scripted 1-sub-premise expansion is NOT retried
- [ ] pytest: a scripted expansion whose child near-verbatim restates the parent triggers ModelRetry; if the retry still duplicates, the duplicate child is dropped and no restatement cell is persisted (the domain event log — events table — records the drop)
- [ ] pytest: Cell instances produced by the decomposer carry relation, and it round-trips through the cells table in petri.sqlite (written via the seam, read back as a typed column)
- [ ] Instruction-content tests: initial agent instructions contain the three BAD CLAIM negative examples; five_whys instructions contain the bedrock-test sentence (string assertions keep prompt regressions loud)
- [ ] Live smoke (manual, documented in PR): seeding a claim with a known axiom sub-premise yields at least one is_atomic stop and at least one relation='limits' or relation='rebuts' cell

---

## M3-decomposer.7 — Stop the decomposer treating layer limits as quotas: per-parent caps in code, no counts in prompts

**Size:** S · **Labels:** migration-v2, decomposer · **Depends on:** Thread dish config through the decomposition path and retire import-time config constants; Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry · **Field issues:** #3

Context: field issue #3 (see docs/field-reports.md). The prompt 'Pick the TOP {max_premises}' (claude_code_provider.py:626, and 671 for decompose_why) makes the model treat the cap as a quota: with max_nodes_per_layer 5 every level has exactly 5 cells; with 13, exactly 13. Architectural root cause: `_provider_decompose` passes `remaining_budget` as `max_premises` (decomposer.py:358-371) — 'the model always sees a number and fills to it'. Prompt-only fixes were partial: L1 became organic but levels 2+ still filled the cap. The v1-v4 experiments showed neutral phrasing ('Let structure determine count') wins; inverse-bias phrasing over-corrects into binary-split tunneling. Post-patch field data shows the ROOT expansion is the remaining saturation point: 5 of 12 post-patch colonies had EXACTLY 8 cells at L1 (the interim patch's L1 ceiling of 8, saturated) while L2+ counts were organic — so the cap discipline must explicitly cover the root/L1 expansion, with no L1 count visible to the model either.

Scope:
- Add `per_parent_cap` (default 4) and `level_one_cap` (default 10) to `DecompositionSettings` (petri/reasoning/decomposition_settings.py, from the config-threading issue) — these are the maintainer's validated `min(4, remaining)` / `min(10, ceiling)` values, now first-class config with the defaults in petri/defaults/petri.yaml.
- Remove ALL numeric caps from prompt visibility, in BOTH agents (initial/root expansion included): agent instructions use the neutral v4-test phrasing ('Let the structure of the claim determine the count; do NOT pad to fill a quota; fewer is fine') with no interpolated N, no 'usually much less than N' editorializing.
- Enforcement moves entirely to code: hard-truncate model output to per_parent_cap per expansion (decomposer.py:379-408 loop) and to level_one_cap at level 1 / the root expansion (decomposer.py:301-325); `max_nodes_per_layer`/`max_decomposition_depth` remain as safety ceilings only (existing truncation at decomposer.py:305, 354-359, 380-381), never as prompt text.
- Delete `remaining_budget`-as-max_premises plumbing (decomposer.py:358, 366-371).
- Coordination: the evals suite (separate issue) asserts that L1 counts vary with claim complexity across fixture claims — guarding the root-cap saturation signature above.

Out of scope: changing debate/grow iteration caps; verdict-driven depth extension (field issue #9 — M7-lifecycle).

Touches: petri/reasoning/decomposer.py:256, 267-273, 301-325, 349-410; petri/reasoning/decomposition_agents.py instructions; petri/reasoning/decomposition_settings.py; petri/defaults/petri.yaml (add per_parent_cap/level_one_cap keys, keep max_nodes_per_layer/max_decomposition_depth documented as ceilings); tests/unit/test_decomposer_caps.py (new).

**Acceptance criteria:**
- [ ] pytest with FunctionModel returning 10 sub-premises: exactly per_parent_cap (4) children are created for that parent; level ceiling still enforced when cumulative children would exceed max_nodes_per_layer
- [ ] pytest: rendered instructions for BOTH agents (initial/root and five_whys) contain no digit interpolated from settings (regex assertion) and no longer contain the strings 'Pick the TOP' or 'AT MOST'
- [ ] pytest: the root/L1 expansion is code-truncated to level_one_cap with no numeric hint in the L1 prompt (regression for the exactly-8-at-L1 saturation seen in 5 of 12 post-patch field colonies)
- [ ] pytest: with FunctionModel returning varied counts (2, 5, 1), per-level cell counts in DecompositionResult mirror the model's counts (capped), demonstrating counts are model-driven not settings-driven
- [ ] pytest: max_decomposition_depth still terminates expansion (existing behaviour preserved, decomposer.py:354)
- [ ] petri/defaults/petri.yaml comments document max_nodes_per_layer/max_decomposition_depth as 'safety ceilings, never targets' (docs drift guard: string assertion in test or review checklist item)

---

## M3-decomposer.8 — Add search_cells tool so decomposition creates cross-colony reference edges instead of duplicate cells

**Size:** M · **Labels:** migration-v2, decomposer, agents, storage · **Depends on:** Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry; Adopt the dish-scoped petri-dishes/<dish_id>/ layout with petri.sqlite as the domain store; Define the petri.sqlite schema and migration mechanism · **Field issues:** #11, relates:#14

Context: field issue #11, the highest-leverage fix, option B (the maintainer's recommended design, locked in D6; see docs/field-reports.md). The decomposer always creates new cells (decomposer.py:349-410) and never checks whether an existing cell in the dish covers the sub-claim: 983 isolated cells vs the hand-crafted equivalent with 128 typed edges, zero cross-connections. The maintainer's interim patch (prompt-injected cell list -> reference nodes -> typed edges persisted at dish level) proved the concept but also its limits: in the wave that produced edges, the aim was coarse — 29 of 36 cross-colony edges attached to colony ROOT cells rather than the specific matching sub-claim — and subsequent post-patch runs produced ZERO cross-references despite overlapping claims. Prompt injection is not reliable; a first-class tool with typed reference output is the design here. Known v1 constraint: `deserialize_colony` drops edges whose target cell is outside the colony (petri/graph/colony.py:384-386), so cross-colony edges could never live in colony.json — under D4 as amended (see docs/ARCHITECTURE-V2.md) they live as rows in the dish-level `edges` table in petri.sqlite (schema issue), which the dish-scoped layout issue in this milestone gives a home. The former 'shared dish-level appends' open question is closed by construction: edges are rows, appended transactionally.

Scope:
- New `petri/reasoning/cell_search.py`: queries the dish's cells/edges tables in petri.sqlite via stdlib sqlite3 (cell id, claim_text, colony, level, status columns); lexical relevance scoring in Python (token overlap — stdlib only, no embedding deps); `search(query, top_k)` returns typed candidates. Build on `graph/colony.py`'s `find_shared_premises` — it is already implemented and unit-tested with ZERO call sites; resurrect its matching logic and port its tests. Keep the entry point behind one stable function signature: M6-storage's petri/query/ SQL views may later back the candidate fetch behind the same signature — keep the seam.
- Register `search_cells` as an `@agent.tool` (RunContext deps carry the dish DB handle/paths) on both decomposition agents, instructed to search before proposing each sub-premise and to reference the MOST SPECIFIC matching cell, not a colony root (the 29/36-roots failure above).
- Schema: sub-premise output becomes a union `PremiseCandidate | ReferencePremise` where `ReferencePremise = {existing_cell_id: str, reason: str}`. When the model returns a reference, the decomposer creates `Edge(from_cell=parent_id, to_cell=existing_cell_id, edge_type='cross_colony')` (Edge model: petri/models.py:276-281; precedent: petri/cli/connect.py:81-95) instead of a new Cell, and does NOT enqueue it for expansion.
- Validation: a `ReferencePremise` naming a nonexistent cell id raises ModelRetry with the valid-id context.
- Canonical edge store (this module is the single owner going forward): new `petri/storage/edge_registry.py`, the sole writer of the `edges` table in petri.sqlite — one typed EdgeRecord ROW per reference (from_cell, to_cell, edge_type, reason, created_at as typed columns, not JSON blobs), appended via the petri.sqlite write seam; rows are append-only through the registry API (no update/delete surface); tombstone records are reserved for later (M7-lifecycle extends this registry with DishGraph queries, cross-colony cycle detection, and tombstones; M6-storage's petri/query/ views read the table). Do NOT create a `dish_edges.py` module, a mutable `edges.json`, or an `edges.jsonl` side file — edges are rows. Include a loader/query utility so `petri graph`/`petri check` can consume edges later.
- Seed-order note in README/CLI help: foundation colonies first ('colonies seeded later benefit from richer context') — wording coordinated with the M7-lifecycle epic, which carries the same guidance.

Out of scope: dashboard rendering of cross-colony edges (field issue #15 — M7-lifecycle edge-registry extension); cross-colony cycle detection (the colony graph layer cannot see these edges yet — flagged as a known limitation in the edge_registry docstring, resolved by M7-lifecycle); retro-linking existing v1 dishes (D8: no compatibility).

Touches: petri/reasoning/decomposition_agents.py, petri/reasoning/decomposition_schemas.py, petri/reasoning/decomposer.py:349-410 (branch on reference vs new cell), petri/graph/colony.py:374-386 (read-only reference; do not change round-trip semantics here) and find_shared_premises (resurrected), petri/models.py:276-281; new petri/reasoning/cell_search.py, petri/storage/edge_registry.py, tests/unit/test_cell_search.py, tests/unit/test_edge_registry.py, tests/unit/test_reference_edges.py.

**Acceptance criteria:**
- [ ] pytest: cell_search over a fixture dish in petri.sqlite (two colonies, ~10 cell rows) returns the expected cell for an exact-phrase query and ranks token-overlap matches deterministically; find_shared_premises's ported unit tests pass against the resurrected logic
- [ ] pytest with FunctionModel scripting a tool call + ReferencePremise reply: seeding colony B against fixture colony A creates a cross_colony row in the edges table, creates no duplicate cell, and does not expand the referenced cell
- [ ] pytest: ReferencePremise with an unknown cell id triggers ModelRetry (observed via scripted second response); after retries exhaust, decomposition fails loudly
- [ ] pytest: the edges table is append-only through the registry — each reference appends exactly one row; records round-trip (insert -> SQL read -> identical EdgeRecord models); sequential seeds append without clobbering earlier rows; the registry API exposes no update/delete surface (API-surface assertion)
- [ ] grep test: no module named dish_edges, no dish-level edges.json, and no edges.jsonl side file is created anywhere
- [ ] Live smoke (manual, documented in PR): re-running the maintainer's two-colony scenario produces >=1 cross-colony reference edge row, attached to a non-root cell

---

## M3-decomposer.9 — Refuse to overwrite an existing colony in petri seed

**Size:** S · **Labels:** migration-v2, lifecycle, decomposer, good-first-issue, breaking-change · **Good first issue** · **Field issues:** #5

Context: field issue #5, CRITICAL (see docs/field-reports.md). `petri seed --colony <name>` on an existing colony silently deletes all cells and starts fresh — the maintainer lost an 88-cell, 8-level colony (~13 minutes of Opus compute) by re-seeding after a rate-limit interruption. Root cause: `petri/cli/seed.py:294-295` wipes any existing colony dir with `shutil.rmtree(colony_path)` before writing. The maintainer's validated patch: refuse when the colony has cells; no `--force` flag — manual delete required. D8 confirms this guard is in scope as a v2 safety feature.

Scope:
- Before the wipe at seed.py:293-295: if `colony_path` exists AND contains >=1 cell (count metadata.json files under it, or `deserialize_colony` cell count), print an error naming the colony, its cell count, and the manual-delete path, then exit non-zero. Do not add `--force`.
- An existing but cell-free leftover directory (0 cells) may still be cleaned and reused (preserves crash-cleanup behaviour).
- The regenerate/abort re-roll `rmtree` calls inside a single seed run (seed.py:396, 456, 469) remain — they only destroy cells created by this run; add a code comment making that invariant explicit.
- Mirror the same guard in the dashboard's `POST /api/seed` handler (petri/dashboard/api.py:422-506) so the web path cannot destroy a colony either.
- Error message should mention `petri seed --resume` once that lands (coordinate wording with the resume issue; acceptable to ship first with a plain message and update later).

Out of scope: `petri backup` (field issue #6 — the M6-storage backup issue); resume itself (field issue #8, separate issue in this milestone plus the M4-dbos re-platform).

Touches: petri/cli/seed.py:287-314 (guard before rmtree at 294-295), petri/cli/seed.py:396/456/469 (comment only), petri/dashboard/api.py:422-506; tests/unit/test_seed_guard.py (new).

**Acceptance criteria:**
- [ ] pytest: seeding onto a fixture colony containing 1+ cells exits with a non-zero code and the colony directory tree is byte-identical afterwards (hash comparison)
- [ ] pytest: seeding onto an existing colony directory with 0 cells proceeds and reuses/recreates the directory
- [ ] pytest: the in-run regenerate loop still rebuilds the colony (existing behaviour preserved)
- [ ] pytest: dashboard POST /api/seed against an existing non-empty colony returns an error status and mutates nothing
- [ ] Error message includes colony name, cell count, and the exact path the user would need to delete manually

---

## M3-decomposer.10 — Make seed resumable: checkpoint decomposition state and add petri seed --resume

**Size:** M · **Labels:** migration-v2, decomposer, durable-execution, lifecycle · **Depends on:** Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry; Add bedrock stop condition, counterargument sub-claims, and negative examples to the decomposition agents; Refuse to overwrite an existing colony in petri seed; Define the petri.sqlite schema and migration mechanism · **Field issues:** relates:#8

Context: field issue #8 (see docs/field-reports.md). An interrupted `petri seed` (rate limit, crash, Ctrl+C) leaves 'leaf cells at every level — some genuinely atomic, some never expanded' with no marker distinguishing them and no resume path (example: 84 cells, 66 leaves, 5 of 9 L1 cells unexpanded). The Five Whys loop is a serial FIFO worklist with per-cell persistence but no worklist checkpoint (decomposer.py:345-410; `on_cell_created` at 327/402 persists cells only). D2 requires killed/rate-limited seed to resume with no lost cells and no re-paid LLM calls.

Ownership note: this issue ships the INTERIM resume so seed safety does not wait for durable execution. The M4-dbos DBOS-on-SQLite validation spike gates this design's assumptions, and a dedicated M4-dbos issue ('Re-platform petri seed onto the DBOS backend', SetWorkflowID = colony id) performs the swap and fully closes field issue #8 — which is why this issue is marked relates:#8 rather than the closer.

Scope:
- Expansion-state marker: every cell created during decomposition records an expansion status — `pending` (created, not yet expanded), `expanded`, or `atomic` (with bedrock_reason from the quality-prompts issue) — persisted via the domain event log (the events table in petri.sqlite; `decomposition_started`/`decomposition_completed` events already exist in the registry; add `cell_expansion_completed`/`cell_marked_atomic` through the AgentStepData-permissive registry, petri/models.py:182-226) so the audit trail stays append-only (D4 as amended: append-only is schema-enforced).
- Re-express the worklist as a small pydantic-graph graph (nodes: ExpandNext -> ExpandNext | End) whose state is {queue of pending cell ids, cells_per_level, settings}, persisted through a `SeedCheckpointStore` after each expansion step. Default checkpoint target: a `seed_checkpoints` table in the dish's petri.sqlite (one row per colony keyed by colony id, JSON state blob, updated_at), added as this issue's own small forward-only migration (bumps `PRAGMA user_version` via the mechanism from the schema issue) — this keeps the dish single-file and the checkpoint write transactional with cell/event writes. A side-file JSON checkpoint remains an acceptable fallback behind the same seam if the pydantic-graph persistence API fits the table poorly. On interrupt, checkpoint + events table fully determine what remains.
- `petri seed --resume --colony <name>`: rebuilds graph state from the checkpoint (fall back to reconstructing the pending set from cell events in the events table if the checkpoint is missing/corrupt — the events table is the source of truth), then continues expansion. Plain `petri seed` on a partially-seeded colony keeps refusing (overwrite guard) but now names `--resume` in the message.
- M4-dbos coordination: keep the persistence driver behind the small `SeedCheckpointStore` interface so the M4-dbos re-platform issue can swap in a DBOS workflow without touching graph/node logic. Do not import dbos here. The seed_checkpoints table is interim state, not domain events — the M4-dbos swap may retire it.
- Open questions (verify against installed packages; do not invent APIs): confirm the pydantic-graph persistence idiom (`set_graph_types` + `load_next`, or a custom BaseStatePersistence) against the pydantic-graph shipped with pydantic-ai 2.9.0 before coding; if the classic BaseNode API fits poorly, the fallback is a plain JSON state blob written by the loop into seed_checkpoints, behind the same SeedCheckpointStore seam.

Out of scope: grow resume (M4-dbos); DBOS wiring (M4-dbos); verdict-driven re-decomposition (field issue #9 — M7-lifecycle); resuming v0.3.x dishes (D8: none).

Touches: petri/reasoning/decomposer.py:345-410 (loop becomes graph-driven), petri/cli/seed.py:175-487 (new --resume option, guard message), petri/models.py:199-226 (event registry additions), petri/storage/migrations (seed_checkpoints migration); new petri/reasoning/seed_graph.py, petri/reasoning/seed_checkpoint.py, tests/unit/test_seed_resume.py.

**Acceptance criteria:**
- [ ] pytest: FunctionModel scripted to raise after N expansions simulates an interrupt; `--resume` completes the colony and the total number of model calls across both runs equals the single-run count (no re-paid expansions)
- [ ] pytest: after an interrupt, every leaf cell is classifiable as pending vs atomic purely from the dish's petri.sqlite (events table + checkpoint row), asserted by a helper the CLI will reuse
- [ ] pytest: `--resume` with a deleted/corrupted checkpoint reconstructs the pending set from the events table and still completes
- [ ] pytest: the seed_checkpoints migration bumps user_version and applies cleanly to a dish created at the base schema version (forward-only migration mechanism exercised)
- [ ] pytest: `--resume` on a fully-seeded colony exits 0 with a 'nothing to resume' message and makes no model calls
- [ ] pytest: plain `petri seed` on a partially-seeded colony refuses and mentions --resume
- [ ] SeedCheckpointStore is the only module reading/writing the seed checkpoint (seed_checkpoints table or fallback file) — grep/API-surface test — preserving the M4-dbos swap seam

---

## M3-decomposer.11 — Add an edit path to decomposition approval in petri seed

**Size:** S · **Labels:** migration-v2, decomposer, cli · **Depends on:** Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry

Context: the original CLI design for seed approval specified a 'Y/n/edit' prompt, but the shipped CLI only offers whole-tree accept / regenerate / abort (petri/cli/seed.py approval loop, ~390-472). When one sub-claim of an otherwise good decomposition is wrong, the user's only options today are to accept the flaw or re-pay for a full regeneration. This capability was specified but never built; with the pydantic-ai rebuild's typed per-node results (PremiseCandidate/WhyExpansion) it becomes straightforward to implement safely.

Scope:
- Add an 'edit' choice to the seed approval prompt alongside accept/regenerate/abort.
- Editing mechanics: dump the proposed sub-claims for the pending expansion as a small YAML block and open it in `$EDITOR` (fallback: an inline numbered-list prompt supporting `edit N <new text>`, `drop N`, `done` for environments without an editor). Users can rewrite claim text, delete a proposed sub-claim, change its relation (supports/limits/rebuts), or mark it atomic.
- On save, re-validate the edited block through the SAME pydantic schemas the agents use (PremiseCandidate/WhyExpansion); invalid edits re-prompt with the validation error rather than crashing or silently accepting.
- Audit trail: edited nodes are recorded with a `decomposition_edited` event (original text, edited text) through the event registry, appended to the domain event log (events table in petri.sqlite), so the append-only history (D4 as amended) shows human intervention.
- Edited results flow into cell creation exactly like model output — caps, near-duplication guard, and persistence behave identically.

Out of scope: editing cells that are already persisted (that is re-decomposition territory, M7-lifecycle); dashboard-side editing (the dashboard seed path stays accept/reject for now); free-form graph surgery (adding edges — `petri connect` already exists for that).

Touches: petri/cli/seed.py approval loop (~390-472); petri/reasoning/decomposition_schemas.py (re-validation entry point); petri/models.py event registry (decomposition_edited); tests/unit/test_seed_edit_approval.py (new).

**Acceptance criteria:**
- [ ] pytest: with a scripted editor function, editing one proposed sub-claim's text results in a created cell carrying the edited text, and the decomposition_edited event records original and edited text
- [ ] pytest: `drop N` removes exactly that sub-claim; remaining sub-claims are created unchanged
- [ ] pytest: an edit producing schema-invalid YAML (e.g. unknown relation value) re-prompts with the validation error and does not create cells
- [ ] pytest: edited output still passes through per-parent cap truncation and the near-duplication guard (an edit that duplicates the parent is rejected the same way model output is)
- [ ] Manual smoke (documented in PR): `petri seed` interactive session shows the edit option, $EDITOR round-trip works, and accept-without-edit behaves exactly as before

---

## M3-decomposer.12 — Add a pydantic_evals regression suite for decomposition quality (organic counts, bedrock stops, counterarguments)

**Size:** M · **Labels:** migration-v2, decomposer, agents, docs · **Depends on:** Rebuild the decomposer as pydantic-ai Agents with typed outputs and automatic retry; Add bedrock stop condition, counterargument sub-claims, and negative examples to the decomposition agents; Stop the decomposer treating layer limits as quotas: per-parent caps in code, no counts in prompts; Add search_cells tool so decomposition creates cross-colony reference edges instead of duplicate cells · **Field issues:** relates:#3, relates:#10, relates:#11, relates:#12, relates:#13

Context: the maintainer ran four prompt iterations on the same root claim; decomposition quality is currently eyeballed. D6 mandates a pydantic_evals regression suite so it is measured. The experiment record (also indexed in docs/field-reports.md) is embedded here so this issue is self-contained:

| Version | Count phrasing | Outcome |
|---|---|---|
| v1 | 'Pick the TOP N' (quota) | Uniform quota-fill: 126 cells across 11 levels, every level filled to the cap |
| v2 | Ceiling phrasing ('at most N') | Still uniform quota-fill at L2+ |
| v3 | Inverse bias ('usually far fewer') | Over-corrected into binary-split tunneling: 79 cells of mostly 2-child expansions |
| v4 | Neutral ('let structure determine count') | Best: organic counts (e.g. L2 count of 4), no quota signature |

pydantic_evals provides Dataset/Case/custom Evaluator subclasses (evaluate over a task fn) plus LLMJudge.

Scope:
- New `evals/decomposer/` package: `dataset.py` (Dataset with Cases), `evaluators.py` (custom Evaluator subclasses), `claims.yaml` (fixture claims: the business-thesis-style root claim family from v1-v4, an axiom-heavy claim, a counterargument-rich contested claim, and a two-colony dish fixture for reference-rate).
- Eval corpus fixtures, committed under `evals/decomposer/fixtures/` (data supplied from the maintainer's field archive, summarized in docs/field-reports.md):
  (a) `claims-graph.json` — the hand-crafted ground-truth graph: 88 nodes / 128 edges, verified acyclic, 60 cross-layer edges — powers a search_cells precision/recall Case (does the tool find the cells a human linked?);
  (b) the archive's organic layer-size distribution (per-layer counts ranged 4-13) — the anti-quota target distribution for count evaluators;
  (c) paired same-claim decompositions across archive waves (~7 colonies) — A/B comparison Cases;
  (d) the principle-vs-measurement a/b splits (claims 08-04a/b, 08-07a/b, 08-09a/b) — split-judgment Cases asserting the decomposer separates a principle claim from its measurement claim instead of fusing them;
  (e) negative Cases the suite must score POORLY: the hotdog-sandwich template output (5 generic subclaims for a definitional claim — template fill, not analysis) and the v0.1.0 `_default_decompose` hardcoded template children.
- Custom evaluators over the DecompositionResult produced by the task fn: (1) NoQuotaFill — fails if >=2 consecutive levels have cell counts exactly equal to the level ceiling (the v1/v2 signature); (2) OrganicVariation — per-level counts are not all identical, scored against the archive's 4-13 organic distribution, AND L1 counts vary across fixture claims of differing complexity (guards the root-cap saturation seen in post-patch field data: 5 of 12 colonies at exactly 8 L1 cells); (3) CounterargumentPresence — fraction of expanded parents with >=1 child whose relation is 'limits' or 'rebuts' above threshold; (4) BedrockStop — axiom fixture terminates with is_atomic within depth 2 and trivial-cell rate < 25% (the v1 failure rate); (5) CrossColonyReferenceRate — a second colony seeded against the fixture dish MUST yield >=1 reference edge (post-patch field runs produced zero cross-references despite overlapping claims), and edges attach to the specific matching cell, not the colony root (29 of 36 interim-patch edges hit roots); (6) DepthOrganic — depth varies across claims rather than always hitting max_decomposition_depth; (7) ParentChildDistinctness — no child near-duplicates its parent (the restatement failure: healthcare-customer 002-001 vs 001-001; swarm-build 002-001), mirroring the decomposer's ModelRetry guard.
- Two run modes: (a) offline/CI — evaluators unit-tested against synthetic DecompositionResults reconstructed from the documented v1 (uniform 13/level) and v4 (organic) shapes above, plus FunctionModel-scripted end-to-end runs: deterministic, zero cost, runs in `uv run pytest tests/evals/`; (b) live — gated behind `PETRI_LIVE_EVALS=1` env var + pytest marker, runs the real agents and prints the Dataset report; README section documents approximate token cost before anyone runs it (cost-warning ethos).
- Optional/stretch (separate commit, may be dropped): LLMJudge case scoring claim falsifiability against the maintainer's quality rubric (quantified thresholds, explicit exclusions) on a cheap model — mark clearly as advisory, not gating.
- CI wiring: offline suite added to the default test run; live suite never runs in CI.

Out of scope: evals for grow-phase agent verdicts (M2-agents); Logfire/OTel upload of eval results (M5-otel, D7).

Touches: reads petri/reasoning/decomposition_agents.py, decomposition_schemas.py, cell_search.py, decomposer.py (task fn wiring only — no production changes); new evals/decomposer/{__init__.py,dataset.py,evaluators.py,claims.yaml,fixtures/}, tests/evals/test_decomposer_evals.py, README eval section.

**Acceptance criteria:**
- [ ] pytest offline: NoQuotaFill evaluator fails a synthetic v1-shaped result (every level exactly at ceiling) and passes a synthetic v4-shaped organic result
- [ ] pytest offline: OrganicVariation fails a synthetic result where every fixture claim yields the same L1 count (root-cap saturation signature) and passes when L1 varies with claim complexity
- [ ] pytest offline: ParentChildDistinctness fails a synthetic result containing a restatement child and passes a distinct-children result
- [ ] pytest offline: the negative fixtures (hotdog-sandwich template output, v0.1.0 _default_decompose children) score poorly on the relevant evaluators (asserted thresholds)
- [ ] pytest offline: the search_cells precision/recall Case runs against the committed claims-graph.json fixture (88 nodes / 128 edges) and reports precision/recall numbers
- [ ] pytest offline: all evaluators run green over FunctionModel-scripted end-to-end decompositions with zero network/subprocess calls (asserted)
- [ ] pytest offline suite is part of `uv run pytest tests/` and adds < 30s runtime
- [ ] Live mode: `PETRI_LIVE_EVALS=1 uv run pytest tests/evals/ -m live` executes the Dataset against the configured model — including the two-colony Case where the second colony MUST produce >=1 cross-colony reference edge — and prints a per-case, per-evaluator report (manually verified once, output pasted in PR)
- [ ] README documents live-eval cost estimate and the offline-by-default policy
- [ ] Deliberate-regression check: a test temporarily reinstating quota phrasing ('Pick the TOP N' with interpolated N) in instructions makes the instruction-content assertions and NoQuotaFill (on the corresponding synthetic shape) fail — demonstrating the suite guards the v1 failure mode

---
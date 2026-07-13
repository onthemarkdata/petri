# Field Reports — Dogfood Findings Index

Petri v0.3.4 was dogfooded on a real research project: 12 root claims seeded as colonies
(148–983 cells depending on patch level) against a hand-crafted baseline of 101 claims connected
by 128 dependency edges. The maintainer logged every limitation as a numbered field issue and
validated fixes as local patches before this plan. v2 GitHub issues cite these numbers
(`#N` = the issue that closes it; `relates:#N` = related work).

Numbers are the original dogfood-repo issue numbers, kept for traceability.

| # | Title | Category | Status after dogfood |
|---|-------|----------|----------------------|
| 2 | Decomposer ignores dish config | Config | Patched locally, validated |
| 3 | Uniform cell count — budget treated as quota | Decomposition | Patched locally; L1 saturation persists |
| 5 | `seed` silently destroys existing colony | Safety (CRITICAL) | Patched locally (refuse if cells exist) |
| 6 | No backup/snapshot workflow | Safety | Open |
| 8 | Cannot resume interrupted decomposition | Lifecycle | Open |
| 9 | No verdict-driven re-decomposition | Lifecycle | Open (needs grow experience) |
| 10 | No triviality/bedrock stop condition | Decomposition | Patched locally (prompt-level only) |
| 11 | No cross-colony cell search during decomposition | Cross-colony (highest leverage) | Patched interim (prompt injection) |
| 12 | No negative examples in decompose prompt | Decomposition | Patched locally |
| 13 | No counterargument sub-claims | Decomposition | Patched locally, validated |
| 14 | Directory structure: dish ID missing as parent dir | Structure | Patched locally |
| 15 | Dashboard bugs + missing drill-down navigation | Dashboard | Patched locally |

(#4 closed — `grow --colony` already existed; #7 split into #8/#9.)

## Summaries

### #2 — Decomposer ignores dish config
`load_config()` is LRU-cached and module-level constants freeze packaged defaults at import time,
so dish `petri.yaml` values (`max_nodes_per_layer`, `max_decomposition_depth`) never reach the
decomposer. Observed: dish config of 13 nodes/depth 10 produced identical 16-cell depth-3 trees
(the packaged 5/3 defaults) across 10 colonies, all of which had to be deleted.

### #3 — Budget treated as quota
The prompt passed `remaining_budget` as `max_premises`; the model fills any number it sees.
Four prompt experiments (v1–v4) on one root claim: "Pick TOP N" → uniform fill; ceiling phrasing →
uniform below L1; inverse bias ("usually much less") → over-corrected into binary tunneling;
neutral "let structure determine count" → best. Post-patch on-disk data shows the residual defect
is at the ROOT expansion: 5 of 12 colonies produced exactly cap-many (8) level-1 children while
deeper levels vary organically. Fix must be code-level per-parent caps with no counts visible to
the model, covering L1 explicitly.

### #5 — Seed destroys existing colony (CRITICAL)
`petri seed --colony <existing>` recreated the directory (rm + mkdir) with no warning. Real loss:
an 88-cell, 8-level colony (~13 minutes of Opus compute) destroyed while retrying after a
rate-limit interruption. Guard patch refuses when cells exist — but the guard without resume (#8)
turned recoverable failures into permanently partial colonies (31 futile guard-blocked retries
observed in wrapper logs). #5 and #8 must ship together.

### #6 — Backup workflow
No snapshot mechanism for colony data representing hours of paid compute. Proposal: `petri backup`
(list/restore) + automatic pre-destructive snapshots to `.petri/backups/<colony>-<timestamp>/`.

### #8 — Resume interrupted decomposition
An interrupted seed leaves leaves at every level — some genuinely atomic, some never expanded —
with no marker distinguishing them and no resume path. Real case: 84 cells across 8 levels, 66
leaves, 5 of 9 L1 cells unexpanded.

### #9 — Verdict-driven re-decomposition
The pipeline is strictly decompose-then-validate. Verdicts that signal "claim too broad"
(`EVIDENCE_INCONCLUSIVE`, `DEFENSIBLE_WITH_CAVEATS`, `CRITICAL_FLAW_FOUND`, `PIPELINE_STALLED`)
have no feedback channel into further decomposition.

### #10 — Triviality/bedrock stop
~25% of cells (243/983) were axioms no one disputes ("OpenAI exists as a legal entity"); each
would trigger the full 13-agent pipeline (~3,159 wasted agent calls projected). The patched
"bedrock test" lives only in prompts: no `is_atomic` value is persisted anywhere in real cell
metadata, so the stop condition is unauditable and unenforceable.

### #11 — Cross-colony cell search (highest leverage)
The decomposer never checks whether an existing cell already covers a sub-claim: 983 cells with
zero cross-colony edges vs the baseline's 101 claims with 128 edges (9.7× inflation). Real cells
reference sibling colonies in prose ("…the use-case concentration from the healthcare-customer
colony") because no edge mechanism existed. The interim patch (inject existing cells into prompts;
model emits reference nodes) produced 36 typed edges but aimed coarse — 29 of 36 attached to
colony roots rather than specific sub-claims. Target design is a `search_cells` tool the
decomposition agent calls (option B in the original analysis).

### #12 — Negative examples
Without bad-claim examples the model produced "safe" claims — logical prerequisites and vague
observations rather than falsifiable assertions with quantified thresholds and explicit exclusions.

### #13 — Counterargument sub-claims
`decompose_why` produced only supporting children; adversarial pressure was deferred to the
red-team stage, after tree structure was fixed. Post-patch data shows counterarguments appearing
as first-class sibling claims (e.g. 2 of 8 L1 cells in one colony are pure counterarguments) —
the patch demonstrably works; v2 types the relation (supports/limits/rebuts) instead of leaving it
implicit in prose.

### #14 — Dish-scoped directory layout
Colony paths were flat (`petri-dishes/<colony>/`); the patch nests them under the dish id
(`petri-dishes/<dish_id>/<colony>/`) so the filesystem is self-documenting. Touched 11+ files —
in v2 this lands early, once, before anything builds on paths.

### #15 — Dashboard bugs
Dashboard showed 0 cells (dish id derived from the wrong source), `launch` auto-repair created
stale dirs, SSE handler called a removed function, logs rendered oldest-first. Patched alongside a
frontend overhaul (dish-wide DAG, sidebar, cross-colony edge rendering).

## Processing/operations findings (unnumbered)

- **No built-in rate-limit handling**: the CLI dies with exit 1, empty stderr, and the reset time
  on *stdout* ("resets 9pm (America/Los_Angeles)"); v0.3.4 classified it as a generic transient
  failure and burned 3 retries in ~5 seconds against a limit hours away. An external 418-line bash
  wrapper (reset-message parsing + 30s buffer + 2m→1h exponential backoff + 30-pass grow loops)
  stood in for engine-level handling.
- **Double-paid work under retries**: prototype logs show one full 7-source research pass recorded
  ~2.5× and one entire iteration duplicated 3 minutes apart — the exactly-once motivation.
- **Self-reported cost tracking does not happen**: `token_usage` was fully specified (schema, API,
  dashboard) and appears once in ~2,700 production events. Cost capture must be automatic at the
  harness/model layer.
- **Typed vocabularies hold; payload shapes drift**: 370+ verdict events with zero vocabulary
  drift, but `blocking_verdicts` flips list↔dict within a single file and casing drifts — typed
  event payloads must land before analytical readers.
- **Silent-fallback verdicts**: a v0.2.x bug made unparseable model output silently become the
  first valid verdict ("silent PASS") — convergence could be satisfied by a parser fallback.
  v2 policy: fail loud; `EXECUTION_ERROR` is a first-class domain event that never passes
  convergence.
- **grow-phase data scarcity**: no dish ever completed a real grow run (`queue.json` empty); all
  validation-pipeline evidence comes from the prototype's logs. v2 produces a committed golden
  end-to-end grow fixture as part of the durable-execution spike.

# Petri v2 integration

Scope frozen on 2026-11-01: 89 open GitHub issues in onthemarkdata/petri.
Newly discovered work is recorded separately unless required to satisfy this scope.

## Delivery contract

- Integration branch: `codex/petri-v2-integration`, based on `dev`.
- Issue-sized PRs integrate after verification. The maintainer approves the final PR into `dev`.
- Reuse suitable existing PR changes, preserving authorship; leave their original PRs open.
- Preserve the public CLI and local-first runtime. Record evidence for design adjustments.
- Live inference uses Google `gemma-4-E2B-it-qat-q4_0-gguf` with Docker Model Runner,
  through both pi and the direct model API. Docker is an optional validation dependency.
- Claude Code receives offline adapter tests; no paid live inference is authorized.
- Require repeatable seed/grow/feed, dashboard, export, backup/restore, and crash-recovery runs.
- Use public cases with known outcomes and live web retrieval. Save model identity, commands,
  outputs, timings, and failure diagnostics; do not confuse model smoke tests with app validation.
- Independent work runs in separate worktrees, with review and tests before integration.
- Private configuration, credentials, local databases, model weights, and account details stay out of Git.
- A feature-branch merge is not a release. Issues remain open until their completion evidence and
  final integration are verified; no automatic release or merge into `main`.

## Frozen issue list

| Issue | Work | Status / evidence |
|---|---|---|
| [#9](https://github.com/onthemarkdata/petri/issues/9) | [EPIC] M1-harness — Harness & inference layer | In progress; see child PRs and validation notes |
| [#10](https://github.com/onthemarkdata/petri/issues/10) | [EPIC] M2-agents — Agents on Pydantic AI | In progress; see child PRs and validation notes |
| [#11](https://github.com/onthemarkdata/petri/issues/11) | [EPIC] M3-decomposer — Agentic decomposer + evals | In progress; see child PRs and validation notes |
| [#12](https://github.com/onthemarkdata/petri/issues/12) | [EPIC] M4-dbos — Durable execution | In progress; see child PRs and validation notes |
| [#13](https://github.com/onthemarkdata/petri/issues/13) | [EPIC] M5-otel — OpenTelemetry | In progress; see child PRs and validation notes |
| [#14](https://github.com/onthemarkdata/petri/issues/14) | [EPIC] M6-storage — Storage & dashboard reads | In progress; see child PRs and validation notes |
| [#15](https://github.com/onthemarkdata/petri/issues/15) | [EPIC] M7-lifecycle — Iterative lifecycle | Pending |
| [#18](https://github.com/onthemarkdata/petri/issues/18) | Create petri/harness error taxonomy and retry core | Integrated in [PR #127](https://github.com/onthemarkdata/petri/pull/127); final v2 acceptance pending |
| [#19](https://github.com/onthemarkdata/petri/issues/19) | Add TestModel and FunctionModel offline fixtures | Integrated in [PR #128](https://github.com/onthemarkdata/petri/pull/128); final v2 acceptance pending |
| [#20](https://github.com/onthemarkdata/petri/issues/20) | Add fake_pi stub and the tests/README testing tiers | Integrated in [PR #129](https://github.com/onthemarkdata/petri/pull/129); final v2 acceptance pending |
| [#21](https://github.com/onthemarkdata/petri/issues/21) | Implement the pi RPC transport | Integrated in [PR #135](https://github.com/onthemarkdata/petri/pull/135); final v2 acceptance pending |
| [#22](https://github.com/onthemarkdata/petri/issues/22) | Implement PiModel over the pi RPC transport | Integrated in [PR #136](https://github.com/onthemarkdata/petri/pull/136); final v2 acceptance pending |
| [#23](https://github.com/onthemarkdata/petri/issues/23) | Port the Claude Code CLI to a ClaudeCodeModel | Integrated in [PR #146](https://github.com/onthemarkdata/petri/pull/146); final v2 acceptance pending |
| [#24](https://github.com/onthemarkdata/petri/issues/24) | Extract prompt builders into reasoning/prompts.py | Integrated in [PR #134](https://github.com/onthemarkdata/petri/pull/134); final v2 acceptance pending |
| [#25](https://github.com/onthemarkdata/petri/issues/25) | Resolve harness backends from petri.yaml | Integrated in [PR #147](https://github.com/onthemarkdata/petri/pull/147); final v2 acceptance pending |
| [#26](https://github.com/onthemarkdata/petri/issues/26) | Bridge decomposition calls to pydantic-ai agents | Integrated in [PR #148](https://github.com/onthemarkdata/petri/pull/148); final v2 acceptance pending |
| [#27](https://github.com/onthemarkdata/petri/issues/27) | Wire the bridge into the CLI with verdict discipline | Integrated in [PR #157](https://github.com/onthemarkdata/petri/pull/157); final v2 acceptance pending |
| [#28](https://github.com/onthemarkdata/petri/issues/28) | Define the typed agent contract and frozen roster | Integrated in [PR #132](https://github.com/onthemarkdata/petri/pull/132); final v2 acceptance pending |
| [#29](https://github.com/onthemarkdata/petri/issues/29) | Add conformance tests pinning verdict enums to YAML | Integrated in [PR #141](https://github.com/onthemarkdata/petri/pull/141); final v2 acceptance pending |
| [#30](https://github.com/onthemarkdata/petri/issues/30) | Ship the v2 petri.yaml agent-content schema | Integrated in [PR #133](https://github.com/onthemarkdata/petri/pull/133); final v2 acceptance pending |
| [#31](https://github.com/onthemarkdata/petri/issues/31) | Build the 13-agent roster via a pydantic-ai factory | Integrated in [PR #142](https://github.com/onthemarkdata/petri/pull/142); final v2 acceptance pending |
| [#32](https://github.com/onthemarkdata/petri/issues/32) | Implement debates as real agent hand-off with cost caps | Integrated in [PR #145](https://github.com/onthemarkdata/petri/pull/145); final v2 acceptance pending |
| [#33](https://github.com/onthemarkdata/petri/issues/33) | Move source-hierarchy policy to an output validator | Integrated in [PR #143](https://github.com/onthemarkdata/petri/pull/143); final v2 acceptance pending |
| [#34](https://github.com/onthemarkdata/petri/issues/34) | Port mechanical convergence to the typed contract | Integrated in [PR #144](https://github.com/onthemarkdata/petri/pull/144); final v2 acceptance pending |
| [#35](https://github.com/onthemarkdata/petri/issues/35) | Add petri agents list and check commands | Integrated in [PR #149](https://github.com/onthemarkdata/petri/pull/149); final v2 acceptance pending |
| [#36](https://github.com/onthemarkdata/petri/issues/36) | Add petri agents run with usage and cost caps | Integrated in [PR #150](https://github.com/onthemarkdata/petri/pull/150); final v2 acceptance pending |
| [#37](https://github.com/onthemarkdata/petri/issues/37) | Thread dish config through the decomposition path | Integrated in [PR #137](https://github.com/onthemarkdata/petri/pull/137); final v2 acceptance pending |
| [#38](https://github.com/onthemarkdata/petri/issues/38) | Consolidate dish-id resolution and cell-ID parsing | Integrated in [PR #130](https://github.com/onthemarkdata/petri/pull/130); final v2 acceptance pending |
| [#39](https://github.com/onthemarkdata/petri/issues/39) | Define the petri.sqlite schema and migrations | Integrated in [PR #139](https://github.com/onthemarkdata/petri/pull/139); final v2 acceptance pending |
| [#40](https://github.com/onthemarkdata/petri/issues/40) | Adopt dish-scoped layout with petri.sqlite store | Integrated in [PR #158](https://github.com/onthemarkdata/petri/pull/158); final v2 acceptance pending |
| [#41](https://github.com/onthemarkdata/petri/issues/41) | Rebuild the decomposer on pydantic-ai agents | Integrated in [PR #159](https://github.com/onthemarkdata/petri/pull/159); final v2 acceptance pending |
| [#42](https://github.com/onthemarkdata/petri/issues/42) | Add bedrock stops and counterargument sub-claims | Integrated in [PR #168](https://github.com/onthemarkdata/petri/pull/168); final v2 acceptance pending |
| [#43](https://github.com/onthemarkdata/petri/issues/43) | Enforce per-parent caps in decomposer code | Integrated in [PR #169](https://github.com/onthemarkdata/petri/pull/169); final v2 acceptance pending |
| [#44](https://github.com/onthemarkdata/petri/issues/44) | Add search_cells tool for cross-colony edges | Integrated in [PR #178](https://github.com/onthemarkdata/petri/pull/178); final v2 acceptance pending |
| [#45](https://github.com/onthemarkdata/petri/issues/45) | Refuse to overwrite an existing colony in seed | Integrated in [PR #126](https://github.com/onthemarkdata/petri/pull/126); final v2 acceptance pending |
| [#46](https://github.com/onthemarkdata/petri/issues/46) | Checkpoint seed state and add petri seed --resume | Integrated in [PR #174](https://github.com/onthemarkdata/petri/pull/174); final v2 acceptance pending |
| [#47](https://github.com/onthemarkdata/petri/issues/47) | Add an edit option to seed approval | Integrated in [PR #207](https://github.com/onthemarkdata/petri/pull/207); final v2 acceptance pending |
| [#48](https://github.com/onthemarkdata/petri/issues/48) | Add pydantic_evals suite for decomposition quality | Integrated in [PR #190](https://github.com/onthemarkdata/petri/pull/190); final v2 acceptance pending |
| [#49](https://github.com/onthemarkdata/petri/issues/49) | Spike: validate DBOS-on-SQLite for durable execution | Integrated in [PR #138](https://github.com/onthemarkdata/petri/pull/138); final v2 acceptance pending |
| [#50](https://github.com/onthemarkdata/petri/issues/50) | Add an ExecutionBackend seam in front of the v1 engine | Integrated in [PR #151](https://github.com/onthemarkdata/petri/pull/151); final v2 acceptance pending |
| [#51](https://github.com/onthemarkdata/petri/issues/51) | Make append_event idempotent with deterministic IDs | Integrated in [PR #140](https://github.com/onthemarkdata/petri/pull/140); final v2 acceptance pending |
| [#52](https://github.com/onthemarkdata/petri/issues/52) | Model the per-cell pipeline as a typed pydantic-graph | Integrated in [PR #152](https://github.com/onthemarkdata/petri/pull/152); final v2 acceptance pending |
| [#53](https://github.com/onthemarkdata/petri/issues/53) | Document the queue-state to pipeline-node mapping | Integrated in [PR #153](https://github.com/onthemarkdata/petri/pull/153); final v2 acceptance pending |
| [#54](https://github.com/onthemarkdata/petri/issues/54) | Port queue transition tests to graph conformance suite | Integrated in [PR #154](https://github.com/onthemarkdata/petri/pull/154); final v2 acceptance pending |
| [#55](https://github.com/onthemarkdata/petri/issues/55) | Port research-side phases into pipeline nodes | Integrated in [PR #160](https://github.com/onthemarkdata/petri/pull/160); final v2 acceptance pending |
| [#56](https://github.com/onthemarkdata/petri/issues/56) | Port decision-side phases and iterate loop into nodes | Integrated in [PR #161](https://github.com/onthemarkdata/petri/pull/161); final v2 acceptance pending |
| [#57](https://github.com/onthemarkdata/petri/issues/57) | Implement DBOSExecutionBackend with durable cell workflows | Integrated in [PR #175](https://github.com/onthemarkdata/petri/pull/175); final v2 acceptance pending |
| [#58](https://github.com/onthemarkdata/petri/issues/58) | Add grow_colony workflow for bottom-up scheduling | Integrated in [PR #176](https://github.com/onthemarkdata/petri/pull/176); final v2 acceptance pending |
| [#59](https://github.com/onthemarkdata/petri/issues/59) | Re-platform petri seed onto the DBOS backend | Integrated in [PR #196](https://github.com/onthemarkdata/petri/pull/196); final v2 acceptance pending |
| [#60](https://github.com/onthemarkdata/petri/issues/60) | Port sync-check as a startup reconciliation step | Integrated in [PR #187](https://github.com/onthemarkdata/petri/pull/187); final v2 acceptance pending |
| [#61](https://github.com/onthemarkdata/petri/issues/61) | Repoint dashboard /api/queue onto the backend seam | Integrated in [PR #202](https://github.com/onthemarkdata/petri/pull/202); final v2 acceptance pending |
| [#62](https://github.com/onthemarkdata/petri/issues/62) | Rewire grow/stop/feed onto the durable backend | Integrated in [PR #206](https://github.com/onthemarkdata/petri/pull/206); final v2 acceptance pending |
| [#63](https://github.com/onthemarkdata/petri/issues/63) | Retire the v1 engine and fcntl queue | Integrated in [PR #218](https://github.com/onthemarkdata/petri/pull/218); final v2 acceptance pending |
| [#64](https://github.com/onthemarkdata/petri/issues/64) | Add OTel bootstrap and petri.yaml telemetry section | Integrated in [PR #162](https://github.com/onthemarkdata/petri/pull/162); final v2 acceptance pending |
| [#65](https://github.com/onthemarkdata/petri/issues/65) | Spike: choose the pydantic-ai OTel instrumentation path | Integrated in [PR #163](https://github.com/onthemarkdata/petri/pull/163); final v2 acceptance pending |
| [#66](https://github.com/onthemarkdata/petri/issues/66) | Enable DBOS OTel spans behind the durability seam | Pending |
| [#67](https://github.com/onthemarkdata/petri/issues/67) | Instrument pydantic-ai agent runs with OTel | Integrated in [PR #164](https://github.com/onthemarkdata/petri/pull/164); final v2 acceptance pending |
| [#68](https://github.com/onthemarkdata/petri/issues/68) | Attach petri.* domain attributes to spans | Integrated in [PR #165](https://github.com/onthemarkdata/petri/pull/165); final v2 acceptance pending |
| [#69](https://github.com/onthemarkdata/petri/issues/69) | Persist spans and usage rows to petri.sqlite | Integrated in [PR #167](https://github.com/onthemarkdata/petri/pull/167); final v2 acceptance pending |
| [#70](https://github.com/onthemarkdata/petri/issues/70) | Add read-only trace API endpoints to the dashboard | Integrated in [PR #170](https://github.com/onthemarkdata/petri/pull/170); final v2 acceptance pending |
| [#71](https://github.com/onthemarkdata/petri/issues/71) | Add a per-cell trace waterfall to the dashboard | Integrated in [PR #171](https://github.com/onthemarkdata/petri/pull/171); final v2 acceptance pending |
| [#72](https://github.com/onthemarkdata/petri/issues/72) | Surface per-cell token and cost totals in the dashboard | Integrated in [PR #172](https://github.com/onthemarkdata/petri/pull/172); final v2 acceptance pending |
| [#73](https://github.com/onthemarkdata/petri/issues/73) | Document observability: local default and OTLP export | Integrated in [PR #179](https://github.com/onthemarkdata/petri/pull/179); final v2 acceptance pending |
| [#74](https://github.com/onthemarkdata/petri/issues/74) | Add a petri/query read layer over petri.sqlite | Integrated in [PR #166](https://github.com/onthemarkdata/petri/pull/166); final v2 acceptance pending |
| [#75](https://github.com/onthemarkdata/petri/issues/75) | Move REST endpoints to petri/query and add /api/edges | Integrated in [PR #180](https://github.com/onthemarkdata/petri/pull/180); final v2 acceptance pending |
| [#76](https://github.com/onthemarkdata/petri/issues/76) | Rebuild /api/stream on events-table tailing | Integrated in [PR #183](https://github.com/onthemarkdata/petri/pull/183); final v2 acceptance pending |
| [#77](https://github.com/onthemarkdata/petri/issues/77) | Retire migrate.py and the disposable dashboard index | Integrated in [PR #184](https://github.com/onthemarkdata/petri/pull/184); final v2 acceptance pending |
| [#78](https://github.com/onthemarkdata/petri/issues/78) | Add petri backup with --list and a snapshot hook | Integrated in [PR #177](https://github.com/onthemarkdata/petri/pull/177); final v2 acceptance pending |
| [#79](https://github.com/onthemarkdata/petri/issues/79) | Add petri backup --restore with an overwrite guard | Integrated in [PR #181](https://github.com/onthemarkdata/petri/pull/181); final v2 acceptance pending |
| [#80](https://github.com/onthemarkdata/petri/issues/80) | Add petri export for git-reviewable derived text | Integrated in [PR #182](https://github.com/onthemarkdata/petri/pull/182); final v2 acceptance pending |
| [#81](https://github.com/onthemarkdata/petri/issues/81) | Replace /api/proc and sync seed with durable runs | Pending |
| [#82](https://github.com/onthemarkdata/petri/issues/82) | Harden colony graph integrity and persistence | Integrated in [PR #192](https://github.com/onthemarkdata/petri/pull/192); final v2 acceptance pending |
| [#83](https://github.com/onthemarkdata/petri/issues/83) | Ship starter analytics views in petri.sqlite | Integrated in [PR #188](https://github.com/onthemarkdata/petri/pull/188); final v2 acceptance pending |
| [#84](https://github.com/onthemarkdata/petri/issues/84) | Write the verdict-driven re-decomposition RFC | Integrated in [PR #185](https://github.com/onthemarkdata/petri/pull/185); final v2 acceptance pending |
| [#85](https://github.com/onthemarkdata/petri/issues/85) | Detect re-decomposition triggers from verdict history | Integrated in [PR #186](https://github.com/onthemarkdata/petri/pull/186); final v2 acceptance pending |
| [#86](https://github.com/onthemarkdata/petri/issues/86) | Build the re-decomposition proposal workflow | Integrated in [PR #199](https://github.com/onthemarkdata/petri/pull/199); final v2 acceptance pending |
| [#87](https://github.com/onthemarkdata/petri/issues/87) | Add the re-decomposition approval gate to petri grow | Integrated in [PR #208](https://github.com/onthemarkdata/petri/pull/208); final v2 acceptance pending |
| [#88](https://github.com/onthemarkdata/petri/issues/88) | Re-validate fed evidence by forking workflows | Pending |
| [#89](https://github.com/onthemarkdata/petri/issues/89) | Re-scope the contradiction scanner and schedule it | Integrated in [PR #189](https://github.com/onthemarkdata/petri/pull/189); final v2 acceptance pending |
| [#90](https://github.com/onthemarkdata/petri/issues/90) | Extend the edge registry with dish-level graph queries | Integrated in [PR #191](https://github.com/onthemarkdata/petri/pull/191); final v2 acceptance pending |
| [#91](https://github.com/onthemarkdata/petri/issues/91) | Compute edge-intelligence metrics from the dish graph | Integrated in [PR #193](https://github.com/onthemarkdata/petri/pull/193); final v2 acceptance pending |
| [#92](https://github.com/onthemarkdata/petri/issues/92) | Surface edge intelligence in CLI and dashboard | Integrated in [PR #197](https://github.com/onthemarkdata/petri/pull/197); final v2 acceptance pending |
| [#93](https://github.com/onthemarkdata/petri/issues/93) | Add opt-in convergence-point priority to grow scheduling | Integrated in [PR #205](https://github.com/onthemarkdata/petri/pull/205); final v2 acceptance pending |
| [#94](https://github.com/onthemarkdata/petri/issues/94) | Add a read-only Analyst for research-health flags | Integrated in [PR #204](https://github.com/onthemarkdata/petri/pull/204); final v2 acceptance pending |
| [#95](https://github.com/onthemarkdata/petri/issues/95) | Write the GitHub feedback re-entry RFC | Integrated in [PR #209](https://github.com/onthemarkdata/petri/pull/209); final v2 acceptance pending |
| [#96](https://github.com/onthemarkdata/petri/issues/96) | Archive merged worktrees and record corpus provenance | Pending |
| [#100](https://github.com/onthemarkdata/petri/issues/100) | Petri v2 migration — master tracker | In progress; see child PRs and validation notes |
| [#117](https://github.com/onthemarkdata/petri/issues/117) | Wire `petri --version` into the CLI | Already shipped in PR #118; version CLI regression tests pass |
| [#120](https://github.com/onthemarkdata/petri/issues/120) | Wire up the pi harness end-to-end | In progress; see child PRs and validation notes |


## Current validation and remaining integration work

- The combined stack through PR #155 passed 1,150 offline tests, mypy across
  80 modules, Ruff 0.15.21, and whitespace checks. The preceding coverage run
  measured 80.31%, above the 68% requirement.
- PR #156 fixes concurrent SQLite WAL startup. A 3,200-opener stress run had no
  lock failures after a failure was reproduced before the fix.
- PR #157 passed local seed and grow dry-run through both pi and the direct
  Docker Model Runner Chat Completions API with the exact Gemma Q4_0 model.
  Full grow, feed, dashboard, export, backup/restore, and crash recovery remain
  required before the final integration PR can be approved.
- The proposed CI branch filter change remains local on `codex/petri-v2-ci`:
  GitHub rejected its push because the active OAuth credential lacks `workflow`
  scope. Local checks continue; no credential settings were changed.

- The combined stack through PR #184 passed 1,332 tests, with one skip and six
  live-test deselections. Mypy checked 115 modules; Ruff 0.15.21 passed.
- Real SQLite-backed DBOS tests exercised interrupted work and reconciliation.
  The final full application crash-recovery run remains pending.
- Live Gemma decomposition still fails acceptance: atomic stopping is unreliable,
  and reference-aware output exhausts validation retries. These failures block
  final integration approval; deterministic fixtures do not replace the live gate.
- Historical research fixtures remain private by the maintainer's explicit choice.
  Public tests use synthetic cases; private archive checks require a local opt-in.
  No private research text, source paths, or review manifest is part of the public stack.

## Integration checks through PR #203

The combined offline suite passed 1,390 tests, with two skips and 16 integration
tests deselected. Five selected real process recovery tests passed: seed SIGKILL
and SIGINT, proposal SIGKILL, and cell/colony SIGKILL. These checks use controlled
inference fixtures; they do not establish live model quality.

Headless Chromium opened the trace waterfall and Lab graph-intelligence view
without JavaScript errors after PR #201 repaired the embedded script boundary.
The browser fixture contains synthetic research data. Full live-dish browser
acceptance remains open.

The historical archive remains private by maintainer decision. Public evaluation
uses synthetic fixtures; historical checks require an explicit local opt-in.
No private research document or local corpus manifest is included in this stack.

Additional fixes in this stack: PR #194 makes graph writes transactional,
PR #198 separates required search from final JSON output, PR #200 corrects the
seed-retirement test baseline, and PR #203 handles complete fenced Pi tool calls.

## Integration checks through PR #218

The v1 retirement stack passed 1,272 offline tests, with one skip and 24
integration or live tests deselected. Mypy checked 139 source files; Ruff and
whitespace checks passed. Tests tied to the removed v1 implementation were
retired with it; [the retirement guide](RETIRE_V1.md) maps their v2 replacements.

PRs #204–209 add the Analyst, opt-in priority scheduling, durable CLI controls,
seed editing, the re-decomposition approval gate, and the feedback RFC.
PRs #210–217 correct decomposition output instructions, add opt-in public-web
retrieval with citation provenance, make defaults restoration exact, resume
failed proposals, isolate manual scans from workflow recovery, retain inference
failure diagnostics, resolve cross-colony premises, and attribute real phase
usage to cells.

The exact local Gemma completed shallow two-colony reference runs through both
Pi and the direct API, including real search and a link to an existing non-root
cell. Full quality acceptance remains open: broader decomposition cases and
continuous seed-to-grow runs still produce validation failures. These failures
are retained for diagnosis; no fallback verdict counts as a pass.

Private archive tests remain opt-in, and worktree housekeeping remains pending.
No private audit or source manifest is included in this public tracker.

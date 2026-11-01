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
| [#9](https://github.com/onthemarkdata/petri/issues/9) | [EPIC] M1-harness — Harness & inference layer | Pending |
| [#10](https://github.com/onthemarkdata/petri/issues/10) | [EPIC] M2-agents — Agents on Pydantic AI | Pending |
| [#11](https://github.com/onthemarkdata/petri/issues/11) | [EPIC] M3-decomposer — Agentic decomposer + evals | Pending |
| [#12](https://github.com/onthemarkdata/petri/issues/12) | [EPIC] M4-dbos — Durable execution | Pending |
| [#13](https://github.com/onthemarkdata/petri/issues/13) | [EPIC] M5-otel — OpenTelemetry | Pending |
| [#14](https://github.com/onthemarkdata/petri/issues/14) | [EPIC] M6-storage — Storage & dashboard reads | Pending |
| [#15](https://github.com/onthemarkdata/petri/issues/15) | [EPIC] M7-lifecycle — Iterative lifecycle | Pending |
| [#18](https://github.com/onthemarkdata/petri/issues/18) | Create petri/harness error taxonomy and retry core | Pending |
| [#19](https://github.com/onthemarkdata/petri/issues/19) | Add TestModel and FunctionModel offline fixtures | Pending |
| [#20](https://github.com/onthemarkdata/petri/issues/20) | Add fake_pi stub and the tests/README testing tiers | Pending |
| [#21](https://github.com/onthemarkdata/petri/issues/21) | Implement the pi RPC transport | Pending |
| [#22](https://github.com/onthemarkdata/petri/issues/22) | Implement PiModel over the pi RPC transport | Pending |
| [#23](https://github.com/onthemarkdata/petri/issues/23) | Port the Claude Code CLI to a ClaudeCodeModel | Pending |
| [#24](https://github.com/onthemarkdata/petri/issues/24) | Extract prompt builders into reasoning/prompts.py | Pending |
| [#25](https://github.com/onthemarkdata/petri/issues/25) | Resolve harness backends from petri.yaml | Pending |
| [#26](https://github.com/onthemarkdata/petri/issues/26) | Bridge decomposition calls to pydantic-ai agents | Pending |
| [#27](https://github.com/onthemarkdata/petri/issues/27) | Wire the bridge into the CLI with verdict discipline | Pending |
| [#28](https://github.com/onthemarkdata/petri/issues/28) | Define the typed agent contract and frozen roster | Pending |
| [#29](https://github.com/onthemarkdata/petri/issues/29) | Add conformance tests pinning verdict enums to YAML | Pending |
| [#30](https://github.com/onthemarkdata/petri/issues/30) | Ship the v2 petri.yaml agent-content schema | Pending |
| [#31](https://github.com/onthemarkdata/petri/issues/31) | Build the 13-agent roster via a pydantic-ai factory | Pending |
| [#32](https://github.com/onthemarkdata/petri/issues/32) | Implement debates as real agent hand-off with cost caps | Pending |
| [#33](https://github.com/onthemarkdata/petri/issues/33) | Move source-hierarchy policy to an output validator | Pending |
| [#34](https://github.com/onthemarkdata/petri/issues/34) | Port mechanical convergence to the typed contract | Pending |
| [#35](https://github.com/onthemarkdata/petri/issues/35) | Add petri agents list and check commands | Pending |
| [#36](https://github.com/onthemarkdata/petri/issues/36) | Add petri agents run with usage and cost caps | Pending |
| [#37](https://github.com/onthemarkdata/petri/issues/37) | Thread dish config through the decomposition path | Pending |
| [#38](https://github.com/onthemarkdata/petri/issues/38) | Consolidate dish-id resolution and cell-ID parsing | Pending |
| [#39](https://github.com/onthemarkdata/petri/issues/39) | Define the petri.sqlite schema and migrations | Pending |
| [#40](https://github.com/onthemarkdata/petri/issues/40) | Adopt dish-scoped layout with petri.sqlite store | Pending |
| [#41](https://github.com/onthemarkdata/petri/issues/41) | Rebuild the decomposer on pydantic-ai agents | Pending |
| [#42](https://github.com/onthemarkdata/petri/issues/42) | Add bedrock stops and counterargument sub-claims | Pending |
| [#43](https://github.com/onthemarkdata/petri/issues/43) | Enforce per-parent caps in decomposer code | Pending |
| [#44](https://github.com/onthemarkdata/petri/issues/44) | Add search_cells tool for cross-colony edges | Pending |
| [#45](https://github.com/onthemarkdata/petri/issues/45) | Refuse to overwrite an existing colony in seed | Pending |
| [#46](https://github.com/onthemarkdata/petri/issues/46) | Checkpoint seed state and add petri seed --resume | Pending |
| [#47](https://github.com/onthemarkdata/petri/issues/47) | Add an edit option to seed approval | Pending |
| [#48](https://github.com/onthemarkdata/petri/issues/48) | Add pydantic_evals suite for decomposition quality | Pending |
| [#49](https://github.com/onthemarkdata/petri/issues/49) | Spike: validate DBOS-on-SQLite for durable execution | Pending |
| [#50](https://github.com/onthemarkdata/petri/issues/50) | Add an ExecutionBackend seam in front of the v1 engine | Pending |
| [#51](https://github.com/onthemarkdata/petri/issues/51) | Make append_event idempotent with deterministic IDs | Pending |
| [#52](https://github.com/onthemarkdata/petri/issues/52) | Model the per-cell pipeline as a typed pydantic-graph | Pending |
| [#53](https://github.com/onthemarkdata/petri/issues/53) | Document the queue-state to pipeline-node mapping | Pending |
| [#54](https://github.com/onthemarkdata/petri/issues/54) | Port queue transition tests to graph conformance suite | Pending |
| [#55](https://github.com/onthemarkdata/petri/issues/55) | Port research-side phases into pipeline nodes | Pending |
| [#56](https://github.com/onthemarkdata/petri/issues/56) | Port decision-side phases and iterate loop into nodes | Pending |
| [#57](https://github.com/onthemarkdata/petri/issues/57) | Implement DBOSExecutionBackend with durable cell workflows | Pending |
| [#58](https://github.com/onthemarkdata/petri/issues/58) | Add grow_colony workflow for bottom-up scheduling | Pending |
| [#59](https://github.com/onthemarkdata/petri/issues/59) | Re-platform petri seed onto the DBOS backend | Pending |
| [#60](https://github.com/onthemarkdata/petri/issues/60) | Port sync-check as a startup reconciliation step | Pending |
| [#61](https://github.com/onthemarkdata/petri/issues/61) | Repoint dashboard /api/queue onto the backend seam | Pending |
| [#62](https://github.com/onthemarkdata/petri/issues/62) | Rewire grow/stop/feed onto the durable backend | Pending |
| [#63](https://github.com/onthemarkdata/petri/issues/63) | Retire the v1 engine and fcntl queue | Pending |
| [#64](https://github.com/onthemarkdata/petri/issues/64) | Add OTel bootstrap and petri.yaml telemetry section | Pending |
| [#65](https://github.com/onthemarkdata/petri/issues/65) | Spike: choose the pydantic-ai OTel instrumentation path | Pending |
| [#66](https://github.com/onthemarkdata/petri/issues/66) | Enable DBOS OTel spans behind the durability seam | Pending |
| [#67](https://github.com/onthemarkdata/petri/issues/67) | Instrument pydantic-ai agent runs with OTel | Pending |
| [#68](https://github.com/onthemarkdata/petri/issues/68) | Attach petri.* domain attributes to spans | Pending |
| [#69](https://github.com/onthemarkdata/petri/issues/69) | Persist spans and usage rows to petri.sqlite | Pending |
| [#70](https://github.com/onthemarkdata/petri/issues/70) | Add read-only trace API endpoints to the dashboard | Pending |
| [#71](https://github.com/onthemarkdata/petri/issues/71) | Add a per-cell trace waterfall to the dashboard | Pending |
| [#72](https://github.com/onthemarkdata/petri/issues/72) | Surface per-cell token and cost totals in the dashboard | Pending |
| [#73](https://github.com/onthemarkdata/petri/issues/73) | Document observability: local default and OTLP export | Pending |
| [#74](https://github.com/onthemarkdata/petri/issues/74) | Add a petri/query read layer over petri.sqlite | Pending |
| [#75](https://github.com/onthemarkdata/petri/issues/75) | Move REST endpoints to petri/query and add /api/edges | Pending |
| [#76](https://github.com/onthemarkdata/petri/issues/76) | Rebuild /api/stream on events-table tailing | Pending |
| [#77](https://github.com/onthemarkdata/petri/issues/77) | Retire migrate.py and the disposable dashboard index | Pending |
| [#78](https://github.com/onthemarkdata/petri/issues/78) | Add petri backup with --list and a snapshot hook | Pending |
| [#79](https://github.com/onthemarkdata/petri/issues/79) | Add petri backup --restore with an overwrite guard | Pending |
| [#80](https://github.com/onthemarkdata/petri/issues/80) | Add petri export for git-reviewable derived text | Pending |
| [#81](https://github.com/onthemarkdata/petri/issues/81) | Replace /api/proc and sync seed with durable runs | Pending |
| [#82](https://github.com/onthemarkdata/petri/issues/82) | Harden colony graph integrity and persistence | Pending |
| [#83](https://github.com/onthemarkdata/petri/issues/83) | Ship starter analytics views in petri.sqlite | Pending |
| [#84](https://github.com/onthemarkdata/petri/issues/84) | Write the verdict-driven re-decomposition RFC | Pending |
| [#85](https://github.com/onthemarkdata/petri/issues/85) | Detect re-decomposition triggers from verdict history | Pending |
| [#86](https://github.com/onthemarkdata/petri/issues/86) | Build the re-decomposition proposal workflow | Pending |
| [#87](https://github.com/onthemarkdata/petri/issues/87) | Add the re-decomposition approval gate to petri grow | Pending |
| [#88](https://github.com/onthemarkdata/petri/issues/88) | Re-validate fed evidence by forking workflows | Pending |
| [#89](https://github.com/onthemarkdata/petri/issues/89) | Re-scope the contradiction scanner and schedule it | Pending |
| [#90](https://github.com/onthemarkdata/petri/issues/90) | Extend the edge registry with dish-level graph queries | Pending |
| [#91](https://github.com/onthemarkdata/petri/issues/91) | Compute edge-intelligence metrics from the dish graph | Pending |
| [#92](https://github.com/onthemarkdata/petri/issues/92) | Surface edge intelligence in CLI and dashboard | Pending |
| [#93](https://github.com/onthemarkdata/petri/issues/93) | Add opt-in convergence-point priority to grow scheduling | Pending |
| [#94](https://github.com/onthemarkdata/petri/issues/94) | Add a read-only Analyst for research-health flags | Pending |
| [#95](https://github.com/onthemarkdata/petri/issues/95) | Write the GitHub feedback re-entry RFC | Pending |
| [#96](https://github.com/onthemarkdata/petri/issues/96) | Archive merged worktrees and record corpus provenance | Pending |
| [#100](https://github.com/onthemarkdata/petri/issues/100) | Petri v2 migration — master tracker | Pending |
| [#117](https://github.com/onthemarkdata/petri/issues/117) | Wire `petri --version` into the CLI | Pending |
| [#120](https://github.com/onthemarkdata/petri/issues/120) | Wire up the pi harness end-to-end | Pending |


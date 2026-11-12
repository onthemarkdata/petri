# Validate the local Petri lifecycle

`scripts.e2e.lifecycle_runner` is an opt-in acceptance runner for issue
[#62](https://github.com/onthemarkdata/petri/issues/62). It runs the public CLI
and reads committed domain rows and DBOS checkpoints. Offline tests use synthetic
SQLite databases and mocked command outcomes. They establish runner behavior,
not live model acceptance.

## Prerequisites

Integrate Deliverable A's `scripts.e2e.local_endpoint` before running inference.
The runner imports its public interface only for `run`; offline checks, status,
and evidence attachment do not need A. Content capture also requires the
telemetry changes that add `telemetry.capture_content` and
`telemetry.redact_env_vars`. The runner does not modify production telemetry.
An explicit system-message profile requires the separate reviewed resolver
contract, `harness.openai_chat_supports_multiple_system_messages`, an optional
strict boolean. The runner rejects an explicit choice while that contract is
absent; it does not implement an adapter or silently discard the choice.

Root must validate the assigned loopback OpenAI-compatible route and device
headroom first. Supply its actual API base, including any path prefix, model ID,
provenance JSON, authentication selection, and configured maximum context size.
Endpoint readiness does not establish research quality. The runner neither
starts model services nor downloads weights or changes device allocations.

Context capacity and output length are separate. `--context-size` records the
selected endpoint's configured maximum; it has no fixed 32,768-token ceiling.
`--max-tokens` bounds each response. Worker concurrency and CLI concurrency stay
at one. Defaults allow 120 requests, 600,000 cumulative tokens, and 100 tool calls
per cell round; each debate permits 24 requests, 100,000 tokens, and 20 tool calls.
Each command has a 14,400-second wall-clock bound. These budgets retain the full
agent roster, source hierarchy, verdict vocabulary, and validation retries.

## Run and resume

Run from the integrated repository with its locked Python environment. Choose a
new empty artifact directory outside the source tree. This example uses bearer
authentication through an existing environment variable; use A's
`--basic-auth-file PATH` or `--no-auth` instead when appropriate. Never put a
credential value in command arguments.

```sh
export PETRI_RUN_LIFECYCLE=1
export PYTHONPATH="$PWD"
python -m scripts.e2e.lifecycle_runner run "$RUN_DIRECTORY" \
  --adapter openai --base-url "$LOCAL_API_BASE" --model "$LOCAL_MODEL_ID" \
  --provenance "$MODEL_PROVENANCE_JSON" --context-size "$MODEL_CONTEXT_SIZE" \
  --api-key-env LOCAL_MODEL_API_KEY --max-tokens 4096 \
  --output-mode prompted --openai-chat-system-messages single \
  --request-limit 120 --token-limit 600000 --tool-limit 100 --timeout 14400
```

The example uses Root's selected profile for a **new** run. `--output-mode`
retains its existing `tool` default; `prompted` is explicit. The compatibility
choice has three values:

| `--openai-chat-system-messages` | Harness YAML field | Identity value |
|---|---|---|
| `default` (default) | omitted | `null` |
| `single` | `openai_chat_supports_multiple_system_messages: false` | `false` |
| `multiple` | `openai_chat_supports_multiple_system_messages: true` | `true` |

`single` selects the SDK's supported merge of consecutive leading system
messages, preserving instruction text and order. B calls the public resolver
for validation of the provider/openai-chat selection. It makes no automatic
fallback after a failed diagnostic. Keep the native tool-mode cap failure and
the original prompted HTTP 400 as separate diagnostics. Root's successful
one-agent profile probe does not satisfy a multi-colony lifecycle stage.

After telemetry integration, explicitly add `--capture-content` and
`--otlp-traces-endpoint "$LOCAL_OTLP_TRACES_ENDPOINT"` for diagnostic acceptance.
The OTLP option accepts a loopback HTTP(S) URL without credentials, query, or
fragment. Capture adds `OPENAI_API_KEY` to `redact_env_vars` so the temporary relay
key is protected. The selected diagnostics and configuration hash enter the saved
identity. Capture is off and external export is unset unless requested.

For a larger finite capture envelope, integrate the reviewed
[`CaptureLimits` policy](observability.md#bounds-and-delivery-limits), create a
UTF-8 JSON file containing only its public fields, and add
`--capture-content --capture-limits "$CAPTURE_LIMITS_JSON"` to a **new** run.
The runner accepts a regular file of at most 4,096 bytes. The complete ceiling
example below is equivalent to `{}`; omitted fields take public schema defaults.
Each field may be reduced to a strict positive integer within its ceiling.

<!-- lifecycle-capture-limits -->
```json
{
  "max_text_chars": 1048576,
  "max_collection_items": 512,
  "max_depth": 64,
  "max_nodes": 16384,
  "max_input_bytes": 8388608,
  "max_span_bytes": 4194304
}
```

The option requires `--capture-content` and is valid only for `run`. Malformed,
nonobject, oversized or duplicate-key JSON, unknown fields, booleans, numeric
strings, nonpositive values and values above public ceilings are rejected before
endpoint construction, readiness, model validation, run-directory writes or
child launch. Errors do not echo file contents. Do not place credentials or
importable code in this public configuration file.

The runner reads and resolves the policy once per invocation through the public
schema. The same resolved values enter all three dishes' telemetry YAML, the
configuration hash and `identity.json` diagnostics. The path and JSON spelling
are not policy identity: an equivalent file at a new path can resume, while
changed resolved values or omission of a previously explicit policy reject
resume before readiness or children. File changes after selection do not alter
the bound invocation. Omitting the option preserves existing configuration and
identity behavior; it adds no `capture_limits` identity field. Changes to live
YAML still fail the existing preflight, even when prepare already passed.

The policy does not change SDK limits, inference budgets, model settings,
stages, source/verdict criteria or acceptance gates. Aggregate overflow drops a
span before either sink; bounded values, missing spans and queue losses still
fail complete-content acceptance. Review the core resource measurement and host
headroom before any live use; finite ceilings do not guarantee that every model
context fits. Capture truncation does not establish inference-input truncation.
Preserve earlier failed captures. A source change requires a fresh run; never
resume an old failed run through this patch or edit generated YAML after its
identity is saved. This runner remains native; Pi execution is separate.

Resume with the identical command plus `--resume`. The saved identity includes
the source revision, real upstream endpoint, model provenance, context and output
limits, configuration hash, command deadline, and diagnostics. A's guarded route
keeps upstream authentication in memory. Its temporary URL and key are passed to
children through the environment; the saved endpoint remains the real upstream
route. Child logs redact both upstream authentication and the temporary key.
Identity also records output mode and the canonical nullable profile value.
A changed output mode, profile, model, concurrency, or diagnostic setting needs
a new empty run directory. An earlier tool-mode run cannot be relabeled as a
prompted/single-system run.

`manifest.json` is saved before and after every stage. Stage attempts have
separate directories with command arguments, PID, exit status, timings, logs,
snapshots, and reports. Earlier failures remain intact. Passed stages reuse their
hashed artifacts. An interrupted stage resumes without replacing the dish;
commands that already committed feed evidence attach through Petri's evidence
identity handling. `missing`, `pending`, `running`, `interrupted`, `skipped`,
`failed`, and `passed` remain distinct. Only every mandatory stage passing can
produce `live_acceptance: true` and exit zero. A completed CLI sequence with
pending Root evidence still exits one.

Before any resumed child starts, the runner compares every existing cycle's
live YAML with its saved configuration and uses Petri's effective loader, even
when prepare already passed. Recovery and feed sidecars must match independently
retained attempt copies. Missing or changed sidecars fail before another command.
The v2 manifest is required; preserve older run evidence rather than rewriting it.

## Cases and stage gates

Three independent dishes, `lifecycle-cycle-1` through `lifecycle-cycle-3`, each
seed `isolated` before `shared`. Both exact claims and their expected statuses
come from `tests/helpers/local_grow_worker.py`:

| Colony | Exact claim | Pre-feed expectation | Post-feed expectation |
|---|---|---|---|
| isolated | In SQLite, two separate connections opened with the exact filename ':memory:' create independent databases; a table created through one is not visible through the other. | VALIDATED, r1 | VALIDATED, r2 for fed cells |
| shared | In SQLite, two separate connections opened with the exact filename ':memory:' share one database; a table created through one is immediately visible through the other. | DISPROVEN, r1 | DISPROVEN, unchanged r1 |

URI shared-cache databases are outside both claims. This is the public
known-outcome case already documented in [local grow validation](local-grow-validation.md).
The source references there are case provenance; live acceptance requires Petri
to retrieve actual sources during each required research round.

The two claims share the exact connection and filename conditions. The explicit
reuse gate requires a `cross_colony` edge from `shared` to a **non-root** cell in
`isolated`. Two isolated colonies, a reference to the foundation root, or an
unrelated edge fail this gate. This exercises the reference component of
[#44](https://github.com/onthemarkdata/petri/issues/44) and
[#48](https://github.com/onthemarkdata/petri/issues/48); it does not replace the
separate decomposition evaluation suite. The runner uses shallow decomposition
safety caps and fails if the model does not produce the required reference.

Each cycle records these stages separately:

1. Prepare; interrupt seed after a durable decomposition step; resume seed;
   read actual approval; compare seed continuity; seed and approve the second
   colony; verify the reference.
2. Interrupt grow after a durable cell-phase checkpoint; resume grow; compare
   committed step/event identities; check both exact roots and their r1 research.
3. Feed actual retrieved source text with provenance to the foundation colony;
   check the resulting forks; check post-feed research; repeat the same feed.
4. Export twice, back up and list, restore into a separate destination, check
   overwrite protection, compare SQLite rows and exports, and retain the warning.
5. Attach dashboard API/SSE, browser, and stored trace readback evidence for this
   same successful dish. A separate deterministic recovery evidence stage is
   mandatory across the run.

`seed --no-questions` auto-accepts the model-generated colony. The runner records
the real `colony_approved` event and labels this approval mode; it does not claim
manual review. The existing single-cell fixture probe remains available as
limited grow diagnostics. It cannot satisfy this runner's model-generated seed,
approval, cross-colony reference, or full lifecycle gates.

Noninteractive `feed --colony isolated --auto-reopen` selects all eligible
terminal cells in that colony, including `VALIDATED`, `DISPROVEN`, `DEFER_OPEN`,
and previously fed `NEW` cells. There is no exact-cell CLI option. This case
supports only an all-validated foundation target set, with every target expected
to remain validated in r2. A disproven or deferred descendant fails preflight
before feed changes the dish; preserve the generated claims for review. The
runner retains the original eligible target set across interrupted attempts,
including previously fed `NEW` cells, and rejects changed eligibility. It feeds
real `source_reviewed` content with URL, timestamp, and
hash, checks the retained evidence file and events, compares copied Socratic
checkpoint outputs, requires new research, and verifies all transitive dependent
flags. Only declared feed targets may receive new validation workflows. At least
one dependent outside the target set must remain unchanged. Repeating the same
evidence must create no new fork or propagation effect.

Research reports use exact dish, colony, cell, round, and workflow identities.
They reconstruct hashed event coordinates and follow usage span ancestry to the
explicit workflow round. Both roots must have the declared terminal status and
successful DBOS workflow. Each required round needs a search with nonzero hits,
a successful public-web fetch with final URL, timezone-aware timestamp, content
hash and text, and an eligible fetched citation supporting the evaluator's
reported verdict. Empty searches, fetch failures, old-round evidence, pending
roots, and `STALLED` results cannot pass. Snapshots retain all previous attempts
and unscoped telemetry for diagnosis.

Recovery compares completed checkpoint identities and output hashes plus existing
committed event identities and payloads. The runner terminates only the child it
started, after observing an active `PENDING` target workflow with a committed
checkpoint, then resumes the same dish. Historical `SUCCESS` workflows cannot
start a new recovery experiment. A retained confirmed kill is reused on retry;
an ambiguous kill or completed command fails closed for Root's inspection. These
observations establish continuity, not an exact external-call count. A call whose
response was not committed may repeat. Root must also supply the deterministic
recovery checks below, which assert exact fixture call counts.

Each child first acknowledges its process birth identity to a durable launch
ticket. It enters Petri only after the parent saves authorization. Catchable
termination cleans up that child. On resume, a surviving authorized child blocks
new commands; Root must inspect its PID, boot ID, start ticks, command, working
directory, and dish ownership before stopping that exact process or waiting for
it to exit. A reused PID is never a signal target. A held execution lock also
blocks resume. SIGKILL can leave an authorized child alive; reconciliation keeps
the next runner from overlapping it.

Storage checks restore only into an attempt's `isolated-restore` directory. The
source database is never the restore target. Restored exports use database-held
fed content without restored evidence files; the original feed source is held
under a temporary name during that export and restored afterward. Preserve the
whole directory after interruption. The captured DBOS/domain desynchronization
warning is mandatory: a domain snapshot does not recover DBOS control state.
These checks cover refusal without `--force`, not a successful forced overwrite.

## Root-operated evidence

Start the dashboard for the relevant cycle using Petri's normal dashboard path
on a port Root has confirmed is free. Do not replace a shared dashboard. Root
owns that server and any browser interaction. During the task-owned feed command,
capture actual API and SSE data in a second terminal:

```sh
python -m scripts.e2e.lifecycle_runner capture-dashboard "$RUN_DIRECTORY" \
  --cycle 1 --dashboard-url "$DASHBOARD_ORIGIN" --cli-pid "$ACTIVE_PETRI_CLI_PID" \
  --capture-timeout 120 --retry-interval 1 --final-status-timeout 14400
```

The PID must be an active Petri CLI command in that exact cycle directory. The
capture retries across per-cell feed ownership gaps for at most 120 seconds,
using the same process birth identity. Each complete attempt checks the execution
owner lock before and after GETs, requires matching
dish/cell/graph identities, nonempty usage and trace payloads, waterfall spans,
and a committed research event from SSE. Request and SSE timeouts respect the
capture deadline. Failed attempts retain their payloads and reasons separately.
If the CLI exits before a complete owned capture, retry during another task-owned
command; do not treat an idle dashboard response as coexistence evidence.

The owned snapshot may contain reopened roots. After a complete owned capture,
the procedure separately waits up to 14,400 seconds for both declared terminal
root outcomes, saving `final-roots.json`. It writes `evidence.json` only when
ownership observation and root confirmation both pass. Expiration retains the
capture but leaves acceptance pending. These observation bounds do not change
inference budgets. Capture can run alongside inference; attachment waits until
the main runner releases its manifest lock and rechecks the current root statuses.

Attach it afterward:

```sh
python -m scripts.e2e.lifecycle_runner attach "$RUN_DIRECTORY" \
  --stage cycle-1/dashboard_api --evidence "$DASHBOARD_EVIDENCE_JSON"
```

HTTP success alone never passes `browser`. Root must inspect graph, cells,
usage, trace waterfall, seed controls, run controls, and read-only coexistence
during CLI ownership, and retain screenshots or browser recordings and console
readback. For each cycle, supply JSON with these mandatory fields:

```json
{
  "observer": "Root",
  "observed_at": "<UTC timestamp>",
  "dish_id": "lifecycle-cycle-1",
  "root_statuses": {
    "lifecycle-cycle-1-isolated-000-000": "VALIDATED",
    "lifecycle-cycle-1-shared-000-000": "DISPROVEN"
  },
  "checks": {
    "graph": false,
    "cells": false,
    "usage": false,
    "trace_waterfall": false,
    "seed_controls": false,
    "run_controls": false,
    "read_only_during_cli": false,
    "no_console_errors": false
  },
  "artifacts": [{"path": "<relative path inside run directory>", "sha256": "<file SHA-256>"}]
}
```

Set a check to true only after inspecting its evidence. Attach using
`--stage cycle-1/browser`; repeat for cycles 2 and 3. Missing attachments,
incorrect hashes, wrong dish identities, or incomplete checks fail attachment.

Stored trace readback is separately pending. Use the same observer, timestamp,
dish, root statuses, and artifact fields; replace `checks` with:

```json
{
  "workflow_step_agent_model_tool_ancestry": false,
  "request_response_readback": false,
  "source_verdict_readback": false,
  "round_usage_readback": false
}
```

Also supply `workflow_ids`, `span_ids`, and `response_ids`. Workflow IDs must
cover both roots' r1 workflows and every declared foundation feed target's r2
workflow. Selected stored spans must include each explicit cell-round workflow
identity and every locally stored model response in those rounds. Keep response
IDs verbatim. Supply `response_readbacks` with one mapping per stored response:

```json
{
  "response_id": "<exact gen_ai.response.id>",
  "span_id": "<local model span ID>",
  "trace_id": "<local trace ID>",
  "workflow_id": "<exact cell ID>:r2",
  "artifact_path": "<relative retained JSON readback path>"
}
```

The named artifact must also appear in `artifacts`. Its JSON must contain the
same four identity fields and a nonempty `remote_record` holding the retrieved
service record. The runner joins each response to the local chat model span and
its same-trace cell/round ancestry; unknown IDs, wrong rounds, missing responses,
and attachments without matching coordinates fail. Root
must retrieve stored request/response content and source/verdict evidence,
inspect workflow → step → agent → model/tool ancestry, and retain that readback
as artifacts. Listing IDs or seeing spans in an HTTP response is insufficient.
Attach using `--stage cycle-1/trace_readback`, then repeat for cycles 2 and 3.

For `deterministic_recovery`, supply `observer`, `observed_at`, `source_sha`
matching `identity.json`, hashed artifacts, and true checks named
`seed_exact_counts`, `grow_exact_counts`, `feed_exact_counts`, and
`stop_semantics`. Root can run the existing offline fixture assertions:

```sh
python -m pytest -q -m integration \
  tests/integration/test_seed_resume.py \
  tests/integration/test_dbos_recovery.py \
  tests/integration/test_feed_fork.py
python -m pytest -q \
  tests/unit/test_cli_stop_file.py tests/unit/test_processor_stop_file.py
```

Retain full output and test counts, including any failures or skips; a failed or
skipped required assertion cannot become a true check. These tests use controlled
inference fixtures, not live models. Attach the evidence with
`--stage deterministic_recovery`.

```sh
python -m scripts.e2e.lifecycle_runner status "$RUN_DIRECTORY"
```

Status exits one until every mandatory stage has passed and its hashed
artifacts remain available. This runner does not publish acceptance results.

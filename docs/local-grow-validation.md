# Local grow diagnostics

This opt-in probe runs actual typed research phases and retrieves public web
sources. It is separate from offline tests and model readiness. It accepts an
explicit loopback OpenAI-compatible endpoint and model through either the native
provider or Pi. See [model readiness](local-model-validation.md) for provenance,
authentication, URL restrictions, and the retained default DMR/Gemma adapter.

The helper records limited grow evidence. It does not evaluate a complete
lifecycle, and every report sets `complete_cycle_passed` to `false`. Multi-colony
seed/approve/grow/feed cycles, crash recovery, dashboard, export, and
backup/restore require separate acceptance work.

Two public claims have opposite expected outcomes. SQLite documents that two
connections opened with the unadorned filename `:memory:` create independent
databases. The `isolated` case states that behavior and expects `VALIDATED`;
the `shared` case asserts the inverse and expects `DISPROVEN`. URI shared-cache
databases are outside both claims' scope. [SQLite in-memory database documentation](https://www.sqlite.org/inmemorydb.html),
[Python sqlite3 reference](https://docs.python.org/3.12/library/sqlite3.html).

By default, the worker prepares a clearly labeled single-cell colony through
the storage API, then calls the actual `petri grow` command. These fixtures
isolate grow behavior; they do not claim model-generated decomposition. Add
`--seed` for a continuous model-generated `seed → approve → grow` run. That path
uses `seed --no-questions`, which auto-accepts the model's generated colony.
All runs retain the configured roster, verdict vocabularies, schema validation,
source gates, retries, and iteration/usage limits. The seed path uses a shallow
tree and a code-enforced level-one cap to bound the number of cells.

For a full-roster diagnostic, `--expanded-budget` explicitly configures 120 requests,
600,000 tokens, and 100 tool calls per cell run, plus 24 requests, 100,000 tokens,
and 20 tool calls per debate. This allows each evidence role to fetch its own
cited sources. Packaged defaults stay unchanged. The report records the actual
configuration and persisted usage; the source and verdict gates still apply.

The probe also bounds native provider responses with `model_settings.max_tokens`
(`--max-tokens`, default 4,096). This is a per-response generation limit, separate
from the cumulative usage budget. Pi uses `maxTokens` in the generated isolated configuration, matching
`--max-tokens`. `--context-size` records the selected context limit and sets Pi's
`contextWindow`; a generic server must already be deployed with that capacity. Petri
rejects native model settings for CLI backends rather than ignoring them.
Truncated output must still pass the full schema and evidence validators.

`--output-mode prompted` configures `harness.output_mode: prompted` for final
assessments, including Socratic analysis. Native provider research still uses
registered host function tools. This selects JSON final-result delivery without
accepting malformed tool arguments or changing the assessment schema. The
packaged native-provider default remains tool output.

Run acceptance probes serially. Automatic llama.cpp slots can share a unified
KV buffer even when each slot advertises the full configured context length.
The recorded DMR/Gemma setup used one 32,768-token slot, with the exact model
unchanged. Docker supports model runtime flags through
[`docker model configure`](https://docs.docker.com/ai/model-runner/configuration/).
Verify the effective slot count before starting a long run; preserve existing
runtime flags when changing the slot setting.

From the repository, with the local model already running and its provenance
manifest prepared:

```sh
export PETRI_RUN_LOCAL_GROW=1
output=$(mktemp -d)
PYTHONPATH=. uv run python tests/helpers/local_grow_worker.py "$output" \
  --harness provider --case isolated \
  --base-url http://127.0.0.1:8000/prefix/v1 --model local-model-alias \
  --provenance /tmp/local-model/provenance.json --context-size 262144 \
  --max-tokens 4096 --no-auth
```

Replace `--no-auth` with the protected bearer or Basic auth option when the route
requires it. Omitting generic connection flags retains the DMR/Gemma defaults;
`--adapter dmr` selects that adapter explicitly. No credentials are replaced
with a no-auth placeholder on an authenticated route.

Repeat in a new empty directory for `--case shared` or `--harness pi`. Pi needs
its executable available through `PETRI_PI_BIN` or the normal discovery path.
The worker creates its own `pi-agent/models.json` and an attempt-specific runtime
configuration, preserving personal settings and inherited `PI_CODING_AGENT_DIR`.
It uses a task-owned guarded relay for both clients; the immutable run identity
and review configuration retain the exact upstream URL. The temporary transport
key is confined to memory/environment, and previous environment settings are
restored after execution.

The worker preserves configuration, domain and DBOS databases, command history,
CLI logs, retrieved-source provenance, verdicts, and failures. Each invocation
writes a new `attempts/<identity>/result.json` and prints its path. Command
outcomes also append to `commands.jsonl`; earlier attempts are retained.
`run-identity.json` records model alias, artifact/revision declarations, upstream
endpoint, adapter/backend, case, fixture kind, limits, source SHA, and the
configuration hash. Diagnostic copies sanitize decoded credential values,
including nested JSON, while preserving protocol keys, public model IDs, and
ordinary text. The relay rejects actual protected credential echoes before
forwarding responses for client persistence. See the
[transport policy](local-model-validation.md) for capture and correlation limits.

A diagnostic pass requires successful seed and grow commands when seed was
requested, the requested colony's exact root at its expected status, successful
search and fetched-source events with current cell/round identities, and no
current-round execution failure. Empty searches, incomplete source provenance,
events from another cell, and an old round's sources cannot satisfy these checks.
All earlier failures remain in the report. Dish-wide usage and event counts are
labeled diagnostics and do not satisfy the pass criteria. A one-cell fixture
can pass this limited diagnostic without establishing a complete cycle.

Allow time for the full roster and configured iteration budget. Poll the
domain events by agent, cell, and iteration while it runs; a source fetch can
precede a slow model response. If an external deadline interrupts the process,
preserve its directory and resume it rather than recreating the fixture:

```sh
PYTHONPATH=. uv run python tests/helpers/local_grow_worker.py "$output" \
  --harness provider --case isolated --resume \
  --base-url http://127.0.0.1:8000/prefix/v1 --model local-model-alias \
  --provenance /tmp/local-model/provenance.json --context-size 262144 \
  --max-tokens 4096 --no-auth
```

Repeat the original connection, budget, and output-mode flags on resume. The
saved identity, Petri YAML, and Pi review configuration must match exactly;
model/backend/case/configuration drift is rejected before inference. Older runs
without a saved identity cannot be silently adopted. Resume inherits the saved
fixture kind, so `--seed` need not be repeated for a generated-seed run. This
helper resumes grow only after a successful recorded seed with the requested
root present. It does not invoke `seed --resume` after a seed interruption or
failure; complete seed-crash recovery requires separate lifecycle validation.

After a recorded terminal execution failure, `--resume --requeue REQUEST_ID`
requests a new validation round through the backend's revalidation API. Use a
stable, explicit request ID. Earlier failed events remain in the report;
requeueing does not edit them or turn them into successful verdicts.

Model/schema/usage failures append `agent_execution_failed` with the phase,
error type, bounded raw output, validation feedback, and available backend
diagnostics. Decomposition failures expose the same diagnostic detail on
`DecompositionError.detail`. Neither record is a validated research verdict.
Failed debates retain bounded raw output and validation feedback in the
`debate_mediated` event's `exchange_summary.failure_detail`, alongside any
completed, validated turns.

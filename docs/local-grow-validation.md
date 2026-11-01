# Validate grow with the local Gemma model

This opt-in probe runs actual typed research phases and retrieves public web
sources. It is separate from offline tests and the seed-only bridge smoke.
It uses the exact model
`huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0` through Docker Model
Runner at `http://127.0.0.1:12434/engines/llama.cpp/v1`; the worker sets that direct
endpoint itself. Pi must have a `petri-local` provider pointing to the same URL.
The worker rejects a different Pi provider endpoint and has no hosted fallback.

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

For full-roster acceptance, `--expanded-budget` explicitly configures 120 requests,
600,000 tokens, and 100 tool calls per cell run, plus 24 requests, 100,000 tokens,
and 20 tool calls per debate. This allows each evidence role to fetch its own
cited sources. Packaged defaults stay unchanged. The report records the actual
configuration and persisted usage; the source and verdict gates still apply.

The probe also bounds native provider responses with `model_settings.max_tokens`
(`--max-tokens`, default 4,096). This is a per-response generation limit, separate
from the cumulative usage budget. Pi uses the explicitly configured `maxTokens`
in its local provider; the probe verifies that it matches `--max-tokens`. Petri
rejects native model settings for CLI backends rather than ignoring them.
Truncated output must still pass the full schema and evidence validators.

From the repository, with the local model already running:

```sh
export PETRI_RUN_LOCAL_GROW=1
output=$(mktemp -d)
PYTHONPATH=. uv run python tests/helpers/local_grow_worker.py "$output" \
  --harness provider --case isolated
```

Repeat in a new empty output directory for `--case shared` and for
`--harness pi`. Pi also needs `PETRI_PI_BIN` and `PI_CODING_AGENT_DIR` configured
for the local provider. Use `--seed` on at least one run when checking app
continuity. The worker preserves its configuration, domain database, DBOS
database, CLI logs, retrieved-source provenance, verdicts, and `result.json`.
A zero CLI exit alone does not pass the probe: the report also requires the
expected terminal cell status, real search/source events, and no execution
failure in the latest round.

Allow time for the full roster and configured iteration budget. Poll the
domain events by agent, cell, and iteration while it runs; a source fetch can
precede a slow model response. If an external deadline interrupts the process,
preserve its directory and resume it rather than recreating the fixture:

```sh
PYTHONPATH=. uv run python tests/helpers/local_grow_worker.py "$output" \
  --harness provider --case isolated --resume
```

After a recorded terminal execution failure, `--resume --requeue REQUEST_ID`
requests a new validation round through the backend's revalidation API. Use a
stable, explicit request ID. Earlier failed events remain in the report;
requeueing does not edit them or turn them into successful verdicts.

Model/schema/usage failures append `agent_execution_failed` with the phase,
error type, bounded raw output, validation feedback, and available backend
diagnostics. Decomposition failures expose the same diagnostic detail on
`DecompositionError.detail`. Neither record is a validated research verdict.

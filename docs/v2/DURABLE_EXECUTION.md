# Durable cell execution

Set the following fields in the dish's `defaults/petri.yaml` to select DBOS.
The default remains `v1` during the migration.

```yaml
execution:
  backend: dbos
  worker_concurrency: 4
  limiter:
    limit: 10
    period: 60
  polling_interval_seconds: 0.05
```

DBOS 3.4.0 stores checkpoints in `.petri/dbos.sqlite`. Research cells, evidence,
and append-only events stay in the dish's `petri.sqlite`. No PostgreSQL server,
Docker container, Conductor, or Console is required. All workflow definitions
are registered once before launch. Queue registration follows launch, as required
by the [DBOS queue API](https://docs.dbos.dev/python/reference/queues).

A cell round has the stable workflow ID `{cell_id}:r{round_number}`. Submitting
that ID again returns its existing execution. Each research or decision phase
is a DBOS step; the terminal status write is a separate step. The public typed
pipeline runner retains transition checks and advances the iteration clock.
Expected inference failures return `NeedsHuman`; rate limits return a small
checkpoint and use durable sleep before another phase attempt.

One process owns a dish's execution at a time. A process lock rejects a second
owner and is released by the operating system after a crash. Read-only clients
may coexist. Application admission also enforces the configured concurrency and
rolling start-rate budget, including the rate window across restarts. The DBOS
queue receives the same limits. This boundary is required by the measured
[SQLite queue limitations](../adr/ADR-M4-dbos-sqlite-spike.md).

Checkpoints are limited to 2 MiB per phase result. They contain control results
and identifiers; evidence remains in the domain database. Treat checkpoint
files as trusted local state because DBOS uses pickle-compatible serialization.
`petri-cell-pipeline-1` pins the workflow shape. Recovery rejects a mismatched
shape; changes to step order require a new application version and a recovery
plan for existing runs.

Completed step results are replayed without repeating their work. A crash after
an external call succeeds but before its event or checkpoint commits can repeat
that call. Deterministic event IDs prevent duplicate domain rows, but do not
promise exactly-once provider calls or billing. Provider idempotency is needed
to close that gap.

Run the local SIGKILL regression explicitly:

```sh
pytest tests/integration/test_dbos_recovery.py -m integration -q
```

The test uses fake phases. It kills a process during research, verifies that a
second owner is rejected, restarts without resubmitting the workflow, and checks
that the checkpointed Socratic phase ran once. The uncheckpointed research call
runs twice, while its domain event remains a single row.

## Colony scheduling and new rounds

A `grow_colony` parent reads the stored graph, enqueues eligible cell rounds, and
waits for their results before checking eligibility again. The unchanged graph
gate requires every dependency to be `VALIDATED`. `DISPROVEN`, `DEFER_OPEN`, and
parked cells therefore block their dependents. The parent reports remaining
`NEW` cells as `not_eligible` and finishes without a second grow invocation.

Parent identities include the colony, requested round, and any explicit cell
scope. Dish-level dispatch applies `--colony` filtering before creating parents.
Dry runs read the eligible list without launching DBOS or creating a workflow.

Revalidation requires a stable request ID. A step uses a transaction to read the
stored rounds, allocate the next number once for that request, and set the cell
to `NEW`. It then appends an idempotent `cell_reopened` event before completing
its checkpoint. Its child uses the distinct ID
`{cell_id}:r{next_round}`. Repeating the request attaches to the same parent and
child; dependents retain their existing status.

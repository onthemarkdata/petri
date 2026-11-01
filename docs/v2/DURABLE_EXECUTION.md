# Durable cell execution

DBOS is the only execution backend. Configure its limits in the dish's
`defaults/petri.yaml`. Selecting `v1` fails with a pointer to v0.3.7.

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

## Startup reconciliation

Phase checkpoints include a small receipt: the cell ID, event count, a hash
of its committed event history, and a hash of its stable cell fields. Before automatic recovery starts, Petri compares
those receipts with the domain database and checks that a successful cell round
has its expected terminal status. Additional events after a checkpoint are
allowed because a crash may occur between an event commit and its checkpoint.

Missing cells, missing checkpointed events, changed event history, and a
successful workflow paired with a different domain status create a persistent
`sync_conflict`. Parking metadata lives in an application-owned table in
`dbos.sqlite`, so restoring an older domain database cannot erase the conflict.
The runtime returns `SyncConflict` and prevents further work for affected cells.
`petri check` shows the divergence reason in both text and JSON output.
Reconciliation does not change cell rows or append, update, or delete events.
It does not automatically repair a conflict or clear a previous parking decision.

## Grow, stop, and feed

Re-running `petri grow` attaches to pending work or resumes cancelled cell rounds
and colony parents with their original checkpoint identities. Completed cell
rounds are reused. New eligible cells added after a completed parent receive a
new parent generation; their cell round numbers still come from the domain
round records. The spinner polls workflow state and new domain events through
the backend.

`petri stop` records cancellation intent in the system database. A running step
finishes and commits its checkpoint before the workflow becomes `CANCELLED`.
`petri stop --force` cancels immediately and interrupts an active async inference
call; that unfinished call can run again on resume. Neither form changes domain
status or writes a stalled verdict. Stopping an idle dish is a no-op. Grow exits
0 for all-terminal completion or a user stop, including Ctrl+C, and 1 for blocked,
conflicted, or failed work. SIGKILL recovery uses the same grow command.

`petri feed` reports transitive impact, checkpoints ingestion, and forks a
completed cell workflow from its first research step. It reuses Socratic output
and appends idempotent propagation flags for dependents. Flags preserve those
cells' statuses and do not enqueue them. Equal evidence attaches to the same
allocated round. See [Feed evidence](FEED.md) for the SQLite fork boundary,
older-checkpoint fallback, typed errors, and SIGKILL validation.

The DBOS adapter accepts an optional async `after_pass(petri_dir) -> bool` hook.
It runs after the selected parents settle and before the runtime closes. A true
result starts another parent generation for newly approved work; the default
has no hook.

The CLI recovery regression covers real SIGKILL and Ctrl+C with local fake
phases:

```sh
pytest tests/integration/test_grow_resume.py -m integration -q
```

## Workflow traces

With telemetry enabled, native DBOS workflow and step spans use the current
dish's provider, exporters, and local SQLite sink. Cell workflow spans carry the
cell ID and round. `telemetry.enabled: false` disables DBOS spans as well as
Petri's other spans. Provider initialization precedes runtime launch.

The DBOS 3.4.0 adapter is confined to the execution backend and restores its SDK
hook when the owner closes. The SQLite test verifies workflow/step parenting,
sequential dish isolation, and disabled telemetry. See the
[D10 verification](../ARCHITECTURE-V2.md#d10-verification-native-dbos-spans-on-sqlite)
for the private SDK boundary and endpoint configuration decision.

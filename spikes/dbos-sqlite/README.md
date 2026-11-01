# DBOS on SQLite: reproducible spike

This spike evaluates issue [#49](https://github.com/onthemarkdata/petri/issues/49).
It uses DBOS 3.4.0, real subprocesses, SQLite files in a fresh temporary directory,
and deterministic fake research calls. It does not import or modify Petri's
production modules, call a model, require credentials, or contact a hosted DBOS
service. Dependency installation requires access to PyPI; execution is offline.

## Run

From the repository root, with Python 3.11 or later and `uv` installed:

```sh
uv run --locked --script spikes/dbos-sqlite/run.py run /tmp/petri-dbos-results.json
```

The output path receives measurements as JSON. Each run creates and removes its
own temporary databases. No database is committed. The script deliberately sends
SIGKILL to a child process it created; run the crash experiment on macOS or Linux.
An exit code of zero means the experiments completed, not that every criterion
passed: inspect each scenario's `pass` field and the final list of failed criteria.
An assertion or subprocess failure indicates an incomplete experiment.

To check the Python floor explicitly:

```sh
uv run --locked --python 3.11 --script spikes/dbos-sqlite/run.py run /tmp/petri-dbos-py311.json
```

For one scenario, use a new empty directory:

```sh
spike_dir=$(mktemp -d)
uv run --locked --script spikes/dbos-sqlite/run.py scenario "$spike_dir" rate
cat "$spike_dir/result.json"
```

The supported scenario names are `semantics`, `replay`, `size`, `queue`, `rate`,
`rate-single`, `rate-fast`, `caps-global`, and `volume-N-E`, where N is 10 or 50 entries and E
is 10 or 500 events per entry. The full `run` command also orchestrates `crash`
and `recover`. Internal `peer` processes provide a second worker on the same
system database. Their executor IDs differ; their application version is pinned.

## What the scenarios measure

- `crash`: kill during an external-effect step, restart the same executor, and
  retrieve the pending workflow without explicitly restarting it. The completed
  step must execute once; the uncheckpointed external effect must appear twice.
- `semantics`: repeated workflow IDs, queue deduplication, two fork boundaries,
  and cooperative cancellation. The active synchronous step finishes; its next
  step must not start.
- `replay`: reproduce two retry shapes with fake call counters, then replay each
  deterministic workflow ID. Compare the resulting domain-event export with the
  committed golden fixture and repeat one event insertion to test its unique ID.
- `queue`: two workers, a global cap of 8 and a worker cap of 4. `caps-global`
  raises the worker cap to 6, making the global cap independently testable.
- `rate` and `rate-single`: two starts per rolling second, with two workers and
  one worker respectively. Report the smallest interval spanning three starts.
  The acceptance threshold allows 100 ms of scheduling tolerance: at least 0.9 s.
  `rate-fast` uses two workers and 64 tasks with a 10 ms hold to expose short-task
  admission races.
- `volume`: concurrent queue entries at the four 10–50 entry / 10–500 event
  corners. Each event is a separate SQLite transaction in a second domain store.
  Global/worker caps scale to N/N÷2. An admission barrier waits for all N entries
  to start before releasing their writes. Report observed concurrency and write
  latency; SQLite still serializes individual write transactions.
- `size`: checkpoint 1 KiB, 1 MiB, and 16 MiB payloads; inspect SQLite's length
  limit; then measure logical database growth for 1,000 one-step queued cells.

Queue timestamps are wall-clock values shared by processes on one machine.
Payload and recovery timings use monotonic clocks. Queue latency includes enqueue
work and backlog delay. Event-write timings include opening a connection and
committing a transaction. Lock errors are collected from child logs; no-error
observations do not establish that SQLite never waits for a lock.

The golden fixture represents this spike's deterministic fake-research workflow.
It is not evidence that Petri's production grow command or a real model completed.
`load_golden()` in `run.py` loads and validates the exported text. The fixture
covers checkpoint/event replay; the production end-to-end suite must separately
cover the real orchestration pipeline.

See [the ADR](../../docs/adr/ADR-M4-dbos-sqlite-spike.md) for measured results,
limitations, and the backend recommendation. These files stay outside the wheel
because the package finder includes only `petri*`.

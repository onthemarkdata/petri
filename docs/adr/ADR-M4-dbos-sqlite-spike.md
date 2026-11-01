# ADR M4: DBOS on SQLite

- Status: **GO-WITH-CONSTRAINTS** for checkpointing behind `ExecutionBackend`;
  **NO-GO** for relying on DBOS 3.4.0's unguarded multi-process SQLite queue limits.
- Issue: [#49](https://github.com/onthemarkdata/petri/issues/49).
- Measured: 2026-11-01, macOS arm64, SQLite 3.50.4, Python 3.11.15 and 3.12.13.
- Dependency: **`dbos==3.4.0`**, the PyPI package from the
  [`dbos-transact-py` repository](https://github.com/dbos-inc/dbos-transact-py).
  Its installed package metadata requires Python >=3.10; this spike requires
  >=3.11 and ran on the project's 3.11 floor. The script lockfile pins transitive
  dependencies without changing Petri's runtime dependencies.

## Decision and conditions

DBOS restored completed checkpoints after SIGKILL and provided usable fork,
cancellation, and deduplication APIs with SQLite. The two-worker experiments
also found failures: a global cap of 8 admitted 10 simultaneous steps, and a
2-starts/second limiter admitted three starts within 2.46 ms. A successful rerun
cannot establish those limits as safe: the failures depend on scheduling.

Proceed only if the backend enforces these conditions:

1. One active execution owner per dish, enforced across CLI and dashboard
   processes. Read-only clients may coexist. An application-level admission
   boundary must enforce strict concurrency and rate budgets; do not delegate
   provider quotas solely to the observed SQLite limiter. This spike establishes
   the need for that boundary, not a validated implementation of it.
2. Keep deterministic workflow identities and deterministic domain-event IDs.
   Domain writes must be idempotent independently of the DBOS checkpoint because
   the two SQLite files do not share a transaction.
3. Record exactly-once **step results**, not a promise of exactly-once external
   calls or billing. A process can die after an external side effect succeeds
   and before its checkpoint commits. Provider idempotency is required to close
   that window; otherwise retries may repeat work.
4. Resolve fork boundaries from stored DBOS step IDs, pin application versions,
   and preserve workflow shape during replay. Do not equate a pipeline's
   zero-based index with DBOS's function ID.
5. Keep checkpoints small and portable. Store large documents in the domain
   store and checkpoint references. Never load an untrusted checkpoint database;
   the default serializer uses pickle-compatible Python objects.

If a single-owner boundary or independent admission control is unsuitable, use
the planned fallback: pydantic-graph `FileStatePersistence` with a Petri-owned
SQLite queue. The fallback still requires idempotent external effects and
transactional queue claims; it does not solve exactly-once model billing.
Re-evaluate DBOS after a verified queue fix before allowing multiple execution
owners on one dish.

## Reproduction and evidence

Run the commands in the [spike README](../../spikes/dbos-sqlite/README.md).
The committed measurements are [Python 3.11](../../spikes/dbos-sqlite/results/python-3.11.json),
[Python 3.12](../../spikes/dbos-sqlite/results/python-3.12.json), and the separate
[fast-workload limiter probe](../../spikes/dbos-sqlite/results/rate-fast-python-3.12.json).
The fast probe ran with Python 3.12.13, DBOS 3.4.0, and the same SQLite build.
It is now also included in the full `run` command. The other result files were
captured before that additional scenario was added to the full-run list.

Each scenario used a fresh local database. No model, PostgreSQL service,
Conductor, Console, or Docker runtime participated. Some runs overlapped on the
same workstation, so timing values describe this workload and host; they are
not isolated hardware benchmarks. Every scenario is bounded by a subprocess
timeout. The experiment runner reports failed criteria separately from crashes.

| Criterion | Observed result | Verdict |
|---|---|---|
| SIGKILL in a step, then automatic recovery of PENDING workflow | Restart to result: 2.30 s (3.11), 2.33 s (3.12). Completed fake call: once; uncheckpointed fake call: twice. Recovery retrieves the workflow after launch without starting it again. | PASS for checkpoint replay; external exactly-once is disproved |
| Submit the same `SetWorkflowID` twice | Three steps executed three times total, not six | PASS |
| Duplicate queued `deduplication_id` | `DBOSQueueDeduplicatedError` raised while original was outstanding | PASS |
| Worker cap 4, two processes | Both participated; peak 4 per process, total 8 | PASS in these runs |
| Independent global cap 8, worker cap 6 | Python 3.11 peak 8; Python 3.12 peak **10**, with both processes participating | **FAIL** |
| Two starts per rolling second | Ordinary workload passed; 64 fast tasks produced a three-start interval of **0.00246 s**, below the 0.9 s tolerance | **FAIL** |
| Polling at 50 ms, `use_listen_notify=False`, 16 entries at concurrency 8 | Enqueue-to-start p50/p95: 0.666/1.124 s (3.11), 0.693/1.194 s (3.12) | Suitable for background research; not a sub-100 ms interactive guarantee |
| Fork logical step 2, copying logical steps 0–1 | Stored function IDs are 1, 2, 3. `start_step=3` copied the first two; `start_step=2` replayed the second and third. | PASS with index translation; literal zero-based assumption is false |
| Cancel synchronous active step | Active fake step finished; next step never started; status CANCELLED | PASS; cancellation is cooperative at the boundary |
| Retry fixtures | Seven-source pass: 18 naive calls versus 7 checkpointed calls. Whole iteration: 4 versus 2. Eleven unique domain events matched the golden export. | PASS after checkpointing |
| Reinsert a domain event independently | Unique deterministic ID retained one row | PASS |
| 1,000 one-step cells | 17.05 s (3.11), 17.20 s (3.12); logical DB growth 946,176 bytes in both runs | Bounded for the tested small outputs |

The ordinary queue includes 0.5-second simulated work and backlog delay. Its p95
below 2 seconds is acceptable for the research CLI/dashboard when the UI exposes
queued status. It is not a measurement of idle dispatch latency alone. Rate-limited
queue latency grows with the backlog by design and is not judged against that
interactive threshold. Polling faster also increases database work.

The global-cap and rate failures are consistent with a read-before-write race:
the installed SQLite adapter uses `isolation_level="IMMEDIATE"`, while the queue
claim reads counts before the first write. Python's legacy sqlite3 transaction
mode does not start that write transaction for a SELECT. This is a source-based
hypothesis, not a proven upstream root cause. The observed violations are enough
to reject unguarded multi-process limits without relying on that hypothesis.

## Volume and checkpoint size

Two worker processes shared one DBOS SQLite database and wrote events to a
separate domain SQLite database. An admission barrier released all 10 or 50
entries together; each event used its own transaction. All rows were present and
no `database is locked` or `SQLITE_BUSY` messages appeared in captured logs.
SQLite still serializes writes; zero logged errors does not mean zero lock waits.

Python 3.12 measurements:

| Concurrent entries | Events per entry | Rows | Observed active entries | Event-write p50 / p95 | Enqueue-to-start p95 |
|---|---|---|---|---|---|
| 10 | 10 | 100 | 10 | 0.235 / 4.894 ms | 0.723 s |
| 10 | 500 | 5,000 | 10 | 0.177 / 1.669 ms | 0.752 s |
| 50 | 10 | 500 | 50 | 0.215 / 42.613 ms | 0.805 s |
| 50 | 500 | 25,000 | 50 | 0.177 / 1.782 ms | 0.749 s |

The short 50-entry burst showed the largest p95 write delay; the longer run
spread more samples over time. Python 3.11 also completed all four corners with
10/50 active entries and all expected rows. The synthetic event payload is small;
these results do not characterize large evidence documents or network filesystems.

Checkpoint payloads of 1 KiB, 1 MiB, and 16 MiB round-tripped successfully. The
16 MiB round-trip took 0.064 s (3.11) and 0.074 s (3.12). SQLite reported a
1,000,000,000-byte `SQLITE_LIMIT_LENGTH`, which is a row/string/BLOB bound, not a
safe application payload budget. Serialization overhead and available memory can
fail earlier. The measured lower bound is **16 MiB supported**; no universal
pickle-size ceiling was established. Use a conservative application limit and
references for larger content rather than allocating near SQLite's limit.

## Answers to the open API questions

**Queue registration:** use `DBOS.register_queue()` after constructing/launching
DBOS. The installed 3.4.0 `Queue` constructor raises when called directly and
instructs callers to use registration/retrieval. The spike exercises persisted
registration from both worker processes. See the official
[queue methods](https://docs.dbos.dev/python/reference/contexts) and
[queue reference](https://docs.dbos.dev/python/reference/queues).

**Drain/deactivation for the CLI:** Python exposes
`DBOS.destroy(workflow_completion_timeout_sec=...)`, which stops background
workers and waits up to the timeout for active workflows. The installed class
has no public `drain`, `deactivate`, or `activate` method. `listen_queues()` is a
pre-launch selection API, not runtime deactivation. Therefore the CLI should
finish or cancel its known workflow handles and then destroy DBOS with a bounded
wait. A timeout can leave external work running without a checkpoint; shutdown
must report that limitation. This does not require an Admin API or hosted
service. See the [DBOS lifecycle reference](https://docs.dbos.dev/python/reference/dbos-class).

## Fixture limits and downstream requirements

The committed [golden export](../../tests/fixtures/dbos-sqlite/golden-grow.jsonl)
is produced by this spike's fake research workflows, with a loader and equality
check in `run.py`. It is a checkpoint/event regression fixture, not a claim that
Petri's production grow pipeline or a real model completed. Production end-to-end
runs remain required when the backend and phase runners are implemented.

The [workflow documentation](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
requires replay to preserve step ordering. Keep the release version explicit and
include a workflow-shape check in the production backend. Recovery behavior for
changed code, distributed/network filesystems, and cancellation of actual model
subprocesses was outside this spike. The next backend work must enforce the
single-owner/admission conditions and retest the observed failures at that seam.

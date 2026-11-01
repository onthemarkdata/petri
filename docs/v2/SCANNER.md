# Domain integrity scans

`petri scan` checks active edge endpoints, stored cell statuses, and event
envelopes and registered payload models. It exits 0 when clean and 1 when
findings exist. It records a `scan_completed` event containing the findings.
Research cells, edges, and prior events remain unchanged. The `--fix` and
`--loop` options and their implementation functions are retired.

Optional scheduling uses the same scan operation in a DBOS workflow:

```yaml
execution:
  backend: dbos
  scan_cron: "0 */6 * * *"
```

The schedule runs while the dish's DBOS owner is running. Five-field expressions
use minute granularity; six-field expressions put seconds first. Removing the
key pauses Petri's scan schedule. A runtime guard rejects a firing recovered
from a removed or changed cron setting. Manual scans do not need model access and do not start DBOS. They inspect the
domain database and append their result without opening or recovering execution
checkpoints, so a pending research workflow remains pending. A manual scan can
also run while the existing owner is active.

Workflow definitions register before launch. DBOS 3.4 requires its system
database to be initialized before `get_schedule` or `create_schedule`, so the
schedule is created immediately after `DBOS.launch()`. A pre-launch call fails
with `System database accessed before DBOS was launched`; this changes the
ordering proposed in #89. See the [DBOS scheduling guide](https://docs.dbos.dev/python/tutorials/scheduled-workflows).

A firing uses its durable workflow ID as the domain-event identity. Replaying
that firing returns its recorded findings without appending another event.
Both manual and scheduled scans emit a root `petri.scan` span with dish ID,
trigger source, and findings count. Scheduled scans never repair records.

## Former scanner categories

| v1 category | v2 disposition | Reason |
|---|---|---|
| Agent names | Retired | The agent registry and validated agent configuration own membership. |
| Verdict vocabulary | Retired | Typed agent outputs validate verdicts before persistence. |
| State-machine text | Retired | Typed graph nodes and DBOS own execution transitions. |
| Event types | Rewritten | Validate each persisted envelope and registered payload. |
| Queue schema | Retired | DBOS replaces the JSON queue; no generated queue schema remains. |
| Convergence text | Retired | Pipeline predicates consume typed verdicts. |
| Cell statuses | Rewritten | Reject unrecognized statuses in active cell rows. |
| Role separation | Retired | Agent specifications declare roles and blocking behavior. |
| Source hierarchy | Retired | Typed source/config validation owns allowed levels. |
| Documentation drift | Retired | Generated adapter prose is no longer an execution contract. |

The v2 scanner additionally reports active edges with missing endpoints. It
reads through `petri.query` and honors tombstone-backed active views when
present. Unknown or invalid event data is reported by event ID without copying
its research contents into the error description.

`test_scanner_v2.py` injects a phantom edge and malformed registered payload.
`test_scan_schedule.py` exercises the real SQLite-backed DBOS scheduler,
restart, event deduplication, root spans, and CLI exit codes.

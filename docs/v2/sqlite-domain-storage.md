# SQLite domain storage

`petri.storage.database` provides the v2 domain store introduced by issue #39.
The dish's `petri.sqlite` holds cells, directed dependency edges, and append-only
events. Durable-execution checkpoints belong in a separate database. This change
adds the storage seam; dish location and caller conversion follow in issue #40.

## Connections and migrations

`connect(path)` creates the parent directory if needed, enables WAL, foreign-key
checks, recursive triggers, and a 30-second busy timeout, then applies packaged
SQL migrations. It returns a stdlib `sqlite3.Connection`; callers must close it.
The connection uses explicit transactions (`isolation_level=None`).

Migrations run in filename order, with one `BEGIN IMMEDIATE` transaction per
script. The schema version is read again after taking the write lock, and
`PRAGMA user_version` advances in the same transaction as that script. A failed
migration rolls back its DDL and version change. A database newer than the code
raises `SchemaVersionError`, naming both versions; it is never downgraded.

## Events

`database.append_event(path, event)` validates the event payload and uses
`INSERT OR IGNORE` against its unique ID. It returns `(stored_event, inserted)`.
A duplicate returns the original event, including its original timestamp and
payload. SQL triggers reject event UPDATE and DELETE. Recursive triggers also
make replacement inserts run the deletion guard. These constraints protect
normal database writes; an administrator who changes the schema can remove them.

The table stores `id`, `cell_id`, `event_type`, `agent`, `iteration`, JSON `payload`,
and `created_at`. Reads reconstruct the existing envelope:
`{id, cell_id, timestamp, type, agent, iteration, data}`. Events may precede cell
creation during seed, so their `cell_id` is indexed without a foreign key.

`stable_event_id(cell_id, *identity)` computes SHA-256 over a compact UTF-8 JSON
array beginning with `"petri-event-v1"`. The cell ID and ordered string/integer
components follow. Array boundaries prevent concatenation ambiguity; integers
remain distinct from strings. The digest is prefixed by the cell ID. This
algorithm does not use process-randomized Python hashes or a presence scan.

`event_log.derive_event_id` fixes the execution identity order to
`(cell_id, round_number, node, agent, iteration, event_type, sequence)`, with an
additional `"execution-v1"` namespace after the cell ID. Round, iteration, and
sequence must be nonnegative integers; names must be nonempty strings. Timestamp
and payload are excluded, so a retry returns the original row even if a fresh
model call would produce different text. The caller must keep the coordinates
stable on retry, increase `round_number` for a new research round, and increase
`sequence` for distinct same-type events in one node/agent iteration.

SQLite append defaults are round 0, node `"default"`, and sequence 0. They identify
one event per cell/type/agent/iteration. A node that emits several sources or
searches must supply sequence numbers; otherwise those events intentionally
resolve to the same identity. Assign sequence numbers from stable workflow order,
not from the current row count. No application presence scan or index participates
in SQLite idempotency: the UNIQUE ID and INSERT OR IGNORE decide the winner. The
append operation then selects that ID to return the stored envelope.

## Compatibility and callers

Existing event helpers accept an explicit SQLite target:

```python
from pathlib import Path
from petri.storage.database import SQLiteEventLog
from petri.storage.event_log import append_event, load_events

target = SQLiteEventLog(Path(".petri/petri.sqlite"), cell_id="dish-colony-1-1")
append_event(target, "dish-colony-1-1", "search_executed", "researcher", 1,
             {"query": "example", "sources_found": 0},
             round_number=0, node="research", sequence=0)
events = load_events(target)
```

A scoped target filters reads to one cell and rejects writes for another cell.
An unscoped `SQLiteEventLog(path)` reads across the dish. `query_events` applies
its filters in SQL. Plain `Path` targets retain the JSONL implementation and
existing event-ID format, allowing the old engine to run until its callers switch.
Nondefault execution coordinates on a plain Path raise an error, so a legacy
append cannot silently imply durable idempotency. The legacy collision scan remains
only on the JSONL path until those callers migrate. No existing research files
are imported or deleted by this change.

`put_cell(path, mapping)` writes the schema columns directly, with defaults for
status, relation, atomicity, and bedrock reason. Callers supply `id`, `colony`,
`level`, `seq`, `claim_text`, and a nonempty `created_at`. `get_cell` and
`list_cells` return row dictionaries. Dependencies live in `edges`, where
`from_cell` depends on `to_cell`; the `dependents` view maps each dependency to
its dependent. `add_edge`, `read_edges`, and `read_dependents` expose that relation.

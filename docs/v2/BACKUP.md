# Back up a dish

`petri backup` snapshots the current dish with SQLite `VACUUM INTO`, then copies
its defaults. Use `--dish <id>` to select a dish, or `--list` to show snapshots
newest first with their cell count and database size.

Snapshots live under `.petri/backups/`. Queries never enumerate that directory.
The copy contains committed domain rows, even when another connection has an
uncommitted transaction. It does not include DBOS execution state.

Seed takes a snapshot before discarding draft rows during regeneration or abort.
Dashboard seed refuses to overwrite an existing colony. Any future overwrite
path must call `snapshot_before_destructive` before modifying rows. A failed
snapshot stops the destructive operation; the error includes a disk-space hint.

## Restore a snapshot

Run `petri backup --restore <snapshot-name>`. An existing dish requires `--force`.
Restore checks the snapshot's integrity, replaces only its named dish database,
removes stale WAL/SHM files, and copies saved defaults back into place. It refuses
while a DBOS executor owns the project. Stop other writers before restoring.

The CLI warns when a DBOS system database is present: restoring domain data does
not restore workflow checkpoints. Run `petri check` before resuming execution.

Validation on 2026-11-01: snapshot and list commands succeeded on the 16-cell
local Gemma quality-smoke dish. Automated restore tests compare cells, edges,
and events, check sibling bytes and timestamps, and run `PRAGMA integrity_check`.

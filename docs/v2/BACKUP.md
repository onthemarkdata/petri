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

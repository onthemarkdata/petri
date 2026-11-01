# Durable replay fixtures

`double-spend.json` models the two retry shapes described in
[the field reports](../../../docs/field-reports.md): a seven-source research pass
repeated about 2.5 times, and a whole iteration repeated twice. The first fixture
uses 7 + 7 + 4 fake calls (18 total); the second uses 2 + 2. These are synthetic
regression inputs, not copied production logs or measured production costs.

`golden-grow.jsonl` is the sorted text export from the spike's two deterministic
fake-research workflows. It contains source results and a final PASS event for
each workflow. The fake verdict is an explicit fixture value; it does not assess
real research quality. `load_golden()` in
[`spikes/dbos-sqlite/run.py`](../../../spikes/dbos-sqlite/run.py) checks unique,
sorted event IDs and loads the export. The `replay` scenario compares actual
SQLite rows with these events after repeated workflow and event submissions.

The crash scenario separately demonstrates why an external call can repeat if
the process dies after the call's side effect but before its step checkpoint.
The golden fixture proves deterministic replay after checkpointing; it cannot
prove exactly-once external execution.

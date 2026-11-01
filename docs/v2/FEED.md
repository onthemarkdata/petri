# Feed evidence into a settled cell

`petri feed` shows settled cells and their transitive impact before submitting
work. Select cells interactively, or use `--auto-reopen` to submit all displayed
cells. `--colony` limits that selection. Dependents receive propagation flags;
feed preserves their statuses and does not enqueue them.

```sh
petri feed ./measurement.txt --colony example
petri feed https://example.org/report --auto-reopen
cat measurement.txt | petri feed - --auto-reopen
```

Ingestion is a DBOS step. Its workflow inputs contain a file path or URL; stdin
and raw text are staged as input files. Successful ingestion writes a complete,
content-addressed JSON document under the dish's `evidence/` directory. The
`evidence_fed` domain event stores the text, content hash, relative path, original
and new workflow IDs, and generation. The `cell_reopened` event records the new
round context. Checkpoints contain only references and small control results.
A missing evidence file can be rebuilt from its committed event after a domain
restore. Research checks its content hash before using it.

A new feed invocation fetches its source again. Equal extracted content for the
same cell attaches to the previously allocated round, including when duplicate
requests overlap. It does not create another concurrent fork. An interrupted
fed round can temporarily have status `NEW`; it remains selectable so rerunning
the same feed command can recover the earlier parent and attach to its round.
The CLI waits for recovered feed parents before closing its execution owner.
A crash during an
uncheckpointed fetch can repeat that fetch; the content-addressed file and
idempotent domain events prevent duplicate committed evidence for equal content.

## Fork boundary and fallback

On DBOS 3.4.0 with SQLite, feed forks a completed cell workflow at the stored
function ID of its first research step. Completed admission, synchronization,
and Socratic checkpoints before that ID are copied. Research and later phases
run again with the new round identity, date, and iteration budget. The first
new phase enforces application admission because the copied admission checkpoint
belongs to the original round.

The offline TestModel fixture recorded these IDs:

```text
r1: 1 admission, 2 sync_check, 3 socratic, 4 research, ... 9 finish
fork: start_step=4; copy IDs 1–3
r2: Socratic calls across both rounds=1; research calls=2
```

`start_step` is a stored function ID, not a zero-based list index. The resolver
uses phase metadata in step outputs and selects the earliest research attempt,
including retry shapes. The earlier [SQLite spike](../adr/ADR-M4-dbos-sqlite-spike.md)
verified that distinction independently. Older checkpoints without phase metadata,
or settled cells without a completed workflow, receive a full new round instead;
that fallback also repeats Socratic work.

The SDK rejects a fork ID that already exists. Petri's checkpointed fork wrapper
checks the stable new round ID first, so recovery can attach after a crash between
the SDK's fork commit and the wrapper's checkpoint. The single execution owner
and a local fork lock serialize duplicate submissions.

The real SIGKILL regressions cover both that window and a kill after publishing
the evidence file but before the ingestion checkpoint:

```sh
pytest tests/integration/test_feed_fork.py -m integration -q
```

Both windows resume to one evidence file and one new cell round. The tests use
local fake phases; no model service participates. The ordinary tests use
TestModel, inspect copied step outputs, check the actual research request, and
exercise duplicate requests, fallback, admission, and evidence larger than the
checkpoint limit.

## Ingestion errors

Failed URL fetches, missing or unreadable files, and PDF extraction failures
raise `IngestError` with the source kind and cause. PDF ingestion requires the
`petri[pdf]` extra. Feed prints the error and exits unsuccessfully. It does not
turn a failure message into evidence or reopen the cell after failed ingestion.

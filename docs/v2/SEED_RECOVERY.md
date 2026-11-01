# Resume an interrupted seed

Select `execution.backend: dbos` in the dish's `defaults/petri.yaml`. Seed reports
its selected backend at startup. Run `petri seed --resume --colony <name>` after
an interruption; add `--no-questions` to accept the completed draft. An approved
colony reports `nothing to resume` without resolving a model provider. A new
seed still refuses to overwrite an existing colony.

The first seed workflow uses the colony ID. Concurrent submissions of that ID
attach to the same execution. A rejected draft's regeneration uses a separate
attempt identity, so its responses cannot be reused for a different directive.

Each durable step permits one new typed model call. It journals the validated
output as a domain event before materializing children, then checkpoints a small
progress reference. Recovery rebuilds the worklist from committed events and
reuses recorded responses if a crash interrupted materialization. The graph,
claims, and model outputs stay in the domain database; DBOS checkpoints remain
below the 2 MiB bound.

Rate limits use durable sleep before another step attempt. Ctrl+C requests model
cancellation before detaching the runtime; SIGKILL relies on startup recovery.
Completed, recorded calls are not repeated. A process killed after a provider
returns but before its response event or checkpoint commits can repeat that
call. Provider idempotency is needed to guarantee exactly-once external calls
or billing.

Cells stopped by a depth or width ceiling remain classified as pending. Explicit
model stops have `cell_marked_atomic` events and a bedrock reason. A forward
migration removes the interim checkpoint table without changing cells, edges,
or events; its historical creation migration remains for existing databases.
The seed worklist and initial/expansion nodes do not import DBOS.

Run the local CLI interruption regression explicitly:

```sh
pytest tests/integration/test_seed_resume.py -m integration -q
```

It covers SIGKILL and Ctrl+C with fake typed calls. In both cases, the completed
call count matches one uninterrupted seed, and the final colony has no duplicate
cells, edges, or expansion events.

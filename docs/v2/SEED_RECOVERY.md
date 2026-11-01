# Resume an interrupted seed

Run `petri seed --resume --colony <name>` from the project. Add `--no-questions`
to accept the completed draft without an interactive approval prompt. An approved
colony reports `nothing to resume` without resolving a model provider.

Seed saves each validated model response as a domain event before writing its
children. After materialization, `SeedCheckpointStore` saves the graph and pending
nodes in the dish database. Resume reuses those responses if the process stopped
between the response event and the checkpoint. A missing or malformed checkpoint
is rebuilt from the response events. Regeneration uses a separate attempt number
so it cannot reuse a rejected draft's responses.

Completed, recorded calls are not repeated. A process killed after the provider
returns but before the response event commits can repeat that call; neither the
checkpoint nor the event log can prove that an unrecorded response completed.
Cells stopped by a depth or width ceiling remain classified as pending, while
explicit model stops have `cell_marked_atomic` events and a bedrock reason.

The checkpoint store also supports an explicit file directory for tests and
alternate runtimes. Only the store accesses checkpoint rows or fallback files.
The seed state and its initial/expansion nodes do not import DBOS.

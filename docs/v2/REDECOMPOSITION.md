# Re-decomposition proposals

A settled trigger can start the `petri.redecompose` workflow. Its identity is the
parent cell plus generation. Repeated submissions attach to that workflow.
The workflow reserves capacity, asks the typed five-whys agent for one expansion,
and publishes the resulting proposal. It never enqueues validation.

Before inference, the reservation checks the cell's round limit, atomic status,
and remaining dish budget. Concurrent proposals share the same transactional
budget. The reservation initially holds the per-parent maximum and releases
unused capacity when the actual children are stored. Rejected children continue
to count against the dish budget because they were persisted.

The model receives the triggering verdict, focused directive, weakest link, and
any flaw or caveat text. Evidence is passed into the workflow as paths relative
to the dish directory; its text is read inside the model step and is absent from
the workflow arguments. Critical-flaw requests require at least one `limits` or
`rebuts` child through the agent's output validator. Truncation preserves that
counterargument. These prompts still require validation during full live grow.

The validated response is journaled before graph materialization. Cells, edges,
proposal status, and the `redecomposition_proposed` event commit together. New
cells use `DEFER_CLOSED`, so the normal scheduler cannot validate them before an
approval. Existing research statuses and dependent cells are unchanged.

`test_redecompose.py` covers guards, reservation sharing, rollback, truncation,
FunctionModel correction, durable deduplication, and checkpoint size. The real
SIGKILL test interrupts execution after the response journal is committed; the
restarted process makes no second model call and publishes one proposal.
A crash between receiving a provider response and recording it can still repeat
that call. Neither DBOS nor the domain journal promises exactly-once provider
billing in that window.

The runtime proposal API is available through `petri.lifecycle.redecompose.propose` while the
dish's durable execution owner is running.

## Approve proposed sub-claims

Re-decomposition stays off unless `grow` receives `--redecompose-on`. Choose
`inconclusive`, `flaw`, `caveats`, or `stalled`, separated by commas. Full verdict
names are also accepted. The gate runs after the selected colonies settle and
shows each matching proposal's sub-claims, trigger, and guard result.

```sh
petri grow --redecompose-on=inconclusive,flaw
petri grow --redecompose-on=inconclusive --approve-all
petri grow --redecompose-on=inconclusive --reject-all
```

An interactive terminal prompts for each decision. A non-interactive run prints
a pending-proposals notice and returns without waiting for input. Setting
`lifecycle.auto_approve: true` bypasses the prompt for enabled triggers; it does
not enable detection by itself. Approval is currently CLI-only.

Approval commits the decision event, new child status, and deterministic child
round together. The next durable grow pass schedules those children. Repeating
approval has no effect, and an interrupted run can recover the committed work.
The gate does not reopen the parent or its dependents. Rejection leaves proposed
children deferred and records the decision. Both decisions appear on the grow
root span.

The source verdict identifies the proposal attempt. A rejected attempt does not
consume an accepted generation, so a later research round may propose a different
set of sub-claims at that generation. Already consumed verdicts do not trigger
another proposal when grow is repeated. The migration that enables these attempts
preserves existing proposals. Rejected cells still count against the dish budget.
Only a recorded convergence circuit breaker enables the stalled trigger; user
stops and stalls without that provenance do not.

When an enabled proposal remains reserved after a failed model step, the next
explicit grow invocation retries that recorded attempt. DBOS records step
exceptions, so resuming an errored workflow alone would replay the same error.
Petri uses the public [workflow rewind API](https://docs.dbos.dev/python/reference/contexts#rewind_workflow)
to restart at the failed step while preserving earlier checkpoints. Cancelled
attempts use resume. Failures outside a recorded step require inspection.

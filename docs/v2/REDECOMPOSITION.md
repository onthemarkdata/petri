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

The approval gate and grow controls are a separate change in #87. The runtime
proposal API is available through `petri.lifecycle.redecompose.propose` while the
dish's durable execution owner is running.

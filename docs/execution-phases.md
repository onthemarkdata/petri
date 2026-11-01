# Research phase adapters

`petri.execution.phases` supplies the Socratic, Research, and Critique callables
for `run_pipeline`. Construct `PhaseServices` with a domain database path, round
number, explicit date, validated `AgentConfig`, built roster, and Socratic agent.
The roster and every debate pairing validate before dependencies are returned.
Decision callables are separate dependencies until the decision-phase port lands.

Socratic questioning keeps the four v1 steps: clarify, challenge assumptions,
map evidence, and verify. Research and critique use the code-owned roster phases.
Critique calls `run_debate`, retaining actual turns from both participants under
one usage budget per debate. A budget failure preserves the partial exchange and
returns a reason for human review.

Each assessment writes a `verdict_issued` event followed by its `source_reviewed`
events. A retry looks up the deterministic verdict ID and reuses the committed
response, including after a crash between the verdict and source writes. A
completed debate is likewise reused. If the process dies after a model response
but before the verdict commit, another request remains possible; event identity
cannot guarantee exactly-once external inference.

The phase result contains only its cell ID, phase, iteration, and an optional
parking reason. Model output and evidence remain in `petri.sqlite`. Tests run the
canonical diamond through real typed agents backed by offline models, replay all
five cells without new events, and check that evidence larger than 64 KB does not
enter graph checkpoints.

`petri.execution.evidence.render_evidence(events)` rebuilds the evidence body
from event envelopes alone. It emits the same appended-section layout as the v1
research and critique formatters, including actual debate summaries. Exporters
can add a document header around this body. The phase adapters write no
`evidence.md` or `metadata.json` files. A progress callback receives typed phase
start/end and agent signals without depending on the CLI spinner.

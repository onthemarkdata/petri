# Petri Vision

> **Grow your own AI context.**
>
> Petri is an open, durable research runtime for turning probabilistic agents into reliable software systems. Its reference workflow reproduces frontier-style deep research with frontier or open models, preserves the result as reusable context, and provides a laboratory for developing entirely new agent orchestrators.

## The opportunity

Frontier AI products have demonstrated that models can conduct meaningful, multi-step research: they can search, compare sources, synthesize evidence, and produce useful reports. But the capability is usually delivered as a closed product experience.

The report is disposable. The execution is opaque. The underlying research state is difficult to inspect, resume, update, or embed in another application. The user cannot easily change the workflow, choose the model allocation, replay failures, compare orchestration strategies, or preserve the evolving body of evidence as durable software state.

At the same time, open models are becoming increasingly capable, but they are often evaluated as isolated chat systems. This misses a more important possibility: a smaller model placed inside the right harness, workflow, tool environment, and validation system may produce outcomes that approach frontier systems at a fraction of the cost and with substantially greater control.

Petri exists to pursue both opportunities.

## The vision

Petri will make deep research:

- **Durable** — research survives individual model calls, sessions, crashes, and reports.
- **Embeddable** — applications can invoke, inspect, resume, and update the research process through stable software interfaces.
- **Auditable** — claims, evidence, decisions, disagreements, model activity, and state transitions remain inspectable.
- **Model-independent** — the same workflow can run across local open models, hosted open models, and frontier providers.
- **Workflow-independent** — the default research method can be replaced with another typed micro-orchestrator without rebuilding the surrounding infrastructure.
- **Economically scalable** — inexpensive specialized models can handle routine work while stronger models and humans are reserved for the decisions where they add the most value.
- **Experimentally testable** — models, prompts, tools, and orchestrators can be compared under controlled conditions rather than judged by demos alone.

The long-term objective is not merely to imitate the interface of a frontier deep-research product. It is to separate **deep research as a capability** from the proprietary model product that currently delivers it.

## A different primary artifact

Most research tools produce a document. Petri produces a **persistent research state**.

That state includes:

- Claims and their logical dependencies.
- Evidence, source provenance, and freshness.
- Supporting and contradicting observations.
- Agent judgments and confidence.
- Disagreements, failed arguments, and unresolved questions.
- Human interventions and escalation decisions.
- Execution history, retries, model usage, and cost.
- The lineage needed to reproduce or revisit a conclusion.

A report is one rendering of this state. It is not the state itself.

This distinction allows research to become a living software asset. New evidence can reopen only the affected claims. A downstream application can query the current conclusion without rerunning the entire investigation. An auditor can trace a synthesis back to its evidence. A future agent can begin with curated context rather than repeat the same search from zero.

Petri turns research from a one-time answer into maintained infrastructure.

## The core thesis: system intelligence is not model size

Petri is built around a simple proposition:

> A sufficiently structured system can convert inexpensive model capacity into high-quality research outcomes.

A model inside Petri does not need to improvise the entire research process in a single response. The surrounding system supplies:

- A constrained role and objective.
- A typed state transition.
- Relevant claim and graph context.
- Search, browsing, retrieval, and analysis tools.
- Explicit evidence requirements.
- Deterministic output validation.
- Retry and recovery behavior.
- Independent criticism and adversarial review.
- Mechanical convergence checks.
- Human escalation when uncertainty remains material.

Capability therefore lives in the combination of:

```text
model + tools + harness + workflow + durable state + evaluation
```

This shifts part of the intelligence from a monolithic model into inspectable software.

The benchmark that matters is not whether a small open model is as capable as a frontier model in isolation. It is whether:

> **Petri plus an open or specialized model can produce research outcomes comparable to frontier deep-research systems while remaining local, durable, inspectable, adaptable, and economical.**

## Agent, harness, and micro-orchestrator

Petri follows a layered view of agentic software.

### Agent

An agent combines a model with external tools. The model provides probabilistic judgment; the tools allow it to act on the world, gather evidence, and complete work beyond text generation.

### Harness

A harness manages the model's operating environment: tool execution, sessions, message history, permissions, provider compatibility, context, and the observe-plan-act-verify loop.

Petri uses the harness as an execution boundary, not as the owner of the research method. The harness should make different models usable under a common contract while allowing Petri to retain control of domain state and orchestration.

### Micro-orchestrator

A micro-orchestrator is a small, domain-specific software package that encodes an expert's judgment about how a repeated workflow should be completed.

It sits between two unsatisfying extremes:

- A large general-purpose orchestration framework that supplies infrastructure but not domain taste.
- A Markdown skill or prompt that describes behavior but cannot reliably enforce execution, state, retries, or invariants.

A micro-orchestrator follows a deterministic process while permitting nondeterministic outcomes within individual steps. It specifies:

- Which tasks must occur.
- Which tasks can run concurrently.
- What context each task receives.
- What outputs are valid.
- Which failures are retryable.
- When a workflow should stop, branch, or escalate.
- What evidence is sufficient.
- What state must be recorded.

Petri's default research pipeline is one micro-orchestrator. The larger platform exists so that other experts can replace it with their own.

## Offloading taste

Traditional packages let developers reuse commodity implementation. Micro-orchestrators let experts reuse **decision processes**.

In a research workflow, taste includes judgments such as:

- What makes a claim appropriately scoped.
- What counts as direct versus indirect evidence.
- Which source types deserve greater weight.
- Which objections are material.
- When additional research has diminishing returns.
- When disagreement requires debate.
- When uncertainty warrants human review.
- What must be preserved for a future audit.

Petri encodes the hard, testable portions of this taste in typed workflows and deterministic validators. A specialized model can learn the fuzzy, recurring judgments that are difficult to express as ordinary code.

The workflow is the explicit specification of taste. A Petri-tuned model is its partially compiled execution policy.

## Petri's four product surfaces

### 1. Petri Research

Petri Research is the reference deep-research workflow.

It decomposes a thesis into a directed graph of claims, validates foundational claims before their dependents, gathers citation-backed evidence, invites specialist criticism, red-teams conclusions, mechanically checks blocking verdicts, and escalates unresolved cells to humans.

Its goal is not agent consensus. Its goal is a durable and inspectable chain from evidence to conclusion.

Petri Research competes with frontier deep-research products, but differentiates through persistence, model choice, workflow control, privacy, auditability, and incremental updating.

### 2. Petri Core

Petri Core is the stable infrastructure required to build reliable micro-orchestrators:

- Typed domain contracts.
- Model and harness abstractions.
- Durable execution and recovery.
- Append-only domain events.
- Queryable local state.
- Tool execution.
- Observability and cost attribution.
- Human escalation.
- Evaluation and replay.

Petri Core treats agentic workflows as distributed data systems. Independent tasks can fail. Deployment must be idempotent. Sequential and concurrent work must share a consistent source of truth. State transitions must be recorded by software rather than entrusted to an agent's memory.

The data moving through this system is context.

### 3. Petri Lab

Petri Lab is the experimental environment for models and orchestrators.

A developer can preserve the domain model, harness, durable execution, telemetry, source corpus, and evaluation suite while replacing the Pydantic workflow graph. The workflow becomes the experimental variable.

This makes it possible to compare:

- Large adversarial colonies against minimal three-agent systems.
- Debate against independent voting.
- Flat decomposition against hierarchical planning.
- Static pipelines against confidence-routed workflows.
- Homogeneous models against heterogeneous model teams.
- Shared research against independently gathered evidence.
- Fixed rounds against adaptive compute allocation.
- Frontier-only, open-only, and hybrid execution.

Petri is therefore an executable orchestration paper: a hypothesis about how agents should work can be implemented, replayed, measured, and distributed as software.

### 4. Petri Model

Petri Model is a compact open model trained specifically to execute Petri protocols.

Its objective is not to become a universal assistant. It should become exceptionally dependable at a narrow collection of behaviors:

- Typed output and tool-call correctness.
- Claim decomposition without uncontrolled graph expansion.
- Evidence selection and entailment judgment.
- Concise, material criticism.
- Calibrated confidence and abstention.
- Revision after counterevidence.
- Correct escalation when evidence is insufficient.
- Efficient operation under explicit token and reasoning budgets.

A local Petri model makes rigorous colonies economical. Routine tasks can execute on owned hardware while frontier models become optional teachers, evaluators, or escalation engines.

## Replaceable models, replaceable orchestrators

Petri should preserve clean boundaries between four independently changeable concerns:

```text
Petri platform
├── Domain state
│   ├── dishes
│   ├── colonies
│   ├── cells
│   ├── claims and edges
│   ├── evidence
│   ├── verdicts
│   └── events
│
├── Stable infrastructure
│   ├── model and harness adapters
│   ├── tools
│   ├── durable execution
│   ├── storage and queries
│   ├── observability
│   └── evaluation
│
├── Replaceable orchestration policy
│   ├── decomposition
│   ├── research topology
│   ├── debate topology
│   ├── convergence policy
│   ├── escalation policy
│   └── synthesis
│
└── Replaceable model policy
    ├── provider and checkpoint
    ├── role routing
    ├── reasoning budget
    ├── quantization
    └── fallback and escalation
```

This separation allows controlled experiments.

A researcher can change the model while preserving the workflow. A workflow author can change orchestration while preserving the models. A harness developer can compare runtimes without changing Petri's domain protocol. A fine-tuned model can be evaluated against its teacher under identical conditions.

Pi is strategically important because it normalizes model and agent execution behind a common harness boundary. Pydantic AI supplies typed model interactions. Pydantic Graph supplies explicit state-machine workflows. Durable execution protects long-running work from crashes, interruptions, and rate limits. Petri remains the owner of the workflow and domain state.

## Forkability is a feature

Petri is Apache 2.0 licensed so it can be copied, modified, and redistributed as the foundation for new micro-orchestrators.

A practitioner should be able to duplicate the repository, retain the infrastructure, and replace the Pydantic workflows with a new domain process. Examples may include:

- Scientific literature adjudication.
- Security threat investigation.
- Compliance-control testing.
- Incident root-cause analysis.
- Technical due diligence.
- Competitive intelligence.
- Clinical or policy evidence review with mandatory human gates.
- Data-quality incident response.

Initially, a complete repository fork is an understandable path because it permits unrestricted experimentation. Over time, Petri should also support versioned workflow packages that consume and produce common domain contracts.

The ecosystem should permit both:

- **Workflow plugins**, for orchestrators that reuse Petri's domain and execution semantics.
- **Full forks**, for systems that intentionally change the underlying data model, execution semantics, or governance.

## Event-sourced context

Agentic workflows are distributed computing systems.

Tasks are independent. Some are sequential and others concurrent. Any task can fail. Retries must not duplicate completed work. A new worker may need to continue after another worker stops. Agents should receive only the context relevant to their current decision.

Petri therefore treats domain events as the source of truth.

Each meaningful action is written as an immutable event by program logic. Current state is derived from that history. This architecture enables:

- Replay after failure.
- Complete audits.
- Idempotent execution.
- Selective recomputation.
- Compact context assembly.
- Reproduction of prior decisions.
- Comparison of model and workflow versions.
- Future training datasets with known lineage.

Event sourcing is not only a reliability mechanism. It is a memory architecture. Rather than asking an agent to retain an entire workflow in its context window, Petri can provide the latest state, relevant evidence, and permitted transitions.

## A model-training flywheel

The durable runtime naturally produces structured training data.

Every Petri execution can record:

- The model, role, and workflow version.
- The input claim and graph context.
- Available tools and permissions.
- Tool requests and observations.
- Raw and parsed model outputs.
- Schema-validation results.
- Retries and corrections.
- Downstream verdicts.
- Human interventions.
- Latency, tokens, and cost.

Successful Qwen-based Petri execution demonstrates that the protocol is not inherently dependent on a single frontier harness or model family. Those trajectories can become teacher data for a smaller Petri-native Nemotron model running on local hardware.

The training strategy should focus on role-level protocol competence rather than indiscriminately copying full colony transcripts. Core tasks include decomposition, research planning, evidence assessment, red-team criticism, adjudication, and bottom-up synthesis.

The improvement loop is:

1. Run Petri using open and frontier teacher models.
2. Capture both successful and failed trajectories.
3. Apply deterministic validation and downstream outcome labels.
4. Repair difficult cases with stronger models or humans.
5. Fine-tune a compact model on the normalized trajectories.
6. Re-run frozen Petri evaluations.
7. Route progressively more work to the specialized local model.
8. Preserve frontier models for novel, ambiguous, or high-impact escalations.

Over time, Petri accumulates not only research context but a dataset describing how to perform research within its own protocol.

## Avoiding model monoculture

One specialized model playing researcher, critic, and judge can become internally consistent while remaining systematically wrong.

Role prompts create behavioral diversity, but they do not guarantee epistemic independence. Petri should therefore support:

- Separate lightweight adapters for builder, skeptic, and judge roles.
- Different model families at critical checkpoints.
- Independent evidence gathering where appropriate.
- Deterministic validation outside every agent.
- Randomized seeds and bounded sampling diversity.
- Human review for consequential or deadlocked claims.

The purpose of multiple agents is not theater. Each additional role must demonstrate measurable marginal value.

## Evaluation before spectacle

A colony completing is not proof that its conclusion is correct. Agent agreement is not truth. A citation is not necessarily supporting evidence.

Petri must distinguish three kinds of validity:

1. **Process validity** — did the system follow the intended protocol?
2. **Evidence validity** — do the sources actually support or contradict the claims assigned to them?
3. **Conclusion validity** — is the resulting synthesis correct and useful?

The first can be strongly enforced through software. The second can be measured and partially automated. The third ultimately requires external evaluation and, in consequential settings, expert judgment.

Petri's benchmark suite should evaluate complete dishes rather than isolated prompts. Important measurements include:

- Claim and conclusion accuracy.
- Citation entailment precision.
- False-support rate.
- Important evidence and counterevidence coverage.
- Appropriate abstention.
- Duplicate-claim creation.
- Graph expansion and omission rates.
- Confidence calibration.
- Revision after criticism.
- End-to-end completion and recovery.
- Tokens, cost, time, and energy per validated claim.
- Human escalation and review burden.

The most important comparison is not whether Petri produces impressive output. It is whether a model-orchestrator combination produces **more reliable research per dollar, hour, and unit of human attention** than simpler alternatives.

Every proposed workflow should be compared against meaningful controls, including a strong single-agent researcher, a minimal researcher-critic pipeline, and available frontier deep-research products.

## Reproducible orchestration experiments

Every Petri result should carry enough lineage to be reproduced:

```json
{
  "petri_core_version": "...",
  "workflow_id": "...",
  "workflow_revision": "...",
  "model": "...",
  "model_revision": "...",
  "prompt_set": "...",
  "schema_version": "...",
  "tool_profile": "...",
  "source_snapshot": "...",
  "dish_snapshot": "..."
}
```

Workflow experiments should freeze or explicitly record:

- Input dishes and claims.
- Model checkpoints and quantizations.
- Prompt and schema versions.
- Source corpora or search caches where possible.
- Tool permissions.
- Token and wall-clock budgets.
- Sampling configuration.
- Human-reviewed reference judgments.

This allows useful claims such as:

> A confidence-routed workflow preserved evidence accuracy while reducing model tokens and human escalations relative to the baseline.

The objective is to turn orchestrator development from folklore into an empirical discipline.

## What Petri should outperform first

Petri does not need to surpass frontier products on every dimension immediately.

Its strongest initial advantages are:

- Durability.
- Auditability.
- Local ownership of state.
- Model portability.
- Privacy.
- Workflow control.
- Incremental updating.
- Failure recovery.
- Reproducible evaluation.

These are valuable even when a frontier model still provides stronger raw synthesis.

The path to adoption is to become the preferred system for research that must be maintained, embedded, inspected, or run inside controlled infrastructure. As specialized open models improve, Petri can progressively close the quality gap while retaining those structural advantages.

## Principles

### The graph is a knowledge asset, not agent exhaust

Petri should preserve concise, useful claims and evidence—not endless transcripts. Context must remain curated, queryable, and economical to consume.

### Deterministic where possible, probabilistic where necessary

Software should enforce schemas, state transitions, identity, idempotency, permissions, graph invariants, and evidence references. Models should handle semantic judgment, ambiguity, synthesis, and material criticism.

### Durable by default

A long-running research workflow should expect interruption. Recovery, replay, cancellation, and selective recomputation are product requirements rather than operational enhancements.

### Evidence over consensus

Agent agreement can inform a decision, but Petri should prefer traceable evidence and explicit uncertainty over forced convergence.

### Human escalation is a successful outcome

A system that correctly identifies its limits is more useful than one that always returns a verdict. Human review is part of the architecture, not an embarrassment hidden at the edge.

### Model pluralism

No provider or model family should own Petri's research state or workflow. Frontier models, local models, specialized models, and human experts should be routable resources.

### Orchestration is intellectual property

The workflow embodies domain judgment and should be versioned, testable, distributable, and comparable like any other software artifact.

### Zero-infrastructure should remain a serious target

Petri should be capable of sophisticated, durable execution on a single machine with local storage. External infrastructure may be optional, but it should not be a prerequisite for using the system meaningfully.

### Open experimentation

The default workflow is a baseline, not dogma. Petri should invite forks, alternative workflows, competing models, and published comparative results.

## Non-goals

Petri is not intended to be:

- A generic replacement for every agent framework.
- A claim that multiple agents are always better than one.
- An autonomous authority for legal, medical, financial, compliance, or safety-critical decisions.
- A mechanism for turning consensus into truth.
- A transcript archive that mistakes volume for context.
- A workflow that requires the largest available model for every step.
- A closed benchmark optimized around Petri's own generated data.

Petri should remain focused on durable, evidence-oriented workflows where explicit state and orchestration provide measurable value.

## The destination

The mature Petri experience should be simple at the surface:

```bash
petri init
petri seed "A consequential claim worth investigating"
petri grow
petri feed new-evidence.pdf
petri report
```

Underneath that interface, Petri should be able to:

- Run on a compact local Petri model.
- Route difficult cells to another open or frontier model.
- Resume after interruption without repeating completed work.
- Expose the full research state to another application.
- Reopen only the claims affected by new evidence.
- Show exactly why a conclusion changed.
- Compare the active orchestrator against a baseline.
- Export a report without discarding the graph that produced it.

For builders, the system should make a different workflow equally approachable:

```text
fork or install Petri Core
replace the typed workflow
retain durable state, tools, telemetry, and evaluation
run the same benchmark dishes
publish the resulting micro-orchestrator
```

## The enduring idea

Petri is built around three separations:

1. **Research from reports** — persistent state instead of disposable output.
2. **Research capability from model vendors** — an open protocol instead of a proprietary product.
3. **System intelligence from model size** — orchestration, tools, and specialization instead of applying the largest model everywhere.

These separations reinforce one another.

The durable runtime generates training data. The specialized model lowers execution cost. Lower cost makes broader and more adversarial research practical. Replaceable workflows allow orchestration strategies to improve independently. A shared evaluation system reveals whether any of those changes actually produce better research.

Petri's ultimate ambition is therefore larger than one research application:

> **Petri is an open laboratory for converting expert judgment into durable agentic software. It provides the state, execution, evidence, and evaluation infrastructure needed to reproduce frontier-style deep research with open models—and allows developers to replace the reference workflow with their own typed micro-orchestrator.**

Orchestration is not merely glue around a model. It is an independently developable, testable, and distributable form of intelligence.

## Further context

- [Petri repository](https://github.com/onthemarkdata/petri)
- [Petri v2 migration plan](https://github.com/onthemarkdata/petri/blob/main/docs/v2/MIGRATION_PLAN.md)
- [Blog: Agent & Harness & Micro-Orchestrator, Oh My!](https://scalingdataops.substack.com/p/agent-and-harness-and-micro-orchestrator)

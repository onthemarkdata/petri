# Petri Constitution

## Core Principles

### I. Opinions Are the Product
Petri ships with opinionated defaults — 13 agent roles, 4 debate pairings, convergence criteria, a source hierarchy, and this constitution. `petri init` installs those defaults; users select their inference provider and customize behavior in `petri.yaml`.

### II. Append-Only Truth
Research events are immutable rows in the dish's `petri.sqlite` database, validated by Pydantic at write time. The same database stores cells, edges, evidence, and verdict history. DBOS keeps execution checkpoints in a separate `dbos.sqlite`; those checkpoints do not replace research evidence. Text exports are derived views.

### III. Mechanical Convergence
Convergence is a boolean check: all blocking verdicts in their pass set. No LLM judges convergence — typed verdict strings determine it mechanically. This is non-negotiable.

### IV. Adversarial by Default
The Red Team builds the strongest case against each node — independent of the original research. The Evidence Evaluator weighs all evidence neutrally. Separating advocacy from judgment is the design.

### V. Autonomous Agents, Human Merge
Agents run the validation pipeline through durable workflows within configured limits. Humans approve seed plans and proposed decomposition changes unless they explicitly enable automatic approval. Agents recommend research outcomes; humans decide how to use them.

### VI. Configurable Inference
Typed Pydantic AI agents use the harness and model selected in `petri.yaml`. Petri supports first-party Pi and Claude Code adapters and configured Pydantic AI providers. Claude Code is required only when selected. Agents use the tools registered for their current role and phase; they must not claim tool access or completed retrieval that the runtime has not provided.

### VII. Simplicity as Default
Local SQLite databases hold research data and durable execution state. Petri does not require an external database, message broker, or Docker. Model services and telemetry exporters are optional integrations chosen by the user.

### VIII. First-Principles Decomposition
Claims decompose from foundational assumptions upward using iterative questioning. Each premise is examined at its root — not borrowing credibility from higher-level claims. The Socratic method (clarify, challenge assumptions, seek evidence, consider alternatives, examine consequences) is the structural backbone of the agent pipeline, not an afterthought. When convergence fails, the decomposition itself is questioned — not just the evidence. Agents are explicitly anchored to first-principles roles: the investigator seeks evidence for irreducible claims, the skeptic challenges assumptions, the pragmatist examines consequences, and the triage agent questions whether the right questions are being asked.

## Governance

- This constitution supersedes ad-hoc decisions.
- Lead agents (Decomposition Lead, Cell Lead, Red Team Lead) re-read this document every iteration.
- Amendments require written rationale.

## V2 amendment rationale

The 2026-11-01 amendment aligns these instructions with the implemented v2 runtime.
The former JSONL event-log, file-queue, and mandatory Claude Code instructions contradicted the
SQLite/DBOS execution path and configurable inference adapters. The human-gate
description now includes seed approval and re-decomposition proposals. Mechanical
convergence, adversarial review, and first-principles decomposition are unchanged.

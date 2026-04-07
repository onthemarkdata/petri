# Petri Constitution

## Core Principles

### I. Opinions Are the Product
Petri ships with opinionated defaults — 13 agent roles, 4 debate pairings, convergence criteria, a source hierarchy, and this constitution. Zero configuration beyond `petri init`. Users customize by editing `petri.yaml`, not by building from scratch.

### II. Append-Only Truth
Every agent action is logged as an immutable event in an append-only JSONL file per node. Events are validated by Pydantic at write time. The event log is the single source of truth for all research data. The workflow queue tracks state only — no verdicts, no evidence.

### III. Mechanical Convergence
Convergence is a boolean check: all blocking verdicts in their pass set. No LLM judges convergence — typed verdict strings determine it mechanically. This is non-negotiable.

### IV. Adversarial by Default
The Red Team builds the strongest case against each node — independent of the original research. The Evidence Evaluator weighs all evidence neutrally. Separating advocacy from judgment is the design.

### V. Autonomous Agents, Human Merge
Agents run the full validation pipeline autonomously. The queue drives execution. The only human gate is the final decision — agents recommend, humans decide.

### VI. Harness-Agnostic Core
The core library imports only stdlib + Pydantic. Adapters bridge to specific harnesses (Claude Code, etc.). `petri.yaml` is the configuration surface.

### VII. Simplicity as Default
JSONL files, JSON queue with file locking, SQLite as a disposable read index. No external infrastructure, no message queues, no Docker required.

### VIII. First-Principles Decomposition
Claims decompose from foundational assumptions upward using iterative questioning. Each premise is examined at its root — not borrowing credibility from higher-level claims. The Socratic method (clarify, challenge assumptions, seek evidence, consider alternatives, examine consequences) is the structural backbone of the agent pipeline, not an afterthought. When convergence fails, the decomposition itself is questioned — not just the evidence. Agents are explicitly anchored to first-principles roles: the investigator seeks evidence for irreducible claims, the skeptic challenges assumptions, the pragmatist examines consequences, and the triage agent questions whether the right questions are being asked.

## Governance

- This constitution supersedes ad-hoc decisions.
- Lead agents (Decomposition Lead, Node Lead, Red Team Lead) re-read this document every iteration.
- Amendments require written rationale.

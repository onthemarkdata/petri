# M2-agents — Issue Backlog

> Tracking epic for milestone **M2-agents**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Re-found Petri's 13-agent roster on pydantic-ai: structural facts (roster membership, phases, blocking semantics, debate protocol, convergence contract) become typed code, while petri.yaml keeps the user-editable content (instructions, verdict vocabularies, debate pairings, per-agent model) per locked decision D9. The roster question is settled: 13 agents (3 leads + 10 specialists); socratic_questioner is a pre-pipeline utility agent outside the roster enum. At the end of this milestone a user can list and validate the typed roster, and run any single specialist end-to-end on the new stack with schema-validated verdicts, source-hierarchy enforcement via ModelRetry, real token accounting, and hard cost caps — while the legacy v0.3.x pipeline keeps working untouched (strangler discipline).

**Shippable release.** A release exposing a new `petri agents` command group: `petri agents list` renders the typed 13-agent roster (phase, blocking mode, verdict vocab, model), `petri agents check` validates a dish's petri.yaml against the v2 contract with actionable errors, and `petri agents run <agent>` executes one specialist against a claim through pydantic-ai with a typed verdict, enforced source-hierarchy policy, and a real token-usage/cost report bounded by UsageLimits. Mechanical convergence and debate hand-off ship as fully unit-tested library code (exercised via TestModel/FunctionModel) ready for M4-dbos (the engine milestone) to wire in.

**Depends on milestones:** M1-harness

**Milestone risks:**
- pydantic-ai 2.9.0 API details in the plan were drawn from documentation summaries, not a source read — Agent kwargs (retries dict form, output modes, UsageLimits fields, output_validator signature) must be verified against the installed package at the start of the factory issue; a mismatch could ripple through the factory, debates, source-policy, and run issues.
- Structured output over CLI-based harness Models (pi default, Claude Code adapter from M1-harness) is unproven: ToolOutput may not work over a print-mode CLI transport and PromptedOutput may degrade verdict fidelity — flagged as an open question in the factory issue; if it turns into real work it may need a small spike issue pulled forward.
- The socratic_questioner disposition is settled (pre-pipeline utility outside the roster enum; roster = 13), but the YAML alignment must relocate — not delete — its content: the clarifying-questions seeding path (consumed by M3-decomposer) depends on it, and dropping the entry instead of moving it would silently lose the capability.
- Debates gain real LLM rebuttal calls that v0.3.x never made (the provider path is stubbed at reasoning/debate.py:197-252), so per-cell token cost strictly increases — the shipped `limits` defaults (request_limit=25 / total_tokens_limit=200_000 per cell run; request_limit=6 / total_tokens_limit=50_000 per debate) are the cost backstop and must stay prominently documented, consistent with the maintainer's cost-warning stance.
- Mechanical convergence is identity-critical ('preserve exactly' per the decision record); the port is guarded by a differential conformance suite against tests/unit/test_convergence.py, but the deterministic weakest-link ordering change (enum order vs YAML dict order) is a subtle behavioral clarification that must be called out in the ADR and release notes.
- v2 petri.yaml schema is a hard break with no migration path (D8): existing dish configs stop loading on the v2 path — release notes and `petri agents check` error messages must make the required edits obvious.
- M2 builds agents that M4-dbos wraps in DBOSAgent; DBOS constraints (unique agent names, construction before DBOS.launch, pickle-serializable deps, no run_stream inside workflows) are encoded as factory acceptance criteria now — if the DBOS-on-SQLite validation spike (D1) invalidates DBOS, the factory still stands since it depends only on pydantic-ai public API.
- M2 library code (debates event append, convergence EXECUTION_ERROR handling) is specified against the amended-D4 domain event log (the `events` table in petri.sqlite) whose schema and write seam land as an early M3-decomposer issue; M2 code must stay behind the event-log write seam so its unit tests (tmp_path) remain valid regardless of when the M3 schema issue merges relative to M2 wiring in M4-dbos.

---

## M2-agents.1 — Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster

**Size:** M · **Labels:** migration-v2, agents

Context: Today the entire agent contract is stringly-typed data. `AgentRole.blocking` is the string 'true'/'false'/'conditional' (petri/models.py:326), forcing a YAML bool-coercion hack (petri/analysis/convergence.py:42-43) and the membership test `role.blocking in ('true', True)` (convergence.py:87). Roster structure (is_lead, phase, blocking, scope, redirect_on_block) lives in petri/defaults/petri.yaml:100-325 with no validation: simplifier declares `blocking: false` yet `verdicts_block: [OVERCOMPLICATED]` (petri.yaml:250-252), and the YAML ships 14 agent entries (3 leads + 11 specialists incl. socratic_questioner) while CLAUDE.md and constitution.md say 13. Per D9, structure moves to typed code.

Settled roster disposition (locked decision — do not reopen): the roster is exactly 13 (3 leads + 10 specialists). socratic_questioner (the 14th YAML entry, petri/defaults/petri.yaml:309-325) is NOT a roster member — it becomes a pre-pipeline utility agent OUTSIDE the AgentName enum, serving the clarifying-questions step before decomposition.

Scope:
- New module `petri/agents/contract.py`: `AgentName` StrEnum (canonical 13-member roster, definition order = deterministic roster order), `BlockingMode` StrEnum (BLOCKING / NON_BLOCKING / CONDITIONAL), `Phase` enum (SOCRATIC, RESEARCH, CRITIQUE, POST_CONVERGENCE, LEAD), and a frozen Pydantic `AgentSpec` (name, display_name, phase, blocking_mode, is_lead, scope, redirect_on_block).
- Ship the canonical `AGENT_SPECS: dict[AgentName, AgentSpec]` registry in code, transcribed from petri/defaults/petri.yaml:100-325 (3 leads at :104-138; 6 blocking specialists at :142-149, :161-169, :179-186, :196-203, :213-220, :230-237; triage conditional + redirect DEFER_OPEN at :258-266).
- Align the shipped petri/defaults/petri.yaml to 13 roster entries: relocate socratic_questioner's content into a distinct pre-pipeline utility section (do not delete it — its clarifying-question capability is consumed by the seeding path and M3-decomposer) and record the disposition as settled in docs/ARCHITECTURE-V2.md.
- Ship the default verdict vocabularies as code StrEnums (one per roster agent, from verdicts_pass/verdicts_block in petri.yaml), satisfying D9's 'verdict enums as typed code'. Exhaustive YAML-vs-enum conformance testing is split into its own issue ('Add verdict-enum ↔ petri.yaml conformance tests for the 13-agent roster').
- Model validators enforcing invariants: BLOCKING and CONDITIONAL agents must have non-empty verdicts_block; CONDITIONAL requires redirect_on_block; leads are never blocking. The NON_BLOCKING-with-verdicts_block combination (simplifier ships `blocking: false` + `verdicts_block: [OVERCOMPLICATED]` today) stays representable here; whether it is legal is decided and documented in the agent-factory issue, which owns eager config validation — the contract validator implements whichever ruling lands there.

Out of scope:
- Building pydantic-ai Agents (factory issue), convergence port, YAML loader changes, exhaustive verdict-vocab conformance tests (own issue), any edits to legacy petri/models.py AgentRole (stays for the v0.3.x path during the strangler migration).

Touched: petri/models.py:320-340 (read-only reference), petri/analysis/convergence.py:42-43, :87 (reference), petri/defaults/petri.yaml:100-325 (source of transcription; socratic_questioner relocation at :309-325). New: petri/agents/__init__.py, petri/agents/contract.py, tests/unit/test_agent_contract.py.

**Acceptance criteria:**
- [ ] petri/agents/contract.py exists with AgentName, BlockingMode, Phase enums and frozen AgentSpec; `AgentSpec(...)` with a mutated field raises (frozen model verified by test)
- [ ] pytest test asserts exactly 3 specs have is_lead=True, exactly 6 have BlockingMode.BLOCKING, exactly 1 (triage) has BlockingMode.CONDITIONAL with redirect_on_block == 'DEFER_OPEN'
- [ ] AgentSpec validator rejects a BLOCKING spec with empty verdicts_block and a CONDITIONAL spec without redirect_on_block (both covered by tests); the NON_BLOCKING-with-verdicts_block case (simplifier) remains representable, with a test pinning current behavior and a comment pointing to the factory issue's ruling
- [ ] A default verdict StrEnum exists for every roster agent (spot-checked in unit tests; exhaustive YAML conformance is the separate conformance-tests issue)
- [ ] len(AGENT_SPECS) == 13 (test); socratic_questioner is not a member of AgentName (test); the shipped petri/defaults/petri.yaml has 13 roster entries with socratic_questioner relocated to a pre-pipeline utility section; the settled disposition is recorded in docs/ARCHITECTURE-V2.md
- [ ] No behavior change to the legacy pipeline: existing tests/unit/test_convergence.py still passes unmodified

---

## M2-agents.2 — Add verdict-enum ↔ petri.yaml conformance tests for the 13-agent roster

**Size:** S · **Labels:** migration-v2, agents, good-first-issue · **Good first issue** · **Depends on:** Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster

Context: Per locked decision D9 the agent contract is deliberately split across two sources of truth: default verdict vocabularies ship as code StrEnums in petri/agents/contract.py (one per roster agent, typed), while petri.yaml keeps the user-editable verdicts_pass/verdicts_block lists. Two sources of truth can drift, and drift here is dangerous: mechanical convergence is a pass-set membership check on exact verdict strings, so a single typo'd vocabulary entry silently changes convergence outcomes with no error. This issue ships the exhaustive conformance suite pinning the code enums to the shipped defaults. It was split out of the typed-contract issue so it stays a compact, self-contained, newcomer-friendly test task.

Scope:
- New tests/unit/test_verdict_conformance.py, parametrized over every AgentName in the roster: assert the agent's code StrEnum values exactly equal the union of verdicts_pass + verdicts_block in petri/defaults/petri.yaml (set equality checked in both directions), and that the pass/block partition matches (a verdict listed as passing in YAML must not be a blocking member of the enum, and vice versa).
- Failure output must be diff-style: name the agent and print the verdicts missing-from-YAML vs missing-from-enum so a contributor can fix drift without spelunking.
- Roster-membership conformance: the set of agent keys in the YAML roster section equals the AgentName values (13), and socratic_questioner is absent from the roster section (it lives in the pre-pipeline utility section per the settled disposition in the typed-contract issue).
- Parse the YAML directly (yaml.safe_load on the packaged defaults), not through the v2 loader, so the suite is loader-independent and stays valid across the v2 schema rewrite (verdicts_pass/verdicts_block keys persist in the v2 shape per the D9 stays-in-YAML table).

Out of scope: runtime cross-validation of user configs (owned by the v2 loader issue); changing any vocabulary content; conformance for custom user vocabularies (the loader validates those at load time).

Touched: petri/agents/contract.py (read-only), petri/defaults/petri.yaml (read-only). New: tests/unit/test_verdict_conformance.py.

**Acceptance criteria:**
- [ ] A parametrized test per roster agent asserts exact pass/block partition equality between the agent's code StrEnum and petri/defaults/petri.yaml; the suite is green on the shipped defaults
- [ ] Corrupting a vocabulary entry in a tmp-path copy of the YAML makes the corresponding test fail with a message naming the agent and the differing verdict strings (meta-test asserting the diff-style output)
- [ ] A roster-membership test asserts YAML roster keys == AgentName values (13 members) and that socratic_questioner does not appear in the roster section
- [ ] The tests parse the YAML with yaml.safe_load and do not import the v2 loader module (asserted; no dependency on the loader issue)

---

## M2-agents.3 — Ship v2 petri.yaml agent-content schema with a validated loader (no import-time constants, no silent fallback)

**Size:** M · **Labels:** migration-v2, agents, breaking-change, docs · **Depends on:** Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster · **Field issues:** #2

Context: config.py is a raw-dict loader: `load_config` is lru_cache'd and silently falls back to packaged defaults when the given path does not exist (petri/config.py:18-27, fallback at :21); ~15 accessors operate on raw dicts with .get() (config.py:140-240, short-circuit derivation at :208-240); and four module-level constants are frozen at import time (config.py:247-250), which petri/models.py:18 imports so QueueEntry.max_iterations (models.py:312) is baked from package defaults. This exact split-brain caused field issue #2 (see docs/field-reports.md): the decomposer read package defaults instead of dish config and 10 colonies were seeded wrong. PetriConfig/AgentRole/Debate (models.py:343-353) exist but are not the actual validation path. Per D9, YAML keeps instructions, verdict vocabularies, debate pairings, and per-agent model choice; structural keys move to code (issue: typed agent contract).

Scope:
- New module `petri/agents/config.py`: Pydantic-validated v2 schema for the agent-content sections: per-agent `AgentContent` (instruction, verdicts_pass, verdicts_block, optional `model` — any pydantic-ai model string per D3, e.g. 'anthropic:claude-sonnet-4-6' or an M1-harness harness string), `debates` list, `source_hierarchy`, `agent_tools`, top-level default `model`, `max_iterations`, and a new `limits` section (per-run and per-debate UsageLimits values: request_limit, total_tokens_limit, tool_calls_limit) supporting the milestone's cost-cap work. Shipped defaults are concrete: request_limit=25, total_tokens_limit=200_000, tool_calls_limit=10 per cell run; request_limit=6, total_tokens_limit=50_000 per debate.
- Loader contract: explicit path required; a nonexistent path RAISES instead of silently falling back; no lru_cache module global; no import-time constants — the loaded config object is passed explicitly (deterministic for future DBOS replay per D2).
- Cross-validation against the typed contract: every AgentName must have content; unknown agent keys error; verdict vocab for default-roster agents is checked against the code enums with a clear diff-style error (this mechanically replaces contradiction-scanner categories 1-2/6 for the v2 path; scanner itself untouched).
- Write the schema-assessment section of ARCHITECTURE-V2: what stays in YAML (instruction, verdicts_pass/verdicts_block, debates incl. pairings and purpose, source_hierarchy petri.yaml:49-75, agent_tools petri.yaml:28-33, model default petri.yaml:10-12, max_iterations petri.yaml:37) vs what leaves for code (is_lead, phase, blocking, scope, redirect_on_block per agent; rounds semantics — see debate issue) vs what is new (`model` per agent, `limits`). Update petri/defaults/petri.yaml to the v2 shape.
- Breaking change (D8: no compatibility): document that v0.3.x dish configs are not readable by the v2 loader.

Out of scope: touching legacy petri/config.py call sites (the v0.3.x pipeline keeps using it until M4-dbos); harness resolution (M1-harness); decomposer config keys (max_decomposition_depth etc. — M3-decomposer).

Touched: petri/config.py:18-27, :140-240, :208-240, :247-250 (referenced, not removed yet), petri/models.py:343-353, petri/defaults/petri.yaml (rewritten to v2 shape). New: petri/agents/config.py, tests/unit/test_agents_config.py.

**Acceptance criteria:**
- [ ] Loading a nonexistent config path raises a typed error (test asserts it does NOT fall back to packaged defaults)
- [ ] The shipped v2 petri/defaults/petri.yaml round-trips through the validated loader with zero errors, and the loaded object exposes typed AgentContent for every AgentName
- [ ] A config with a typo'd agent name, a missing roster agent, or a verdict string absent from the code enum fails loading with an error message naming the offending key (three tests)
- [ ] Structural keys (is_lead, phase, blocking, redirect_on_block) present in a v2 YAML are rejected with an error pointing to the ADR migration note
- [ ] No module in petri/agents/ reads config at import time (test imports the package with a broken PETRI config path and asserts success)
- [ ] `limits` section parses into values usable as pydantic-ai UsageLimits kwargs; shipped defaults are exactly request_limit=25, total_tokens_limit=200_000, tool_calls_limit=10 per cell run and request_limit=6, total_tokens_limit=50_000 per debate, documented inline in the YAML (test pins the numbers)
- [ ] ARCHITECTURE-V2 contains the stays/moves/new schema table for petri.yaml

---

## M2-agents.4 — Build the 13-agent pydantic-ai roster via an agent factory (structure from code, content from YAML)

**Size:** M · **Labels:** migration-v2, agents, harness · **Depends on:** Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster; Ship v2 petri.yaml agent-content schema with a validated loader (no import-time constants, no silent fallback)

Context: Today each agent invocation is a hand-built prompt through ClaudeCodeProvider.assess_cell (petri/reasoning/claude_code_provider.py:690-847) with three-tier regex JSON extraction and substring verdict parsing (_extract_json/_coerce_str/_parse_verdict, claude_code_provider.py:170-244), plus generated markdown agent files (petri/adapters/generators.py:185-234, _AGENT_DESCRIPTIONS at :124-182, lead constitution re-read instruction at :216-224). Per D9 a factory builds pydantic-ai Agents from the typed contract plus YAML content; per D3 any pydantic-ai model string is accepted, resolved through the M1-harness seam (pi default, Claude Code adapter, direct API). v1 validated config lazily inside individual calls, so a bad pairing or vocab typo surfaced mid-run after tokens were already spent — v2 validates everything up front.

Scope:
- New `petri/agents/outputs.py`: per-agent structured output models — AssessmentResult-shaped (petri/models.py:388-397) with `verdict` constrained to that agent's vocabulary (built dynamically as a Literal from the loaded YAML vocab so custom vocabularies stay configurable; defaults coincide with the code enums), `sources_cited: list[SourceCitation]` (models.py:359-385), summary/arguments/confidence.
- New `petri/agents/factory.py`: `build_agents(config) -> dict[AgentName, Agent]` constructing one pydantic-ai Agent per roster member with: unique `name` (required for later durable DBOS wrapping — DBOS-wrapped agents need unique, stable names; see docs/ARCHITECTURE-V2.md); `instructions` composed from constitution.md text + the agent's YAML instruction + verdict contract; model resolved from per-agent `model` override else the config default via the M1-harness model-resolver seam; `retries` for output validation. Because pydantic-ai `instructions` are re-sent on every run, the leads' 're-read the constitution every iteration' rule (generators.py:216-224; field learning: constitution re-reading prevents agent drift) becomes a mechanical guarantee instead of prompt hope.
- Eager config validation at startup: build_agents() validates the ENTIRE loaded config — all roles present, every verdict vocabulary well-formed against the contract, every debate pairing referencing roster members — before constructing any Agent and before any LLM call (v1's lazy in-call validation is the anti-pattern being retired).
- Faithfully model the full configuration surface: tri-state blocking (true/false/conditional plus redirect_on_block), fractional debate rounds (the shipped `rounds: 1.5` 'skeptic gets final word' encoding, normalized into the typed DebateProtocol of the debates issue), and phase as Optional[int] at the YAML boundary (shipped entries carry integer phases; leads may carry none), mapped onto the contract's Phase enum.
- Decide and document (ADR) whether verdicts_block on a NON_BLOCKING agent is legal: simplifier ships `blocking: false` with `verdicts_block: [OVERCOMPLICATED]` today. Either ruling is acceptable — advisory vocabulary that never gates convergence, or an illegal combination that eager validation rejects (with the shipped simplifier config fixed accordingly) — but the ruling lands here, and both the contract validator and eager validation must implement it consistently.
- Agents must be constructible as module-level/global objects without I/O at import (DBOS requires agents to be constructed before DBOS.launch()), taking config explicitly.
- Tests use TestModel/FunctionModel (documented pydantic-ai test doubles) — zero API calls: verdict outside vocab triggers automatic retry-with-reflection; valid output parses to the typed model.
- Open question (do not guess): which structured-output delivery mode works over CLI-based harness Models from M1-harness — pydantic-ai 2.9.x documents ToolOutput (default), NativeOutput, and PromptedOutput ('works everywhere'). The factory must make the mode selectable per model backend; verify against the installed pydantic-ai 2.9.x and the M1-harness pi backend before hardcoding a default.

Out of scope: wiring into engine/processor.py phase runners (M4-dbos); DBOSAgent wrapping (M4-dbos); debates (separate issue); deleting claude_code_provider.py or adapters/generators.py (legacy path stays live).

Touched (reference): petri/reasoning/claude_code_provider.py:170-244, :690-847; petri/adapters/generators.py:124-234; petri/defaults/constitution.md; petri/models.py:359-397. New: petri/agents/outputs.py, petri/agents/factory.py, tests/unit/test_agent_factory.py.

**Acceptance criteria:**
- [ ] build_agents() returns one Agent per AgentName with unique, stable names; a uniqueness test asserts no collisions
- [ ] build_agents() performs eager validation of all roles, verdict vocabularies, and debate pairings before any Agent is constructed: a config with a pairing naming a non-roster agent, or a role with a malformed vocabulary, fails at build time with a named error and zero model construction (tests)
- [ ] Tri-state blocking (incl. CONDITIONAL + redirect_on_block), `rounds: 1.5`, and integer or absent `phase` values all round-trip through the factory without error, verified against the shipped defaults (tests)
- [ ] The simplifier verdicts_block-on-non-blocking ruling is recorded in the ADR and enforced identically by the contract validator and the factory's eager validation (a test pins the ruling)
- [ ] Every agent has output verdicts constrained to its YAML vocabulary: a FunctionModel test emitting an out-of-vocab verdict shows the model receiving a retry prompt, and exhausting retries raises (not silently coercing) — asserted with capture_run_messages or equivalent
- [ ] A FunctionModel test emitting a valid payload yields a typed output with .verdict in the agent's pass-or-block set and typed SourceCitation entries
- [ ] Per-agent `model` override in YAML changes the model the Agent is constructed with (asserted without network, e.g. via TestModel injection/override)
- [ ] Lead agents' composed instructions contain the constitution text; a test asserts the constitution appears in instructions, not in persisted message history
- [ ] Importing petri.agents.factory performs no file/network I/O (import test with a broken cwd passes)
- [ ] The chosen structured-output mode and its verification against the installed pydantic-ai + M1-harness backends is recorded in the ADR (or the open question is escalated with findings)

---

## M2-agents.5 — Implement debates as programmatic agent hand-off with message_history, shared usage, and UsageLimits caps

**Size:** M · **Labels:** migration-v2, agents · **Depends on:** Build the 13-agent pydantic-ai roster via an agent factory (structure from code, content from YAML)

Context: v0.3.x debates are fake: mediate_debate (petri/reasoning/debate.py:61-113) only re-formats each agent's original output via static formatters whose provider branches are literal stubs — `if provider is not None: pass` (debate.py:197-200, :224-226, :249-252). 'Skeptic gets final word' is encoded as the magic float `rounds: 1.5` (petri/defaults/petri.yaml:80-95, Debate.rounds float at petri/models.py:335-340). The 4 pairings are mediated in _run_phase2, which KeyErrors if a pairing references a missing agent (petri/engine/processor.py:1227-1228). Field learning: agents can't message each other, so debates must be relayed and logged as debate_mediated events.

Scope:
- New `petri/agents/debates.py` implementing the documented pydantic-ai programmatic hand-off pattern: agent_a.run(...) then agent_b.run(..., message_history=result.all_messages()), then a real agent_a rebuttal turn when the protocol grants a final word — actual LLM rebuttals for the first time.
- Replace the 1.5-rounds float with a typed `DebateProtocol` (pairing: tuple[AgentName, AgentName], rounds: int, final_word: AgentName | None, purpose: str) in the typed contract; YAML keeps the pairings/purpose per D9, parsed by the v2 loader (legacy float accepted at the YAML boundary and normalized, given the shipped defaults use 1.5).
- Cost controls per the milestone mandate: one shared usage object across both debaters and UsageLimits from the config `limits` section (shipped per-debate defaults: request_limit=6, total_tokens_limit=50_000) enforced per debate; exceeding limits terminates the debate with a typed outcome, not an unhandled exception.
- Pairings are validated against the roster at load — delivered via the factory's eager startup validation; this issue supplies the typed DebateProtocol it validates against (kills the processor.py:1227-1228 KeyError class).
- Keep DebateResult/DebateExchange models and the debate_mediated event append (log_debate, debate.py:119-143): on the v2 path the event is appended to the domain event log (the `events` table in petri.sqlite) through the single event-log write seam — the append-only event log is sacred per D4 as amended (rows, never edited, never deleted; see docs/ARCHITECTURE-V2.md). Keep load_debate_pairings semantics (debate.py:35-55) in the v2 loader. Carry over get_held_messages' cross-iteration hand-off intent (debate.py:149-185) as retained message_history in the debate result, for M4-dbos to thread.
- Tests via FunctionModel scripting both debaters: assert turn order, that agent_b's run receives agent_a's messages, final-word turn occurs only when protocol says so, and usage accumulates across both agents.

Out of scope: wiring into _run_phase2/engine (M4-dbos consumes run_debate directly); mediator-as-lead-agent LLM summarization (keep the mechanical summary); deleting reasoning/debate.py (legacy path stays).

Touched (reference): petri/reasoning/debate.py:11 (unused math import), :35-55, :61-113, :119-143, :149-185, :197-252; petri/defaults/petri.yaml:80-95; petri/models.py:335-340; petri/engine/processor.py:1152-1250, :1227-1228. New: petri/agents/debates.py, tests/unit/test_agent_debates.py.

**Acceptance criteria:**
- [ ] run_debate() with FunctionModel doubles produces a DebateResult whose exchanges show A-present, B-respond-with-A's-message_history, and an A rebuttal turn iff final_word == A (asserted for skeptic/champion and skeptic/pragmatist protocols)
- [ ] A test asserts agent_b's model call actually received agent_a's prior messages (inspected via FunctionModel), not a re-formatted string
- [ ] Shared usage: total usage after a debate equals the sum of both agents' runs (asserted via the run results' usage)
- [ ] A debate configured with total_tokens_limit small enough to trip mid-exchange returns a typed limit-exceeded outcome and the partial exchanges; no unhandled UsageLimitExceeded escapes
- [ ] A pairing referencing an agent not in the roster fails at config load with a named error (no runtime KeyError)
- [ ] log_debate appends a schema-valid debate_mediated event to the domain event log (`events` table in petri.sqlite) through the event-log write seam, using the uniform envelope observed in all real dish data: {id, cell_id, timestamp, type, agent, iteration, data} (tmp_path test)
- [ ] rounds: 1.5 in YAML normalizes to DebateProtocol(rounds=2 turns for A... final_word=pair[0]) with the mapping documented in the ADR

---

## M2-agents.6 — Port source-hierarchy enforcement to a pure policy core plus @output_validator raising ModelRetry

**Size:** S · **Labels:** migration-v2, agents · **Depends on:** Ship v2 petri.yaml agent-content schema with a validated loader (no import-time constants, no silent fallback); Build the 13-agent pydantic-ai roster via an agent factory (structure from code, content from YAML)

Context: validate_terminal_sources (petri/analysis/validators.py:51-115) couples policy to raw JSONL reads via load_events, returns an untyped dict with a 'pass' key, and misleadingly names the strongest level 'highest_level' while computing min(levels) (validators.py:104-105). Its config loader has three overlapping fallback paths and an optional-yaml import guard (validators.py:13-16, :21-48). Enforcement today happens after the fact in the evaluation phase; the milestone mandate is to move it to the agent boundary as a pydantic-ai @output_validator (a documented pydantic-ai pattern) so the model self-corrects via ModelRetry. Separately, v0.3.x citation URL checking is warn-only, which allowed fabricated or missing URLs to flow into evidence (the failure class behind the 0.3.4 'force agents to web-search instead of fabricating URLs' release).

Scope:
- New `petri/agents/source_policy.py`: pure function `check_terminal_sources(sources: list[SourceCitation], minimum_terminal_level: int) -> SourcePolicyResult` (typed Pydantic result with passes: bool, strongest_level: int | None — fixing the inverted naming — and a human-readable detail). No file I/O; hierarchy config comes from the v2 loader (source_hierarchy stays in YAML, petri/defaults/petri.yaml:49-75, minimum_terminal_level: 4 at :50).
- Factory attachment: agents whose verdicts can drive a terminal decision — evidence_evaluator (EVIDENCE_CONFIRMS/EVIDENCE_REFUTES, petri.yaml:289-305; mapped to VALIDATED/DISPROVEN in _run_evaluation, petri/engine/processor.py:1422-1501) — get an @output_validator that raises ModelRetry when the output's verdict is terminal-driving but sources_cited contains no level-1..minimum source. Non-terminal verdicts (EVIDENCE_INCONCLUSIVE) pass through untouched.
- Tighten SourceCitation URL validation from warn-only to blocking at the typed-output layer: a citation with a missing or malformed URL raises ModelRetry (via a validator on SourceCitation in petri/agents/outputs.py and/or the @output_validator path), so the model self-corrects instead of the pipeline recording unusable citations. Warn-only behavior does not exist on the v2 path.
- Exhausted-retries behavior must be explicit and typed, not an unhandled UnexpectedModelBehavior at the call site. Open question for maintainer: on exhaustion, fail the run or degrade the verdict to EVIDENCE_INCONCLUSIVE? Current code has no equivalent (it just gates later); the ADR must record the choice.
- Keep validators.py untouched for the legacy path; the v2 policy is the port target only.

Out of scope: red-team/investigator citation-count policies (their vocabularies don't directly produce terminal statuses); event-log-reading wrappers (M4-dbos supplies sources from typed step outputs, not reads of the domain event log); freshness policy.

Touched (reference): petri/analysis/validators.py:13-16, :21-48, :51-115, :104-105; petri/defaults/petri.yaml:49-75; petri/models.py:359-385 (SourceCitation); petri/engine/processor.py:1422-1501 (terminal mapping reference). New: petri/agents/source_policy.py, tests/unit/test_source_policy.py; edits to petri/agents/outputs.py (URL validation) and petri/agents/factory.py (validator attachment).

**Acceptance criteria:**
- [ ] check_terminal_sources is pure (no filesystem access; property test or import-level assertion) and returns SourcePolicyResult with strongest_level == min(levels) and passes iff strongest_level <= minimum_terminal_level — parity cases ported from the current validators.py semantics including 'no sources' and 'sources without hierarchy_level'
- [ ] FunctionModel test: evidence_evaluator emitting EVIDENCE_CONFIRMS with only level-5/6 sources receives a ModelRetry reflection naming the policy; emitting the same verdict with a level-2 source passes on first try
- [ ] FunctionModel test: EVIDENCE_INCONCLUSIVE with zero sources does NOT trigger the validator
- [ ] A citation with a missing URL or a non-URL string triggers ModelRetry at the typed-output layer (FunctionModel test covers both cases); a well-formed http(s) URL passes — no warn-only path exists in v2 (blocking, unlike v0.3.x)
- [ ] Exhausted retries produce the documented typed outcome (per the recorded ADR decision), covered by a test
- [ ] minimum_terminal_level is read from the v2 config, and changing it in a test config changes validator behavior (no hardcoded 4)
- [ ] Legacy analysis/validators.py and its tests are untouched and still green

---

## M2-agents.7 — Port mechanical convergence to the typed contract with semantics preserved exactly

**Size:** S · **Labels:** migration-v2, agents · **Depends on:** Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster

Context: Mechanical convergence is a core identity feature and must be preserved exactly (locked decision: 'boolean verdict-set check, no LLM'). Current implementation: check_convergence folds verdicts last-write-wins per agent (petri/analysis/convergence.py:70-75), skips leads (:83-85), tests blocking via the string hack (:87), includes conditional agents in blocking_results with redirect entries (:97-98, :100-101), and converges iff all blocking pass AND missing_blocking is empty (:112-113). identify_weakest_link ranks missing over failing but depends on dict/YAML iteration order (:130-147, order noted at :143-145). evaluate_short_circuits redundantly recomputes check_convergence (:170) and reads rules derived by config.py get_short_circuit_rules (petri/config.py:208-240: redirect_on_block -> defer-type rules, CANNOT_DETERMINE -> needs_experiment). compute_circuit_breaker is pure relative counting (:201-211). ConvergenceOutcome.outcome is a bare str (petri/models.py:400-405).

Scope:
- New `petri/agents/convergence.py` operating on the typed contract: inputs are Verdict records plus the code AGENT_SPECS registry (no YAML re-load, no load_agent_roles legacy path convergence.py:25-45); BlockingMode enum replaces the string test; outputs are typed (ConvergenceDecision with a Literal/StrEnum outcome replacing the bare strings of models.py:400-405, typed per-agent entries replacing dict['passes']).
- Preserve semantics exactly: last-write-wins per agent, leads excluded, conditional agents counted as convergence voters with redirect on block, converged = all blocking-or-conditional pass and none missing, missing outranks failing for weakest link, short-circuit fires only when the triggering agent's verdict matches AND all other blocking pass AND none missing, relative circuit-breaker counting (iteration - cycle_start_iteration) >= max_iterations (field-validated pattern: humans can grant more iterations without resetting history).
- EXECUTION_ERROR handling: per the M1-harness fail-loud invariant, exhausted-retry harness failures are appended to the domain event log (the `events` table in petri.sqlite; see docs/ARCHITECTURE-V2.md) as an explicit EXECUTION_ERROR verdict (a domain event, not a crash). The v2 port must treat EXECUTION_ERROR as counting AGAINST convergence and never satisfying it: it is never a member of any agent's pass set, a blocking or conditional agent whose latest verdict is EXECUTION_ERROR blocks convergence, and it ranks as a failing (not missing) verdict for weakest-link purposes.
- Determinism fix, semantics-safe: weakest-link iteration order becomes the AgentName enum definition order (stable by construction) instead of YAML dict order; document in ADR.
- evaluate_short_circuits accepts an optional precomputed convergence result (removes the double computation at :170); short-circuit rule derivation moves from config.py:208-240 into typed code derived from AgentSpec.redirect_on_block + vocab (rules are structural, per D9).
- Conformance: port every case in tests/unit/test_convergence.py (leads excluded, latest-wins, non-blocking ignored, triage LOW_VALUE_DEFER redirect) to run against the v2 module; legacy module and tests stay untouched.

Out of scope: pydantic-graph node wrapping (M4-dbos); iteration bookkeeping storage (M4-dbos); any LLM involvement (none exists, none added).

Touched (reference): petri/analysis/convergence.py:25-45, :51-124, :130-147, :153-195, :201-211; petri/config.py:208-240; petri/models.py:400-405; tests/unit/test_convergence.py. New: petri/agents/convergence.py, tests/unit/test_agents_convergence.py.

**Acceptance criteria:**
- [ ] Every scenario from tests/unit/test_convergence.py has a v2 twin producing the same converged/weakest_link/redirect answers (conformance suite green)
- [ ] A differential test feeds the same randomized verdict sets to legacy check_convergence and the v2 port and asserts identical converged booleans and weakest-link agents (with roster order pinned to match)
- [ ] EXECUTION_ERROR counts against convergence and never satisfies it: a cell with all other blocking verdicts passing plus one EXECUTION_ERROR from a blocking agent does NOT converge; a test asserts EXECUTION_ERROR is absent from every agent's pass set; an EXECUTION_ERROR verdict ranks as failing (not missing) for weakest-link
- [ ] Weakest-link selection is deterministic under dict-key shuffling of the input verdicts (test permutes insertion order)
- [ ] Short-circuit parity: triage LOW_VALUE_DEFER with all others passing yields a defer-type ShortCircuit decision; investigator CANNOT_DETERMINE with all others passing yields needs_experiment; either with a missing blocking verdict yields none
- [ ] compute_circuit_breaker parity across boundary cases (iteration - cycle_start == max-1, == max, > max)
- [ ] No function in petri/agents/convergence.py performs I/O or touches YAML (import + call with pure inputs only)
- [ ] Outcome type is an enum/Literal — assigning an unknown outcome string fails validation (test)

---

## M2-agents.8 — Add `petri agents list` and `petri agents check` CLI commands

**Size:** S · **Labels:** migration-v2, agents, good-first-issue · **Good first issue** · **Depends on:** Define typed agent contract: AgentName, BlockingMode, Phase enums and frozen AgentSpec roster; Ship v2 petri.yaml agent-content schema with a validated loader (no import-time constants, no silent fallback) · **Field issues:** relates:#2

Context: M2 needs a user-visible surface (strangler rule: every milestone ships). The CLI uses a per-module `register(app)` pattern (e.g. petri/cli/grow.py:25-47) with shared bootstrap in petri/cli/_bootstrap.py (find_petri_dir at :40-48). Today there is no way to see the agent roster or validate a dish's petri.yaml without running the pipeline; config errors fail silently mid-run (the field issue #2 failure class — a stale-defaults config bug that mis-seeded 10 colonies; see docs/field-reports.md). This issue adds read-only introspection over the new typed contract and v2 loader.

Scope:
- New `petri/cli/agents_cmd.py` registering a `petri agents` Typer sub-app with two subcommands:
  - `petri agents list`: render the roster from AGENT_SPECS + loaded content — name, display name, phase, blocking mode, lead flag, pass/block verdicts, resolved model string. Plain-text table via existing petri/cli_ui.py primitives; --json flag for machine output.
  - `petri agents check [--config PATH]`: run the v2 loader + contract cross-validation against the dish's petri.yaml (default: .petri/defaults/petri.yaml via find_petri_dir) and print each violation with its YAML key; exit 0 when clean, exit 1 with violations, exit 3 when no .petri/ found (matching _bootstrap.py:40-48 convention).
- Register in the CLI assembly next to the existing 11 commands; add a README section (cost-neutral commands — no LLM calls, worth stating given the maintainer's cost-warning stance).

Out of scope: running agents (separate issue), mutating config, touching legacy `petri scan` (the scanner still lints the legacy .claude/ adapter output).

Touched: petri/cli/ (command registration assembly), petri/cli/_bootstrap.py:40-48 (reuse), petri/cli_ui.py (reuse), README.md. New: petri/cli/agents_cmd.py, tests/unit/test_cli_agents.py (Typer CliRunner-based).

**Acceptance criteria:**
- [ ] `petri agents list` in an initialized dish prints one row per roster agent with phase, blocking mode, and verdict vocab; `--json` output parses and contains the same agent count as AGENT_SPECS (CliRunner test)
- [ ] `petri agents check` exits 0 on the shipped defaults, exits 1 and names the offending key when the config has (a) an unknown agent, (b) an out-of-vocab verdict, (c) a structural key like `blocking:` (three CliRunner tests)
- [ ] Running outside a .petri/ directory exits with code 3 and the standard missing-dish message
- [ ] Neither subcommand spawns a subprocess or makes a network/LLM call (test asserts no provider/model construction)
- [ ] README documents both subcommands

---

## M2-agents.9 — Add `petri agents run <agent>` one-shot execution with real usage and cost caps

**Size:** M · **Labels:** migration-v2, agents, harness, observability · **Depends on:** Build the 13-agent pydantic-ai roster via an agent factory (structure from code, content from YAML); Port source-hierarchy enforcement to a pure policy core plus @output_validator raising ModelRetry; Add `petri agents list` and `petri agents check` CLI commands

Context: This is M2's end-to-end proof: one specialist running on the new stack against a real model. It also lands the first real token accounting — docs/field-reports.md records that v0.3.x cost tracking is agent-estimated, order-of-magnitude only, and that true token measurement would require harness-level instrumentation; pydantic-ai's RunUsage (per-run input/output/cache token counts, with accumulated usage checked against UsageLimits after each model response) provides exactly that instrumentation. The legacy analogue is the hand-built assess_cell path (petri/reasoning/claude_code_provider.py:690-847) with its retry/backoff and stderr-substring failure classification (:33-81) — none of which is reimplemented here.

Scope:
- Extend petri/cli/agents_cmd.py with `petri agents run <agent> --claim TEXT [--cell CELL_ID] [--model MODEL] [--max-tokens N] [--max-requests N]`: build the single named Agent via the factory, run it once against the claim text (or a cell's claim + evidence.md context when --cell is given, resolved via the existing colony deserialization used by petri/cli/check.py:100-110), print the typed verdict, summary, and sources, then a usage report (requests, input/output tokens) and — when the model resolver can price the model — an estimated cost line; otherwise print tokens only (no fabricated pricing).
- Enforce UsageLimits from the config `limits` section (shipped defaults: request_limit=25, total_tokens_limit=200_000 per run), overridable by flags; a tripped limit prints a clear over-budget message and exits non-zero with partial output.
- The source-hierarchy @output_validator must be active on this path (a terminal-driving verdict without qualifying sources visibly triggers self-correction; surface retry count in verbose mode).
- Cost warning before execution when running interactively, consistent with the maintainer's prominent-cost-warning stance; --yes to skip.
- Tests: CliRunner + TestModel/FunctionModel for verdict rendering, limit tripping, and validator retries; one optional marked-live test (skipped by default) against the configured default model.

Out of scope: multi-agent cell pipelines, debates from the CLI, DBOS wrapping, streaming display (event_stream_handler work belongs to M5-otel given the no-run_stream-in-DBOS-workflows constraint).

Touched: petri/cli/agents_cmd.py (extend), petri/cli/check.py:100-110 (reuse cell/evidence resolution), petri/agents/factory.py (entry point reuse), README.md (cost warning + usage docs). New: tests/unit/test_cli_agents_run.py.

**Acceptance criteria:**
- [ ] `petri agents run skeptic --claim '...'` with an injected FunctionModel prints a verdict from the skeptic vocabulary plus a usage block with non-zero request and token counts (CliRunner test)
- [ ] `--max-requests 1` against a FunctionModel scripted to require a retry exits non-zero with an over-budget message (test)
- [ ] Running evidence_evaluator with a scripted terminal verdict lacking level-1..4 sources shows at least one validator retry in --verbose output, and the final output obeys the policy (test)
- [ ] `--model` flag overrides the config model for this run (asserted via injected model resolver)
- [ ] Unknown agent name exits non-zero listing valid AgentName values
- [ ] No cost estimate is printed for models without pricing data — tokens only (test asserts absence of a fabricated dollar figure)
- [ ] Interactive invocation without --yes prompts with a cost warning; non-TTY skips the prompt (test via CliRunner input simulation)

---
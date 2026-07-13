# M1-harness — Issue Backlog

> Tracking epic for milestone **M1-harness**. See `docs/v2/MIGRATION_PLAN.md` for the roadmap and `docs/field-reports.md` for field-issue context. Storage follows the amended D4 (petri.sqlite domain store; text via `petri export`).

**Goal.** Ship Petri's new inference layer: a first-party harness abstraction built as pydantic-ai Model subclasses, with the pi harness (--mode rpc, LF-delimited JSONL) as the validated default backend, the Claude Code CLI ported as a supported adapter, and passthrough for any pydantic-ai provider string from petri.yaml. Users gain model-agnostic inference (D3/D5) without losing zero-API-key subscription auth, and the v1 engine keeps working unchanged — the new layer lands alongside and is reachable via an opt-in bridge, so a user can run `petri seed`/`petri grow` on pi, Claude Code, or a direct API provider today. docs/ARCHITECTURE-V2.md (the locked decision record, module-fate table, ADR index, and label semantics) and docs/field-reports.md (public index of field issues #2–#15) were seeded when this backlog was committed, and the tracking epics were created alongside these issues — so every issue may cite them instead of restating rationale; this milestone's first issue audits and finalizes the seeded architecture doc. The engine bridge lands in two parts (decomposition path first, then assess_cell/match_evidence + CLI wiring), each independently landable.

**Shippable release.** petri-grow 0.4.0: backend-selectable inference. A new `harness:` block in petri.yaml selects pi (new default for fresh dishes), Claude Code CLI, or any pydantic-ai provider string ('anthropic:...', 'openai:...', local); `petri inspect` preflights pi/Node availability; the existing v1 seed/grow pipeline runs end-to-end on whichever backend is selected via the opt-in HarnessInferenceProvider bridge, while dishes without a harness block behave exactly as v0.3.4. Offline TestModel/FunctionModel test infrastructure replaces CLI-subprocess mocking for all new code. Ships alongside docs/ARCHITECTURE-V2.md and docs/field-reports.md, the reference docs the whole v2 plan cites.

**Depends on milestones:** none

**Milestone risks:**
- pi protocol assumptions may not survive the spike (structured output reliability, session semantics, error surfacing) — the spike gates the transport and PiModel issues; fallback transports are pi's --mode json or -p per D5, and the transport seam keeps pi swappable (same discipline D1 demands for DBOS)
- pydantic-ai Model ABC signatures were researched from documentation, not source — each Model issue mandates verifying against the installed 2.9.0 package first; the 2.x line is young and may churn, so pin >=2.9,<3 and expect small adaptations
- Node.js dependency friction for Python-first users: flipping the init template default to pi can strand users without Node — mitigated by preflight checks with install guidance, the claude-code and provider-string backends remaining fully supported, and existing dishes (no harness block) behaving identically, which is why the resolver issue carries no breaking-change label
- pi is a young external project (earendil-works/pi): version-pin it in docs, capture protocol transcripts in the spike doc so regressions are diagnosable, and keep all pi-specific logic behind PiRpcTransport
- Bridged path loses WebSearch/WebFetch on backends without equivalent tools — fabricated-citation risk that v0.3.4 specifically hardened against (commit 7eee82b); M1 documents the gap per backend and M2-agents (toolsets) closes it
- on_progress streaming parity is deliberately partial in M1 — CLI spinner UX degrades to coarse progress when the bridge is enabled; acceptable for opt-in, must be resolved before the default engine switches
- Retry-policy stacking: backend-level retries must be configurable to zero before M4-dbos wraps calls in DBOS steps (disable provider retries under DBOS) — designed into petri/harness/retry.py now
- Verdict vocabulary constraint in the bridge depends on config-loaded verdict lists (config.py:180) — the known 13-vs-14 roster and simplifier verdicts_block inconsistencies in defaults/petri.yaml could surface as validation errors; treat any such failure as a config bug to fix in defaults (M2-agents owns the roster alignment), not to paper over in the bridge

---

## M1-harness.1 — Finalize docs/ARCHITECTURE-V2.md: ADR index, module-fate audit, and label semantics review

**Size:** S · **Labels:** migration-v2, docs · **Field issues:** relates:#2, relates:#3, relates:#4, relates:#5, relates:#6, relates:#7, relates:#8, relates:#9, relates:#10, relates:#11, relates:#12, relates:#13, relates:#14, relates:#15

**Context.** `docs/ARCHITECTURE-V2.md` was seeded mechanically from the migration plan's decision record (docs/v2/MIGRATION_PLAN.md §2–§4) when the v2 docs corpus was committed, so that every migration issue could cite it from day one. The milestone tracking epics, labels, and milestones were created by `scripts/create_v2_issues.py` at the same time. What remains is the human pass: verifying the seeded document against reality and completing the pieces a mechanical assembly cannot write.

**Scope.**
- Audit the module-fate table against the actual `petri/` tree: every top-level module has a fate (keep / port / replace / retire) and an owning milestone slug from the canonical set (M1-harness, M2-agents, M3-decomposer, M4-dbos, M5-otel, M6-storage, M7-lifecycle). Fix any drift.
- Write **ADR-0001** in full — the D4 storage amendment (single Petri-owned petri.sqlite per dish as domain truth; DBOS keeps its own separate system DB; text becomes a derived `petri export` artifact; stdlib sqlite3 only). It is currently only an index-skeleton entry.
- Verify the label-semantics section matches the label set as actually created on the repo (`breaking-change` = observable behavior change for existing users; `spike` = timeboxed investigation; `good first issue` = newcomer-safe, self-contained), name-for-name.
- Cross-link docs/field-reports.md entries from the decisions they motivated (e.g. the rate-limit findings from D2, the decomposition findings from D6).

**Out of scope.**
- No code changes under petri/ or tests/.
- No re-litigation of the locked decisions — this issue verifies and completes their documentation.
- No new ADRs beyond ADR-0001.

**Touched files.** docs/ARCHITECTURE-V2.md only.

**Acceptance criteria:**
- [ ] The module-fate table is verified against the real petri/ tree: every top-level module present in the repo appears with a fate and an owning milestone slug, and no listed module is absent from the tree
- [ ] ADR-0001 (the D4 storage amendment) is written in full and linked from the ADR index
- [ ] The label-semantics section matches the repository's actual label set name-for-name (breaking-change, spike, good first issue, size:S/M/L, area labels)
- [ ] Each decision D1–D10 that was motivated by field evidence links the relevant docs/field-reports.md entry
- [ ] git diff shows zero changes under petri/ and tests/

---

## M1-harness.2 — SPIKE: Validate pi --mode rpc end-to-end and document the Node.js dependency story

**Size:** M · **Labels:** migration-v2, harness, spike, docs

**Context.** D5 locks pi (github.com/earendil-works/pi) as Petri v2's default harness, driven via `--mode rpc` (strict LF-delimited JSONL over stdin/stdout). Before we build a transport and Model subclass on it, we must validate every protocol assumption with real transcripts — everything Petri currently hand-rolls against the claude CLI (subprocess mgmt, stream parsing, retry classification in petri/reasoning/claude_code_provider.py:33-167, 284-506) will be rebuilt on this protocol. Time-box: ~3 days.

**Scope.**
- Install pi (TypeScript/Node); record exact install commands, minimum Node version, and whether `npx` works for a Python-first audience with no global npm setup.
- Drive `pi --mode rpc` from a throwaway Python script: spawn subprocess, write LF-delimited JSON requests to stdin, read LF-delimited JSON responses from stdout. Capture full request/response transcripts.
- Validate and transcript each of: (a) plain prompt round-trip; (b) structured output — can pi return schema-conformant JSON reliably, or do we plan on pydantic-ai PromptedOutput + validation retry on top of plain text? (c) tool allowlists via `--tools` / `--no-tools` (enumerate available tools — is there a web-search equivalent to Petri's load-bearing WebSearch/WebFetch grants in defaults/petri.yaml agent_tools?); (d) session reuse/fork/resume across requests and its context/cost implications; (e) error and rate-limit surfacing — force an invalid model, an auth failure, and (if feasible) a provider 429; record what appears on stdout/stderr and whether transient vs permanent is classifiable; (f) `--provider anthropic|openai|google` and local-model switching; (g) `-p` print mode and `--mode json` as fallback transports.
- Answer explicitly (or mark unanswered): does pi emit token usage per request? does rpc mode stream partial output? is the rpc session strictly sequential (one in-flight request) or multiplexed?
- Write docs/spikes/pi-rpc.md: transcripts, a transient/permanent error classification table, the Node.js dependency story (install guidance, version floor, failure modes when Node is absent), and a go/no-go recommendation per capability. This doc is the contract for the transport, fake_pi stub, and PiModel issues.

**Out of scope.**
- No production code under petri/.
- No decision on flipping any default (that lands with harness resolution).

**Touched files.** None in petri/ (read-only reference: petri/reasoning/claude_code_provider.py). New: docs/spikes/pi-rpc.md, docs/spikes/pi_rpc_probe.py (throwaway probe script, committed for reproducibility, not packaged).

**Acceptance criteria:**
- [ ] docs/spikes/pi-rpc.md is committed and contains reproducible transcripts for: prompt round-trip, structured-output attempt, tool-allowlist run, session-reuse run, and at least two forced error cases
- [ ] A transient-vs-permanent error classification table exists, with the raw pi output that motivated each row
- [ ] The Node.js dependency story section states minimum Node version, at least one verified install path, and the observed behavior when Node/pi is missing
- [ ] Each of {structured output strategy, session strategy, error taxonomy, usage reporting, streaming} has an explicit go/no-go/unknown verdict; unknowns are listed as open questions for the transport and PiModel issues
- [ ] The probe script runs against an installed pi with a documented one-line invocation

---

## M1-harness.3 — Create petri/harness package with typed error taxonomy and shared retry/backoff policy

**Size:** S · **Labels:** migration-v2, harness

**Context.** Both new backends (pi, Claude Code CLI) and the provider passthrough need one shared failure vocabulary and retry policy. Today this logic lives inline in the legacy provider: stderr-substring classification `_is_transient_failure` (petri/reasoning/claude_code_provider.py:33-74 — brittle: the permanent marker 'model' anywhere in stderr, e.g. 'model overloaded', suppresses retries) and exponential backoff with jitter `_retry_delay_seconds` (claude_code_provider.py:77-81), with `_MAX_RETRIES = 2` (line 28). Real-run evidence (indexed in docs/field-reports.md) shows why stderr-only classification fails: when a provider rate limit hit mid-run, the claude CLI exited 1 with EMPTY stderr — the recorded failure string was literally "claude CLI exited 1. stderr: (empty)" — while the reset hint appeared on STDOUT ("resets 9pm (America/Los_Angeles)"). v1 saw nothing classifiable and burned all 3 retry attempts in ~5 seconds against a limit that would not reset for hours, then hard-failed the run; users had to wrap Petri in an external backoff script. A second v1 anti-pattern this package designs against is the 'silent PASS' bug: unparseable verdict output was silently coerced into the first valid verdict that matched as a substring. v2's fail-loud invariant: malformed harness output becomes a TYPED error, never a coerced value — and when retries are exhausted, the failure must still be recordable in the domain event log (events table) as an explicit non-passing EXECUTION_ERROR domain event, so the error types defined here must carry everything that write needs. This issue creates the new package and the shared core; it does NOT modify the legacy provider (strangler: copy-adapt, old code untouched).

**Scope.**
- New package petri/harness/ with petri/harness/__init__.py.
- petri/harness/errors.py: HarnessError base carrying an optional structured FailureDetail (exit_code, stderr_tail, stdout_tail, last_stream_event, partial_output — JSON-serializable and small, so consumers can persist it into the domain event log (events table) under the uniform envelope's data field ({id, cell_id, timestamp, type, agent, iteration, data})); TransientHarnessError; PermanentHarnessError; RateLimitedError(TransientHarnessError) carrying optional retry_after_seconds; AuthExpiredError(PermanentHarnessError) with an actionable message (long-running CLI-auth sessions can expire mid-run).
- petri/harness/retry.py: a RetryPolicy (max_retries, base delay, cap, jitter) as a plain dataclass + pure delay function, generalizing claude_code_provider.py:77-81; policy must be configurable down to zero retries (when DBOS step retries arrive in M4-dbos, backend-level retries must be disable-able to avoid double-retry stacking). The policy honors RateLimitedError.retry_after_seconds: when the reset lies beyond the policy's delay cap, it stops retrying and surfaces the error immediately instead of burning attempts (regression target: v1's 3-retries-in-~5s against an hours-long limit); M4-dbos later consumes retry_after_seconds to durable-sleep the workflow until reset.
- A parse_reset_hint(text) helper that extracts reset hints into RateLimitedError.retry_after_seconds from BOTH observed formats: relative "try again in N minutes" and clock-style "resets 9pm (America/Los_Angeles)" (compute seconds until the next 9pm in the named zone; clock injectable for deterministic tests).
- Classification is pluggable per backend and takes exit code, stderr AND stdout (the real rate-limit hint arrived on stdout with empty stderr); each backend supplies its marker sets / structured error codes; preserve the documented permanent-wins-over-transient precedence (behavior pinned today by tests/unit/test_claude_code_provider.py:336-514, which remain untouched and green).
- Add pydantic-ai>=2.9,<3 to pyproject.toml core dependencies (Python floor 3.11 is compatible; pydantic-ai needs >=3.10).

**Out of scope.**
- Any transport or Model subclass; any change to petri/reasoning/claude_code_provider.py or its tests; writing events to the domain event log / events table (consumers own that — the bridge issue wires it).

**Touched files.** pyproject.toml (dependency add only). Reference (unmodified): petri/reasoning/claude_code_provider.py:28, 33-81; tests/unit/test_claude_code_provider.py:336-514. New: petri/harness/__init__.py, petri/harness/errors.py, petri/harness/retry.py, tests/unit/harness/__init__.py, tests/unit/harness/test_errors.py, tests/unit/harness/test_retry.py.

**Acceptance criteria:**
- [ ] uv run pytest tests/unit/harness/ passes with no network, no claude CLI, no pi installed
- [ ] Retry delay test mirrors test_retry_delay_grows_exponentially (tests/unit/test_claude_code_provider.py:500): exponential base growth, jitter bounded, cap respected; a RetryPolicy with max_retries=0 performs exactly one attempt
- [ ] Classification reads stdout as well as stderr: a fixture with exit code 1, empty stderr, and stdout containing 'resets 9pm (America/Los_Angeles)' classifies as RateLimitedError with retry_after_seconds populated; permanent-wins-over-transient precedence preserved
- [ ] parse_reset_hint extracts 'try again in N minutes' as N*60 seconds AND 'resets 9pm (America/Los_Angeles)' as seconds-until-next-9pm in that zone under a frozen injected clock; returns None when no hint is present
- [ ] A RateLimitedError whose retry_after_seconds exceeds the policy's delay cap causes no further attempts — the error surfaces immediately (regression test for v1 burning 3 retries in ~5s against an hours-long limit)
- [ ] FailureDetail round-trips through JSON serialization and slots into the domain event envelope's data field shape ({id, cell_id, timestamp, type, agent, iteration, data})
- [ ] python -c 'import pydantic_ai' succeeds after uv pip install -e "."
- [ ] git diff shows zero changes under petri/reasoning/ and tests/unit/test_claude_code_provider.py; full existing suite green

---

## M1-harness.4 — Build TestModel/FunctionModel offline test harness for harness-backed code

**Size:** S · **Labels:** migration-v2, harness

**Context.** Petri's LLM boundary is currently mocked at two ad-hoc seams: the dict-returning FakeProvider (tests/conftest.py:21-112) and ClaudeCodeProvider subclasses stubbing `_ask` (tests/unit/test_claude_code_provider.py:30-49, 395-514); the real subprocess/streaming boundary is deliberately untested (tests/conftest.py:15-18: 'Anything that would require simulating the real claude CLI through monkeypatching is not automated'). pydantic-ai ships TestModel and FunctionModel for fully offline agent tests (FunctionModel scripts ModelResponse/ToolCallPart sequences) — the direct replacement for CLI-subprocess mocking. This issue builds the shared fixtures that the PiModel and ClaudeCodeModel test suites adopt (adoption is an acceptance criterion on those issues, not this one) and that every later milestone (M2-agents, M3-decomposer evals) will reuse. It is deliberately dependency-free: its core deliverables need nothing from the other harness issues — only the pydantic-ai dependency itself, whose pyproject line is owned by the harness-package issue; if this issue lands first, it adds pydantic-ai>=2.9,<3 to pyproject.toml itself (whichever lands first owns the line).

**Scope.**
- tests/harness_utils.py + shared conftest fixtures: factories for TestModel and FunctionModel pre-loaded with Petri-shaped canned outputs typed against the existing result models — AssessmentResult (petri/models.py:388-397), DecompositionResult (models.py:484-490), ClarifyingQuestion (models.py:493-498), EvidenceMatch (models.py:501-506).
- A scripted-sequence helper: build a FunctionModel that emits an N-turn sequence (e.g. schema-invalid payload, then valid) to test pydantic-ai's ModelRetry/validation-retry paths deterministically.
- Offline guarantee: an autouse guard (env flag or socket-blocking fixture) for the tests/unit/harness/ tree so no test can silently hit the network or a real CLI; document how to run the @pytest.mark.integration tier separately.
- FakeProvider stays in place for v1-engine tests (strangler; it dies with M4-dbos's retire-v1 work).

**Out of scope.**
- Deleting or refactoring FakeProvider or any existing v1 test; testing the v1 processor against the new fixtures.
- The fake_pi transport stub and tests/README.md (they have their own good-first-issue: "Add fake_pi test stub and tests/README.md three-tier testing pattern doc").
- Refactoring the PiModel/ClaudeCodeModel test suites to adopt these fixtures (that adoption is an acceptance criterion on those two issues).

**Touched files.** tests/conftest.py (fixture registration only — FakeProvider at lines 21-112 untouched); pyproject.toml only if this lands before the harness-package issue (pydantic-ai dependency line). New: tests/harness_utils.py.

**Acceptance criteria:**
- [ ] An example test builds an Agent with output_type=AssessmentResult against a scripted FunctionModel (bad payload then good) and asserts the retry count and final validated output
- [ ] TestModel-based smoke test produces a validated DecompositionResult without any custom scripting
- [ ] The autouse offline guard covers the tests/unit/harness/ tree and trips loudly if any test attempts network or real-CLI access; the @pytest.mark.integration escape hatch is documented in the fixture docstrings
- [ ] FakeProvider (tests/conftest.py:21-112) is byte-identical; full existing suite green

---

## M1-harness.5 — Add fake_pi test stub and tests/README.md three-tier testing pattern doc

**Size:** S · **Labels:** migration-v2, harness, good-first-issue · **Good first issue** · **Depends on:** SPIKE: Validate pi --mode rpc end-to-end and document the Node.js dependency story

**Context.** The pi transport and PiModel test suites need an offline stand-in for the pi process: an executable stub that speaks pi's `--mode rpc` protocol (strict LF-delimited JSONL over stdin/stdout) and can be scripted per test scenario. The protocol contract is docs/spikes/pi-rpc.md (the spike's committed transcripts) — everything a contributor needs is in that doc and this issue. Alongside the stub, tests/README.md documents the three-tier testing pattern that all v2 test code follows. One invariant the stub must support testing: v2 is fail-loud — a malformed frame from the harness must surface as a typed error, never be coerced (the motivating anti-pattern is a v1 bug where unparseable verdict output was silently coerced into a valid-looking verdict) — so the stub must be able to EMIT malformed frames on demand.

**Scope.**
- tests/fixtures/fake_pi.py: an executable Python stub speaking LF-delimited JSONL on stdin/stdout, scriptable per test (via a scenario file or environment variable): canned responses; in-band error injection; rate-limit injection with and without a reset-hint string; slow response (for timeout tests); mid-request process death; malformed-frame emission (a non-JSON line, a truncated JSON object).
- Register fake_pi as a shared pytest fixture in tests/conftest.py: the fixture yields the stub path plus a scenario-scripting helper usable from any test.
- tests/README.md documenting the three-tier pattern: tier 1 — TestModel for smoke/shape tests; tier 2 — FunctionModel for scripted multi-turn/retry behavior; tier 3 — fake_pi for transport framing; plus the @pytest.mark.integration escape hatch for real-harness runs and how to run each tier. Cross-reference tests/harness_utils.py (built by the TestModel/FunctionModel fixtures issue) for tiers 1–2.

**Out of scope.**
- The transport implementation itself (the pi RPC transport issue consumes this stub).
- The TestModel/FunctionModel factory helpers (own issue).
- Any changes to existing tests.

**Touched files.** tests/conftest.py (fixture registration only). Reference (unmodified): docs/spikes/pi-rpc.md. New: tests/fixtures/fake_pi.py, tests/README.md.

**Acceptance criteria:**
- [ ] fake_pi runs standalone: a documented one-line invocation writes a request to its stdin and receives the scripted canned response on stdout, LF-framed
- [ ] Scenario hooks exist and are exercised by unit tests: canned response, error injection, rate-limit injection carrying a reset-hint string, delayed response, mid-request death, and malformed-frame emission (non-JSON line and truncated JSON)
- [ ] The registered pytest fixture returns the stub path plus a scenario-scripting helper usable from any test module
- [ ] tests/README.md documents the three tiers with one concrete example each and the @pytest.mark.integration escape hatch
- [ ] Everything runs offline: no network, no Node, no pi required; full existing suite green

---

## M1-harness.6 — Implement pi RPC transport (LF-delimited JSONL over stdin/stdout)

**Size:** M · **Labels:** migration-v2, harness · **Depends on:** SPIKE: Validate pi --mode rpc end-to-end and document the Node.js dependency story; Create petri/harness package with typed error taxonomy and shared retry/backoff policy; Add fake_pi test stub and tests/README.md three-tier testing pattern doc

**Context.** The pi backend needs a transport layer below the pydantic-ai Model: spawn `pi --mode rpc`, frame LF-delimited JSONL both ways, manage a long-lived session, and map failures into the petri/harness error taxonomy. This is the replacement for the subprocess plumbing Petri hand-rolls today (`_ask`/`_ask_oneshot`/`_ask_streaming`/Popen management at petri/reasoning/claude_code_provider.py:327-506) — but for pi, and reusable. Protocol specifics (response framing/termination, in-band vs stderr errors, sequential vs multiplexed sessions) come from the spike's docs/spikes/pi-rpc.md; where the spike marked an item unknown, this issue must resolve it against a real pi install and update the spike doc. Fail-loud invariant: a malformed frame (non-JSON line, truncated JSON) is a protocol violation and must raise a typed HarnessError — never be skipped, coerced into response text, or partially parsed. The motivating anti-pattern is v1's 'silent PASS' bug, where unparseable model output was silently coerced into a valid-looking verdict.

**Scope.**
- petri/harness/pi/transport.py: PiRpcTransport class — binary discovery (PETRI_PI_BIN env var, then shutil.which('pi')); spawn `pi --mode rpc` with configured `--provider`/model/tool flags; strict LF framing (one compact JSON object per line; content newlines only inside JSON strings); write request / read response per the spike-documented protocol; per-request timeout (default aligned with the legacy 300s at claude_code_provider.py:374-412, configurable); graceful shutdown and kill-on-timeout; detect process death mid-request, raise, and support transparent restart of the session on next use.
- Concurrency: unless the spike proved multiplexing, enforce one in-flight request per transport with a lock.
- Error mapping: stderr, stdout, and in-band error payloads → TransientHarnessError / PermanentHarnessError / RateLimitedError via the pluggable classification from the harness core, using the spike's classification table; malformed frames raise a typed HarnessError carrying the offending raw line in its FailureDetail.
- All unit tests run against the fake_pi stub (tests/fixtures/fake_pi.py, built by its own issue); extend the stub's scenario hooks additively if a needed scenario is missing.
- One real-pi integration test marked @pytest.mark.integration, auto-skipped when pi is not on PATH.

**Out of scope.**
- pydantic-ai Model mapping (next issue); session fork/resume semantics beyond simple reuse (defer until M2-agents needs them); any engine/reasoning changes; creating fake_pi (own good-first-issue).

**Touched files.** Reference (unmodified): petri/reasoning/claude_code_provider.py:327-506; docs/spikes/pi-rpc.md (update if unknowns get resolved). Modified (additive scenario hooks only, if needed): tests/fixtures/fake_pi.py. New: petri/harness/pi/__init__.py, petri/harness/pi/transport.py, tests/unit/harness/test_pi_transport.py.

**Acceptance criteria:**
- [ ] Offline pytest against fake_pi: request/response round-trip; injected transient error → TransientHarnessError; injected permanent error → PermanentHarnessError; injected rate-limit → RateLimitedError with retry_after populated when the payload carries a hint
- [ ] Framing test: a response whose JSON string fields contain embedded \n and \r\n round-trips intact (no CRLF/LF ambiguity)
- [ ] Malformed-frame test: fake_pi emitting a non-JSON line or truncated JSON raises a typed HarnessError whose FailureDetail carries the raw offending line — the frame is never skipped, coerced into text, or partially parsed (fail-loud invariant)
- [ ] Timeout test: slow fake_pi response raises within the configured timeout and the process is reaped (no zombie — asserted via returncode/poll)
- [ ] Mid-request process death raises, and a subsequent request on the same transport transparently restarts the session and succeeds
- [ ] Integration test (skipped without pi on PATH) completes one real round-trip against `pi --mode rpc`
- [ ] No existing files modified except optional spike-doc updates and additive fake_pi scenario hooks; full existing suite green

---

## M1-harness.7 — Implement PiModel: pydantic-ai Model subclass backed by the pi RPC transport

**Size:** M · **Labels:** migration-v2, harness · **Depends on:** Implement pi RPC transport (LF-delimited JSONL over stdin/stdout); Build TestModel/FunctionModel offline test harness for harness-backed code

**Context.** D3/D5: Petri's inference layer becomes first-party pydantic-ai Model subclass(es), with pi as the default harness. Subclassing the abstract pydantic_ai.models.Model is the supported, documented path for custom providers (reference implementation: OpenAIChatModel) — but that finding is doc-level, so the FIRST task in this issue is to read the installed pydantic-ai 2.9.0 pydantic_ai.models.Model source and record the exact abstract methods/signatures in the PR description. Do not invent APIs beyond that.

**Scope.**
- petri/harness/pi/model.py: class PiModel(pydantic_ai.models.Model) wrapping PiRpcTransport. One transport (session) per PiModel instance; document concurrency expectations.
- Request mapping: translate the pydantic-ai message history/request into the pi rpc payload per the spike contract; provider/model selection maps to `--provider anthropic|openai|google` / local at process start.
- Tool policy: constructor arg for allowed tools mapping to `--tools`; default `--no-tools` (Petri's agents get their research tools re-established in M2-agents; keep M1 conservative).
- Structured output: per the spike recommendation. Baseline assumption (works everywhere per pydantic-ai docs): pi returns text and pydantic-ai PromptedOutput + validation-retry produces typed results. If the spike proved a native schema mode, support it behind the same interface. OPEN QUESTION to carry in the issue: which pydantic-ai output modes the Model should advertise (verify against the installed Model ABC's capability surface — do not guess).
- Usage reporting: populate token usage from pi response metadata if the spike found it; otherwise report zeros and document the limitation. Token/cost capture must ultimately happen automatically at this layer: in ~2,700 events from a real prototype run, exactly ONE token_usage event was self-reported by agents despite full schema, API, and dashboard support — self-reporting does not work, this Model IS the instrumentation point, and M5-otel consumes it.
- Retry: wrap transport calls with petri/harness/retry.py policy (default mirrors legacy _MAX_RETRIES=2, claude_code_provider.py:28), configurable to zero.
- Tests adopt the shared TestModel/FunctionModel fixtures from tests/harness_utils.py and the shared fake_pi fixture; no canned-output literals duplicated with other suites.

**Out of scope.**
- Streaming/partial output (unless the spike showed rpc-mode streaming is trivial — otherwise document as a limitation; M5-otel owns streaming UX); session fork/resume; any engine wiring.

**Touched files.** Reference (unmodified): docs/spikes/pi-rpc.md; petri/harness/pi/transport.py; petri/harness/retry.py; tests/harness_utils.py. New: petri/harness/pi/model.py, tests/unit/harness/test_pi_model.py.

**Acceptance criteria:**
- [ ] PR description records the verified pydantic_ai.models.Model abstract method signatures from the installed 2.9.0 package
- [ ] Offline test: an Agent constructed with PiModel over fake_pi, output_type set to a Pydantic model, returns a validated instance
- [ ] Offline test: fake_pi scripts a schema-invalid first response then a valid one; pydantic-ai's validation-retry loop recovers and the test asserts two transport round-trips occurred
- [ ] Offline test: fake_pi rate-limit injection surfaces as RateLimitedError after the retry policy is exhausted (and retries the configured number of times before that)
- [ ] Model name and provider are accessible on responses (asserted in test) so later OTel spans can attribute cost
- [ ] Test suite consumes the shared fixtures from tests/harness_utils.py and the shared fake_pi pytest fixture — no canned-output literals duplicated across the Model test suites
- [ ] Integration smoke test (skipped without pi): one structured-output round-trip against real pi
- [ ] Full existing suite green; no legacy modules modified

---

## M1-harness.8 — Port Claude Code CLI backend to a pydantic-ai Model subclass (ClaudeCodeModel)

**Size:** M · **Labels:** migration-v2, harness · **Depends on:** Create petri/harness package with typed error taxonomy and shared retry/backoff policy; Build TestModel/FunctionModel offline test harness for harness-backed code

**Context.** D5 keeps Claude Code CLI as a supported adapter (it carries the zero-API-key subscription auth story, D3). The legacy ClaudeCodeProvider (petri/reasoning/claude_code_provider.py, 897 lines) mixes transport with parsing; most of the parsing is made obsolete by pydantic-ai structured output. This issue ports ONLY the transport-worthy parts into a new Model subclass; the legacy provider stays untouched and remains the default path for the v1 engine (strangler).

**Scope.**
- petri/harness/claude_code/model.py: class ClaudeCodeModel(pydantic_ai.models.Model) (same verified-ABC discipline as PiModel).
- PORT: command construction `_build_claude_command` (claude_code_provider.py:284-325) including the `--allowedTools=` equals-form workaround for the variadic flag parser; one-shot `-p` transport `_oneshot_attempt` (claude_code_provider.py:374-412, 300s timeout); stream-json parsing `_extract_text_delta` (84-126) and `_process_stream_lines` (127-167) for streamed text assembly.
- REPLACE: the hand-rolled retry loops in `_ask_oneshot`/`_ask_streaming` (350-446) with petri/harness/retry.py + the shared error taxonomy. The transient/permanent markers from claude_code_provider.py:33-74 become this backend's classification config, extended to read STDOUT as well as stderr — real-run evidence shows rate limits surface as exit 1 with EMPTY stderr and the reset hint on stdout ("resets 9pm (America/Los_Angeles)"); v1's stderr-only classification burned all its retries in ~5 seconds against that hours-long limit.
- DO NOT PORT (obsolete under pydantic-ai): three-tier fenced-JSON extraction `_extract_json` (170-192), `_coerce_str` (195-210), `_parse_verdict` substring fallback (236-244) — the substring fallback is the v1 'silent PASS' bug (unparseable verdict output silently matched the first valid verdict substring), and the fail-loud invariant forbids porting it. Claude Code CLI has no provider-native structured-output channel, so this Model serves text and pydantic-ai PromptedOutput + validation-retry produces typed results; note this expected output mode in the module docstring.
- Auth: rely on the already-authenticated claude CLI; classify auth-expiry stderr as AuthExpiredError with an actionable message.
- Tests stub the subprocess boundary at the new transport seam, following the `_ask`-stub pattern from tests/unit/test_claude_code_provider.py:30-49 (which itself remains untouched), and adopt the shared TestModel/FunctionModel fixtures from tests/harness_utils.py (no canned-output literals duplicated with other suites).

**Out of scope.**
- Any modification to petri/reasoning/claude_code_provider.py or its tests; wiring into resolve_provider (separate issue); tool allowlist redesign (pass through allowed_tools as today, config.py:83).

**Touched files.** Reference (unmodified): petri/reasoning/claude_code_provider.py:28, 33-74, 84-167, 284-325, 350-446; tests/unit/test_claude_code_provider.py:30-49; tests/harness_utils.py. New: petri/harness/claude_code/__init__.py, petri/harness/claude_code/model.py, tests/unit/harness/test_claude_code_model.py.

**Acceptance criteria:**
- [ ] Offline test: Agent with ClaudeCodeModel over a stubbed CLI returns validated structured output; invalid-JSON first response then valid response exercises pydantic-ai validation-retry (asserted: two subprocess invocations)
- [ ] Command construction pinned by tests: model flag, --allowedTools= equals form, print vs stream-json mode selection
- [ ] Transient stderr (rate limit, 5xx, network) triggers retry per policy; permanent stderr raises PermanentHarnessError with no retry; 'model overloaded'-style text is classified transient (regression test for the legacy misclassification at claude_code_provider.py:45-52)
- [ ] Regression fixture from real logs: exit code 1, empty stderr, stdout containing 'resets 9pm (America/Los_Angeles)' → RateLimitedError with retry_after_seconds populated, and the retry policy does NOT burn its attempts against the hours-long reset
- [ ] grep confirms petri/harness/claude_code/ contains no port of _extract_json/_coerce_str/_parse_verdict
- [ ] Test suite consumes the shared fixtures from tests/harness_utils.py — no canned-output literals duplicated across the Model test suites
- [ ] git diff shows zero changes under petri/reasoning/; full existing suite green

---

## M1-harness.9 — Extract ClaudeCodeProvider prompt builders into petri/reasoning/prompts.py (behavior-preserving)

**Size:** S · **Labels:** migration-v2, harness, good-first-issue · **Good first issue** · **Field issues:** relates:#3, relates:#12, relates:#13

**Context.** All of Petri's prompt text — the domain IP the migration must carry forward verbatim — is inlined as f-strings inside ClaudeCodeProvider method bodies: assess_claim_substance (petri/reasoning/claude_code_provider.py:508-552), generate_clarifying_questions (554-587), the 4-step BRAINSTORM/PRIORITIZE/SELECT/EMIT decompose_claim prompt (589-640), the Five Whys decompose_why prompt with is_atomic escape hatch (642-688), the per-agent-role assess_cell prompt with freshness/WebSearch directive (690-847), and summarize_evidence (849+). These must be ported as pydantic-ai system/user prompts rather than rewritten. The upcoming bridge (and M2-agents) must share these with the legacy provider without divergence. Pure refactor: zero behavior change.

**Scope.**
- New petri/reasoning/prompts.py: one pure function per prompt (e.g. build_decompose_claim_prompt(claim, max_premises, clarifications, regenerate_guidance) -> str; build_assess_cell_prompt(...); etc.), producing byte-identical output to today's inline construction for identical inputs.
- Keep the JSON-emission instructions ('EMIT JSON on final lines', fenced-output directives) as SEPARATE composable suffix functions, so the pydantic-ai bridge can compose prompts without them (pydantic-ai owns output formatting there) while the legacy provider keeps appending them.
- ClaudeCodeProvider methods delegate to the new builders; no signature, retry, parsing, or subprocess changes.
- Snapshot tests pinning each builder's output for fixed inputs (these snapshots become the regression net when the bridge and M2-agents reuse the prompts).

**Out of scope.**
- Any prompt WORDING change (the maintainer's v1-v4 prompt experiments proved phrasing is fragile in both directions — the wording improvements catalogued in docs/field-reports.md, relates:#3/#12/#13, belong to M3-decomposer, not here); any change to parsing or transport; touching decomposer.py.

**Touched files.** petri/reasoning/claude_code_provider.py (method bodies delegate to builders; lines 508-552, 554-587, 589-640, 642-688, 690-847, 849+). New: petri/reasoning/prompts.py, tests/unit/test_prompts.py.

**Acceptance criteria:**
- [ ] All existing tests pass unchanged — in particular the prompt-content assertions in tests/unit/test_claude_code_provider.py (e.g. test_assess_cell_prompt_injects_todays_date_and_websearch_directive, line 73)
- [ ] Snapshot tests exist for every extracted builder with fixed inputs, including at least one with clarifications/regenerate_guidance populated (claude_code_provider.py:597-607 path)
- [ ] JSON-emission suffixes are separate functions; a test composes a prompt with and without the suffix and asserts the body is identical modulo the suffix
- [ ] No inline prompt literals remain in the extracted ClaudeCodeProvider method bodies (reviewer checklist item)
- [ ] InferenceProvider Protocol (petri/models.py:614-693) and all public signatures unchanged

---

## M1-harness.10 — Add harness resolution and pydantic-ai provider-string passthrough from petri.yaml

**Size:** M · **Labels:** migration-v2, harness, docs · **Depends on:** Implement PiModel: pydantic-ai Model subclass backed by the pi RPC transport; Port Claude Code CLI backend to a pydantic-ai Model subclass (ClaudeCodeModel) · **Field issues:** relates:#2

**Context.** D3: petri.yaml must accept pi, Claude Code, or any pydantic-ai model string. Today model resolution is hardcoded to ClaudeCodeProvider in petri/cli/_bootstrap.py:16-37 (resolve_provider reads model.name via load_dish_config and constructs ClaudeCodeProvider), and config loading has a known split-brain trap: load_config is lru_cached with silent fallback to packaged defaults (petri/config.py:19-27) and module-level constants frozen at import time (config.py:247-250) — the root cause of the field-validated failure where the decomposer ignored dish config because constants were frozen at import time (docs/field-reports.md #2). The new resolver must be an explicit, cache-free, dish-config-driven path. Legacy resolution stays byte-identical for dishes without the new config (strangler) — dishes without a harness block behave identically, so this is not a breaking change for existing users.

**Scope.**
- petri/harness/resolve.py: resolve_model(dish_config: dict) returning a constructed PiModel / ClaudeCodeModel, or a passthrough pydantic-ai provider string (e.g. 'anthropic:claude-sonnet-4-6', 'openai:...') handed to pydantic-ai's model inference untouched. Takes the config dict explicitly; never reads config at import time; never silently falls back to packaged defaults when a dish config exists.
- Config schema: a new `harness:` block in petri.yaml, e.g. `harness: {backend: pi|claude-code|provider, model: ..., provider: ..., tools: [...]}` — validated by a Pydantic settings model with actionable errors. OPEN QUESTION flagged in the issue: final key naming to be settled in review against docs/ARCHITECTURE-V2.md (bootstrapped by this milestone's first issue; D10); the resolver's public function signature is the stable seam.
- Default flip (D5): update the packaged template petri/defaults/petri.yaml so `petri init` writes a harness block with pi as the default backend, with a fallback comment pointing at the Node.js install docs. Existing dishes (no harness block) are untouched: petri/cli/_bootstrap.py resolve_provider keeps returning ClaudeCodeProvider exactly as today.
- Preflight: extend petri/engine/preflight.py (currently: Python version + claude CLI checks) with check_pi() — pi binary found, Node present and >= the version floor from docs/spikes/pi-rpc.md — surfaced by `petri inspect` with install guidance when the dish config selects pi.
- Docs: README section 'Choosing an inference backend' covering all three backends, the Node.js dependency story (from the spike), and the cost-warning posture per backend (README cost warnings are a standing requirement).

**Out of scope.**
- Wiring the resolved Model into seed/grow (the bridge issues); unifying the rest of config.py's accessors/constants (M4-dbos owns full config.py retirement — this issue only carves the harness path out of the trap).

**Touched files.** petri/defaults/petri.yaml (template: add harness block); petri/engine/preflight.py (add check_pi); README.md. Referenced, behavior-preserved: petri/cli/_bootstrap.py:16-37; petri/config.py:19-27, 44-56, 83, 247-250. New: petri/harness/resolve.py, tests/unit/harness/test_resolve.py.

**Acceptance criteria:**
- [ ] Unit tests: backend 'pi' → PiModel; 'claude-code' → ClaudeCodeModel; 'provider' → the exact string passed through unmodified; malformed block → validation error naming the offending key
- [ ] Missing pi binary with backend 'pi' raises an actionable error that names the Node.js install doc; check_pi() reports pi/Node status and `petri inspect` displays it (manual verification steps in PR)
- [ ] petri init on a fresh directory writes a petri.yaml containing the harness block with pi default
- [ ] A dish config WITHOUT a harness block resolves exactly as v0.3.4: test asserts _bootstrap.resolve_provider returns a ClaudeCodeProvider instance and no petri/harness import occurs on that path
- [ ] resolve_model never consults functools.lru_cache-d state and never reads packaged defaults when given a dish config (asserted via test with a temp dish config differing from packaged defaults)
- [ ] Full existing suite green, including seeded_petri_dir-based integration tests unchanged

---

## M1-harness.11 — Add HarnessInferenceProvider bridge (part 1 of 2): decomposition-path methods on pydantic-ai Agents

**Size:** M · **Labels:** migration-v2, harness · **Depends on:** Extract ClaudeCodeProvider prompt builders into petri/reasoning/prompts.py (behavior-preserving); Build TestModel/FunctionModel offline test harness for harness-backed code

**Context.** Strangler payoff for M1: the OLD engine keeps working, and users who opt in via the new `harness:` block get `petri seed`/`petri grow` running on pi, Claude Code, or any pydantic-ai provider — with structured output replacing regex extraction on the bridged path. The seam already exists: the v1 engine consumes the runtime_checkable InferenceProvider Protocol (petri/models.py:614-693 — assess_claim_substance, generate_clarifying_questions, decompose_claim, decompose_why, assess_cell, match_evidence, each with an optional on_progress callback), resolved in exactly one place (petri/cli/_bootstrap.py:16-37). The processor already tolerates real Pydantic results via its dict-or-model duck-typing (petri/engine/processor.py:230-354), so returning validated models is safe. The bridge lands in two parts at the natural seam: THIS issue implements the four decomposition-path methods; part 2 adds assess_cell + match_evidence, verdict discipline, and the CLI wiring that makes the bridge reachable.

**Scope.**
- petri/harness/legacy_bridge.py: HarnessInferenceProvider class over a resolved pydantic-ai Model, implementing assess_claim_substance, generate_clarifying_questions, decompose_claim, and decompose_why as pydantic-ai Agents — output_type set to the existing result models (AssessmentResult petri/models.py:388-397, DecompositionResult 484-490, ClarifyingQuestion 493-498), prompts from petri/reasoning/prompts.py WITHOUT the JSON-emission suffixes (pydantic-ai owns output formatting).
- Failure-semantics fix on the bridged path: decompose_why must distinguish 'genuinely atomic' from 'parse/validation failure' — validation failure raises/retries instead of silently returning [] (the legacy ambiguity at claude_code_provider.py:682-688 silently truncates trees).
- on_progress: accept the callback; emit coarse progress (call start/end, plus text deltas where the Model exposes them). Full streaming parity with the CLI MultiSpinner is a documented limitation for M1.
- The class is NOT yet reachable from the CLI (part 2 wires petri/cli/_bootstrap.py); assess_cell and match_evidence raise NotImplementedError with a pointer to part 2 until then.

**Out of scope.**
- assess_cell, match_evidence, verdict discipline, CLI wiring, and integration tests (all part 2); any processor/engine change; DBOS; removing or deprecating ClaudeCodeProvider; streaming parity.

**Touched files.** Referenced, unmodified: petri/models.py:614-693; petri/engine/processor.py:230-354; petri/reasoning/prompts.py. New: petri/harness/legacy_bridge.py, tests/unit/harness/test_legacy_bridge.py.

**Acceptance criteria:**
- [ ] Offline FunctionModel tests cover all four decomposition-path methods (assess_claim_substance, generate_clarifying_questions, decompose_claim, decompose_why) returning their typed results
- [ ] decompose_why regression test: a validation-failing model output triggers retry (not a silent [] return); a genuine is_atomic=true output returns [] — the two cases are distinguishable in test
- [ ] Prompts are composed without the JSON-emission suffixes (test asserts via the prompts.py suffix helpers)
- [ ] on_progress callback receives start/end events in offline tests
- [ ] git diff shows zero changes under petri/cli/, petri/engine/, and petri/reasoning/; full existing suite green

---

## M1-harness.12 — Add HarnessInferenceProvider bridge (part 2 of 2): assess_cell verdict discipline, match_evidence, and CLI wiring

**Size:** M · **Labels:** migration-v2, harness · **Depends on:** Add HarnessInferenceProvider bridge (part 1 of 2): decomposition-path methods on pydantic-ai Agents; Add harness resolution and pydantic-ai provider-string passthrough from petri.yaml

**Context.** Completes the HarnessInferenceProvider bridge started in part 1: the two research-path Protocol methods (assess_cell, match_evidence) plus the CLI wiring that makes the bridge reachable via the `harness:` block, so `petri seed`/`petri grow` run end-to-end on the selected backend. Verdict discipline is the heart of this issue, motivated by v1's 'silent PASS' bug: the legacy substring fallback could coerce unparseable verdict output into the first valid verdict that happened to appear as a substring, silently converting failures into passes. v2's fail-loud invariant has two sides: invalid verdicts are reflected back for model self-correction, and when retries are exhausted the failure is STILL preserved as an explicit non-passing EXECUTION_ERROR verdict appended to the domain event log (events table) — never dropped, never coerced — with structured failure detail landing in the event's data field (uniform envelope: {id, cell_id, timestamp, type, agent, iteration, data}).

**Scope.**
- assess_cell: constrain/validate the verdict field against the role's configured vocabulary (config get_agent_verdicts, petri/config.py:180) via an output validator so invalid verdicts are reflected back for self-correction; on exhausted retries return the structured EXECUTION_ERROR verdict exactly as the legacy provider does (claude_code_provider.py assess_cell failure channel) so processor semantics are unchanged, carrying the harness FailureDetail (exit code, last stream event, partial output) so it is persisted into the domain event log (events table).
- match_evidence: output_type EvidenceMatch (petri/models.py:501-506), prompts from petri/reasoning/prompts.py without JSON-emission suffixes.
- Wiring: petri/cli/_bootstrap.py resolve_provider gains one guarded branch — dish config HAS a harness block → petri.harness.resolve + HarnessInferenceProvider; otherwise the existing ClaudeCodeProvider path runs byte-identically (default unchanged for existing dishes).
- OPEN QUESTION recorded in the issue: whether pi's tool allowlist can grant a WebSearch-equivalent so bridged research agents retain live citation verification (spike findings decide); if not, document that assess_cell on the pi backend runs without web tools in M1 and M2-agents restores tools as pydantic-ai toolsets.
- Integration tests for the end-to-end opt-in path.

**Out of scope.**
- Any processor/engine change; DBOS; removing or deprecating ClaudeCodeProvider; streaming parity; summarize_evidence beyond a straight port if the CLI path requires it.

**Touched files.** petri/cli/_bootstrap.py:16-37 (guarded branch). petri/harness/legacy_bridge.py (extend part 1's class). Referenced, unmodified: petri/models.py:614-693; petri/engine/processor.py:230-354; petri/config.py:180. Extended: tests/unit/harness/test_legacy_bridge.py. New: tests/integration/test_bridge_seed.py (integration-marked).

**Acceptance criteria:**
- [ ] isinstance(HarnessInferenceProvider(...), InferenceProvider) passes against the runtime_checkable Protocol (petri/models.py:614) — all six Protocol methods implemented
- [ ] Offline FunctionModel tests cover assess_cell and match_evidence returning their typed results
- [ ] assess_cell test: out-of-vocabulary verdict is reflected back and corrected via retry; exhausted retries yield verdict=EXECUTION_ERROR with the same shape the processor expects today, and the emitted event's data field carries the structured failure detail (exit code, last stream event, partial output) — the failure is a domain event, never a coerced pass
- [ ] With no harness block in dish config, resolve_provider returns a ClaudeCodeProvider (type-asserted) and the full v1 test suite passes unchanged
- [ ] Integration test (skipped without pi): `petri seed` on a scratch dish with harness backend pi produces a serialized colony and `petri grow --dry-run` lists eligible cells
- [ ] Manual verification steps in PR: same seed flow exercised with backend 'provider' + an 'anthropic:...' string (requires API key, documented as manual)

---
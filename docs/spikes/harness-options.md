# Inference harness options for Petri v2

The inference harness is the layer that runs model calls, classifies failures, and reports
usage. Petri v2 replaces the hand-built v1 layer, and three candidate backends exist. This
document describes the tested behavior and tradeoffs of each so that the team can choose.
It does not select an option.

The candidates are [pi](https://github.com/earendil-works/pi), a multi-provider agent
harness driven over JSON-RPC; the Claude Code CLI behind an adapter; and direct provider
APIs through pydantic-ai. The pi option was validated against a live install of pi 0.80.9,
with 240 recorded call attempts; transcripts are committed under `docs/spikes/`. In brief:
pi offers provider flexibility and cost telemetry, Claude Code offers subscription-covered
usage and built-in web tools, and direct APIs offer simplicity with no external binary.
The migration plan's working proposal (decision D5) names pi as the default; the evidence
below is what that proposal rests on, presented for the team to confirm or revise.

**Validation at a glance**

- 9 capabilities probed: 5 validated, 2 conditional, 2 untested, 0 failed
- 240 recorded call attempts (3 runs × 80; 185 completed, the rest error-path)
- 3 distinct error channels, one of which reports success on failure
- $0.045 measured per call without session reuse; 66% of tokens were cache-write

> **Notation.** D1–D10 are the decisions recorded in `docs/ARCHITECTURE-V2.md` (D3:
> zero-API-key subscription auth must remain possible; D5: the pi-as-default proposal this
> document informs). M1–M7 are the v2 migration milestones (tracker
> [#100](https://github.com/onthemarkdata/petri/issues/100)). #N are repository issues.

## What the harness layer replaces

Petri v1 implements its inference layer directly against the `claude` CLI: roughly 900
lines of subprocess management, stream parsing, stderr-only retry classification, and
regex JSON extraction (`claude_code_provider.py`). Two production incidents trace to this
layer ([field-reports.md](../field-reports.md)). In the first, a rate limit placed its
reset hint on stdout while the classifier read only stderr, so every retry was spent in
about five seconds. In the second, model output that failed to parse was silently coerced
into a valid verdict.

The v2 design replaces this with a harness layer that supports three interchangeable
backends behind one resolver. The implementation chain is the same whichever backend
becomes the default: error taxonomy (#18), fake_pi fixtures (#20), RPC transport (#21),
PiModel (#22), backend resolution (#25), and the CLI bridge (#27). The choice of default
affects configuration and documentation, not the shape of the code.

## Tested behavior of pi

pi is the only candidate that required validation, since the other two are known
quantities. Each row records one protocol assumption from the implementation issues and
the behavior observed against the live binary. Raw transcripts are committed under
`docs/spikes/`.

| Capability | Status | Observed behavior | Relevant to |
|---|---|---|---|
| RPC transport and LF-JSONL framing | VALIDATED | The command, response, and event stream behave as documented. Newlines embedded in JSON strings survive the round trip. The fallback transports `--mode json` and `-p` both function. | #21 |
| Session model | VALIDATED | pi acknowledges commands immediately but executes them serially from a queue, so a transport should hold one request in flight. Persistent sessions and session forking are available. | #21 #22 |
| Usage and cost metadata | VALIDATED | Every assistant message carries token counts and dollar cost, and `get_session_stats` aggregates them per session. Summed costs reconcile with billed spend. This closes a v1 gap: agents self-reported usage once in roughly 2,700 events ([field-reports.md](../field-reports.md)). | #22 #36 M5 |
| Error surfacing | VALIDATED | Failures surface on three classifiable channels. The channel depends on the transport mode, and one of them reports success at the command level while the failure rides inside the message. The next section covers this in detail. | #18 #21 |
| Subscription OAuth | CONDITIONAL | Claude Pro/Max OAuth works, including headless flows in which the authorization URL and code are relayed through another channel. Billing is per token against the account's extra-usage balance rather than plan usage, and an exhausted balance fails every call with an HTTP 400. API-key environment variables are the simpler headless path. | #25 D3 |
| Multi-provider and local models | VALIDATED | One protocol covers Anthropic, OpenAI, Google, Bedrock, Ollama, and OpenRouter, with per-model pricing in the catalog. The other two candidates do not offer this. | #25 |
| Tool surface | CONDITIONAL | pi ships no WebSearch or WebFetch equivalent, so research roles cannot satisfy the citation policy on an unmodified pi backend. The `pi.registerTool()` extension API can supply equivalent tools. | #22 M2 |
| Structured output | UNTESTED | A native schema mode is not confirmed. The working baseline is pydantic-ai PromptedOutput with validation retry over plain text, which functions with the existing prompts. | #22 |
| Streaming and live rate-limit shape | UNTESTED | The event types exist in the protocol (`text_delta`, and `auto_retry_*` carrying a `delayMs` hint), but neither has been observed on a live generation. Both appear in the [pi-rpc.md](./pi-rpc.md) runbook. | #18 M5 |

> **Findings attributable to Petri rather than pi.** Two validation observations belong to
> Petri's side of the boundary. A stall under concurrent load traces to the v1 concurrent
> processor, which M4 replaces; serial pi calls completed in about nine seconds each
> throughout. Colony non-convergence under default verdict thresholds is an
> agent-calibration concern owned by M2. pi executed every call it was given correctly.

## Error channels

Where a failure surfaces depends on the transport mode and on when it occurs: before the
turn starts (A), during generation (B), or at the process boundary (C). Channel B carries
the most integration risk, because the command reports success while the failure rides
inside the message; an integration that checks only the command result will repeat v1's
silent-PASS incident. The transport (#21) checks all three channels, and the
classification table in [pi-rpc.md](./pi-rpc.md) records each channel's raw output.

### Channel A — RPC mode, command-level

The command fails before any turn starts. This channel classifies cleanly as a permanent
authentication error.

```json
{"type":"response",
 "success":false,
 "error":"No API key found for anthropic…"}
```

### Channel B — RPC mode, in-turn

A credential or provider failure during generation. The command returns `success:true`
and the turn completes normally; the failure is inside the message.

```json
"message":{
  "stopReason":"error",
  "errorMessage":"UnrecognizedClientException: 403…"}
```

### Channel C — JSON / -p mode, process boundary

The same failure as Channel A, surfaced on stderr with a nonzero exit. It is easy to
misclassify as an empty response, which discards the diagnostic.

```text
$ pi --mode json …
exit 1
stderr: No API key found for anthropic.
```

## The event stream as an OTel source

M5 specifies OTel spans with domain attributes, a local spans table, and per-cell cost
accounting (#64–#72). pi does not emit OTel natively, but its event stream carries the
raw material. The transport is the natural mapping point, with one pi call becoming one
generation span. The claude-code and direct-API backends offer less here: the claude CLI
exposes no usage telemetry, and pydantic-ai reports tokens without pricing.

A recorded event sequence, annotated (from `docs/spikes/transcripts/`; the marked fields
become span data):

```text
{"type":"session","version":3,"id":"019f6c6f-b8d6-…",             ← trace / session correlation
 "timestamp":"2026-07-16T19:38:08.982Z","cwd":"…"}
{"type":"agent_start"}                                            ← span start event
{"type":"turn_start"}                                             ← generation span opens
{"type":"message_start","message":{"role":"user",
 "timestamp":1784230533415, …}}                                   ← span timestamps (epoch ms)
{"type":"message_end","message":{"role":"assistant",
 "api":"…","provider":"anthropic","model":"claude-sonnet-4-6",    ← gen_ai.system / request.model
 "usage":{"input":3,"output":6,"cacheRead":0,
          "cacheWrite":1557,"totalTokens":1566,
          "cost":{"total":0.00593775}},                           ← gen_ai.usage.* + petri.cost.usd
 "stopReason":"stop", …}}                                         ← span status (error path: Channel B)
{"type":"turn_end", …}                                            ← generation span closes
{"type":"agent_end","willRetry":false, …}                         ← retry event attribute
{"type":"agent_settled"}                                          ← span end event
```

| pi output field | Status | OTel target | Consumer |
|---|---|---|---|
| `session.id` · `session.timestamp` | RECORDED | Session correlation attribute (`petri.session_id`) | #68 #69 |
| `turn_start` / `turn_end` · message timestamps | RECORDED | Generation-span boundaries and span timestamps | #67 #71 |
| `message.provider` · `message.model` · `message.api` | RECORDED | `gen_ai.system` and `gen_ai.request.model` | #67 |
| `usage.input/output/cacheRead/cacheWrite` · `cost.total` | RECORDED | `gen_ai.usage.*` tokens and `petri.cost.usd`, the per-cell cost rows | #69 #72 #36 |
| `stopReason` · `errorMessage` · `agent_end.willRetry` | RECORDED | Span status Error with `exception.message`, plus a retry attribute | #18 #67 |
| `get_session_stats` · `contextUsage.percent` | RECORDED | Context-window pressure gauge on the session span | #68 |
| `message_update` / `text_delta` | DOCUMENTED | Stream progress events for live-progress display | M5 #76 |
| `tool_execution_start/update/end` | DOCUMENTED | Child tool spans under the `gen_ai` tool conventions | #67 M2 |
| `auto_retry_start/end {attempt, maxAttempts, delayMs}` | DOCUMENTED | Retry span events and a backoff hint for the #18 policy. These fire only when pi's auto-retry is enabled, which risk R4 recommends against in steady state, so they are a diagnostic-mode signal. | #18 |

Petri supplies the domain half of every span (dish, colony, cell ID, agent role, verdict,
iteration — #68) and pi supplies the execution half (model, provider, timing, tokens,
cost). The transport is where they join. Two capture gaps constrain the design. First,
`-p` mode emits plain text with no events, so instrumented calls should not use it.
Second, Channel-C errors bypass the event stream, so the telemetry layer must also record
process exit data. RECORDED means present in the committed transcripts or run logs;
DOCUMENTED means specified in pi's `docs/rpc.md` but not yet observed live.

## Session reuse determines cost

Spawning a fresh `--no-session` pi process for each call re-pays the prompt-cache write
on every call. In the measured runs, two thirds of all billed tokens were avoidable cache
writes, which makes the long-lived session in #21 a cost requirement rather than a
refinement.

Token composition of the heaviest validation run — 745,857 tokens across 79 completed
calls, one process per call ([pi-e2e-report.md](./pi-e2e-report.md)):

| Category | Tokens | Share |
|---|---:|---:|
| Cache write (the avoidable share) | 492,291 | 66% |
| Output | 131,274 | 18% |
| Cache read | 122,055 | 16% |
| Input | 237 | <0.1% |

The measured $0.045 per call on claude-sonnet-4-6 is the one-process-per-call figure. The
cost after session reuse has not been measured, but the 66% cache-write share bounds the
improvement. A second cost lever is routing routine agent roles to cheaper or local
models, which is only available under a multi-provider backend. pi's per-request pricing
metadata makes both levers measurable, and feeds the cost caps in #36 and the M5
accounting.

## Three backends, compared

The #25 resolver supports all three backends regardless of which becomes the default, so
the choice is about defaults and documentation rather than architecture.

### Option 1 — pi

An open-source, multi-provider agent harness driven over LF-delimited JSON-RPC. Its
strengths are one protocol across Anthropic, OpenAI, Google, Bedrock, and local models,
and per-request token and dollar telemetry that no other option provides. Its weaknesses
are youth (a fast-moving v0.x release line), no built-in web tools, and subscription
billing that draws on a prepaid balance. It is the only option with a dedicated risk
register, below, because it is the only one that introduces a new external dependency.

### Option 2 — Claude Code CLI behind an adapter

The v1 backend, retained behind the new harness interface. Its strengths are usage
covered by an Anthropic subscription plan at no marginal cost, built-in WebSearch and
WebFetch (which the research roles' citation policy requires), and a mature,
vendor-maintained binary. Its weaknesses are a single provider, no usage or cost
telemetry from the CLI, and no path to local models.

### Option 3 — direct provider APIs through pydantic-ai

Provider strings such as `anthropic:claude-sonnet-4-6` resolved natively by pydantic-ai.
Its strengths are pure-Python operation with no external binary and access to any
pydantic-ai provider. Its weaknesses are API-key-only billing, no web tools until the M2
toolsets exist, and token counts without pricing (pydantic-ai RunUsage reports tokens;
dollar cost requires a separate price table).

| | pi | claude-code adapter | provider string |
|---|---|---|---|
| **Providers** | Anthropic, OpenAI, Google, Bedrock, Ollama, local — one protocol | Anthropic only | Any pydantic-ai provider |
| **Auth and billing** | Subscription OAuth or API key; OAuth bills per token from the extra-usage balance | Subscription; usage covered by the plan | API key or cloud credentials; per-token billing |
| **Web tools** | None built in; an extension can supply them | WebSearch and WebFetch built in | None until the M2 toolsets |
| **Runtime dependency** | External binary; a Node-free standalone build exists (the npm path needs Node ≥22.19) | External binary; native installer, no Node | None; pure Python |
| **Usage and cost telemetry** | Tokens and dollars per message, plus session totals | None exposed by the CLI | Tokens via RunUsage; pricing separate |
| **Maturity** | v0.x and fast-moving; requires a version pin | Mature and vendor-controlled | Mature |

**How to choose.** No option wins every row, so the default follows from which axis
matters most. If provider flexibility and a path to local models matter most, pi is the
strongest option and the only one that also carries full cost telemetry. If zero marginal
cost under an existing Anthropic subscription and working web tools matter most, the
claude-code adapter is the strongest option. If minimizing external dependencies matters
most, provider strings are the strongest option. The resolver keeps the other two
available whichever default is chosen, and the transport seam keeps pi replaceable if its
release churn becomes a burden.

## Risks specific to the pi option

These apply only if pi becomes the default. Each maps to an existing backlog issue, and
R1 and R2 have the largest immediate impact. The other options' principal weaknesses are
covered in their descriptions above.

### R1 — Channel-B errors require first-class handling

A generation failure inside a successful command is pi's version of the silent-PASS bug.
The failure mode is recorded in the spike transcripts and is easy to reintroduce.
**Owner:** the #18 classification (exit code, stderr, stdout, and in-band fields) and the
#21 transport tests.

### R2 — No web tools on an unmodified pi backend

Research roles require WebSearch and WebFetch for citation integrity, which bare pi
cannot satisfy. **Owner:** the capability manifest, under which search-requiring roles
refuse pi until the M2 toolsets or a vetted pi extension exist.

### R3 — Per-call processes waste two thirds of spend

Without session reuse, every call re-pays the prompt-cache write. **Owner:** the
long-lived session in #21, and one session per PiModel in #22.

### R4 — Retry stacking

pi retries automatically with its own backoff, Petri has a retry policy, and M4 adds step
retries from DBOS, the durable-execution engine. Three retry layers against one rate
limit would repeat the v1 incident at larger scale. **Owner:** disable pi's auto-retry
via `set_auto_retry`; the #18 policy is configurable to zero, so Petri owns all
throttling. Disabling auto-retry also silences its telemetry events, an accepted tradeoff
noted in the telemetry table.

### R5 — Subscription billing can surprise users

pi's Claude OAuth draws on prepaid extra-usage credit rather than plan usage — Anthropic
bills third-party harness traffic from the prepaid balance, while Claude Code's own
traffic is covered by the plan — and the two are easy to conflate. An exhausted balance
fails cleanly, with errors becoming EXECUTION_ERROR escalations, but it stops the run.
**Owner:** the per-backend cost warnings in the #25 README section, and the cost caps in
#36.

### R6 — A v0.x dependency for a Python-first audience

pi is young and changes quickly, and its npm install path requires Node ≥22.19.
**Owner:** D5's constraint that pi is never a pip dependency; the Node-free standalone
binary as the only documented install; the `check_pi()` preflight with a version pin in
#25; and the committed transcripts, which make protocol regressions diagnosable.

## What the validation did not establish

| Question | Why it matters | Where it lands |
|---|---|---|
| Whether pi has a native structured-output mode | It would replace the PromptedOutput baseline with schema enforcement | #22 |
| Streaming content on a live generation | CLI progress parity before the default engine changes | M5 / runbook |
| The live rate-limit shape and reset hints | It completes the #18 classification for rate limits | #18 / runbook |
| OAuth on the standalone binary and headless seeding | The Node-free install path must also support subscription login | #25 / runbook |

None of these blocks the shared implementation work in #18, #20, or #21. Each has a
reproduction step in the [pi-rpc.md](./pi-rpc.md) runbook.

---

**Evidence:** [pi-rpc.md](./pi-rpc.md) (protocol contract) ·
[pi-e2e-report.md](./pi-e2e-report.md) · [pi-e2e-evidence.md](./pi-e2e-evidence.md)
(claim-by-claim evidence map) ·
[PR #119](https://github.com/onthemarkdata/petri/pull/119).
**Related records:** `docs/ARCHITECTURE-V2.md` (D3, D5) · `VISION.md` · issues #9, #17,
#18, #20–#23, #25, #27, #36, #64–#72. All measurements were taken with pi 0.80.9 and
`anthropic/claude-sonnet-4-6`, 2026-07-16 to 07-17.

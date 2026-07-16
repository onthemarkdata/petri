# Spike: `pi --mode rpc` and the Node.js dependency story

> **Issue:** #17 / `M1-harness.2`. **Status:** protocol validated against a real `pi`
> install; model-generation items pending a provider credential (see
> [Open questions & runbook](#open-questions--runbook)).
> **pi version probed:** `0.80.9` (npm `@earendil-works/pi-coding-agent`), Node `v22.22.2`,
> Linux x64. **Date:** 2026-07-16.
>
> This doc is the **protocol contract** for the pi RPC transport (`M1-harness.6`), the
> `fake_pi` stub (`M1-harness.5`), `PiModel` (`M1-harness.7`), the shared error taxonomy
> (`M1-harness.3`), and harness preflight (`M1-harness.10`). Raw transcripts live in
> [`docs/spikes/transcripts/`](transcripts/); the probe that produced them is
> [`docs/spikes/pi_rpc_probe.py`](pi_rpc_probe.py).

## How this was run

```bash
python docs/spikes/pi_rpc_probe.py --self-test               # offline framing check (no pi)
python docs/spikes/pi_rpc_probe.py --with-model --out docs/spikes/transcripts
```

All transcripts here were captured **without a provider credential** (none is available in
the spike sandbox). That still exercises every protocol path except a *successful*
generation, and it directly produced the auth-failure / provider-error artifacts the
acceptance criteria require. The observed default model in the sandbox resolved to
`us.anthropic.claude-opus-4-6-v1` via `amazon-bedrock`, whose ambient credentials are
invalid (403) — which is why generations return provider errors rather than text.

## Verdict per capability

| Capability | Verdict | Basis |
|---|---|---|
| RPC transport & LF-JSONL framing | **GO** | Real round-trip of commands/events; framing confirmed (`introspection.jsonl`, `--self-test`) |
| Error taxonomy (transient/permanent) | **GO (rich)** | Three distinct error channels observed & classifiable (below) |
| Usage / cost reporting | **GO** | Per-message `usage{input,output,cacheRead,cacheWrite,totalTokens,cost{...}}` + `get_session_stats` |
| Session model (sequential vs multiplexed) | **GO w/ constraint** | Commands ack immediately but execute **queued/serialized** → lock to one in-flight |
| Provider switching | **GO** | `--provider`, `--model provider/id`, `get_available_models`, `set_model` |
| Streaming | **UNKNOWN (mechanism present)** | `message_update`/`text_delta` event type exists; needs a real generation to confirm content deltas |
| Structured output | **UNKNOWN → baseline chosen** | No successful generation yet; adopt pydantic-ai `PromptedOutput` + validation-retry; native JSON-schema mode is an open question |
| Tool surface / **web search** | **NO web search** | Built-ins are coding tools only (`read/bash/edit/write`, +`grep/find/ls`); **no WebSearch/WebFetch equivalent** |
| Node-free install | **GO (production)** | Standalone Bun binaries on GitHub Releases; npm path used in this sandbox only |

## RPC protocol (validated)

- **Framing:** strict LF-delimited JSONL. Split on `\n` only, tolerate a trailing `\r`; do
  **not** use a Unicode-aware splitter (pi's `docs/rpc.md` calls out Node `readline` as
  non-compliant because it also breaks on U+2028/U+2029). The probe reads bytes and frames
  itself; `--self-test` proves an embedded `\n` inside a JSON string round-trips intact.
- **Requests** (stdin): `{"id": "...", "type": "prompt", "message": "..."}` and control
  commands `get_available_models`, `get_session_stats`, `get_state`, `set_model`,
  `set_auto_retry`, `fork`, session ops, etc. `id` is optional but correlates the response.
- **Command responses** (stdout): `{"id","type":"response","command":<cmd>,"success":<bool>,"data"?,"error"?}`.
- **Turn events** (stdout, no `id`): `agent_start` → `turn_start` → `message_start`/
  `message_update`(`text_delta`)/`message_end` → `turn_end` → `agent_end` → `agent_settled`.
  Assistant messages carry `api`, `provider`, `model`, `usage`, and — on failure —
  `stopReason:"error"` + `errorMessage`.

Example (`get_session_stats`, `introspection.jsonl`):

```json
{"id":"stats","type":"response","command":"get_session_stats","success":true,
 "data":{"tokens":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},
         "cost":0,"contextUsage":{"tokens":0,"contextWindow":1000000,"percent":0}}}
```

## Error taxonomy — THREE channels (the key finding for `M1-harness.3`)

pi does **not** put all errors in one place. The transport and the shared classifier must
inspect all three, and the fail-loud invariant hangs on Channel B:

**Channel A — command-level pre-flight failure** (`error_missing_key.jsonl`). A misconfig
(e.g. no API key) fails the `prompt` command itself, before any turn:

```json
{"id":"r","type":"response","command":"prompt","success":false,
 "error":"No API key found for anthropic.\n\nUse /login to log into a provider via OAuth or API key. ..."}
```
→ classify from `error` string; this one is **permanent/auth** (`AuthExpiredError`).

**Channel B — in-turn provider failure** (`prompt_roundtrip.jsonl`, `session_multiplex.jsonl`).
The command returns `success:true` and the turn reaches `agent_settled`, but the assistant
message carries the failure:

```json
{"type":"turn_end","message":{"role":"assistant","content":[],
 "usage":{"input":0,"output":0,"totalTokens":0,"cost":{"total":0}},
 "stopReason":"error","errorMessage":"UnrecognizedClientException: 403: [object Object]"},
 "toolResults":[]}
{"type":"agent_end", "...":"...", "willRetry":false}
```
→ **A naive top-level `success` check MISSES this** — a failed generation looks "settled".
This is exactly Petri's v1 *silent-PASS* anti-pattern. The transport must treat
`message.stopReason == "error"` (empty content, `errorMessage` set) as a typed
`HarnessError`, never as a valid (empty) response. `agent_end.willRetry` signals pi's own
retry intent; `errorMessage` is the classification input (403 → auth/permanent here).

**Channel C — non-RPC print mode** (`fallback_print.jsonl`). `pi -p` surfaces the same
provider error on **stderr** and exits non-zero (`rc=1`, stderr
`"UnrecognizedClientException: 403: [object Object]"`). (One run instead *hung* on the
model call and had to be timed out — `-p` is the least suitable transport for a daemon.)

**Rate-limit (429) shape — OPEN.** Not reproducible without a live provider. Expected via
Channel B (`stopReason:"error"`, `errorMessage` containing 429/throttling) and/or pi's
`auto_retry_start`/`auto_retry_end` events (`{attempt,maxAttempts,delayMs}`). `set_auto_retry`
is accepted (`auto_retry_off.jsonl`). **Recommendation:** disable pi's built-in auto-retry
(`set_auto_retry:false`) so Petri owns throttling — otherwise pi's retries stack under
`petri/harness/retry.py` and later DBOS step retries (the reconciled-throttling principle,
MIGRATION_PLAN §6.1; `retry.py` must be configurable to zero). Note pi surfaces a `delayMs`
back-off hint, **not** a raw provider reset timestamp, so the "resets 9pm (America/
Los_Angeles)" clock-form parsing in `M1-harness.3` remains a Claude-Code-backend concern,
not a pi one.

## Session model — sequential vs multiplexed

Two `prompt` commands sent back-to-back are **both ack'd immediately**
(`{"id":"a",...success:true}` then `{"id":"b",...success:true}`) but the turn events then
run through to `agent_settled` in order (`session_multiplex.jsonl`) — pi **queues** work
(there is a `queue_update` event type). **Contract:** treat the RPC session as
single-in-flight; `M1-harness.6` should enforce one outstanding `prompt` per transport with
a lock. (A real-generation test should confirm the queue does not interleave token streams
before relaxing this.)

## Usage / cost — pi is the instrumentation point

Every assistant message includes
`usage:{input,output,cacheRead,cacheWrite,totalTokens,cost:{input,output,cacheRead,cacheWrite,total}}`,
and `get_session_stats` returns cumulative `tokens`, `cost`, and
`contextUsage{tokens,contextWindow,percent}`; `get_state`/`get_available_models` carry
per-model `cost` rates. This answers the migration's cost-capture gap (1 self-reported
usage event in ~2,700): **`PiModel` reads usage from pi metadata** and M5-otel consumes it —
no agent self-reporting.

## Tool surface — the web-search gap (gates `M2` capability manifest)

pi's built-ins are coding tools: `read, bash, edit, write` (tagline) plus `grep/find/ls`
(docs); controlled by `--tools`/`--no-tools`/`--exclude-tools`/`--no-builtin-tools`.
`get_state` exposes **no** tool list. **There is no built-in WebSearch/WebFetch equivalent.**
Petri's `agent_tools` (`petri/defaults/petri.yaml:28-33`) treat `WebSearch`/`WebFetch` as
load-bearing for citation integrity — so per the capability-manifest principle
(MIGRATION_PLAN §6.8), **search-requiring roles must refuse the pi backend** until web tools
are supplied as pydantic-ai toolsets in M2 (or via a pi extension). `PiModel` in M1 should
default to `--no-tools` and document this gap per provider.

## Structured output

No successful generation yet, so reliability is unmeasured. **Adopt the portable baseline:**
pydantic-ai `PromptedOutput` + validation-retry over plain text (works on any backend). If a
later live run proves a native JSON-schema mode, expose it behind the same interface. Do
**not** port v1's substring/fenced-JSON coercion — the fail-loud invariant forbids it.

## The Node.js dependency story

- **Production install is Node-free.** pi ships self-contained Bun-compiled binaries on its
  GitHub Releases (`pi-linux-x64.tar.gz`, `pi-darwin-arm64.tar.gz`, … + `SHA256SUMS`) that
  embed their runtime — download, verify checksum, run; **no Node, npm, or Bun required at
  runtime.** This is the path `petri inspect`/`check_pi` (`M1-harness.10`) should document,
  consistent with D5 ("pi is an optional runtime dependency, never a pip dependency").
- **⚠️ This sandbox used the npm path, which we do NOT want as the dependency.** GitHub
  Releases and `pi.dev` are blocked by the sandbox egress proxy (403), so the spike installed
  `pi 0.80.9` via `npm install -g @earendil-works/pi-coding-agent` **purely to unblock live
  probing**. The npm path requires **Node ≥ 22.19.0** (`package.json` `engines`) — recorded
  here only as the fallback/dev path. Petri stays a pure-Python package; pi is always spawned
  as an external subprocess and never enters Petri's packaging.
- **Version floor:** Node ≥ 22.19.0 (npm path only). `npx` was not validated; the standalone
  binary makes it moot for end users.
- **Missing-pi behavior:** before install, invoking `pi` yields `pi: command not found` — the
  signal `check_pi()` should surface with install guidance. Missing-**Node** behavior for the
  npm path was not observable here (Node 22.22.2 is present) — a runbook item.

## Recommendation

**GO: keep pi as the default backend.** Its strategic value is *model pluralism* — one RPC
harness normalizing frontier **and local open** providers (VISION.md:247; D3/D5), which is
what lets Petri run on Qwen today and a local Petri model later. The Node-free standalone
binary satisfies the "avoid Node" constraint.

Framing the alternatives on the one axis that matters (subscription-auth + web tools vs
model pluralism):
- **`claude-code` (native binary):** a **scoped Anthropic-only fallback** — Node-free, keeps
  zero-API-key subscription auth *and* WebSearch/WebFetch. Best for the Anthropic-subscription
  user, but **not** a substitute for pi's cross-provider/local reach.
- **pydantic-ai native + API key:** pure-Python, any provider, but loses subscription auth and
  built-in web tools (fabricated-citation risk until M2 toolsets).

## Open questions & runbook

To finalize the UNKNOWN verdicts, run on a machine with a provider credential
(`export ANTHROPIC_API_KEY=...` or `pi` `/login`), preferably also fetching the **standalone
binary** to validate the Node-free path:

```bash
# 1. (production path) download + verify the Node-free binary, then point the probe at it:
#    pi-linux-x64.tar.gz + SHA256SUMS from the pi GitHub release; extract; --pi-bin ./pi
# 2. run the model-touching probes:
python docs/spikes/pi_rpc_probe.py --with-model --provider anthropic \
    --model anthropic/claude-sonnet-4-6 --out docs/spikes/transcripts
```

Then resolve and fold back into this doc:
1. **Structured output** — does pi offer a native JSON-schema mode, or is `PromptedOutput` +
   validation-retry the only path? (`M1-harness.7` output-mode question.)
2. **Streaming** — confirm `message_update`/`text_delta` carries incremental content on a real
   generation.
3. **Multiplexing** — confirm the queue serializes and does not interleave two turns' token
   streams (relax or keep the single-in-flight lock in `M1-harness.6`).
4. **Rate-limit (429)** — capture the real Channel-B shape and whether a reset hint appears,
   to finalize the `M1-harness.3` classification row.
5. **Subscription auth headless** — confirm `pi /login` (Claude Pro/Max) works non-interactively
   for the RPC daemon and how creds seed `~/.pi/agent/auth.json`.
6. **Node-free binary** — validate download + checksum + run, and record missing-Node behavior.

## Acceptance-criteria status

| # (from #17) | Status |
|---|---|
| Reproducible transcripts (round-trip, structured-output, tool-allowlist, session-reuse, ≥2 forced errors) | **Partial** — real transcripts for command flow, ≥2 forced errors (missing-key, invalid-model/403), session/multiplex, fallbacks; happy-path round-trip + structured-output need a credential (runbook) |
| Transient-vs-permanent classification table w/ raw pi output | **Done** — three channels documented with captured raw output |
| Node story (min version, one install path, missing behavior) | **Done** — Node-free binary (production) + npm ≥22.19.0 (sandbox); `pi: command not found` recorded |
| go/no-go/unknown per {structured output, session, error taxonomy, usage, streaming} | **Done** — verdict table; unknowns listed as open questions |
| Probe runs against installed pi with a one-line invocation | **Done** — see [How this was run](#how-this-was-run) |

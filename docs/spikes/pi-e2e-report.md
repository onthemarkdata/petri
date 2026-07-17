# pi end-to-end report — running Petri v0.3.6 on the pi harness

> Follow-on to the [pi RPC spike](pi-rpc.md) (#17). The spike validated the protocol;
> this report answers the next question: what actually happens when the whole Petri
> pipeline — seed, decompose, 13-agent grow — runs on pi against a live model.
> Scratch code only; none of it is meant to merge. Everything cited here is in
> [`docs/spikes/e2e/`](e2e/), with claim-by-claim pointers in the
> [evidence report](pi-e2e-evidence.md).

**Setup.** pi 0.80.9 (npm install — the dependency we don't want; the Node-free
standalone binary stays the production path), Claude Max subscription via pi's
OAuth `/login`, model `anthropic/claude-sonnet-4-6` throughout. The integration is
one scratch file: `PiInferenceProvider` subclasses `ClaudeCodeProvider` and overrides
only `__init__` and `_ask`, so every prompt builder and parser is reused. Test claim
for all runs: *"Regular expressions cannot parse arbitrary HTML."*

## The runs

| Run | Config | Real calls | Timeouts | Cost | Seed | Grow outcome |
|-----|--------|-----------|----------|------|------|--------------|
| A | no web tools, concurrency 2, temp dish | 80 (cap) | 9 | $3.85 | 16 cells, 15 edges | 0 validated; 2 deferred, 11 needs_human |
| B | + WebSearch/WebFetch extension, concurrency 2, temp dish | 80 (cap) | 1 | $3.61 | 16 cells, 15 edges | 0 validated; 13 needs_human |
| C | + tools, **concurrency 1**, persistent dish | 27 real, then credit exhaustion | 1 | $1.13 | 16 cells, 15 edges | 0 validated; 13 needs_human (see F7) |

All three runs capped at 80 provider calls (`PI_MAX_CALLS` guardrail) to bound spend.
The pipeline wanted more — runs A and B each rejected 40+ calls past the cap, so
late cells stalled on guardrail errors rather than genuine verdicts. Early cells got
the full treatment; read the outcomes with that split in mind.

Total spend across everything (runs, smoke tests, killed partials): **~$9.20** of
Max extra-usage credit.

## Findings

### F1 — The integration seam is one method
Overriding `_ask` (prompt string in, assistant text out) was the entire port. All
prompt construction, JSON extraction, verdict parsing, and EXECUTION_ERROR handling
came along for free. This is direct evidence for the M1 plan: the transport really is
the only backend-specific layer, and `PiModel` + the bridge should stay that small.

### F2 — pi's error surface is mode-dependent
The same failure (no API key) arrives three different ways: in-band
`success:false` in `--mode rpc`, stderr + exit 1 in `--mode json`, and stderr in
`-p`. A provider-level 400 arrives in-band as `message.stopReason:"error"` with the
command still reporting success. The first draft of the scratch transport mislabeled
the json-mode case as "empty response" and swallowed the real diagnostic — the exact
silent-failure shape v2's error taxonomy (M1-harness.3) exists to kill. The
classification table in the spike doc now covers both surfaces.

### F3 — Decomposition on pi is good
Every run produced a coherent 16-cell, 3-level colony: non-regularity of HTML,
regex ↔ finite automata equivalence, the pumping lemma, parsing-vs-matching, and a
counterargument cell for PCRE/recursive engines. Subjectively the trees are as good
as the claude-CLI baseline — and one cell_lead summary (run C, call 27) correctly
separated "non-regularity is proven" from "the strictly-harder-than-CFL sub-claim is
architecturally plausible but formally unproven," which is exactly the kind of
discrimination the pipeline is supposed to produce.

### F4 — Web tools change evidence quality, not convergence
pi has no built-in web search, so run A's research agents ran blind against prompts
that demand WebSearch/WebFetch and threaten memory-cited URLs as fabricated. Giving
pi the tools (a `registerTool` extension serving a pre-gathered, real-URL corpus —
the sandbox blocks live egress) fixed what it should: timeouts dropped 9 → 1,
freshness_checker actually verified every URL it cited, and verdicts moved from
evidence-starved to evidence-based (`EVIDENCE_SUFFICIENT`, `STALE_BUT_HOLDS`).
It did not move convergence: 0 cells validated in both A and B.

### F5 — Convergence is blocked by role calibration, not plumbing (the headline)
In run B the fully-processed cells burned all 3 circuit-breaker iterations and
escalated, because two blocking specialists never left their block sets:

- **skeptic → `CRITICAL_FLAW_FOUND`, 7/7 assessments.** An untuned Sonnet playing
  skeptic finds a critical flaw in every premise — including well-evidenced,
  true ones — and never concedes.
- **dependency_auditor → `UNVALIDATED_DEPS`, 7/7.** Bottom-up deadlock: parents
  can't pass while children are unvalidated, and nothing validates, so the block
  propagates up the DAG.

Mechanical convergence needs all 6 blocking verdicts in their pass sets
simultaneously; with these two pinned, it is unreachable regardless of evidence.
This is live confirmation of the VISION thesis: role prompts alone don't produce
calibrated behavior, and the adversarial roster needs either tuned models
(Petri Model), calibrated verdict thresholds, or convergence-policy changes before
any colony converges on a stock model. Filed against M2's roster/debate work rather
than M1 — the harness did its job.

### F6 — Fail-loud held everywhere
Across ~240 provider calls and four distinct failure modes (missing key, invalid
model, 300s timeouts, credit exhaustion), not one failure was coerced into a
verdict. Everything surfaced as a typed error → EXECUTION_ERROR → escalation. The
silent-PASS bug class stayed dead, which was the point of the invariant.

### F7 — Run C is a credit-exhaustion case study, not a replication
Run C completed 27 real calls ($1.13), then the account's extra-usage balance hit
zero and every subsequent call returned
`400 "You're out of extra usage. Add more at claude.ai/settings/usage"`. The
pipeline kept going: errors became EXECUTION_ERROR verdicts, cells stalled to
needs_human, and the run exited cleanly with a persisted, inspectable dish. Treat
its convergence numbers accordingly — F5 rests on run B. Incidentally this is the
strongest durability datapoint of the day: mid-run budget death produced no
corruption and no false state, only escalations. (Also a product note: subscription
OAuth on pi bills as per-token extra usage, not plan usage — an exhaustible pot is
the wrong funding model for a pipeline this call-hungry.)

### F8 — v1's concurrent processor hung; serial didn't
With `max_concurrent=2` against the persistent dish, grow stalled twice at the same
place — worker threads parked in `poll()` inside subprocess coordination, zero CPU,
no in-flight pi process — while a single serial call completed in 9.4s. Dropping to
`max_concurrent=1` cleared it completely. Not diagnosed to root cause; consistent
with the migration's judgment that the hand-rolled concurrent processor + fcntl
queue is the component to replace (M4-dbos), not to debug.

### F9 — Cost shape: ~$0.045/call, and sessions are the lever
Steady-state cost was ~$0.045 per call across runs, with one-time prompt-cache
writes dominating (run A: 492k cacheWrite tokens of 746k total). The scratch
transport spawns a fresh `--no-session` pi process per call, so every call re-pays
the cache write. pi has persistent sessions and fork; a real PiModel that holds a
session per cell (or per colony) should cut cost several-fold. At the observed
~9+ calls per cell per iteration, an uncapped 16-cell colony on flat Sonnet is a
$5–10 claim — which is the concrete argument for role-routing cheap/local models
(VISION's model pluralism) once the harness supports it.

## What this changes for the migration

1. **M1 as planned, no scope growth.** Transport-only integration works; keep
   PiModel thin. Add the mode-dependent error table (F2) to M1-harness.3's inputs.
2. **M2 gains a hard requirement:** convergence calibration. A stock model cannot
   converge a colony under the default block sets (F5). The capability manifest
   (search-requiring roles refuse tool-less backends) is confirmed necessary but
   not sufficient.
3. **M4's replacement target is validated by failure** (F8) — the v1 concurrent
   path is the thing that hangs.
4. **M5 cost attribution has real numbers to test against** (F9), and pi's
   per-message usage reporting proved reliable enough to be the sole
   instrumentation point.
5. **Product/docs:** document pi subscription auth = extra-usage billing (F7), and
   keep the standalone binary as the only recommended install.

## Artifacts

Everything under [`docs/spikes/e2e/`](e2e/):

| Path | What |
|------|------|
| `code/pi_provider_scratch.py` | The scratch provider + fail-loud `pi_ask` transport |
| `code/pi_e2e_run.py` | Driver: dish → seed → grow → summary |
| `code/run_persistent.sh`, `code/pi_tool_smoke.py` | Run C launcher; tool smoke test |
| `extension/web_tools.ts`, `extension/petri_web_corpus.json` | pi WebSearch/WebFetch extension + real-URL corpus |
| `logs/run-A-notools.log` · `run-B-webtools.log` · `run-C-serial-persistent.log` | Full transcripts: every prompt, response, error, usage dict |
| `dish/` | Run C's persisted colony: per-cell `events.jsonl`, `evidence.md`, `metadata.json`, `colony.json` |

The dish is also live at `.petri/` (gitignored): `uv run petri check` shows the
colony; the 13 needs_human cells are waiting for exactly the human-review pass the
pipeline escalated them for.

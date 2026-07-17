# pi end-to-end evidence report

Companion to [pi-e2e-report.md](pi-e2e-report.md). Each finding maps to raw
artifacts under [`docs/spikes/e2e/`](e2e/) with a way to check it yourself —
grep commands run from the repo root. Logs are complete transcripts: every prompt
sent to pi, every response, every error, every per-call usage dict.

Abbreviations: **A** = `docs/spikes/e2e/logs/run-A-notools.log`,
**B** = `run-B-webtools.log`, **C** = `run-C-serial-persistent.log`.

## Run accounting

| Claim | Evidence | Check |
|---|---|---|
| 80 real pi calls in A and B; 80 attempted in C | One `[pi call #N] REQUEST` block per call | `grep -c '] REQUEST' docs/spikes/e2e/logs/*.log` |
| Costs: A $3.85 / B $3.61 / C $1.13 | Per-call usage dicts with `cost.total` | sum `'total': X}` values per log; 79/79/26 dicts respectively |
| Timeouts: A 9, B 1, C 1 | `pi timed out after 300.0s` in ClaudeCLIError messages | `grep -c 'timed out after 300' <log>` |
| Call cap enforced (A: 41 rejected, B/C: 46) | Guardrail error text | `grep -c 'PI_MAX_CALLS guardrail' <log>` |
| Seed: 16 cells, 15 edges every run | Driver seed summary line | `grep 'Seed ->' <log>` |
| Grow outcomes (0 validated anywhere) | Driver summary counters | `grep 'grow\.' <log>` — A: processed 13 / succeeded 2 (the two DEFER_OPEN) / stalled 11; B, C: processed 13 / stalled 13 |

## F1 — one-method integration seam

- `e2e/code/pi_provider_scratch.py` — `PiInferenceProvider` overrides `__init__`
  and `_ask` only; no other ClaudeCodeProvider method is touched.
- Inherited behavior demonstrably ran: A/B/C logs show the untouched prompt
  builders (the 4-step decompose prompt, per-role assess prompts) and the inherited
  EXECUTION_ERROR channel (`claude CLI exited 1` strings emitted by the *parent
  class's* error path wrapping pi failures).

## F2 — mode-dependent error surface

- rpc mode, in-band: `docs/spikes/transcripts/error_missing_key.jsonl` —
  `{"success":false,"error":"No API key found for anthropic..."}` (from the #17 probe).
- json mode, stderr + exit 1: same failure, different channel —
  `grep -m1 -A3 'ERROR channel=process' docs/spikes/e2e/logs/run-A-notools.log`
  shows `pi exited 1: No API key found for anthropic` captured from stderr.
- In-band provider 400 while the command "succeeds":
  `grep -m1 'channel=generation' docs/spikes/e2e/logs/run-C-serial-persistent.log`
  — `message.stopReason:"error"` carrying the 400 body.
- The mislabel-then-fix is visible in `e2e/code/pi_provider_scratch.py` — the
  comment block in `pi_ask` explaining why a bare "empty" parse must not mask a
  nonzero exit.

## F3 — decomposition quality

- Colony structure: `e2e/dish/petri-dishes/demo/colony.json` (16 cells, 15 edges,
  3 levels) and the four level-directories under `e2e/dish/petri-dishes/demo/`.
- The counterargument cell (PCRE/recursive engines):
  `grep -l 'PCRE' e2e/dish/petri-dishes/demo/*/*/metadata.json` — cell 001-005.
- The cell_lead discrimination quoted in the report:
  `grep -A8 'pi call #27] RESPONSE' docs/spikes/e2e/logs/run-C-serial-persistent.log`
  — "The non-regularity sub-claim is proven; the 'strictly harder than CFL'
  sub-claim ... remains formally un[proven]".

## F4 — web tools

- No built-in web search in pi: `pi --help` lists built-ins (`read, bash, edit,
  write`); the #17 spike doc records the gap. The extension that closed it:
  `e2e/extension/web_tools.ts` (registers `WebSearch`/`WebFetch` via
  `pi.registerTool`) + `e2e/extension/petri_web_corpus.json` (8 real-URL sources:
  Wikipedia pumping lemma, Fitch & Friederici 2012, CMU 15-411 notes, ODU CS390,
  johndcook.com, neilmadden.blog, codinghorror.com, HN thread).
- Tools actually used by agents:
  `grep -m2 'confirmed live' docs/spikes/e2e/logs/run-B-webtools.log` —
  freshness_checker: "All 7 distinct URLs confirmed live and accessible via
  WebFetch."
- Timeout drop 9 → 1: compare `grep -c 'timed out' ` on A vs B.
- Evidence-based verdicts appear only in B/C:
  `grep -c 'EVIDENCE_SUFFICIENT\|STALE_BUT_HOLDS' <log>` — 0 in A.

## F5 — convergence blocked by role calibration (run B)

- Skeptic pinned: `grep -c '"verdict": "CRITICAL_FLAW_FOUND"'
  docs/spikes/e2e/logs/run-B-webtools.log` → 7 of 7 skeptic assessments.
- Dependency auditor pinned: `grep -c '"verdict": "UNVALIDATED_DEPS"'` → 7 of 7.
- These are block-set verdicts by config, not interpretation:
  `petri/defaults/petri.yaml` — skeptic `verdicts_block: [CRITICAL_FLAW_FOUND,
  UNADDRESSED_COUNTERARGUMENT]` (line ~203), dependency_auditor `verdicts_block:
  [UNVALIDATED_DEPS, CIRCULAR_REASONING]` (line ~186).
- Full circuit-breaker arcs: every processed cell reports
  `final_state='needs_human', iterations=7` — `grep -o "final_state='needs_human',
  iterations=7" run-B-webtools.log | wc -l` → 13.
- Deepest single-cell record: `e2e/dish/petri-dishes/demo/001-parse-context-means-
  complete-structural/002-html-contextfree-language-strictly/events.jsonl` (248KB —
  every agent verdict, source citation, and iteration for one cell).

## F6 — fail-loud held

- Four failure modes, all surfaced as typed errors, none coerced:
  missing key (`No API key found`, A), invalid model (probe transcripts), timeouts
  (`timed out after 300.0s`, A×9), credit exhaustion (`out of extra usage`, C×~53).
- No false passes: `grep -c 'VALIDATED' <any log>` on final states → 0; every
  terminal state is DEFER_OPEN, STALLED/needs_human, or NEW (unreached).

## F7 — credit exhaustion mid-run (run C)

- Boundary is visible at calls 27→28:
  `grep -A2 'pi call #28] ERROR' docs/spikes/e2e/logs/run-C-serial-persistent.log`
  → `400 ... "You're out of extra usage. Add more at claude.ai/settings/usage"`.
- 26 usage dicts (calls 1–27, one timed out), then zero — the $1.13 total.
- Clean exit + intact dish afterwards: `e2e/dish/` is the post-exhaustion state;
  `uv run petri check` against the live `.petri/` shows 13 needs_human, 3 NEW,
  no corrupted states.

## F8 — concurrent hang, serial fix

- Two stalls at the same spot (grow start, `max_concurrent=2`, persistent dish):
  run C's log predecessor attempts show REQUEST counts frozen at 3–4 with the
  driver alive; thread states captured during diagnosis showed workers in
  `poll_schedule_timeout` with no pi subprocess alive.
- Serial control: single call completed in 9.4s during the stall
  (`e2e/code/pi_tool_smoke.py`), and the `PI_CONCURRENCY=1` run (C) sailed past the
  stall point to 80 attempts. Launcher diff: `e2e/code/run_persistent.sh`
  (`PI_CONCURRENCY=1`).

## F9 — cost shape

- Cache-write domination (run A totals): `input 237, output 131,274, cacheRead
  122,055, cacheWrite 492,291, total 745,857` — the driver's accumulated-usage
  summary at the end of A.
- Fresh process per call: `pi_ask` argv in `e2e/code/pi_provider_scratch.py`
  includes `--no-session`.
- Per-call steady state ≈ $0.045: divide any run's cost by its usage-dict count
  (A: 3.85/79, B: 3.61/79, C: 1.13/26).

## Provenance notes

- All model output in these artifacts came from `anthropic/claude-sonnet-4-6`
  through pi 0.80.9 between 2026-07-16 19:35 and 2026-07-17 01:20 UTC.
- The web corpus was gathered by the supervising agent's live WebSearch on
  2026-07-16 (the sandbox blocks direct egress); URLs are real and were surfaced
  by genuine searches, summaries were written from search-result content. Agents
  could verify URL existence only against this corpus — a limitation to note
  before treating any cell's citations as independently verified.
- Costs are the sums of pi's per-message `cost.total` fields; billed against Claude
  Max extra-usage, not plan limits.

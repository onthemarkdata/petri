# Legacy provider bridge

`HarnessInferenceProvider` adapts Pydantic AI agents to the synchronous
`InferenceProvider` interface. Each instance owns one serialized worker loop, so
pi requests and subprocess cleanup stay on the loop that created them. Use the
provider as a context manager or call `close()` in a `finally` block. A finalizer
also closes abandoned instances. Progress callbacks run on the caller's thread
and report start, completion, or failure; token streaming is not implemented.

The live protocol differs from the frozen #26/#27 descriptions in three places:

- It has seven methods, including `summarize_evidence`.
- `assess_claim_substance` returns a dict with `is_substantive`, `reason`, and
  `suggested_rewrite`. `AssessmentResult` is a role verdict and cannot represent
  that result. The bridge validates a separate `SubstanceAssessment` and returns
  its dict to preserve existing callers.
- `decompose_claim` promises `DecompositionResult`, but the legacy implementation
  returns a `nodes` dict. The bridge validates a small premise schema and returns
  typed cells with temporary `pending-premises` identifiers. Those identifiers
  carry no dish identity; the decomposer must assign real cell identifiers before
  persistence. Models do not choose filesystem identifiers.

The bridge uses the shared prompt bodies without their legacy JSON suffixes.
Pydantic AI supplies the output schema and validation feedback. A malformed
Five Whys result consumes the configured output retries and then raises; only a
validated atomic result returns an empty list. Outputs over the requested limit
also trigger validation feedback rather than being silently truncated.

## CLI and failure behavior

A mapping-valued `harness` block opts a dish into the bridge. The older
`harness: claude-code` string and dishes without a harness block retain the
legacy provider. Click closes bridge resources when a CLI command exits,
including failed seed operations. Library callers still own the context manager.
The execution adapter checks the selected inference backend before grow.

Role assessments accept only the configured agent identity and verdict vocabulary.
Invalid outputs receive validation feedback. Exhausted retries and backend errors
produce `EXECUTION_ERROR` with `failure_detail`; they never select a passing verdict.
The processor preserves that detail in both JSONL and SQLite domain events.

Two small reader changes are necessary despite the frozen #27 exclusion of engine
changes: the decomposer must accept the protocol's typed results, and the verdict
projection/event schema must retain failure details. Without those changes, typed
cells fail on dict access and diagnostic fields disappear before persistence.

All structured bridge calls use prompted output, including direct providers.
The generated premise schema contains the requested `maxItems` limit. The bridge
asks for selected results without intermediate brainstorming, and distinguishes
result data from the JSON schema. This matters for smaller local models: the
initial Gemma run copied schema fields and emitted too many premises. The bridge
rejected those outputs; no parser extracted or repaired a partial result.

## Web research tools

| Backend | WebSearch equivalent in this bridge |
| --- | --- |
| Claude Code | Built-in `WebSearch` and `WebFetch`, when explicitly granted in `harness.tools`. |
| pi | None among its built-in coding tools. A separately implemented extension or domain-tool bridge is required. |
| Direct provider | No web tool is registered by this bridge. Provider capability alone does not enable browsing. |

The [pi protocol spike](https://github.com/onthemarkdata/petri/pull/119) records the
missing built-in web tools; the [local validation](spikes/pi-rpc-local-validation.md)
records the adapter/runtime tested here. A successful decomposition smoke test
does not establish that a backend can perform sourced web research.

## Local end-to-end check

The explicitly enabled test uses Docker Model Runner at
`http://127.0.0.1:12434/engines/llama.cpp/v1` and the exact model
`huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0`.
It creates an isolated dish, runs `seed --no-questions`, checks serialized cells,
and runs `grow --dry-run` to list eligible cells. It does not execute a research
pipeline or contact a paid provider.

Configure pi's `petri-local` provider as described in the
[local model validation runbook](local-model-validation.md), then set `PETRI_PI_BIN` and `PI_CODING_AGENT_DIR` to that installation and
configuration. Run:

```sh
PETRI_RUN_LOCAL_BRIDGE=1 uv run pytest tests/integration/test_bridge_seed.py -m integration -q
```

The direct-provider variant sets the local endpoint and a dummy key within its
test and uses `openai-chat:`. In pydantic-ai 2.64, `openai:` selects the Responses
API, which this Docker Model Runner endpoint does not implement. The test leaves
provider model strings untouched rather than rewriting that prefix.

For a manually authorized hosted-provider check, use an isolated dish with
`harness: {backend: provider, model: anthropic:claude-sonnet-4-6}` and your own
provider credentials, then run the same seed and dry-run commands. That check
incurs provider charges and is not part of the automated local validation.

The typed decomposition runner accepts `agent_runner` and `db_path` explicitly.
Its named agents return `InitialDecomposition` and `WhyExpansion`; malformed
results exhaust validation retries with `DecompositionError`. An atomic stop
requires both `is_atomic: true` and a nonempty `bedrock_reason`. Cells receive
UTC creation timestamps and persist through the database seam. Existing
provider injection remains available for legacy callers and offline fixtures.
The typed path does not parse free text or infer atomicity from an empty result.

Streaming is explicit in `run_decomposition_agent(stream=True)`: capable models
deliver text chunks through the progress callback. Pi currently returns a
completed response, so its bridge uses start/completion progress instead.
Callbacks on the bridge's typed methods run on its owned worker thread.

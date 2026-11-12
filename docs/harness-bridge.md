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
validated atomic result returns an empty list. The decomposer truncates selected
premises in code; count limits do not appear in prompts or output schemas.

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
The premise schema has no `maxItems` quota. The bridge asks for selected results without intermediate brainstorming, and distinguishes
result data from the JSON schema. This matters for smaller local models: the
initial Gemma run copied schema fields and emitted too many premises. Malformed
schema-shaped output is rejected; valid premise lists are capped by the
decomposer. No parser extracts or repairs partial results.

## Web research tools

### OpenAI Chat compatibility profile

For an OpenAI-compatible Chat Completions server that requires one leading
system message, set this dish option explicitly:

```yaml
harness:
  backend: provider
  model: openai-chat:local-model-alias
  output_mode: prompted
  openai_chat_supports_multiple_system_messages: false
```

The pinned Pydantic AI SDK joins consecutive leading system messages with two
newlines, preserving their text and order. Explicit `true` keeps multiple system
messages. Omission or `null` leaves the resolver's existing provider-string path
unchanged. The option accepts only YAML booleans or `null`; it requires
`backend: provider` and an effective `openai-chat:` model with a nonblank identifier.
`harness.model` takes precedence over the top-level `model`. CLI backends, other
model prefixes, and explicit `harness.provider` or nonempty `harness.tools` are
incompatible with this option.

The resolver uses the ordinary OpenAI provider and overlays only this profile
key. Configure the server and authentication through the existing
`OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables; URL path prefixes
are preserved. There is no server detection or API fallback. `openai:` and
`openai-responses:` retain their Responses API selection when the option is unset.
Configuration support does not establish compatibility with a particular server.

This option applies when resolving the selected dish harness model. Per-agent
model selections remain independent: `build_agents` replaces the harness
selection for an agent with its own `model`, so this option does not become a
global override for those agents. Output-mode defaults, retrieval requirements,
source validation, retries, and usage limits are unchanged.

### Backend tool availability

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

The Five Whys instructions test for bedrock before generating children. Each
multi-child expansion must include a `limits` or `rebuts` premise; a single
indispensable premise is exempt. The validator gives one retry for missing
counterarguments or parent restatements. A repeated counterargument failure
raises `DecompositionError`.

Parent restatements use Jaccard similarity of case-folded Unicode word-token
sets, ignoring punctuation. Similarity at or above `0.8` triggers the retry.
Children that still meet that threshold are dropped before persistence, with a
`decomposition_warning` event on the parent when a database path is supplied.
The remaining children must still satisfy the counterargument rule. This guard
compares a child with its parent only; it does not detect cross-colony duplicates.

Decomposition settings include `per_parent_cap` (default `4`) and `level_one_cap`
(default `10`). Both must be positive and are bounded by `max_nodes_per_layer`.
The root uses the level-one cap; later expansions use the parent cap and the
remaining layer capacity. Truncation retains a counterargument when at least two
children fit. The existing depth and layer settings remain safety ceilings,
never prompt targets. Both agents use neutral count guidance: “Let the structure
determine the count; do not pad to a quota; fewer is fine.” The legacy
`max_premises` argument remains accepted for compatibility but is not interpolated
into decomposition prompts or schemas.

### Local decomposition quality check

On November 1, 2026, the direct local Gemma CLI seed used a claim about the
definition of a triangle and diagonal bracing in a timber frame. The first run
created 29 cells, including eight `limits`/`rebuts` cells and no atomic stops.
Making the stop instruction explicit still produced no atomic stops. After
putting `is_atomic` and `bedrock_reason` before `sub_premises` in the output
schema, a depth-two run created 16 cells: one atomic stop with a stored reason
and three counterargument cells. These counts satisfy the smoke's storage
checks, but the model still expanded the triangle definition and called a
mechanical claim atomic. The check demonstrates the typed stop path and
relation persistence; it does not establish reliable semantic bedrock judgment.
The local pi and direct-provider seed/grow-dry-run checks also passed.


### Dish search and reference premises

Both named decomposition agents expose `search_cells`. The tool reads typed cell
candidates from SQLite, ranks positive word-token Jaccard overlap, and breaks ties
by exact phrase and cell ID. It excludes the colony currently being seeded.
Agents must search before proposing each premise. When earlier colonies exist,
validation rejects an output that never called the search tool, including
paraphrased candidates. A `ReferencePremise` contains
an `existing_cell_id` and a nonempty `reason`; unknown IDs or IDs in the current
draft receive validation feedback. Reference IDs must also have appeared in the
current call's search results. New premises exactly matching an earlier claim
(case and punctuation ignored) receive a retry requesting a reference. Valid
references append a `cross_colony` edge
to the existing cell, without cloning or expanding it.

`edge_registry` owns edge inserts and exposes append/read operations. Replaying
an edge keeps its original reason and creation time. It does not perform
dish-wide cycle detection; lifecycle work owns that check and future tombstones.
No dish-level edge side file is written. Legacy draft cancellation still clears
unapproved draft rows through the colony storage layer.

Pi and Claude Code carry host function calls in complete JSON envelopes. Pydantic
AI validates the arguments, executes registered tools, and returns results to
the model. These host tools are separate from each CLI's built-in tool allowlist.
Malformed envelopes remain text and receive normal structured-output validation;
no parser extracts a call from surrounding prose.


The November 1 local two-colony smoke did not satisfy the live reference gate.
Pi first duplicated an existing premise without searching. With duplicate
validation enabled, it exhausted retries after inventing a reference ID. The
direct local Gemma run paraphrased premises and returned no cross-colony edge.
Offline scripted tool calls verify the search/reference/storage path, including
a real fake-pi subprocess tool round trip; they do not establish that this local
model reliably chooses the tool. The live reference acceptance check remains
open.

### Required search and final-result formatting

The raw local responses exposed two separate failures: nested child expansions
that violated the flat schema, and required function calls returned as ordinary
JSON text. The premise union now selects its validation branch from the input's
keys, so errors describe the matching shape instead of unrelated missing fields.
Instructions also state that children cannot contain nested expansions.

When earlier colonies exist, the first request requires `search_cells`. That
request omits final-result JSON mode and its schema instructions; after the tool
returns, prompted JSON mode and the final-result schema are restored. Pydantic
validation and the reference-ID checks still apply. This distinction matters
for the tested llama.cpp endpoint: combining required tools with JSON-object
response mode returned text; a tool-only request returned a real function call.
A subsequent direct local Gemma CLI smoke seeded two colonies and persisted a
reference to the first colony's level-one cell with its reason. This closes the
specific direct-provider reference smoke; it does not imply that all quality
Dataset cases pass.

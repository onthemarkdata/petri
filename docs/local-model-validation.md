# Local model readiness

This optional harness checks real inference through a loopback OpenAI-compatible
server or Docker Model Runner, directly and through Pi. Run it explicitly when
validating a model integration; the normal test suite remains offline. Readiness
does not establish seed, grow, feed, dashboard, export, or recovery acceptance.

## Generic local endpoints

Supply the exact API base, request model ID, context limit, response cap, and a
JSON provenance manifest. For example, save this public template outside the
repository and replace its placeholders with the deployed artifact's identity:

```json
{
  "artifact": "organization/model-repository",
  "revision": "immutable-upstream-revision",
  "model_name": "local-model-alias",
  "response_model": "local-model-alias",
  "provider": "local-server"
}
```

`model_name` records the operator's request alias; `artifact` and `revision`
identify the declared weights. If completion responses use a different served
ID, set `response_model` to that exact ID. A name returned by the server does not
prove a weight digest. Generic verification reports the declaration without
claiming independent weight verification. Optional `weights_sha256` and `files`
remain supplied provenance. Optional `chat_template_kwargs` are recorded and
applied to requests; generic endpoints receive no Gemma-specific defaults.

```sh
python3 scripts/e2e/model_runner.py check \
  --base-url http://127.0.0.1:8000/prefix/v1 --model local-model-alias \
  --provenance /tmp/local-model/provenance.json --context-size 262144 \
  --max-tokens 4096 --no-auth --output /tmp/local-model/readiness.json
```

Use `--no-auth` only after confirming that route requires no authentication.
For bearer auth, use `--api-key-env VARIABLE_NAME` to read a protected environment
value; an existing `OPENAI_API_KEY` is also supported. For HTTP Basic, use
`--basic-auth-file /absolute/private/client.toml`, containing top-level `username`
and `password` fields. The file must be owned by the current user with mode
`0600`. A provenance manifest may instead supply a `credential_file` reference;
explicit auth flags take precedence. Never put credential values in the URL,
manifest, command arguments, or public examples. Credentials are retained in
memory for requests. Passwords, bearer keys, authorization values, and relay
keys are removed from decoded diagnostic values before serialization, including
JSON nested inside text. Protocol keys and configured public model IDs are
preserved. Basic usernames are removed from explicit `username` fields; their
occurrence as an ordinary word or part of a public model ID is preserved.

Only HTTP loopback URLs are accepted. Embedded credentials, remote hosts, query
strings, and fragments are rejected before requests. The API path prefix is
preserved. Requests ignore proxy environment variables and refuse redirects,
including redirects to another loopback endpoint. There is no hosted fallback.
Generic servers need neither DMR metadata nor Gemma's lock; `configure` is a
DMR-only action. Use the server's configured context maximum, independently of the output cap.
Every supplied manifest `max_model_len` and `context_size` must match
`--context-size`; conflicting declarations are rejected before requests. Both
supplied declarations are recorded in provenance. Context remains an operator
declaration for generic servers; the
harness does not change their deployment configuration.

Readiness reports each plain-completion, JSON-schema, function-call, and
function-result-replay capability separately. A failed or unsupported mode
remains visible even when later checks pass. HTTP 501 is recorded as
`unsupported`; other HTTP errors retain their status and body as failures.
Every required capability must pass, with complete output, the declared response
model, exact tool arguments, and the replayed result. There is no automatic
output-mode fallback. A separate Petri run may select a supported output mode
while preserving these capability results.

Reports include the upstream identity, configured limits, configuration hash,
source SHA, timestamps, command outcome, and diagnostic copies of responses.
Failed checks retain sanitized response diagnostics. `--output` creates a new file exclusively so an earlier
result cannot be overwritten.

## DMR/Gemma adapter and provenance

The selected artifact is [Google's Gemma 4 E2B instruction-tuned QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf).
[The lock](../models/gemma.lock.json) records the Hugging Face revision, file
SHA-256 hashes and Docker Model Runner content ID. This is Google's artifact;
the reference workshop uses a different, Unsloth-converted Q4_K_M artifact.
The weights are about 3.35 GB; the upstream projector adds about 0.99 GB.
The checks exercise text only.

The CLI pull reference retains the upstream capitalization. The API normalizes
the repository name to lowercase but retains the `Q4_0` tag. Use the model ID
from the lock when configuring clients.

## Start and verify

Install Docker Model Runner using [Docker's setup instructions](https://docs.docker.com/ai/model-runner/get-started/).
On macOS, enable Docker Desktop's native Model Runner and host TCP support on
port 12434. Native inference can use Metal; a Linux container on Docker Desktop
uses CPU inference. These instructions reuse the host runner and leave other
projects' services intact.

```sh
docker model pull hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:Q4_0
python3 scripts/e2e/model_runner.py verify
python3 scripts/e2e/model_runner.py configure
python3 scripts/e2e/model_runner.py check --runs 3 --output /tmp/gemma-direct.json
```

The tag is a download locator, not an integrity guarantee. Verification compares
the local content ID to the lock and fails if upstream content changes. Do not
update the lock merely to bypass that failure: compare the pinned upstream
revision and file hashes first. No model weights belong in Git.

Configuration sets a 32,768-token context, enables the embedded Jinja template,
and disables thinking. It applies only to this model. Checks require three exact
readiness responses, a schema-constrained JSON result, a function call with exact
arguments, and a final answer grounded in the supplied tool result. A download,
HTTP 200 response or truncated completion cannot pass readiness.

The [OpenAI-compatible API](https://docs.docker.com/ai/model-runner/api-reference/)
base is `http://127.0.0.1:12434/engines/llama.cpp/v1`. No paid provider or API key
is needed. The harness accepts loopback HTTP endpoints only.

## Exercise pi without changing personal settings

The validated pi version is pinned in the model lock. Install it in a temporary
prefix, then run a read-only tool check:

```sh
pi_runtime=$(mktemp -d)
npm install --prefix "$pi_runtime" --no-audit --no-fund @mariozechner/pi-coding-agent@0.73.1
python3 scripts/e2e/model_runner.py pi-check \
  --pi "$pi_runtime/node_modules/.bin/pi" --output /tmp/gemma-pi.json
```

The package is now deprecated in favor of `@earendil-works/pi-coding-agent`;
this harness keeps the tested version until an upgrade is validated. The test
creates an isolated `PI_CODING_AGENT_DIR`, disables startup telemetry, extensions,
skills and context-file discovery, and permits only pi's `read` tool. It requires
an actual tool call and the fixture's exact verification code. Temporary session
configuration and the fixture are removed after the run.

For a later Petri workflow run, generate a separate configuration:

```sh
pi_agent=$(mktemp -d)
python3 scripts/e2e/model_runner.py pi-config --directory "$pi_agent"
export PI_CODING_AGENT_DIR="$pi_agent"
export PI_OFFLINE=1 PI_TELEMETRY=0
export PATH="$pi_runtime/node_modules/.bin:$PATH"
```

Use provider `petri-local` and the selected model ID. Generic endpoints use the
same connection flags shown above with `pi-config` or `pi-check`. The generated
`models.json` records the exact upstream URL and selected context/output limits.
The command refuses to overwrite an existing file. Bearer keys are environment
references; Basic auth uses the `PETRI_LOCAL_AUTHORIZATION` header reference,
which a direct Pi caller must populate privately. This file can also serve as a
reviewable upstream configuration without launching Pi.

`pi-check` and the grow helper generate a second isolated runtime configuration
for a task-owned loopback relay. The relay preserves the upstream path prefix,
adds protected auth in memory, checks request and response model identity, and
refuses upstream redirects before Pi or the provider SDK can follow them. Its
temporary key is passed through the environment and never written to a file.
Original successful response bytes are validated and forwarded unchanged.
Before credential inspection, streamed text is assembled by choice and tool
arguments by choice/tool identity, including interleaved deltas. Rejected streams
retain sanitized logical diagnostics without raw fragments; if logical assembly
fails, raw SSE is omitted from diagnostics. Responses that echo protected
authentication data are rejected with a generic error and a sanitized diagnostic
copy, rather than rewritten into a research
result. HTTP-error bodies are diagnostic copies. Responses, including SSE,
are buffered; the upstream timeout is 300 seconds. This transport supports
Chat Completions only.

SSE requires successful terminal `stop` or `tool_calls` reasons for every
observed choice and the final `[DONE]` marker. Usage-only chunks, comments,
multiline data, and LF/CRLF/CR framing are supported. EOF, missing terminal
reasons, truncation, and failed finishes are rejected with retained diagnostics.
Pi readiness requires a final assistant with `stopReason: stop`, plus the
read-call/tool-result/final-answer sequence; a matching partial answer cannot pass.

The relay forwards only caller-supplied `traceparent`, `tracestate`,
`x-request-id`, `x-petri-run-id`, `x-petri-workflow-id`, and `x-petri-cell-id` by
default. Python callers may extend this with
`guarded_route(correlation_allowlist={"x-custom-run"})`. Header values must be
printable ASCII and at most 4,096 characters. Authentication and cookie headers
remain excluded even when requested; upstream authentication comes from the
protected route configuration. The relay creates no correlation IDs and makes
no claim that a client already supplies them.

Keep test configuration outside the repository and delete it after collecting
the results. Petri workflow checks must use isolated research
data and separately verify seed, grow, evidence updates, dashboard,
export/backup/restore and recovery. A passing model check establishes none of
those application outcomes.

## Recorded checks

The committed [direct API evidence](validation/gemma-direct.json) and
[pi evidence](validation/gemma-pi.json) contain generated public test inputs and
results only. The local validation used native llama.cpp Metal backend `ac4cdde`,
backend digest `sha256:aa3e239c5737a7fa56f3908a865eb75dfae214b9cfec2cdd929e4d5353c2d714`,
and Docker Model Runner plugin v1.1.29. The model is pinned; Docker Desktop owns
backend upgrades, so record backend identity with each new validation run.
These short checks do not measure full-workflow memory use or research quality.

The opt-in `tests/unit/test_cli_agents_run.py` test also passed against this
model using the shipped skeptic instructions, constitution, typed output, and
usage reporting. A repeated invocation is recorded in
[gemma-agent-cli.json](validation/gemma-agent-cli.json): one request, 1,108 input
tokens, and 173 output tokens. Both runs used prompted JSON output. This verifies
a single-agent CLI run; it does not establish that a multi-agent research workflow
passes end to end.

# Local pi RPC validation

The transport was checked on 2026-11-01 with
`@mariozechner/pi-coding-agent` 0.73.1 and local Docker Model Runner serving
`huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0`.
This supplements the earlier #17 spike, which probed
`@earendil-works/pi-coding-agent` 0.80.9 without working model credentials.

An explicit integration run of `test_real_pi_roundtrip` passed against the local
model: one test passed in 1.15 seconds. It exercised subprocess startup, the
`set_auto_retry` command, prompt acceptance, successful assistant output, and
shutdown. It does not establish cross-version compatibility for every pi command.

The implementation follows the installed package's `docs/rpc.md`:

- Commands and responses use JSON objects separated by LF. The reader splits on
  the LF byte; embedded newlines and Unicode line separators stay inside JSON
  strings. A trailing CR before LF is accepted as JSON whitespace.
- A successful `prompt` response acknowledges acceptance. Output continues until
  `agent_end`, whose `messages` contain the completed assistant output and usage.
  Turn events have no request ID, so a transport serializes concurrent calls.
- Provider failures appear as assistant messages with `stopReason: "error"` and
  `errorMessage`. A rejected command instead has `success: false` and `error`.
- Petri sends `set_auto_retry` with `enabled: false` at each process start to avoid
  nesting pi retries underneath the shared Petri retry policy.
- The local generation confirms that completed assistant output and usage are
  available. Native JSON-schema enforcement remains unverified; PiModel uses
  prompted JSON and Pydantic AI validation. Streaming remains outside this adapter.

Run the explicit local integration check after configuring pi's local provider:

```sh
PETRI_PI_BIN=/path/to/pi \
PETRI_TEST_PI_PROVIDER=your-local-provider \
PETRI_TEST_PI_MODEL=your-local-model \
uv run pytest tests/unit/harness/test_pi_transport.py -m integration -q
```

The test skips when the executable or explicit provider/model settings are
missing. Routine tests use `fake_pi`; they cover framing, rejection, provider
errors, malformed/truncated frames, timeout, cancellation, process death and
restart, and lock serialization without model access.

## PiModel and structured output

`test_real_pi_structured_output` passed against the same local model: one test
passed in 1.28 seconds. Pydantic AI returned a validated object with `status`
set to `ready`. This validates prompted JSON output; it does not establish a
provider-native schema capability.

The implementation inspected the installed, lockfile-resolved Pydantic AI
**2.64.0**. An isolated inspection of 2.9.0 confirmed the same abstract members,
but 2.9 lacks `ModelRequestParameters.declared_function_tools`, which this
adapter uses. The dependency floor is therefore `>=2.64,<3`; the lockfile stays
on the tested 2.64.0 version. The abstract
`Model` members in the installed package are:

```python
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse: ...

@property
def model_name(self) -> str: ...

@property
def system(self) -> str: ...
```

PiModel calls `prepare_request` and advertises prompted structured output, with
native schemas and Pydantic function tools disabled. A pi tool allowlist remains
available through the constructor. Unsupported per-request settings fail with an
explanation rather than being silently ignored. Streaming is not implemented.

Pydantic AI owns the complete conversation history. PiModel resets the pi
conversation before each replay, while reusing its process. This keeps validation
feedback and prior messages without duplicating them in pi's internal history.
The entire reset-and-prompt operation is serialized per model instance.

The live run exposed a pi 0.73.1 behavior: `new_session` writes session files even
when the process started with `--no-session`. The transport now supplies a
private temporary session directory, enables persistence only there, and deletes
it on shutdown. Combining `--no-session` with `--session-dir` does not work in
this pi version: the in-memory session discards the directory before reset. No inference
transcripts belong in the source tree.

Usage is collected from every assistant message in `agent_end`, including turns
before pi tool calls. Input counts include cache reads and writes; output counts
sum the reported output tokens. Missing counts remain zero, and the response
marks usage as unreported. Counters are not estimated from text.

After the temporary-directory fix, one follow-up returned `success` for a field
whose schema allowed any string. The smoke test now constrains that field to
`Literal["ready"]`; the local run passed in 1.04 seconds. This checks the intended
value through schema validation, not just JSON syntax. The corrected run left no
session files in the worktree.

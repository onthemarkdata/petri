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

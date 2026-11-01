# Local Gemma validation

This optional harness checks real inference through Docker Model Runner and pi.
It does not add Docker to Petri's runtime dependencies. Run it explicitly when
validating a model integration; the normal test suite remains offline.

## Model provenance

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

Use provider `petri-local` and the lock's model ID. The command refuses to
overwrite an existing `models.json`. Keep this directory outside the repository
and delete it after testing. Petri workflow checks must use isolated research
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

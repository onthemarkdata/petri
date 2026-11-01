# Testing Petri

Run the default suite without a model account, Node.js, or pi:

```sh
uv sync --locked --extra test
uv run pytest tests/ -q -m 'not integration'
```

Use the smallest boundary that can verify the behavior. Existing v1 engine tests
continue to use `FakeProvider`; v2 inference tests use the three tiers below.

## 1. TestModel: output shape

Pydantic AI's `TestModel` creates schema-valid output without inference. Use it to
check an agent's wiring and typed output. The shared factories in
`tests/harness_utils.py` also provide Petri-shaped results.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from petri.models import AssessmentResult


def test_assessment_shape():
    model = TestModel(custom_output_args={"agent": "skeptic", "verdict": "PASS"})
    result = Agent(model, output_type=AssessmentResult).run_sync("Assess this claim")
    assert result.output.verdict == "PASS"
```

## 2. FunctionModel: scripted agent turns

Use `FunctionModel` for tool calls and validation retries. A scripted invalid
output followed by a valid output verifies that the agent retries validation:

```python
from pydantic_ai import Agent
from petri.models import AssessmentResult
from tests.harness_utils import make_scripted_output_model


def test_assessment_validation_retry():
    model = make_scripted_output_model([
        {"summary": "Missing required fields"},
        {"agent": "skeptic", "verdict": "PASS"},
    ])
    result = Agent(model, output_type=AssessmentResult, retries=1).run_sync("Assess")
    assert result.output.verdict == "PASS"
```

Run these two tiers with `uv run pytest tests/unit/harness/test_offline_models.py -q`.

## 3. fake_pi: process and framing behavior

Use `fake_pi` for process lifecycle, LF framing, timeout, and protocol failures.
It is a Python-only executable implementing the relevant subset of pi 0.73.1's
RPC contract. A successful `prompt` response acknowledges acceptance;
`agent_end` completes the run. Events do not carry request IDs.

The fixture returns an executable path and a scenario writer. Calling the writer
sets `FAKE_PI_SCENARIO` for subprocesses in that test:

```python
import json
import subprocess
import sys


def test_pi_response(fake_pi):
    path, script = fake_pi
    script({"text": "first\nsecond"})
    result = subprocess.run(
        [sys.executable, str(path), "--mode", "rpc"],
        input='{"id":"one","type":"prompt","message":"hello"}\n',
        text=True, capture_output=True, timeout=5, check=True,
    )
    final = json.loads(result.stdout.splitlines()[-1])
    assert final["messages"][0]["content"][0]["text"] == "first\nsecond"
```

Run this tier with `uv run pytest tests/unit/harness/test_fake_pi.py -q`.

The standalone default needs no scenario file:

```sh
printf '%s\n' '{"id":"one","type":"prompt","message":"hello"}' | python3 tests/fixtures/fake_pi.py --mode rpc
```

For a custom run, put a JSON array in a file and pass `--scenario path.json` or
set `FAKE_PI_SCENARIO=path.json`. Each prompt consumes one entry; the final entry
repeats for later prompts. Other commands receive a success response without
consuming an entry. The supported entries are:

| Entry | Behavior |
|---|---|
| `{"text":"answer"}` | Accepted prompt, text delta, completed message, `agent_end` |
| `{"kind":"error","error":"invalid model"}` | Rejected command response |
| `{"kind":"error","after_accept":true,"error":"network timeout"}` | Accepted prompt followed by an assistant error |
| `{"kind":"rate_limit","reset_hint":"try again in 3 minutes"}` | Assistant rate-limit error; omit `reset_hint` to test an unknown wait |
| `{"delay":0.2,"text":"later"}` | Delay after acceptance, before the completed output |
| `{"delay":30,"ignore_sigterm":true}` | Ignore termination so shutdown tests must escalate to kill |
| `{"kind":"death","exit_code":7}` | Exit after acceptance, before completion |
| `{"kind":"malformed","raw":"not-json"}` | Emit an invalid JSON line |
| `{"kind":"malformed","raw":"{\"type\":","newline":false}` | Emit a truncated frame without LF |
| `{"kind":"frames","frames":[{"type":"custom"}]}` | Emit the supplied JSON frames exactly |

Any entry can set `stderr` to write diagnostics. The fixture allows only this
stub through the harness subprocess guard, rejects shell launches and executable
substitution, and leaves the network and model API guards enabled. Keep these
process tests in the offline tier.

## Explicit live integration tests

Tests requiring a real CLI or model use `@pytest.mark.integration`. Run them only
when the needed runtime and model are configured:

```sh
uv run pytest tests/ -m integration -q
```

That marker bypasses the harness offline guard. It does not install pi, configure
a model, or authorize paid inference. Live pi tests should skip when pi is
unavailable and accept an explicit local model configuration. Save the model
identifier, runtime version, commands, and results for end-to-end validation.

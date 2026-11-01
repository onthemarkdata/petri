"""PiModel exercises the shared fake process and Petri-shaped output fixtures."""

import json
import os
import shutil
from typing import Literal

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from petri.harness.errors import PermanentHarnessError, RateLimitedError
from petri.harness.pi.model import PiModel
from petri.harness.messages import render_history
from petri.harness.pi.transport import PiRpcTransport
from petri.harness.retry import RetryPolicy
from petri.models import AssessmentResult
from tests.harness_utils import CANNED_ASSESSMENT_ARGS, INVALID_ASSESSMENT_ARGS


def model_for(fake_pi, **kwargs):
    return PiModel("fake-pi", provider="fake", transport=PiRpcTransport(binary=str(fake_pi.path)), **kwargs)


@pytest.mark.asyncio
async def test_plain_roundtrip_and_usage(fake_pi):
    fake_pi.scenario({"text": "hello"})
    async with model_for(fake_pi) as model:
        result = await Agent(model).run("Say hello")
        assert result.output == "hello"
        response = result.response
        assert response.model_name == "fake-pi"
        assert response.provider_name == "fake"
        assert response.usage.input_tokens == 1
        assert response.usage.output_tokens == 1
        assert response.provider_details["usage_reported"] is True


@pytest.mark.asyncio
async def test_typed_output(fake_pi):
    fake_pi.scenario({"text": json.dumps(CANNED_ASSESSMENT_ARGS)})
    async with model_for(fake_pi) as model:
        result = await Agent(model, output_type=AssessmentResult).run("Assess")
    assert result.output == AssessmentResult(**CANNED_ASSESSMENT_ARGS)


@pytest.mark.asyncio
async def test_validation_retry(fake_pi):
    fake_pi.scenario(
        {"text": json.dumps(INVALID_ASSESSMENT_ARGS)},
        {"text": json.dumps(CANNED_ASSESSMENT_ARGS)},
    )
    async with model_for(fake_pi) as model:
        result = await Agent(model, output_type=AssessmentResult, retries=1).run("Assess")
        assert model.round_trips == 2
    assert result.output.verdict == CANNED_ASSESSMENT_ARGS["verdict"]


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [0, 2])
async def test_rate_limit_retries_exactly_configured_count(fake_pi, retries):
    fake_pi.scenario({"kind": "rate_limit"})
    async with model_for(fake_pi, retry_policy=RetryPolicy(max_retries=retries, base=0, jitter=0)) as model:
        with pytest.raises(RateLimitedError):
            await Agent(model).run("hello")
        assert model.round_trips == retries + 1


def test_tools_map_to_transport(monkeypatch):
    monkeypatch.setenv("PETRI_PI_BIN", "/test/pi")
    assert "--no-tools" in PiModel("gemma").transport.command()
    command = PiModel("gemma", provider="local", tools=["read", "grep"]).transport.command()
    assert command[command.index("--tools") + 1] == "read,grep"
    assert command[command.index("--provider") + 1] == "local"


def test_tool_validation_feedback_is_preserved():
    history = [ModelRequest(parts=[UserPromptPart("Assess"), ToolReturnPart("answer", "Missing verdict")])]
    assert "Missing verdict" in render_history(history, ModelRequestParameters())


def test_missing_usage_is_marked():
    model = PiModel("gemma", provider="local")
    result = model._response({"messages": [{
        "role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "ok"}],
    }]})
    assert result.usage.input_tokens == 0
    assert result.usage.unmeasured_requests == 1
    assert result.provider_details["usage_reported"] is False
    assert result.model_name == "gemma"
    assert result.provider_name == "local"


@pytest.mark.parametrize("message", [
    {"role": "assistant", "stopReason": "toolUse", "content": []},
    {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": 1}]},
])
def test_malformed_output_is_not_coerced(message):
    with pytest.raises(PermanentHarnessError):
        PiModel("fake")._response({"messages": [message]})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_pi_structured_output():
    binary = os.environ.get("PETRI_PI_BIN") or shutil.which("pi")
    provider, model_name = os.environ.get("PETRI_TEST_PI_PROVIDER"), os.environ.get("PETRI_TEST_PI_MODEL")
    if not binary or not provider or not model_name:
        pytest.skip("Configure pi and PETRI_TEST_PI_PROVIDER/MODEL for an explicit live run")

    class Answer(BaseModel):
        status: Literal["ready"]

    transport = PiRpcTransport(binary=binary, provider=provider, model=model_name)
    async with PiModel(model_name, provider=provider, transport=transport) as model:
        result = await Agent(model, output_type=Answer).run("Set status to ready.")
    assert result.output.status.lower() == "ready"


def test_usage_sums_tool_loop_messages():
    messages = [
        {"role": "assistant", "stopReason": "toolUse", "content": [],
         "usage": {"input": 2, "output": 3, "cacheRead": 4, "cacheWrite": 5}},
        {"role": "toolResult", "content": [{"type": "text", "text": "tool result"}]},
        {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "done"}],
         "usage": {"input": 6, "output": 7, "cacheRead": 8, "cacheWrite": 9}},
    ]
    response = PiModel("fake")._response({"messages": messages})
    assert response.usage.input_tokens == 34
    assert response.usage.output_tokens == 10
    assert response.usage.cache_read_tokens == 12
    assert response.usage.cache_write_tokens == 14


@pytest.mark.asyncio
async def test_history_and_validation_feedback_reach_pi(fake_pi):
    class RecordingTransport(PiRpcTransport):
        def __init__(self):
            super().__init__(binary=str(fake_pi.path))
            self.payloads = []

        async def request(self, payload):
            self.payloads.append(payload)
            return await super().request(payload)

    fake_pi.scenario(
        {"text": json.dumps(INVALID_ASSESSMENT_ARGS)},
        {"text": json.dumps(CANNED_ASSESSMENT_ARGS)},
    )
    transport = RecordingTransport()
    async with PiModel("fake", transport=transport) as model:
        await Agent(model, output_type=AssessmentResult, instructions="Use primary evidence.").run("Assess")
    assert [p["type"] for p in transport.payloads] == ["new_session", "prompt", "new_session", "prompt"]
    assert "Use primary evidence." in transport.payloads[1]["message"]
    assert "json" in transport.payloads[1]["message"].lower()
    assert "verdict" in transport.payloads[-1]["message"]
    assert "required" in transport.payloads[-1]["message"].lower()

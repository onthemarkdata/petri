"""Claude adapter tests inject the CLI seam and never invoke a paid model."""

import asyncio
import json
from datetime import UTC, datetime
from subprocess import CompletedProcess
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent

from petri.harness.claude_code.model import ClaudeCodeModel, extract_text_delta, run_cli
from petri.harness.errors import AuthExpiredError, PermanentHarnessError, RateLimitedError
from petri.harness.retry import RetryPolicy
from petri.models import AssessmentResult
from tests.harness_utils import CANNED_ASSESSMENT_ARGS


def result(stdout="", stderr="", code=0):
    return CompletedProcess(["claude"], code, stdout, stderr)


@pytest.mark.asyncio
async def test_structured_output_and_invalid_json_retry():
    runner = AsyncMock(side_effect=[result("not JSON"), result(json.dumps(CANNED_ASSESSMENT_ARGS))])
    model = ClaudeCodeModel(runner=runner)
    response = await Agent(model, output_type=AssessmentResult, retries=1).run("Assess")
    assert response.output == AssessmentResult(**CANNED_ASSESSMENT_ARGS)
    assert runner.await_count == 2
    assert response.response.provider_name == "anthropic"
    assert response.response.usage.unmeasured_requests == 1


def test_command_flags():
    model = ClaudeCodeModel("opus", allowed_tools=["WebSearch", "WebFetch"])
    assert model.command("prompt") == ["claude", "--print", "--model", "opus", "--allowedTools=WebSearch,WebFetch", "prompt"]
    stream = ClaudeCodeModel(stream_json=True).command("prompt")
    assert "--tools=" in stream
    assert stream[-1] == "prompt"
    assert stream[stream.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in stream


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic", ["model overloaded", "network timeout", "503 unavailable"])
async def test_transient_retry(diagnostic):
    runner = AsyncMock(side_effect=[result(stderr=diagnostic, code=1), result("ok")])
    model = ClaudeCodeModel(runner=runner, retry_policy=RetryPolicy(base=0, jitter=0))
    assert (await Agent(model).run("hello")).output == "ok"
    assert runner.await_count == 2


@pytest.mark.asyncio
async def test_permanent_no_retry():
    runner = AsyncMock(return_value=result(stderr="model not found", code=1))
    with pytest.raises(PermanentHarnessError):
        await Agent(ClaudeCodeModel(runner=runner)).run("hello")
    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_stdout_only_reset_hint_stops_retry():
    runner = AsyncMock(return_value=result(stdout="resets 9pm (America/Los_Angeles)", code=1))
    model = ClaudeCodeModel(runner=runner, clock=lambda: datetime(2026, 1, 1, 20, tzinfo=UTC))
    with pytest.raises(RateLimitedError) as raised:
        await Agent(model).run("hello")
    assert raised.value.retry_after_seconds == 9 * 3600
    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_auth_expiry_is_actionable():
    runner = AsyncMock(return_value=result(stderr="authentication token expired; rate limit", code=1))
    with pytest.raises(AuthExpiredError, match="claude auth login"):
        await Agent(ClaudeCodeModel(runner=runner)).run("hello")
    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_stream_result_and_usage():
    frames = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial"}}},
        {"type": "result", "subtype": "success", "result": json.dumps(CANNED_ASSESSMENT_ARGS),
         "usage": {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": 4}},
    ]
    runner = AsyncMock(return_value=result("\n".join(map(json.dumps, frames))))
    response = await Agent(ClaudeCodeModel(runner=runner, stream_json=True), output_type=AssessmentResult).run("Assess")
    assert response.output == AssessmentResult(**CANNED_ASSESSMENT_ARGS)
    assert response.response.usage.input_tokens == 6
    assert response.response.usage.output_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["not-json\n", '{"type":', '{"type":"assistant","message":{"content":[]}}\n'])
async def test_malformed_or_partial_stream_fails(output):
    runner = AsyncMock(return_value=result(output))
    with pytest.raises(PermanentHarnessError):
        await Agent(ClaudeCodeModel(runner=runner, stream_json=True)).run("hello")
    assert runner.await_count == 1


def test_full_assistant_and_delta_shapes():
    assert extract_text_delta({"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}}) == "hello"
    assert extract_text_delta({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "delta"}}) == "delta"


@pytest.mark.asyncio
async def test_runner_reaps_on_cancellation(monkeypatch):
    class Process:
        returncode = None

        def __init__(self):
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            return b"", b""

        def kill(self):
            self.returncode = -9
            self.finished.set()

    process = Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    task = asyncio.create_task(run_cli(["claude"], 300))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -9


@pytest.mark.asyncio
async def test_runner_timeout_reaps_and_keeps_diagnostics(monkeypatch):
    class Process:
        returncode = None

        def __init__(self):
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            return b"partial", b"slow"

        def kill(self):
            self.returncode = -9
            self.finished.set()

    process = Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    from petri.harness.errors import TransientHarnessError
    with pytest.raises(TransientHarnessError, match="timed out") as raised:
        await run_cli(["claude"], 0.01)
    assert process.returncode == -9
    assert raised.value.detail.stdout_tail == "partial"

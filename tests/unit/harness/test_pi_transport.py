"""Offline transport tests launch only the shared fake pi executable."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from petri.harness.errors import PermanentHarnessError, RateLimitedError, TransientHarnessError
from petri.harness.pi.transport import PiRpcTransport


@pytest.mark.asyncio
async def test_roundtrip_and_session_reuse(fake_pi):
    fake_pi.scenario({"text": "first\nsecond\r\nlast"})
    async with PiRpcTransport(binary=str(fake_pi.path)) as transport:
        result = await transport.request({"type": "prompt", "message": "a\nb\r\nc"})
        process = transport.process
        session_dir = Path(transport._session_dir.name)
        assert result["messages"][0]["content"][0]["text"] == "first\nsecond\r\nlast"
        await transport.request({"type": "prompt", "message": "again"})
        assert transport.process is process
    assert process.returncode is not None
    assert not session_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,error", [
    ({"kind": "error", "error": "network timeout"}, TransientHarnessError),
    ({"kind": "error", "error": "429 quota exhausted"}, RateLimitedError),
    ({"kind": "error", "error": "model not found"}, PermanentHarnessError),
    ({"kind": "rate_limit", "reset_hint": "try again in 2 minutes"}, RateLimitedError),
])
async def test_failures(fake_pi, scenario, error):
    fake_pi.scenario(scenario)
    transport = PiRpcTransport(binary=str(fake_pi.path))
    with pytest.raises(error) as raised:
        await transport.request({"type": "prompt", "message": "hello"})
    assert transport.process.returncode is not None
    if scenario.get("reset_hint"):
        assert raised.value.retry_after_seconds == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,newline", [("not-json", True), ('{"type":', False), ('[]', True)])
async def test_malformed_frames(fake_pi, raw, newline):
    fake_pi.scenario({"kind": "malformed", "raw": raw, "newline": newline})
    transport = PiRpcTransport(binary=str(fake_pi.path), timeout=0.2, shutdown_timeout=0.05)
    task = asyncio.create_task(transport.request({"type": "prompt", "message": "hello"}))
    with pytest.raises(PermanentHarnessError) as raised:
        await task
    assert raw in raised.value.detail.stdout_tail
    assert transport.process.returncode is not None


@pytest.mark.asyncio
async def test_timeout_reaps_stubborn_process(fake_pi):
    fake_pi.scenario({"delay": 30, "ignore_sigterm": True})
    transport = PiRpcTransport(binary=str(fake_pi.path), timeout=0.1, shutdown_timeout=0.05)
    with pytest.raises(TransientHarnessError, match="timed out"):
        await asyncio.wait_for(transport.request({"type": "prompt", "message": "hello"}), 2)
    assert transport.process.returncode is not None


@pytest.mark.asyncio
async def test_cancellation_reaps_process(fake_pi):
    fake_pi.scenario({"delay": 30})
    transport = PiRpcTransport(binary=str(fake_pi.path), shutdown_timeout=0.05)
    task = asyncio.create_task(transport.request({"type": "prompt", "message": "hello"}))
    while transport.process is None:
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.process.returncode is not None


@pytest.mark.asyncio
async def test_death_then_restart(fake_pi):
    fake_pi.scenario({"kind": "death", "stderr": "network lost"})
    transport = PiRpcTransport(binary=str(fake_pi.path))
    with pytest.raises(TransientHarnessError):
        await transport.request({"type": "prompt", "message": "hello"})
    old = transport.process
    fake_pi.scenario({"text": "recovered"})
    async with transport:
        result = await transport.request({"type": "prompt", "message": "again"})
        assert result["messages"][0]["content"][0]["text"] == "recovered"
        assert transport.process.pid != old.pid


@pytest.mark.asyncio
async def test_concurrent_calls_serialize(fake_pi):
    fake_pi.scenario({"text": "first", "delay": 0.02}, {"text": "second"})
    async with PiRpcTransport(binary=str(fake_pi.path)) as transport:
        results = await asyncio.gather(*[
            transport.request({"type": "prompt", "message": str(n)}) for n in range(2)
        ])
    assert [r["messages"][0]["content"][0]["text"] for r in results] == ["first", "second"]


def test_binary_and_flags(monkeypatch):
    monkeypatch.setenv("PETRI_PI_BIN", "/test/pi")
    transport = PiRpcTransport(provider="local", model="gemma", tools=["read", "grep"])
    assert transport.command() == [
        "/test/pi", "--mode", "rpc", "--no-session", "--provider", "local",
        "--model", "gemma", "--tools", "read,grep",
    ]
    assert "--no-tools" in PiRpcTransport().command()
    monkeypatch.delenv("PETRI_PI_BIN")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(PermanentHarnessError, match="PETRI_PI_BIN"):
        PiRpcTransport().command()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_pi_roundtrip():
    binary = os.environ.get("PETRI_PI_BIN") or shutil.which("pi")
    provider, model = os.environ.get("PETRI_TEST_PI_PROVIDER"), os.environ.get("PETRI_TEST_PI_MODEL")
    if not binary or not provider or not model:
        pytest.skip("Configure pi and PETRI_TEST_PI_PROVIDER/MODEL for an explicit live run")
    async with PiRpcTransport(binary=binary, provider=provider, model=model) as transport:
        response = await transport.request({"type": "prompt", "message": "Reply with the word READY."})
    assert "READY" in json.dumps(response["messages"])


@pytest.mark.asyncio
async def test_stderr_is_drained_without_pipe_deadlock(fake_pi):
    fake_pi.scenario({"stderr": "x" * 100000, "text": "done"})
    async with PiRpcTransport(binary=str(fake_pi.path), timeout=2) as transport:
        result = await transport.request({"type": "prompt", "message": "hello"})
        assert result["messages"][0]["content"][0]["text"] == "done"
        assert len(transport._stderr) <= 4096


@pytest.mark.asyncio
async def test_invalid_payload_does_not_leak_process(fake_pi):
    transport = PiRpcTransport(binary=str(fake_pi.path))
    with pytest.raises(TypeError):
        await transport.request({"type": "prompt", "message": object()})
    assert transport.process.returncode is not None

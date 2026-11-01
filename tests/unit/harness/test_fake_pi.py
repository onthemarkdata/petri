"""Run the scriptable pi fixture as a real local process, without inference."""

import json
import subprocess
import sys
import time

import pytest


def run_stub(fake_pi, *scenarios, requests=None):
    fake_pi.scenario(*scenarios)
    result = subprocess.run(
        [sys.executable, str(fake_pi.path), "--mode", "rpc"],
        input=requests or '{"id":"one","type":"prompt","message":"hello"}\n',
        capture_output=True, text=True, timeout=5,
    )
    return result


def test_response_and_lf_framing(fake_pi):
    text = "first\nsecond\r\nthird"
    result = run_stub(fake_pi, {"text": text})
    assert result.returncode == 0
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.stdout.endswith("\n")
    assert len(frames) == 5
    assert frames[0] == {"id": "one", "type": "response", "command": "prompt", "success": True}
    assert frames[-1]["messages"][0]["content"][0]["text"] == text
    assert all("id" not in frame for frame in frames[1:])


def test_canned_sequence(fake_pi):
    requests = ''.join(json.dumps({"type": "prompt", "id": n}) + "\n" for n in range(3))
    result = run_stub(fake_pi, {"text": "first"}, {"text": "second"}, requests=requests)
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert [e["messages"][0]["content"][0]["text"] for e in messages if e["type"] == "agent_end"] == [
        "first", "second", "second",
    ]


def test_rejected_request(fake_pi):
    result = run_stub(fake_pi, {"kind": "error", "error": "model not found"})
    assert json.loads(result.stdout) == {
        "type": "response", "command": "prompt", "id": "one",
        "success": False, "error": "model not found",
    }


@pytest.mark.parametrize("hint", [None, "try again in 3 minutes"])
def test_rate_limit_after_acceptance(fake_pi, hint):
    result = run_stub(fake_pi, {"kind": "rate_limit", "reset_hint": hint})
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert frames[0]["success"] is True
    message = frames[-1]["messages"][0]
    assert message["stopReason"] == "error"
    assert "rate limit" in message["errorMessage"]
    assert ("try again" in message["errorMessage"]) == bool(hint)


def test_in_band_transient_error(fake_pi):
    result = run_stub(fake_pi, {"kind": "error", "after_accept": True, "error": "network timeout"})
    message = json.loads(result.stdout.splitlines()[-1])["messages"][0]
    assert message["errorMessage"] == "network timeout"
    assert message["content"] == []


def test_slow_response(fake_pi):
    start = time.monotonic()
    result = run_stub(fake_pi, {"delay": 0.05})
    assert time.monotonic() - start >= 0.05
    assert json.loads(result.stdout.splitlines()[-1])["type"] == "agent_end"


def test_mid_request_death(fake_pi):
    result = run_stub(fake_pi, {"kind": "death", "exit_code": 7, "stderr": "network lost"})
    assert result.returncode == 7
    assert result.stderr == "network lost"
    assert "agent_start" in result.stdout
    assert "agent_end" not in result.stdout


@pytest.mark.parametrize("raw,newline", [("not-json", True), ('{"type":', False)])
def test_malformed_frames(fake_pi, raw, newline):
    result = run_stub(fake_pi, {"kind": "malformed", "raw": raw, "newline": newline})
    assert result.stdout == raw + ("\n" if newline else "")
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_explicit_frames(fake_pi):
    result = run_stub(fake_pi, {"kind": "frames", "frames": [{"type": "custom"}]})
    assert json.loads(result.stdout) == {"type": "custom"}


def test_other_programs_stay_blocked(fake_pi):
    with pytest.raises(RuntimeError, match="only its local stub"):
        subprocess.run([sys.executable, "-c", "print('should not run')"])
    with pytest.raises(RuntimeError, match="only its local stub"):
        subprocess.run([str(fake_pi.path)], shell=True)
    with pytest.raises(RuntimeError, match="only its local stub"):
        subprocess.run([str(fake_pi.path)], executable="/bin/sh")


def test_explicit_scenario_file_and_executable(fake_pi):
    scenario = fake_pi.scenario({"text": "explicit file"})
    result = subprocess.run(
        [str(fake_pi.path), "--scenario", str(scenario)],
        input='{"type":"prompt"}\n', capture_output=True, text=True, timeout=5, check=True,
    )
    assert "explicit file" in result.stdout


def test_stubborn_process_requires_kill(fake_pi):
    fake_pi.scenario({"delay": 30, "ignore_sigterm": True})
    process = subprocess.Popen(
        [sys.executable, str(fake_pi.path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        process.stdin.write('{"type":"prompt"}\n')
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["type"] == "response"
        assert json.loads(process.stdout.readline())["type"] == "agent_start"
        process.terminate()
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.05)
    finally:
        process.kill()
        process.communicate(timeout=5)
    assert process.returncode != 0

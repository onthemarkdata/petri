#!/usr/bin/env python3
"""Scriptable, offline subset of pi 0.73.1's LF-delimited RPC protocol."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def emit(frame):
    sys.stdout.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def respond(request, scenario):
    command = request.get("type", "prompt")
    response = {"type": "response", "command": command, "success": True}
    if "id" in request:
        response["id"] = request["id"]
    kind = scenario.get("kind", "response")
    if scenario.get("stderr"):
        sys.stderr.write(scenario["stderr"])
        sys.stderr.flush()
    if kind == "malformed":
        raw = scenario.get("raw", "not-json")
        sys.stdout.write(raw + ("\n" if scenario.get("newline", True) else ""))
        sys.stdout.flush()
        return
    if kind == "frames":
        for frame in scenario["frames"]:
            emit(frame)
        return
    if kind == "error" and not scenario.get("after_accept", False):
        emit({**response, "success": False, "error": scenario.get("error", "invalid model")})
        return
    if scenario.get("ignore_sigterm", False):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    emit(response)
    if command != "prompt":
        return
    emit({"type": "agent_start"})
    if kind == "death":
        raise SystemExit(scenario.get("exit_code", 7))
    time.sleep(scenario.get("delay", 0))
    text = scenario.get("text", "Hello from fake pi.")
    message = {
        "role": "assistant", "content": [{"type": "text", "text": text}],
        "api": "openai-completions", "provider": "fake", "model": "fake-pi",
        "usage": {
            "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop", "timestamp": 0,
    }
    if kind in ("error", "rate_limit"):
        error = scenario.get("error", "rate limit" if kind == "rate_limit" else "network timeout")
        if scenario.get("reset_hint"):
            error += "; " + scenario["reset_hint"]
        message.update(content=[], stopReason="error", errorMessage=error)
    else:
        emit({
            "type": "message_update", "message": message,
            "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": text, "partial": message,
            },
        })
    emit({"type": "message_end", "message": message})
    emit({"type": "agent_end", "messages": [message]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=os.environ.get("FAKE_PI_SCENARIO"))
    args, _ = parser.parse_known_args()
    scenarios = json.loads(Path(args.scenario).read_text()) if args.scenario else [{}]
    if not isinstance(scenarios, list) or not scenarios:
        parser.error("The scenario file must contain a nonempty JSON array.")
    index = 0
    for line in sys.stdin:
        request = json.loads(line)
        scenario = scenarios[min(index, len(scenarios) - 1)]
        if request.get("type") == "prompt":
            index += 1
        respond(request, scenario)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Throwaway probe for pi's `--mode rpc` protocol — committed for reproducibility (M1-harness.2 / issue #17).

This is NOT packaged Petri code. It spawns a real `pi` process, drives the
RPC / json / print transports over stdin/stdout, and records raw transcripts so
docs/spikes/pi-rpc.md can cite exact output. Pure stdlib — no third-party deps,
no network of its own.

One-line invocation (against an installed pi):

    python docs/spikes/pi_rpc_probe.py --out docs/spikes/transcripts

Useful flags:
    --pi-bin PATH     pi executable (default: "pi" on PATH)
    --provider NAME   provider to probe (default: let pi resolve its own default)
    --model PATTERN   model pattern/id
    --only NAME[,..]  run a subset of probes (default: all auth-less probes)
    --with-model      also run probes that need a working provider credential
    --self-test       validate the LF-JSONL framing offline against a bundled
                      fake pi (no real pi, no network) — proves the client itself

Probes with `needs_model=True` require a real provider credential (env var or
`pi /login`). Without one they still run and the transcript captures pi's error
surface (which is itself an acceptance-criterion artifact).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

# pi's RPC framing is strict LF-delimited JSONL. Per pi's docs/rpc.md we must
# split on "\n" ONLY (and tolerate a trailing "\r") — NOT with a Unicode-aware
# line splitter such as Node's readline, which also breaks on U+2028/U+2029 and
# is explicitly called out as non-compliant. Python's text-mode iteration splits
# on universal newlines, so we read bytes and split ourselves.


@dataclass
class Transcript:
    """Raw record of one probe: everything written to and read from pi."""

    name: str
    argv: list[str]
    sent: list[dict] = field(default_factory=list)
    stdout_lines: list[str] = field(default_factory=list)
    stderr_text: str = ""
    returncode: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        head = {
            "_probe": self.name,
            "_argv": self.argv,
            "_returncode": self.returncode,
            "_notes": self.notes,
            "_sent": self.sent,
        }
        lines = ["# " + json.dumps(head)]
        lines += self.stdout_lines
        if self.stderr_text.strip():
            lines.append("# STDERR: " + json.dumps(self.stderr_text))
        return "\n".join(lines) + "\n"


class PiRpc:
    """Drive one long-lived `pi --mode rpc` session over stdin/stdout."""

    def __init__(self, argv: list[str]):
        self.argv = argv
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # we do our own buffering / framing
        )
        self._q: Queue[str] = Queue()
        self._raw: list[str] = []
        self._buf = b""
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                text = line.rstrip(b"\r").decode("utf-8", "replace")
                if text == "":
                    continue
                self._raw.append(text)
                self._q.put(text)

    def send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def drain(self, seconds: float, stop_on: set[str] | None = None) -> list[dict]:
        """Collect parsed events for `seconds`, or until an event whose `type` is in stop_on."""
        events: list[dict] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=min(0.25, seconds))
            except Empty:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                events.append({"_malformed": line})
                continue
            events.append(ev)
            if stop_on and ev.get("type") in stop_on:
                break
        return events

    @property
    def raw_lines(self) -> list[str]:
        return list(self._raw)

    def close(self) -> tuple[int, str]:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        time.sleep(0.1)
        stderr = b""
        try:
            if self.proc.stderr:
                stderr = self.proc.stderr.read() or b""
        except Exception:
            pass
        return self.proc.returncode or 0, stderr.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Probes. Each returns a Transcript. `needs_model` ones require a credential.
# --------------------------------------------------------------------------- #

def _base_argv(cfg: argparse.Namespace, *extra: str) -> list[str]:
    argv = [cfg.pi_bin, "--mode", "rpc", "--no-session"]
    if cfg.provider:
        argv += ["--provider", cfg.provider]
    if cfg.model:
        argv += ["--model", cfg.model]
    argv += list(extra)
    return argv


def probe_introspection(cfg) -> Transcript:
    """get_available_models / get_session_stats / get_state — no model call, no auth."""
    argv = _base_argv(cfg)
    t = Transcript("introspection", argv)
    rpc = PiRpc(argv)
    for cmd in (
        {"id": "models", "type": "get_available_models"},
        {"id": "stats", "type": "get_session_stats"},
        {"id": "state", "type": "get_state"},
    ):
        t.sent.append(cmd)
        rpc.send(cmd)
        rpc.drain(2.0, stop_on=None)
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append("get_session_stats exposes tokens{input,output,cacheRead,cacheWrite,total}, cost, contextUsage.")
    t.notes.append("get_state keys do NOT include a tool list; tool surface is controlled by --tools/--no-tools only.")
    return t


def probe_missing_key(cfg) -> Transcript:
    """Force the command-level auth error: a provider with no API key set."""
    argv = _base_argv(cfg, "--provider", "anthropic", "--model", "anthropic/claude-sonnet-4-6")
    t = Transcript("error_missing_key", argv)
    env_had = bool(os.environ.get("ANTHROPIC_API_KEY"))
    rpc = PiRpc(argv)
    cmd = {"id": "r", "type": "prompt", "message": "hi"}
    t.sent.append(cmd)
    rpc.send(cmd)
    rpc.drain(6.0, stop_on={"agent_settled"})
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append(
        "CHANNEL A (command-level): a provider with no key fails the `prompt` command itself: "
        '{"type":"response","success":false,"error":"No API key found for <provider>..."}. '
        "Fail-fast, clearly classifiable as permanent/auth."
    )
    if env_had:
        t.notes.append("NOTE: ANTHROPIC_API_KEY was set in env; this probe may not have reproduced the no-key path.")
    return t


def probe_invalid_model(cfg) -> Transcript:
    """Invalid model id — observe warning + error channel."""
    argv = _base_argv(cfg, "--provider", "anthropic", "--model", "anthropic/does-not-exist-xyz")
    t = Transcript("error_invalid_model", argv)
    rpc = PiRpc(argv)
    cmd = {"id": "r", "type": "prompt", "message": "hi"}
    t.sent.append(cmd)
    rpc.send(cmd)
    rpc.drain(6.0, stop_on={"agent_settled"})
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    return t


def probe_multiplex(cfg) -> Transcript:
    """Fire two prompts with distinct ids back-to-back; observe whether the
    command acks/turns interleave (multiplexed) or are strictly serialized."""
    argv = _base_argv(cfg)
    t = Transcript("session_multiplex", argv)
    rpc = PiRpc(argv)
    c1 = {"id": "a", "type": "prompt", "message": "first"}
    c2 = {"id": "b", "type": "prompt", "message": "second"}
    t.sent += [c1, c2]
    rpc.send(c1)
    rpc.send(c2)
    rpc.drain(8.0, stop_on=None)
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append(
        "Inspect ordering of the two `response`/`agent_*` blocks to decide sequential vs multiplexed "
        "(drives M1-harness.6: lock to one in-flight request unless multiplexing is proven)."
    )
    return t


def probe_auto_retry_off(cfg) -> Transcript:
    """set_auto_retry off, then prompt — checks Petri can own throttling (no double-retry stacking)."""
    argv = _base_argv(cfg)
    t = Transcript("auto_retry_off", argv)
    rpc = PiRpc(argv)
    c0 = {"id": "cfg", "type": "set_auto_retry", "enabled": False}
    c1 = {"id": "r", "type": "prompt", "message": "hi"}
    t.sent += [c0, c1]
    rpc.send(c0)
    rpc.drain(1.5, stop_on=None)
    rpc.send(c1)
    rpc.drain(6.0, stop_on={"agent_settled"})
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append("Verifies the command is accepted; live 429 behavior with retry off is a runbook item.")
    return t


def probe_prompt_roundtrip(cfg) -> Transcript:
    """needs_model — a real generation: round-trip, streaming deltas, usage."""
    argv = _base_argv(cfg)
    t = Transcript("prompt_roundtrip", argv)
    rpc = PiRpc(argv)
    cmd = {"id": "r", "type": "prompt", "message": "Reply with the single word: pong."}
    t.sent.append(cmd)
    rpc.send(cmd)
    rpc.drain(45.0, stop_on={"agent_settled"})
    rpc.send({"id": "s", "type": "get_session_stats"})
    rpc.drain(3.0, stop_on=None)
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append("Look for message_update/text_delta (streaming) and non-zero usage on the assistant message.")
    return t


def probe_structured_output(cfg) -> Transcript:
    """needs_model — ask for schema-conformant JSON to gauge reliability."""
    argv = _base_argv(cfg)
    t = Transcript("structured_output", argv)
    rpc = PiRpc(argv)
    msg = (
        'Return ONLY a JSON object matching {"verdict": "PASS"|"FAIL", "confidence": number}. '
        "No prose. Verdict PASS, confidence 0.9."
    )
    cmd = {"id": "r", "type": "prompt", "message": msg}
    t.sent.append(cmd)
    rpc.send(cmd)
    rpc.drain(45.0, stop_on={"agent_settled"})
    t.returncode, t.stderr_text = rpc.close()
    t.stdout_lines = rpc.raw_lines
    t.notes.append("Baseline strategy is pydantic-ai PromptedOutput + validation-retry over plain text.")
    return t


PROBES = {
    "introspection": (probe_introspection, False),
    "error_missing_key": (probe_missing_key, False),
    "error_invalid_model": (probe_invalid_model, False),
    "session_multiplex": (probe_multiplex, False),
    "auto_retry_off": (probe_auto_retry_off, False),
    "prompt_roundtrip": (probe_prompt_roundtrip, True),
    "structured_output": (probe_structured_output, True),
}


# --------------------------------------------------------------------------- #
# Non-RPC transports (fallbacks): -p and --mode json.
# --------------------------------------------------------------------------- #

def probe_print_mode(cfg) -> Transcript:
    argv = [cfg.pi_bin, "-p", "say hi", "--no-session"]
    if cfg.provider:
        argv += ["--provider", cfg.provider]
    t = Transcript("fallback_print", argv)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        t.stdout_lines = r.stdout.splitlines()
        t.stderr_text = r.stderr
        t.returncode = r.returncode
    except subprocess.TimeoutExpired as e:
        t.notes.append("print mode TIMED OUT (25s) — pi -p blocked on the model call.")
        t.stdout_lines = (e.stdout or b"").decode("utf-8", "replace").splitlines() if isinstance(e.stdout, bytes) else (e.stdout or "").splitlines()
        t.returncode = 124
    return t


def probe_json_mode(cfg) -> Transcript:
    argv = [cfg.pi_bin, "--mode", "json", "say hi", "--no-session"]
    if cfg.provider:
        argv += ["--provider", cfg.provider]
    t = Transcript("fallback_json", argv)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=25, input="")
        t.stdout_lines = r.stdout.splitlines()
        t.stderr_text = r.stderr
        t.returncode = r.returncode
        t.notes.append("--mode json emits a `session` header then the same event stream as rpc, then exits.")
    except subprocess.TimeoutExpired:
        t.notes.append("json mode TIMED OUT (25s).")
        t.returncode = 124
    return t


NON_RPC = {
    "fallback_print": (probe_print_mode, True),
    "fallback_json": (probe_json_mode, True),
}


# --------------------------------------------------------------------------- #
# Offline self-test: a fake pi that speaks LF-JSONL, to validate the client.
# --------------------------------------------------------------------------- #

FAKE_PI = r'''#!/usr/bin/env python3
import json, sys
# Minimal pi-rpc impersonator: echoes an ack + a fake turn for `prompt`,
# and a canned data payload for get_state. Includes an embedded \n inside a
# JSON string field to exercise the client's LF framing.
for raw in sys.stdin:
    line = raw.rstrip("\n").rstrip("\r")
    if not line:
        continue
    try:
        cmd = json.loads(line)
    except json.JSONDecodeError:
        sys.stdout.write(json.dumps({"type":"response","success":False,"error":"bad frame"})+"\n"); sys.stdout.flush(); continue
    cid = cmd.get("id")
    typ = cmd.get("type")
    if typ == "prompt":
        sys.stdout.write(json.dumps({"id":cid,"type":"response","command":"prompt","success":True})+"\n")
        sys.stdout.write(json.dumps({"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"po\nng"}})+"\n")
        sys.stdout.write(json.dumps({"type":"agent_settled"})+"\n")
    elif typ == "get_state":
        sys.stdout.write(json.dumps({"id":cid,"type":"response","command":"get_state","success":True,"data":{"model":"fake"}})+"\n")
    else:
        sys.stdout.write(json.dumps({"id":cid,"type":"response","command":typ,"success":True,"data":{}})+"\n")
    sys.stdout.flush()
'''


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "fake_pi.py"
        fake.write_text(FAKE_PI)
        argv = [sys.executable, str(fake)]
        rpc = PiRpc(argv)
        rpc.send({"id": "1", "type": "prompt", "message": "x"})
        evs = rpc.drain(4.0, stop_on={"agent_settled"})
        rpc.send({"id": "2", "type": "get_state"})
        evs += rpc.drain(2.0, stop_on=None)
        rpc.close()
        got_ack = any(e.get("type") == "response" and e.get("id") == "1" and e.get("success") for e in evs)
        delta = next((e for e in evs if e.get("type") == "message_update"), None)
        embedded_ok = bool(delta and "\n" in delta["assistantMessageEvent"]["delta"])
        state = any(e.get("id") == "2" and e.get("data", {}).get("model") == "fake" for e in evs)
        ok = got_ack and embedded_ok and state
        print(f"[self-test] ack={got_ack} embedded_newline_framing={embedded_ok} state={state} => {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe pi --mode rpc; record transcripts.")
    ap.add_argument("--pi-bin", default="pi")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="docs/spikes/transcripts")
    ap.add_argument("--only", default=None, help="comma-separated probe names")
    ap.add_argument("--with-model", action="store_true", help="also run probes needing a credential")
    ap.add_argument("--self-test", action="store_true")
    cfg = ap.parse_args()

    if cfg.self_test:
        return run_self_test()

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = set(cfg.only.split(",")) if cfg.only else None

    all_probes = {**{k: v for k, v in PROBES.items()}, **NON_RPC}
    summary = []
    for name, (fn, needs_model) in all_probes.items():
        if selected and name not in selected:
            continue
        if needs_model and not cfg.with_model:
            summary.append({"probe": name, "status": "skipped (needs --with-model + credential)"})
            continue
        print(f"[probe] {name} ...", flush=True)
        try:
            t = fn(cfg)
            (out / f"{name}.jsonl").write_text(t.to_jsonl())
            summary.append({"probe": name, "status": "ran", "returncode": t.returncode,
                            "stdout_lines": len(t.stdout_lines)})
        except Exception as e:  # noqa: BLE001 — probe harness must not abort the suite
            summary.append({"probe": name, "status": f"error: {e!r}"})

    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

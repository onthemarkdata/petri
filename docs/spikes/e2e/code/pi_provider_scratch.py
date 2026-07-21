"""THROWAWAY SCRATCH provider: route Petri inference through the `pi` harness.

NOT production code, not for merge. Proves an end-to-end Petri run (seed ->
grow) on `pi` (0.80.9) instead of the `claude` CLI.

  * ``PiInferenceProvider`` SUBCLASSES ``ClaudeCodeProvider`` and overrides only
    ``__init__`` (drop the claude-binary guard) and the transport ``_ask`` — so
    every prompt-builder and all response parsing are inherited verbatim.
  * Transport is ``pi --mode json <prompt> --no-session --no-tools`` one-shot,
    via the fail-loud ``pi_ask`` helper. It RAISES on all three of pi's error
    channels and on empty output; on failure ``_ask`` re-raises as
    ``ClaudeCLIError`` so inherited guards (assess_cell -> EXECUTION_ERROR)
    behave identically.

Config via env: PI_PROVIDER (default anthropic), PI_MODEL
(default anthropic/claude-sonnet-4-6), PI_BIN (default pi).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

from petri.reasoning.claude_code_provider import ClaudeCodeProvider, ClaudeCLIError

DEFAULT_PI_PROVIDER = os.environ.get("PI_PROVIDER", "anthropic")
DEFAULT_PI_MODEL = os.environ.get("PI_MODEL", "anthropic/claude-sonnet-4-6")
DEFAULT_PI_BIN = os.environ.get("PI_BIN", "pi")


class PiAskError(RuntimeError):
    """A pi one-shot call failed on a documented channel: 'command' (A),
    'generation' (B), 'process' (C), or 'empty'."""

    def __init__(self, message, *, channel, error_message="", exit_code=None, stderr="", usage=None):
        self.channel = channel
        self.error_message = error_message
        self.exit_code = exit_code
        self.stderr = stderr
        self.usage = usage or {}
        super().__init__(message)


@dataclass
class PiResult:
    text: str
    usage: dict = field(default_factory=dict)


def _iter_events(stdout: str):
    for raw in stdout.split("\n"):
        line = raw.rstrip("\r")
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _assistant_text(msg: dict) -> str:
    parts = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _parse_events(stdout: str) -> PiResult:
    final_msg = None
    delta_buf = []
    usage: dict = {}
    settled = False
    for ev in _iter_events(stdout):
        etype = ev.get("type")
        if etype == "response" and ev.get("success") is False:  # Channel A
            raise PiAskError(
                f"pi command '{ev.get('command')}' failed: {ev.get('error')}",
                channel="command", error_message=str(ev.get("error", "")),
            )
        if etype == "message_update":
            ame = ev.get("assistantMessageEvent") or {}
            if ame.get("type") == "text_delta" and isinstance(ame.get("delta"), str):
                delta_buf.append(ame["delta"])
            continue
        msg = ev.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            final_msg = msg
            if isinstance(msg.get("usage"), dict):
                usage = msg["usage"]
        if etype == "agent_settled":
            settled = True
    if final_msg is not None and final_msg.get("stopReason") == "error":  # Channel B
        raise PiAskError(
            f"pi generation error: {final_msg.get('errorMessage')}",
            channel="generation", error_message=str(final_msg.get("errorMessage", "")), usage=usage,
        )
    text = (_assistant_text(final_msg) if final_msg else "") or "".join(delta_buf)
    text = text.strip()
    if not text:
        raise PiAskError(
            f"pi produced no assistant text (settled={settled}, saw_assistant_msg={final_msg is not None})",
            channel="empty", usage=usage,
        )
    return PiResult(text=text, usage=usage)


def pi_ask(prompt, *, provider=None, model=None, timeout=300.0, pi_bin="pi", tools=None) -> PiResult:
    argv = [pi_bin, "--mode", "json", prompt, "--no-session"]
    if tools:
        argv += ["--tools", ",".join(tools)]
    else:
        argv += ["--no-tools"]
    if provider:
        argv += ["--provider", provider]
    if model:
        argv += ["--model", model]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input="")
    except FileNotFoundError:
        raise PiAskError(f"pi executable {pi_bin!r} not found on PATH.", channel="process") from None
    except subprocess.TimeoutExpired as e:
        raise PiAskError(
            f"pi timed out after {timeout}s.", channel="process", exit_code=124,
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
        ) from None
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # A structured in-stream error (A/B) is most informative; prefer it. But a
        # bare "empty" parse on a nonzero exit means the real diagnostic went to
        # stderr (Channel C) — e.g. in --mode json the "No API key found" error is
        # stderr+exit-1, NOT an in-band response like in --mode rpc. Don't mask it.
        try:
            return _parse_events(proc.stdout)
        except PiAskError as e:
            if e.channel in ("command", "generation"):
                raise
        except Exception:
            pass
        raise PiAskError(
            f"pi exited {proc.returncode}: {stderr or '(no stderr)'}",
            channel="process", exit_code=proc.returncode, stderr=stderr, error_message=stderr,
        )
    return _parse_events(proc.stdout)


class PiInferenceProvider(ClaudeCodeProvider):
    def __init__(self, model=DEFAULT_PI_MODEL, *, provider=DEFAULT_PI_PROVIDER,
                 pi_bin=DEFAULT_PI_BIN, allowed_tools=None, timeout=300.0, on_transport=None):
        # NB: deliberately skip super().__init__ (it runs shutil.which("claude")).
        self.model = model
        self.provider = provider
        self.pi_bin = pi_bin
        self.timeout = timeout
        if allowed_tools is None:
            self.allowed_tools = [t for t in os.environ.get("PI_TOOLS", "").split(",") if t]
        else:
            self.allowed_tools = list(allowed_tools)
        self._on_transport = on_transport
        self.total_usage: dict = {}
        # Cost guardrail: hard cap on real model calls for this scratch run.
        self.max_calls = int(os.environ.get("PI_MAX_CALLS", "80"))
        self.call_count = 0

    def _accumulate_usage(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                self.total_usage[k] = self.total_usage.get(k, 0) + v

    def _ask(self, prompt: str, on_progress: Optional[Callable[[str], None]] = None) -> str:
        self.call_count += 1
        if self.call_count > self.max_calls:
            raise ClaudeCLIError(
                exit_code=1,
                stderr=f"PI_MAX_CALLS guardrail hit ({self.max_calls}) — stopping scratch run to cap spend.",
            )
        try:
            result = pi_ask(prompt, provider=self.provider, model=self.model,
                            timeout=self.timeout, pi_bin=self.pi_bin,
                            tools=self.allowed_tools or None)
        except PiAskError as e:
            if self._on_transport is not None:
                try:
                    self._on_transport(prompt, "", None, e)
                except Exception:
                    pass
            raise ClaudeCLIError(
                exit_code=e.exit_code if e.exit_code is not None else 1,
                stderr=e.error_message or e.stderr or str(e),
            ) from e
        self._accumulate_usage(result.usage)
        if self._on_transport is not None:
            try:
                self._on_transport(prompt, result.text, result, None)
            except Exception:
                pass
        if on_progress is not None:
            try:
                on_progress(result.text)
            except Exception:
                pass
        return result.text

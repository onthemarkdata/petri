"""A serialized, long-lived pi RPC session with strict LF framing."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence

from petri.harness.errors import (
    FailureClassifier, FailureDetail, PermanentHarnessError, TransientHarnessError,
)


class PiRpcTransport:
    """Run one command at a time; a prompt ends at agent_end, not its acceptance ack.

    Use as an async context manager or call aclose(). A failed request closes the
    session so the next request starts a fresh process. This layer never retries.
    """

    def __init__(
        self, *, binary: str | None = None, provider: str | None = None,
        model: str | None = None, tools: Sequence[str] = (), timeout: float = 300,
        shutdown_timeout: float = 0.5, env: Mapping[str, str] | None = None,
        cwd: str | None = None, extra_args: Sequence[str] = (),
        classifier: FailureClassifier | None = None,
    ):
        for name, value in (("timeout", timeout), ("shutdown_timeout", shutdown_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.binary = binary
        self.provider = provider
        self.model = model
        self.tools = tuple(tools)
        self.timeout = timeout
        self.shutdown_timeout = shutdown_timeout
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.extra_args = tuple(extra_args)
        self.classifier = classifier or FailureClassifier()
        self.process: asyncio.subprocess.Process | None = None
        self._session_dir: tempfile.TemporaryDirectory | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr = ""
        self._stdout = ""
        self._last_event: dict | None = None
        self._lock = asyncio.Lock()
        self.request_count = 0

    def command(self) -> list[str]:
        binary = self.binary or os.environ.get("PETRI_PI_BIN") or shutil.which("pi")
        if not binary:
            raise PermanentHarnessError("pi is unavailable. Install pi or set PETRI_PI_BIN.")
        command = [binary, "--mode", "rpc"]
        if self._session_dir is not None:
            command += ["--session-dir", self._session_dir.name]
        else:
            command += ["--no-session"]
        if self.provider:
            command += ["--provider", self.provider]
        if self.model:
            command += ["--model", self.model]
        command += ["--tools", ",".join(self.tools)] if self.tools else ["--no-tools"]
        return [*command, *self.extra_args]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    def _detail(self) -> FailureDetail:
        return FailureDetail(
            exit_code=self.process.returncode if self.process else None,
            stderr_tail=self._stderr, stdout_tail=self._stdout, last_stream_event=self._last_event,
        )

    def _failure(self, stdout: str):
        status = re.search(r"(?<![\w-])(401|403|404|429|500|502|503|504)(?![\w-])", f"{self._stderr}\n{stdout}")
        return self.classifier.classify(
            self.process.returncode if self.process else None,
            self._stderr, stdout, code=int(status[1]) if status else None, detail=self._detail(),
        )

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(4096):
            self._stderr = (self._stderr + chunk.decode("utf-8", errors="replace"))[-4096:]

    async def _start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        await self._stop()
        self._stderr = ""
        self._session_dir = tempfile.TemporaryDirectory(prefix="petri-pi-session-")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command(), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self.env, cwd=self.cwd, limit=1024 * 1024,
            )
        except OSError as exc:
            raise PermanentHarnessError(f"Cannot start pi: {exc}") from exc
        assert self.process.stderr is not None
        self._stderr_task = asyncio.create_task(self._drain_stderr(self.process.stderr))
        await self._exchange({"type": "set_auto_retry", "enabled": False})

    async def _stop(self) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), self.shutdown_timeout)
            except TimeoutError:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), self.shutdown_timeout)
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
            self._stderr_task = None
        if self._session_dir is not None:
            self._session_dir.cleanup()
            self._session_dir = None

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop()

    async def _read_frame(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        try:
            raw = await self.process.stdout.readline()
        except ValueError as exc:
            raise PermanentHarnessError("pi emitted a frame larger than 1 MiB.", detail=self._detail()) from exc
        if not raw:
            await self.process.wait()
            if self._stderr_task:
                await self._stderr_task
            # Successful RPC frames may contain numeric IDs or token counts.
            # Keep them in diagnostics, but classify process death from stderr.
            raise self._failure("")
        self._stdout = (self._stdout + raw.decode("utf-8", errors="replace"))[-4096:]
        try:
            if not raw.endswith(b"\n"):
                raise ValueError("Frame must end in LF")
            frame = json.loads(raw.decode("utf-8"))
            if not isinstance(frame, dict) or not isinstance(frame.get("type"), str):
                raise ValueError("Frame must be an object with a type")
        except (ValueError, UnicodeError) as exc:
            raise PermanentHarnessError("pi emitted a malformed RPC frame.", detail=self._detail()) from exc
        self._last_event = frame
        return frame

    async def _exchange(self, payload: dict) -> dict:
        assert self.process is not None and self.process.stdin is not None
        request_id = payload.get("id") or str(uuid.uuid4())
        wire = json.dumps({**payload, "id": request_id}, ensure_ascii=False, allow_nan=False)
        self.process.stdin.write(wire.encode("utf-8") + b"\n")
        await self.process.stdin.drain()
        accepted = False
        completed = None
        while True:
            frame = await self._read_frame()
            if frame["type"] == "response":
                if frame.get("id") != request_id or frame.get("command") != payload["type"]:
                    raise PermanentHarnessError("pi returned an unmatched response.", detail=self._detail())
                if type(frame.get("success")) is not bool:
                    raise PermanentHarnessError("pi response lacks a boolean success field.", detail=self._detail())
                if not frame["success"]:
                    raise self._failure(json.dumps(frame))
                if payload["type"] != "prompt":
                    return frame
                accepted = True
            elif frame["type"] == "agent_end" and payload["type"] == "prompt":
                messages = frame.get("messages")
                if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
                    raise PermanentHarnessError("pi completed without valid messages.", detail=self._detail())
                for message in messages:
                    if message.get("role") == "assistant" and message.get("stopReason") in ("error", "aborted"):
                        error = message.get("errorMessage") or "pi inference aborted"
                        raise self._failure(error if isinstance(error, str) else json.dumps(error))
                completed = frame
            if accepted and completed is not None:
                return completed

    async def request(self, payload: dict) -> dict:
        """Send an RPC command; timeouts, protocol errors, and cancellation reap pi."""
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise ValueError("RPC payload must be an object with a string type")
        async with self._lock:
            self._stdout = ""
            self._last_event = None
            self.request_count += 1
            try:
                async with asyncio.timeout(self.timeout):
                    await self._start()
                    return await self._exchange(payload)
            except TimeoutError as exc:
                await self._stop()
                raise TransientHarnessError("pi RPC request timed out.", detail=self._detail()) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self._stop()
                raise TransientHarnessError("pi closed its input stream.", detail=self._detail()) from exc
            except BaseException:
                await self._stop()
                raise

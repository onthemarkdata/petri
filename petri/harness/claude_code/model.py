"""Claude Code transport with PromptedOutput validation owned by Pydantic AI.

This adapter returns CLI text without JSON extraction or verdict coercion. The
optional stream-json mode consumes CLI events but exposes a completed response;
Pydantic AI streaming is not implemented. Plain output has no measured usage.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from subprocess import CompletedProcess

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from petri.harness.errors import (
    AuthExpiredError, FailureClassifier, FailureDetail, PermanentHarnessError,
    TransientHarnessError,
)
from petri.harness.messages import render_history
from petri.harness.function_tools import response_parts, tool_instructions
from petri.harness.retry import RetryPolicy, retry

Runner = Callable[[Sequence[str], float], Awaitable[CompletedProcess[str]]]


async def run_cli(command: Sequence[str], timeout: float) -> CompletedProcess[str]:
    """Capture one CLI invocation and reap it on timeout or cancellation."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise PermanentHarnessError(
            "Cannot start Claude Code. Install the claude CLI and run 'claude auth login'.",
        ) from exc
    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        stdout, stderr = await communication
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise TransientHarnessError(
            "Claude Code request timed out.",
            detail=FailureDetail(
                exit_code=process.returncode,
                stdout_tail=stdout.decode("utf-8", errors="replace"),
                stderr_tail=stderr.decode("utf-8", errors="replace"),
            ),
        ) from exc
    assert process.returncode is not None
    try:
        return CompletedProcess(command, process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PermanentHarnessError("Claude Code emitted invalid UTF-8.") from exc


def extract_text_delta(event: dict) -> str:
    """Read documented Claude text delta and complete-assistant shapes."""
    if event.get("type") == "stream_event":
        nested = event.get("event")
        return extract_text_delta(nested) if isinstance(nested, dict) else ""
    if event.get("type") == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            text = delta.get("text")
            if not isinstance(text, str):
                raise PermanentHarnessError("Claude Code emitted a non-text delta.")
            return text
    if event.get("type") == "assistant":
        message = event.get("message")
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            raise PermanentHarnessError("Claude Code emitted invalid assistant content.")
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise PermanentHarnessError("Claude Code emitted a non-text content block.")
                parts.append(text)
        return "".join(parts)
    return ""


class ClaudeCodeModel(Model):
    """A fresh CLI process per request; injectable runner keeps tests offline."""

    def __init__(
        self, model_name: str = "sonnet", *, allowed_tools: Sequence[str] = (),
        binary: str = "claude", stream_json: bool = False, timeout: float = 300,
        retry_policy: RetryPolicy | None = None, runner: Runner = run_cli,
        clock: Callable[[], datetime] | None = None,
    ):
        super().__init__(profile=ModelProfile(
            supports_tools=True, supports_json_schema_output=False,
            supports_json_object_output=False, default_structured_output_mode="prompted",
            supports_image_input=False, supports_document_input=False,
            supports_audio_file_input=False, supports_video_input=False,
        ))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._model_name = model_name
        self.allowed_tools = tuple(allowed_tools)
        self.binary = binary
        self.stream_json = stream_json
        self.timeout = timeout
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.runner = runner
        self.clock = clock
        self.classifier = FailureClassifier()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "anthropic"

    def command(self, prompt: str) -> list[str]:
        command = [self.binary, "--print", "--model", self.model_name]
        command.append(f"--allowedTools={','.join(self.allowed_tools)}" if self.allowed_tools else "--tools=")
        if self.stream_json:
            command += ["--output-format", "stream-json", "--include-partial-messages", "--verbose"]
        return [*command, prompt]

    def _failure(self, result: CompletedProcess[str], detail: FailureDetail | None = None):
        status = re.search(r"(?<![\w-])(401|403|404|429|500|502|503|504)(?![\w-])", f"{result.stderr}\n{result.stdout}")
        error = self.classifier.classify(
            result.returncode, result.stderr, result.stdout,
            code=int(status[1]) if status else None, detail=detail, clock=self.clock,
        )
        if isinstance(error, AuthExpiredError):
            return AuthExpiredError(
                "Claude Code authentication expired. Run 'claude auth login', then retry.", detail=error.detail,
            )
        return error

    def _stream_text(self, result: CompletedProcess[str]) -> tuple[str, dict | None]:
        terminal = None
        partial = ""
        last_event = None
        for line in result.stdout.split("\n"):
            if not line.strip():
                continue
            detail = FailureDetail(
                exit_code=result.returncode, stdout_tail=line, stderr_tail=result.stderr,
                last_stream_event=last_event, partial_output=partial,
            )
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise ValueError("Expected an event object")
            except ValueError as exc:
                raise PermanentHarnessError("Claude Code emitted a malformed stream frame.", detail=detail) from exc
            last_event = event
            partial = (partial + extract_text_delta(event))[-4096:]
            if event["type"] == "result":
                if terminal is not None:
                    raise PermanentHarnessError("Claude Code emitted multiple result frames.", detail=detail)
                terminal = event
        detail = FailureDetail(
            exit_code=result.returncode, stdout_tail=result.stdout, stderr_tail=result.stderr,
            last_stream_event=last_event, partial_output=partial,
        )
        if terminal is None:
            raise PermanentHarnessError("Claude Code stream ended without a result.", detail=detail)
        if terminal.get("is_error") is True or terminal.get("subtype", "success") != "success":
            raise self._failure(result, detail)
        text = terminal.get("result")
        if not isinstance(text, str):
            raise PermanentHarnessError("Claude Code result has no text.", detail=detail)
        return text, terminal.get("usage")

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        settings, parameters = self.prepare_request(model_settings, model_request_parameters)
        if settings and (set(settings) != {"tool_choice"} or settings["tool_choice"] not in ("auto", "required")):
            raise UserError("ClaudeCodeModel does not support per-request model settings.")
        if parameters.output_tools:
            raise UserError("ClaudeCodeModel supports prompted output and CLI tool allowlists.")
        prompt = render_history(messages, parameters) + tool_instructions(parameters, required=bool(settings and settings.get("tool_choice") == "required"))

        async def invoke():
            result = await self.runner(self.command(prompt), self.timeout)
            if result.returncode != 0:
                raise self._failure(result)
            return self._stream_text(result) if self.stream_json else (result.stdout, None)

        text, usage = await retry(invoke, self.retry_policy)
        counts = {}
        if usage is not None:
            if not isinstance(usage, dict):
                raise PermanentHarnessError("Claude Code returned invalid usage metadata.")
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                value = usage.get(key, 0)
                if type(value) is not int or value < 0:
                    raise PermanentHarnessError("Claude Code returned invalid token counts.")
                counts[key] = value
        reported = isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage
        return ModelResponse(
            parts=response_parts(text, parameters), model_name=self.model_name, provider_name=self.system,
            usage=RequestUsage(
                input_tokens=counts.get("input_tokens", 0) + counts.get("cache_read_input_tokens", 0) + counts.get("cache_creation_input_tokens", 0),
                output_tokens=counts.get("output_tokens", 0),
                cache_read_tokens=counts.get("cache_read_input_tokens", 0),
                cache_write_tokens=counts.get("cache_creation_input_tokens", 0),
                unmeasured_requests=0 if reported else 1,
            ),
            provider_details={"usage_reported": reported}, finish_reason="stop",
        )

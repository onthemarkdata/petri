"""Pydantic AI model using pi text output and prompted JSON validation.

The model reuses one pi process, serializes requests, and resets conversation state
before replaying Pydantic AI's complete history. Streaming, native JSON schemas,
and Pydantic function tools are unsupported. Explicit pi tool allowlists apply to
pi's own tools. Missing usage stays zero and is marked as unreported.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    InstructionPart, ModelMessage, ModelRequest, ModelResponse, RetryPromptPart,
    SystemPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from petri.harness.errors import FailureDetail, PermanentHarnessError
from petri.harness.pi.transport import PiRpcTransport
from petri.harness.retry import RetryPolicy, retry


def render_history(messages: list[ModelMessage], parameters: ModelRequestParameters) -> str:
    """Serialize supported text history, preserving validation feedback and role labels."""
    history = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                history.append({"role": "system", "content": part.content})
            elif isinstance(part, UserPromptPart):
                content = part.content
                if not isinstance(content, str):
                    if not all(isinstance(item, str) for item in content):
                        raise UserError("PiModel currently accepts text input only.")
                    content = "\n".join(item for item in content if isinstance(item, str))
                history.append({"role": "user", "content": content})
            elif isinstance(part, TextPart):
                history.append({"role": "assistant", "content": part.content})
            elif isinstance(part, ToolReturnPart):
                history.append({"role": "tool", "name": part.tool_name, "content": part.model_response_str()})
            elif isinstance(part, RetryPromptPart):
                history.append({"role": "user", "content": part.model_response()})
            elif isinstance(part, ToolCallPart):
                history.append({"role": "assistant", "tool": part.tool_name, "arguments": part.args_as_json_str()})
            elif isinstance(part, ThinkingPart):
                continue
            else:
                raise UserError(f"PiModel cannot send message part {type(part).__name__}.")
    if parameters.instruction_parts is not None:
        instructions = InstructionPart.join(parameters.instruction_parts)
    else:
        instructions = next(
            (m.instructions for m in reversed(messages) if isinstance(m, ModelRequest) and m.instructions), "",
        )
        if parameters.prompted_output_instructions:
            instructions += "\n" + parameters.prompted_output_instructions
    return (
        "Continue the conversation below and answer its latest request. "
        "The JSON entries preserve each message's role.\n\n"
        + (f"Instructions:\n{instructions}\n\n" if instructions else "")
        + "Conversation:\n" + json.dumps(history, ensure_ascii=False)
    )


class PiModel(Model):
    """One reusable pi process; use an async context manager to ensure shutdown."""

    def __init__(
        self, model_name: str, *, provider: str = "anthropic",
        tools: Sequence[str] = (), transport: PiRpcTransport | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        super().__init__(profile=ModelProfile(
            supports_tools=False, supports_json_schema_output=False,
            supports_json_object_output=False, default_structured_output_mode="prompted",
            supports_image_input=False, supports_document_input=False,
            supports_audio_file_input=False, supports_video_input=False,
        ))
        self._model_name = model_name
        self._system = provider
        self.transport = transport or PiRpcTransport(provider=provider, model=model_name, tools=tools)
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._lock = asyncio.Lock()
        self.round_trips = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return self._system

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            await self.transport.aclose()

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        settings, parameters = self.prepare_request(model_settings, model_request_parameters)
        if settings:
            raise UserError("PiModel does not support per-request model settings; configure the pi provider.")
        if parameters.output_tools or parameters.declared_function_tools:
            raise UserError("PiModel supports prompted output and pi tool allowlists, not Pydantic function tools.")
        prompt = render_history(messages, parameters)
        async with self._lock:
            async def invoke():
                # Pydantic AI owns history; resetting avoids duplicating it in pi.
                await self.transport.request({"type": "new_session"})
                self.round_trips += 1
                return await self.transport.request({"type": "prompt", "message": prompt})

            result = await retry(invoke, self.retry_policy)
        return self._response(result)

    def _response(self, result: dict) -> ModelResponse:
        messages = [m for m in result["messages"] if m.get("role") == "assistant"]
        detail = FailureDetail(last_stream_event=result)
        if not messages:
            raise PermanentHarnessError("pi completed without an assistant message.", detail=detail)
        final = messages[-1]
        content = final.get("content")
        if final.get("stopReason") not in ("stop", "length") or not isinstance(content, list):
            raise PermanentHarnessError("pi returned an incomplete assistant message.", detail=detail)
        text = []
        for part in content:
            if not isinstance(part, dict):
                raise PermanentHarnessError("pi returned an invalid content block.", detail=detail)
            if part.get("type") == "text":
                if not isinstance(part.get("text"), str):
                    raise PermanentHarnessError("pi returned a non-text answer.", detail=detail)
                text.append(part["text"])
        if not text:
            raise PermanentHarnessError("pi completed without text output.", detail=detail)
        counts = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        usage_reported = True
        for message in messages:
            usage = message.get("usage")
            if not isinstance(usage, dict):
                usage_reported = False
                continue
            for key in counts:
                value = usage.get(key)
                if value is None:
                    usage_reported = False
                    continue
                if type(value) is not int or value < 0:
                    raise PermanentHarnessError("pi returned invalid token usage.", detail=detail)
                counts[key] += value
        return ModelResponse(
            parts=[TextPart("".join(text))],
            model_name=final.get("model") or self.model_name,
            provider_name=final.get("provider") or self.system,
            usage=RequestUsage(
                input_tokens=counts["input"] + counts["cacheRead"] + counts["cacheWrite"],
                output_tokens=counts["output"], cache_read_tokens=counts["cacheRead"],
                cache_write_tokens=counts["cacheWrite"],
                unmeasured_requests=0 if usage_reported else 1,
            ),
            provider_details={"usage_reported": usage_reported},
            finish_reason="length" if final["stopReason"] == "length" else "stop",
        )

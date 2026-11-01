"""Strict prompted function-call envelopes for text-only CLI transports.

Pydantic AI validates arguments, executes registered tools, and feeds back their
results. The CLI only returns text; its built-in tool allowlist is independent.
"""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters


class _Call(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    arguments: dict


class _Calls(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_calls: list[_Call] = Field(min_length=1)


def tool_instructions(parameters: ModelRequestParameters, *, required: bool = False, completed: bool = False) -> str:
    tools = parameters.declared_function_tools
    if not tools:
        return ""
    definitions = [{"name": tool.name, "description": tool.description,
                    "parameters": tool.parameters_json_schema} for tool in tools]
    required_note = (
        "The next response must be a host function-call envelope; a final result is not allowed yet.\n"
        if required else ""
    )
    completion = ""
    if completed and not required:
        completion = (
            "\nHost tool results are available. Continue the original task using those results. "
            "Call any further tools needed to meet its evidence requirements; a search result "
            "does not substitute for fetching a cited source. Do not repeat an identical successful "
            "call without a reason. When ready, return the final result in the required format:\n"
            + (parameters.prompted_output_instructions or "Return the final answer.")
        )
    return required_note + (
        "\nPetri host function tools are available, independently of CLI built-in tools. "
        'To call them, return exactly {"tool_calls": [{"name": "tool_name", "arguments": {}}]}. '
        "Use real arguments matching the schema. Return no final result in the same response. "
        "The host will execute the calls and supply tool results before you continue. "
        "When finished, return the requested final result instead of a tool-call envelope.\n"
        + json.dumps(definitions, ensure_ascii=False)
    ) + completion


def response_parts(text: str, parameters: ModelRequestParameters) -> list[TextPart | ToolCallPart]:
    """Interpret only complete, schema-valid tool envelopes; preserve other text."""
    if parameters.declared_function_tools:
        try:
            candidate = text.strip()
            fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, flags=re.DOTALL)
            if fenced:
                candidate = fenced[1]
            envelope = _Calls.model_validate_json(candidate)
        except ValidationError:
            pass
        else:
            return [ToolCallPart(call.name, call.arguments) for call in envelope.tool_calls]
    return [TextPart(text)]

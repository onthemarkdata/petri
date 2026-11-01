"""Strict prompted function-call envelopes for text-only CLI transports.

Pydantic AI validates arguments, executes registered tools, and feeds back their
results. The CLI only returns text; its built-in tool allowlist is independent.
"""

import json

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


def tool_instructions(parameters: ModelRequestParameters) -> str:
    tools = parameters.declared_function_tools
    if not tools:
        return ""
    definitions = [{"name": tool.name, "description": tool.description,
                    "parameters": tool.parameters_json_schema} for tool in tools]
    return (
        "\nPetri host function tools are available, independently of CLI built-in tools. "
        'To call them, return exactly {"tool_calls": [{"name": "tool_name", "arguments": {}}]}. '
        "Use real arguments matching the schema. Return no final result in the same response. "
        "The host will execute the calls and supply tool results before you continue. "
        "When finished, return the requested final result instead of a tool-call envelope.\n"
        + json.dumps(definitions, ensure_ascii=False)
    )


def response_parts(text: str, parameters: ModelRequestParameters) -> list[TextPart | ToolCallPart]:
    """Interpret only complete, schema-valid tool envelopes; preserve other text."""
    if parameters.declared_function_tools:
        try:
            envelope = _Calls.model_validate_json(text)
        except ValidationError:
            pass
        else:
            return [ToolCallPart(call.name, call.arguments) for call in envelope.tool_calls]
    return [TextPart(text)]

"""Text history mapping shared by CLI-backed Pydantic AI models."""

import json

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    InstructionPart, ModelMessage, ModelRequest, RetryPromptPart, SystemPromptPart,
    TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters


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


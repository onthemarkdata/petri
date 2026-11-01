"""Require an investigation attempt before requesting the final assessment."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionPart, ModelMessage, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings


def has_retrieval_attempt(messages: Sequence[ModelMessage]) -> bool:
    """Count actual tool returns, including failed retrieval attempts."""
    return any(isinstance(part, ToolReturnPart) and part.tool_name in {"web_search", "web_fetch"}
               for message in messages for part in message.parts)


def investigation_settings(base: ModelSettings) -> Callable[[RunContext[Any]], ModelSettings]:
    def settings(context: RunContext[Any]) -> ModelSettings:
        result = base.copy()
        if not has_retrieval_attempt(context.messages):
            result["tool_choice"] = "required"
        return result
    return settings


class RequiredInvestigationOutput(AbstractCapability[Any]):
    """Request a host tool call before enabling final-result JSON mode."""

    async def prepare_model_request(
        self, context: RunContext[Any], request: ModelRequestContext,
    ) -> ModelRequestContext:
        parameters = request.model_request_parameters
        if (request.model_settings or {}).get("tool_choice") == "required" and parameters.output_mode == "prompted":
            instructions = list(parameters.instruction_parts or [])
            instructions.append(InstructionPart(
                "This request is the required investigation stage. Call web_search or web_fetch. "
                "Wait for the tool result before returning the final assessment. A failed retrieval "
                "attempt is not evidence and may justify an inconclusive result."
            ))
            request.model_request_parameters = replace(
                parameters, output_mode="text", output_object=None, prompted_output_template=None,
                instruction_parts=instructions,
            )
        return request


class NativeFunctionPromptedOutput(AbstractCapability[Any]):
    """Keep native function calls available while validating prompted final output."""

    async def prepare_model_request(
        self, context: RunContext[Any], request: ModelRequestContext,
    ) -> ModelRequestContext:
        parameters = request.model_request_parameters
        if parameters.output_mode == "prompted" and parameters.declared_function_tools:
            prepared = request.model.customize_request_parameters(parameters)
            prepared = replace(prepared, prompted_output_template=(
                parameters.prompted_output_template or request.model.profile.get("prompted_output_template")))
            final_instructions = prepared.prompted_output_instructions
            if not final_instructions:
                return request
            instructions = list(parameters.instruction_parts or [])
            instructions.append(InstructionPart(final_instructions))
            request.model_request_parameters = replace(
                parameters, output_mode="text", output_object=None, prompted_output_template=None,
                instruction_parts=instructions,
            )
        return request

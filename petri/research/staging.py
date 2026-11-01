"""Require an investigation attempt before requesting the final assessment."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, OutputContext, RawOutput
from pydantic_ai.messages import InstructionPart, ModelMessage, RetryFeedbackPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings


def has_retrieval_attempt(messages: Sequence[ModelMessage]) -> bool:
    """Count completed retrieval tools, whose results can contain HTTP errors."""
    return any(isinstance(part, ToolReturnPart) and part.outcome == "success"
               and part.tool_name in {"web_search", "web_fetch"}
               for message in messages for part in message.parts)


def investigation_settings(base: ModelSettings) -> Callable[[RunContext[Any]], ModelSettings]:
    def settings(context: RunContext[Any]) -> ModelSettings:
        result = base.copy()
        if not has_retrieval_attempt(context.messages):
            result["tool_choice"] = "required"
        return result
    return settings


def require_retrieval_attempt(messages: Sequence[ModelMessage]) -> None:
    if not has_retrieval_attempt(messages):
        raise ModelRetry(
            "Call web_search or web_fetch before completing the investigation. "
            "A failed retrieval attempt may justify an inconclusive result; it is not evidence."
        )


class RequiredInvestigationOutput(AbstractCapability[Any]):
    """Request a host tool call before enabling final-result JSON mode."""

    async def before_output_validate(
        self, context: RunContext[Any], *, output_context: OutputContext, output: RawOutput,
    ) -> RawOutput:
        if not context.partial_output:
            require_retrieval_attempt(context.messages)
        return output

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
            parts = request.messages[-1].parts if request.messages else []
            syntax_retry = any(
                isinstance(part, RetryFeedbackPart) and isinstance(part.content, list)
                and bool(part.content)
                and all(error.get("type") == "json_invalid" for error in part.content)
                for part in parts
            )
            if (syntax_retry and has_retrieval_attempt(context.messages)
                    and not any(isinstance(part, ToolReturnPart) for part in parts)
                    and request.model.profile.get("supports_json_object_output", False)):
                request.model_request_parameters = replace(
                    parameters, function_tools=[], instruction_parts=[
                        *(parameters.instruction_parts or []), InstructionPart(
                            "The previous final assessment was invalid JSON. Return one complete JSON "
                            "object matching the schema. This is a final-output syntax retry; tools are "
                            "unavailable for this request. Report missing source verification honestly. "
                            "Evidence and verdict validation still apply."
                        ),
                    ],
                )
                request.model_settings = (request.model_settings or {}).copy()
                request.model_settings.pop("tool_choice", None)
                return request
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

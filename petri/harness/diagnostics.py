"""Bound failed model output and retry feedback for durable diagnostics."""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryFeedbackPart,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from petri.harness.errors import FailureDetail


def model_failure_detail(error: BaseException, messages: list[ModelMessage]) -> FailureDetail:
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    partial = ""
    if responses:
        partial = "\n".join(
            part.content if isinstance(part, TextPart) else part.args_as_json_str()
            for part in responses[-1].parts
            if isinstance(part, (TextPart, ToolCallPart))
        )
    feedback = [
        part.model_response()
        if isinstance(part, RetryPromptPart)
        else part.model_response_str()
        if isinstance(part, ToolReturnPart)
        else str(part)
        for message in messages
        for part in message.parts
        if isinstance(part, (RetryPromptPart, RetryFeedbackPart))
        or isinstance(part, ToolReturnPart)
        and part.outcome == "retried"
    ]
    inherited = getattr(error, "detail", None)
    return FailureDetail.model_validate(
        {
            **(inherited.model_dump() if isinstance(inherited, FailureDetail) else {}),
            "partial_output": partial
            or (inherited.partial_output if isinstance(inherited, FailureDetail) else ""),
            "last_stream_event": {
                "validation_feedback": feedback[-3:],
                "backend_event": inherited.last_stream_event
                if isinstance(inherited, FailureDetail)
                else None,
            },
        }
    )

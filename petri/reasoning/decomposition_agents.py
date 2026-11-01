"""Named decomposition agents with explicit models and validated output retries."""

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models import Model

from petri.harness.errors import FailureDetail, HarnessError
from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion
from petri.reasoning.prompts import build_decompose_claim_prompt, build_decompose_why_prompt

Progress = Callable[[str], None] | None
OUTPUT_TEMPLATE = (
    "Return one JSON value containing the requested result data. "
    "The following JSON schema defines its shape. Do not copy schema keywords "
    "such as properties, $defs, type, or required into the result.\n{schema}"
)


class DecompositionError(RuntimeError):
    """The model failed or exhausted validation retries; no atomic stop is implied."""

    def __init__(self, message: str, *, detail: FailureDetail | None = None):
        super().__init__(message)
        self.detail = detail or FailureDetail(partial_output=message)


@dataclass
class DecompositionInput:
    claim: str
    clarifications: list[dict] = field(default_factory=list)
    guidance: str = ""
    parent_level: int = 0
    parent_seq: int = 0
    max_premises: int = 5


def initial_instructions(request: DecompositionInput) -> str:
    return build_decompose_claim_prompt(
        request.claim, request.clarifications, request.guidance, request.max_premises,
    ) + "STEP 4 — EMIT JSON.\nReturn the selected premises using the supplied output schema."


def why_instructions(request: DecompositionInput) -> str:
    return build_decompose_why_prompt(
        request.claim, request.parent_level, request.parent_seq, request.max_premises,
    ) + (
        "If the parent is already ATOMIC (a directly verifiable fact, definition, or measurement), "
        "return is_atomic=true, no sub-premises, and a bedrock_reason.\n\n"
        "STEP 4 — EMIT JSON.\nReturn the selected sub-premises using the supplied output schema."
    )


initial_decomposer = Agent(
    name="initial_decomposer", deps_type=DecompositionInput,
    output_type=PromptedOutput(InitialDecomposition, template=OUTPUT_TEMPLATE),
    retries={"output": 2}, defer_model_check=True,
)
five_whys = Agent(
    name="five_whys", deps_type=DecompositionInput,
    output_type=PromptedOutput(WhyExpansion, template=OUTPUT_TEMPLATE),
    retries={"output": 2}, defer_model_check=True,
)


@initial_decomposer.instructions
def initial_prompt(context: RunContext[DecompositionInput]) -> str:
    return initial_instructions(context.deps)


@five_whys.instructions
def why_prompt(context: RunContext[DecompositionInput]) -> str:
    return why_instructions(context.deps)


async def run_decomposition_agent(
    agent: Agent[DecompositionInput, Any], request: DecompositionInput, *, model: Model,
    on_progress: Progress = None, stream: bool = False,
) -> Any:
    """Run with a caller-owned model; streaming is explicit for capable adapters."""
    async def events(_: RunContext[DecompositionInput], source: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in source:
            if on_progress is None:
                continue
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                on_progress(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                on_progress(event.delta.content_delta)

    if on_progress:
        on_progress(f"{agent.name}: started\n")
    try:
        result = await agent.run(
            "Apply the decomposition instructions to the supplied claim. Return only the selected result, without intermediate analysis.",
            deps=request, model=model, event_stream_handler=events if stream else None,
        )
    except (AgentRunError, HarnessError) as error:
        if on_progress:
            on_progress(f"{agent.name}: failed\n")
        detail = error.detail if isinstance(error, HarnessError) else FailureDetail(partial_output=str(error))
        raise DecompositionError(f"{agent.name} failed: {error}", detail=detail) from error
    if on_progress:
        on_progress(f"{agent.name}: completed\n")
    return result.output

"""Named decomposition agents with explicit models and validated output retries."""

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any
import logging
import re

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models import Model

from petri.harness.errors import FailureDetail, HarnessError
from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion
from petri.reasoning.prompts import build_decompose_claim_prompt, build_decompose_why_prompt

Progress = Callable[[str], None] | None
DUPLICATE_THRESHOLD = 0.8
QUALITY_BAR = """QUALITY BAR
BAD CLAIM (obvious fact): "Water exists." It adds no testable assumption.
BAD CLAIM (logical prerequisite): "For demand to increase, demand must exist." It merely restates a prerequisite.
BAD CLAIM (too vague): "The technology will be good." It has no measurable outcome or scope.
Prefer claims with quantified thresholds where the claim supports them, explicit scope and exclusions, and a clear falsification test. Do not invent measurements or precision.

"""
BEDROCK_TEST = """FIRST — BEDROCK TEST.
Before generating children, ask whether the parent is a definition, an axiom, or a directly verifiable measurement. If so, STOP: skip all brainstorming, prioritization, and selection below. Return is_atomic=true, an empty sub_premises list, and a specific bedrock_reason. Do not turn its wording into more premises.
For example, for "One meter equals one hundred centimeters", return {"is_atomic": true, "sub_premises": [], "bedrock_reason": "The metric unit conversion is definitional."}.
A contested causal or predictive claim is not bedrock merely because it sounds plausible. Only if the bedrock test fails should you follow the expansion steps below.

"""
COUNTERARGUMENT = (
    "For a non-atomic expansion, include the strongest counterargument as a sub-claim "
    "with relation limits or rebuts; other premises may support the parent. "
    "A single indispensable sub-premise is exempt from this requirement.\n\n"
)

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
    on_warning: Callable[[str, str, float], None] | None = None


def initial_instructions(request: DecompositionInput) -> str:
    return QUALITY_BAR + build_decompose_claim_prompt(
        request.claim, request.clarifications, request.guidance,
    ) + "STEP 4 — EMIT JSON.\nReturn the selected premises using the supplied output schema."


def why_instructions(request: DecompositionInput) -> str:
    return BEDROCK_TEST + build_decompose_why_prompt(
        request.claim, request.parent_level, request.parent_seq,
    ) + (
        COUNTERARGUMENT + "If the parent is already ATOMIC (a directly verifiable fact, definition, or measurement), "
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
    retries={"output": 1}, defer_model_check=True,
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


def parent_overlap(parent: str, child: str) -> float:
    """Jaccard similarity of Unicode word-token sets, ignoring case and punctuation."""
    parent_tokens = set(re.findall(r"\w+", parent.casefold()))
    child_tokens = set(re.findall(r"\w+", child.casefold()))
    union = parent_tokens | child_tokens
    return len(parent_tokens & child_tokens) / len(union) if union else 0.0


@five_whys.output_validator
def validate_expansion(context: RunContext[DecompositionInput], output: WhyExpansion) -> WhyExpansion:
    if output.is_atomic:
        return output
    duplicates = [child for child in output.sub_premises
                  if parent_overlap(context.deps.claim, child.claim_text) >= DUPLICATE_THRESHOLD]
    missing_counterargument = len(output.sub_premises) >= 2 and all(
        child.relation == "supports" for child in output.sub_premises
    )
    if context.retry == 0 and (duplicates or missing_counterargument):
        problems = []
        if duplicates:
            problems.append("Replace children that restate the parent with distinct, testable premises.")
        if missing_counterargument:
            problems.append("Include the strongest counterargument with relation limits or rebuts.")
        raise ModelRetry(" ".join(problems))
    if duplicates:
        for child in duplicates:
            reason = "Dropped a child that still restates its parent after a validation retry."
            logging.getLogger(__name__).warning("%s %s", reason, child.claim_text)
            if context.deps.on_warning:
                context.deps.on_warning(reason, child.claim_text, DUPLICATE_THRESHOLD)
        output = output.model_copy(update={"sub_premises": [
            child for child in output.sub_premises if child not in duplicates
        ]})
    if len(output.sub_premises) >= 2 and all(
        child.relation == "supports" for child in output.sub_premises
    ):
        raise DecompositionError("five_whys failed: the retry still lacks a counterargument.")
    return output

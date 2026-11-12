"""Named decomposition agents with explicit models and validated output retries."""

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import logging
import re

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import InstructionPart, AgentStreamEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.settings import ModelSettings

from petri.harness.errors import FailureDetail, HarnessError
from petri.harness.diagnostics import model_failure_detail
from petri.reasoning.decomposition_schemas import InitialDecomposition, PremiseCandidate, WhyExpansion
from petri.reasoning.cell_search import CellSearchCandidate, search
from petri.storage import database
from petri.reasoning.prompts import build_decompose_claim_prompt, build_decompose_why_prompt

Progress = Callable[[str], None] | None
DUPLICATE_THRESHOLD = 0.8
QUALITY_BAR = """QUALITY BAR
BAD CLAIM (obvious fact): "Water exists." It adds no testable assumption.
BAD CLAIM (logical prerequisite): "For demand to increase, demand must exist." It merely restates a prerequisite.
BAD CLAIM (too vague): "The technology will be good." It has no measurable outcome or scope.
Prefer claims with quantified thresholds where the claim supports them, explicit scope and exclusions, and a clear falsification test. Do not invent measurements or precision.

"""
SCOPE_CONTEXT = (
    "Make each child independently assessable. Preserve the parent's named system, version, "
    "mode, quantities, and conditions whenever they are needed for that child's truth. "
    "Do not turn a scoped statement into an unrestricted claim by dropping its conditions. "
    "When a child describes a named system's behavior, include that system's name in the child. "
    "A later assessor sees that child alone; context in the parent or a neighboring child is insufficient. "
    "Do not invent scope or copy the entire parent merely to carry context; state a distinct "
    "premise with only the context needed to test it.\n\n"
)
BEDROCK_TEST = """FIRST — BEDROCK TEST.
Before generating children, ask whether the parent is a definition, an axiom, or a directly verifiable measurement. If so, STOP: skip all brainstorming, prioritization, and selection below. Return is_atomic=true, an empty sub_premises list, and a specific bedrock_reason. Do not turn its wording into more premises.
For example, for "One meter equals one hundred centimeters", return {"is_atomic": true, "sub_premises": [], "bedrock_reason": "The metric unit conversion is definitional."}.
A contested causal or predictive claim is not bedrock merely because it sounds plausible. Only if the bedrock test fails should you follow the expansion steps below.

"""
COUNTERARGUMENT_SHAPE = (
    "Use relation limits or rebuts only on a new PremiseCandidate: "
    '{"claim_text": "<distinct testable claim>", "relation": "limits"} '
    '(use "rebuts" instead when appropriate). '
    "A ReferencePremise has only existing_cell_id and reason; do not add relation to it. "
    "A reason describing a reference as a counterargument does not satisfy the typed counterargument requirement. "
    "Retain valid references and supply a genuinely distinct, testable counterargument candidate when required. "
    "Call search_cells for each proposed claim first; reference existing matches instead of recreating them. "
    "Do not invent a counterargument, remove valid children to claim the singleton exemption, "
    "or mark a non-atomic parent atomic to evade this requirement."
)
COUNTERARGUMENT = (
    "For a non-atomic expansion, include the strongest counterargument as a sub-claim "
    "with relation limits or rebuts; other premises may support the parent. "
    "A single indispensable sub-premise is exempt from this requirement. "
    + COUNTERARGUMENT_SHAPE + "\n\n"
)

SEARCH_INSTRUCTIONS = (
    "Before proposing each premise, call search_cells with the specific claim you intend to add. "
    "When a matching existing cell already represents that premise, return a ReferencePremise "
    "with existing_cell_id and a reason instead of creating a duplicate. Prefer the specific "
    "matching non-root cell over its colony root. Search across earlier colonies only; "
    "seed foundation colonies before dependent colonies.\n\n"
)

FLAT_RESULT = (
    "Return only the current expansion level. Each new premise has only claim_text and relation. "
    "Each reference has only existing_cell_id and reason. The values of claim_text, "
    "existing_cell_id, reason, and a non-null bedrock_reason are plain JSON strings, never nested objects. "
    "Never put sub_premises, nodes, "
    "is_atomic, or bedrock_reason inside a child premise; the runtime expands children separately. "
    "The final-result schema does not replace tool calls: call search_cells before the final result.\n\n"
)

OUTPUT_TEMPLATE = (
    "Return one JSON value containing the requested result data. "
    "The following JSON schema defines its shape. Do not copy schema keywords "
    "such as properties, $defs, $ref, type, or required into the result.\n{schema}"
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
    db_path: Path | None = None
    colony_id: str = ""
    search_results: set[str] = field(default_factory=set)
    search_performed: bool = False
    require_counterargument: bool = False


def initial_instructions(request: DecompositionInput) -> str:
    return SEARCH_INSTRUCTIONS + FLAT_RESULT + QUALITY_BAR + SCOPE_CONTEXT + build_decompose_claim_prompt(
        request.claim, request.clarifications, request.guidance, show_reasoning=False,
    ) + ("STEP 4 — EMIT JSON.\nReturn the selected premises using the supplied output schema. "
         "The only top-level key is nodes; do not wrap the result in InitialDecomposition or another type name.")


def why_instructions(request: DecompositionInput) -> str:
    context = f"Guidance and trigger context:\n{request.guidance}\n\n" if request.guidance else ""
    return BEDROCK_TEST + context + SEARCH_INSTRUCTIONS + FLAT_RESULT + SCOPE_CONTEXT + build_decompose_why_prompt(
        request.claim, request.parent_level, request.parent_seq, show_reasoning=False,
    ) + (
        COUNTERARGUMENT + "If the parent is already ATOMIC (a directly verifiable fact, definition, or measurement), "
        "return is_atomic=true, no sub-premises, and a bedrock_reason.\n\n"
        "STEP 4 — EMIT JSON.\nReturn the selected sub-premises using the supplied output schema. "
        "The top-level keys are is_atomic, bedrock_reason, and sub_premises; do not wrap the result in WhyExpansion or another type name."
    )


def has_earlier_cells(request: DecompositionInput) -> bool:
    return bool(request.db_path is not None and request.db_path.is_file()
                and any(row["colony"] != request.colony_id for row in database.list_cells(request.db_path)))


def search_model_settings(context: RunContext[DecompositionInput]) -> ModelSettings:
    if not context.deps.search_performed and has_earlier_cells(context.deps):
        return {"tool_choice": "required"}
    return {}


class RequiredSearchOutput(AbstractCapability[DecompositionInput]):
    """Keep final-result JSON mode out of a required function-call step."""

    async def prepare_model_request(
        self, context: RunContext[DecompositionInput], request: ModelRequestContext,
    ) -> ModelRequestContext:
        parameters = request.model_request_parameters
        if (request.model_settings or {}).get("tool_choice") == "required" and parameters.output_mode == "prompted":
            instructions = list(parameters.instruction_parts or [])
            instructions.append(InstructionPart(
                "This request is the required dish-search stage. Call search_cells for a specific premise. "
                "Do not return the final decomposition result before the tool result arrives."
            ))
            request.model_request_parameters = replace(
                parameters, output_mode="text", output_object=None, prompted_output_template=None,
                instruction_parts=instructions,
            )
        return request


initial_decomposer = Agent(
    name="initial_decomposer", deps_type=DecompositionInput,
    output_type=PromptedOutput(InitialDecomposition, template=OUTPUT_TEMPLATE),
    retries={"output": 2}, defer_model_check=True, model_settings=search_model_settings, capabilities=[RequiredSearchOutput()],
)
five_whys = Agent(
    name="five_whys", deps_type=DecompositionInput,
    output_type=PromptedOutput(WhyExpansion, template=OUTPUT_TEMPLATE),
    retries={"output": 1}, defer_model_check=True, model_settings=search_model_settings, capabilities=[RequiredSearchOutput()],
)


@initial_decomposer.instructions
def initial_prompt(context: RunContext[DecompositionInput]) -> str:
    return initial_instructions(context.deps)


@five_whys.instructions
def why_prompt(context: RunContext[DecompositionInput]) -> str:
    return why_instructions(context.deps)


@initial_decomposer.tool
@five_whys.tool
def search_cells(context: RunContext[DecompositionInput], query: str, top_k: int = 5) -> list[CellSearchCandidate]:
    """Find existing premises in earlier colonies of this dish by word overlap."""
    context.deps.search_performed = True
    if context.deps.db_path is None:
        return []
    # Exclude this draft's cells so references cannot point back into its ancestors.
    if type(top_k) is not int or top_k <= 0:
        raise ModelRetry("top_k must be a positive integer")
    results = search(context.deps.db_path, query, top_k, exclude_colony=context.deps.colony_id)
    context.deps.search_results.update(cell.id for cell in results)
    return results


def validate_references(context: RunContext[DecompositionInput], candidates) -> None:
    if candidates and has_earlier_cells(context.deps) and not context.deps.search_performed:
        raise ModelRetry("Call search_cells before proposing premises, including paraphrases. Search for each specific premise.")
    for candidate in candidates:
        if isinstance(candidate, PremiseCandidate):
            if context.deps.db_path is not None:
                matches = search(context.deps.db_path, candidate.claim_text, exclude_colony=context.deps.colony_id)
                words = re.findall(r"\w+", candidate.claim_text.casefold())
                if any(re.findall(r"\w+", match.claim_text.casefold()) == words for match in matches):
                    raise ModelRetry(
                        "This premise already exists in an earlier colony. Call search_cells for "
                        f"{candidate.claim_text!r} and return a ReferencePremise instead of duplicating it."
                    )
            continue
        row = (database.get_cell(context.deps.db_path, candidate.existing_cell_id)
               if context.deps.db_path is not None else None)
        if (row is None or row["colony"] == context.deps.colony_id
                or candidate.existing_cell_id not in context.deps.search_results):
            raise ModelRetry("Reference an existing cell returned by search_cells from an earlier colony.")


@initial_decomposer.output_validator
def validate_initial_references(context: RunContext[DecompositionInput], output: InitialDecomposition) -> InitialDecomposition:
    validate_references(context, output.nodes)
    return output


async def run_decomposition_agent(
    agent: Agent[DecompositionInput, Any], request: DecompositionInput, *, model: Model,
    on_progress: Progress = None, stream: bool = False,
    model_settings: ModelSettings | None = None,
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
    with capture_run_messages() as messages:
        try:
            result = await agent.run(
                "Apply the decomposition instructions to the supplied claim. Return only the selected result, without intermediate analysis.",
                deps=request, model=model, model_settings=model_settings,
                event_stream_handler=events if stream else None,
            )
        except (AgentRunError, HarnessError, DecompositionError) as error:
            if on_progress:
                on_progress(f"{agent.name}: failed\n")
            detail = model_failure_detail(error, messages)
            if isinstance(error, DecompositionError):
                error.detail = detail
                raise
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
    validate_references(context, output.sub_premises)
    if context.deps.require_counterargument and not any(
        isinstance(child, PremiseCandidate) and child.relation in ("limits", "rebuts")
        for child in output.sub_premises
    ):
        raise ModelRetry(
            "Critical-flaw re-decomposition requires a limits or rebuts premise, even for a singleton or atomic response. "
            + COUNTERARGUMENT_SHAPE
        )
    if output.is_atomic:
        return output
    duplicates = [child for child in output.sub_premises
                  if isinstance(child, PremiseCandidate) and parent_overlap(context.deps.claim, child.claim_text) >= DUPLICATE_THRESHOLD]
    missing_counterargument = len(output.sub_premises) >= 2 and all(
        not isinstance(child, PremiseCandidate) or child.relation == "supports" for child in output.sub_premises
    )
    if context.retry == 0 and (duplicates or missing_counterargument):
        problems = []
        if duplicates:
            problems.append("Replace children that restate the parent with distinct, testable premises.")
        if missing_counterargument:
            problems.append("Include the strongest counterargument with relation limits or rebuts. " + COUNTERARGUMENT_SHAPE)
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
    if context.deps.require_counterargument and not any(
        isinstance(child, PremiseCandidate) and child.relation in ("limits", "rebuts")
        for child in output.sub_premises
    ):
        raise ModelRetry(
            "Retain a distinct limits or rebuts premise for this critical-flaw re-decomposition. "
            + COUNTERARGUMENT_SHAPE
        )
    if len(output.sub_premises) >= 2 and all(
        not isinstance(child, PremiseCandidate) or child.relation == "supports" for child in output.sub_premises
    ):
        raise DecompositionError("five_whys failed: the retry still lacks a counterargument.")
    return output

"""Typed inference behind the synchronous v1 provider protocol.

A private event loop owns each model's async resources. Calls are serialized;
close() reaps subprocesses and the worker thread, including after failed runs.
"""

from __future__ import annotations

import asyncio
from datetime import date
import threading
import weakref
from collections.abc import Callable, Coroutine, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelRetry, PromptedOutput
from pydantic_ai.models import Model, infer_model
from pydantic_ai.exceptions import AgentRunError

from petri.models import AssessmentResult, Cell, ClarifyingQuestion, DecompositionResult, EvidenceMatch
from petri.harness.errors import FailureDetail, HarnessError
from petri.reasoning import prompts

if TYPE_CHECKING:
    from petri.reasoning.decomposition_agents import DecompositionInput
    from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion

T = TypeVar("T")
Progress = Callable[[str], None] | None


class _Worker:
    """Keep the model on one loop even when callers use different threads."""

    def __init__(self, model: Model):
        self.model = model
        self.entered = False
        self.active_task: asyncio.Task | None = None
        self.lock = threading.RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.closed = False

    def run(self, operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
        with self.lock:
            if self.closed:
                raise RuntimeError("HarnessInferenceProvider is closed.")
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                ready = threading.Event()

                def serve():
                    assert self.loop is not None
                    asyncio.set_event_loop(self.loop)
                    ready.set()
                    try:
                        self.loop.run_forever()
                    finally:
                        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                        self.loop.run_until_complete(self.loop.shutdown_default_executor())
                        self.loop.close()

                self.thread = threading.Thread(target=serve, name="petri-harness", daemon=True)
                self.thread.start()
                ready.wait()
            async def invoke():
                if not self.entered:
                    await self.model.__aenter__()
                    self.entered = True
                self.active_task = asyncio.current_task()
                try:
                    return await operation()
                finally:
                    self.active_task = None

            future = asyncio.run_coroutine_threadsafe(invoke(), self.loop)
            try:
                return future.result()
            except BaseException:
                future.cancel()
                raise

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            try:
                if self.loop is not None:
                    async def cleanup():
                        if self.active_task is not None:
                            self.active_task.cancel()
                            await asyncio.gather(self.active_task, return_exceptions=True)
                        if self.entered:
                            await self.model.__aexit__(None, None, None)
                    asyncio.run_coroutine_threadsafe(cleanup(), self.loop).result()
            finally:
                self.closed = True
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    assert self.thread is not None
                    self.thread.join()


class SubstanceAssessment(BaseModel):
    """The substance check has a different contract from a role assessment."""

    model_config = ConfigDict(extra="forbid")
    is_substantive: bool = Field(strict=True)
    reason: str
    suggested_rewrite: str = ""


class BridgeAssessment(AssessmentResult):
    failure_detail: FailureDetail | None = None


class _Premise(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: str = Field(min_length=1)
    is_atomic: bool = Field(default=False, strict=True)


class _ClaimDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[_Premise] = Field(min_length=1)


class _WhyDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_atomic: bool = Field(strict=True)
    sub_premises: list[_Premise]
    reason: str = ""

    @model_validator(mode="after")
    def require_atomic_or_children(self):
        if self.is_atomic and self.sub_premises:
            raise ValueError("An atomic premise cannot have sub-premises.")
        if not self.is_atomic and not self.sub_premises:
            raise ValueError("A non-atomic premise must have at least one sub-premise.")
        return self


class HarnessInferenceProvider:
    """Validate every response; close explicitly or use as a context manager."""

    def __init__(self, model: Model | str, *, config: Mapping[str, Any] | None = None, retries: int = 2):
        if retries < 0:
            raise ValueError("retries must be nonnegative.")
        self.model = infer_model(model) if isinstance(model, str) else model
        self.config = dict(config) if config is not None else None
        self.retries = retries
        self._worker = _Worker(self.model)
        self._finalizer = weakref.finalize(self, self._worker.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        """Close the model on its owning loop, then stop and join that loop."""
        self._finalizer()

    def _run(self, name: str, prompt: str, schema: Any, on_progress: Progress = None,
             validator: Callable | None = None) -> Any:
        output_type = schema
        if schema is not str:
            output_type = PromptedOutput(schema, template=(
                "Return one JSON value containing the requested result data. "
                "The following JSON schema defines its shape. Do not copy schema keywords "
                "such as properties, $defs, type, or required into the result.\n{schema}"
            ))
        agent = Agent(
            self.model, output_type=output_type, retries={"output": self.retries},
            instructions=(
                "Return only the requested result. Perform brainstorming and ranking internally; "
                "do not include the intermediate analysis in the response. "
                "Write each claim as a concise sentence in plain text, without LaTeX notation."
            ),
        )
        if validator is not None:
            agent.output_validator(validator)
        if on_progress:
            on_progress(f"{name}: started\n")

        async def invoke():
            return (await agent.run(prompt)).output

        try:
            output = self._worker.run(invoke)
        except BaseException:
            if on_progress:
                on_progress(f"{name}: failed\n")
            raise
        if on_progress:
            on_progress(f"{name}: completed\n")
        return output

    @staticmethod
    def _limit(value: int) -> None:
        if type(value) is not int or value < 1:
            raise ValueError("The requested output limit must be a positive integer.")

    def assess_claim_substance(self, claim: str, on_progress: Progress = None) -> dict:
        result = self._run("substance", prompts.build_assess_claim_substance_prompt(claim),
                           SubstanceAssessment, on_progress)
        return result.model_dump()

    def generate_clarifying_questions(self, claim: str, max_questions: int = 5,
                                      on_progress: Progress = None) -> list[ClarifyingQuestion]:
        self._limit(max_questions)

        def validate(output: list[ClarifyingQuestion]):
            if len(output) > max_questions:
                raise ModelRetry(f"Return at most {max_questions} questions.")
            return output

        return self._run("clarification", prompts.build_generate_clarifying_questions_prompt(claim, max_questions),
                         list[ClarifyingQuestion], on_progress, validate)

    def decompose_claim(self, claim: str, clarifications: list[dict], guidance: str = "",
                        max_premises: int = 5, on_progress: Progress = None) -> DecompositionResult:
        self._limit(max_premises)

        result = self._run("decomposition", prompts.build_decompose_claim_prompt(
            claim, clarifications, guidance), _ClaimDecomposition, on_progress)
        # These placeholders carry no dish identity. The decomposer assigns real IDs.
        cells = [Cell(id=f"pending-premises-001-{index:03d}", colony_id="pending-premises",
                      claim_text=node.claim_text, level=1) for index, node in enumerate(result.nodes, 1)]
        return DecompositionResult(cells=cells, center_claim=claim)

    def decompose_why(self, premise: str, parent_level: int, parent_seq: int,
                      max_premises: int = 5, on_progress: Progress = None) -> list[dict]:
        self._limit(max_premises)

        prompt = prompts.build_decompose_why_prompt(premise, parent_level, parent_seq)
        prompt += "If this is directly verifiable as a fact, definition, or measurement, mark it atomic with no sub-premises."
        result = self._run("five-whys", prompt, _WhyDecomposition, on_progress)
        return [item.model_dump() for item in result.sub_premises]

    def initial_decomposition(self, request: DecompositionInput, on_progress: Progress = None) -> InitialDecomposition:
        from petri.reasoning.decomposition_agents import initial_decomposer, run_decomposition_agent

        return self._worker.run(lambda: run_decomposition_agent(
            initial_decomposer, request, model=self.model, on_progress=on_progress,
        ))

    def expand_why(self, request: DecompositionInput, on_progress: Progress = None) -> WhyExpansion:
        from petri.reasoning.decomposition_agents import five_whys, run_decomposition_agent

        return self._worker.run(lambda: run_decomposition_agent(
            five_whys, request, model=self.model, on_progress=on_progress,
        ))

    def assess_cell(self, cell_id: str, claim_text: str, context: dict, agent_role: str,
                    on_progress: Progress = None) -> AssessmentResult:
        from petri.config import get_agent_instruction, get_agent_verdicts, load_config

        config = self.config if self.config is not None else load_config()
        if not config.get("agents"):
            raise ValueError("Agent content is required for cell assessment.")
        verdicts = get_agent_verdicts(agent_role, config)
        if not verdicts:
            raise ValueError(f"No verdict vocabulary configured for agent {agent_role!r}.")

        def validate(output: AssessmentResult):
            if output.agent != agent_role:
                raise ModelRetry(f"Return agent={agent_role!r}.")
            if output.verdict not in verdicts:
                raise ModelRetry(f"Invalid verdict {output.verdict!r}; return one of {verdicts}.")
            return output

        prompt = prompts.build_assess_cell_prompt(
            cell_id, claim_text, context, agent_role, valid_verdicts=verdicts,
            role_instruction=get_agent_instruction(agent_role, config), today_iso=date.today().isoformat(),
        )
        try:
            return self._run("assessment", prompt, AssessmentResult, on_progress, validate)
        except (HarnessError, AgentRunError) as error:
            detail = error.detail if isinstance(error, HarnessError) else FailureDetail(partial_output=str(error))
            return BridgeAssessment(agent=agent_role, verdict="EXECUTION_ERROR",
                                    summary=f"Assessment failed: {error}", failure_detail=detail)

    def summarize_evidence(self, cell_id: str, claim_text: str, evidence_md: str, iteration: int) -> str:
        return self._run("summary", prompts.build_summarize_evidence_prompt(
            cell_id, claim_text, evidence_md, iteration), str)

    def match_evidence(self, content: str, cells: list[Cell]) -> list[EvidenceMatch]:
        valid_ids = {cell.id for cell in cells}

        def validate(output: list[EvidenceMatch]):
            if any(item.cell_id not in valid_ids for item in output):
                raise ModelRetry("Return only cell IDs from the supplied candidates.")
            if any(not 0 <= item.relevance <= 1 for item in output):
                raise ModelRetry("Relevance must be between 0 and 1.")
            if len({item.cell_id for item in output}) != len(output):
                raise ModelRetry("Return each matching cell once.")
            return output

        candidates = [{"cell_id": cell.id, "claim_text": cell.claim_text} for cell in cells]
        return self._run("evidence-matching", prompts.build_match_evidence_prompt(content, candidates),
                         list[EvidenceMatch], validator=validate)

"""Typed inference behind the synchronous v1 provider protocol.

A private event loop owns each model's async resources. Calls are serialized;
close() reaps subprocesses and the worker thread, including after failed runs.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelRetry, PromptedOutput
from pydantic_ai.models import Model, infer_model

from petri.models import Cell, ClarifyingQuestion, DecompositionResult
from petri.reasoning import prompts

T = TypeVar("T")
Progress = Callable[[str], None] | None


class _Worker:
    """Keep the model on one loop even when callers use different threads."""

    def __init__(self, model: Model):
        self.model = model
        self.entered = False
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
                return await operation()

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
        if isinstance(self.model, Model) and self.model.profile.get("default_structured_output_mode") == "prompted":
            output_type = PromptedOutput(schema)
        agent = Agent(self.model, output_type=output_type, retries={"output": self.retries})
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

        def validate(output: _ClaimDecomposition):
            if len(output.nodes) > max_premises:
                raise ModelRetry(f"Return at most {max_premises} premises.")
            return output

        result = self._run("decomposition", prompts.build_decompose_claim_prompt(
            claim, clarifications, guidance, max_premises), _ClaimDecomposition, on_progress, validate)
        # These placeholders carry no dish identity. The decomposer assigns real IDs.
        cells = [Cell(id=f"pending-premises-001-{index:03d}", colony_id="pending-premises",
                      claim_text=node.claim_text, level=1) for index, node in enumerate(result.nodes, 1)]
        return DecompositionResult(cells=cells, center_claim=claim)

    def decompose_why(self, premise: str, parent_level: int, parent_seq: int,
                      max_premises: int = 5, on_progress: Progress = None) -> list[dict]:
        self._limit(max_premises)

        def validate(output: _WhyDecomposition):
            if len(output.sub_premises) > max_premises:
                raise ModelRetry(f"Return at most {max_premises} sub-premises.")
            return output

        prompt = prompts.build_decompose_why_prompt(premise, parent_level, parent_seq, max_premises)
        prompt += "If this is directly verifiable as a fact, definition, or measurement, mark it atomic with no sub-premises."
        result = self._run("five-whys", prompt, _WhyDecomposition, on_progress, validate)
        return [item.model_dump() for item in result.sub_premises]

    def assess_cell(self, cell_id, claim_text, context, agent_role, on_progress=None):
        raise NotImplementedError("Cell assessment is implemented in issue #27.")

    def summarize_evidence(self, cell_id, claim_text, evidence_md, iteration):
        raise NotImplementedError("Evidence summaries are implemented in issue #27.")

    def match_evidence(self, content, cells):
        raise NotImplementedError("Evidence matching is implemented in issue #27.")

"""Typed per-cell control flow with injected phase callables.

V1 dispositions (all 14 queue states):
    queued -> DispatchNode (entry and sync guard)
    socratic_active -> SocraticNode
    research_active -> ResearchNode
    critique_active -> CritiqueNode
    mediating -> ConvergenceCheckNode
    converged -> obsolete: ConvergenceCheckNode routes directly to RedTeamNode or End
    stalled -> obsolete: End[NeedsHuman] parks the failed round
    needs_human -> End[NeedsHuman]
    red_team_active -> RedTeamNode
    evaluating -> EvaluateNode
    done -> End[Validated | Disproven]
    deferred_open -> End[DeferOpen]
    deferred_closed -> End[DeferClosed]
    sync_conflict -> End[SyncConflict]

Node return annotations define the permitted transitions. The runner checks
these annotations before dispatch: the library can route an untyped node return
outside its declared edges, so static types alone do not enforce runtime safety. Pydantic Graph 2.64
uses GraphBuilder to assemble BaseNode classes; it no longer provides the old
Graph(nodes=...) constructor. We retain BaseNode classes for explicit typed
edges, using the builder only for assembly and the entry step. The colony DAG
remains separate runtime data.

Run through run_pipeline so the dispatcher owns the iteration clock. Phase
callables receive a frozen snapshot and cannot assign the live iteration.
This module performs no inference, event writes, queue mutations, or DBOS calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_graph import BaseNode, End, EndMarker, GraphBuilder, GraphRunContext, StepContext

from pydantic_graph.exceptions import GraphRuntimeError

from petri.execution.models import CellPipelineState, PipelineOutcome


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Validated(_FrozenModel):
    status: Literal["validated"] = "validated"


class Disproven(_FrozenModel):
    status: Literal["disproven"] = "disproven"


class DeferOpen(_FrozenModel):
    status: Literal["defer_open"] = "defer_open"
    reason: str = ""


class DeferClosed(_FrozenModel):
    status: Literal["defer_closed"] = "defer_closed"
    reason: str = ""


class NeedsHuman(_FrozenModel):
    status: Literal["needs_human"] = "needs_human"
    reason: str = Field(min_length=1)


class SyncConflict(_FrozenModel):
    status: Literal["sync_conflict"] = "sync_conflict"
    reason: str = Field(min_length=1)


EvaluationEnd: TypeAlias = Validated | Disproven | DeferOpen | DeferClosed
PipelineEnd: TypeAlias = EvaluationEnd | NeedsHuman | SyncConflict
ConvergenceEnd: TypeAlias = DeferOpen | DeferClosed | NeedsHuman


class PipelineInput(_FrozenModel):
    """Read-only phase input, copied from the runner's state for each call."""

    cell_id: str
    iteration: int
    weakest_link: str | None
    focused_directive: str | None
    cycle_start_iteration: int
    max_iterations: int


class PhaseResult(_FrozenModel):
    """A phase may complete or park the round for human intervention."""

    needs_human_reason: str | None = Field(default=None, min_length=1)


class RedTeamResult(_FrozenModel):
    defer_open_reason: str | None = Field(default=None, min_length=1)


class ConvergenceDecision(_FrozenModel):
    outcome: PipelineOutcome
    weakest_link: str | None = None
    focused_directive: str | None = None
    short_circuit: Literal["needs_experiment", "defer_open", "defer_closed"] | None = None
    reason: str = ""

    @model_validator(mode="after")
    def require_short_circuit_destination(self) -> ConvergenceDecision:
        if self.outcome == PipelineOutcome.short_circuit and self.short_circuit is None:
            raise ValueError("A short circuit requires its destination")
        return self


class EvaluationVerdict(str, Enum):
    VALIDATED = "VALIDATED"
    DISPROVEN = "DISPROVEN"
    DEFER_OPEN = "DEFER_OPEN"
    DEFER_CLOSED = "DEFER_CLOSED"


@dataclass(frozen=True)
class PipelineDeps:
    socratic: Callable[[PipelineInput], Awaitable[PhaseResult]]
    research: Callable[[PipelineInput], Awaitable[PhaseResult]]
    critique: Callable[[PipelineInput], Awaitable[PhaseResult]]
    convergence: Callable[[PipelineInput], Awaitable[ConvergenceDecision]]
    red_team: Callable[[PipelineInput], Awaitable[RedTeamResult]]
    evaluate: Callable[[PipelineInput], Awaitable[EvaluationVerdict]]
    sync_check: Callable[[PipelineInput], Awaitable[SyncConflict | None]]


def _input(state: CellPipelineState) -> PipelineInput:
    return PipelineInput.model_validate(state.model_dump())


@dataclass
class DispatchNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> SocraticNode | End[SyncConflict]:
        conflict = await ctx.deps.sync_check(_input(ctx.state))
        if conflict is not None:
            return End(conflict)
        return SocraticNode()


@dataclass
class SocraticNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> ResearchNode | End[NeedsHuman]:
        result = await ctx.deps.socratic(_input(ctx.state))
        if result.needs_human_reason:
            return End(NeedsHuman(reason=result.needs_human_reason))
        return ResearchNode()


@dataclass
class ResearchNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> CritiqueNode | End[NeedsHuman]:
        result = await ctx.deps.research(_input(ctx.state))
        if result.needs_human_reason:
            return End(NeedsHuman(reason=result.needs_human_reason))
        return CritiqueNode()


@dataclass
class CritiqueNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> ConvergenceCheckNode | End[NeedsHuman]:
        result = await ctx.deps.critique(_input(ctx.state))
        if result.needs_human_reason:
            return End(NeedsHuman(reason=result.needs_human_reason))
        return ConvergenceCheckNode()


@dataclass
class ConvergenceCheckNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> RedTeamNode | ResearchNode | End[ConvergenceEnd]:
        decision = await ctx.deps.convergence(_input(ctx.state))
        if decision.outcome == PipelineOutcome.converged:
            return RedTeamNode()
        if decision.outcome == PipelineOutcome.circuit_breaker:
            return End(
                NeedsHuman(reason=decision.reason or "The convergence cycle exhausted its budget")
            )
        if decision.outcome == PipelineOutcome.short_circuit:
            if decision.short_circuit == "defer_open":
                return End(DeferOpen(reason=decision.reason))
            if decision.short_circuit == "defer_closed":
                return End(DeferClosed(reason=decision.reason))
            return End(NeedsHuman(reason=decision.reason or "An experiment requires human input"))
        ctx.state.weakest_link = decision.weakest_link
        ctx.state.focused_directive = decision.focused_directive or (
            f"Focus on addressing {decision.weakest_link} concerns"
            if decision.weakest_link
            else None
        )
        return ResearchNode()


@dataclass
class RedTeamNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> EvaluateNode | End[DeferOpen]:
        result = await ctx.deps.red_team(_input(ctx.state))
        if result.defer_open_reason:
            return End(DeferOpen(reason=result.defer_open_reason))
        return EvaluateNode()


@dataclass
class EvaluateNode(BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]):
    async def run(
        self, ctx: GraphRunContext[CellPipelineState, PipelineDeps]
    ) -> End[EvaluationEnd]:
        verdict = await ctx.deps.evaluate(_input(ctx.state))
        endings: dict[EvaluationVerdict, type[EvaluationEnd]] = {
            EvaluationVerdict.VALIDATED: Validated,
            EvaluationVerdict.DISPROVEN: Disproven,
            EvaluationVerdict.DEFER_OPEN: DeferOpen,
            EvaluationVerdict.DEFER_CLOSED: DeferClosed,
        }
        return End(endings[verdict]())


PIPELINE_NODES = (
    DispatchNode,
    SocraticNode,
    ResearchNode,
    CritiqueNode,
    ConvergenceCheckNode,
    RedTeamNode,
    EvaluateNode,
)


def declared_transitions(
    node: type[BaseNode[CellPipelineState, PipelineDeps, PipelineEnd]],
) -> tuple[set[type], set[type]]:
    """Read next-node classes and terminal payload classes from run's annotation."""
    annotation = get_type_hints(node.run)["return"]
    alternatives = get_args(annotation) if get_origin(annotation) is not End else (annotation,)
    next_nodes: set[type] = set()
    endings: set[type] = set()
    for alternative in alternatives:
        if get_origin(alternative) is End:
            payload = get_args(alternative)[0]
            endings.update(get_args(payload) or (payload,))
        else:
            next_nodes.add(alternative)
    return next_nodes, endings


# Capture before execution; runtime monkeypatches cannot weaken this contract.
_DECLARED = {node.__name__: declared_transitions(node) for node in PIPELINE_NODES}

_builder = GraphBuilder(
    name="cell_pipeline",
    state_type=CellPipelineState,
    deps_type=PipelineDeps,
    output_type=cast(type[PipelineEnd], PipelineEnd),
    auto_instrument=False,
)


@_builder.step
async def start(ctx: StepContext) -> DispatchNode:
    return DispatchNode()


_builder.add(
    *(_builder.node(node) for node in PIPELINE_NODES),
    _builder.edge_from(_builder.start_node).to(start),
)
pipeline_graph = _builder.build()


async def run_pipeline(state: CellPipelineState, deps: PipelineDeps) -> PipelineEnd:
    """Run one round; only the dispatcher advances iteration on a retry edge.

    Iteration is constant within a research/critique/convergence pass. The
    ConvergenceCheckNode -> ResearchNode edge advances it exactly once, before
    the next research callable sees state. Normal node/callable assignment to
    iteration raises a Pydantic frozen-field error. This is an API invariant,
    not a sandbox against Python code using object.__setattr__ or private data.
    """
    previous_node: str | None = None
    node_names = {node.__name__ for node in PIPELINE_NODES}
    async with pipeline_graph.iter(state=state, deps=deps) as run:
        async for tasks in run:
            if isinstance(tasks, EndMarker):
                if previous_node is None or type(tasks.value) not in _DECLARED[previous_node][1]:
                    raise GraphRuntimeError(f"Undeclared terminal result from {previous_node}")
                return tasks.value
            for task in tasks:
                if task.node_id not in node_names:
                    continue
                if previous_node is not None:
                    permitted = {node.__name__ for node in _DECLARED[previous_node][0]}
                    if task.node_id not in permitted:
                        raise GraphRuntimeError(
                            f"Undeclared transition: {previous_node} -> {task.node_id}"
                        )
                if previous_node == "ConvergenceCheckNode" and task.node_id == "ResearchNode":
                    object.__setattr__(state, "iteration", state.iteration + 1)
                previous_node = task.node_id
    raise RuntimeError("The pipeline finished without a typed result")

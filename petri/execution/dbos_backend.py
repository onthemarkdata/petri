"""Durable cell rounds with small checkpoints and replay-safe domain writes."""

from __future__ import annotations

import asyncio
import pickle
from collections import Counter
from datetime import datetime, timezone
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from dbos import DBOS, DBOSClient, SetEnqueueOptions, SetWorkflowID
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from petri.execution.config import ExecutionConfig
from petri.execution.dbos_app import (
    APPLICATION_VERSION,
    CHECKPOINT_LIMIT,
    QUEUE_NAME,
    DBOSApp,
    active_app,
)
from petri.execution.decisions import build_pipeline_dependencies
from petri.execution.models import CellPipelineState, GrowOutcome, PreflightCheck
from petri.execution.phases import PhaseServices, build_socratic_agent
from petri.execution.pipeline import (
    NeedsHuman,
    PipelineDeps,
    PipelineEnd,
    PipelineInput,
    run_pipeline,
)
from petri.harness.errors import PermanentHarnessError, RateLimitedError
from petri.models import ProcessCellResult, QueueProcessingResult
from petri.storage import database
from petri.storage.paths import dish_db_path


@dataclass(frozen=True)
class CellRunSpec:
    cell_id: str
    round_number: int
    today_iso: str
    max_iterations: int = 5
    application_version: str = APPLICATION_VERSION

    def __post_init__(self):
        if self.round_number < 0 or self.max_iterations < 1 or not self.today_iso:
            raise ValueError("Provide a nonnegative round, positive iteration budget, and date")

    @property
    def workflow_id(self) -> str:
        return f"{self.cell_id}:r{self.round_number}"


@dataclass(frozen=True)
class StepResult:
    value: Any = None
    retry_after_seconds: float | None = None
    failure: str | None = None


def _bounded(value):
    if len(pickle.dumps(value)) > CHECKPOINT_LIMIT:
        raise ValueError("Checkpoint exceeds 2 MiB; store evidence in the domain database")
    return value


class _ParkRound(Exception):
    pass


@DBOS.step(name="petri.admit_cell")
def admit_cell(spec: CellRunSpec) -> float:
    return active_app().admission_delay(spec.workflow_id)


@DBOS.step(name="petri.cell_phase")
async def cell_phase(spec: CellRunSpec, phase: str, value: PipelineInput) -> StepResult:
    deps = active_app().dependencies(spec)
    try:
        output = await getattr(deps, phase)(value)
        return _bounded(StepResult(value=output))
    except RateLimitedError as error:
        return StepResult(
            retry_after_seconds=(
                error.retry_after_seconds if error.retry_after_seconds is not None else 1.0
            )
        )
    except (PermanentHarnessError, UnexpectedModelBehavior, UsageLimitExceeded) as error:
        return StepResult(failure=str(error)[-4096:] or type(error).__name__)


@DBOS.step(name="petri.finish_cell")
async def finish_cell(spec: CellRunSpec, value: PipelineInput, result: PipelineEnd) -> None:
    finish = active_app().dependencies(spec).finish
    if finish is not None:
        await finish(value, result)


async def _phase(spec: CellRunSpec, name: str, value: PipelineInput):
    while True:
        result = await cell_phase(spec, name, value)
        if result.retry_after_seconds is not None:
            await DBOS.sleep_async(result.retry_after_seconds)
            continue
        if result.failure:
            raise _ParkRound(result.failure)
        return result.value


@DBOS.workflow(name="petri.validate_cell")
async def validate_cell(spec: CellRunSpec) -> PipelineEnd:
    if spec.application_version != APPLICATION_VERSION:
        raise RuntimeError("The workflow shape changed; run recovery with its original version")
    app = active_app()
    await app.acquire_slot()
    try:
        while delay := await asyncio.to_thread(admit_cell, spec):
            await DBOS.sleep_async(delay)
        state = CellPipelineState(cell_id=spec.cell_id, max_iterations=spec.max_iterations)
        deps = PipelineDeps(
            socratic=partial(_phase, spec, "socratic"),
            research=partial(_phase, spec, "research"),
            critique=partial(_phase, spec, "critique"),
            convergence=partial(_phase, spec, "convergence"),
            red_team=partial(_phase, spec, "red_team"),
            evaluate=partial(_phase, spec, "evaluate"),
            sync_check=partial(_phase, spec, "sync_check"),
            finish=partial(finish_cell, spec),
        )
        try:
            return _bounded(await run_pipeline(state, deps))
        except _ParkRound as error:
            result = NeedsHuman(reason=str(error))
            await finish_cell(spec, PipelineInput.model_validate(state.model_dump()), result)
            return result
    finally:
        app.release_slot()


class DBOSExecutionBackend:
    status_interval = 0.25

    def __init__(
        self,
        *,
        config: dict | None = None,
        services_factory: Callable[[CellRunSpec], PhaseServices] | None = None,
    ):
        self.config = config or {}
        self.settings = ExecutionConfig.model_validate(self.config.get("execution", {}))
        self.services_factory = services_factory
        self.app: DBOSApp | None = None

    def preflight(self) -> PreflightCheck:
        settings = self.settings
        rate = (
            f"{settings.limiter.limit} starts/{settings.limiter.period:g}s"
            if settings.limiter
            else "unlimited starts"
        )
        return PreflightCheck(
            "dbos",
            True,
            f"DBOS: {settings.worker_concurrency} concurrent cells; {rate}; "
            "binding concurrency constraint: execution.worker_concurrency (one execution owner per dish).",
        )

    def start(
        self,
        petri_dir: Path,
        *,
        db_path: Path,
        dependencies: Callable[[CellRunSpec], PipelineDeps] | None = None,
    ) -> DBOSApp:
        if dependencies is None:
            if self.services_factory is None:
                self.services_factory = self._services_factory(petri_dir, db_path)

            def dependencies(spec):
                return build_pipeline_dependencies(self.services_factory(spec))

        self.app = DBOSApp(petri_dir, db_path, self.settings, dependencies).start()
        return self.app

    async def start_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.start, *args, **kwargs)

    async def close_async(self):
        await asyncio.to_thread(self.close)

    async def enqueue_cell(self, spec: CellRunSpec, *, level: int = 0):
        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dish runtime before enqueueing a cell")
        if level < 0:
            raise ValueError("Cell level must be nonnegative")
        _bounded(spec)
        with SetWorkflowID(spec.workflow_id), SetEnqueueOptions(priority=level + 1):
            return await DBOS.enqueue_workflow_async(QUEUE_NAME, validate_cell, spec)

    def close(self):
        if self.app:
            self.app.close()

    def _services_factory(self, petri_dir: Path, db_path: Path):
        from petri.agents.config import AgentConfig
        from petri.agents.factory import build_agents
        from petri.storage.dish import resolve_dish_id

        config = AgentConfig.model_validate(self.config)
        constitution = (petri_dir / "defaults" / "constitution.md").read_text()
        agents = build_agents(config, constitution=constitution)
        questioner = build_socratic_agent(config, constitution=constitution)

        def create(spec):
            return PhaseServices(
                db_path,
                spec.round_number,
                spec.today_iso,
                config,
                agents,
                questioner,
                dish_id=resolve_dish_id(petri_dir),
            )

        return create

    def eligible_cells(self, petri_dir, *, cell_ids=None, colony_filter=None):
        from petri.storage.colonies import list_colonies

        db_path = dish_db_path(petri_dir)
        if not db_path.is_file():
            return []
        eligible = []
        for graph, colony in list_colonies(db_path):
            if colony_filter and colony_filter not in {
                colony.id,
                colony.id.removeprefix(f"{colony.dish}-"),
            }:
                continue
            statuses = {cell.id: cell.status for cell in graph.get_all_cells()}
            eligible.extend(
                cell
                for cell in graph.get_eligible_for_validation(statuses)
                if not cell_ids or cell.id in cell_ids
            )
        return eligible

    async def enqueue_colony(self, spec):
        from petri.execution.scheduler import grow_colony

        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dish runtime before enqueueing a colony")
        _bounded(spec)
        with SetWorkflowID(spec.workflow_id):
            return await DBOS.start_workflow_async(grow_colony, spec)

    async def grow_dish(
        self, petri_dir, *, colony_filter=None, cell_ids=None, dry_run=False, round_number=1
    ):
        from petri.execution.scheduler import ColonyRunSpec
        from petri.storage.colonies import list_colonies

        if dry_run:
            return [
                cell.id
                for cell in self.eligible_cells(
                    petri_dir, cell_ids=cell_ids, colony_filter=colony_filter
                )
            ]
        handles = []
        for _, colony in list_colonies(dish_db_path(petri_dir)):
            if colony_filter and colony_filter not in {
                colony.id,
                colony.id.removeprefix(f"{colony.dish}-"),
            }:
                continue
            handles.append(
                await self.enqueue_colony(
                    ColonyRunSpec(
                        colony.id,
                        round_number,
                        datetime.now(timezone.utc).date().isoformat(),
                        max_iterations=self.config.get("max_iterations", 5),
                        cell_ids=tuple(sorted(cell_ids or [])),
                    )
                )
            )
        return handles

    async def reopen_cell(self, request):
        from petri.execution.scheduler import revalidate_cell

        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dish runtime before reopening a cell")
        if not request.request_id:
            raise ValueError("Provide a stable request ID to deduplicate revalidation")
        _bounded(request)
        with SetWorkflowID(request.workflow_id):
            return await DBOS.start_workflow_async(revalidate_cell, request)

    def process_queue(
        self,
        *,
        petri_dir,
        provider=None,
        max_concurrent=4,
        cell_ids=None,
        colony_filter=None,
        all_cells=True,
        dry_run=False,
        on_event=None,
    ):
        if dry_run:
            return QueueProcessingResult(
                dry_run=True,
                would_process=[
                    cell.id
                    for cell in self.eligible_cells(
                        petri_dir, cell_ids=cell_ids, colony_filter=colony_filter
                    )
                ],
            )
        path = dish_db_path(petri_dir)
        if not path.is_file():
            return QueueProcessingResult()
        self.start(petri_dir, db_path=path)

        async def run():
            handles = await self.grow_dish(
                petri_dir, colony_filter=colony_filter, cell_ids=cell_ids
            )
            colonies = await asyncio.gather(*(handle.get_result() for handle in handles))
            completed = sorted({cell for colony in colonies for cell in colony.completed})
            results = [
                ProcessCellResult(cell_id=cell, final_state=database.get_cell(path, cell)["status"])
                for cell in completed
            ]
            return QueueProcessingResult(
                processed=len(results),
                succeeded=sum(
                    result.final_state in {"VALIDATED", "DISPROVEN", "DEFER_OPEN", "DEFER_CLOSED"}
                    for result in results
                ),
                stalled=sum(
                    result.final_state in {"STALLED", "NEEDS_EXPERIMENT"} for result in results
                ),
                results=results,
            )

        try:
            return asyncio.run(run())
        finally:
            self.close()

    def grow_loop(self, *, run_one_pass, get_states, is_stopped):
        result = run_one_pass()
        states = get_states()
        return GrowOutcome(
            "not_eligible" if states.get("queued") else "all_terminal", states, result, 1
        )

    def entries(self, petri_dir: Path) -> dict[str, dict]:
        db_path = dish_db_path(petri_dir)
        if not db_path.is_file():
            return {}
        states = {
            "NEW": "queued",
            "RESEARCH": "research_active",
            "RED_TEAM": "red_team_active",
            "EVALUATE": "evaluating",
            "VALIDATED": "done",
            "DISPROVEN": "done",
            "DEFER_OPEN": "deferred_open",
            "DEFER_CLOSED": "deferred_closed",
            "STALLED": "needs_human",
            "NEEDS_EXPERIMENT": "needs_human",
        }
        return {
            row["id"]: {
                "cell_id": row["id"],
                "queue_state": states[row["status"]],
                "colony_id": row["colony"],
                "level": row["level"],
                "priority": row["level"] + 1,
                "iteration": 0,
            }
            for row in database.list_cells(db_path)
        }

    def list_entries(self, petri_dir: Path) -> list[dict]:
        return list(self.entries(petri_dir).values())

    def state_summary(self, petri_dir: Path) -> dict[str, int]:
        return dict(Counter(entry["queue_state"] for entry in self.list_entries(petri_dir)))

    def format_state_summary(self, states: dict[str, int]) -> str:
        return ", ".join(f"{name}: {count}" for name, count in sorted(states.items()))

    def clear_stop(self, petri_dir: Path) -> None:
        pass

    def is_stopped(self, petri_dir: Path) -> bool:
        return False

    def request_stop(self, petri_dir: Path) -> None:
        path = petri_dir / "dbos.sqlite"
        if not path.exists():
            return
        client = DBOSClient(system_database_url=f"sqlite:///{path.resolve()}")
        try:
            for run in client.list_workflows(status=["PENDING", "ENQUEUED"]):
                client.cancel_workflow(run.workflow_id)
        finally:
            client.destroy()

    def update_state(self, petri_dir: Path, cell_id: str, state: str) -> None:
        if state == "queued":
            raise ValueError("Start a new validation round to reopen a cell")
        self.request_stop(petri_dir)

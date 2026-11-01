"""Durable cell rounds with small checkpoints and replay-safe domain writes."""

from __future__ import annotations

import asyncio
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, replace, asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from dbos import DBOS, DBOSClient, SetEnqueueOptions, SetWorkflowID
from dbos._error import DBOSWorkflowCancelledError, DBOSAwaitedWorkflowCancelledError
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
from petri.execution.models import CellPipelineState, GrowOutcome, PreflightCheck, ProgressEvent
from petri.execution.run_controls import RunRequest, RunStatus, SeedRequest
from petri.execution.phases import PhaseServices, build_socratic_agent
from petri.execution.pipeline import (
    NeedsHuman,
    PipelineDeps,
    PipelineEnd,
    PipelineInput,
    SyncConflict,
    run_pipeline,
)
from petri.harness.errors import PermanentHarnessError, RateLimitedError
from petri.execution.reconciliation import DomainReceipt, capture_receipt, read_conflicts
from petri.models import CellStatus, ProcessCellResult, QueueProcessingResult, parse_key
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
class ParkedCellHandle:
    workflow_id: str
    reason: str

    async def get_result(self) -> SyncConflict:
        return SyncConflict(reason=self.reason)


@dataclass(frozen=True)
class StepResult:
    value: Any = None
    retry_after_seconds: float | None = None
    failure: str | None = None
    domain_receipt: DomainReceipt | None = None


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
    from petri.execution.control import cancellation_mode

    app = active_app()
    deps = app.dependencies(spec)
    with app.step_activity():
        task = asyncio.create_task(getattr(deps, phase)(value))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                if not task.done() and cancellation_mode(
                    app.petri_dir / "dbos.sqlite", spec.workflow_id
                ) == "force":
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise DBOSWorkflowCancelledError("Force stop requested by user")
            output = task.result()
            return _bounded(
                StepResult(value=output, domain_receipt=capture_receipt(app.db_path, spec.cell_id))
            )
        except RateLimitedError as error:
            return StepResult(retry_after_seconds=(
                error.retry_after_seconds if error.retry_after_seconds is not None else 1.0
            ))
        except (PermanentHarnessError, UnexpectedModelBehavior, UsageLimitExceeded) as error:
            return StepResult(failure=str(error)[-4096:] or type(error).__name__)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@DBOS.step(name="petri.finish_cell")
async def finish_cell(
    spec: CellRunSpec, value: PipelineInput, result: PipelineEnd
) -> DomainReceipt | None:
    if isinstance(result, SyncConflict):
        return None
    finish = active_app().dependencies(spec).finish
    if finish is not None:
        await finish(value, result)
    return _bounded(capture_receipt(active_app().db_path, spec.cell_id))


async def _finish(spec, value, result) -> None:
    await finish_cell(spec, value, result)


async def _phase(spec: CellRunSpec, name: str, value: PipelineInput):
    from petri.execution.control import cancel_at_boundary

    app = active_app()
    while True:
        result = await cell_phase(spec, name, value)
        cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
        if result.retry_after_seconds is not None:
            await DBOS.sleep_async(result.retry_after_seconds)
            continue
        if result.failure:
            raise _ParkRound(result.failure)
        return result.value


@DBOS.workflow(name="petri.validate_cell")
async def validate_cell(spec: CellRunSpec) -> PipelineEnd:
    from petri.execution.otel import set_cell_workflow_attributes

    set_cell_workflow_attributes(spec, active_app(), DBOS.span)
    if spec.application_version != APPLICATION_VERSION:
        raise RuntimeError("The workflow shape changed; run recovery with its original version")
    app = active_app()
    from petri.execution.control import cancel_at_boundary

    cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
    if spec.cell_id in app.conflicts:
        return SyncConflict(reason=app.conflicts[spec.cell_id]["reason"])
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
            finish=partial(_finish, spec),
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
    backend_name = "dbos"

    def __init__(
        self,
        *,
        config: dict | None = None,
        services_factory: Callable[[CellRunSpec], PhaseServices] | None = None,
        after_pass: Callable[[Path], Awaitable[bool]] | None = None,
    ):
        self.config = config or {}
        self.settings = ExecutionConfig.model_validate(self.config.get("execution", {}))
        self.services_factory = services_factory
        self.after_pass = after_pass
        self.app: DBOSApp | None = None
        self._stopped = False

    async def start_owner(
        self, petri_dir: Path, *, on_phase_progress=None, event_stream_handler_factory=None
    ) -> None:
        if self.app is not None and self.app.started:
            return
        await self.start_async(
            petri_dir,
            db_path=dish_db_path(petri_dir),
            on_phase_progress=on_phase_progress,
            event_stream_handler_factory=event_stream_handler_factory,
        )

    async def shutdown_owner(self) -> None:
        if self.app is None or not self.app.started:
            return
        app = self.app
        if app._active_steps or await DBOS.list_workflows_async(status=["PENDING", "ENQUEUED"], limit=1):
            await asyncio.to_thread(app.detach)
        else:
            await self.close_async()

    def _run_status(self, status) -> RunStatus:
        result = status.output
        if is_dataclass(result) and not isinstance(result, type):
            result = asdict(result)
        elif isinstance(result, list):
            result = [
                asdict(item) if is_dataclass(item) and not isinstance(item, type) else item
                for item in result
            ]
        return RunStatus(
            id=status.workflow_id,
            command=status.name,
            status=status.status,
            created_at=status.created_at,
            updated_at=status.updated_at,
            result=result,
            error=str(status.error)[-4096:] if status.error is not None else None,
        )

    async def get_run(self, petri_dir: Path, run_id: str) -> RunStatus | None:
        system = petri_dir / "dbos.sqlite"
        if not system.is_file():
            return None
        client = DBOSClient(system_database_url=f"sqlite:///{system.resolve()}")
        try:
            rows = await client.list_workflows_async(workflow_ids=[run_id])
            return self._run_status(rows[0]) if rows else None
        finally:
            client.destroy()

    async def cancel_run(self, petri_dir: Path, run_id: str) -> RunStatus | None:
        current = await self.get_run(petri_dir, run_id)
        if current is None:
            return None
        client = DBOSClient(
            system_database_url=f"sqlite:///{(petri_dir / 'dbos.sqlite').resolve()}"
        )
        try:
            await client.cancel_workflow_async(run_id, cancel_children=True)
        finally:
            client.destroy()
        return await self.get_run(petri_dir, run_id)

    async def submit_seed(self, request: SeedRequest) -> RunStatus:
        from petri.execution.dashboard_runs import DashboardSeedSpec, dashboard_seed
        from petri.execution.seed_workflow import SeedRunSpec
        from petri.query.cells import list_colonies
        from petri.reasoning.decomposition_settings import DecompositionSettings

        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dashboard execution owner before submitting a run.")
        dish_id = self.app.db_path.parent.name
        colony_id = f"{dish_id}-{request.colony}"
        existing = next(
            (row for row in list_colonies(self.app.db_path) if row["id"] == colony_id), None
        )
        initial = await DBOS.get_workflow_status_async(colony_id)
        if not request.regenerate:
            if initial is not None:
                if existing is not None and existing["center_claim"] != request.claim:
                    raise ValueError(
                        "This colony contains a different claim; request regeneration explicitly."
                    )
                return self._run_status(initial)
            if existing is not None:
                raise ValueError(
                    "This colony already exists; request regeneration to replace its draft."
                )
        elif initial is not None and initial.status in {"PENDING", "ENQUEUED"}:
            raise ValueError("Wait for or cancel the existing seed run before regenerating it.")
        attempt = 1
        settings = DecompositionSettings.from_dish(self.app.petri_dir)
        while True:
            seed = SeedRunSpec(
                colony_id, dish_id, str(request.colony), settings, attempt, request.guidance
            )
            previous = await DBOS.get_workflow_status_async(seed.workflow_id)
            if previous is None:
                break
            if previous.status in {"PENDING", "ENQUEUED"}:
                raise ValueError("Wait for or cancel the existing seed run before regenerating it.")
            attempt += 1
        spec = DashboardSeedSpec(request, seed)
        with SetWorkflowID(spec.workflow_id):
            handle = await DBOS.start_workflow_async(dashboard_seed, spec)
        status = await DBOS.get_workflow_status_async(handle.workflow_id)
        return self._run_status(status)

    async def submit_run(self, request: RunRequest) -> RunStatus:
        from petri.execution.dashboard_runs import DashboardGrowSpec, dashboard_grow
        from petri.execution.scheduler import ColonyRunSpec, ReopenRequest
        from petri.query.cells import get_cell, list_colonies

        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dashboard execution owner before submitting a run.")
        if request.command == "cell":
            cell = get_cell(self.app.db_path, str(request.cell_id))
            if cell is None:
                raise KeyError("Cell not found.")
            handle = await self.enqueue_cell(
                CellRunSpec(
                    str(request.cell_id),
                    request.round_number,
                    request.today.isoformat(),
                    self.config.get("max_iterations", 3),
                ),
                level=cell["level"],
            )
        elif request.command == "revalidate":
            handle = await self.reopen_cell(
                ReopenRequest(
                    str(request.cell_id),
                    request.request_id or str(uuid4()),
                    request.today.isoformat(),
                    max_iterations=self.config.get("max_iterations", 3),
                )
            )
        else:
            colonies = list_colonies(self.app.db_path)
            if request.colony is not None:
                colonies = [
                    row
                    for row in colonies
                    if request.colony in {row["id"], row["id"].removeprefix(row["dish"] + "-")}
                ]
            if not colonies:
                raise KeyError("Colony not found.")
            specs = tuple(
                ColonyRunSpec(
                    row["id"],
                    request.round_number,
                    request.today.isoformat(),
                    self.config.get("max_iterations", 3),
                    tuple(request.cell_ids),
                )
                for row in colonies
            )
            spec = DashboardGrowSpec("grow:" + str(uuid4()), specs)
            with SetWorkflowID(spec.id):
                handle = await DBOS.start_workflow_async(dashboard_grow, spec)
        status = await DBOS.get_workflow_status_async(handle.workflow_id)
        return self._run_status(status)

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
        seed_runner_factory=None,
        seed_on_progress=None,
        seed_on_cell_created=None,
        on_phase_progress=None,
        event_stream_handler_factory=None,
    ) -> DBOSApp:
        if dependencies is None:
            if self.services_factory is None:
                self.services_factory = self._services_factory(petri_dir, db_path)

            def dependencies(spec):
                services = self.services_factory(spec)
                previous = services.on_progress
                def progress(event):
                    if previous is not None:
                        previous(event)
                    if on_phase_progress is not None:
                        on_phase_progress(event)
                services = replace(services,
                    on_progress=progress if on_phase_progress is not None else previous,
                    event_stream_handler_factory=(event_stream_handler_factory
                                                  or services.event_stream_handler_factory))
                return build_pipeline_dependencies(services)

        self.app = DBOSApp(
            petri_dir,
            db_path,
            self.settings,
            dependencies,
            seed_runner_factory=seed_runner_factory or self._default_seed_runner,
            seed_on_progress=seed_on_progress,
            seed_on_cell_created=seed_on_cell_created,
            owns_seed_runners=seed_runner_factory is None,
        ).start()
        return self.app

    async def start_async(self, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            call = partial(copy_context().run, partial(self.start, *args, **kwargs))
            return await asyncio.get_running_loop().run_in_executor(executor, call)

    async def close_async(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            await asyncio.get_running_loop().run_in_executor(executor, self.close)

    async def enqueue_cell(self, spec: CellRunSpec, *, level: int = 0):
        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dish runtime before enqueueing a cell")
        if level < 0:
            raise ValueError("Cell level must be nonnegative")
        _bounded(spec)
        if spec.cell_id in self.app.conflicts:
            return ParkedCellHandle(spec.workflow_id, self.app.conflicts[spec.cell_id]["reason"])
        from petri.execution.priorities import enqueue_with_priority

        status = await DBOS.get_workflow_status_async(spec.workflow_id)
        if status is not None and status.status == "CANCELLED":
            return await DBOS.resume_workflow_async(spec.workflow_id, queue_name=QUEUE_NAME)
        return await enqueue_with_priority(self.app, spec, level)

    def close(self):
        if self.app:
            self.app.close()

    def _default_seed_runner(self, spec):
        from petri.harness.legacy_bridge import HarnessInferenceProvider
        from petri.harness.resolve import resolve_model

        return HarnessInferenceProvider(resolve_model(self.config), config=self.config)

    async def enqueue_seed(self, spec, *, resume=False):
        from petri.execution.seed_workflow import seed_colony

        if self.app is None or not self.app.started:
            raise RuntimeError("Start the dish runtime before enqueueing a seed")
        _bounded(spec)
        if resume:
            status = await DBOS.get_workflow_status_async(spec.workflow_id)
            if status is not None and status.status == "ERROR":
                steps = await DBOS.list_workflow_steps_async(spec.workflow_id)
                failed = [step["function_id"] for step in steps if step["error"] is not None]
                if failed:
                    return await DBOS.rewind_workflow_async(
                        spec.workflow_id, start_step=min(failed)
                    )
            if status is not None and status.status in {
                "CANCELLED",
                "MAX_RECOVERY_ATTEMPTS_EXCEEDED",
            }:
                return await DBOS.resume_workflow_async(spec.workflow_id)
        with SetWorkflowID(spec.workflow_id), SetEnqueueOptions(priority=1):
            return await DBOS.enqueue_workflow_async(QUEUE_NAME, seed_colony, spec)

    def seed_colony(
        self,
        *,
        petri_dir,
        claim,
        clarifications,
        dish_id,
        colony_name,
        provider,
        settings,
        guidance="",
        db_path=None,
        on_progress=None,
        on_cell_created=None,
        center=None,
        agent_runner=None,
        resume=False,
    ):
        from petri.execution.seed_workflow import SeedRunSpec
        from petri.reasoning.seed_events import current_attempt
        from petri.storage.colonies import load_colony
        from petri.models import DecompositionResult

        path = db_path or dish_db_path(petri_dir, dish_id)
        graph, colony = load_colony(path, f"{dish_id}-{colony_name}")
        spec = SeedRunSpec(
            colony.id,
            dish_id,
            colony_name,
            settings,
            current_attempt(path, colony.center_cell_id),
            guidance,
        )
        runner = agent_runner or provider
        if not hasattr(runner, "initial_decomposition"):
            raise ValueError("DBOS seed requires the configured typed decomposition runner")

        def cell_dependencies(spec):
            raise RuntimeError("Cell validation is unavailable in a seed-only runtime")

        def report_created(cell, edges):
            if cell.colony_id == spec.colony_id and on_cell_created:
                on_cell_created(cell, edges)

        self.start(
            petri_dir,
            db_path=path,
            dependencies=None if "agents" in self.config else cell_dependencies,
            seed_runner_factory=lambda spec: runner,
            seed_on_progress=on_progress,
            seed_on_cell_created=report_created,
        )
        try:

            async def run():
                handle = await self.enqueue_seed(spec, resume=resume)
                return await handle.get_result()

            asyncio.run(run())
        except KeyboardInterrupt:
            self.app.closing = True
            if hasattr(runner, "cancel_active_run"):
                runner.cancel_active_run()
            if hasattr(runner, "close"):
                runner.close()
            self.app.detach()
            raise
        finally:
            self.close()
        graph, colony = load_colony(path, colony.id)
        return DecompositionResult(
            cells=graph.get_all_cells(),
            edges=graph.get_edges(),
            colony_name=colony_name,
            center_claim=colony.center_claim,
        )

    def _services_factory(self, petri_dir: Path, db_path: Path):
        from petri.agents.config import AgentConfig
        from petri.agents.factory import build_agents
        from petri.storage.dish import resolve_dish_id

        config = AgentConfig.model_validate(self.config)
        constitution = (petri_dir / "defaults" / "constitution.md").read_text()
        agents = None
        questioner = None

        def create(spec):
            nonlocal agents, questioner
            if agents is None:
                agents = build_agents(config, constitution=constitution)
                questioner = build_socratic_agent(config, constitution=constitution)
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
        statuses = {row["id"]:CellStatus(row["status"]) for row in database.list_cells(db_path)}
        for graph, colony in list_colonies(db_path):
            if colony_filter and colony_filter not in {
                colony.id,
                colony.id.removeprefix(f"{colony.dish}-"),
            }:
                continue
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
        status = await DBOS.get_workflow_status_async(spec.workflow_id)
        if status is not None and status.status == "CANCELLED":
            return await DBOS.resume_workflow_async(spec.workflow_id)
        with SetWorkflowID(spec.workflow_id):
            return await DBOS.start_workflow_async(grow_colony, spec)

    async def grow_dish(
        self, petri_dir, *, colony_filter=None, cell_ids=None, dry_run=False, round_number=None
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
            generation = round_number
            if generation is None:
                prior = await DBOS.list_workflows_async(name="petri.grow_colony")
                prefix = f"{colony.id}:grow:r"
                selected = tuple(sorted(cell_ids or []))
                suffix = ColonyRunSpec(colony.id, 1, "generation", cell_ids=selected).workflow_id.removeprefix(prefix + "1")
                matches = []
                for run in prior:
                    if not run.workflow_id.startswith(prefix):
                        continue
                    number, _, scope = run.workflow_id[len(prefix):].partition(":")
                    if number.isdigit() and (f":{scope}" if scope else "") == suffix:
                        matches.append((int(number), run.status))
                generation, status = max(matches, default=(1, "NEW"))
                if status == "SUCCESS" and self.eligible_cells(
                    petri_dir, cell_ids=cell_ids, colony_filter=colony.id
                ):
                    generation += 1
            handles.append(
                await self.enqueue_colony(
                    ColonyRunSpec(
                        colony.id,
                        generation,
                        datetime.now(timezone.utc).date().isoformat(),
                        max_iterations=self.config.get("max_iterations", 5),
                        cell_ids=tuple(sorted(cell_ids or [])),
                    )
                )
            )
        return handles

    def reopen_cell(self, request, *, petri_dir=None):
        from petri.execution.scheduler import revalidate_cell

        async def submit():
            if self.app is None or not self.app.started:
                raise RuntimeError("Start the dish runtime before reopening a cell")
            if not request.request_id:
                raise ValueError("Provide a stable request ID to deduplicate revalidation")
            _bounded(request)
            with SetWorkflowID(request.workflow_id):
                return await DBOS.start_workflow_async(revalidate_cell, request)

        if petri_dir is None:
            return submit()
        self.start(petri_dir, db_path=dish_db_path(petri_dir))
        async def run():
            handle = await submit()
            return await handle.get_result()
        try:
            return asyncio.run(run())
        finally:
            self.close()

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
        self.clear_stop(petri_dir)
        self.start(petri_dir, db_path=path)

        async def poll_progress(done):
            slots = {}
            last = {}
            while not done.is_set():
                runs = await DBOS.list_workflows_async(name="petri.validate_cell")
                active = {
                    run.workflow_id.rpartition(":r")[0]
                    for run in runs if run.status == "PENDING"
                }
                for cell_id in list(slots):
                    if cell_id not in active:
                        on_event(ProgressEvent(slots.pop(cell_id), cell_id, "finished"))
                        last.pop(cell_id, None)
                for cell_id in sorted(active):
                    if cell_id not in slots:
                        available = set(range(self.settings.worker_concurrency)) - set(slots.values())
                        if not available:
                            continue
                        slots[cell_id] = min(available)
                        on_event(ProgressEvent(slots[cell_id], cell_id, "started"))
                    events = database.read_events(path, cell_id=cell_id)
                    row = database.get_cell(path, cell_id) or {}
                    latest = events[-1] if events else {}
                    state = (row.get("status", "NEW"), latest.get("id"))
                    if last.get(cell_id) != state:
                        last[cell_id] = state
                        on_event(ProgressEvent(
                            slots[cell_id], cell_id, "phase", phase=state[0].lower(),
                            agent=latest.get("agent"), iteration=latest.get("iteration"),
                        ))
                try:
                    await asyncio.wait_for(done.wait(), timeout=self.status_interval)
                except TimeoutError:
                    pass
            for cell_id, slot in slots.items():
                on_event(ProgressEvent(slot, cell_id, "finished"))

        async def run():
            # Resume children outside their parent replay so no new parent step is inserted.
            cancelled = await DBOS.list_workflows_async(
                name="petri.validate_cell", status=["CANCELLED"]
            )
            for child in cancelled:
                await DBOS.resume_workflow_async(child.workflow_id, queue_name=QUEUE_NAME)
            done = asyncio.Event()
            poller = asyncio.create_task(poll_progress(done)) if on_event else None
            try:
                colonies = []
                while True:
                    handles = await self.grow_dish(
                        petri_dir, colony_filter=colony_filter, cell_ids=cell_ids
                    )
                    colonies.extend(await asyncio.gather(*(handle.get_result() for handle in handles)))
                    # Another colony may have validated a previously blocking premise.
                    if any(
                        cell.id not in self.app.conflicts
                        for cell in self.eligible_cells(
                            petri_dir, cell_ids=cell_ids, colony_filter=colony_filter
                        )
                    ):
                        continue
                    if self.after_pass is None or not await self.after_pass(petri_dir):
                        break
                completed = sorted({
                    cell for colony in colonies for cell in colony.completed
                    if cell not in self.app.conflicts
                })
                results = [
                    ProcessCellResult(cell_id=cell, final_state=database.get_cell(path, cell)["status"])
                    for cell in completed
                ]
                return QueueProcessingResult(
                    processed=len(results),
                    succeeded=sum(result.final_state in {
                        "VALIDATED", "DISPROVEN", "DEFER_OPEN", "DEFER_CLOSED"
                    } for result in results),
                    stalled=sum(result.final_state in {
                        "STALLED", "NEEDS_EXPERIMENT"
                    } for result in results),
                    results=results,
                )
            except (DBOSWorkflowCancelledError, DBOSAwaitedWorkflowCancelledError):
                self._stopped = True
                while self.app._active_steps:
                    await asyncio.sleep(0.05)
                return QueueProcessingResult()
            finally:
                done.set()
                if poller:
                    await poller

        try:
            return asyncio.run(run())
        except BaseException:
            self.request_stop(petri_dir, force=True)
            self.app.detach()
            raise
        finally:
            self.close()

    def grow_loop(self, *, run_one_pass, get_states, is_stopped):
        result = run_one_pass()
        states = get_states()
        reason = (
            "user_stop" if self._stopped else
            "sync_conflict"
            if states.get("sync_conflict")
            else "not_eligible"
            if states.get("queued")
            else "all_terminal"
        )
        return GrowOutcome(reason, states, result, 1)

    def entries(self, petri_dir: Path) -> dict[str, dict]:
        db_path = dish_db_path(petri_dir)
        rows = database.list_cells(db_path) if db_path.is_file() else []
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
        entries = {
            row["id"]: {
                "cell_id": row["id"],
                "queue_state": states[row["status"]],
                "colony_id": row["colony"],
                "level": row["level"],
                "priority": row["level"] + 1,
                "iteration": 0,
            }
            for row in rows
        }
        for cell_id, conflict in read_conflicts(petri_dir / "dbos.sqlite").items():
            entries.setdefault(
                cell_id,
                {
                    "cell_id": cell_id,
                    "colony_id": parse_key(cell_id).get("colony_prefix", ""),
                    "level": 0,
                    "iteration": 0,
                },
            )
            entries[cell_id].update(
                queue_state="sync_conflict",
                error=conflict["reason"],
                sync_conflict_reason=conflict["reason"],
            )
        return entries

    def list_entries(self, petri_dir: Path) -> list[dict]:
        return list(self.entries(petri_dir).values())

    def state_summary(self, petri_dir: Path) -> dict[str, int]:
        return dict(Counter(entry["queue_state"] for entry in self.list_entries(petri_dir)))

    def format_state_summary(self, states: dict[str, int]) -> str:
        return ", ".join(f"{name}: {count}" for name, count in sorted(states.items()))

    def clear_stop(self, petri_dir: Path) -> None:
        from petri.execution.control import clear_cancellation

        clear_cancellation(petri_dir / "dbos.sqlite")
        self._stopped = False

    def is_stopped(self, petri_dir: Path) -> bool:
        from petri.execution.control import has_cancellation

        return self._stopped or has_cancellation(petri_dir / "dbos.sqlite")

    def request_stop(self, petri_dir: Path, *, force: bool = False) -> int:
        from petri.execution.control import record_cancellation

        path = petri_dir / "dbos.sqlite"
        if not path.exists():
            return 0
        client = DBOSClient(system_database_url=f"sqlite:///{path.resolve()}")
        try:
            runs = client.list_workflows(status=["PENDING", "ENQUEUED"])
            ids = [run.workflow_id for run in runs]
            if ids:
                record_cancellation(path, ids, force=force)
                immediate = ids if force else [
                    run.workflow_id for run in runs if run.status == "ENQUEUED"
                ]
                if immediate:
                    client.cancel_workflows(immediate)
                self._stopped = True
            return len(ids)
        finally:
            client.destroy()

    def update_state(self, petri_dir: Path, cell_id: str, state: str) -> None:
        if state == "queued":
            raise ValueError("Start a new validation round to reopen a cell")
        self.request_stop(petri_dir)

"""Durable typed decomposition with model outputs journaled before cell writes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dbos import DBOS

from petri.execution.dbos_app import active_app
from petri.execution.dbos_backend import _bounded
from petri.harness.errors import RateLimitedError
from petri.models import ClarifyingQuestion
from petri.reasoning.decomposition_runner import decompose_typed
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.reasoning.seed_events import record_completion
from petri.storage import database
from petri.storage.colonies import load_colony
from petri.storage.event_log import append_event


@dataclass(frozen=True)
class SeedRunSpec:
    colony_id: str
    dish_id: str
    colony_name: str
    settings: DecompositionSettings
    attempt: int = 0
    guidance: str = ""

    @property
    def workflow_id(self):
        return (
            self.colony_id if self.attempt <= 1 else f"{self.colony_id}:seed-attempt{self.attempt}"
        )


@dataclass(frozen=True)
class SeedProgress:
    colony_id: str
    complete: bool = False
    cell_count: int = 0
    retry_after_seconds: float | None = None


class _NextStep(Exception):
    pass


class _OneCallRunner:
    """Limit a step to one new model call; committed outputs come from events."""

    def __init__(self, runner):
        self.runner = runner
        self.called = False

    def _call(self, method, request, on_progress):
        if self.called:
            raise _NextStep()
        self.called = True
        return getattr(self.runner, method)(request, on_progress)

    def initial_decomposition(self, request, on_progress=None):
        return self._call("initial_decomposition", request, on_progress)

    def expand_why(self, request, on_progress=None):
        return self._call("expand_why", request, on_progress)


def _rate_limit(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, RateLimitedError):
            return error.retry_after_seconds if error.retry_after_seconds is not None else 1.0
        error = error.__cause__
    return None


@DBOS.step(name="petri.decompose_seed_step")
def decompose_seed_step(spec: SeedRunSpec) -> SeedProgress:
    """Run one new call and its domain writes, returning only a progress reference."""
    app = active_app()
    with app.step_activity():
        return _decompose_seed_step(spec, app)


def _decompose_seed_step(spec, app):
    graph, colony = load_colony(app.db_path, spec.colony_id)
    center = graph.get_cell(colony.center_cell_id)
    runner = _OneCallRunner(app.seed_runner(spec))

    def created(cell, edges):
        append_event(
            database.SQLiteEventLog(app.db_path),
            cell.id,
            "cell_created",
            "decomposition_lead",
            0,
            {"level": cell.level, "parents": [edge.from_cell for edge in edges]},
            node="seed",
        )
        if app.seed_on_cell_created:
            app.seed_on_cell_created(cell, edges)

    try:
        result = decompose_typed(
            colony.center_claim,
            [ClarifyingQuestion.model_validate(item) for item in colony.clarifications],
            spec.dish_id,
            spec.colony_name,
            runner,
            settings=spec.settings,
            guidance=spec.guidance,
            db_path=app.db_path,
            center=center,
            resume=True,
            on_progress=app.seed_on_progress,
            on_cell_created=created,
        )
        record_completion(
            app.db_path, center.id, attempt=spec.attempt, cell_count=len(result.cells)
        )
        return _bounded(SeedProgress(spec.colony_id, complete=True, cell_count=len(result.cells)))
    except _NextStep:
        return SeedProgress(spec.colony_id)
    except Exception as error:
        if app.closing:
            return SeedProgress(spec.colony_id)
        delay = _rate_limit(error)
        if delay is not None:
            return SeedProgress(spec.colony_id, retry_after_seconds=delay)
        raise


@DBOS.step(name="petri.admit_seed")
def admit_seed(spec: SeedRunSpec) -> float:
    return active_app().admission_delay(f"seed:{spec.workflow_id}")


@DBOS.workflow(name="petri.seed_colony")
async def seed_colony(spec: SeedRunSpec) -> SeedProgress:
    app = active_app()
    await app.acquire_slot()
    try:
        while delay := await asyncio.to_thread(admit_seed, spec):
            await DBOS.sleep_async(delay)
        while True:
            progress = await asyncio.to_thread(decompose_seed_step, spec)
            if progress.complete:
                return progress
            if progress.retry_after_seconds is not None:
                await DBOS.sleep_async(progress.retry_after_seconds)
    finally:
        app.release_slot()

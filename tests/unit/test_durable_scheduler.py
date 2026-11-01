"""Dependency ordering, terminal gates, and durable round identity."""

import time
from dataclasses import replace

import pytest
from dbos import DBOS

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.pipeline import EvaluationVerdict, NeedsHuman, Validated
from petri.execution.scheduler import ColonyRunSpec, ReopenRequest
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.fixture
def scheduler_fixture(tmp_path, canonical_colony):
    (tmp_path / "petri.yaml").write_text("name: test-dish\n")
    path = dish_db_path(tmp_path)
    save_colony(path, canonical_colony["graph"], canonical_colony["colony_model"])
    starts = []
    verdicts = {}

    def dependencies(spec):
        phases = StubPhases(verdict=verdicts.get(spec.cell_id, EvaluationVerdict.VALIDATED))

        async def research(value):
            for edge in database.read_edges(path):
                if edge["from_cell"] == value.cell_id:
                    assert database.get_cell(path, edge["to_cell"])["status"] == "VALIDATED"
            starts.append((value.cell_id, spec.round_number, time.monotonic()))
            return await phases.research(value)

        async def finish(value, result):
            status = result.cell_status if isinstance(result, NeedsHuman) else result.status.upper()
            with database._write(path) as conn:
                conn.execute("UPDATE cells SET status = ? WHERE id = ?", (status, value.cell_id))

        return replace(phases.dependencies(), research=research, finish=finish)

    backend = DBOSExecutionBackend(
        config={"execution": {"backend": "dbos", "worker_concurrency": 2}}
    )
    return tmp_path, path, backend, dependencies, starts, verdicts


@pytest.mark.asyncio
async def test_one_parent_grows_three_levels_once(scheduler_fixture, canonical_colony):
    directory, path, backend, dependencies, starts, _ = scheduler_fixture
    await backend.start_async(directory, db_path=path, dependencies=dependencies)
    try:
        spec = ColonyRunSpec(canonical_colony["colony_id"], 1, "2026-11-01")
        first = await backend.enqueue_colony(spec)
        second = await backend.enqueue_colony(spec)
        result = await first.get_result()
        assert await second.get_result() == result
        assert len(result.completed) == 5
        assert result.not_eligible == ()
        assert len(starts) == 5
        assert [database.get_cell(path, cell)["level"] for cell, _, _ in starts] == [2, 2, 1, 1, 0]
        assert len(await DBOS.list_workflows_async(name="petri.grow_colony")) == 1
        assert len(await DBOS.list_workflows_async(name="petri.validate_cell")) == 5
    finally:
        await backend.close_async()


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [EvaluationVerdict.DISPROVEN, EvaluationVerdict.DEFER_OPEN])
async def test_nonvalidated_leaf_blocks_dependents(scheduler_fixture, canonical_colony, verdict):
    directory, path, backend, dependencies, starts, verdicts = scheduler_fixture
    verdicts[canonical_colony["cell1"].id] = verdict
    await backend.start_async(directory, db_path=path, dependencies=dependencies)
    try:
        handle = await backend.enqueue_colony(
            ColonyRunSpec(canonical_colony["colony_id"], 1, "2026-11-01")
        )
        result = await handle.get_result()
        assert set(result.not_eligible) == {
            canonical_colony["center"].id,
            canonical_colony["premise1"].id,
        }
        assert len(starts) == 3
        assert canonical_colony["premise2"].id in result.completed
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_revalidation_resolves_and_deduplicates_the_next_round(
    scheduler_fixture, canonical_colony
):
    directory, path, backend, dependencies, starts, _ = scheduler_fixture
    await backend.start_async(directory, db_path=path, dependencies=dependencies)
    try:
        grow = await backend.enqueue_colony(
            ColonyRunSpec(canonical_colony["colony_id"], 1, "2026-11-01")
        )
        await grow.get_result()
        cell = canonical_colony["cell1"].id
        request = ReopenRequest(cell, "new-evidence-1", "2026-11-01")
        first = await backend.reopen_cell(request)
        second = await backend.reopen_cell(request)
        assert isinstance(await first.get_result(), Validated)
        assert isinstance(await second.get_result(), Validated)
        assert [(name, round_) for name, round_, _ in starts if name == cell] == [
            (cell, 1),
            (cell, 2),
        ]
        runs = await DBOS.list_workflows_async(name="petri.validate_cell")
        assert {run.workflow_id for run in runs if run.workflow_id.startswith(cell)} == {
            f"{cell}:r1",
            f"{cell}:r2",
        }
        events = database.read_events(path, cell_id=cell, event_type="cell_reopened")
        assert len(events) == 1
        assert events[0]["data"]["prior_status"] == "VALIDATED"
        assert database.get_cell(path, canonical_colony["premise1"].id)["status"] == "VALIDATED"
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_dish_filter_and_pure_dry_run(scheduler_fixture, canonical_colony):
    directory, path, backend, dependencies, _, _ = scheduler_fixture
    expected = [canonical_colony["cell1"].id, canonical_colony["cell2"].id]
    assert await backend.grow_dish(directory, dry_run=True) == expected
    assert await backend.grow_dish(directory, dry_run=True, colony_filter="other") == []
    assert not (directory / "dbos.sqlite").exists()
    await backend.start_async(directory, db_path=path, dependencies=dependencies)
    try:
        assert await backend.grow_dish(directory, colony_filter="other") == []
        handles = await backend.grow_dish(directory, colony_filter=canonical_colony["colony_name"])
        assert len(handles) == 1
        assert len((await handles[0].get_result()).completed) == 5
    finally:
        await backend.close_async()

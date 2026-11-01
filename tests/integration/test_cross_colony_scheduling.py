"""External premises participate in eligibility with their actual dish statuses."""

from dataclasses import replace

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.scheduler import ColonyRunSpec
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.edge_registry import EdgeRecord, append
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.mark.integration
@pytest.mark.asyncio
async def test_only_dependent_of_validated_external_premise_runs(tmp_path):
    (tmp_path / "petri.yaml").write_text("name: dish\n")
    path = dish_db_path(tmp_path)
    ready, blocked = "dish-alpha-000-001", "dish-alpha-000-002"
    validated, unresolved = "dish-beta-001-001", "dish-beta-001-002"
    for colony_id, members in [
        ("dish-alpha", [ready, blocked]),
        ("dish-beta", [validated, unresolved]),
    ]:
        graph = ColonyGraph(colony_id)
        for cell_id in members:
            graph.add_cell(
                Cell(
                    id=cell_id,
                    colony_id=colony_id,
                    claim_text="Public fixture claim",
                    level=0,
                    status="VALIDATED" if cell_id == validated else "NEW",
                )
            )
        save_colony(
            path,
            graph,
            Colony(
                id=colony_id,
                dish="dish",
                center_claim="Public fixture claim",
                center_cell_id=members[0],
            ),
        )
    for dependent, premise in [(ready, validated), (blocked, unresolved)]:
        append(
            path,
            EdgeRecord(
                from_cell=dependent,
                to_cell=premise,
                edge_type="cross_colony",
                created_at="2026-11-01",
            ),
        )
    phases = StubPhases()
    model = Agent(TestModel(custom_output_text="Local fixture verdict"))
    starts = []

    async def research(value):
        assert value.cell_id == ready
        assert database.get_cell(path, validated)["status"] == "VALIDATED"
        starts.append(value.cell_id)
        await model.run("Return the local fixture verdict.")
        return await phases.research(value)

    async def finish(value, result):
        with database.write_connection(path) as connection:
            connection.execute(
                "UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id)
            )

    deps = replace(phases.dependencies(), research=research, finish=finish)
    backend = DBOSExecutionBackend(
        config={"execution": {"backend": "dbos", "worker_concurrency": 1}}
    )
    assert [cell.id for cell in backend.eligible_cells(tmp_path, colony_filter="alpha")] == [ready]
    await backend.start_async(tmp_path, db_path=path, dependencies=lambda spec: deps)
    try:
        handle = await backend.enqueue_colony(ColonyRunSpec("dish-alpha", 1, "2026-11-01"))
        result = await handle.get_result()
        assert result.completed == (ready,) and result.not_eligible == (blocked,)
        assert starts == [ready]
        assert database.get_cell(path, blocked)["status"] == "NEW"
        assert database.get_cell(path, unresolved)["status"] == "NEW"
    finally:
        await backend.close_async()

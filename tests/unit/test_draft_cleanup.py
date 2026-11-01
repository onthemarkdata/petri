"""Discard frees draft names without losing history or replaying prior attempts."""

from contextlib import closing

import pytest

from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database, edge_registry
from petri.storage.colonies import clear_unapproved_colony, colony_exists, load_colony, save_colony
from petri.storage.event_log import append_event


def draft(path, name):
    colony_id = f"dish-{name}"
    graph = ColonyGraph(colony_id)
    center = Cell(id=f"{colony_id}-000-000", colony_id=colony_id, claim_text=name, level=0)
    graph.add_cell(center)
    colony = Colony(
        id=colony_id, dish="dish", center_claim=name, center_cell_id=center.id,
        clarifications=[{"question": "Scope?", "answer": "Local"}],
    )
    save_colony(path, graph, colony)
    append_event(database.SQLiteEventLog(path), center.id, "seed_started", "decomposition_lead", 0, {})
    return colony


def rows(path):
    with closing(database.connect(path)) as connection:
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("cells", "colonies", "edges", "events")
        }


def test_discard_releases_only_draft_and_retains_events(tmp_path):
    path = tmp_path / "petri.sqlite"
    discarded = draft(path, "draft")
    unrelated = draft(path, "approved")
    append_event(database.SQLiteEventLog(path), unrelated.center_cell_id, "colony_approved", "human", 0, {})
    before = rows(path)
    clear_unapproved_colony(path, discarded.id)
    after = rows(path)
    assert not colony_exists(path, discarded.id)
    assert colony_exists(path, unrelated.id)
    assert after["events"] == before["events"]
    assert after["cells"] == [row for row in before["cells"] if row["colony"] == unrelated.id]
    assert after["colonies"] == [row for row in before["colonies"] if row["id"] == unrelated.id]
    assert load_colony(path, unrelated.id)[1].clarifications == unrelated.clarifications


@pytest.mark.parametrize("guard", ["approved", "external_reference"])
def test_discard_preserves_protected_colonies(tmp_path, guard):
    path = tmp_path / "petri.sqlite"
    target = draft(path, "draft")
    if guard == "approved":
        append_event(database.SQLiteEventLog(path), target.center_cell_id, "colony_approved", "human", 0, {})
    else:
        other = draft(path, "other")
        edge_registry.append(path, edge_registry.EdgeRecord(
            from_cell=other.center_cell_id, to_cell=target.center_cell_id,
            edge_type="cross_colony", created_at="2026-11-01T00:00:00Z",
        ))
    before = rows(path)
    with pytest.raises(ValueError):
        clear_unapproved_colony(path, target.id)
    assert rows(path) == before


def test_discard_joins_transaction_and_replacement_retains_metadata(tmp_path):
    path = tmp_path / "petri.sqlite"
    target = draft(path, "draft")
    before = rows(path)
    with pytest.raises(RuntimeError, match="Interrupted discard"):
        with database.write_connection(path) as connection:
            clear_unapproved_colony(path, target.id, connection=connection)
            assert connection.execute("SELECT 1 FROM colonies WHERE id=?", (target.id,)).fetchone() is None
            raise RuntimeError("Interrupted discard")
    assert rows(path) == before
    clear_unapproved_colony(path, target.id, release_reservation=False)
    assert colony_exists(path, target.id)
    assert rows(path)["colonies"] == before["colonies"]


def test_attempt_lookup_reads_explicit_identity_without_creating_missing_database(tmp_path):
    from petri.reasoning.seed_events import current_attempt

    path = tmp_path / "missing/petri.sqlite"
    center = "dish-units-000-000"
    assert current_attempt(path, center) == 0
    assert not path.parent.exists()
    append_event(database.SQLiteEventLog(path), center, "decomposition_started", "decomposition_lead", 0, {})
    append_event(database.SQLiteEventLog(path), center, "decomposition_started", "decomposition_lead", 0,
                 {"attempt": 5}, sequence=1)
    assert current_attempt(path, center) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_reservation", [False, True])
async def test_dashboard_discard_then_new_claim_uses_current_attempt_and_keeps_old_status(
    tmp_path, failed_reservation
):
    from petri.cli.init import create_petri_dish
    from petri.execution.dbos_backend import DBOSExecutionBackend
    from petri.execution.run_controls import SeedRequest
    from petri.execution.seed_workflow import SeedRunSpec
    from petri.reasoning.decomposition_schemas import InitialDecomposition
    from petri.reasoning.seed_events import current_attempt
    from petri.reasoning.decomposition_settings import DecompositionSettings
    from petri.storage.paths import dish_db_path
    from tests.pipeline_stubs import StubPhases
    from tests.unit.test_durable_seed import SeedRunner
    from dbos import DBOS

    create_petri_dish(tmp_path, dish_name="dish")
    (tmp_path / "defaults/petri.yaml").write_text(
        "name: dish\nmax_decomposition_depth: 1\ntelemetry:\n  enabled: false\n"
    )
    path = dish_db_path(tmp_path, "dish")
    calls = []

    class Runner(SeedRunner):
        def initial_decomposition(self, request, on_progress=None):
            calls.append(request.claim)
            return InitialDecomposition.model_validate({"nodes": [
                {"claim_text": request.claim + " meter premise"},
                {"claim_text": request.claim + " second premise", "relation": "limits"},
            ]})

    backend = DBOSExecutionBackend()
    await backend.start_async(
        tmp_path, db_path=path, dependencies=lambda spec: StubPhases().dependencies(),
        seed_runner_factory=lambda spec: Runner(),
    )
    try:
        first = await backend.submit_seed(SeedRequest(claim="Original units", colony="units"))
        await (await DBOS.retrieve_workflow_async(first.id)).get_result()
        old = await backend.get_run(tmp_path, first.id)
        history = {event["id"]: event for event in database.read_events(path)}
        clear_unapproved_colony(path, "dish-units")
        if failed_reservation:
            failed = await backend.enqueue_seed(SeedRunSpec(
                "dish-units", "dish", "units", DecompositionSettings.from_dish(tmp_path), 2
            ))
            with pytest.raises(KeyError, match="Colony not found"):
                await failed.get_result()
        second = await backend.submit_seed(SeedRequest(claim="Replacement units", colony="units"))
        attempt = 3 if failed_reservation else 2
        assert second.id == f"dish-units:seed-attempt{attempt}" and second.id != first.id
        await (await DBOS.retrieve_workflow_async(second.id)).get_result()
        current = await backend.get_run(tmp_path, second.id)
        assert current.status == "SUCCESS" and current.result["complete"]
        repeated = await backend.submit_seed(SeedRequest(claim="Replacement units", colony="units"))
        assert repeated.id == second.id
        assert (await backend.get_run(tmp_path, first.id)).model_dump() == old.model_dump()
        assert calls == ["Original units", "Replacement units"]
        assert current_attempt(path, "dish-units-000-000") == attempt
        graph, colony = load_colony(path, "dish-units")
        assert colony.center_claim == "Replacement units"
        assert all(cell.claim_text.startswith("Replacement units") for cell in graph.get_all_cells())
        events = {event["id"]: event for event in database.read_events(path)}
        assert all(events[key] == event for key, event in history.items())
    finally:
        await backend.close_async()

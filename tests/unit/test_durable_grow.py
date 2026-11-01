"""Grow polls backend state and can admit approved work after a settled pass."""

from dataclasses import replace

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.storage import database
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


def test_grow_fixture_and_after_pass_add_only_new_work(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    directory = env["petri_dir"]
    path = dish_db_path(directory)
    phases = StubPhases()
    signals = []
    callbacks = []
    new_id = f"{env['colony_id']}-002-099"

    async def finish(value, result):
        with database._write(path) as conn:
            conn.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))

    async def after_pass(petri_dir):
        assert petri_dir == directory
        assert all(row["status"] == "VALIDATED" for row in database.list_cells(path))
        callbacks.append("settled")
        if len(callbacks) == 1:
            row = database.get_cell(path, env["cell1"].id)
            database.put_cell(path, {**row, "id": new_id, "seq": 99, "status": "NEW"})
            return True
        return False

    backend = DBOSExecutionBackend(
        config={"execution": {"backend": "dbos", "worker_concurrency": 2}},
        after_pass=after_pass,
    )
    backend.status_interval = 0.01
    original_start = backend.start
    deps = replace(phases.dependencies(), finish=finish)
    monkeypatch.setattr(backend, "start", lambda directory, **options: original_start(
        directory, **options, dependencies=lambda _: deps
    ))
    result = backend.process_queue(petri_dir=directory, on_event=signals.append)
    assert result.processed == 6
    assert callbacks == ["settled", "settled"]
    assert len([name for name, _ in phases.calls if name == "socratic"]) == 6
    assert any(event.kind == "started" for event in signals)
    assert any(event.kind == "phase" for event in signals)
    assert all(row["status"] == "VALIDATED" for row in database.list_cells(path))
    assert not (directory / "queue.json").exists()

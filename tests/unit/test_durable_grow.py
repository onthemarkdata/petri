"""Grow polls backend state and can admit approved work after a settled pass."""

import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

from dbos import DBOS
from dbos._error import DBOSAwaitedWorkflowCancelledError
from filelock import FileLock, Timeout
import pytest

from petri.execution.control import cancellation_mode, record_cancellation
from petri.execution.dbos_app import DBOSApp, QUEUE_NAME
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


@pytest.mark.parametrize("old_parent", [False, True])
def test_graceful_stop_waits_for_child_publication_and_retains_owner(
    seeded_petri_dir, monkeypatch, old_parent
):
    env = seeded_petri_dir
    directory = env["petri_dir"]
    path = dish_db_path(directory)
    backend = DBOSExecutionBackend()
    app = DBOSApp(directory, path, backend.settings, lambda _: StubPhases().dependencies())
    backend.app = app
    parent_id = "colony:grow:r1"
    children = ["cell-a:r1", "cell-b:r1"]
    rows = {
        parent_id: SimpleNamespace(workflow_id=parent_id, parent_workflow_id=None, status="PENDING"),
        "intermediate": SimpleNamespace(
            workflow_id="intermediate", parent_workflow_id=parent_id, status="SUCCESS"
        ),
        "queued": SimpleNamespace(
            workflow_id="queued", parent_workflow_id=parent_id, status="ENQUEUED"
        ),
    }
    for child in children:
        rows[child] = SimpleNamespace(
            workflow_id=child,
            parent_workflow_id="earlier-grow" if old_parent else "intermediate",
            status="PENDING",
        )
    if old_parent:
        record_cancellation(directory / "dbos.sqlite", children, force=False)
    entered = [threading.Event() for _ in children]
    publications = [threading.Event() for _ in children]
    finish_phases = threading.Event()
    parent_cancelled = threading.Event()
    publication_gap = threading.Event()
    one_pending = threading.Event()
    close_attempted = threading.Event()
    closed = threading.Event()
    calls = []
    outcome = []
    errors = []

    def start(*args, **kwargs):
        app._owner.acquire(timeout=0)
        app.started = True
        return app

    async def child_step(index):
        with app.step_activity():
            entered[index].set()
            assert await asyncio.to_thread(finish_phases.wait, 5)
            calls.append((children[index], "research_finished"))
        assert await asyncio.to_thread(publications[index].wait, 5)
        rows[children[index]].status = "CANCELLED"

    class ParentHandle:
        workflow_id = parent_id

        async def get_result(self):
            for event in entered:
                assert await asyncio.to_thread(event.wait, 5)
            rows[parent_id].status = "CANCELLED"
            rows["queued"].status = "CANCELLED"
            parent_cancelled.set()
            raise DBOSAwaitedWorkflowCancelledError("Stopped at a step boundary")

    tasks = []

    async def grow_dish(*args, **kwargs):
        tasks.extend(asyncio.create_task(child_step(index)) for index in range(2))
        return [ParentHandle()]

    async def list_runs(**kwargs):
        assert kwargs == {"load_input": False, "load_output": False}
        pending = sum(rows[child].status == "PENDING" for child in children)
        if parent_cancelled.is_set() and app._active_steps == 0:
            publication_gap.set()
            if pending == 1:
                one_pending.set()
        return list(rows.values())

    def pending_runs(**kwargs):
        return [row for row in rows.values() if row.status in kwargs["status"]][:kwargs["limit"]]

    original_close = app.close

    def close():
        close_attempted.set()
        return original_close()

    # Control status publication while retaining the real close guard and owner lock.
    monkeypatch.setattr(app, "close", close)
    monkeypatch.setattr(backend, "start", start)
    monkeypatch.setattr(backend, "_resume_cancelled_cells", AsyncMock(
        return_value=set(children) if old_parent else set()
    ))
    monkeypatch.setattr(backend, "grow_dish", grow_dish)
    monkeypatch.setattr(DBOS, "list_workflows_async", list_runs)
    monkeypatch.setattr(DBOS, "list_workflows", pending_runs)
    monkeypatch.setattr(DBOS, "destroy", lambda **_: closed.set())

    def grow():
        try:
            outcome.append(backend.grow_loop(
                run_one_pass=lambda: backend.process_queue(petri_dir=directory),
                get_states=lambda: backend.state_summary(directory),
                is_stopped=lambda: backend.is_stopped(directory),
            ))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=grow)
    worker.start()
    owner = FileLock(directory / ".execution-owner.lock")
    try:
        assert parent_cancelled.wait(5)
        assert app._active_steps == 2
        assert not close_attempted.is_set()
        assert not closed.is_set()
        finish_phases.set()
        assert publication_gap.wait(5)
        assert app._active_steps == 0
        assert all(rows[child].status == "PENDING" for child in children)
        with pytest.raises(Timeout):
            owner.acquire(timeout=0)
        assert not close_attempted.is_set()
        assert not closed.is_set()
        publications[0].set()
        assert one_pending.wait(5)
        with pytest.raises(Timeout):
            owner.acquire(timeout=0)
        assert not close_attempted.is_set()
        assert not closed.is_set()
        publications[1].set()
        worker.join(5)
        assert not worker.is_alive()
        assert not errors
        assert outcome[0].reason == "user_stop"
        assert outcome[0].final_states["queued"] == 5
        assert sorted(calls) == [(child, "research_finished") for child in children]
        assert close_attempted.is_set()
        assert closed.is_set()
        with owner.acquire(timeout=0):
            pass
    finally:
        finish_phases.set()
        for event in publications:
            event.set()
        worker.join(10)
        app._owner.release()


def test_stopped_workflow_drain_preserves_close_guard(tmp_path, monkeypatch):
    record_cancellation(tmp_path / "dbos.sqlite", ["unrelated"], force=False)
    backend = DBOSExecutionBackend()
    app = DBOSApp(tmp_path, tmp_path / "domain.sqlite", backend.settings, lambda _: None)
    backend.app = app
    app._owner.acquire(timeout=0)
    app.started = True
    rows = [
        SimpleNamespace(workflow_id="owned", parent_workflow_id=None, status="CANCELLED"),
        SimpleNamespace(workflow_id="unrelated", parent_workflow_id=None, status="PENDING"),
    ]
    monkeypatch.setattr(DBOS, "list_workflows_async", AsyncMock(return_value=rows))
    monkeypatch.setattr(DBOS, "list_workflows", lambda **_: [rows[1]])
    try:
        asyncio.run(asyncio.wait_for(backend._wait_for_stopped_workflows({"owned"}), 2))
        with pytest.raises(RuntimeError, match="Wait for or cancel active workflows"):
            backend.close()
        assert app.started
        with pytest.raises(Timeout):
            FileLock(tmp_path / ".execution-owner.lock").acquire(timeout=0)
    finally:
        app._owner.release()


def test_scoped_resume_reports_only_selected_resumed_workflows(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    backend = DBOSExecutionBackend()
    backend.app = SimpleNamespace(
        started=True, petri_dir=env["petri_dir"], db_path=dish_db_path(env["petri_dir"]),
        conflicts={},
    )
    selected = f"{env['cell1'].id}:r1"
    outside = f"{env['cell2'].id}:r1"
    record_cancellation(env["petri_dir"] / "dbos.sqlite", [selected, outside], force=False)
    monkeypatch.setattr("petri.execution.scheduler.current_round", lambda *_: 1)
    monkeypatch.setattr(DBOS, "list_workflows_async", AsyncMock(return_value=[
        SimpleNamespace(workflow_id=selected, parent_workflow_id="old-scope"),
        SimpleNamespace(workflow_id=outside, parent_workflow_id="unselected-scope"),
    ]))
    resume = AsyncMock()
    monkeypatch.setattr(DBOS, "resume_workflow_async", resume)
    resumed = asyncio.run(backend._resume_cancelled_cells(cell_ids=[env["cell1"].id]))
    assert resumed == {selected}
    resume.assert_awaited_once_with(selected, queue_name=QUEUE_NAME)
    assert cancellation_mode(env["petri_dir"] / "dbos.sqlite", selected) is None
    assert cancellation_mode(env["petri_dir"] / "dbos.sqlite", outside) == "graceful"

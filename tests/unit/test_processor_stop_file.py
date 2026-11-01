"""Durable cancellation preserves completed steps and distinguishes force stop."""

import asyncio
import threading
from dataclasses import replace

import pytest
from dbos import DBOS
from dbos._error import DBOSAwaitedWorkflowCancelledError

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.execution.pipeline import Validated
from tests.pipeline_stubs import StubPhases


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_cancel_and_resume_at_step_boundary(tmp_path, force):
    phases = StubPhases()
    entered = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    calls = []

    async def research(value):
        calls.append("started")
        entered.set()
        try:
            while not released.is_set():
                await asyncio.sleep(0.01)
            calls.append("completed")
            return await phases.research(value)
        finally:
            finished.set()

    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    deps = replace(phases.dependencies(), research=research)
    await backend.start_async(tmp_path, db_path=tmp_path / "domain.sqlite", dependencies=lambda _: deps)
    spec = CellRunSpec("dish-colony-002-001", 1, "2026-11-01")
    try:
        handle = await backend.enqueue_cell(spec)
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        assert await asyncio.to_thread(backend.request_stop, tmp_path, force=force) == 1
        assert (await DBOS.get_workflow_status_async(spec.workflow_id)).status == (
            "CANCELLED" if force else "PENDING"
        )
        if force:
            await asyncio.wait_for(asyncio.to_thread(finished.wait), 3)
            assert calls == ["started"]
        else:
            await asyncio.sleep(0.15)
            assert not finished.is_set()
        released.set()
        await asyncio.wait_for(asyncio.to_thread(finished.wait), 3)
        with pytest.raises(DBOSAwaitedWorkflowCancelledError):
            await handle.get_result()
        # Wait until the old execution releases its slot before resuming it.
        while backend.app._active_steps:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        backend.clear_stop(tmp_path)
        resumed = await backend.enqueue_cell(spec)
        assert isinstance(await resumed.get_result(), Validated)
        assert [name for name, _ in phases.calls].count("socratic") == 1
        assert calls.count("completed") == 1
        assert calls.count("started") == (2 if force else 1)
        assert not (tmp_path / ".stop").exists()
    finally:
        released.set()
        await backend.close_async()


def test_stop_without_a_runtime_is_a_noop(tmp_path):
    backend = DBOSExecutionBackend()
    assert backend.request_stop(tmp_path) == 0
    assert backend.request_stop(tmp_path, force=True) == 0
    assert list(tmp_path.iterdir()) == []

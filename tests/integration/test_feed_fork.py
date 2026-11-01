"""Offline evidence forks against the real SQLite DBOS system database."""

import asyncio
from dataclasses import replace
import hashlib

from dbos import DBOS
import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.execution.models import FeedRequest
from petri.execution.pipeline import Validated
from petri.lifecycle.revalidate import research_boundary
from petri.storage import database
from tests.unit import test_research_phases

phase_services = test_research_phases.phase_services


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_feed_copies_socratic_and_repeats_research_once(
    phase_services, canonical_colony, tmp_path, monkeypatch, legacy
):
    services, signals = phase_services
    from petri.storage.colonies import save_colony
    from petri.storage.paths import dish_db_path

    directory = services.db_path.parent
    (directory / "petri.yaml").write_text("name: test-dish\n")
    services = replace(services, db_path=dish_db_path(directory))
    save_colony(services.db_path, canonical_colony["graph"], canonical_colony["colony_model"])
    research_messages = []

    class RecordingTestModel(TestModel):
        async def request(self, messages, model_settings, model_request_parameters):
            research_messages.append(str(messages))
            return await super().request(messages, model_settings, model_request_parameters)

    for name, agent in services.agents.items():
        agent.model = RecordingTestModel(
            custom_output_args={
                "agent": name.value,
                "verdict": services.config.agents[name].verdicts_pass[0],
                "sources_cited": [test_research_phases.SOURCE],
            }
        )
    rounds = {}

    def factory(spec):
        if spec.workflow_id not in rounds:
            rounds[spec.workflow_id] = replace(
                services, round_number=spec.round_number, today_iso=spec.today_iso, usage=RunUsage()
            )
        return rounds[spec.workflow_id]

    backend = DBOSExecutionBackend(services_factory=factory)
    await backend.start_async(directory, db_path=services.db_path)
    try:
        cell_id = canonical_colony["cell1"].id
        original = await backend.enqueue_cell(CellRunSpec(cell_id, 1, "2026-11-01"))
        assert isinstance(await original.get_result(), Validated)
        steps = await DBOS.list_workflow_steps_async(original.workflow_id)
        boundary = research_boundary(steps)
        assert boundary is not None
        source = tmp_path / "evidence.txt"
        source.write_text("A new measurement supports the claim.")
        if legacy:
            import petri.lifecycle.revalidate as module

            monkeypatch.setattr(module, "research_boundary", lambda steps: None)
        requests = [
            FeedRequest(
                cell_id,
                hashlib.sha256(identity.encode()).hexdigest(),
                str(source),
                "2026-11-02",
                max_iterations=4,
            )
            for identity in ("first", "duplicate")
        ]
        handles = await asyncio.gather(*(backend.feed_evidence(request) for request in requests))
        assert all(
            isinstance(result, Validated)
            for result in await asyncio.gather(*(handle.get_result() for handle in handles))
        )
        runs = await DBOS.list_workflows_async(name="petri.validate_cell")
        assert sorted(run.workflow_id for run in runs) == [f"{cell_id}:r1", f"{cell_id}:r2"]
        assert rounds[f"{cell_id}:r2"].today_iso == "2026-11-02"
        fork_steps = await DBOS.list_workflow_steps_async(f"{cell_id}:r2")
        if not legacy:
            assert [step["output"] for step in fork_steps if step["function_id"] < boundary] == [
                step["output"] for step in steps if step["function_id"] < boundary
            ]
        assert sum(
            event.kind == "phase_start" and event.phase == "socratic" for event in signals
        ) == (2 if legacy else 1)
        assert (
            sum(event.kind == "phase_start" and event.phase == "research" for event in signals) == 2
        )
        events = database.read_events(services.db_path, cell_id=cell_id)
        fed = [event for event in events if event["type"] == "evidence_fed"]
        assert len(fed) == 1
        assert fed[0]["data"]["content"] == "A new measurement supports the claim."
        assert any(
            "A new measurement supports the claim." in messages for messages in research_messages
        )
        assert fed[0]["data"]["original_workflow_id"] == original.workflow_id
        assert fed[0]["data"]["forked_workflow_id"] == f"{cell_id}:r2"
        assert fed[0]["data"]["generation"] == 2
        assert len(list((services.db_path.parent / "evidence").glob("*.json"))) == 1
        propagated = database.read_events(services.db_path, event_type="propagation_triggered")
        expected_dependents = set(canonical_colony["graph"].get_dependents(cell_id))
        assert expected_dependents <= {event["cell_id"] for event in propagated}
        assert all(
            database.get_cell(services.db_path, event["cell_id"])["status"] == "NEW"
            for event in propagated
        )
        assert all(run.workflow_id.startswith(cell_id + ":r") for run in runs)
        from petri.execution.phases import _prior_evidence
        from petri.execution.pipeline import PipelineInput
        from petri.execution.models import CellPipelineState

        document = services.db_path.parent / fed[0]["data"]["evidence_path"]
        document.unlink()
        assert "A new measurement supports the claim." in _prior_evidence(
            services, PipelineInput.model_validate(CellPipelineState(cell_id=cell_id).model_dump())
        )
        assert document.is_file()
    finally:
        await backend.close_async()


@pytest.mark.integration
@pytest.mark.parametrize("point", ["ingest", "fork"])
def test_feed_recovers_sigkill_without_duplicate_files_or_forks(tmp_path, point):
    import json
    import os
    from pathlib import Path
    import signal
    import sqlite3
    import subprocess
    import sys
    import time
    from petri.storage.paths import dish_db_path

    root = Path(__file__).parents[2]
    command = [
        sys.executable,
        str(root / "tests/helpers/dbos_feed_worker.py"),
        str(tmp_path),
        point,
    ]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "interrupted").exists():
                assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            os.kill(worker.pid, signal.SIGKILL)
            assert worker.wait(timeout=5) == -signal.SIGKILL
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "validated"
    path = dish_db_path(tmp_path)
    assert len(list((path.parent / "evidence").glob("*.json"))) == 1
    fed = database.read_events(path, event_type="evidence_fed")
    assert len(fed) == 1 and fed[0]["data"]["generation"] == 2
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls WHERE phase='socratic'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM calls WHERE phase='research'").fetchone()[0] == 2
    with sqlite3.connect(tmp_path / "dbos.sqlite") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM workflow_status WHERE name='petri.validate_cell'"
            ).fetchone()[0]
            == 2
        )


@pytest.mark.parametrize(
    "phases",
    [
        ["sync_check", "socratic", "research", "critique", "research"],
        ["sync_check", "socratic", "socratic", "sync_check", "research", "research"],
        ["sync_check", "socratic", "convergence"],
    ],
)
def test_research_boundary_uses_stored_ids_for_each_shape(phases):
    from petri.execution.dbos_backend import StepResult

    steps = [
        {
            "function_id": 2 * index + 3,
            "function_name": "petri.cell_phase",
            "output": StepResult(phase=phase),
        }
        for index, phase in enumerate(phases)
    ]
    expected = next(
        (step["function_id"] for step in steps if step["output"].phase == "research"), None
    )
    assert research_boundary(steps) == expected
    assert (
        research_boundary(
            [{"function_id": 9, "function_name": "petri.cell_phase", "output": StepResult()}]
        )
        is None
    )


@pytest.mark.asyncio
async def test_failed_checkpointed_ingest_writes_no_evidence(
    tmp_path, canonical_colony, monkeypatch
):
    import httpx
    from petri.reasoning.ingest import IngestError
    from petri.storage.colonies import save_colony
    from petri.storage.paths import dish_db_path
    from tests.pipeline_stubs import StubPhases

    (tmp_path / "petri.yaml").write_text("name: test-dish\ntelemetry:\n  enabled: false\n")
    path = dish_db_path(tmp_path)
    save_colony(path, canonical_colony["graph"], canonical_colony["colony_model"])
    cell_id = canonical_colony["cell1"].id
    with database._write(path) as conn:
        conn.execute("UPDATE cells SET status='VALIDATED' WHERE id=?", (cell_id,))

    def fail(*args, **options):
        raise httpx.ConnectError("Offline URL failure")

    monkeypatch.setattr(httpx, "get", fail)
    backend = DBOSExecutionBackend()
    await backend.start_async(
        tmp_path, db_path=path, dependencies=lambda _: StubPhases().dependencies()
    )
    try:
        for index, source in enumerate(
            ("https://example.invalid/evidence", str(tmp_path / "missing.txt"))
        ):
            request = FeedRequest(
                cell_id, hashlib.sha256(str(index).encode()).hexdigest(), source, "2026-11-01"
            )
            handle = await backend.feed_evidence(request)
            with pytest.raises(IngestError) as caught:
                await handle.get_result()
            assert caught.value.source_kind == ("url" if index == 0 else "file")
            steps = await DBOS.list_workflow_steps_async(request.workflow_id)
            assert steps[0]["function_name"] == "petri.ingest_evidence"
            assert steps[0]["error"] is not None
        assert not (path.parent / "evidence").exists()
        assert database.get_cell(path, cell_id)["status"] == "VALIDATED"
        assert database.read_events(path, event_type="evidence_fed") == []
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_large_evidence_stays_out_of_checkpoints_and_fork_obeys_admission(
    tmp_path, canonical_colony
):
    import pickle
    import time
    from petri.execution.dbos_app import CHECKPOINT_LIMIT
    from petri.storage.colonies import save_colony
    from petri.storage.paths import dish_db_path
    from tests.pipeline_stubs import StubPhases

    (tmp_path / "petri.yaml").write_text("name: test-dish\ntelemetry:\n  enabled: false\n")
    path = dish_db_path(tmp_path)
    save_colony(path, canonical_colony["graph"], canonical_colony["colony_model"])
    starts = []
    phases = StubPhases()

    async def research(value):
        starts.append(time.monotonic())
        return await phases.research(value)

    async def finish(value, result):
        with database._write(path) as conn:
            conn.execute(
                "UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id)
            )

    deps = replace(phases.dependencies(), research=research, finish=finish)
    backend = DBOSExecutionBackend(config={"execution": {"limiter": {"limit": 1, "period": 1.2}}})
    await backend.start_async(tmp_path, db_path=path, dependencies=lambda _: deps)
    try:
        cell_id = canonical_colony["cell1"].id
        original = await backend.enqueue_cell(CellRunSpec(cell_id, 1, "2026-11-01"))
        await original.get_result()
        source = tmp_path / "large.txt"
        source.write_text("Measured evidence.\n" * 160_000)
        request = FeedRequest(
            cell_id, hashlib.sha256(b"large").hexdigest(), str(source), "2026-11-01"
        )
        handle = await backend.feed_evidence(request)
        assert isinstance(await handle.get_result(), Validated)
        steps = await DBOS.list_workflow_steps_async(request.workflow_id)
        assert all(len(pickle.dumps(step["output"])) < CHECKPOINT_LIMIT for step in steps)
        assert (
            len(database.read_events(path, event_type="evidence_fed")[0]["data"]["content"])
            > CHECKPOINT_LIMIT
        )
        assert starts[1] - starts[0] >= 1.1
    finally:
        await backend.close_async()

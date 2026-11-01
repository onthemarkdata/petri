"""Offline durability, typed failure, and admission tests against DBOS SQLite."""

import asyncio
import pickle
import time
from dataclasses import replace

import pytest
from dbos import DBOS
from pydantic_ai.models.test import TestModel

from petri.execution.config import ExecutionConfig
from petri.execution.dbos_app import CHECKPOINT_LIMIT, DBOSApp, ExecutionOwnerError
from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend, _bounded
from petri.execution.pipeline import NeedsHuman, Validated
from petri.harness.errors import PermanentHarnessError, RateLimitedError
from petri.storage import database
from tests.pipeline_stubs import StubPhases
from tests.unit import test_research_phases

phase_services = test_research_phases.phase_services


@pytest.mark.asyncio
async def test_diamond_cell_round_is_deduplicated_and_small(phase_services, canonical_colony):
    services, signals = phase_services
    for name, agent in services.agents.items():
        agent.model = TestModel(
            custom_output_args={
                "agent": name.value,
                "verdict": services.config.agents[name].verdicts_pass[0],
                "sources_cited": [test_research_phases.SOURCE],
            }
        )
    backend = DBOSExecutionBackend(
        config={"execution": {"backend": "dbos", "worker_concurrency": 2}},
        services_factory=lambda spec: replace(services, round_number=spec.round_number),
    )
    await backend.start_async(services.db_path.parent, db_path=services.db_path)
    try:
        spec = CellRunSpec(canonical_colony["cell1"].id, 1, "2026-11-01")
        first = await backend.enqueue_cell(spec, level=2)
        second = await backend.enqueue_cell(spec, level=2)
        assert first.workflow_id == second.workflow_id == spec.workflow_id
        assert isinstance(await first.get_result(), Validated)
        assert isinstance(await second.get_result(), Validated)
        before = database.read_events(services.db_path)
        assert before
        again = await backend.enqueue_cell(spec)
        assert isinstance(await again.get_result(), Validated)
        assert database.read_events(services.db_path) == before
        assert database.get_cell(services.db_path, spec.cell_id)["status"] == "VALIDATED"
        runs = await DBOS.list_workflows_async(workflow_ids=[spec.workflow_id])
        assert len(runs) == 1
        assert runs[0].priority == 3
        steps = await DBOS.list_workflow_steps_async(spec.workflow_id)
        assert {step["function_name"] for step in steps} >= {
            "petri.admit_cell",
            "petri.cell_phase",
            "petri.finish_cell",
        }
        assert all(len(pickle.dumps(step["output"])) < CHECKPOINT_LIMIT for step in steps)
        assert services.db_path.name == "petri.sqlite"
        assert (services.db_path.parent / "dbos.sqlite").is_file()
        assert len([signal for signal in signals if signal.kind == "phase_start"]) == 6
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_rate_limit_sleeps_without_retrying_completed_steps(tmp_path):
    phases = StubPhases()
    calls = []

    async def research(value):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RateLimitedError("Wait for reset", retry_after_seconds=0.12)
        return await phases.research(value)

    deps = replace(phases.dependencies(), research=research)
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    await backend.start_async(
        tmp_path, db_path=tmp_path / "petri.sqlite", dependencies=lambda spec: deps
    )
    try:
        spec = CellRunSpec("dish-colony-002-001", 1, "2026-11-01")
        handle = await backend.enqueue_cell(spec)
        assert isinstance(await handle.get_result(), Validated)
        assert calls[1] - calls[0] >= 0.11
        assert [name for name, _ in phases.calls].count("socratic") == 1
        assert (await DBOS.get_workflow_status_async(spec.workflow_id)).status == "SUCCESS"
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_expected_failure_parks_the_round_as_success(tmp_path):
    phases = StubPhases()
    endings = []

    async def research(value):
        raise PermanentHarnessError("Repair the model configuration")

    async def finish(value, result):
        endings.append(result)

    deps = replace(phases.dependencies(), research=research, finish=finish)
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    await backend.start_async(
        tmp_path, db_path=tmp_path / "petri.sqlite", dependencies=lambda spec: deps
    )
    try:
        spec = CellRunSpec("dish-colony-002-001", 1, "2026-11-01")
        handle = await backend.enqueue_cell(spec)
        result = await handle.get_result()
        assert isinstance(result, NeedsHuman)
        assert result.reason == "Repair the model configuration"
        assert endings == [result]
        assert (await DBOS.get_workflow_status_async(spec.workflow_id)).status == "SUCCESS"
    finally:
        await backend.close_async()


@pytest.mark.asyncio
async def test_owner_lock_excludes_another_runtime_and_queue_limits_starts(tmp_path):
    active = 0
    peak = 0
    starts = []
    phases = StubPhases()

    async def research(value):
        nonlocal active, peak
        starts.append(time.monotonic())
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.16)
            return await phases.research(value)
        finally:
            active -= 1

    config = {
        "execution": {
            "backend": "dbos",
            "worker_concurrency": 2,
            "limiter": {"limit": 2, "period": 0.3},
        }
    }
    deps = replace(phases.dependencies(), research=research)
    backend = DBOSExecutionBackend(config=config)
    await backend.start_async(
        tmp_path, db_path=tmp_path / "petri.sqlite", dependencies=lambda spec: deps
    )
    try:
        other = DBOSApp(tmp_path, tmp_path / "petri.sqlite", backend.settings, lambda spec: deps)
        with pytest.raises(ExecutionOwnerError):
            await asyncio.to_thread(other.start)
        handles = [
            await backend.enqueue_cell(
                CellRunSpec(f"dish-colony-002-{index:03d}", 1, "2026-11-01"),
                level=2,
            )
            for index in range(5)
        ]
        await asyncio.gather(*(handle.get_result() for handle in handles))
        assert peak <= 2
        assert len(starts) == 5
        assert starts[2] - starts[0] >= 0.28
        assert starts[4] - starts[2] >= 0.28
        assert "2 concurrent cells" in backend.preflight().message
        assert "2 starts/0.3s" in backend.preflight().message
    finally:
        await backend.close_async()


def test_settings_and_checkpoint_bound():
    settings = ExecutionConfig.model_validate(
        {
            "backend": "dbos",
            "worker_concurrency": 2,
            "limiter": {"limit": 3, "period": 1},
        }
    )
    assert ExecutionConfig.model_validate(settings.model_dump()) == settings
    with pytest.raises(ValueError, match="2 MiB"):
        _bounded("x" * CHECKPOINT_LIMIT)


def test_cli_reports_dbos_resources_and_dry_run_has_no_workflows(
    tmp_path,
    canonical_colony,
    monkeypatch,
):
    import importlib
    import typer
    from typer.testing import CliRunner
    from petri.execution.backend import ExecutionBackend
    from petri.storage.colonies import save_colony
    from petri.storage.paths import dish_db_path

    (tmp_path / "petri.yaml").write_text(
        "name: test-dish\nexecution:\n  backend: dbos\n  worker_concurrency: 2\n"
        "  limiter:\n    limit: 3\n    period: 1\n"
    )
    save_colony(dish_db_path(tmp_path), canonical_colony["graph"], canonical_colony["colony_model"])
    module = importlib.import_module("petri.cli.grow")
    monkeypatch.setattr(module, "find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "resolve_provider", lambda path: None)
    app = typer.Typer()
    module.register(app)
    result = CliRunner().invoke(app, ["--dry-run"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "2 concurrent cells" in result.output
    assert "3 starts/1s" in result.output
    assert "execution.worker_concurrency" in result.output
    assert not (tmp_path / "dbos.sqlite").exists()
    assert not (tmp_path / ".execution-owner.lock").exists()
    assert isinstance(DBOSExecutionBackend(), ExecutionBackend)


def test_application_admission_preserves_the_rate_window_after_restart(tmp_path, monkeypatch):
    import petri.execution.dbos_app as app_module

    settings = ExecutionConfig.model_validate(
        {
            "backend": "dbos",
            "worker_concurrency": 2,
            "limiter": {"limit": 2, "period": 1},
        }
    )
    clock = [100.0]
    monkeypatch.setattr(app_module.time, "time", lambda: clock[0])
    first = DBOSApp(
        tmp_path, tmp_path / "petri.sqlite", settings, lambda spec: StubPhases().dependencies()
    )
    first._prepare_admission()
    assert first.admission_delay("one") == 0
    assert first.admission_delay("one") == 0
    assert first.admission_delay("two") == 0
    assert first.admission_delay("three") == 1
    second = DBOSApp(
        tmp_path, tmp_path / "petri.sqlite", settings, lambda spec: StubPhases().dependencies()
    )
    clock[0] = 100.75
    assert second.admission_delay("three") == 0.25
    clock[0] = 101.0
    assert second.admission_delay("three") == 0

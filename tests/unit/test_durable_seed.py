"""Typed seed steps keep model outputs in SQLite and small progress in DBOS."""

import asyncio
import pickle
import time

import pytest
from dbos import DBOS

from petri.execution.dbos_app import CHECKPOINT_LIMIT
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.seed_workflow import SeedRunSpec
from petri.graph.colony import ColonyGraph
from petri.harness.errors import RateLimitedError
from petri.models import Cell, Colony
from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database
from petri.storage.colonies import save_colony
from tests.pipeline_stubs import StubPhases


class SeedRunner:
    def __init__(self, rate_limit=False):
        self.calls = []
        self.rate_limit = rate_limit

    def initial_decomposition(self, request, on_progress=None):
        self.calls.append(("initial", time.monotonic()))
        return InitialDecomposition.model_validate(
            {
                "nodes": [
                    {"claim_text": "The meter is a defined unit"},
                    {"claim_text": "The second is a defined unit", "relation": "limits"},
                ]
            }
        )

    def expand_why(self, request, on_progress=None):
        self.calls.append((request.claim, time.monotonic()))
        if self.rate_limit:
            self.rate_limit = False
            raise RateLimitedError("Local reset", retry_after_seconds=0.1)
        return WhyExpansion.model_validate(
            {"is_atomic": True, "bedrock_reason": "A unit definition", "sub_premises": []}
        )


@pytest.fixture
def seed_fixture(tmp_path):
    path = tmp_path / "petri.sqlite"
    graph = ColonyGraph(colony_id="dish-units")
    graph.add_cell(
        Cell(id="dish-units-000-000", colony_id="dish-units", claim_text="Physical units", level=0)
    )
    save_colony(
        path,
        graph,
        Colony(
            id="dish-units",
            dish="dish",
            center_claim="Physical units",
            center_cell_id="dish-units-000-000",
        ),
    )
    spec = SeedRunSpec(
        "dish-units", "dish", "units", DecompositionSettings(depth=3, per_layer=8, model="test")
    )
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    return path, spec, backend


@pytest.mark.asyncio
@pytest.mark.parametrize("rate_limit", [False, True])
async def test_seed_round_deduplicates_and_sleeps(seed_fixture, rate_limit):
    path, spec, backend = seed_fixture
    runner = SeedRunner(rate_limit=rate_limit)
    await backend.start_async(
        path.parent,
        db_path=path,
        dependencies=lambda spec: StubPhases().dependencies(),
        seed_runner_factory=lambda spec: runner,
    )
    try:
        first = await backend.enqueue_seed(spec)
        second = await backend.enqueue_seed(spec)
        result = await first.get_result()
        assert await second.get_result() == result
        assert result.complete and result.cell_count == 3
        assert first.workflow_id == second.workflow_id == "dish-units"
        assert len(runner.calls) == (4 if rate_limit else 3)
        assert [name for name, _ in runner.calls].count("initial") == 1
        if rate_limit:
            assert runner.calls[2][1] - runner.calls[1][1] >= 0.09
        before = database.read_events(path)
        again = await backend.enqueue_seed(spec)
        assert await again.get_result() == result
        assert database.read_events(path) == before
        assert len(database.list_cells(path)) == 3
        assert len(database.read_edges(path)) == 2
        assert len(database.read_events(path, event_type="cell_expansion_completed")) == 3
        steps = await DBOS.list_workflow_steps_async(spec.workflow_id)
        assert all(len(pickle.dumps(step["output"])) < CHECKPOINT_LIMIT for step in steps)
        assert all(row["is_atomic"] for row in database.list_cells(path) if row["level"])
    finally:
        await backend.close_async()


def test_seed_provider_cancellation_releases_its_run_lock():
    import concurrent.futures
    import threading
    from pydantic_ai.models.function import FunctionModel
    from petri.harness.legacy_bridge import HarnessInferenceProvider
    from petri.reasoning.decomposition_agents import DecompositionInput

    entered = threading.Event()

    async def block(messages, info):
        entered.set()
        await asyncio.Event().wait()

    provider = HarnessInferenceProvider(FunctionModel(block))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        call = executor.submit(provider.initial_decomposition, DecompositionInput("Physical units"))
        try:
            assert entered.wait(timeout=5)
            provider.cancel_active_run()
            with pytest.raises(concurrent.futures.CancelledError):
                call.result(timeout=5)
        finally:
            provider.cancel_active_run()
            provider.close()


@pytest.mark.asyncio
async def test_retry_failed_seed_keeps_completed_steps(seed_fixture):
    path, spec, backend = seed_fixture

    class FailingRunner(SeedRunner):
        fail = True

        def expand_why(self, request, on_progress=None):
            if self.fail:
                self.fail = False
                raise RuntimeError("Interrupted expansion")
            return super().expand_why(request, on_progress)

    runner = FailingRunner()
    start = dict(
        db_path=path,
        dependencies=lambda spec: StubPhases().dependencies(),
        seed_runner_factory=lambda spec: runner,
    )
    await backend.start_async(path.parent, **start)
    handle = await backend.enqueue_seed(spec)
    with pytest.raises(RuntimeError, match="Interrupted expansion"):
        await handle.get_result()
    await backend.close_async()
    await backend.start_async(path.parent, **start)
    try:
        handle = await backend.enqueue_seed(spec, resume=True)
        assert (await handle.get_result()).complete
        assert len(runner.calls) == 3
        assert [name for name, _ in runner.calls].count("initial") == 1
        assert len(database.read_events(path, event_type="cell_expansion_completed")) == 3
    finally:
        await backend.close_async()


def test_seed_cli_uses_a_fake_backend_without_importing_dbos(tmp_path, monkeypatch):
    import builtins
    from typer.testing import CliRunner
    from petri.cli import app
    from petri.cli import seed as seed_module
    from petri.reasoning.decomposer import decompose_claim
    from tests.conftest import FakeProvider

    class FakeBackend:
        backend_name = "fake"
        calls = 0

        def seed_colony(self, *, petri_dir, **kwargs):
            self.calls += 1
            return decompose_claim(**kwargs)

    backend = FakeBackend()
    (tmp_path / "petri.yaml").write_text("name: dish\ntelemetry:\n  enabled: false\n")
    monkeypatch.setattr(seed_module, "find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr(seed_module, "get_dish_id", lambda _: "dish")
    monkeypatch.setattr(seed_module, "resolve_backend", lambda _: backend)
    monkeypatch.setattr(seed_module, "resolve_provider", lambda _: FakeProvider())
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "dbos" or name.startswith("dbos."):
            raise AssertionError("The fake backend must not import DBOS")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = CliRunner().invoke(
        app, ["seed", "A new research claim", "--colony", "fresh", "--no-questions"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert backend.calls == 1
    assert "Execution backend: fake" in result.output
    assert not (tmp_path / "dbos.sqlite").exists()


def test_retiring_interim_seed_state_preserves_domain_tables(tmp_path):
    import sqlite3
    from contextlib import closing
    from importlib.resources import files

    path = tmp_path / "petri.sqlite"
    source = files("petri.storage.migrations")
    scripts = tuple(
        source.joinpath(name).read_text() for name in database.MIGRATIONS if int(name[:3]) <= 4
    )
    with closing(sqlite3.connect(path)) as conn:
        database.migrate(conn, scripts)
        conn.execute(
            "INSERT INTO cells(id,colony,level,seq,claim_text,status,created_at) "
            "VALUES ('dish-units-000-000','dish-units',0,0,'Physical units','NEW','2026-11-01')"
        )
        conn.execute(
            "INSERT INTO events VALUES ('event','dish-units-000-000','verdict_issued',"
            "'investigator',0,'{\"verdict\":\"EVIDENCE_SUFFICIENT\"}','2026-11-01')"
        )
        conn.commit()
        tables = ("cells", "edges", "events")
        rows = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
        schemas = {
            table: conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
            for table in tables
        }
        database.migrate(conn)
        assert {
            table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables
        } == rows
        assert {
            table: conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
            for table in tables
        } == schemas
        assert not conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE '%checkpoint%'"
        ).fetchall()

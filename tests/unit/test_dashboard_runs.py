"""Exercise dashboard submission, status, and cancellation without model calls."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from petri.dashboard.api import create_app
from petri.dashboard.config import DashboardConfig
from petri.execution.dashboard_runs import DashboardSeedSpec, _prepare_dashboard_seed
from petri.execution.run_controls import RunStatus, SeedRequest
from petri.execution.seed_workflow import SeedRunSpec
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database
from petri.storage.paths import dish_db_path


class FakeBackend:
    def __init__(self):
        self.started = 0
        self.closed = 0
        self.requests = []
        self.runs = {}
        self.progress = None
        self.stream_factory = None

    async def start_owner(
        self, petri_dir, *, on_phase_progress=None, event_stream_handler_factory=None
    ):
        self.started += 1
        self.progress = on_phase_progress
        self.stream_factory = event_stream_handler_factory

    async def shutdown_owner(self):
        self.closed += 1

    async def submit_seed(self, request):
        self.requests.append(request)
        run = RunStatus(id="dish-" + request.colony, command="seed", status="PENDING")
        self.runs[run.id] = run
        return run

    async def submit_run(self, request):
        self.requests.append(request)
        run = RunStatus(id="grow:123", command=request.command, status="PENDING")
        self.runs[run.id] = run
        return run

    async def get_run(self, petri_dir, run_id):
        return self.runs.get(run_id)

    async def cancel_run(self, petri_dir, run_id):
        if run_id not in self.runs:
            return None
        self.runs[run_id].status = "CANCELLED"
        return self.runs[run_id]


def test_submit_poll_cancel_and_shutdown_without_domain_writes(tmp_path):
    backend = FakeBackend()
    (tmp_path / "petri.yaml").write_text("name: dish\n")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with TestClient(create_app(tmp_path, execution_backend=backend)) as client:
        response = client.post(
            "/api/seed", json={"claim": "A meter is a defined unit", "colony": "units"}
        )
        assert response.status_code == 202
        assert response.json() == {
            "id": "dish-units",
            "workflow_id": "dish-units",
            "status": "PENDING",
            "status_url": "/api/runs/dish-units",
            "colony_id": "dish-units",
        }
        assert response.headers["location"] == "/api/runs/dish-units"
        assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
        assert client.get(response.json()["status_url"]).json()["status"] == "PENDING"
        assert client.post("/api/runs/dish-units/cancel").json()["status"] == "CANCELLED"
        grow = client.post("/api/runs", json={"command": "grow", "colony": "units"})
        assert grow.status_code == 202
        assert client.get(grow.json()["status_url"]).json()["id"] == "grow:123"
        assert backend.started == 1
        assert backend.progress and backend.stream_factory
    assert backend.closed == 1


@pytest.mark.parametrize(
    "url,method", [("/api/runs/missing", "get"), ("/api/runs/missing/cancel", "post")]
)
def test_unknown_run_is_404(tmp_path, url, method):
    client = TestClient(create_app(tmp_path, execution_backend=FakeBackend()))
    assert getattr(client, method)(url).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"claim": "   "},
        {"claim": "claim", "colony": "../other"},
        {"claim": "claim", "unknown": True},
    ],
)
def test_invalid_seed_is_rejected_before_start(tmp_path, body):
    backend = FakeBackend()
    client = TestClient(create_app(tmp_path, execution_backend=backend))
    assert client.post("/api/seed", json=body).status_code == 422
    assert backend.started == 0


def test_run_validation_and_backend_errors(tmp_path):
    class RejectingBackend(FakeBackend):
        async def submit_seed(self, request):
            raise ValueError("Draft already exists.")

        async def submit_run(self, request):
            raise KeyError("Colony not found.")

    client = TestClient(create_app(tmp_path, execution_backend=RejectingBackend()))
    assert client.post("/api/runs", json={"command": "cell"}).status_code == 422
    assert client.post("/api/seed", json={"claim": "A claim"}).status_code == 409
    assert client.post("/api/runs", json={"command": "grow"}).status_code == 404


@pytest.mark.parametrize(
    "origin,allowed",
    [
        ("http://127.0.0.1:8090", True),
        ("http://localhost:8090", True),
        ("https://remote.example", False),
    ],
)
def test_cors_defaults_are_local(tmp_path, origin, allowed):
    client = TestClient(create_app(tmp_path, execution_backend=FakeBackend()))
    response = client.options(
        "/api/seed",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert ("access-control-allow-origin" in response.headers) == allowed
    if allowed:
        assert response.headers["access-control-allow-origin"] == origin


def test_explicit_cors_override(tmp_path):
    client = TestClient(
        create_app(
            tmp_path, execution_backend=FakeBackend(), allowed_origins=["https://lab.example"]
        )
    )
    assert (
        client.get("/api/ping", headers={"Origin": "https://lab.example"}).headers[
            "access-control-allow-origin"
        ]
        == "https://lab.example"
    )
    assert (
        "access-control-allow-origin"
        not in client.get("/api/ping", headers={"Origin": "http://localhost:8090"}).headers
    )
    with pytest.raises(ValidationError):
        DashboardConfig(allowed_origins=["*"])


def test_dashboard_has_no_process_bridge(tmp_path):
    client = TestClient(create_app(tmp_path, execution_backend=FakeBackend()))
    assert client.post("/api/proc/start", json={"command": "grow"}).status_code == 404
    source = Path("petri/dashboard/api.py").read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Import)
        and any(name.name in {"subprocess", "pty"} for name in node.names)
        for node in ast.walk(tree)
    )
    html = client.get("/").text
    assert "/api/proc/" not in html
    assert "/api/runs/" in html and "petri.activeRun" in html


def seed_spec(claim, *, regenerate=False, attempt=1):
    request = SeedRequest(claim=claim, colony="units", regenerate=regenerate)
    return DashboardSeedSpec(
        request,
        SeedRunSpec(
            "dish-units",
            "dish",
            "units",
            DecompositionSettings(depth=2, per_layer=4, model="test"),
            attempt,
        ),
    )


def test_durable_preparation_is_atomic_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "dishes/dish/petri.sqlite"
    app = SimpleNamespace(petri_dir=tmp_path, db_path=path)
    spec = seed_spec("A defined unit")
    original = database.append_event

    def fail_event(*args, **kwargs):
        raise RuntimeError("Interrupted materialization")

    monkeypatch.setattr(database, "append_event", fail_event)
    with pytest.raises(RuntimeError, match="Interrupted"):
        _prepare_dashboard_seed(app, spec)
    assert not database.list_cells(path)
    monkeypatch.setattr(database, "append_event", original)
    _prepare_dashboard_seed(app, spec)
    before = database.read_events(path)
    _prepare_dashboard_seed(app, spec)
    assert len(database.list_cells(path)) == 1
    assert database.get_cell(path, "dish-units-000-000")["claim_text"] == "A defined unit"
    assert database.read_events(path) == before
    assert [row["type"] for row in before] == ["seed_started", "decomposition_started"]


def test_regeneration_snapshots_before_replacement_and_keeps_history(tmp_path, monkeypatch):
    from petri.execution import dashboard_runs

    (tmp_path / "defaults").mkdir()
    (tmp_path / "defaults/petri.yaml").write_text("name: dish\n")
    path = dish_db_path(tmp_path, "dish")
    app = SimpleNamespace(petri_dir=tmp_path, db_path=path)
    _prepare_dashboard_seed(app, seed_spec("Original unit claim"))
    original = dashboard_runs.snapshot_before_destructive
    snapshots = []

    def snapshot(petri_dir, dish_id):
        assert database.get_cell(path, "dish-units-000-000")["claim_text"] == "Original unit claim"
        result = original(petri_dir, dish_id)
        snapshots.append(result)
        return result

    monkeypatch.setattr(dashboard_runs, "snapshot_before_destructive", snapshot)
    backend = FakeBackend()
    client = TestClient(create_app(tmp_path, execution_backend=backend))
    response = client.post(
        "/api/seed",
        json={
            "claim": "Replacement unit claim",
            "colony": "units",
            "regenerate": True,
        },
    )
    assert response.status_code == 202
    assert snapshots == []
    assert database.get_cell(path, "dish-units-000-000")["claim_text"] == "Original unit claim"
    queued = DashboardSeedSpec(
        backend.requests[0], seed_spec("Replacement unit claim", attempt=2).seed
    )
    _prepare_dashboard_seed(app, queued)
    assert len(snapshots) == 1
    assert database.get_cell(path, "dish-units-000-000")["claim_text"] == "Replacement unit claim"
    assert len(database.read_events(path, event_type="seed_started")) == 2
    _prepare_dashboard_seed(app, seed_spec("Replacement unit claim", regenerate=True, attempt=2))
    assert len(snapshots) == 1


@pytest.mark.asyncio
async def test_real_backend_runs_prepared_seed_and_returns_saved_status(tmp_path):
    import asyncio
    from httpx import ASGITransport, AsyncClient
    from petri.cli.init import create_petri_dish
    from petri.execution.dbos_backend import DBOSExecutionBackend
    from tests.pipeline_stubs import StubPhases
    from tests.unit.test_durable_seed import SeedRunner

    create_petri_dish(tmp_path, dish_name="dish")
    path = dish_db_path(tmp_path, "dish")
    runner = SeedRunner()
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    await backend.start_async(
        tmp_path,
        db_path=path,
        dependencies=lambda spec: StubPhases().dependencies(),
        seed_runner_factory=lambda spec: runner,
    )
    app = create_app(tmp_path, execution_backend=backend)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost:8090"
        ) as client:
            response = await client.post(
                "/api/seed", json={"claim": "Physical units", "colony": "units"}
            )
            assert response.status_code == 202, response.text
            run_id = response.json()["id"]
            assert run_id == "dish-units"
            for _ in range(150):
                response = await client.get("/api/runs/" + run_id)
                assert response.status_code == 200, response.text
                if response.json()["status"] not in {"PENDING", "ENQUEUED"}:
                    break
                await asyncio.sleep(0.05)
            assert response.json()["status"] == "SUCCESS", response.text
            assert response.json()["result"]["complete"]
            assert len(database.list_cells(path)) == 3
            repeated = await client.post(
                "/api/seed", json={"claim": "Physical units", "colony": "units"}
            )
            assert repeated.json()["id"] == run_id
            conflict = await client.post(
                "/api/seed", json={"claim": "Different claim", "colony": "units"}
            )
            assert conflict.status_code == 409
            missing = await client.get("/api/runs/missing")
            assert missing.status_code == 404
    finally:
        await backend.close_async()


def test_seed_submission_does_not_bootstrap_the_owner_in_request(tmp_path):
    class UnstartedBackend(FakeBackend):
        async def start_owner(self, *args, **kwargs):
            raise AssertionError("Owner startup belongs to lifespan or initialization.")

        async def submit_seed(self, request):
            raise RuntimeError("Start the dashboard execution owner before submitting a run.")

    client = TestClient(create_app(tmp_path, execution_backend=UnstartedBackend()))
    assert client.post("/api/seed", json={"claim": "Physical units"}).status_code == 503
    assert not list(tmp_path.rglob("*"))


def test_owner_contention_keeps_reads_available_but_other_startup_errors_escape(tmp_path):
    from petri.execution.dbos_app import ExecutionOwnerError

    (tmp_path / "petri.yaml").write_text("name: dish\n")

    class OccupiedBackend(FakeBackend):
        async def start_owner(self, *args, **kwargs):
            raise ExecutionOwnerError("Another process owns this dish.")

    with TestClient(create_app(tmp_path, execution_backend=OccupiedBackend())) as client:
        assert client.get("/api/ping").status_code == 200
        assert client.get("/api/cells").json() == []
        response = client.post("/api/seed", json={"claim": "A claim"})
        assert response.status_code == 503
        assert "restart the dashboard" in response.json()["detail"]

    class BrokenBackend(FakeBackend):
        async def start_owner(self, *args, **kwargs):
            raise RuntimeError("Invalid runtime configuration")

    with pytest.raises(RuntimeError, match="Invalid runtime configuration"):
        with TestClient(create_app(tmp_path, execution_backend=BrokenBackend())):
            pass

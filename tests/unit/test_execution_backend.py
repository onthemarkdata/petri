"""Exercise the CLI execution seam without a queue file or inference call."""

import ast
import importlib
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from petri.execution.config import ExecutionConfig
from petri.execution import resolve_backend
from petri.execution.backend import ExecutionBackend
from petri.execution.models import CellPipelineState, GrowOutcome, PipelineOutcome, PreflightCheck
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.models import QueueProcessingResult


class MemoryBackend:
    status_interval = 0.01

    def __init__(self):
        self.calls = []
        self.data = {
            "dish-colony-L0-001": {
                "cell_id": "dish-colony-L0-001",
                "queue_state": "research_active",
            }
        }

    async def start_owner(self, petri_dir, **kwargs):
        self.calls.append("start_owner")

    async def shutdown_owner(self):
        self.calls.append("shutdown_owner")

    async def submit_seed(self, request):
        raise NotImplementedError

    async def submit_run(self, request):
        raise NotImplementedError

    async def get_run(self, petri_dir, run_id):
        return None

    async def cancel_run(self, petri_dir, run_id):
        return None

    def seed_colony(self, **kwargs):
        self.calls.append("seed_colony")
        return None

    def preflight(self):
        self.calls.append("preflight")
        return PreflightCheck("memory", True, "Ready")

    def process_queue(self, **kwargs):
        self.calls.append("process_queue")
        return QueueProcessingResult(dry_run=kwargs["dry_run"], would_process=list(self.data))

    def grow_loop(self, **kwargs):
        self.calls.append("grow_loop")
        result = kwargs["run_one_pass"]()
        assert kwargs["get_states"]() == {"research_active": 1}
        assert not kwargs["is_stopped"]()
        return GrowOutcome("all_terminal", {"done": 1}, result, 1)

    def state_summary(self, petri_dir):
        self.calls.append("state_summary")
        return {"research_active": 1}

    def format_state_summary(self, states):
        return str(states)

    def clear_stop(self, petri_dir):
        self.calls.append("clear_stop")

    def is_stopped(self, petri_dir):
        return False

    def request_stop(self, petri_dir, *, force=False):
        self.calls.append("request_stop")
        return 1

    def reopen_cell(self, request, *, petri_dir):
        self.calls.append("reopen_cell")

    def entries(self, petri_dir):
        self.calls.append("entries")
        return self.data

    def list_entries(self, petri_dir):
        self.calls.append("list_entries")
        return list(self.data.values())

    def update_state(self, petri_dir, cell_id, state):
        self.calls.append("update_state")
        self.data[cell_id]["queue_state"] = state
        return self.data[cell_id]


@pytest.mark.parametrize(
    "command,arguments,expected",
    [
        ("grow", ["--dry-run"], "process_queue"),
        ("grow", [], "grow_loop"),
        ("check", ["--json"], "entries"),
        ("stop", [], "request_stop"),
    ],
)
def test_cli_uses_memory_backend(
    command, arguments, expected, monkeypatch, tmp_path, canonical_colony
):
    module = importlib.import_module(f"petri.cli.{command}")
    backend = MemoryBackend()
    assert isinstance(backend, ExecutionBackend)
    monkeypatch.setattr(module, "find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "resolve_backend", lambda path: backend)
    if command == "grow":
        monkeypatch.setattr(module, "resolve_provider", lambda path: None)
    elif command == "check":
        monkeypatch.setattr(module, "get_dish_id", lambda path: "dish")
        monkeypatch.setattr(
            module,
            "load_colonies",
            lambda *args: [(canonical_colony["graph"], canonical_colony["colony_model"])],
        )
    app = typer.Typer()
    module.register(app)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    assert expected in backend.calls
    assert not (tmp_path / "queue.json").exists()


@pytest.mark.parametrize("command", ["grow", "check", "stop"])
def test_cli_rejects_unknown_backend_before_execution(command, monkeypatch, tmp_path):
    module = importlib.import_module(f"petri.cli.{command}")
    (tmp_path / "petri.yaml").write_text("execution:\n  backend: misspelled\n")
    monkeypatch.setattr(module, "find_petri_dir", lambda: tmp_path)
    app = typer.Typer()
    module.register(app)
    result = CliRunner().invoke(app, [])
    assert result.exit_code != 0
    assert "Unknown execution.backend" in result.output
    assert not (tmp_path / "queue.json").exists()


def test_config_and_state_round_trip(tmp_path):
    assert ExecutionConfig().backend == "dbos"
    assert isinstance(resolve_backend(tmp_path), DBOSExecutionBackend)
    with pytest.raises(ValueError, match="v0.3.7"):
        ExecutionConfig(backend="v1")
    with pytest.raises(ValueError):
        PipelineOutcome("unknown")
    state = CellPipelineState(
        cell_id="cell",
        iteration=3,
        weakest_link="sources",
        focused_directive="Verify sources",
        cycle_start_iteration=2,
    )
    assert CellPipelineState.model_validate(state.model_dump()) == state
    assert {outcome.value for outcome in PipelineOutcome} == {
        "converged",
        "iterate",
        "circuit_breaker",
        "short_circuit",
    }


def test_cli_and_protocol_import_boundaries():
    root = Path(__file__).resolve().parents[2]
    for filename in ["cli/grow.py", "cli/check.py", "cli/stop.py", "execution/backend.py"]:
        tree = ast.parse((root / "petri" / filename).read_text())
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert all(
                not name.startswith(("petri.engine", "petri.storage.queue", "dbos"))
                for name in modules
            ), filename

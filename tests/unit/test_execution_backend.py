"""Exercise the CLI execution seam without a queue file or inference call."""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from petri.config import get_execution_backend
from petri.execution import resolve_backend
from petri.execution.backend import ExecutionBackend
from petri.execution.models import CellPipelineState, GrowOutcome, PipelineOutcome, PreflightCheck
from petri.execution.v1_engine import V1ExecutionBackend
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
    assert get_execution_backend({}) == "v1"
    assert isinstance(resolve_backend(tmp_path), V1ExecutionBackend)
    with pytest.raises(ValueError, match="execution must be a mapping"):
        get_execution_backend({"execution": []})
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


def test_v1_adapter_delegations(monkeypatch, tmp_path):
    from petri.execution import v1_engine as adapter

    backend = V1ExecutionBackend()
    assert isinstance(backend, ExecutionBackend)
    calls = []

    def record(name, result=None):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    check = adapter.preflight.CheckResult("cli", True, "ready")
    monkeypatch.setattr(adapter.preflight, "check_claude_cli", record("preflight", check))
    assert backend.preflight() == PreflightCheck("cli", True, "ready")
    result = QueueProcessingResult(processed=1)
    monkeypatch.setattr(adapter.processor, "process_queue", record("process_queue", result))
    assert backend.process_queue(petri_dir=tmp_path) is result
    outcome = SimpleNamespace(
        reason="terminal", final_states={"done": 1}, last_result=result, passes_run=1
    )
    monkeypatch.setattr(adapter.loop, "grow_loop", record("grow_loop", outcome))
    assert backend.grow_loop(run_one_pass="stub").last_result is result
    monkeypatch.setattr(adapter.loop, "format_state_summary", record("format", "done: 1"))
    assert backend.format_state_summary({"done": 1}) == "done: 1"
    assert backend.status_interval == adapter.loop.GROW_STATUS_INTERVAL_SECONDS
    for name in ["clear_stop_file", "is_stop_file_present", "request_stop", "request_stop_file"]:
        monkeypatch.setattr(adapter.processor, name, record(name, name == "is_stop_file_present"))
    backend.clear_stop(tmp_path)
    assert backend.is_stopped(tmp_path)
    backend.request_stop(tmp_path)
    for name, value in [
        ("get_state_summary", {"done": 1}),
        ("load_queue", {"entries": {"c": {}}}),
        ("list_queue", [{"cell_id": "c"}]),
        ("update_state", None),
    ]:
        monkeypatch.setattr(adapter.queue, name, record(name, value))
    assert backend.state_summary(tmp_path) == {"done": 1}
    assert backend.entries(tmp_path) == {"c": {}}
    assert backend.list_entries(tmp_path) == [{"cell_id": "c"}]
    assert backend.update_state(tmp_path, "c", "stalled") is None
    assert calls == [
        ("preflight", (), {}),
        ("process_queue", (), {"petri_dir": tmp_path}),
        ("grow_loop", (), {"run_one_pass": "stub"}),
        ("format", ({"done": 1},), {}),
        ("clear_stop_file", (tmp_path,), {}),
        ("is_stop_file_present", (tmp_path,), {}),
        ("request_stop", (), {}),
        ("request_stop_file", (tmp_path,), {}),
        ("get_state_summary", (tmp_path / "queue.json",), {}),
        ("load_queue", (tmp_path / "queue.json",), {}),
        ("list_queue", (tmp_path / "queue.json",), {}),
        ("update_state", (tmp_path / "queue.json", "c", "stalled"), {}),
    ]


def test_adapter_translates_provider_error(monkeypatch):
    from petri.execution import v1_engine as adapter
    from petri.execution.models import NoProviderError

    def fail(**kwargs):
        raise adapter.processor.NoProviderError()

    monkeypatch.setattr(adapter.processor, "process_queue", fail)
    with pytest.raises(NoProviderError, match="No InferenceProvider configured"):
        V1ExecutionBackend().process_queue()

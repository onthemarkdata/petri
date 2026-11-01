"""Feed reports impact before submitting evidence through the durable seam."""

from types import SimpleNamespace
import importlib

from typer.testing import CliRunner

from petri.cli import app
from petri.storage import database
from petri.storage.paths import dish_db_path


def test_feed_delegates_and_flags_idempotently(petri_env_with_colony, monkeypatch):
    env = petri_env_with_colony
    directory = env["petri_dir"]
    path = dish_db_path(directory)
    (directory / "queue.json").unlink(missing_ok=True)
    module = importlib.import_module("petri.cli.feed")
    requests = []
    monkeypatch.setattr(module, "find_petri_dir", lambda: directory)
    monkeypatch.setattr(module, "resolve_backend", lambda _: SimpleNamespace(
        feed_evidence=lambda request, **options: requests.append(request)
    ))
    before = database.list_cells(path)
    for _ in range(2):
        result = CliRunner().invoke(app, ["feed", "New local evidence", "--auto-reopen"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Impact:" in result.output
    assert requests
    midpoint = len(requests) // 2
    assert [r.cell_id for r in requests[:midpoint]] == [r.cell_id for r in requests[midpoint:]]
    assert all(r.source == "New local evidence" for r in requests)
    assert database.list_cells(path) == before
    assert not (directory / "queue.json").exists()

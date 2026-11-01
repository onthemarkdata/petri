"""The stop command delegates durable cancellation without domain writes."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from petri.cli import app


@pytest.mark.parametrize("force", [False, True])
def test_stop_delegates_force_without_writing_files(tmp_path, monkeypatch, force):
    calls = []
    backend = SimpleNamespace(request_stop=lambda path, **options: calls.append((path, options)))
    monkeypatch.setattr("petri.cli.stop.find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr("petri.cli.stop.resolve_backend", lambda _: backend)
    result = CliRunner().invoke(app, ["stop", *(["--force"] if force else [])])
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, {"force": force})]
    assert "No active runs" in result.output
    assert list(tmp_path.iterdir()) == []

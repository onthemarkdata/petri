"""Seeding must not overwrite previously created research."""

import hashlib
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from petri.cli import app
from petri.dashboard.api import create_app
from petri.storage.seed_guard import ColonyExistsError, require_empty_colony
from tests.conftest import TypedFakeProvider as FakeProvider


def snapshot(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*") if p.is_file()
    }


def test_cli_preserves_existing_colony(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    before = snapshot(env["petri_dir"])
    provider = FakeProvider()
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    result = CliRunner().invoke(app, [
        "seed", "New claim", "--colony", env["colony_name"], "--no-questions",
    ])
    assert result.exit_code == 1, result.output
    assert env["colony_name"] in result.output
    assert "already exists" in result.output
    assert snapshot(env["petri_dir"]) == before
    assert provider.decompose_calls == []


def test_dashboard_preserves_existing_colony(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    def unexpected_provider(self, spec):
        pytest.fail("Existing colonies must be rejected before resolving inference.")

    monkeypatch.setattr("petri.execution.dbos_backend.DBOSExecutionBackend._default_seed_runner", unexpected_provider)
    with TestClient(create_app(env["petri_dir"])) as client:
        before = {name: digest for name, digest in snapshot(env["petri_dir"]).items()
                  if not name.endswith(("-wal", "-shm"))}
        result = client.post("/api/seed", json={"claim": "New", "colony": env["colony_name"]})
        assert result.status_code == 409
        assert "already exists" in result.json()["detail"]
        after = {name: digest for name, digest in snapshot(env["petri_dir"]).items()
                 if not name.endswith(("-wal", "-shm"))}
        assert after == before


@pytest.mark.parametrize("surface", ["cli", "dashboard"])
def test_cell_free_directory_can_be_reused(seeded_petri_dir, monkeypatch, surface):
    env = seeded_petri_dir
    target = env["petri_dir"] / "petri-dishes" / "fresh"
    target.mkdir()
    provider = FakeProvider()
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    monkeypatch.setattr("petri.cli._bootstrap.resolve_provider", lambda _: provider)
    if surface == "cli":
        result = CliRunner().invoke(app, [
            "seed", "A new claim", "--colony", "fresh", "--no-questions",
        ])
        assert result.exit_code == 0, result.output
    else:
        monkeypatch.setattr("petri.execution.dbos_backend.DBOSExecutionBackend._default_seed_runner",
                            lambda self, spec: provider)
        with TestClient(create_app(env["petri_dir"])) as client:
            result = client.post("/api/seed", json={"claim": "A new claim", "colony": "fresh"})
            assert result.status_code == 202, result.text
            for _ in range(150):
                status = client.get(result.json()["status_url"])
                if status.json()["status"] not in {"PENDING", "ENQUEUED"}:
                    break
                time.sleep(0.05)
            assert status.json()["status"] == "SUCCESS", status.text
    from petri.storage.database import list_cells
    from petri.storage.paths import dish_db_path

    assert len(list_cells(dish_db_path(env["petri_dir"]), colony=f"{env['dish_id']}-fresh")) == 3
    assert not list(target.iterdir())


def test_incomplete_colony_is_protected(tmp_path):
    cell = tmp_path / "partial" / "cell"
    cell.mkdir(parents=True)
    (cell / "metadata.json").write_text("incomplete")
    with pytest.raises(ColonyExistsError, match="1 cell"):
        require_empty_colony(cell.parent)
    assert (cell / "metadata.json").read_text() == "incomplete"


def test_symlink_destination_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ColonyExistsError, match="symbolic link"):
        require_empty_colony(link)


def test_regeneration_replaces_only_current_run(seeded_petri_dir, monkeypatch):
    from click.testing import _NamedTextIOWrapper

    env = seeded_petri_dir
    before = snapshot(env["colony_path"])
    provider = FakeProvider()
    provider.questions_response = []
    actions = iter(["Regenerate with guidance", "Accept"])
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda _: True)
    monkeypatch.setattr("questionary.select", lambda *a, **kw: SimpleNamespace(ask=lambda: next(actions)))
    monkeypatch.setattr("questionary.text", lambda *a, **kw: SimpleNamespace(ask=lambda: "Focus on evidence"))
    result = CliRunner().invoke(app, ["seed", "New claim", "--colony", "fresh"])
    assert result.exit_code == 0, result.output
    assert len(provider.decompose_calls) == 2
    assert provider.decompose_calls[-1]["guidance"] == "Focus on evidence"
    assert snapshot(env["colony_path"]) == before

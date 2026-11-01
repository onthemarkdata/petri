"""Derived exports preserve append order, deterministic bytes, and scoped writes."""

import json
from contextlib import closing

from typer.testing import CliRunner

from petri.cli import app
from petri.query.export import export_snapshot
from petri.storage import database
from petri.storage.export import export_dish, export_incremental
from tests.unit.test_cli_backup import seed


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def add_verdict(path, event_id="verdict", cell_id="dish-colony-001-000"):
    with closing(database.connect(path)) as connection:
        connection.execute("INSERT INTO events VALUES (?,?, 'verdict_issued','cell_lead',1,?,?)", (
            event_id, cell_id, json.dumps({"verdict": "VALIDATED", "summary": "The measured result agrees."}),
            "2025-01-01"))
        connection.execute("UPDATE cells SET status='VALIDATED' WHERE id=?", (cell_id,))


def test_full_export_is_deterministic_and_uses_append_order(tmp_path):
    path = seed(tmp_path)
    add_verdict(path)
    output = export_dish(tmp_path, "dish")
    first = tree(output)
    export_dish(tmp_path, "dish")
    assert tree(output) == first
    snapshot = export_snapshot(path)
    for cell in snapshot.cells:
        events = [json.loads(line) for line in (output / "cells" / cell["id"] / "events.jsonl").read_text().splitlines()]
        expected = [{key: value for key, value in event.items() if key != "rowid"}
                    for event in snapshot.events if event["cell_id"] == cell["id"]]
        assert events == expected
    markdown = (output / "cells/dish-colony-001-000/cell.md").read_text()
    assert "A test claim" in markdown and "VALIDATED" in markdown
    assert "The measured result agrees." in markdown


def test_incremental_rewrites_only_changed_cells(tmp_path):
    path = seed(tmp_path)
    output = export_dish(tmp_path, "dish")
    unchanged = output / "cells/dish-colony-001-001/cell.md"
    before = unchanged.read_bytes(), unchanged.stat().st_mtime_ns
    watermark = json.loads((output / "manifest.json").read_text())["watermark"]
    add_verdict(path)
    export_dish(tmp_path, "dish", since=watermark)
    assert (unchanged.read_bytes(), unchanged.stat().st_mtime_ns) == before
    incremental = tree(output)
    export_dish(tmp_path, "dish")
    assert tree(output) == incremental
    current = unchanged.stat().st_mtime_ns
    export_incremental(tmp_path, "dish")
    assert unchanged.stat().st_mtime_ns == current


def test_scopes_and_snapshot_export(tmp_path, monkeypatch):
    path = seed(tmp_path)
    add_verdict(path)
    output = export_dish(tmp_path, "dish", cell_id="dish-colony-001-000")
    assert sorted(p.name for p in (output / "cells").iterdir()) == ["dish-colony-001-000"]
    export_dish(tmp_path, "dish", colony="colony")
    assert len(list((output / "cells").iterdir())) == 2
    from petri.cli import backup

    monkeypatch.setattr(backup, "find_petri_dir", lambda: tmp_path)
    result = CliRunner().invoke(app, ["backup", "--dish", "dish", "--with-export"])
    assert result.exit_code == 0, result.output
    snapshot = next((tmp_path / "backups").iterdir())
    assert (snapshot / "exports/manifest.json").is_file()
    assert (snapshot / "exports/cells/dish-colony-001-000/cell.md").read_bytes() == (output / "cells/dish-colony-001-000/cell.md").read_bytes()


def test_derived_files_never_change_database_queries(tmp_path):
    from petri.query.connection import enumerate_dishes
    from petri.query.cells import list_cells

    path = seed(tmp_path)
    output = export_dish(tmp_path, "dish")
    (output / "cells/dish-colony-001-000/cell.md").write_text("edited derived text")
    assert list_cells(path)[0]["claim_text"] == "A test claim"
    assert [dish.dish_id for dish in enumerate_dishes(tmp_path)] == ["dish"]


def test_grow_completion_honors_export_flag(tmp_path, monkeypatch):
    from petri.cli import grow
    from tests.unit.test_execution_backend import MemoryBackend

    seed(tmp_path)
    monkeypatch.setattr(grow, "find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr(grow, "resolve_backend", lambda path: MemoryBackend())
    monkeypatch.setattr(grow, "resolve_provider", lambda path: None)
    result = CliRunner().invoke(app, ["grow", "--export"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "petri-dishes/dish/exports/manifest.json").is_file()


def test_restored_feed_evidence_appears_in_markdown_without_disk_files(tmp_path):
    import hashlib
    import shutil
    from petri.models import Event
    from petri.storage.backup import snapshot_before_destructive, restore_snapshot

    path = seed(tmp_path)
    content = "Public evidence: SQLite transactions are atomic.\nSource: https://www.sqlite.org/transactional.html"
    payload = {
        "content": content, "evidence_path": "evidence/public.json",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "source_kind": "file",
        "original_workflow_id": "dish-colony-001-000:r1",
        "forked_workflow_id": "dish-colony-001-000:r2", "generation": 2,
    }
    database.append_event(path, Event(
        id="fed", cell_id="dish-colony-001-000", timestamp="2026-11-01T00:00:00Z",
        type="evidence_fed", agent="execution_backend", iteration=0, data=payload,
    ))
    with database._write(path) as connection:
        connection.execute("UPDATE cells SET evidence_md=? WHERE id=?", (
            "Existing canonical evidence.", "dish-colony-001-000",
        ))
    evidence_file = path.parent / payload["evidence_path"]
    evidence_file.parent.mkdir()
    evidence_file.write_text(json.dumps({"content": content}))
    export_dish(tmp_path, "dish")
    original_events = (path.parent / "exports/cells/dish-colony-001-000/events.jsonl").read_bytes()
    snapshot = snapshot_before_destructive(tmp_path, "dish")
    shutil.rmtree(path.parent)
    restore_snapshot(tmp_path, snapshot.name)
    assert not evidence_file.exists()
    output = export_dish(tmp_path, "dish")
    markdown = (output / "cells/dish-colony-001-000/cell.md").read_text()
    assert "Existing canonical evidence." in markdown
    assert "### Fed evidence (round 2)" in markdown
    assert "> Public evidence: SQLite transactions are atomic." in markdown
    assert (output / "cells/dish-colony-001-000/events.jsonl").read_bytes() == original_events
    assert database.read_events(path, event_type="evidence_fed")[0]["data"] == payload

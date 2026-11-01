"""Regenerate reviewable text from the domain database; never import it back."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from petri.query.export import export_snapshot
from petri.storage.paths import dish_db_path

FORMAT_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _component(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Cell ID is not a safe export filename: {value!r}")
    return value


def export_dish(
    petri_dir: Path, dish_id: str, *, colony: str | None = None,
    cell_id: str | None = None, since: int | None = None, output_dir: Path | None = None,
    source_db: Path | None = None,
) -> Path:
    """Export a read-only snapshot; a rowid watermark limits changed cell writes."""
    if since is not None and (type(since) is not int or since < 0):
        raise ValueError("--since must be a nonnegative event rowid")
    if colony is not None and cell_id is not None:
        raise ValueError("Choose either --colony or --cell")
    db_path = source_db or dish_db_path(petri_dir, dish_id)
    snapshot = export_snapshot(db_path)
    target = output_dir or db_path.parent / "exports"
    if target.is_symlink() or (target / "cells").is_symlink():
        raise ValueError("Export directories must not be symlinks")
    if since is not None and since > snapshot.watermark:
        since = None
    selected = [cell for cell in snapshot.cells if
                (colony is None or cell["colony"] in {colony, f"{dish_id}-{colony}"})
                and (cell_id is None or cell["id"] == cell_id)]
    if (colony or cell_id) and not selected:
        raise ValueError("No cells match the export scope")
    selected_ids = {cell["id"] for cell in selected}
    if colony is None and cell_id is None and (target / "cells").is_dir():
        for directory in (target / "cells").iterdir():
            if directory.is_symlink():
                raise ValueError("A cell export directory must not be a symlink")
            if directory.is_dir() and directory.name not in selected_ids:
                shutil.rmtree(directory)
    changed = {event["cell_id"] for event in snapshot.events if since is None or event["rowid"] > since}
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for event in snapshot.events:
        by_cell.setdefault(event["cell_id"], []).append({key: value for key, value in event.items() if key != "rowid"})
    for cell in selected:
        directory = target / "cells" / _component(cell["id"])
        if directory.is_symlink():
            raise ValueError("A cell export directory must not be a symlink")
        if since is not None and cell["id"] not in changed and (directory / "cell.md").is_file():
            continue
        events = by_cell.get(cell["id"], [])
        dependencies = sorted({edge["to_cell"] for edge in snapshot.edges if edge["from_cell"] == cell["id"]})
        verdicts = [event for event in events if event["type"] == "verdict_issued"]
        verdict = verdicts[-1]["data"] if verdicts else {}
        evidence = cell.get("evidence_md", "") or "\n\n".join(
            event["data"].get("summary", "") for event in events if event["type"] == "evidence_appended")
        summary = cell.get("summary_md", "") or verdict.get("summary", "")
        markdown = (f"# {cell['id']}\n\n{cell['claim_text']}\n\n"
                    f"Status: {cell['status']}\n\nLevel: {cell['level']}\n\n"
                    f"Dependencies: {', '.join(dependencies) or 'None'}\n\n"
                    f"Verdict: {verdict.get('verdict', 'None')}\n\n"
                    f"## Summary\n\n{summary}\n\n## Evidence\n\n{evidence}\n")
        _write(directory / "events.jsonl", "".join(_json(event) for event in events))
        _write(directory / "cell.md", markdown)
    edges = [edge for edge in snapshot.edges if edge["from_cell"] in selected_ids or edge["to_cell"] in selected_ids]
    _write(target / "edges.jsonl", "".join(_json(edge) for edge in edges))
    _write(target / "manifest.json", _json({
        "format_version": FORMAT_VERSION, "dish_id": dish_id, "watermark": snapshot.watermark,
        "scope": {"colony": colony, "cell": cell_id}, "cells": sorted(selected_ids),
    }))
    return target


def export_incremental(petri_dir: Path, dish_id: str) -> Path:
    """Read only export metadata; domain rows always come from SQLite."""
    manifest = dish_db_path(petri_dir, dish_id).parent / "exports" / "manifest.json"
    watermark = None
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata["format_version"] == FORMAT_VERSION and metadata["scope"] == {"colony": None, "cell": None}:
            watermark = metadata["watermark"]
    except (OSError, ValueError, KeyError):
        pass
    return export_dish(petri_dir, dish_id, since=watermark)

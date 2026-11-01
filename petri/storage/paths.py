"""Canonical v2 dish paths and temporary legacy path helpers.

V2 stores cells, edges, and events in
``<petri_dir>/petri-dishes/<dish_id>/petri.sqlite`` and derived files in
that dish's ``exports/`` directory. Only ``dish_dir`` constructs the dish
path. Legacy cell-directory helpers remain for the v1 engine until its
retirement; v2 seed and read paths do not use them.
"""

from __future__ import annotations

import warnings

from pathlib import Path

from petri.models import parse_key
from typing import Iterator

__all__ = [
    "dish_dir",
    "dish_db_path",
    "parse_cell_id",
    "colony_dir",
    "cell_dir",
    "events_path",
    "metadata_path",
    "iter_events_files",
]


def dish_dir(petri_dir: Path, dish_id: str | None = None) -> Path:
    """Return the v2 dish directory without creating it."""
    from petri.storage.dish import resolve_dish_id

    name = resolve_dish_id(petri_dir) if dish_id is None else dish_id
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Dish ID must be one nonempty path component")
    path = petri_dir / "petri-dishes" / name
    if path.is_symlink():
        raise ValueError(f"Dish directory must not be a symbolic link: {path}")
    return path


def dish_db_path(petri_dir: Path, dish_id: str | None = None) -> Path:
    """Return the domain database path for the configured dish."""
    return dish_dir(petri_dir, dish_id) / "petri.sqlite"


def parse_cell_id(cell_id: str, dish_id: str | None = None) -> tuple[str, str, int, int]:
    """Deprecated tuple wrapper around ``models.parse_key``.

    Supply ``dish_id`` to distinguish hyphens in dish and colony names. Without
    it, the historical first-segment dish heuristic remains for compatibility.
    """
    warnings.warn("Use petri.models.parse_key instead.", DeprecationWarning, stacklevel=2)
    parsed = parse_key(cell_id, dish_id, strict=True)
    if dish_id is None:
        dish, _, colony = parsed["colony_prefix"].partition("-")
    else:
        dish, colony = parsed["dish"], parsed["colony"]
    return dish, colony, parsed["level"], parsed["seq"]


def colony_dir(petri_dir: Path, dish_id: str, colony_id: str) -> Path:
    """Return the on-disk directory for a colony.

    The actual layout on disk is
    ``<petri_dir>/petri-dishes/<colony_slug>/`` — there is no
    intermediate ``<dish_id>/`` subdirectory (see
    :func:`petri.graph.colony.serialize_colony` and the call sites in
    ``petri/cli.py``).

    ``colony_id`` is accepted in either form: the full
    ``{dish_id}-{colony_slug}`` produced by ``seed`` (cli.py:442) or
    the bare ``{colony_slug}``. If the ID starts with ``{dish_id}-``
    that prefix is stripped so the returned path points at the
    per-colony directory and not a non-existent doubly-prefixed one.

    Args:
        petri_dir: The root ``.petri`` directory.
        dish_id: The current dish ID. Used only to strip a redundant
            prefix from ``colony_id``.
        colony_id: Either ``"{dish_id}-{slug}"`` or just ``"{slug}"``.

    Returns:
        ``<petri_dir>/petri-dishes/<colony_slug>/``.
    """
    dish_prefix = f"{dish_id}-"
    if colony_id.startswith(dish_prefix):
        colony_slug = colony_id[len(dish_prefix):]
    else:
        colony_slug = colony_id
    return petri_dir / "petri-dishes" / colony_slug


def cell_dir(colony_path: Path, level: int, seq: int) -> Path:
    """Return the fallback on-disk directory for a cell within a colony.

    Uses the ``{level:03d}-{seq:03d}`` zero-padded naming convention
    that :func:`petri.graph.colony.serialize_colony` falls back to
    when ``colony.cell_paths`` does not contain an entry for the cell.
    Whenever possible, callers should prefer the path stored in
    ``colony.cell_paths`` — this helper is for the fallback case and
    for callers that only have a cell ID in hand.

    Args:
        colony_path: The on-disk colony directory (from
            :func:`colony_dir`).
        level: Cell level (``Cell.level``).
        seq: Cell sequence number within its level.

    Returns:
        ``<colony_path>/<level:03d>-<seq:03d>/``.
    """
    return colony_path / f"{level:03d}-{seq:03d}"


def events_path(cell_path: Path) -> Path:
    """Return the ``events.jsonl`` file inside a cell directory."""
    return cell_path / "events.jsonl"


def metadata_path(cell_path: Path) -> Path:
    """Return the ``metadata.json`` file inside a cell directory."""
    return cell_path / "metadata.json"


def iter_events_files(petri_dir: Path) -> Iterator[Path]:
    """Yield every ``events.jsonl`` file under ``petri-dishes/``.

    Walks ``<petri_dir>/petri-dishes/`` recursively. Used by the grow
    status loop and the stop command to scan all cell event logs
    without having to reconstruct per-cell paths. If
    ``petri-dishes/`` does not exist the iterator yields nothing
    rather than raising, so callers do not need to pre-check the
    directory.

    Args:
        petri_dir: The root ``.petri`` directory.

    Yields:
        Every ``events.jsonl`` file found under
        ``<petri_dir>/petri-dishes/``, in the order returned by
        :meth:`pathlib.Path.rglob`.
    """
    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.is_dir():
        return
    yield from dishes_dir.rglob("events.jsonl")

"""Centralized filesystem path helpers for petri's on-disk layout.

All of petri's persistent state lives under ``<petri_dir>/petri-dishes/``
in a fixed tree::

    <petri_dir>/petri-dishes/
        <colony_slug>/
            colony.json
            <level:03d>-<seq:03d>/          # fallback node directory name
                events.jsonl
                metadata.json
                evidence.md

In practice the node directory on disk is whatever
:func:`petri.graph.colony.serialize_colony` wrote into
``colony.node_paths`` (a slugified ``<level>-<level_slug>/<seq>-<node_slug>``
layout), but callers that have only a node ID fall back to the
``<level:03d>-<seq:03d>`` convention — which is what the helpers here
build.

Node IDs follow the schema defined by
:func:`petri.models.build_node_key`::

    {dish}-{colony}-{level:03d}-{seq:03d}

Because both the dish ID and the colony slug may themselves contain
hyphens, parsing is always performed right-to-left: the final two
hyphen-separated segments are the level and seq, and everything before
them is the ``{dish}-{colony}`` prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

__all__ = [
    "parse_node_id",
    "colony_dir",
    "node_dir",
    "events_path",
    "metadata_path",
    "iter_events_files",
]


def parse_node_id(node_id: str) -> tuple[str, str, int, int]:
    """Parse a composite node ID into its four logical parts.

    Node IDs use the schema ``{dish}-{colony}-{level:03d}-{seq:03d}``.
    Both the dish and colony slugs may contain hyphens, so parsing is
    done right-to-left: the last two hyphen-separated segments are
    ``level`` and ``seq``, and everything before them is the combined
    ``{dish}-{colony}`` prefix.

    When the ID has exactly four hyphen-separated parts, the first is
    treated as the dish and the second as the colony slug. When it has
    more, the first segment is taken as the dish and the remaining
    middle segments are joined back together as the colony slug. This
    is a lossy heuristic — the node ID alone does not carry enough
    information to unambiguously recover a multi-hyphen dish — but it
    matches the right-to-left convention used elsewhere in the
    codebase (see :func:`petri.models.parse_key`).

    Args:
        node_id: Composite node ID string.

    Returns:
        ``(dish, colony_slug, level, seq)`` with level and seq as ints.

    Raises:
        ValueError: If ``node_id`` has fewer than four hyphen-separated
            parts, or if the level/seq segments are not integers.
    """
    if not isinstance(node_id, str):
        raise ValueError(f"node_id must be a string, got {type(node_id).__name__}")

    parts = node_id.split("-")
    if len(parts) < 4:
        raise ValueError(
            f"node_id {node_id!r} has fewer than 4 hyphen-separated parts; "
            f"expected '{{dish}}-{{colony}}-{{level}}-{{seq}}'"
        )

    level_str = parts[-2]
    seq_str = parts[-1]
    try:
        level_int = int(level_str)
    except ValueError as exc:
        raise ValueError(
            f"node_id {node_id!r} level segment {level_str!r} is not an integer"
        ) from exc
    try:
        seq_int = int(seq_str)
    except ValueError as exc:
        raise ValueError(
            f"node_id {node_id!r} seq segment {seq_str!r} is not an integer"
        ) from exc

    dish = parts[0]
    colony_slug = "-".join(parts[1:-2])
    return dish, colony_slug, level_int, seq_int


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


def node_dir(colony_path: Path, level: int, seq: int) -> Path:
    """Return the fallback on-disk directory for a node within a colony.

    Uses the ``{level:03d}-{seq:03d}`` zero-padded naming convention
    that :func:`petri.graph.colony.serialize_colony` falls back to
    when ``colony.node_paths`` does not contain an entry for the node.
    Whenever possible, callers should prefer the path stored in
    ``colony.node_paths`` — this helper is for the fallback case and
    for callers that only have a node ID in hand.

    Args:
        colony_path: The on-disk colony directory (from
            :func:`colony_dir`).
        level: Node level (``Node.level``).
        seq: Node sequence number within its level.

    Returns:
        ``<colony_path>/<level:03d>-<seq:03d>/``.
    """
    return colony_path / f"{level:03d}-{seq:03d}"


def events_path(node_path: Path) -> Path:
    """Return the ``events.jsonl`` file inside a node directory."""
    return node_path / "events.jsonl"


def metadata_path(node_path: Path) -> Path:
    """Return the ``metadata.json`` file inside a node directory."""
    return node_path / "metadata.json"


def iter_events_files(petri_dir: Path) -> Iterator[Path]:
    """Yield every ``events.jsonl`` file under ``petri-dishes/``.

    Walks ``<petri_dir>/petri-dishes/`` recursively. Used by the grow
    status loop and the stop command to scan all node event logs
    without having to reconstruct per-node paths. If
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

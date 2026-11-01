"""Short-lived, read-only connections to existing per-dish databases."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DishDatabase:
    dish_id: str
    db_path: Path


def enumerate_dishes(petri_dir: Path) -> list[DishDatabase]:
    """Recognize immediate v2 dish directories; never walk backups or exports."""
    root = Path(petri_dir) / "petri-dishes"
    if not root.is_dir():
        return []
    return [
        DishDatabase(path.name, path / "petri.sqlite")
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and path.name not in {"backups", "exports"}
        and (path / "petri.sqlite").is_file()
    ]


@contextmanager
def read_connection(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Read a WAL snapshot without creating a database, migrating, or changing journal mode."""
    path = Path(db_path)
    if not path.is_file():
        yield None
        return
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table','view')", (name,)
        ).fetchone()
        is not None
    )

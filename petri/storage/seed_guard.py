"""Protect existing colonies when a new seed run chooses a destination."""

from pathlib import Path


class ColonyExistsError(ValueError):
    """The seed destination contains research that must be preserved."""


def require_empty_colony(path: Path) -> None:
    """Reject destinations containing cells, including incomplete colonies."""
    if path.is_symlink():
        raise ColonyExistsError(f"Cannot seed through a symbolic link: {path.absolute()}")
    count = sum(1 for entry in path.rglob("metadata.json") if entry.is_file())
    if count:
        raise ColonyExistsError(
            f"Colony '{path.name}' already contains {count} cell(s). "
            f"Existing research was preserved. To replace it, manually delete {path.resolve()} "
            "after saving any research you need."
        )

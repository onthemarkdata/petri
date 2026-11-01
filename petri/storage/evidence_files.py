"""Publish complete evidence files that can be rebuilt from domain events."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_once(path: Path, content: str) -> None:
    """Publish a complete evidence file; retries reuse its deterministic path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

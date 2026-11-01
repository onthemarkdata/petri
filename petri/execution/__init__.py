"""Select the execution backend without exposing engine imports to CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petri.execution.backend import ExecutionBackend


def resolve_backend(petri_dir: Path) -> ExecutionBackend:
    """Validate the dish's backend choice before constructing its adapter."""
    from petri.config import load_dish_config
    from petri.execution.dbos_backend import DBOSExecutionBackend

    config = load_dish_config(petri_dir)
    return DBOSExecutionBackend(config=config.model_dump(mode="json"))

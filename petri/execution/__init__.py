"""Select the execution backend without exposing engine imports to CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petri.execution.backend import ExecutionBackend


def resolve_backend(petri_dir: Path) -> ExecutionBackend:
    """Validate the dish's backend choice before constructing its adapter."""
    from petri.config import get_execution_backend, load_dish_config

    config = load_dish_config(petri_dir)
    if get_execution_backend(config) == "dbos":
        from petri.execution.dbos_backend import DBOSExecutionBackend

        return DBOSExecutionBackend(config=config)
    from petri.execution.v1_engine import V1ExecutionBackend

    return V1ExecutionBackend(config=config)

"""Select the execution backend without exposing engine imports to CLI commands."""

from pathlib import Path

from petri.config import get_execution_backend, load_dish_config
from petri.execution.backend import ExecutionBackend


def resolve_backend(petri_dir: Path) -> ExecutionBackend:
    """Validate the dish's backend choice before constructing its adapter."""
    get_execution_backend(load_dish_config(petri_dir))
    from petri.execution.v1_engine import V1ExecutionBackend

    return V1ExecutionBackend()

"""Local tracing with explicit configuration and optional HTTP export."""

from petri.telemetry.config import TelemetryConfig
from petri.telemetry.setup import (
    get_tracer,
    get_tracer_provider,
    init_telemetry,
    shutdown_telemetry,
)

__all__ = [
    "TelemetryConfig",
    "get_tracer",
    "get_tracer_provider",
    "init_telemetry",
    "shutdown_telemetry",
]

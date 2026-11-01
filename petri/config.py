"""Load validated settings at call time without global configuration state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from petri.agents.config import AgentConfig


def load_config(config_path: Path | None = None) -> AgentConfig:
    """Load exactly the requested v2 file, or the packaged defaults."""
    from petri.agents.config import load_agent_config

    path = config_path or Path(__file__).parent / "defaults" / "petri.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Petri config not found: {path}")
    return load_agent_config(path)


def load_dish_config(petri_dir: Path) -> AgentConfig:
    """Validate current dish overrides against packaged v2 defaults on each call."""
    from petri.agents.config import AgentConfig, _UniqueKeyLoader

    path = petri_dir / "defaults" / "petri.yaml"
    if not path.exists():
        path = petri_dir / "petri.yaml"
    values = load_config().model_dump(mode="json")
    values["name"] = petri_dir.parent.name
    if path.exists():
        overrides = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        if not isinstance(overrides, dict):
            raise ValueError(f"Dish config must be a mapping: {path}")
        if isinstance(overrides.get("harness"), dict):
            # Adapter options belong to the supplied block. Keep only the default
            # backend for partial overrides; omitted model IDs use the top level.
            values["harness"] = {"backend": values["harness"]["backend"]}
        def merge(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value
        merge(values, overrides)
    return AgentConfig.model_validate(values)


def get_telemetry_config(petri_dir: Path | None = None):
    """Read current dish settings and explicit OTLP environment overrides."""
    import os
    from urllib.parse import unquote

    from petri.telemetry.config import TelemetryConfig

    dish = load_dish_config(petri_dir if petri_dir is not None else Path.cwd() / ".petri")
    cfg = TelemetryConfig.model_validate(dish.telemetry.model_dump())
    raw = cfg.model_dump()
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        raw["otlp"]["traces_endpoint"] = endpoint.rstrip("/") + "/v1/traces"
        raw["otlp"]["logs_endpoint"] = endpoint.rstrip("/") + "/v1/logs"
    headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    if headers is not None:
        parsed = {}
        for item in filter(None, headers.split(",")):
            key, separator, value = item.partition("=")
            if not separator or not key.strip():
                raise ValueError("OTEL_EXPORTER_OTLP_HEADERS must contain key=value pairs.")
            parsed[unquote(key.strip())] = unquote(value.strip())
        raw["otlp"]["headers"] = parsed
    return TelemetryConfig.model_validate(raw)

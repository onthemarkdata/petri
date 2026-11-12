"""Shared tracer providers with explicit, opt-in HTTP export."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict, cast

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import NoOpTracerProvider, Tracer, TracerProvider

from petri.telemetry.config import TelemetryConfig
from petri.telemetry.redaction import RedactingSpanProcessor, Redactor

_lock = threading.RLock()
_providers: dict[str, TracerProvider] = {}
_active: TracerProvider = NoOpTracerProvider()
_configurations: dict[str, TelemetryConfig] = {}


class _BatchOptions(TypedDict, total=False):
    """Optional batch keywords for the elevated capture envelope."""

    max_queue_size: int
    max_export_batch_size: int


class _CaptureMetadata(Protocol):
    """Capture setting attached to Petri-owned SDK providers."""

    _petri_capture_content: bool


def _version() -> str:
    try:
        return version("petri-grow")
    except PackageNotFoundError:
        return "0.0.0"


def init_telemetry(
    config: TelemetryConfig | dict, *, dish_id: str = "petri", petri_dir: Path | None = None,
    protected_secret_values: tuple[str, ...] = (),
) -> TracerProvider:
    """Reuse one provider per configuration and dish; setup failures never stop research.

    Consumers use the returned provider or this module's accessors. This does not
    replace an embedding application's process-global OpenTelemetry provider.
    Invalid configuration is raised before setup so field errors remain visible.
    """
    global _active
    cfg = TelemetryConfig.model_validate(config)
    protected = [*protected_secret_values, *cfg.otlp.headers.values()]
    protected.extend(os.environ[name] for name in cfg.redact_env_vars if os.environ.get(name))
    redactor = Redactor(protected, limits=cfg.capture_limits)
    # Export credentials never enter cache keys, even as hashes. Compare configs in memory.
    public_config = cfg.model_dump(mode="json", exclude={"otlp"})
    fingerprint = hashlib.sha256(
        json.dumps([public_config, dish_id, str(petri_dir.resolve()) if petri_dir is not None else None], sort_keys=True).encode()
    ).hexdigest()
    with _lock:
        key = fingerprint
        index = 0
        while key in _providers:
            if _configurations[key] == cfg:
                _active = _providers[key]
                processor = getattr(_active, "_active_span_processor", None)
                if isinstance(processor, RedactingSpanProcessor):
                    processor.redactor.protect(protected)
                return _active
            index += 1
            key = f"{fingerprint}:{index}"
        provider: TracerProvider = NoOpTracerProvider()
        try:
            if cfg.enabled:
                # The larger span envelope needs bounded queues at both owned sinks.
                batch_options: _BatchOptions = (
                    {"max_queue_size": 16, "max_export_batch_size": 4}
                    if cfg.capture_limits is not None else {}
                )
                limits = SpanLimits()
                sdk = SDKTracerProvider(
                    span_limits=limits,
                    active_span_processor=RedactingSpanProcessor(redactor, limits),
                    resource=Resource.create(
                        {
                            "service.name": cfg.service_name,
                            "service.version": _version(),
                            "petri.dish_id": dish_id,
                        }
                    ),
                    sampler=ParentBased(TraceIdRatioBased(cfg.sample_ratio)),
                )
                if cfg.local.enabled and petri_dir is not None:
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    from petri.telemetry.span_store import SQLiteSpanExporter

                    db_path = petri_dir / "petri-dishes" / dish_id / "petri.sqlite"
                    sdk.add_span_processor(BatchSpanProcessor(SQLiteSpanExporter(db_path), **batch_options))
                if cfg.otlp.traces_endpoint:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    exporter = OTLPSpanExporter(
                        endpoint=cfg.otlp.traces_endpoint,
                        headers=cfg.otlp.headers,
                    )
                    sdk.add_span_processor(BatchSpanProcessor(exporter, **batch_options))
                cast(_CaptureMetadata, sdk)._petri_capture_content = cfg.capture_content
                provider = sdk
        except Exception as error:  # noqa: BLE001 — telemetry setup must not stop research
            warnings.warn(
                f"Telemetry setup failed ({type(error).__name__}); continuing without tracing.",
                RuntimeWarning,
                stacklevel=2,
            )
        _providers[key] = provider
        _configurations[key] = cfg.model_copy(deep=True)
        _active = provider
        return provider


def get_tracer_provider() -> TracerProvider:
    """Return the active provider without reading configuration or creating exporters."""
    with _lock:
        return _active


def get_tracer(name: str = "petri") -> Tracer:
    return get_tracer_provider().get_tracer(name)


def shutdown_telemetry() -> None:
    """Flush and close owned providers; callers can initialize them again afterward."""
    global _active
    with _lock:
        for provider in _providers.values():
            if isinstance(provider, SDKTracerProvider):
                provider.shutdown()
        _providers.clear()
        _configurations.clear()
        _active = NoOpTracerProvider()

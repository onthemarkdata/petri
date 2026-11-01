"""Shared tracer providers with explicit, opt-in HTTP export."""

from __future__ import annotations

import hashlib
import json
import threading
import warnings
from importlib.metadata import PackageNotFoundError, version

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import NoOpTracerProvider, Tracer, TracerProvider

from petri.telemetry.config import TelemetryConfig

_lock = threading.RLock()
_providers: dict[str, TracerProvider] = {}
_active: TracerProvider = NoOpTracerProvider()


def _version() -> str:
    try:
        return version("petri-grow")
    except PackageNotFoundError:
        return "0.0.0"


def init_telemetry(config: TelemetryConfig | dict, *, dish_id: str = "petri") -> TracerProvider:
    """Reuse one provider per configuration and dish; setup failures never stop research.

    Consumers use the returned provider or this module's accessors. This does not
    replace an embedding application's process-global OpenTelemetry provider.
    Invalid configuration is raised before setup so field errors remain visible.
    """
    global _active
    cfg = TelemetryConfig.model_validate(config)
    fingerprint = hashlib.sha256(
        json.dumps([cfg.model_dump(mode="json"), dish_id], sort_keys=True).encode()
    ).hexdigest()
    with _lock:
        if fingerprint in _providers:
            _active = _providers[fingerprint]
            return _active
        provider: TracerProvider = NoOpTracerProvider()
        try:
            if cfg.enabled:
                sdk = SDKTracerProvider(
                    resource=Resource.create(
                        {
                            "service.name": cfg.service_name,
                            "service.version": _version(),
                            "petri.dish_id": dish_id,
                        }
                    ),
                    sampler=ParentBased(TraceIdRatioBased(cfg.sample_ratio)),
                )
                if cfg.otlp.traces_endpoint:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    exporter = OTLPSpanExporter(
                        endpoint=cfg.otlp.traces_endpoint,
                        headers=cfg.otlp.headers,
                    )
                    sdk.add_span_processor(BatchSpanProcessor(exporter))
                provider = sdk
        except Exception as error:  # noqa: BLE001 — telemetry setup must not stop research
            warnings.warn(
                f"Telemetry setup failed ({type(error).__name__}); continuing without tracing.",
                RuntimeWarning,
                stacklevel=2,
            )
        _providers[fingerprint] = provider
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
        _active = NoOpTracerProvider()

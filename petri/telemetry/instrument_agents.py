"""Route Pydantic AI spans to Petri's shared provider without cloud configuration."""

import threading

from opentelemetry.trace import TracerProvider
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from petri.telemetry.setup import get_tracer_provider

_lock = threading.Lock()
_active_provider: TracerProvider | None = None
_settings: InstrumentationSettings | None = None
_capture_content = False


def instrument_agents(provider: TracerProvider | None = None, *, capture_content: bool | None = None) -> InstrumentationSettings:
    """Instrument all agents once per active provider, including agents built later."""
    global _active_provider, _settings, _capture_content
    selected = provider if provider is not None else get_tracer_provider()
    capture = getattr(selected, "_petri_capture_content", False) if capture_content is None else capture_content
    with _lock:
        if selected is _active_provider and capture == _capture_content and _settings is not None:
            return _settings
        settings = InstrumentationSettings(
            tracer_provider=selected,
            include_content=capture,
            include_binary_content=False,
            include_model_request_parameters=capture,
            version=5,
        )
        Agent.instrument_all(settings)
        _active_provider, _settings, _capture_content = selected, settings, capture
        return settings

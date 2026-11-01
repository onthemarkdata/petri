"""Route DBOS 3.4 native spans through the current dish's owned provider."""

from importlib.metadata import version
from typing import Any

from opentelemetry import trace

from petri.config import get_telemetry_config
from petri.storage.dish import resolve_dish_id
from petri.telemetry.instrument_agents import instrument_agents
from petri.telemetry.setup import init_telemetry


class DBOSTelemetry:
    """Own a scoped adapter for the pinned SDK's global-tracer-only API."""

    def __init__(self, petri_dir):
        self.config = get_telemetry_config(petri_dir)
        self.dish_id = resolve_dish_id(petri_dir)
        self.provider = init_telemetry(self.config, dish_id=self.dish_id, petri_dir=petri_dir)
        instrument_agents(self.provider)
        self._native: Any = None
        self._original: Any = None

    def dbos_config(self):
        # Petri owns exporters, headers, sampling, and the local sink. DBOS only emits spans.
        return {
            "enable_otlp": self.config.enabled,
            "otel_attribute_format": "semconv",
            "otlp_traces_endpoints": [],
            "otlp_logs_endpoints": [],
        }

    def bind(self):
        if not self.config.enabled:
            return
        if version("dbos") != "3.4.0":
            raise RuntimeError("Review the DBOS telemetry adapter before changing dbos==3.4.0")
        from dbos._tracer import dbos_tracer
        from dbos._utils import GlobalParams

        self._native = dbos_tracer
        self._original = dbos_tracer.start_span

        def start_span(attributes, parent=None):
            tracer = self.provider.get_tracer("dbos-tracer")
            context = trace.set_span_in_context(parent) if parent else None
            span = tracer.start_span(name=attributes["name"], context=context)
            values = {
                **attributes,
                "applicationID": dbos_tracer.app_id,
                "applicationVersion": GlobalParams.app_version,
                "executorID": GlobalParams.executor_id,
            }
            for key, value in values.items():
                if key != "name" and isinstance(value, (str, bool, int, float)):
                    span.set_attribute(dbos_tracer._resolve_attribute_name(key), value)
            span.set_attributes(dbos_tracer.otlp_attributes)
            return span

        setattr(dbos_tracer, "start_span", start_span)

    def close(self):
        if self._native is not None and self._original is not None:
            setattr(self._native, "start_span", self._original)
            self._native = self._original = None
        flush = getattr(self.provider, "force_flush", None)
        if flush:
            flush()


def set_cell_workflow_attributes(spec, app, span):
    """Attach explicit round identity without putting DBOS imports in domain phases."""
    if span is None:
        return
    from petri.models import parse_key

    identity = parse_key(spec.cell_id)
    span.set_attributes({
        "petri.cell_id": spec.cell_id,
        "petri.round": spec.round_number,
        "petri.dish_id": app.telemetry.dish_id,
        **({"petri.colony_id": identity["colony_prefix"]} if "colony_prefix" in identity else {}),
        **({"petri.level": identity["level"], "petri.seq": identity["seq"]} if "level" in identity else {}),
    })

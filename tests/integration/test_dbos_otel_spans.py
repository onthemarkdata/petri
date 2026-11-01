"""Native DBOS spans use Petri providers on SQLite without global-provider changes."""

from contextlib import closing

from dbos import DBOS
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.otel import DBOSTelemetry
from petri.storage import database
from tests.pipeline_stubs import StubPhases


@DBOS.step(name="fixture.first")
async def first():
    return 1


@DBOS.step(name="fixture.second")
async def second(value):
    return value + 1


@DBOS.workflow(name="fixture.two_steps")
async def two_steps():
    return await second(await first())


@pytest.mark.asyncio
async def test_sqlite_native_spans_route_sequential_dishes_and_restore_provider(tmp_path, monkeypatch):
    from dbos._tracer import dbos_tracer
    import petri.execution.otel as wiring

    original = dbos_tracer.start_span
    global_provider = trace.get_tracer_provider()
    providers = []
    exporters = []
    order = []
    def initialize(config, **options):
        order.append("init")
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        providers.append(provider)
        exporters.append(exporter)
        return provider
    monkeypatch.setattr(wiring, "init_telemetry", initialize)
    launch = DBOS.launch
    def launch_recorded():
        order.append("launch")
        return launch()
    monkeypatch.setattr(DBOS, "launch", launch_recorded)
    for name in ("first-dish", "second-dish"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "petri.yaml").write_text(f"name: {name}\ntelemetry:\n  local:\n    enabled: false\n")
        backend = DBOSExecutionBackend()
        await backend.start_async(directory, db_path=directory / "domain.sqlite",
                                  dependencies=lambda _: StubPhases().dependencies())
        try:
            handle = await DBOS.start_workflow_async(two_steps)
            assert await handle.get_result() == 2
        finally:
            await backend.close_async()
        assert dbos_tracer.start_span == original
    assert order == ["init", "launch", "init", "launch"]
    assert trace.get_tracer_provider() is global_provider
    for exporter in exporters:
        spans = [span for span in exporter.get_finished_spans() if span.name.startswith("fixture.")]
        assert len(spans) == 3
        workflow = next(span for span in spans if span.name == "fixture.two_steps")
        children = [span for span in spans if span is not workflow]
        assert workflow.attributes["dbos.operation.type"] == "workflow"
        assert all(span.attributes["dbos.operation.type"] == "step" for span in children)
        assert all(span.parent.span_id == workflow.context.span_id for span in children)
    assert exporters[0].get_finished_spans()[0].context.trace_id != exporters[1].get_finished_spans()[0].context.trace_id
    for provider in providers:
        provider.shutdown()


@pytest.mark.asyncio
async def test_disabled_dbos_emits_no_spans(tmp_path, monkeypatch):
    import petri.execution.otel as wiring

    (tmp_path / "petri.yaml").write_text("name: disabled\ntelemetry:\n  enabled: false\n")
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(wiring, "init_telemetry", lambda *args, **options: provider)
    backend = DBOSExecutionBackend()
    await backend.start_async(tmp_path, db_path=tmp_path / "domain.sqlite",
                              dependencies=lambda _: StubPhases().dependencies())
    try:
        assert backend.app.telemetry.dbos_config()["enable_otlp"] is False
        handle = await DBOS.start_workflow_async(two_steps)
        assert await handle.get_result() == 2
    finally:
        await backend.close_async()
        provider.shutdown()
    assert exporter.get_finished_spans() == ()


def test_endpoint_reaches_the_owned_exporter_boundary_without_duplicate_dbos_export(tmp_path, monkeypatch):
    import petri.execution.otel as wiring
    from opentelemetry.trace import NoOpTracerProvider

    endpoint = "http://127.0.0.1:4318/v1/traces"
    (tmp_path / "petri.yaml").write_text(f"telemetry:\n  otlp:\n    traces_endpoint: {endpoint}\n")
    seen = []
    monkeypatch.setattr(wiring, "init_telemetry", lambda config, **options: seen.append(config) or NoOpTracerProvider())
    bridge = DBOSTelemetry(tmp_path)
    assert seen[0].otlp.traces_endpoint == endpoint
    assert bridge.dbos_config() == {
        "enable_otlp": True, "otel_attribute_format": "semconv",
        "otlp_traces_endpoints": [], "otlp_logs_endpoints": [],
    }
    bridge.close()
    assert not (tmp_path / "domain.sqlite").exists()


@pytest.mark.asyncio
async def test_cell_workflow_identity_is_written_to_the_local_sink(tmp_path):
    from petri.execution.dbos_backend import CellRunSpec

    (tmp_path / "petri.yaml").write_text("name: dish\n")
    backend = DBOSExecutionBackend()
    path = tmp_path / "petri-dishes" / "dish" / "petri.sqlite"
    await backend.start_async(tmp_path, db_path=path, dependencies=lambda _: StubPhases().dependencies())
    try:
        handle = await backend.enqueue_cell(CellRunSpec("dish-colony-002-001", 1, "2026-11-01"))
        await handle.get_result()
    finally:
        await backend.close_async()
    import json
    with closing(database.connect(path)) as conn:
        attributes = [json.loads(row[0]) for row in conn.execute("SELECT attributes FROM spans")]
    assert any(row.get("petri.cell_id") == "dish-colony-002-001" and row.get("petri.round") == 1
               and row.get("petri.level") == 2 for row in attributes)

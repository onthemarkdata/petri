"""Manual scans and persisted cron firings share findings and event identity."""

import time
from datetime import UTC, datetime

import pytest
from dbos import DBOS, SetWorkflowID
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError
from typer.testing import CliRunner

from petri.cli import app
from petri.execution.config import ExecutionConfig
from petri.execution.dbos_app import DBOSApp
from petri.execution.scan_workflow import SCHEDULE_NAME, run_scan_now, scan_workflow
from petri.storage import database
from tests.unit.test_scanner_v2 import corrupt, seed


def runtime(tmp_path, path, cron=None):
    return DBOSApp(tmp_path, path, ExecutionConfig(scan_cron=cron), lambda spec: None).start()


def test_manual_scheduled_findings_event_dedup_and_root_spans(tmp_path, monkeypatch):
    path = seed(tmp_path)
    corrupt(path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("petri.execution.scan_workflow.get_tracer", provider.get_tracer)
    running = runtime(tmp_path, path)
    try:
        assert DBOS.get_schedule(SCHEDULE_NAME) is None
        manual = run_scan_now()
        for _ in range(2):
            with SetWorkflowID("scheduled-scan"):
                scheduled = scan_workflow(datetime(2026, 1, 1, tzinfo=UTC), {"trigger": "scheduled"})
        assert [item.model_dump() for item in manual] == scheduled
        events = database.read_events(path, event_type="scan_completed")
        assert len(events) == 2
        assert [event["data"]["findings_count"] for event in events] == [3, 3]
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(span.parent is None for span in spans)
        assert {span.attributes["petri.scan.trigger"] for span in spans} == {"manual", "scheduled"}
        assert all(span.attributes["petri.dish_id"] == "dish" for span in spans)
    finally:
        running.close()
        provider.shutdown()


def wait_for_firing(path, count):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        events = database.read_events(path, event_type="scan_completed")
        if len(events) >= count:
            return events
        time.sleep(0.05)
    raise AssertionError("The scan cron did not fire within 12 seconds")


def test_real_cron_persists_across_restart(tmp_path):
    path = seed(tmp_path)
    running = runtime(tmp_path, path, "*/2 * * * * *")
    try:
        first = wait_for_firing(path, 1)
        DBOS.pause_schedule(SCHEDULE_NAME)
    finally:
        running.close()
    running = runtime(tmp_path, path, "*/2 * * * * *")
    try:
        events = wait_for_firing(path, len(first) + 1)
        DBOS.pause_schedule(SCHEDULE_NAME)
        assert len({event["id"] for event in events}) == len(events)
        schedules = DBOS.list_schedules()
        assert len(schedules) == 1
        runs = DBOS.list_workflows()
        assert all(event["data"]["trigger"] == "scheduled" for event in events)
        assert len(runs) == len(events)
    finally:
        running.close()
    running = runtime(tmp_path, path)
    try:
        assert DBOS.get_schedule(SCHEDULE_NAME)["status"] == "PAUSED"
    finally:
        running.close()


def test_cron_validation():
    with pytest.raises(ValidationError, match="cron"):
        ExecutionConfig(scan_cron="invalid")


def test_cli_returns_zero_or_one(tmp_path, monkeypatch):
    path = seed(tmp_path)
    monkeypatch.setattr("petri.cli.scan.find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr("petri.cli.scan.get_dish_id", lambda _: "dish")
    monkeypatch.setattr("petri.cli.scan.bootstrap_telemetry", lambda _: None)
    clean = CliRunner().invoke(app, ["scan"])
    assert clean.exit_code == 0, clean.exception
    corrupt(path)
    broken = CliRunner().invoke(app, ["scan"])
    assert broken.exit_code == 1, broken.exception
    assert "phantom_edge" in broken.output and "event_payload" in broken.output


def test_manual_scan_does_not_start_an_execution_runtime(tmp_path,monkeypatch):
    path = seed(tmp_path)
    monkeypatch.setattr('petri.cli.scan.find_petri_dir',lambda:tmp_path)
    monkeypatch.setattr('petri.cli.scan.get_dish_id',lambda _: 'dish')
    monkeypatch.setattr('petri.cli.scan.bootstrap_telemetry',lambda _:None)
    def fail_start(*args):
        raise AssertionError('Manual scan started execution recovery')
    monkeypatch.setattr(DBOSApp,'start',fail_start)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('petri.analysis.scan_run.get_tracer',provider.get_tracer)
    result = CliRunner().invoke(app,['scan'])
    assert result.exit_code==0,result.exception
    assert len(database.read_events(path,event_type='scan_completed'))==1
    assert not (tmp_path/'dbos.sqlite').exists()
    spans = exporter.get_finished_spans()
    assert len(spans)==1 and spans[0].parent is None
    assert spans[0].attributes['petri.dish_id']=='dish'
    assert spans[0].attributes['petri.scan.trigger']=='manual'
    assert spans[0].attributes['petri.scan.findings_count']==0
    provider.shutdown()

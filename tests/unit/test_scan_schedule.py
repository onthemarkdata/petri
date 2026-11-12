"""Manual scans and persisted cron firings share findings and event identity."""

import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from dbos import DBOS, SetWorkflowID
from filelock import FileLock, Timeout
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


def pause_and_wait_for_scans():
    DBOS.pause_schedule(SCHEDULE_NAME)
    deadline = time.monotonic() + 12
    while True:
        runs = DBOS.list_workflows(
            schedule_name=SCHEDULE_NAME, load_input=False, load_output=False
        )
        pending = [run for run in runs if run.status in {"PENDING", "ENQUEUED", "DELAYED"}]
        if not pending:
            for run in runs:
                DBOS.retrieve_workflow(run.workflow_id).get_result()
            return runs
        assert time.monotonic() < deadline, (
            "Scheduled scans did not settle: "
            + ", ".join(f"{run.workflow_id}={run.status}" for run in pending)
        )
        time.sleep(0.01)


@contextmanager
def runtime(tmp_path, path, cron=None):
    running = DBOSApp(tmp_path, path, ExecutionConfig(scan_cron=cron), lambda spec: None).start()
    try:
        yield running
    finally:
        try:
            if DBOS.get_schedule(SCHEDULE_NAME) is not None:
                pause_and_wait_for_scans()
            running.close()
        finally:
            # A failed wait or close must not leave this test owning the next runtime.
            if running.started:
                running.detach()


def test_manual_scheduled_findings_event_dedup_and_root_spans(tmp_path, monkeypatch):
    path = seed(tmp_path)
    corrupt(path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("petri.execution.scan_workflow.get_tracer", provider.get_tracer)
    try:
        with runtime(tmp_path, path):
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
    with runtime(tmp_path, path, "*/2 * * * * *"):
        wait_for_firing(path, 1)
        pause_and_wait_for_scans()
        first = database.read_events(path, event_type="scan_completed")
    with runtime(tmp_path, path, "*/2 * * * * *"):
        wait_for_firing(path, len(first) + 1)
        runs = pause_and_wait_for_scans()
        events = database.read_events(path, event_type="scan_completed")
        assert len({event["id"] for event in events}) == len(events)
        schedules = DBOS.list_schedules()
        assert len(schedules) == 1
        assert all(event["data"]["trigger"] == "scheduled" for event in events)
        assert all(run.status == "SUCCESS" for run in runs)
        assert len(runs) == len(events)
    with runtime(tmp_path, path):
        assert DBOS.get_schedule(SCHEDULE_NAME)["status"] == "PAUSED"


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


def test_runtime_waits_after_scan_event_until_workflow_completes(tmp_path, monkeypatch):
    from petri.execution.scan_workflow import run_domain_scan

    path = seed(tmp_path)
    release = threading.Event()
    pending_observed = threading.Event()
    close_called = threading.Event()
    finished = threading.Event()
    run_ids = []
    errors = []
    original_list = DBOS.list_workflows
    original_close = DBOSApp.close

    def held_scan(*args, **kwargs):
        findings = run_domain_scan(*args, **kwargs)
        assert release.wait(15), "The test did not release the scheduled scan"
        return findings

    def list_runs(**kwargs):
        runs = original_list(**kwargs)
        if kwargs.get("schedule_name") == SCHEDULE_NAME and any(
            run.status == "PENDING" for run in runs
        ):
            pending_observed.set()
        return runs

    def close(running):
        close_called.set()
        return original_close(running)

    monkeypatch.setattr("petri.execution.scan_workflow.run_domain_scan", held_scan)
    monkeypatch.setattr(DBOS, "list_workflows", list_runs)
    monkeypatch.setattr(DBOSApp, "close", close)

    def scan_and_close():
        try:
            with runtime(tmp_path, path, "0 0 1 1 *"):
                run_ids.append(DBOS.trigger_schedule(SCHEDULE_NAME).get_workflow_id())
                wait_for_firing(path, 1)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=scan_and_close)
    worker.start()
    try:
        assert pending_observed.wait(8)
        assert len(database.read_events(path, event_type="scan_completed")) == 1
        assert DBOS.get_workflow_status(run_ids[0]).status == "PENDING"
        assert DBOS.get_schedule(SCHEDULE_NAME)["status"] == "PAUSED"
        assert not close_called.is_set()
        assert not finished.is_set()
        with pytest.raises(Timeout):
            FileLock(tmp_path / ".execution-owner.lock").acquire(timeout=0)
        release.set()
        worker.join(10)
        assert not worker.is_alive()
        assert not errors
        assert close_called.is_set()
        assert finished.is_set()
        with FileLock(tmp_path / ".execution-owner.lock").acquire(timeout=0):
            pass
    finally:
        release.set()
        worker.join(20)
    next_directory = tmp_path / "next-runtime"
    with runtime(next_directory, seed(next_directory)):
        pass


@pytest.mark.parametrize("failure", ["wait", "close"])
def test_runtime_cleanup_failure_releases_execution_owner(tmp_path, monkeypatch, failure):
    path = seed(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError(f"Injected {failure} failure")

    with monkeypatch.context() as patch:
        if failure == "wait":
            patch.setattr(f"{__name__}.pause_and_wait_for_scans", fail)
        else:
            patch.setattr(DBOSApp, "close", fail)
        with pytest.raises(RuntimeError, match=f"Injected {failure} failure"):
            with runtime(tmp_path, path, "0 0 1 1 *"):
                pass
    with FileLock(tmp_path / ".execution-owner.lock").acquire(timeout=0):
        pass
    with runtime(tmp_path, path):
        pass

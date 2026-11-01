"""Span persistence is bounded, local, and separate from domain history."""

import json
import logging
import multiprocessing
import sqlite3
from contextlib import closing

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from petri.storage.database import connect
from petri.telemetry import init_telemetry, shutdown_telemetry
from petri.telemetry.span_store import SQLiteSpanExporter, prune, query_spans


def make_spans(count=100, start=1_000_000_000):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('fixture')
    with tracer.start_as_current_span('root', start_time=start) as root:
        root.set_attribute('petri.cell_id', 'cell-a')
        root.set_attribute('petri.phase', 'research')
        root.set_attribute('petri.tokens_input', 10)
        root.set_attribute('petri.tokens_output', 5)
        for index in range(count - 1):
            child = tracer.start_span('child', start_time=start + index + 1)
            child.set_attribute('petri.cell_id', 'cell-b')
            child.end(end_time=start + index + 100)
    spans = exporter.get_finished_spans()
    provider.shutdown()
    return spans


def export_in_child(path, queue):
    exporter = SQLiteSpanExporter(path)
    queue.put(exporter.export(make_spans()).name)


def test_roundtrip_usage_filters_and_domain_preservation(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    with closing(connect(path)) as connection:
        connection.execute("INSERT INTO events VALUES ('domain','cell-a','search_executed','agent',1,'{}','2026-01-01')")
        before = {table: connection.execute('SELECT * FROM ' + table).fetchall() for table in ['events', 'cells', 'edges']}
    spans = make_spans()
    exporter = SQLiteSpanExporter(path)
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    rows = query_spans(tmp_path, limit=None)
    assert len(rows) == 100
    assert len(query_spans(tmp_path, cell_id='cell-a')) == 1
    assert len(query_spans(tmp_path, trace_id=rows[0]['trace_id'])) == 100
    root = next(row for row in rows if row['name'] == 'root')
    assert root['parent_span_id'] is None
    assert root['attributes']['petri.tokens_input'] == 10
    assert all(row['parent_span_id'] == root['span_id'] for row in rows if row['name'] == 'child')
    with closing(connect(path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM usage').fetchone()[0] == 1
        usage = connection.execute('SELECT * FROM usage').fetchone()
        assert (usage['tokens_input'], usage['tokens_output'], usage['cost_usd']) == (10, 5, None)
        assert connection.execute('SELECT cell_id,phase,SUM(tokens_input) FROM usage GROUP BY cell_id,phase').fetchone()[:] == ('cell-a', 'research', 10)
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 3
        assert connection.execute('PRAGMA index_list(usage)').fetchall()
        for table in before:
            assert connection.execute('SELECT * FROM ' + table).fetchall() == before[table]
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    assert len(query_spans(tmp_path, limit=None)) == 100
    assert not (tmp_path / 'petri.sqlite').exists()


def test_locked_database_drops_batch_and_warns_once(tmp_path, caplog):
    path = tmp_path / 'petri.sqlite'
    exporter = SQLiteSpanExporter(path)
    spans = make_spans(1)
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    with closing(connect(path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        with caplog.at_level(logging.WARNING):
            assert exporter.export(spans) == SpanExportResult.FAILURE
            assert exporter.export(spans) == SpanExportResult.FAILURE
        writer.rollback()
    assert len(caplog.records) == 1
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    with closing(connect(path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        writer.rollback()


def test_readonly_export_failure_never_raises(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'petri.sqlite'
    with closing(connect(path)):
        pass
    monkeypatch.setattr('petri.telemetry.span_store._connect', lambda _: sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, isolation_level=None))
    exporter = SQLiteSpanExporter(path)
    assert exporter.export(make_spans(1)) == SpanExportResult.FAILURE
    assert exporter.export(make_spans(1)) == SpanExportResult.FAILURE
    assert len(caplog.records) == 1


def test_migration_upgrades_prior_schema_and_reopens(tmp_path):
    from petri.storage.database import migrate
    from importlib.resources import files

    path = tmp_path / 'petri.sqlite'
    source = files('petri.storage.migrations')
    with sqlite3.connect(path) as connection:
        migrate(connection, [source.joinpath(name).read_text() for name in ['001_domain.sql', '002_colonies.sql']])
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 2
    with closing(connect(path)) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 3
    with closing(connect(path)) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 3


def test_prune_keeps_recent_rows_and_domain_history(tmp_path):
    path = tmp_path / 'petri.sqlite'
    exporter = SQLiteSpanExporter(path)
    assert exporter.export(make_spans(1, start=1)) == SpanExportResult.SUCCESS
    now = 40 * 86_400_000_000_000
    assert exporter.export(make_spans(1, start=now)) == SpanExportResult.SUCCESS
    with closing(connect(path)) as connection:
        connection.execute("INSERT INTO events VALUES ('keep','cell','search_executed','agent',1,'{}','2020-01-01')")
    assert prune(path, 30, now_ns=now) == 1
    assert len(query_spans(path)) == 1
    with closing(connect(path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM usage').fetchone()[0] == 1
        assert connection.execute('SELECT id FROM events').fetchone()[0] == 'keep'


def test_multiprocess_writer_and_readonly_reader(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    with closing(connect(path)):
        pass
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=export_in_child, args=(path, queue))
    process.start()
    while process.is_alive():
        assert len(query_spans(tmp_path, limit=None)) in (0, 100)
        process.join(timeout=0.01)
    assert process.exitcode == 0
    assert queue.get(timeout=5) == 'SUCCESS'
    assert len(query_spans(tmp_path, limit=None)) == 100
    queue.close()


def test_provider_registers_local_exporter_and_shutdown_flushes(tmp_path, monkeypatch):
    shutdown_telemetry()
    monkeypatch.setattr('socket.socket.connect', lambda *args: pytest.fail('network'))
    provider = init_telemetry({}, dish_id='dish', petri_dir=tmp_path)
    with provider.get_tracer('fixture').start_as_current_span('flushed'):
        pass
    shutdown_telemetry()
    assert len(query_spans(tmp_path)) == 1
    assert not (tmp_path / 'petri.sqlite').exists()
    disabled = init_telemetry({'local': {'enabled': False}}, dish_id='dish', petri_dir=tmp_path)
    assert not disabled._active_span_processor._span_processors
    shutdown_telemetry()


def test_missing_telemetry_returns_no_rows_without_writes(tmp_path):
    assert query_spans(tmp_path) == []
    assert prune(tmp_path) == 0
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / 'petri-dishes' / 'old' / 'petri.sqlite'
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE old_table (value TEXT)')
    assert query_spans(tmp_path) == []
    assert prune(tmp_path) == 0
    assert json.loads(json.dumps(query_spans(tmp_path))) == []

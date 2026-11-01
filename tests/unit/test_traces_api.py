"""Trace routes expose the local store without a runtime or model call."""

import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from petri.dashboard.api import create_app
from petri.storage.database import connect
from petri.telemetry.span_store import SQLiteSpanExporter


def seed_trace(path, *, cell_id='cell-a', start=100):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(path)))
    tracer = provider.get_tracer('fixture')
    root = tracer.start_span('command', start_time=start)
    from opentelemetry import trace
    with trace.use_span(root):
        child = tracer.start_span('agent', start_time=start + 10, attributes={'petri.cell_id': cell_id, 'petri.colony_id': 'colony-a'})
        with trace.use_span(child):
            leaf = tracer.start_span('model', start_time=start + 20)
            leaf.end(end_time=start + 40)
        child.set_attribute('petri.verdict', 'VALIDATED')
        child.end(end_time=start + 50)
    root.end(end_time=start + 100)
    provider.shutdown()
    return f'{root.context.trace_id:032x}'


@pytest.fixture
def client_and_path(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    return TestClient(create_app(tmp_path, tmp_path / 'unused.sqlite')), path


def test_trace_summary_and_parent_tree(client_and_path):
    client, path = client_and_path
    trace_id = seed_trace(path)
    summary = client.get('/api/traces').json()
    assert summary == [{'trace_id': trace_id, 'name': 'command', 'start_time_ns': 100, 'duration_ns': 100, 'cell_id': 'cell-a', 'verdict': 'VALIDATED'}]
    spans = client.get('/api/traces/' + trace_id).json()
    assert len(spans) == 3
    assert spans[0]['parent_span_id'] is None
    assert spans[1]['parent_span_id'] == spans[0]['span_id']
    assert spans[2]['parent_span_id'] == spans[1]['span_id']


def test_any_span_filters_newest_first_and_limit(client_and_path):
    client, path = client_and_path
    old = seed_trace(path, cell_id='cell-a', start=100)
    new = seed_trace(path, cell_id='cell-b', start=1000)
    assert [row['trace_id'] for row in client.get('/api/traces').json()] == [new, old]
    assert client.get('/api/traces?limit=1').json()[0]['trace_id'] == new
    assert [row['trace_id'] for row in client.get('/api/traces?cell_id=cell-a').json()] == [old]
    assert len(client.get('/api/traces?colony=colony-a').json()) == 2
    assert client.get('/api/traces?colony=missing').json() == []
    assert client.get('/api/traces?limit=0').status_code == 422


def test_unknown_trace_returns_json_404(client_and_path):
    client, _ = client_and_path
    response = client.get('/api/traces/unknown')
    assert response.status_code == 404
    assert response.json() == {'detail': 'Trace not found.'}


def test_missing_and_pretelemetry_dishes_are_empty(client_and_path, tmp_path):
    client, path = client_and_path
    assert client.get('/api/traces').json() == []
    assert not path.exists()
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE legacy(value TEXT)')
    assert client.get('/api/traces').json() == []
    assert not (tmp_path / 'unused.sqlite').exists()


def test_routes_do_not_read_domain_events_or_colony_files(client_and_path, monkeypatch):
    client, path = client_and_path
    seed_trace(path)
    monkeypatch.setattr('petri.graph.colony.deserialize_colony', lambda *args: pytest.fail('colony read'))
    with closing(connect(path)) as connection:
        connection.execute('DROP TABLE events')
    assert len(client.get('/api/traces').json()) == 1

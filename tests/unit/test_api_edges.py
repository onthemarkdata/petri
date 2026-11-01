"""Dependency routes and CLI views share one deterministic report."""

import json

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typer.testing import CliRunner

from petri.analysis.edge_intelligence import EdgeIntelligenceReport
from petri.cli import app
from petri.dashboard.api import create_app
from petri.dashboard.frontend import build_frontend_html
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.presentation import EDGE_STYLES
from petri.storage.colonies import save_colony
from petri.storage.edge_registry import EdgeRecord, append
from petri.storage.paths import dish_db_path


@pytest.fixture
def cross_colony(seeded_petri_dir):
    env = seeded_petri_dir
    colony_id = env['dish_id'] + '-other'
    cell = Cell(id=colony_id + '-000-000', colony_id=colony_id, claim_text='Other claim', level=0)
    graph = ColonyGraph(colony_id)
    graph.add_cell(cell)
    path = dish_db_path(env['petri_dir'])
    save_colony(path, graph, Colony(id=colony_id, dish=env['dish_id'], center_claim=cell.claim_text, center_cell_id=cell.id))
    append(path, EdgeRecord(from_cell=cell.id, to_cell=env['cell2'].id, edge_type='cross_colony', reason='Shared premise', created_at='2026-11-01T00:00:00Z'))
    return env, cell


def test_typed_edges_report_and_cli_agree_without_legacy_reads(cross_colony, monkeypatch):
    env, other = cross_colony
    monkeypatch.setattr('petri.graph.colony.deserialize_colony', lambda *args: pytest.fail('Legacy read'))
    client = TestClient(create_app(env['petri_dir']))
    response = client.get('/api/edges')
    assert response.status_code == 200
    edges = [EdgeRecord.model_validate(row) for row in response.json()]
    assert len(edges) == 6
    assert len(client.get('/api/edges', params={'cell_id':other.id}).json()) == 1
    report = EdgeIntelligenceReport.model_validate(client.get('/api/intelligence').json())
    assert report.convergence_points == [env['cell2'].id]
    metric = report.cells[env['cell2'].id]
    checked = CliRunner().invoke(app, ['check', '--targets', '1'])
    assert checked.exit_code == 0, checked.output
    assert f"{metric.cell_id}  importance={metric.importance}  fragility={metric.fragility}" in checked.output
    text = CliRunner().invoke(app, ['graph']).output
    assert 'convergence point' in text and 'Across colonies:' in text
    dot = CliRunner().invoke(app, ['graph', '--format', 'dot']).output
    assert 'Convergence point' in dot and 'style=dashed, color=blue' in dot
    assert CliRunner().invoke(app, ['check', '--targets', '0']).exit_code != 0


def test_dependency_routes_emit_domain_spans(cross_colony, monkeypatch):
    env, other = cross_colony
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('petri.telemetry.get_tracer', lambda: provider.get_tracer('fixture'))
    client = TestClient(create_app(env['petri_dir']))
    assert client.get('/api/edges', params={'cell_id':other.id, 'colony':other.colony_id}).status_code == 200
    assert client.get('/api/intelligence').status_code == 200
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ['petri.dashboard.edges', 'petri.dashboard.intelligence']
    assert all(span.attributes['petri.dish_id'] == env['dish_id'] and span.attributes['petri.outcome'] == 'ok' for span in spans)
    assert spans[0].attributes['petri.cell_id'] == other.id
    assert spans[0].attributes['petri.colony_id'] == other.colony_id
    provider.shutdown()


def test_empty_read_creates_nothing(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/edges').json() == []
    assert client.get('/api/intelligence').json()['cells'] == {}
    assert list(tmp_path.iterdir()) == []


def test_frontend_receives_canonical_edge_styles_and_targets():
    html = build_frontend_html()
    assert 'var EDGE_STYLES = ' + json.dumps(EDGE_STYLES) in html
    assert "fetchJSON('/api/intelligence')" in html
    assert 'validation-targets-panel' in html
    assert "document.createTextNode(style.label)" in html

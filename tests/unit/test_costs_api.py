"""SQL usage rollups preserve missing token and price measurements."""

import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from petri.dashboard.api import create_app
from petri.dashboard.frontend import build_frontend_html
from petri.storage.database import connect
from petri.telemetry.span_store import aggregate_usage


@pytest.fixture
def client(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    with closing(connect(path)) as connection:
        for cell, colony in [('a','alpha'), ('b','alpha'), ('c','beta'), ('empty','alpha')]:
            connection.execute('INSERT INTO cells (id,colony,level,seq,claim_text,status,created_at) VALUES (?,?,0,0,?,\'NEW\',\'2026-01-01\')', (cell,colony,cell))
        for span, cell, phase, tokens_in, tokens_out, cost in [('1','a','research',10,5,0.1), ('2','a','critique',20,8,0.2), ('3','b','research',30,9,0.3), ('4','c','research',40,10,None)]:
            connection.execute("INSERT INTO spans VALUES (?,? ,NULL,'span','INTERNAL',1,2,'OK',NULL,'{}','{}')", (span,'trace'))
            connection.execute('INSERT INTO usage VALUES (?,?,?,?,?,?,?,?)', (span,'trace',cell,phase,'agent',tokens_in,tokens_out,cost))
        connection.execute("INSERT INTO spans VALUES ('no-usage','trace',NULL,'no usage','INTERNAL',1,2,'OK',NULL,'{}','{}')")
    return TestClient(create_app(tmp_path))


def test_totals_cells_phases_and_filters(client):
    response = client.get('/api/costs')
    assert response.status_code == 200
    usage = response.json()
    assert usage['total']['tokens_input'] == 100
    assert usage['total']['tokens_output'] == 32
    assert usage['total']['cost_usd'] is None
    cells = {row['cell_id']: row for row in usage['cells']}
    assert cells['a']['tokens_input'] == 30
    assert cells['a']['tokens_output'] == 13
    assert cells['a']['cost_usd'] == pytest.approx(0.3)
    assert cells['empty']['tokens_input'] is None
    assert cells['empty']['tokens_output'] is None
    assert cells['empty']['cost_usd'] is None
    phases = {(row['cell_id'],row['phase']):row for row in usage['phases']}
    assert phases['a','research']['tokens_input'] == 10
    assert phases['a','critique']['tokens_output'] == 8
    assert len(phases) == 4
    cell = client.get('/api/costs?cell_id=a').json()
    assert cell['total']['tokens_input'] == 30
    assert len(cell['cells']) == 1
    colony = client.get('/api/costs?colony=alpha').json()
    assert colony['total']['tokens_input'] == 60
    assert colony['total']['cost_usd'] == pytest.approx(0.6)
    assert len(colony['cells']) == 3


def test_no_store_or_premigration_store_returns_empty(tmp_path):
    client = TestClient(create_app(tmp_path))
    expected = {'total':{'cell_id':None,'phase':None,'tokens_input':None,'tokens_output':None,'cost_usd':None}, 'cells':[], 'phases':[]}
    assert client.get('/api/costs').json() == expected
    path = tmp_path / 'petri-dishes' / 'old' / 'petri.sqlite'
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE old_table(value TEXT)')
    assert client.get('/api/costs').json() == expected
    assert not (tmp_path / 'unused.sqlite').exists()


def test_frontend_has_dish_cell_and_phase_usage():
    html = build_frontend_html()
    assert 'id="costs-panel"' in html
    assert 'cell-costs-panel' in html
    assert '/api/costs' in html
    assert 'usage.phases.forEach' in html
    assert "value == null ? 'n/a'" in html


def test_grouping_is_validated(tmp_path):
    with pytest.raises(ValueError, match='group_by'):
        aggregate_usage(tmp_path, group_by='DROP TABLE cells')

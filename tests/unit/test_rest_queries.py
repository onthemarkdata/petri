"""Dashboard REST reads use per-dish SQLite without a disposable index."""

import itertools
import json
import sqlite3
from contextlib import closing
from importlib.resources import files

import pytest
from fastapi.testclient import TestClient

from petri import query
from petri.dashboard.api import create_app
from petri.storage import database


@pytest.fixture
def rest_env(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    with closing(database.connect(path)) as connection:
        for cell, colony, level in [('parent','alpha',0),('child','beta',1)]:
            connection.execute('INSERT INTO cells (id,colony,level,seq,claim_text,status,created_at) VALUES (?,?,?,0,?,\'NEW\',\'2026-01-01\')', (cell,colony,level,cell+' claim'))
        connection.execute("INSERT INTO edges VALUES ('parent','child','cross_colony','Shared premise','2026-01-01')")
        for index, (cell, kind, agent, iteration, payload, timestamp) in enumerate([
            ('parent','evidence_appended','investigator',1,{'summary':'Evidence one'},'2026-01-01'),
            ('parent','verdict_issued','cell_lead',1,{'summary':'Final summary','verdict':'VALIDATED'},'2026-01-02'),
            ('child','source_reviewed','investigator',1,{'url':'https://example.com'},'2026-01-02'),
            ('parent','evidence_appended','investigator',2,{'summary':'Evidence two'},'2026-01-02'),
        ]):
            connection.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', (str(index),cell,kind,agent,iteration,json.dumps(payload),timestamp))
    (tmp_path / 'queue.json').write_text(json.dumps({'entries':{'parent':{'queue_state':'queued'}}}))
    legacy = tmp_path / 'petri-dishes' / 'old-colony'
    legacy.mkdir()
    (legacy / 'colony.json').write_text('invalid legacy colony')
    (tmp_path / 'petri.sqlite').write_bytes(b'stale index must remain untouched')
    (tmp_path / 'combined.jsonl').write_bytes(b'stale rollup must remain untouched')
    return TestClient(create_app(tmp_path)), path


def test_all_rest_filters_match_query_layer(rest_env):
    client, path = rest_env
    for cell, agent, iteration, kind in itertools.product([None,'parent'],[None,'investigator'],[None,1],[None,'evidence_appended']):
        params = {k:v for k,v in dict(cell_id=cell,agent=agent,iteration=iteration,event_type=kind).items() if v is not None}
        response = client.get('/api/events',params=params)
        assert response.status_code == 200
        assert response.json() == query.query_events(path, **params)
    assert len(client.get('/api/events?limit=1').json()) == 1


def test_cells_detail_edges_and_legacy_exclusion(rest_env, monkeypatch):
    client, path = rest_env
    monkeypatch.setattr('petri.cli._bootstrap.load_colonies', lambda *args: pytest.fail('colony assembly'))
    monkeypatch.setattr('petri.graph.colony.deserialize_colony', lambda *args: pytest.fail('legacy read'))
    cells = client.get('/api/cells').json()
    assert len(cells) == 2
    assert set(cells[0]) == {'cell_id','colony_id','claim_text','level','status','dependencies','dependents'}
    parent = client.get('/api/cell/parent').json()
    assert parent['dependencies'] == ['child']
    assert parent['evidence_md'] == 'Evidence one\n\nEvidence two'
    assert parent['summary_md'] == 'Final summary'
    assert parent['events'] == query.query_events(path,cell_id='parent')
    assert client.get('/api/cell/unknown').status_code == 404
    assert client.get('/api/edges').json() == query.get_edges(path)
    assert client.get('/api/edges?cell_id=missing').json() == []
    assert [dish['id'] for dish in client.get('/api/dishes').json()] == ['dish']
    root = path.parents[2]
    assert (root / 'petri.sqlite').read_bytes() == b'stale index must remain untouched'
    assert (root / 'combined.jsonl').read_bytes() == b'stale rollup must remain untouched'


def test_stats_preserve_event_and_queue_contracts(rest_env):
    client, path = rest_env
    stats = client.get('/api/stats').json()
    for key, value in query.event_stats(path).items():
        assert stats[key] == value
    assert stats['queue_size'] == 1
    assert stats['cells_by_state'] == {'queued':1}


def test_no_database_returns_empty_contracts(tmp_path):
    client = TestClient(create_app(tmp_path))
    for endpoint in ['events','cells','edges','dishes']:
        assert client.get('/api/'+endpoint).json() == []
    assert client.get('/api/stats').json()['total_events'] == 0
    assert not (tmp_path / 'petri.sqlite').exists()


def test_content_migration_backfills_and_tracks_only_new_events(tmp_path):
    path = tmp_path / 'petri.sqlite'
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        source = files('petri.storage.migrations')
        database.migrate(connection, [source.joinpath(name).read_text() for name in database.MIGRATIONS[:4]])
        connection.execute("INSERT INTO cells (id,colony,level,seq,claim_text,status,created_at) VALUES ('cell','colony',0,0,'Claim','NEW','2026-01-01')")
        connection.execute("INSERT INTO events VALUES ('e1','cell','evidence_appended','investigator',1,'{\"summary\":\"Before migration\"}','2026-01-01')")
    with closing(database.connect(path)) as connection:
        assert connection.execute('SELECT evidence_md FROM cells').fetchone()[0] == 'Before migration'
        connection.execute("INSERT OR IGNORE INTO events VALUES ('e1','cell','evidence_appended','investigator',1,'{\"summary\":\"Duplicate\"}','2026-01-01')")
        connection.execute("INSERT INTO events VALUES ('e2','cell','verdict_issued','cell_lead',1,'{\"summary\":\"Latest\"}','2026-01-02')")
        assert connection.execute('SELECT evidence_md,summary_md FROM cells').fetchone()[:] == ('Before migration','Latest')
        assert connection.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute("DELETE FROM events WHERE id='e1'")

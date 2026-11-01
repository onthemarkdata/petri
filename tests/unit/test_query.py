"""Read-only query contracts over per-dish domain databases."""

import itertools
import json
import sqlite3
from contextlib import closing

import pytest

from petri import query
from petri.storage import database


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    with closing(database.connect(path)) as connection:
        for cell, colony, level in [('parent', 'alpha', 0), ('child', 'beta', 1)]:
            connection.execute(
                'INSERT INTO cells (id,colony,level,seq,claim_text,status,created_at) VALUES (?,?,?,?,?,?,?)',
                (cell, colony, level, 0, cell + ' claim', 'NEW', '2026-01-01'),
            )
        connection.execute("INSERT INTO edges VALUES ('parent','child','depends_on','premise','2026-01-01')")
        for index, (cell, kind, agent, iteration, payload, timestamp) in enumerate([
            ('parent', 'verdict_issued', 'skeptic', 1, {'verdict': 'VALIDATED', 'summary': 'Supported'}, '2026-01-02'),
            ('child', 'source_reviewed', 'investigator', 1, {'url': 'https://example.com/a'}, '2026-01-01'),
            ('parent', 'source_reviewed', 'investigator', 2, {'url': 'https://example.com/a'}, '2026-01-02'),
            ('parent', 'source_reviewed', 'investigator', 2, {'url': 'https://example.com/b'}, '2026-01-02'),
        ]):
            connection.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', (str(index), cell, kind, agent, iteration, json.dumps(payload), timestamp))
    return path


def test_missing_database_creates_nothing(tmp_path):
    path = tmp_path / 'absent' / 'petri.sqlite'
    assert query.query_events(path) == []
    assert query.events_since(path, 0) == []
    assert query.list_cells(path) == []
    assert query.get_cell(path, 'absent') is None
    assert query.get_edges(path) == []
    assert query.list_colonies(path) == []
    assert query.event_stats(path)['total_events'] == 0
    assert list(tmp_path.iterdir()) == []


def test_connections_reject_writes(db_path):
    with query.read_connection(db_path) as connection:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute("INSERT INTO cells (id) VALUES ('forbidden')")


def test_enumeration_excludes_legacy_backups_exports_and_symlinks(db_path):
    root = db_path.parents[2]
    for relative in ['petri-dishes/legacy/cell/events.jsonl', 'backups/snapshot/petri.sqlite', 'exports/dish/petri.sqlite', 'petri-dishes/backups/petri.sqlite', 'petri-dishes/exports/petri.sqlite', 'petri-dishes/.hidden/petri.sqlite']:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / 'petri-dishes' / 'alias').symlink_to(db_path.parent, target_is_directory=True)
    assert query.enumerate_dishes(root) == [query.DishDatabase('dish', db_path)]


def test_event_filter_combinations_match_envelopes(db_path):
    all_rows = query.query_events(db_path, limit=None)
    for cell, agent, iteration, kind in itertools.product([None, 'parent'], [None, 'investigator'], [None, 2], [None, 'source_reviewed']):
        filters = dict(cell_id=cell, agent=agent, iteration=iteration, event_type=kind)
        expected = [r for r in all_rows if (cell is None or r['cell_id'] == cell) and (agent is None or r['agent'] == agent) and (iteration is None or r['iteration'] == iteration) and (kind is None or r['type'] == kind)]
        assert query.query_events(db_path, **filters) == expected
    assert [r['id'] for r in all_rows] == ['1', '0', '2', '3']
    assert [r['id'] for r in query.query_events(db_path, since='2026-01-02', limit=2)] == ['0', '2']
    assert set(all_rows[0]) == {'id', 'cell_id', 'timestamp', 'type', 'agent', 'iteration', 'data'}


def test_rowid_cursor_resumes_across_timestamp_ties(db_path):
    first = query.events_since(db_path, 0, limit=2)
    second = query.events_since(db_path, first[-1]['rowid'], limit=2)
    assert [r['id'] for r in first + second] == ['0', '1', '2', '3']
    assert query.events_since(db_path, second[-1]['rowid']) == []


@pytest.mark.parametrize('limit', [0, -1, 5001, True])
def test_invalid_limit_rejected(db_path, limit):
    with pytest.raises(ValueError):
        query.query_events(db_path, limit=limit)


def test_cells_edges_verdicts_sources_and_stats(db_path):
    parent, child = query.list_cells(db_path)
    assert parent['dependencies'] == ['child']
    assert parent['dependents'] == []
    assert child['dependents'] == ['parent']
    assert parent['created_at']
    assert query.get_cell(db_path, 'parent') == parent
    assert query.list_cells(db_path, colony='beta', level=1, status='NEW') == [child]
    assert query.get_edges(db_path, cell_id='child', colony='alpha')[0]['reason'] == 'premise'
    assert query.get_edges(db_path, cell_id='missing') == []
    assert query.get_verdicts(db_path, cell_id='parent')[0].verdict == 'VALIDATED'
    assert [r['id'] for r in query.get_sources(db_path)] == ['1', '3']
    assert [r['id'] for r in query.get_sources(db_path, cell_id='parent')] == ['2', '3']
    stats = query.event_stats(db_path)
    assert stats['total_events'] == 4
    assert stats['cells_with_events'] == 2
    assert stats['events_by_type'][0] == {'type': 'source_reviewed', 'count': 3}
    assert stats['top_cells'][0] == {'cell_id': 'parent', 'event_count': 3}


def test_wal_snapshot_reads_only_committed_rows_during_writer(db_path):
    with closing(database.connect(db_path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        writer.execute("INSERT INTO events VALUES ('pending','parent','search_executed','investigator',1,'{}','2026-01-02')")
        assert len(query.query_events(db_path)) == 4
        writer.commit()
        assert len(query.query_events(db_path)) == 5
    assert not (db_path.parents[2] / 'combined.jsonl').exists()

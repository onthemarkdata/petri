"""Research-health SQL views match known event histories."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from importlib.resources import files

import pytest

from petri.query.analytics import blocking_verdict_patterns, cell_velocity, convergence_rates, stalled_cells
from petri.storage import database


@pytest.fixture
def path(tmp_path):
    path = tmp_path / 'petri.sqlite'
    with closing(database.connect(path)) as connection:
        for cell,colony in [('converged','alpha'),('old','alpha'),('fresh','beta'),('no-events','alpha')]:
            connection.execute('INSERT INTO cells (id,colony,level,seq,claim_text,status,created_at) VALUES (?,?,1,0,?,\'NEW\',\'2020-01-01\')',(cell,colony,cell))
        history = [
            ('converged','search_executed','investigator',{},'2026-01-01T00:00:00Z'),
            ('converged','convergence_checked','cell_lead',{'converged':True},'2026-01-01T01:00:00Z'),
            ('old','verdict_issued','skeptic',{'verdict':'CRITICAL_FLAW_FOUND'},'2026-01-01T00:00:00Z'),
            ('old','verdict_issued','skeptic',{'verdict':'UNADDRESSED_COUNTERARGUMENT'},'2026-01-01T01:00:00Z'),
            ('old','verdict_issued','champion',{'verdict':'STRONG_CASE'},'2026-01-01T02:00:00Z'),
            ('fresh','verdict_issued','skeptic',{'verdict':'CRITICAL_FLAW_FOUND'},'2026-01-03T11:00:00Z'),
            ('fresh','verdict_issued','simplifier',{'verdict':'OVERCOMPLICATED'},'2026-01-03T11:00:00Z'),
        ]
        for index,(cell,kind,agent,payload,time) in enumerate(history):
            connection.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',(str(index),cell,kind,agent,1,json.dumps(payload),time))
    return path


def test_blocking_patterns_count_only_nonpassing_blocking_roles(path):
    rows = blocking_verdict_patterns(path)
    assert rows[0] == {'colony':'alpha','level':1,'agent':'skeptic','blocked_verdicts':2,'blocked_cells':1,'cell_ids':['old'],'last_blocked_at':'2026-01-01T01:00:00Z'}
    assert len(rows) == 2
    assert rows[1]['cell_ids'] == ['fresh']


def test_velocity_convergence_and_stalled_window(path):
    rows = {row['cell_id']:row for row in cell_velocity(path)}
    assert rows['converged']['seconds_to_convergence'] == pytest.approx(3600,abs=0.001)
    assert rows['old']['seconds_to_convergence'] is None
    assert 'no-events' not in rows
    old = stalled_cells(path,window_hours=24,now=datetime(2026,1,3,12,tzinfo=timezone.utc))
    assert [row['cell_id'] for row in old] == ['old']
    rates = {row['colony']:row for row in convergence_rates(path)}
    assert rates['alpha']['total_cells'] == 3
    assert rates['alpha']['converged_cells'] == 1
    assert rates['alpha']['convergence_rate'] == pytest.approx(1/3)
    assert rates['beta']['convergence_rate'] == 0


def test_empty_or_missing_database_yields_no_rows(tmp_path):
    path = tmp_path / 'petri.sqlite'
    for reader in [blocking_verdict_patterns,cell_velocity,stalled_cells,convergence_rates]:
        assert reader(path) == []
    assert not path.exists()
    with closing(database.connect(path)):
        pass
    for reader in [blocking_verdict_patterns,cell_velocity,stalled_cells,convergence_rates]:
        assert reader(path) == []


def test_migration_upgrades_once_and_reopens(tmp_path):
    path = tmp_path / 'petri.sqlite'
    source = files('petri.storage.migrations')
    with sqlite3.connect(path,isolation_level=None) as connection:
        database.migrate(connection,[source.joinpath(name).read_text() for name in database.MIGRATIONS[:5]])
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 5
    for _ in range(2):
        with closing(database.connect(path)) as connection:
            assert connection.execute('PRAGMA user_version').fetchone()[0] == database.SCHEMA_VERSION
            assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='cell_velocity'").fetchone()[0] == 1

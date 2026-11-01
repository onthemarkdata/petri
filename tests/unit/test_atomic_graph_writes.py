"""Graph and event writes can share one caller-owned transaction."""

import sqlite3
from contextlib import closing

import pytest

from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony, Event
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.edge_registry import EdgeRecord


def topology():
    graph = ColonyGraph('dish-alpha')
    cell = Cell(id='dish-alpha-000-000',colony_id='dish-alpha',level=0,claim_text='Claim')
    graph.add_cell(cell)
    colony = Colony(id='dish-alpha',dish='dish',center_cell_id=cell.id,center_claim=cell.claim_text)
    event = Event(id='atomic-event',cell_id=cell.id,type='search_executed',agent='investigator',
                  iteration=1,timestamp='2026-01-01',data={'query':'fixture','sources_found':0})
    return graph,colony,event


def test_caller_rollback_removes_all_materialized_rows(tmp_path):
    path = tmp_path / 'petri.sqlite'
    graph,colony,event = topology()
    with pytest.raises(RuntimeError,match='rollback'):
        with database.write_connection(path) as connection:
            save_colony(path,graph,colony,connection=connection)
            database.append_event(path,event,connection=connection)
            assert connection.in_transaction
            raise RuntimeError('rollback')
    with closing(database.connect(path)) as connection:
        for table in ['cells','colonies','edges','events']:
            assert connection.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] == 0


def test_caller_commit_keeps_rows_and_duplicate_event_is_idempotent(tmp_path):
    path = tmp_path / 'petri.sqlite'
    graph,colony,event = topology()
    with database.write_connection(path) as connection:
        save_colony(path,graph,colony,connection=connection)
        assert database.append_event(path,event,connection=connection)[1]
        assert not database.append_event(path,event,connection=connection)[1]
        assert connection.in_transaction
    assert database.get_cell(path,event.cell_id) is not None
    assert len(database.read_events(path)) == 1


def test_default_colony_save_rolls_back_after_invalid_external_edge(tmp_path):
    path = tmp_path / 'petri.sqlite'
    graph,colony,event = topology()
    graph.add_edge(EdgeRecord(from_cell=event.cell_id,to_cell='dish-missing-000-000',
        edge_type='cross_colony',created_at='2026-01-01'))
    with pytest.raises(sqlite3.IntegrityError,match='FOREIGN KEY'):
        save_colony(path,graph,colony)
    assert database.list_cells(path) == []
    with closing(database.connect(path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM colonies').fetchone()[0] == 0


def test_supplied_connection_must_have_an_active_transaction(tmp_path):
    path = tmp_path / 'petri.sqlite'
    graph,colony,_ = topology()
    with closing(database.connect(path)) as connection:
        with pytest.raises(ValueError,match='active writer transaction'):
            save_colony(path,graph,colony,connection=connection)

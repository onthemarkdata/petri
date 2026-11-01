"""Canonical topology survives persistence, removals, and invalid neighbors."""

import random
import sqlite3
from contextlib import closing

import pytest

from petri import query
from petri.graph.colony import ColonyGraph, GraphIntegrityError
from petri.models import Cell, Colony, Edge
from petri.storage import database
from petri.storage.colonies import list_colonies, load_colony, save_colony
from petri.storage.edge_registry import EdgeRecord


def cell(colony,index):
    return Cell(id=f'{colony}-001-{index:03d}',colony_id=colony,claim_text=str(index),level=1,
                created_at='2026-01-01')


def metadata(colony,center):
    return Colony(id=colony,dish='dish',center_claim=center.claim_text,center_cell_id=center.id,
                  created_at='2026-01-01')


@pytest.fixture
def topology(tmp_path):
    path = tmp_path / 'dish/petri.sqlite'
    other = ColonyGraph('dish-beta')
    b = cell('dish-beta',0)
    other.add_cell(b)
    save_colony(path,other,metadata('dish-beta',b))
    graph = ColonyGraph('dish-alpha')
    a = [cell('dish-alpha',index) for index in range(3)]
    for item in a:
        graph.add_cell(item)
    for source,target,kind in [(a[0].id,a[1].id,'intra_colony'),(a[1].id,a[2].id,'intra_colony'),(a[2].id,b.id,'cross_colony')]:
        graph.add_edge(EdgeRecord(from_cell=source,to_cell=target,edge_type=kind,
            reason='Original reason',created_at='2026-01-01'))
    colony = metadata('dish-alpha',a[0])
    save_colony(path,graph,colony)
    return path,graph,colony,a,b


def dump(path):
    with closing(database.connect(path)) as connection:
        return {table:[tuple(row) for row in connection.execute('SELECT * FROM '+table+' ORDER BY rowid')]
                for table in ['cells','edges','colonies','edge_tombstones','cell_tombstones']}


def assert_views_agree(graph):
    for item in graph.get_all_cells():
        assert item.dependencies == graph.get_dependencies(item.id)
        assert item.dependents == graph.get_dependents(item.id)


def test_save_load_save_fixpoint_keeps_cross_edges_and_metadata(topology):
    path,graph,colony,a,b = topology
    before = dump(path)
    loaded,meta = load_colony(path,colony.id)
    assert_views_agree(loaded)
    assert loaded.get_cell(a[2].id).dependencies == [b.id]
    assert loaded.external_references[b.id].colony_id == 'dish-beta'
    assert loaded.get_edge_records()[-1].reason == 'Original reason'
    assert loaded.validate_dag()
    save_colony(path,loaded,meta)
    assert dump(path) == before
    beta,_ = load_colony(path,'dish-beta')
    assert beta.get_cell(b.id).dependents == [a[2].id]


def test_missing_local_endpoint_is_typed_and_duplicate_edges_are_one(topology):
    path,graph,colony,a,_ = topology
    with pytest.raises(GraphIntegrityError,match='Missing local cell'):
        graph.add_edge(Edge(from_cell=a[0].id,to_cell='dish-alpha-001-999'))
    graph.add_edge(Edge(from_cell=a[0].id,to_cell=a[1].id))
    graph.add_edge(Edge(from_cell=a[0].id,to_cell=a[1].id))
    save_colony(path,graph,colony)
    assert len(graph.get_edges()) == 3
    with closing(database.connect(path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM active_edges').fetchone()[0] == 3


def test_remove_cell_retires_active_rows_without_dangling_references(topology):
    path,graph,colony,a,_ = topology
    before = dump(path)
    graph.remove_cell(a[1].id)
    assert_views_agree(graph)
    save_colony(path,graph,colony)
    loaded,_ = load_colony(path,colony.id)
    assert_views_agree(loaded)
    assert query.get_cell(path,a[1].id) is None
    assert a[1].id not in {item.id for item in loaded.get_all_cells()}
    with closing(database.connect(path)) as connection:
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
        assert connection.execute('SELECT * FROM dependents WHERE cell_id=?',(a[2].id,)).fetchall() == []
        assert connection.execute('SELECT * FROM active_edges WHERE from_cell NOT IN (SELECT id FROM active_cells) OR to_cell NOT IN (SELECT id FROM active_cells)').fetchall() == []
    after = dump(path)
    assert after['edges'] == before['edges']
    assert after['cells'] == before['cells']
    assert len(after['edge_tombstones']) == 2
    assert len(after['cell_tombstones']) == 1


def test_arbitrary_add_remove_sequences_keep_cell_projections_consistent():
    rng = random.Random(82)
    graph = ColonyGraph('dish-sequence')
    for index in range(12):
        graph.add_cell(cell('dish-sequence',index))
    for _ in range(100):
        items = graph.get_all_cells()
        if len(items) < 2:
            break
        first,second = sorted(rng.sample(items,2),key=lambda item:item.id)
        if rng.random() < 0.2:
            graph.remove_cell(second.id)
        else:
            graph.add_edge(Edge(from_cell=first.id,to_cell=second.id))
        assert_views_agree(graph)
        assert graph.validate_dag()


def test_loader_reports_invalid_colony_and_keeps_healthy_colonies(topology,caplog):
    path,_,colony,a,_ = topology
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA foreign_keys=OFF')
        connection.execute('INSERT INTO edges VALUES (?,?,?,?,?)',(a[0].id,'dish-missing-001-999','cross_colony','','2026-01-02'))
        connection.commit()
    loaded = list_colonies(path)
    assert [meta.id for _,meta in loaded] == ['dish-beta']
    assert 'dish dish' in caplog.text
    assert colony.id in caplog.text
    assert 'missing dish cell' in caplog.text


def test_loader_reports_invalid_cell_model_without_hiding_healthy_colony(topology,caplog):
    path,_,colony,a,_ = topology
    with closing(database.connect(path)) as connection:
        connection.execute("UPDATE cells SET status='INVALID' WHERE id=?",(a[0].id,))
    assert [meta.id for _,meta in list_colonies(path)] == ['dish-beta']
    assert colony.id in caplog.text
    assert 'validation error' in caplog.text

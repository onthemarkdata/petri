"""Dish-wide dependencies preserve immutable history and reject cycles."""

from contextlib import closing

import pytest
from typer.testing import CliRunner

from petri import query
from petri.cli import app
from petri.storage import database, edge_registry


@pytest.fixture
def dish(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    petri = tmp_path / '.petri'
    config = petri / 'defaults/petri.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('name: dish\n')
    path = petri / 'petri-dishes/dish/petri.sqlite'
    ids = ['dish-alpha-000-000','dish-beta-000-000','dish-beta-001-000']
    for index,cell_id in enumerate(ids):
        database.put_cell(path,{'id':cell_id,'colony':cell_id.rsplit('-',2)[0],
            'level':index,'seq':0,'claim_text':cell_id,'created_at':'2026-01-01'})
    return path,ids


def edge(source,target,created_at='2026-01-01'):
    return edge_registry.EdgeRecord(from_cell=source,to_cell=target,edge_type='cross_colony',
                                    created_at=created_at,reason='Shared dependency')


def test_cross_colony_reachability_and_tombstone_history(dish):
    path,(a,b,c) = dish
    assert edge_registry.append(path,edge(a,b))
    assert edge_registry.append(path,edge(b,c))
    graph = edge_registry.DishGraph.read(path)
    assert graph.dependencies(a) == [b]
    assert graph.dependents(b) == [a]
    assert graph.reachable(a) == [b,c]
    assert graph.reachable(c,reverse=True) == [a,b]
    with closing(database.connect(path)) as connection:
        before = [tuple(row) for row in connection.execute('SELECT * FROM edges ORDER BY rowid')]
    tombstones = edge_registry.remove(path,from_cell=a,to_cell=b,reason='No longer required')
    assert len(tombstones) == 1
    assert isinstance(tombstones[0],edge_registry.EdgeTombstone)
    assert edge_registry.read_tombstones(path) == tombstones
    assert edge_registry.DishGraph.read(path).dependencies(a) == []
    assert query.get_cell(path,b)['dependents'] == []
    assert len(query.get_edges(path)) == 1
    assert edge_registry.remove(path,from_cell=a,to_cell=b) == []
    with closing(database.connect(path)) as connection:
        assert [tuple(row) for row in connection.execute('SELECT * FROM edges ORDER BY rowid')] == before
        with pytest.raises(Exception,match='append-only'):
            connection.execute('DELETE FROM edge_tombstones')


def test_readding_after_removal_appends_a_new_version(dish):
    path,(a,b,_) = dish
    first = edge(a,b)
    assert edge_registry.append(path,first)
    edge_registry.remove(path,from_cell=a,to_cell=b)
    assert not edge_registry.append(path,first)
    assert edge_registry.append(path,edge(a,b,created_at='2026-01-02'))
    assert edge_registry.DishGraph.read(path).dependencies(a) == [b]
    with closing(database.connect(path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM edges').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM edge_tombstones').fetchone()[0] == 1


def test_dish_graph_deduplicates_duplicate_rows(dish):
    path,(a,b,_) = dish
    assert edge_registry.append(path,edge(a,b))
    with closing(database.connect(path)) as connection:
        connection.execute('INSERT INTO edges VALUES (?,?,?,?,?)',(a,b,'cross_colony','duplicate','2026-01-02'))
    graph = edge_registry.DishGraph.read(path)
    assert graph.dependencies(a) == [b]
    assert len(graph.edges) == 1
    assert len(edge_registry.remove(path,from_cell=a,to_cell=b)) == 2
    assert edge_registry.DishGraph.read(path).dependencies(a) == []


def test_registry_and_connect_reject_cross_colony_cycles(dish):
    path,(a,b,c) = dish
    runner = CliRunner()
    result = runner.invoke(app,['connect',a,b])
    assert result.exit_code == 0,result.output
    assert edge_registry.append(path,edge(b,c))
    with pytest.raises(edge_registry.EdgeCycleError,match='dish-level cycle'):
        edge_registry.append(path,edge(c,a))
    result = runner.invoke(app,['connect',c,a])
    assert result.exit_code != 0
    assert 'dish-level cycle' in result.output
    assert len(edge_registry.read(path)) == 2


def test_connect_missing_arguments_in_non_tty_prints_usage(dish,monkeypatch):
    monkeypatch.setattr('petri.cli.connect.detect_interactive_mode',lambda:(False,None))
    result = CliRunner().invoke(app,['connect'])
    assert result.exit_code != 0
    assert 'Usage: petri connect CELL1 CELL2' in result.output


def test_tombstone_identity_survives_database_maintenance(dish):
    path,(a,b,_) = dish
    edge_registry.append(path,edge(a,b))
    edge_registry.remove(path,from_cell=a,to_cell=b)
    with closing(database.connect(path)) as connection:
        connection.execute('VACUUM')
    assert edge_registry.DishGraph.read(path).dependencies(a) == []

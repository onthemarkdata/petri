"""Hand-computed graph metrics over a diamond with a second colony."""

import ast
from pathlib import Path

from petri.analysis.edge_intelligence import EdgeIntelligenceReport, analyze_edges, convergence_points, fragility, importance
from petri.storage.edge_registry import DishGraph, EdgeRecord


def graph(include_cross=True):
    cells = [{'id':name,'colony':'alpha' if name!='other' else 'beta','level':level}
             for name,level in [('root',0),('left',1),('right',1),('shared',2),('leaf',3),('other',0)]]
    pairs = [('root','left'),('root','right'),('left','shared'),('right','shared'),('shared','leaf')]
    if include_cross:
        pairs.append(('other','shared'))
    return DishGraph(cells,[EdgeRecord(from_cell=a,to_cell=b,edge_type='cross_colony' if a=='other' else 'intra_colony',created_at='2026-01-01') for a,b in pairs])


def test_diamond_metrics_match_hand_computed_values():
    dish = graph()
    assert importance(dish,'shared') == (3,1)
    assert importance(dish,'leaf') == (1,0)
    assert fragility(dish,'shared') == (3,4,7)
    assert fragility(dish,'leaf') == (3,5,8)
    assert fragility(dish,'root') == (3,0,3)
    report = analyze_edges(dish)
    assert report.convergence_points == ['shared']
    assert report.importance_ranking[0] == 'shared'
    assert report.fragility_ranking[0] == 'leaf'
    assert report.cells['shared'].depending_colonies == ('alpha','beta')


def test_convergence_point_disappears_when_cross_reference_is_removed():
    assert [item.cell_id for item in convergence_points(graph())] == ['shared']
    assert convergence_points(graph(include_cross=False)) == []


def test_directionality_and_model_roundtrip():
    report = analyze_edges(graph())
    assert report.cells['leaf'].fragility > report.cells['left'].fragility
    assert EdgeIntelligenceReport.model_validate(report.model_dump(mode='json')) == report
    empty = analyze_edges(DishGraph([],[]))
    assert empty.cells == {} and empty.convergence_points == []


def test_module_has_no_engine_reasoning_dashboard_or_dbos_imports():
    source = Path(__file__).parents[2] / 'petri/analysis/edge_intelligence.py'
    tree = ast.parse(source.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            modules.append(node.module or '')
    assert not any(name.startswith(('petri.engine','petri.reasoning','petri.dashboard','dbos','pydantic_ai')) for name in modules)


def test_equal_degree_leaf_has_more_fragility_than_an_independent_root():
    cells = [{'id':name,'colony':'alpha','level':0} for name in ['a','b','x','y']]
    dish = DishGraph(cells,[EdgeRecord(from_cell=a,to_cell=b,created_at='2026-01-01') for a,b in [('a','b'),('x','y')]])
    assert len(dish.dependencies('b')) + len(dish.dependents('b')) == 1
    assert len(dish.dependencies('x')) + len(dish.dependents('x')) == 1
    assert fragility(dish,'b')[2] > fragility(dish,'x')[2]

"""Priority plans preserve defaults and one computation per grow pass."""

from types import SimpleNamespace

import pytest

from petri.execution.config import SchedulingConfig
from petri.execution.priorities import assigned_priority, prepare_priority_pass, priority_plan
from tests.unit.test_edge_intelligence import graph
from petri.analysis.edge_intelligence import analyze_edges


def test_opt_in_plan_ranks_convergence_then_importance_then_level():
    report = analyze_edges(graph())
    plan = priority_plan(report)
    assert plan['shared']==1
    rest = [cell for cell in sorted(plan,key=plan.get) if cell!='shared']
    assert rest == sorted(rest,key=lambda cell:(-report.cells[cell].importance,report.cells[cell].level,cell))


def test_absent_flag_preserves_level_derived_priorities(tmp_path):
    app = SimpleNamespace(petri_dir=tmp_path,target_priority_plans={})
    assert SchedulingConfig().prioritize_targets is False
    for level in [0,1,9]:
        assert assigned_priority(app,f'dish-alpha-{level:03}-001',level)==level+1
    with pytest.raises(ValueError):
        SchedulingConfig(prioritize_targets='false')


def test_enabled_pass_computes_once_and_replay_can_rebuild(tmp_path,monkeypatch):
    (tmp_path/'petri.yaml').write_text('scheduling:\n  prioritize_targets: true\n')
    calls = []
    monkeypatch.setattr('petri.execution.priorities.DishGraph.read',lambda path:graph())
    def analyze(value):
        calls.append(value)
        return analyze_edges(value)
    monkeypatch.setattr('petri.execution.priorities.analyze_edges',analyze)
    app = SimpleNamespace(petri_dir=tmp_path,db_path=tmp_path/'unused',target_priority_plans={})
    prepare_priority_pass(app,'dish-alpha')
    assert assigned_priority(app,'dish-alpha-001-001',1)==2
    assert assigned_priority(app,'dish-alpha-001-002',1)==2
    assert len(calls)==1
    app.target_priority_plans.clear()
    assigned_priority(app,'dish-alpha-001-001',1)
    assert len(calls)==2

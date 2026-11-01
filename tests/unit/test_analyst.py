"""Analyst boundaries, timestamp evidence, and append-only storage."""

import ast
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from petri.analysis.analyst import AnalystConfig, HealthSnapshot, ResearchHealthReport, analyze_health
from petri.analysis.edge_intelligence import analyze_edges
from petri.analysis.health import run_analyst
from petri.cli import app
from petri.dashboard.api import create_app
from petri.dashboard.frontend import build_frontend_html
from petri.models import Event
from petri.query.health import health_snapshot, latest_health
from petri.storage import database
from petri.storage.analyst import append_report
from petri.storage.edge_registry import DishGraph, EdgeRecord
from petri.storage.paths import dish_db_path

NOW = datetime(2026,11,1,tzinfo=timezone.utc)


def snapshot(count=5):
    cells = [{'id':str(i),'colony':'alpha','level':0,'status':'NEW','created_at':'1900-01-01'} for i in range(count)]
    graph = DishGraph(cells,[])
    return HealthSnapshot(cells=cells,events=[],blocking_patterns=[{'agent':'skeptic','cell_ids':list(graph.cells)}],
                          intelligence=analyze_edges(graph),dependencies={})


def rules(report):
    return {flag.rule:flag for flag in report.flags}


def analyze(inputs, config=None):
    return analyze_health(inputs,report_id='fixture',dish_id='dish',now=NOW,config=config)


def test_systemic_agent_boundary_and_distinct_cell_count():
    assert rules(analyze(snapshot(5)))['systemic_agent_block'].affected_agents == ['skeptic']
    assert rules(analyze(snapshot(5)))['systemic_agent_block'].affected_cells == ['0','1','2','3','4']
    assert 'systemic_agent_block' not in rules(analyze(snapshot(4)))
    inputs = snapshot(4)
    inputs.blocking_patterns *= 2
    assert 'systemic_agent_block' not in rules(analyze(inputs))


def test_verdict_velocity_uses_eligible_cells_and_recent_verdicts():
    inputs = snapshot(2)
    inputs.dependencies = {'0':['1']}
    assert rules(analyze(inputs))['stalled_verdict_velocity'].affected_cells == ['1']
    inputs.events = [{'cell_id':'1','type':'verdict_issued','timestamp':'2026-10-31T23:00:00Z'}]
    assert 'stalled_verdict_velocity' not in rules(analyze(inputs))
    inputs.events[0]['timestamp'] = '2026-10-30T00:00:00Z'
    assert 'stalled_verdict_velocity' in rules(analyze(inputs))


def test_staleness_uses_transitive_dependents_and_event_age_never_cell_creation():
    inputs = snapshot(4)
    graph = DishGraph(inputs.cells,[EdgeRecord(from_cell=str(i),to_cell=str(i+1),created_at='2026-01-01') for i in range(3)])
    inputs.intelligence = analyze_edges(graph)
    inputs.events = [{'cell_id':'3','type':'source_reviewed','timestamp':'2026-10-01T00:00:00Z'}]
    flag = rules(analyze(inputs))['transitive_staleness']
    assert flag.affected_cells == ['3'] and flag.evidence['transitive_dependents'] == 3
    inputs.events[0]['timestamp'] = '2026-10-31T23:00:00Z'
    assert 'transitive_staleness' not in rules(analyze(inputs))
    inputs.events = []
    assert 'transitive_staleness' not in rules(analyze(inputs))


def test_reports_append_without_domain_writes_and_feed_cli_api(seeded_petri_dir):
    env = seeded_petri_dir
    path = dish_db_path(env['petri_dir'])
    with closing(database.connect(path)) as conn:
        before = {table:list(conn.execute('SELECT * FROM '+table)) for table in ['cells','events','edges']}
        assert conn.execute('PRAGMA user_version').fetchone()[0] == database.SCHEMA_VERSION
    report = run_analyst(path,dish_id=env['dish_id'],now=NOW,report_id='one')
    run_analyst(path,dish_id=env['dish_id'],now=NOW,report_id='two')
    assert not append_report(path,report)
    with closing(database.connect(path)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM analyst_reports').fetchone()[0] == 2
        assert {table:list(conn.execute('SELECT * FROM '+table)) for table in before} == before
        with pytest.raises(sqlite3.IntegrityError,match='append-only'):
            conn.execute("UPDATE analyst_reports SET created_at='changed'")
        with pytest.raises(sqlite3.IntegrityError,match='append-only'):
            conn.execute('DELETE FROM analyst_reports')
    client = TestClient(create_app(env['petri_dir']))
    response = ResearchHealthReport.model_validate(client.get('/api/health').json())
    assert response.id == 'two' and response.flags
    checked = CliRunner().invoke(app,['check'])
    assert checked.exit_code == 0,checked.output
    assert 'Active health flags:' in checked.output and '[warning]' in checked.output
    assert response.flags[0].affected_cells[0] in checked.output
    assert 'research-health-panel' in build_frontend_html()


def test_query_views_supply_systemic_agent_evidence(seeded_petri_dir):
    env = seeded_petri_dir
    path = dish_db_path(env['petri_dir'])
    for i,cell in enumerate(database.list_cells(path)):
        database.append_event(path,Event(id=f'block-{i}',cell_id=cell['id'],timestamp='2026-10-31T00:00:00Z',
            type='verdict_issued',agent='skeptic',iteration=1,data={'verdict':'CRITICAL_FLAW_FOUND','rationale':'Fixture'}))
    report = analyze(health_snapshot(path))
    flag = rules(report)['systemic_agent_block']
    assert len(flag.affected_cells) == 5 and flag.evidence['query'] == 'blocking_verdict_patterns'


def test_missing_health_read_and_replay_create_no_new_reports(tmp_path):
    path = tmp_path/'missing.sqlite'
    assert latest_health(path) is None and not path.exists()
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/health').json() is None
    first = run_analyst(path,dish_id='dish',now=NOW,report_id='stable')
    assert run_analyst(path,dish_id='dish',now=NOW,report_id='stable') == first
    with closing(database.connect(path)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM analyst_reports').fetchone()[0] == 1


def test_analyst_modules_have_no_model_provider_or_dbos_imports():
    root = Path(__file__).parents[2]
    for relative in ['petri/analysis/analyst.py','petri/analysis/health.py']:
        tree = ast.parse((root/relative).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        assert not any(name and name.startswith(('dbos','pydantic_ai','petri.harness','petri.providers')) for name in imports)


@pytest.mark.parametrize('settings',[{'systemic_block_cells':0},{'verdict_window_hours':0},{'schedule':'invalid'}])
def test_invalid_thresholds_and_schedule_are_rejected(settings):
    with pytest.raises(ValueError):
        AnalystConfig.model_validate(settings)


def test_analyst_migration_is_forward_only(tmp_path):
    from importlib.resources import files
    path = tmp_path/'dish.sqlite'
    scripts = [files('petri.storage.migrations').joinpath(name).read_text() for name in database.MIGRATIONS]
    with sqlite3.connect(path) as conn:
        database.migrate(conn,scripts[:-1])
        before = conn.execute('PRAGMA user_version').fetchone()[0]
        database.migrate(conn,scripts)
        assert conn.execute('PRAGMA user_version').fetchone()[0] == before+1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='analyst_reports'").fetchone()
        with pytest.raises(database.SchemaVersionError):
            database.migrate(conn,scripts[:-1])


def test_analyst_options_survive_strict_v2_config_loading():
    from petri.agents.config import AgentConfig, load_agent_config
    defaults = Path(__file__).parents[2]/'petri/defaults/petri.yaml'
    config = load_agent_config(defaults)
    assert config.analyst.schedule is None
    raw = config.model_dump()
    raw['analyst'] = {'schedule':'0 * * * *','systemic_block_cells':7}
    configured = AgentConfig.model_validate(raw)
    assert configured.analyst.systemic_block_cells == 7

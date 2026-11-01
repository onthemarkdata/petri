"""Launch reads domain databases and leaves retired artifacts untouched."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from petri.cli import app
from petri.dashboard.stream_api import stream_messages
from petri.models import Event
from petri.storage.database import append_event
from petri.storage.paths import dish_db_path


def capture_launch(monkeypatch):
    captured = {}
    monkeypatch.setattr('petri.cli.launch._free_port_or_exit', lambda _: None)
    monkeypatch.setattr('threading.Timer', lambda *args, **kwargs: SimpleNamespace(start=lambda: None))
    def run(application, **options):
        with TestClient(application) as client:
            captured['events'] = client.get('/api/events').json()
            captured['cells'] = client.get('/api/cells').json()
            captured['dishes'] = client.get('/api/dishes').json()
        captured['app'] = application
    monkeypatch.setattr('uvicorn.run',run)
    return captured


def test_fresh_launch_creates_only_per_dish_database(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = capture_launch(monkeypatch)
    result = CliRunner().invoke(app,['launch'])
    assert result.exit_code == 0,result.output
    petri = tmp_path / '.petri'
    assert captured['events'] == []
    assert len(captured['dishes']) == 1
    assert (petri / 'petri-dishes' / tmp_path.name / 'petri.sqlite').is_file()
    assert not (petri / 'petri.sqlite').exists()
    assert not (petri / 'combined.jsonl').exists()
    assert 'Building event index' not in result.output


def test_seeded_launch_ignores_stale_index_and_rollup(seeded_petri_dir,monkeypatch):
    env = seeded_petri_dir
    petri = env['petri_dir']
    paths = [petri / 'petri.sqlite',petri / 'combined.jsonl']
    for path in paths:
        path.write_bytes(b'old artifact: do not read or rewrite')
    before = [(path.read_bytes(),path.stat().st_mtime_ns) for path in paths]
    append_event(dish_db_path(petri), Event(id='launch-event',cell_id=env['center'].id,
        timestamp='2026-11-01T00:00:00Z',type='search_executed',agent='investigator',iteration=1,
        data={'query':'fixture','sources_found':1}))
    captured = capture_launch(monkeypatch)
    result = CliRunner().invoke(app,['launch'])
    assert result.exit_code == 0,result.output
    assert any(event['id'] == 'launch-event' for event in captured['events'])
    assert len(captured['cells']) == 5
    async def read_stream():
        stream = stream_messages(petri)
        assert (await anext(stream))['event'] == 'connected'
        event = await asyncio.wait_for(anext(stream),1)
        await stream.aclose()
        return event
    assert asyncio.run(read_stream())['event'] == 'event_inserted'
    assert [(path.read_bytes(),path.stat().st_mtime_ns) for path in paths] == before


def test_retired_dashboard_index_symbols_are_absent():
    root = Path(__file__).parents[2] / 'petri'
    retired = ['petri.dashboard.migrate','rebuild_sqlite','incremental_sync','combined.jsonl','rollup_to_combined']
    assert not (root / 'dashboard/migrate.py').exists()
    for path in root.rglob('*.py'):
        source = path.read_text()
        assert not any(symbol in source for symbol in retired),path

"""A real local DBOS cron appends exactly one diagnostic report per firing."""

import time
from contextlib import closing

import pytest
from dbos import DBOS
from fastapi.testclient import TestClient

from petri.analysis.analyst import ResearchHealthReport
from petri.execution.analyst_schedule import SCHEDULE_NAME, register_analyst_schedule
from petri.execution.config import ExecutionConfig
from petri.execution.dbos_app import DBOSApp
from petri.dashboard.api import create_app
from petri.storage import database
from petri.storage.paths import dish_db_path


def reports(path):
    with closing(database.connect(path)) as conn:
        return [ResearchHealthReport.model_validate_json(row[0]) for row in conn.execute('SELECT report_json FROM analyst_reports')]


def runtime(petri_dir):
    return DBOSApp(petri_dir,dish_db_path(petri_dir),ExecutionConfig(),lambda spec:None).start()


@pytest.mark.integration
def test_absent_key_registers_nothing_and_real_cron_appends_once_per_firing(tmp_path):
    petri_dir = tmp_path/'.petri'
    petri_dir.mkdir()
    (petri_dir/'petri.yaml').write_text('name: dish\n')
    path = dish_db_path(petri_dir)
    running = runtime(petri_dir)
    try:
        assert DBOS.get_schedule(SCHEDULE_NAME) is None
        assert reports(path) == []
    finally:
        running.close()
    (petri_dir/'petri.yaml').write_text('name: dish\nanalyst:\n  schedule: "*/2 * * * * *"\n')
    running = runtime(petri_dir)
    try:
        register_analyst_schedule()
        assert len(DBOS.list_schedules(schedule_name_prefix=SCHEDULE_NAME)) == 1
        deadline = time.monotonic()+12
        while len(reports(path)) < 2:
            assert time.monotonic() < deadline,'Analyst cron did not fire twice.'
            time.sleep(0.05)
        DBOS.pause_schedule(SCHEDULE_NAME)
        rows = reports(path)
        runs = DBOS.list_workflows(name=['petri.analyst_tick'],status=['SUCCESS'])
        assert len(rows) == len(runs) == 2
        assert len({row.id for row in rows}) == 2
        latest = ResearchHealthReport.model_validate(TestClient(create_app(petri_dir)).get('/api/health').json())
        assert latest.id in {row.id for row in rows}
    finally:
        running.close()

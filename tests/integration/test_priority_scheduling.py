"""A real DBOS queue applies opt-in priorities without bypassing dependencies."""

import asyncio
import threading
import time
from dataclasses import replace

import pytest
from dbos import DBOS
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.execution.scheduler import ColonyRunSpec
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony, Edge
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.edge_registry import EdgeRecord, append
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.fixture
def priority_dish(tmp_path):
    (tmp_path/'petri.yaml').write_text('name: dish\nscheduling:\n  prioritize_targets: true\n')
    path = dish_db_path(tmp_path)
    ids = ['dish-alpha-000-000','dish-alpha-001-001','dish-alpha-001-002','dish-beta-000-000']
    parent,leaf,target,other = ids
    for colony_id, members in [('dish-alpha',ids[:3]),('dish-beta',[other])]:
        graph = ColonyGraph(colony_id)
        for cell_id in members:
            graph.add_cell(Cell(id=cell_id,colony_id=colony_id,level=0 if cell_id in [parent,other] else 1,claim_text='Fixture'))
        if colony_id=='dish-alpha':
            graph.add_edge(Edge(from_cell=parent,to_cell=target))
        save_colony(path,graph,Colony(id=colony_id,dish='dish',center_cell_id=members[0],center_claim='Fixture'))
    append(path,EdgeRecord(from_cell=other,to_cell=target,edge_type='cross_colony',created_at='2026-11-01'))
    return tmp_path,path,parent,leaf,target


def dependencies(path, starts, *, hold=None, release=None):
    phases = StubPhases()
    model = Agent(TestModel(custom_output_text='Fixture verdict'))

    async def research(value):
        if value.cell_id == 'dish-hold-000-000':
            hold.set()
            assert await asyncio.to_thread(release.wait,10)
        else:
            for edge in database.read_edges(path):
                if edge['from_cell']==value.cell_id:
                    assert database.get_cell(path,edge['to_cell'])['status']=='VALIDATED'
            starts.append(value.cell_id)
        await model.run('Return the local fixture verdict.')
        return await phases.research(value)

    async def finish(value,result):
        with database.write_connection(path) as connection:
            connection.execute('UPDATE cells SET status=? WHERE id=?',(result.status.upper(),value.cell_id))
    return replace(phases.dependencies(),research=research,finish=finish)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_convergence_target_executes_first_with_one_worker_and_emits_priority(priority_dish,monkeypatch):
    directory,path,parent,leaf,target = priority_dish
    hold,release = threading.Event(),threading.Event()
    starts = []
    deps = dependencies(path,starts,hold=hold,release=release)
    backend = DBOSExecutionBackend(config={'execution':{'backend':'dbos','worker_concurrency':1}})
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('petri.execution.priorities.get_tracer',lambda:provider.get_tracer('fixture'))
    await backend.start_async(directory,db_path=path,dependencies=lambda spec:deps)
    try:
        blocker = await backend.enqueue_cell(CellRunSpec('dish-hold-000-000',1,'2026-11-01'))
        assert await asyncio.to_thread(hold.wait,8)
        colony = await backend.enqueue_colony(ColonyRunSpec('dish-alpha',1,'2026-11-01',cell_ids=(leaf,target)))
        deadline = time.monotonic()+8
        while len(await DBOS.list_workflows_async(workflow_ids=[leaf+':r1',target+':r1'],status=['ENQUEUED'])) < 2:
            assert time.monotonic()<deadline
            await asyncio.sleep(0.02)
        release.set()
        await blocker.get_result()
        await colony.get_result()
        assert starts == [target,leaf]
        queued = {event['cell_id']:event['data']['priority'] for event in database.read_events(path,event_type='cell_queued')}
        assert queued[target] == 1 and queued[leaf]>queued[target] and parent not in queued
        spans = {span.attributes['petri.cell_id']:span for span in exporter.get_finished_spans()}
        assert spans[target].attributes['petri.queue.priority'] == queued[target]
        assert spans[leaf].attributes['petri.queue.priority'] == queued[leaf]
    finally:
        release.set()
        await backend.close_async()
        provider.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_target_priority_never_overrides_dependency_eligibility(priority_dish):
    directory,path,parent,leaf,target = priority_dish
    append(path,EdgeRecord(from_cell=target,to_cell=leaf,created_at='2026-11-01'))
    starts = []
    deps = dependencies(path,starts)
    backend = DBOSExecutionBackend(config={'execution':{'backend':'dbos','worker_concurrency':1}})
    await backend.start_async(directory,db_path=path,dependencies=lambda spec:deps)
    try:
        colony = await backend.enqueue_colony(ColonyRunSpec('dish-alpha',1,'2026-11-01',cell_ids=(leaf,target)))
        await colony.get_result()
        assert starts == [leaf,target]
        assert database.get_cell(path,parent)['status']=='NEW'
    finally:
        await backend.close_async()

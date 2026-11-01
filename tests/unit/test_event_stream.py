"""SSE cursors and bounded live progress work without inference."""

import asyncio
import json
from contextlib import closing

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

from petri.dashboard import broker
from petri.dashboard.api import create_app
from petri.dashboard.stream_api import POLL_INTERVAL, stream_messages
from petri.engine.streaming import progress_handler
from petri.storage import database


def append(path, index):
    with closing(database.connect(path)) as connection:
        connection.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', (str(index),'cell','search_executed','agent',index,'{}','2026-01-01'))


@pytest.mark.asyncio
async def test_live_append_and_reconnect_cursor_preserve_append_order(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    append(path,0)
    stream = stream_messages(tmp_path)
    assert (await anext(stream))['event'] == 'connected'
    first = await asyncio.wait_for(anext(stream),1)
    assert json.loads(first['data']) == {'id':'0','cell_id':'cell','type':'search_executed','agent':'agent','iteration':0}
    await stream.aclose()
    append(path,1)
    append(path,2)
    resumed = stream_messages(tmp_path,json.loads(first['id']))
    await anext(resumed)
    second = await asyncio.wait_for(anext(resumed),1)
    third = await asyncio.wait_for(anext(resumed),1)
    assert [json.loads(item['data'])['id'] for item in [second,third]] == ['1','2']
    pending = asyncio.create_task(anext(resumed))
    await asyncio.sleep(0)
    append(path,3)
    live = await asyncio.wait_for(pending,POLL_INTERVAL + 1)
    assert json.loads(live['data'])['id'] == '3'
    await resumed.aclose()


@pytest.mark.asyncio
async def test_progress_adapter_reaches_two_sse_subscribers(tmp_path):
    first, second = stream_messages(tmp_path), stream_messages(tmp_path)
    await anext(first)
    await anext(second)
    event = PartDeltaEvent(index=0,delta=TextPartDelta(content_delta='Measured result'))
    async def events():
        yield event
    await progress_handler(cell_id='cell',agent='investigator')(None,events())
    for stream in [first,second]:
        message = await asyncio.wait_for(anext(stream),1)
        assert message['event'] == 'agent_progress'
        assert json.loads(message['data']) == {'cell_id':'cell','agent':'investigator','kind':'text_delta','text_delta':'Measured result'}
    await first.aclose()
    broker.publish({'cell_id':'cell','agent':'investigator','kind':'text_delta','text_delta':'Still connected'})
    assert json.loads((await asyncio.wait_for(anext(second),1))['data'])['text_delta'] == 'Still connected'
    await second.aclose()


@pytest.mark.asyncio
async def test_slow_or_disconnected_subscriber_does_not_block_publish():
    async with broker.listen() as queue:
        for index in range(150):
            broker.publish({'cell_id':'cell','agent':'agent','kind':'text_delta','text_delta':str(index)})
        await asyncio.sleep(0)
        assert queue.qsize() == 100
        assert queue.get_nowait()['text_delta'] == '50'
    broker.publish({'cell_id':'cell','agent':'agent','kind':'text_delta','text_delta':'disconnected'})
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_poll_failure_is_logged_and_retries(tmp_path,monkeypatch,caplog):
    attempts = 0
    def enumerate_dishes(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError('temporary read failure')
        return []
    monkeypatch.setattr('petri.dashboard.stream_api.query.enumerate_dishes', enumerate_dishes)
    stream = stream_messages(tmp_path)
    await anext(stream)
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(POLL_INTERVAL + 0.05)
    broker.publish({'cell_id':'cell','agent':'agent','kind':'text_delta','text_delta':'recovered'})
    assert (await asyncio.wait_for(task,1))['event'] == 'agent_progress'
    assert attempts >= 2
    assert 'Dashboard event polling failed' in caplog.text
    await stream.aclose()


def test_invalid_cursor_is_json_400_before_streaming(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get('/api/stream?cursor=invalid')
    assert response.status_code == 400
    assert response.json() == {'detail':'Invalid event cursor.'}


@pytest.mark.asyncio
async def test_two_asgi_connections_receive_domain_and_progress_wire_messages(tmp_path):
    path = tmp_path / 'petri-dishes' / 'dish' / 'petri.sqlite'
    app = create_app(tmp_path)
    connected = 0
    received = []

    async def connection():
        done = asyncio.Event()
        bodies = []
        scope = {'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'GET',
                 'scheme':'http','path':'/api/stream','raw_path':b'/api/stream','query_string':b'',
                 'headers':[],'client':('test',1),'server':('test',80),'root_path':''}
        async def receive():
            await done.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            nonlocal connected
            if message['type'] != 'http.response.body':
                return
            body = message.get('body',b'').decode()
            bodies.append(body)
            if 'event: connected' in body:
                connected += 1
                if connected == 2:
                    append(path,1)
                    broker.publish({'cell_id':'cell','agent':'investigator','kind':'text_delta','text_delta':'Live text'})
            combined = ''.join(bodies)
            if 'event: event_inserted' in combined and 'event: agent_progress' in combined:
                received.append(combined)
                done.set()
        await app(scope,receive,send)

    await asyncio.wait_for(asyncio.gather(connection(),connection()),3)
    assert len(received) == 2
    for body in received:
        assert '"cell_id": "cell"' in body
        assert '"text_delta": "Live text"' in body
        assert '"type": "search_executed"' in body
        assert 'id: {"dish":1}' in body

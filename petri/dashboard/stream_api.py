"""Replayable domain events and transient agent progress over one SSE stream."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from petri import query
from petri.dashboard.broker import listen

logger = logging.getLogger(__name__)
POLL_INTERVAL = 0.2


def _cursor(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        if not isinstance(data, dict) or any(not isinstance(key, str) or type(rowid) is not int or rowid < 0 for key, rowid in data.items()):
            raise ValueError
        return data
    except (ValueError, TypeError) as error:
        raise HTTPException(400, 'Invalid event cursor.') from error


async def stream_messages(
    petri_dir: Path, watermark: dict[str, int] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Use one rowid cursor per dish, preserving each database's append order."""
    cursors = dict(watermark or {})
    async with listen() as progress:
        yield {'event': 'connected', 'data': json.dumps({'status': 'ok', 'watermarks': cursors})}
        while True:
            try:
                for dish in query.enumerate_dishes(petri_dir):
                    events = query.events_since(dish.db_path, cursors.get(dish.dish_id, 0), limit=500)
                    for event in events:
                        cursors[dish.dish_id] = event['rowid']
                        yield {
                            'event': 'event_inserted',
                            'id': json.dumps(cursors, sort_keys=True, separators=(',', ':')),
                            'data': json.dumps({key: event[key] for key in ('id', 'cell_id', 'type', 'agent', 'iteration')}),
                        }
            except Exception:  # noqa: BLE001 — retain cursors and retry after logging
                logger.exception('Dashboard event polling failed; retrying from the last cursor.')
            try:
                progress_event = await asyncio.wait_for(progress.get(), timeout=POLL_INTERVAL)
                yield {'event': 'agent_progress', 'data': json.dumps(progress_event)}
                while not progress.empty():
                    yield {'event': 'agent_progress', 'data': json.dumps(progress.get_nowait())}
            except TimeoutError:
                continue


def create_stream_router(petri_dir: Path) -> APIRouter:
    router = APIRouter()

    @router.get('/api/stream')
    async def event_stream(request: Request):
        raw = request.headers.get('last-event-id') or request.query_params.get('cursor')
        return EventSourceResponse(stream_messages(petri_dir, _cursor(raw)))

    return router

"""Bounded in-process fan-out for transient agent progress."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from typing_extensions import TypedDict


class AgentProgress(TypedDict):
    cell_id: str
    agent: str
    kind: str
    text_delta: str


_subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue[AgentProgress]]] = {}
_lock = threading.Lock()


def _deliver(queue: asyncio.Queue[AgentProgress], event: AgentProgress) -> None:
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(event)


def publish(event: AgentProgress) -> None:
    """Schedule each subscriber independently; slow clients retain recent progress."""
    with _lock:
        subscribers = list(_subscribers.values())
    for loop, queue in subscribers:
        try:
            loop.call_soon_threadsafe(_deliver, queue, event.copy())
        except RuntimeError:
            # A closing event loop must not interrupt another subscriber.
            with _lock:
                _subscribers.pop(id(queue), None)


@asynccontextmanager
async def listen() -> AsyncIterator[asyncio.Queue[AgentProgress]]:
    queue: asyncio.Queue[AgentProgress] = asyncio.Queue(maxsize=100)
    with _lock:
        _subscribers[id(queue)] = (asyncio.get_running_loop(), queue)
    try:
        yield queue
    finally:
        with _lock:
            _subscribers.pop(id(queue), None)


async def subscribe() -> AsyncIterator[AgentProgress]:
    async with listen() as queue:
        while True:
            yield await queue.get()

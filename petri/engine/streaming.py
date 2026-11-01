"""Adapt Pydantic AI stream events for the dashboard's live progress channel."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ToolCallPart

from petri.dashboard.broker import AgentProgress, publish


def adapt_stream_event(event: Any, *, cell_id: str, agent: str) -> AgentProgress | None:
    kind, text = '', ''
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        kind, text = 'text_delta', event.delta.content_delta
    elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        kind, text = 'text_start', event.part.content
    elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
        kind, text = 'tool_start', event.part.tool_name
    if not kind:
        return None
    return AgentProgress(cell_id=cell_id, agent=agent, kind=kind, text_delta=text)


def progress_handler(*, cell_id: str, agent: str):
    """Build the event handler used by dashboard-owned agent runs."""
    async def handler(context: Any, events: AsyncIterable[Any]) -> None:
        async for event in events:
            progress = adapt_stream_event(event, cell_id=cell_id, agent=agent)
            if progress is not None:
                publish(progress)
    return handler

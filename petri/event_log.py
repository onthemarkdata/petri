"""Append-only JSONL event log for Petri research nodes.

Each node stores its events in a dedicated ``events.jsonl`` file located at
``.petri/petri-dishes/{colony}/{level}-{seq}/events.jsonl``.  Events are
immutable -- they are only ever appended, never updated or deleted.  This
module is the sole writer; all other code treats the JSONL files as read-only.

A combined rollup concatenates per-node files into a single
``combined.jsonl`` that downstream consumers (e.g. the SQLite read index) can
ingest in one pass.
"""

from __future__ import annotations

import json
import secrets
import warnings
from datetime import datetime, timezone
from pathlib import Path

from petri.models import Event, EventType, build_event_key, validate_event_data


# ── Core Write Operations ────────────────────────────────────────────────


def append_event(
    events_path: Path,
    node_id: str,
    event_type: str,
    agent: str,
    iteration: int,
    data: dict,
) -> Event:
    """Append a validated event to a node's JSONL file.

    1. Validates *data* against the Pydantic model for *event_type*.
    2. Generates a collision-free event ID.
    3. Serialises the event as compact JSON and appends one line.
    4. Returns the created ``Event`` model instance.
    """
    validated_data = validate_event_data(event_type, data)

    # Load existing IDs so we can detect the (very rare) collision.
    existing_ids: set[str] = set()
    if events_path.exists():
        for entry in load_events(events_path):
            existing_ids.add(entry.get("id", ""))

    # Generate a unique event ID, re-rolling on collision.
    event_id = build_event_key(node_id, secrets.token_hex(4))
    while event_id in existing_ids:
        event_id = build_event_key(node_id, secrets.token_hex(4))

    timestamp = datetime.now(timezone.utc).isoformat()

    event = Event(
        id=event_id,
        node_id=node_id,
        timestamp=timestamp,
        type=EventType(event_type),
        agent=agent,
        iteration=iteration,
        data=validated_data,
    )

    # Ensure the parent directory exists.
    events_path.parent.mkdir(parents=True, exist_ok=True)

    with open(events_path, "a") as f:
        f.write(json.dumps(event.model_dump(), separators=(",", ":")) + "\n")

    return event


# ── Core Read Operations ─────────────────────────────────────────────────


def load_events(events_path: Path) -> list[dict]:
    """Read all events from a JSONL file.

    Malformed lines are skipped with a warning so that a single corrupt line
    does not prevent the rest of the log from being read.
    """
    if not events_path.exists():
        return []

    events: list[dict] = []
    content = events_path.read_text()

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(f"Skipping malformed line {line_num} in {events_path}")

    return events


def query_events(
    events_path: Path,
    node_id: str | None = None,
    iteration: int | None = None,
    event_type: str | None = None,
    agent: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Load events and filter by any combination of parameters.

    All supplied filters are combined with AND logic.  *since* is an ISO 8601
    timestamp -- only events with ``timestamp >= since`` are returned.
    """
    events = load_events(events_path)
    filtered: list[dict] = []

    for evt in events:
        if node_id is not None and evt.get("node_id") != node_id:
            continue
        if iteration is not None and evt.get("iteration") != iteration:
            continue
        if event_type is not None and evt.get("type") != event_type:
            continue
        if agent is not None and evt.get("agent") != agent:
            continue
        if since is not None and evt.get("timestamp", "") < since:
            continue
        filtered.append(evt)

    return filtered


def get_verdicts(
    events_path: Path,
    node_id: str | None = None,
    iteration: int | None = None,
    agent: str | None = None,
) -> list[dict]:
    """Return verdict_issued events with extracted verdict and summary."""
    events = query_events(
        events_path,
        node_id=node_id,
        iteration=iteration,
        event_type="verdict_issued",
        agent=agent,
    )
    verdicts: list[dict] = []
    for evt in events:
        data = evt.get("data", {})
        verdicts.append(
            {
                "node_id": evt.get("node_id"),
                "agent": evt.get("agent"),
                "iteration": evt.get("iteration"),
                "verdict": data.get("verdict"),
                "summary": data.get("summary"),
            }
        )
    return verdicts


def get_sources(events_path: Path) -> list[dict]:
    """Return source_reviewed events, deduplicated by URL."""
    events = query_events(events_path, event_type="source_reviewed")
    seen_urls: set[str] = set()
    sources: list[dict] = []

    for evt in events:
        url = evt.get("data", {}).get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(evt)

    return sources


def get_searches(events_path: Path) -> list[dict]:
    """Return search_executed events."""
    return query_events(events_path, event_type="search_executed")


# ── Rollup Operations ────────────────────────────────────────────────────


def rollup_to_combined(dish_path: Path) -> Path:
    """Concatenate all per-node ``events.jsonl`` files into ``combined.jsonl``.

    Walks ``dish_path/petri-dishes/{colony}/{node}/events.jsonl`` in sorted
    order and writes every non-blank line to ``dish_path/combined.jsonl``.
    The combined file is what the SQLite read index is built from.
    """
    combined_path = dish_path / "combined.jsonl"
    petri_dishes = dish_path / "petri-dishes"

    with open(combined_path, "w") as combined:
        if petri_dishes.is_dir():
            for events_file in sorted(petri_dishes.rglob("events.jsonl")):
                for line in events_file.read_text().splitlines():
                    line = line.strip()
                    if line:
                        combined.write(line + "\n")

    return combined_path

"""Local span batches and read-only telemetry queries using stdlib SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, TypedDict

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from petri.query.connection import enumerate_dishes, has_table, read_connection
from petri.storage.database import migrate

logger = logging.getLogger(__name__)
BUSY_TIMEOUT_MS = 100


class SpanRow(TypedDict):
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time_ns: int
    end_time_ns: int
    status_code: str
    status_message: str | None
    attributes: dict[str, Any]
    resource: dict[str, Any]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    try:
        connection.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
        migrate(connection)
        return connection
    except BaseException:
        connection.close()
        raise


class SQLiteSpanExporter(SpanExporter):
    """Commit one short transaction per batch; drop failed batches and warn once.

    The 100 ms lock wait bounds contention with research writes. Connections close
    after each batch, including failures. The processor owns buffering and flushes.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._warned = False
        self._lock = threading.Lock()
        self._closed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        if not spans:
            return SpanExportResult.SUCCESS
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            with closing(_connect(self.db_path)) as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    for span in spans:
                        context = span.context
                        if context is None:
                            continue
                        attributes = dict(span.attributes or {})
                        span_id = f'{context.span_id:016x}'
                        trace_id = f'{context.trace_id:032x}'
                        connection.execute(
                            'INSERT OR IGNORE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (span_id, trace_id, f'{span.parent.span_id:016x}' if span.parent else None,
                             span.name, span.kind.name, span.start_time, span.end_time,
                             span.status.status_code.name, span.status.description,
                             json.dumps(attributes), json.dumps(dict(span.resource.attributes))),
                        )
                        if any(key in attributes for key in ('petri.tokens_input', 'petri.tokens_output', 'petri.cost_usd')):
                            connection.execute(
                                'INSERT OR IGNORE INTO usage VALUES (?,?,?,?,?,?,?,?)',
                                (span_id, trace_id, attributes.get('petri.cell_id'),
                                 attributes.get('petri.phase'), attributes.get('petri.agent_role'),
                                 attributes.get('petri.tokens_input'), attributes.get('petri.tokens_output'),
                                 attributes.get('petri.cost_usd')),
                            )
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
            return SpanExportResult.SUCCESS
        except Exception as error:  # noqa: BLE001 — telemetry must never stop research
            with self._lock:
                if not self._warned:
                    logger.warning('Local telemetry batch dropped (%s); research continues.', type(error).__name__)
                    self._warned = True
            return SpanExportResult.FAILURE
        finally:
            detach(token)

    def shutdown(self) -> None:
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _paths(petri_dir: Path) -> list[Path]:
    path = Path(petri_dir)
    if path.suffix == '.sqlite':
        return [path] if path.is_file() else []
    return [dish.db_path for dish in enumerate_dishes(path)]


def query_spans(
    petri_dir: Path,
    *,
    cell_id: str | None = None,
    trace_id: str | None = None,
    colony: str | None = None,
    limit: int | None = 500,
) -> list[SpanRow]:
    """Read span rows across v2 dishes; missing telemetry tables yield no rows."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or None.')
    result: list[SpanRow] = []
    for path in _paths(petri_dir):
        with read_connection(path) as connection:
            if connection is None or not has_table(connection, 'spans'):
                continue
            sql = (
                'SELECT * FROM spans WHERE (? IS NULL OR json_extract(attributes,\'$."petri.cell_id"\')=?) '
                'AND (? IS NULL OR trace_id=?) '
                'AND (? IS NULL OR json_extract(attributes,\'$."petri.colony_id"\')=?) '
                'ORDER BY start_time_ns DESC,span_id'
            )
            values: list[Any] = [cell_id, cell_id, trace_id, trace_id, colony, colony]
            if limit is not None:
                sql += ' LIMIT ?'
                values.append(limit)
            for row in connection.execute(sql, values):
                result.append(SpanRow(
                    span_id=row['span_id'], trace_id=row['trace_id'], parent_span_id=row['parent_span_id'],
                    name=row['name'], kind=row['kind'], start_time_ns=row['start_time_ns'],
                    end_time_ns=row['end_time_ns'], status_code=row['status_code'],
                    status_message=row['status_message'], attributes=json.loads(row['attributes']),
                    resource=json.loads(row['resource']),
                ))
    result.sort(key=lambda row: (-row['start_time_ns'], row['span_id']))
    return result[:limit] if limit is not None else result


def prune(petri_dir: Path, retention_days: int = 30, *, now_ns: int | None = None) -> int:
    """Prune telemetry only; domain tables remain untouched, including event history."""
    if type(retention_days) is not int or retention_days <= 0:
        raise ValueError('retention_days must be a positive integer.')
    cutoff = (time.time_ns() if now_ns is None else now_ns) - retention_days * 86_400_000_000_000
    removed = 0
    for path in _paths(petri_dir):
        try:
            with closing(sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)) as connection:
                if not has_table(connection, 'spans'):
                    continue
                connection.execute('BEGIN IMMEDIATE')
                connection.execute('DELETE FROM usage WHERE span_id IN (SELECT span_id FROM spans WHERE start_time_ns<?)', (cutoff,))
                removed += connection.execute('DELETE FROM spans WHERE start_time_ns<?', (cutoff,)).rowcount
                connection.commit()
        except sqlite3.Error as error:
            logger.warning('Local telemetry retention skipped (%s).', type(error).__name__)
    return removed

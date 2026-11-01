"""Read-only trace routes over the local span-store query contract."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from petri.telemetry.span_store import SpanRow, aggregate_usage, query_spans


def create_traces_router(petri_dir: Path) -> APIRouter:
    router = APIRouter()

    @router.get('/api/traces')
    def traces(
        cell_id: str | None = None,
        colony: str | None = None,
        limit: int = Query(default=100, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[SpanRow]] = {}
        for span in query_spans(petri_dir, limit=None):
            grouped.setdefault(span['trace_id'], []).append(span)
        result: list[dict[str, Any]] = []
        for trace_id, spans in grouped.items():
            if cell_id is not None and not any(span['attributes'].get('petri.cell_id') == cell_id for span in spans):
                continue
            if colony is not None and not any(span['attributes'].get('petri.colony_id') == colony for span in spans):
                continue
            ordered = sorted(spans, key=lambda span: (span['start_time_ns'], span['span_id']))
            root = next((span for span in ordered if span['parent_span_id'] is None), ordered[0])
            start = min(span['start_time_ns'] for span in spans)
            end = max(span['end_time_ns'] for span in spans)
            result.append({
                'trace_id': trace_id,
                'name': root['name'],
                'start_time_ns': start,
                'duration_ns': max(0, end - start),
                'cell_id': root['attributes'].get('petri.cell_id') or next((span['attributes']['petri.cell_id'] for span in ordered if 'petri.cell_id' in span['attributes']), None),
                'verdict': next((span['attributes']['petri.verdict'] for span in reversed(ordered) if 'petri.verdict' in span['attributes']), None),
            })
        result.sort(key=lambda row: (-row['start_time_ns'], row['trace_id']))
        return result[:limit]

    @router.get('/api/traces/{trace_id}')
    def trace(trace_id: str) -> list[SpanRow]:
        spans = query_spans(petri_dir, trace_id=trace_id, limit=None)
        if not spans:
            raise HTTPException(status_code=404, detail='Trace not found.')
        return sorted(spans, key=lambda span: (span['start_time_ns'], span['span_id']))

    @router.get('/api/costs')
    def costs(cell_id: str | None = None, colony: str | None = None) -> dict[str, Any]:
        options = {'cell_id': cell_id, 'colony': colony}
        totals = aggregate_usage(petri_dir, group_by='total', **options)
        return {
            'total': totals[0] if totals else {'cell_id': None, 'phase': None, 'tokens_input': None, 'tokens_output': None, 'cost_usd': None},
            'cells': aggregate_usage(petri_dir, group_by='cell', **options),
            'phases': aggregate_usage(petri_dir, group_by='cell_phase', **options),
        }

    return router

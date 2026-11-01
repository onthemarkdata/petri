"""Pure, model-free research-health rules over a committed query snapshot."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from petri.analysis.edge_intelligence import EdgeIntelligenceReport


class AnalystConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schedule: str | None = None
    systemic_block_cells: int = Field(default=5, ge=1, strict=True)
    verdict_window_hours: float = Field(default=24, gt=0, allow_inf_nan=False)
    stale_after_hours: float = Field(default=168, gt=0, allow_inf_nan=False)
    stale_min_dependents: int = Field(default=3, ge=1, strict=True)


    @field_validator('schedule')
    @classmethod
    def valid_schedule(cls, value: str | None) -> str | None:
        from croniter import croniter
        if value is not None and not croniter.is_valid(value,second_at_beginning=True):
            raise ValueError('analyst.schedule must be a valid cron expression')
        return value


class HealthFlag(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    rule: Literal['systemic_agent_block', 'stalled_verdict_velocity', 'transitive_staleness']
    severity: Literal['warning', 'critical']
    message: str
    affected_cells: list[str]
    affected_agents: list[str]
    evidence: dict[str, Any]


class ResearchHealthReport(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: str
    dish_id: str
    created_at: str
    flags: list[HealthFlag]
    thresholds: AnalystConfig


class HealthSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cells: list[dict[str, Any]]
    events: list[dict[str, Any]]
    blocking_patterns: list[dict[str, Any]]
    intelligence: EdgeIntelligenceReport
    dependencies: dict[str, list[str]]


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def analyze_health(snapshot: HealthSnapshot, *, report_id: str, dish_id: str,
                   now: datetime, config: AnalystConfig | None = None) -> ResearchHealthReport:
    """Diagnose active cells without mutating the snapshot or its domain records."""
    settings = config or AnalystConfig()
    flags: list[HealthFlag] = []
    cells = {str(cell['id']):cell for cell in snapshot.cells}
    blocked: dict[str, set[str]] = {}
    patterns: dict[str, list[dict[str, Any]]] = {}
    for pattern in snapshot.blocking_patterns:
        agent = str(pattern['agent'])
        blocked.setdefault(agent,set()).update(set(pattern['cell_ids']) & cells.keys())
        patterns.setdefault(agent,[]).append(pattern)
    for agent, affected in sorted(blocked.items()):
        if len(affected) >= settings.systemic_block_cells:
            flags.append(HealthFlag(rule='systemic_agent_block',severity='critical',
                message=f'{agent} has blocking verdicts on {len(affected)} active cells.',
                affected_cells=sorted(affected),affected_agents=[agent],
                evidence={'query':'blocking_verdict_patterns','blocked_cells':len(affected),'rows':patterns[agent]}))

    events: dict[str, list[dict[str, Any]]] = {}
    for event in snapshot.events:
        if event['cell_id'] in cells:
            events.setdefault(event['cell_id'],[]).append(event)
    cutoff = now - timedelta(hours=settings.verdict_window_hours)
    eligible = [cell_id for cell_id,cell in sorted(cells.items()) if cell['status']=='NEW'
        and all(cells.get(dependency,{}).get('status')=='VALIDATED'
                for dependency in snapshot.dependencies.get(cell_id,[]))]
    stalled = [cell_id for cell_id in eligible if not any(event['type']=='verdict_issued'
        and cutoff <= _timestamp(event['timestamp']) <= now for event in events.get(cell_id,[]))]
    if stalled:
        flags.append(HealthFlag(rule='stalled_verdict_velocity',severity='warning',
            message=f'{len(stalled)} eligible cells have no verdicts in the last {settings.verdict_window_hours:g} hours.',
            affected_cells=stalled,affected_agents=[],evidence={'query':'events:verdict_issued',
                'window_start':cutoff.isoformat(),'window_end':now.isoformat(),'eligible_cells':eligible,
                'recent_verdict_counts':{cell_id:0 for cell_id in stalled}}))

    for cell_id,metric in sorted(snapshot.intelligence.cells.items()):
        if cell_id not in cells or metric.transitive_dependents < settings.stale_min_dependents:
            continue
        timestamps = [_timestamp(event['timestamp']) for event in events.get(cell_id,[])
                      if _timestamp(event['timestamp']) <= now]
        if not timestamps:
            continue
        latest = max(timestamps)
        age = (now-latest).total_seconds()/3600
        if age >= settings.stale_after_hours:
            flags.append(HealthFlag(rule='transitive_staleness',severity='warning',
                message=f'{cell_id} has stale research activity and {metric.transitive_dependents} transitive dependents.',
                affected_cells=[cell_id],affected_agents=[],evidence={'query':'events + edge_intelligence',
                    'last_event_at':latest.isoformat(),'age_hours':age,
                    'transitive_dependents':metric.transitive_dependents}))
    return ResearchHealthReport(id=report_id,dish_id=dish_id,created_at=now.isoformat(),flags=flags,thresholds=settings)

"""Offer settled-round proposals only when the grow command opts in."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import typer
from opentelemetry import trace

from petri.agents.contract import AgentName
from petri.lifecycle.approval import decide_proposal, list_proposals
from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.redecompose import RedecomposeSpec, propose
from petri.lifecycle.triggers import TriggerContext, detect_triggers
from petri.models import Verdict
from petri.storage import database
from petri.storage.paths import dish_db_path

ALIASES = {
    "flaw": "CRITICAL_FLAW_FOUND", "inconclusive": "EVIDENCE_INCONCLUSIVE",
    "caveats": "DEFENSIBLE_WITH_CAVEATS", "stalled": "PIPELINE_STALLED",
}


def parse_triggers(value: str) -> frozenset[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALIASES and name not in ALIASES.values()]
    if unknown:
        raise ValueError(f"Unknown re-decomposition trigger: {', '.join(unknown)}. Use {', '.join(ALIASES)}.")
    return frozenset(ALIASES.get(name, name) for name in names)


class ApprovalGate:
    def __init__(self, *, enabled: frozenset[str], config: LifecycleConfig,
                 approve_all: bool = False, reject_all: bool = False,
                 selected_cells: list[str] | None = None, colony: str | None = None,
                 interactive: bool | None = None, echo: Callable[[str], None] = typer.echo):
        self.enabled, self.config = enabled, config
        self.approve_all, self.reject_all = approve_all, reject_all
        self.selected_cells, self.colony = selected_cells, colony
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.echo = echo
        self.span = trace.get_current_span()
        self.decisions: list[str] = []

    def _selected(self, row: dict) -> bool:
        return (self.selected_cells is None or row["id"] in self.selected_cells) and (
            self.colony is None or row["colony"] == self.colony or row["colony"].endswith("-" + self.colony))

    async def __call__(self, petri_dir: Path) -> bool:
        if not self.enabled:
            return False
        path = dish_db_path(petri_dir)
        proposals = list_proposals(path)
        by_cell: dict[str, list[dict]] = defaultdict(list)
        for event in database.read_events(path):
            by_cell[event["cell_id"]].append(event)
        for row in database.list_cells(path):
            if not self._selected(row) or row["status"] not in {"VALIDATED", "DISPROVEN", "DEFER_OPEN", "DEFER_CLOSED", "NEEDS_EXPERIMENT", "STALLED"}:
                continue
            prior = [p for p in proposals if p.parent_cell_id == row["id"]]
            if any(p.state in {"reserved", "proposed"} for p in prior):
                continue
            events = by_cell[row["id"]]
            # A reopened research round cannot reuse the previous round's verdicts.
            start = max((i + 1 for i, event in enumerate(events) if event["type"] == "cell_reopened"), default=0)
            verdict_events = [event for event in events[start:] if event["type"] == "verdict_issued"]
            if not verdict_events:
                continue
            checks = [event for event in events[start:] if event["type"] == "convergence_checked"]
            final_verdict = verdict_events[-1]
            final_check = checks[-1] if checks else None
            check_is_later = final_check is not None and events.index(final_check) > events.index(final_verdict)
            source = final_check["id"] if check_is_later and final_check else final_verdict["id"]
            if any(p.trigger.source_event_id == source for p in prior):
                continue
            context = dict(final_check["data"]) if final_check else {}
            if not check_is_later:
                context.pop("stall_reason", None)
            verdicts = [Verdict(cell_id=row["id"], agent=e["agent"], iteration=e["iteration"],
                verdict=e["data"]["verdict"], summary=e["data"].get("summary", "")) for e in verdict_events]
            if context.get("stall_reason") == "circuit_breaker":
                verdicts.append(Verdict(cell_id=row["id"], agent="cell_lead", iteration=verdicts[-1].iteration,
                    verdict="PIPELINE_STALLED", summary="The convergence cycle exhausted its budget."))
            triggers = detect_triggers(verdicts, TriggerContext(
                generation=sum(p.state == "approved" for p in prior),
                stall_reason=context.get("stall_reason"), weakest_link=context.get("weakest_link"),
                focused_directive=context.get("focused_directive") or "",
            ), {agent.value: True for agent in AgentName} | {"cell_lead": True}, self.config)
            triggers = [t for t in triggers if t.trigger_verdict in self.enabled]
            if triggers:
                priority = list(ALIASES.values())
                chosen = min(triggers, key=lambda t: priority.index(t.trigger_verdict))
                proposals.append(await propose(RedecomposeSpec(
                    trigger=chosen.model_copy(update={"source_event_id": source}), config=self.config)))
        admitted = False
        for proposal in proposals:
            if proposal.state != "proposed" or proposal.trigger.trigger_verdict not in self.enabled:
                continue
            parent = database.get_cell(path, proposal.parent_cell_id)
            if parent is None or not self._selected(parent):
                continue
            self.echo(f"Proposal {proposal.id}: {proposal.trigger.trigger_verdict}; guard: {proposal.reason}")
            for child in proposal.child_ids:
                cell = database.get_cell(path, child)
                self.echo(f"  {cell['claim_text']}" if cell else f"  Missing cell: {child}")
            if self.reject_all:
                approve = False
            elif self.approve_all or self.config.auto_approve:
                approve = True
            elif self.interactive:
                approve = typer.confirm("Approve these sub-claims for validation?", default=False)
            else:
                self.echo("Pending proposals require approval. Rerun with --redecompose-on and --approve-all or --reject-all.")
                continue
            decided, changed = decide_proposal(path, proposal.id, approve=approve, config=self.config)
            self.decisions.append(f"{proposal.id}:{decided.state}")
            self.span.set_attribute("petri.redecomposition.decisions", self.decisions)
            if changed and approve and proposal.child_ids:
                admitted = True
                if self.selected_cells is not None:
                    self.selected_cells.extend(child for child in proposal.child_ids if child not in self.selected_cells)
        return admitted

"""Create durable proposals without enqueueing their validation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel, ConfigDict, Field

from petri.execution.dbos_app import active_app
from petri.execution.dbos_backend import _bounded
from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.proposals import (
    Proposal, get_proposal, journal_output, materialize_proposal, recorded_output, reserve_proposal, proposal_identity,
)
from petri.lifecycle.triggers import RedecompositionTrigger
from petri.models import parse_key
from petri.reasoning.decomposition_agents import DecompositionInput
from petri.storage import database
from petri.telemetry.setup import get_tracer


class RedecomposeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trigger: RedecompositionTrigger
    config: LifecycleConfig = Field(default_factory=LifecycleConfig)
    evidence_paths: tuple[str, ...] = ()

    @property
    def workflow_id(self) -> str:
        suffix = f":{self.trigger.source_event_id}" if self.trigger.source_event_id else ""
        return f"{self.trigger.cell_id}:redecompose:{self.trigger.generation + 1}{suffix}"


class ExpansionProgress(BaseModel):
    proposal_id: str
    retry_after_seconds: float | None = None


@DBOS.step(name="petri.reserve_redecomposition")
def reserve_step(spec: RedecomposeSpec) -> Proposal:
    return reserve_proposal(active_app().db_path, spec.trigger, spec.config)


def _evidence_context(root: Path, paths: tuple[str, ...]) -> str:
    snippets = []
    remaining = 64_000
    for supplied in paths:
        path = (root / supplied).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Proposal evidence paths must stay inside the dish directory")
        text = path.read_text(encoding="utf-8")[:remaining]
        snippets.append(f"Evidence from {supplied}:\n{text}")
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(snippets)


@DBOS.step(name="petri.redecompose_why")
def expand_step(spec: RedecomposeSpec, proposal_id: str) -> ExpansionProgress:
    """Journal the validated response before returning its small proposal reference."""
    from petri.execution.seed_workflow import _rate_limit

    app = active_app()
    with app.step_activity():
        if recorded_output(app.db_path, proposal_id) is not None:
            return ExpansionProgress(proposal_id=proposal_id)
        row = database.get_cell(app.db_path, spec.trigger.cell_id)
        if row is None:
            raise ValueError("The triggering cell no longer exists")
        dish = app.db_path.parent.name
        identity = parse_key(spec.trigger.cell_id, dish, strict=True)
        runner = app.seed_runner(spec)
        context = "\n".join(filter(None, [
            f"Reconsider this claim after {spec.trigger.trigger_verdict}.",
            spec.trigger.focused_directive, spec.trigger.flaw_text, spec.trigger.caveat_text,
            f"Weakest link: {spec.trigger.weakest_link}" if spec.trigger.weakest_link else "",
            _evidence_context(app.petri_dir, spec.evidence_paths),
        ]))
        request = DecompositionInput(claim=row["claim_text"], guidance=context,
            parent_level=row["level"], parent_seq=identity["seq"], db_path=app.db_path,
            colony_id=row["colony"], require_counterargument=spec.trigger.trigger_verdict == "CRITICAL_FLAW_FOUND")
        try:
            output = runner.expand_why(request)
        except Exception as error:
            delay = _rate_limit(error)
            if delay is None:
                raise
            return ExpansionProgress(proposal_id=proposal_id, retry_after_seconds=delay)
        _bounded(output)
        journal_output(app.db_path, proposal_id, output)
        return ExpansionProgress(proposal_id=proposal_id)


@DBOS.step(name="petri.materialize_redecomposition")
def materialize_step(proposal_id: str) -> Proposal:
    return _bounded(materialize_proposal(active_app().db_path, proposal_id))


@DBOS.workflow(name="petri.redecompose")
async def redecompose(spec: RedecomposeSpec) -> Proposal:
    _bounded(spec)
    app = active_app()
    identity = parse_key(spec.trigger.cell_id, app.db_path.parent.name, strict=True)
    with get_tracer("petri.lifecycle").start_as_current_span("petri.re-decompose", attributes={
        "petri.command": "re-decompose", "petri.dish_id": app.db_path.parent.name,
        "petri.cell_id": spec.trigger.cell_id, "petri.verdict": spec.trigger.trigger_verdict,
        "petri.colony_id": identity["colony_prefix"], "petri.level": identity["level"],
        "petri.seq": identity["seq"], "petri.agent_role": "decomposition_lead",
        "petri.generation": spec.trigger.generation + 1,
    }) as span:
        proposal = await asyncio.to_thread(reserve_step, spec)
        if proposal.state != "reserved":
            span.set_attribute("petri.outcome", proposal.state)
            return proposal
        while True:
            await app.acquire_slot()
            try:
                progress = await asyncio.to_thread(expand_step, spec, proposal.id)
            finally:
                app.release_slot()
            if progress.retry_after_seconds is None:
                break
            await DBOS.sleep_async(progress.retry_after_seconds)
        result = await asyncio.to_thread(materialize_step, proposal.id)
        span.set_attribute("petri.outcome", result.state)
        return result


async def propose(spec: RedecomposeSpec) -> Proposal:
    """Attach to the recorded attempt and resume failed or cancelled inference."""
    status = await DBOS.get_workflow_status_async(spec.workflow_id)
    if status is not None and status.status == "ERROR":
        steps = await DBOS.list_workflow_steps_async(spec.workflow_id)
        failed = next((step for step in steps if step.get("error") is not None), None)
        if failed is None:
            raise RuntimeError("The proposal failed outside a recorded step; inspect its workflow before retrying.")
        handle = await DBOS.rewind_workflow_async(spec.workflow_id, start_step=failed["function_id"])
    elif status is not None and status.status == "CANCELLED":
        handle = await DBOS.resume_workflow_async(spec.workflow_id)
    else:
        with SetWorkflowID(spec.workflow_id):
            handle = await DBOS.start_workflow_async(redecompose, spec)
    await handle.get_result()
    return get_proposal(active_app().db_path,
        proposal_identity(spec.trigger))

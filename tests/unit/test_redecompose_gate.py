"""Approval persists before scheduling and consumes each settled verdict once."""

import asyncio
from dataclasses import replace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.lifecycle.approval import decide_proposal, list_proposals
from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.gate import ApprovalGate, parse_triggers
from petri.lifecycle.proposals import journal_output, materialize_proposal, reserve_proposal
from petri.lifecycle.triggers import RedecompositionTrigger
from petri.models import Event, validate_event_data
from petri.storage import database
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases
from tests.unit.test_redecompose import output


def proposed(env, source="first"):
    path = dish_db_path(env["petri_dir"])
    trigger = RedecompositionTrigger(cell_id=env["cell1"].id, trigger_verdict="EVIDENCE_INCONCLUSIVE",
        agent="evidence_evaluator", fire_point="settled_round", source_event_id=source)
    proposal = reserve_proposal(path, trigger, LifecycleConfig(per_parent_cap=1))
    journal_output(path, proposal.id, output())
    return materialize_proposal(path, proposal.id)


def test_rejection_allows_a_later_source_without_advancing_generation(seeded_petri_dir):
    env = seeded_petri_dir
    path = dish_db_path(env["petri_dir"])
    first = proposed(env)
    rejected, changed = decide_proposal(path, first.id, approve=False, config=LifecycleConfig())
    assert changed and rejected.state == "rejected"
    assert not decide_proposal(path, first.id, approve=True, config=LifecycleConfig())[1]
    second = proposed(env, "later-round")
    assert second.id != first.id and second.generation == first.generation == 1
    assert database.get_cell(path, first.child_ids[0])["status"] == "DEFER_CLOSED"
    event = database.read_events(path, event_type="redecomposition_rejected")[0]
    assert validate_event_data(event["type"], event["data"])


def test_approval_is_atomic_idempotent_and_records_child_generation(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    path = dish_db_path(env["petri_dir"])
    proposal = proposed(env)
    from petri.lifecycle import approval
    original = approval._event
    monkeypatch.setattr(approval, "_event", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        decide_proposal(path, proposal.id, approve=True, config=LifecycleConfig())
    assert database.get_cell(path, proposal.child_ids[0])["status"] == "DEFER_CLOSED"
    monkeypatch.setattr(approval, "_event", original)
    assert decide_proposal(path, proposal.id, approve=True, config=LifecycleConfig())[1]
    assert not decide_proposal(path, proposal.id, approve=True, config=LifecycleConfig())[1]
    assert database.get_cell(path, proposal.child_ids[0])["status"] == "NEW"
    events = database.read_events(path, event_type="redecomposition_approved")
    assert len(events) == 1 and events[0]["data"]["generation"] == 1
    assert validate_event_data(events[0]["type"], events[0]["data"])


def test_no_flag_does_not_read_surface_or_approve_even_with_auto_approve(tmp_path, monkeypatch):
    from petri.lifecycle import gate
    monkeypatch.setattr(gate, "list_proposals", lambda _: pytest.fail("Detection must remain off"))
    callback = ApprovalGate(enabled=frozenset(), config=LifecycleConfig(auto_approve=True), echo=lambda _: pytest.fail("No output"))
    assert not asyncio.run(callback(tmp_path))


def test_noninteractive_pending_notice_and_explicit_rejection(seeded_petri_dir):
    env = seeded_petri_dir
    proposal = proposed(env)
    lines = []
    callback = ApprovalGate(enabled=parse_triggers("inconclusive"), config=LifecycleConfig(), interactive=False, echo=lines.append)
    assert not asyncio.run(callback(env["petri_dir"]))
    assert any("Pending proposals require approval" in line for line in lines)
    callback.reject_all = True
    assert not asyncio.run(callback(env["petri_dir"]))
    assert list_proposals(dish_db_path(env["petri_dir"]))[0].state == "rejected"
    assert proposal.child_ids


def test_approved_children_run_once_in_the_next_durable_pass(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    directory = env["petri_dir"]
    path = dish_db_path(directory)
    proposal = proposed(env)
    phases = StubPhases()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    async def finish(value, result):
        with database.write_connection(path) as conn:
            conn.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))
    with provider.get_tracer("test").start_as_current_span("grow"):
        callback = ApprovalGate(enabled=parse_triggers("inconclusive"), config=LifecycleConfig(auto_approve=True), echo=lambda _: None)
        backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}}, after_pass=callback)
        start = backend.start
        deps = replace(phases.dependencies(), finish=finish)
        monkeypatch.setattr(backend, "start", lambda directory, **options: start(directory, **options, dependencies=lambda _: deps))
        result = backend.process_queue(petri_dir=directory)
    assert result.processed == 6
    assert len([name for name, _ in phases.calls if name == "socratic"]) == 6
    assert database.get_cell(path, proposal.child_ids[0])["status"] == "VALIDATED"
    assert len(database.read_events(path, cell_id=proposal.child_ids[0], event_type="cell_queued")) == 1
    assert len(database.read_events(path, event_type="redecomposition_approved")) == 1
    assert exporter.get_finished_spans()[-1].attributes["petri.redecomposition.decisions"]
    provider.shutdown()


def test_settled_source_is_consumed_once_and_user_stalls_are_excluded(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    directory = env["petri_dir"]
    path = dish_db_path(directory)
    parent = env["cell1"].id
    row = database.get_cell(path, parent)
    database.put_cell(path, {**row, "status": "DEFER_OPEN"})
    database.append_event(path, Event(id="source-verdict", cell_id=parent, timestamp="2026-11-01T00:00:00Z",
        type="verdict_issued", agent="evidence_evaluator", iteration=1,
        data={"verdict": "EVIDENCE_INCONCLUSIVE", "summary": "Measurement is unresolved"}))
    from petri.lifecycle import gate
    calls = []
    async def create(spec):
        calls.append(spec)
        proposal = reserve_proposal(path, spec.trigger, spec.config)
        journal_output(path, proposal.id, output())
        return materialize_proposal(path, proposal.id)
    monkeypatch.setattr(gate, "propose", create)
    callback = ApprovalGate(enabled=parse_triggers("inconclusive,stalled"), config=LifecycleConfig(), reject_all=True, echo=lambda _: None)
    assert not asyncio.run(callback(directory))
    assert not asyncio.run(callback(directory))
    assert len(calls) == 1 and calls[0].trigger.source_event_id == "source-verdict"
    database.append_event(path, Event(id="user-stop", cell_id=parent, timestamp="2026-11-01T01:00:00Z",
        type="verdict_issued", agent="cell_lead", iteration=1, data={"verdict": "PIPELINE_STALLED"}))
    callback.enabled = parse_triggers("stalled")
    assert not asyncio.run(callback(directory)) and len(calls) == 1


def test_trigger_aliases_reject_unknown_values():
    assert parse_triggers("inconclusive,CRITICAL_FLAW_FOUND") == {"EVIDENCE_INCONCLUSIVE", "CRITICAL_FLAW_FOUND"}
    with pytest.raises(ValueError, match="Unknown"):
        parse_triggers("everything")


def test_circuit_breaker_provenance_does_not_survive_a_later_user_stop(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    path = dish_db_path(env["petri_dir"])
    parent = env["cell1"].id
    database.put_cell(path, {**database.get_cell(path, parent), "status": "STALLED"})
    def record(key, kind, data, agent="cell_lead"):
        database.append_event(path, Event(id=key, cell_id=parent, timestamp="2026-11-01T00:00:00Z",
            type=kind, agent=agent, iteration=1, data=data))
    record("evidence", "verdict_issued", {"verdict": "EVIDENCE_INCONCLUSIVE"}, "evidence_evaluator")
    record("breaker", "convergence_checked", {"converged": False, "stall_reason": "circuit_breaker"})
    from petri.lifecycle import gate
    calls = []
    async def create(spec):
        calls.append(spec)
        proposal = reserve_proposal(path, spec.trigger, spec.config)
        journal_output(path, proposal.id, output())
        return materialize_proposal(path, proposal.id)
    monkeypatch.setattr(gate, "propose", create)
    callback = ApprovalGate(enabled=parse_triggers("stalled"), config=LifecycleConfig(), reject_all=True, echo=lambda _: None)
    assert not asyncio.run(callback(env["petri_dir"]))
    assert len(calls) == 1 and calls[0].trigger.source_event_id == "breaker"
    record("later-stop", "verdict_issued", {"verdict": "PIPELINE_STALLED"})
    assert not asyncio.run(callback(env["petri_dir"]))
    assert len(calls) == 1


def test_cli_rejects_conflicting_decisions_before_starting_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from typer.testing import CliRunner
    from petri.cli import app
    monkeypatch.setattr("petri.cli.grow.find_petri_dir", lambda: tmp_path)
    monkeypatch.setattr("petri.cli.grow.resolve_backend", lambda _: SimpleNamespace())
    result = CliRunner().invoke(app, ["grow", "--redecompose-on=inconclusive", "--approve-all", "--reject-all"])
    assert result.exit_code != 0 and "Choose either" in result.output

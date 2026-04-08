"""Tests for the optional ``on_event`` lifecycle callback plumbing.

Phase 2 of the multi-spinner plan adds an ``on_event`` parameter to
``process_node``/``process_queue`` so the CLI can paint a per-slot status
row whenever a node transitions phase, fires an agent, or finishes.

These tests:
  * confirm the existing call sites are not affected (``on_event`` defaults
    to ``None``)
  * confirm the callback receives ``started``/``phase``/``agent``/
    ``verdict``/``finished`` events in a sensible order
  * confirm callback exceptions never break the engine
  * confirm slot indices are stable per node and unique across concurrent
    nodes
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from petri.engine.processor import (
    NodeProgressEvent,
    process_node,
    process_queue,
    reset_stop,
)
from petri.storage.queue import update_state


@pytest.fixture(autouse=True)
def _reset_processor_stop_signal():
    """Clear the global stop signal before AND after every test in this file.

    The ``petri stop`` CLI tests call ``request_stop()`` and never reset
    it, leaving the module-level ``_stop_event`` in a "set" state for any
    later test that calls ``process_node`` directly.  ``process_node``
    short-circuits at the very first stop check, which makes our
    callback assertions fail in the full suite while passing in
    isolation.  Resetting around every test isolates us from that
    legacy leakage.
    """
    reset_stop()
    yield
    reset_stop()


DISH_ID = "test-dish"
COLONY_NAME = "callbacks"
NODE_ID_TEMPLATE = f"{DISH_ID}-{COLONY_NAME}-001-{{seq:03d}}"


# ── Deterministic InferenceProvider ─────────────────────────────────────


class CallbackTestProvider:
    """Deterministic InferenceProvider that drives every phase to a verdict.

    Returns canned verdicts so the convergence and red-team paths terminate
    cleanly:

      * Socratic steps return ``CLARIFIED``/``ASSUMPTIONS_CHALLENGED``/
        ``EVIDENCE_MAPPED``/``VERIFIED``.
      * Phase 1 (research) and Phase 2 (critique) agents return ``PASS``
        with one Level-1 source so the convergence check sees a unanimous
        blocking verdict matrix and the source-hierarchy validator passes.
      * Red team returns ``CANNOT_DISPROVE``.
      * Evidence evaluator returns ``EVIDENCE_CONFIRMS``.

    A small artificial delay (``call_delay``) makes it possible to overlap
    multiple ``process_node`` invocations under ``ThreadPoolExecutor``.
    """

    def __init__(self, call_delay: float = 0.0) -> None:
        self.call_delay = call_delay
        self.assess_calls: list[dict] = []

    def assess_node(
        self,
        node_id: str,
        claim_text: str,
        context: dict,
        agent_role: str,
    ) -> dict:
        self.assess_calls.append(
            {
                "node_id": node_id,
                "agent_role": agent_role,
                "phase": context.get("phase", ""),
            }
        )
        if self.call_delay:
            time.sleep(self.call_delay)

        phase = context.get("phase", "")
        verdict = self._verdict_for(phase, agent_role)
        return {
            "verdict": verdict,
            "summary": f"summary for {agent_role}",
            "arguments": "",
            "confidence": "HIGH",
            "sources_cited": [
                {
                    "url": "https://example.org/source",
                    "title": "Test source",
                    "hierarchy_level": 1,
                    "finding": "matches the claim exactly",
                    "supports_or_contradicts": "supports",
                    "confidence": "HIGH",
                }
            ],
        }

    @staticmethod
    def _verdict_for(phase: str, agent_role: str) -> str:
        if phase == "socratic_clarify":
            return "CLARIFIED"
        if phase == "socratic_challenge_assumptions":
            return "ASSUMPTIONS_CHALLENGED"
        if phase == "socratic_identify_evidence_needed":
            return "EVIDENCE_MAPPED"
        if phase == "socratic_verification":
            return "VERIFIED"
        if phase == "red_team":
            return "CANNOT_DISPROVE"
        if phase == "evaluation":
            return "EVIDENCE_CONFIRMS"
        # Default for phase 1/phase 2 agents — matches the canonical
        # convergence "pass" set so check_convergence terminates.
        return "PASS"

    # Other InferenceProvider methods are unused by the processor pipeline
    # but the Protocol declares them.
    def assess_claim_substance(self, claim, on_progress=None):
        return {"is_substantive": True, "reason": "", "suggested_rewrite": ""}

    def generate_clarifying_questions(self, claim, max_questions=5, on_progress=None):
        return []

    def decompose_claim(self, claim, clarifications, guidance="", max_premises=5, on_progress=None):
        return {"nodes": [], "edges": []}

    def decompose_why(self, premise, parent_level, parent_seq, max_premises=5, on_progress=None):
        return []

    def match_evidence(self, content, nodes):
        return []


# ── Filesystem fixture builder ──────────────────────────────────────────


def _build_callback_dish(tmp_path: Path, seqs: list[int]) -> dict:
    """Build a minimal .petri/ layout with one node per seq.

    Each node uses the flat fallback layout
    (``petri-dishes/<colony>/{level}-{seq}/``) so we can sidestep the
    colony.json deserialisation that ``process_queue`` would otherwise
    require for ``find_eligible_nodes``.

    Returned dict carries:
      petri_dir: the .petri root
      node_ids:  ordered list of node IDs created
      events_files: parallel list of events.jsonl paths
    """
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # ``_get_dish_id`` reads ``defaults/petri.yaml`` for the dish name.
    defaults_dir = petri_dir / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "petri.yaml").write_text(f"name: {DISH_ID}\n")

    node_ids: list[str] = []
    events_files: list[Path] = []
    queue_entries: dict = {}

    for seq in seqs:
        node_id = NODE_ID_TEMPLATE.format(seq=seq)
        node_ids.append(node_id)

        node_dir = petri_dir / "petri-dishes" / COLONY_NAME / f"001-{seq:03d}"
        node_dir.mkdir(parents=True)

        events_file = node_dir / "events.jsonl"
        events_file.touch()
        events_files.append(events_file)

        (node_dir / "evidence.md").write_text(
            f"# {node_id}\n\n**Status:** pending\n\n"
        )
        (node_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": node_id,
                    "claim_text": f"test claim {seq}",
                    "level": 1,
                    "status": "NEW",
                    "dependencies": [],
                    "dependents": [],
                },
                indent=2,
            )
            + "\n"
        )

        queue_entries[node_id] = {
            "node_id": node_id,
            "queue_state": "queued",
            "iteration": 0,
            "entered_at": "2026-01-01T00:00:00+00:00",
            "last_activity": "2026-01-01T00:00:00+00:00",
        }

    queue_path = petri_dir / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "last_updated": "2026-01-01T00:00:00+00:00",
                "entries": queue_entries,
            },
            indent=2,
        )
        + "\n"
    )

    return {
        "petri_dir": petri_dir,
        "node_ids": node_ids,
        "events_files": events_files,
    }


@pytest.fixture
def single_node_dish(tmp_path):
    """A petri dish with exactly one node ready in the ``queued`` state."""
    return _build_callback_dish(tmp_path, seqs=[1])


@pytest.fixture
def multi_node_dish(tmp_path):
    """A petri dish with three nodes ready in the ``queued`` state."""
    return _build_callback_dish(tmp_path, seqs=[1, 2, 3])


# ── Tests ───────────────────────────────────────────────────────────────


def test_no_event_callback_means_no_overhead(single_node_dish):
    """``on_event=None`` is the default and the existing path keeps working.

    Calls ``process_node`` directly (rather than ``process_queue``) so the
    test doesn't depend on a fully serialised colony — the lifecycle
    callback wiring is what's under test, not eligibility discovery.
    """
    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    result = process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
    )

    assert result.node_id == node_id
    assert result.error is None
    assert provider.assess_calls, "provider was never invoked"


def test_callback_fires_started_and_finished_for_one_node(single_node_dish):
    """Single node — the callback receives a started + finished event."""
    captured: list[NodeProgressEvent] = []

    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
        slot_idx=2,
        on_event=captured.append,
    )

    started = [event for event in captured if event.kind == "started"]
    finished = [event for event in captured if event.kind == "finished"]

    assert len(started) == 1, f"expected exactly one started event, got {started}"
    assert len(finished) == 1, f"expected exactly one finished event, got {finished}"
    assert started[0].node_id == node_id
    assert finished[0].node_id == node_id
    # slot_idx must round-trip back to the caller
    assert started[0].slot_idx == 2
    assert finished[0].slot_idx == 2
    assert finished[0].slot_idx >= 0


def test_callback_fires_phase_events_in_order(single_node_dish):
    """Phase events appear in pipeline order: socratic → research → ... → evaluating.

    The fake provider drives a clean happy path so all six phases fire.
    """
    captured: list[NodeProgressEvent] = []
    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
        slot_idx=0,
        on_event=captured.append,
    )

    phase_events = [event.phase for event in captured if event.kind == "phase"]
    expected_order = [
        "socratic", "research", "critique", "mediating", "red_team", "evaluating",
    ]

    # The test asserts only the relative ordering of the phases that DID
    # fire — the FakeProvider is configured for a one-shot happy path so
    # every phase should appear exactly once, but iteration loops in the
    # mediating phase could in principle insert extra research/critique
    # cycles which would also be valid.
    fired = [phase for phase in expected_order if phase in phase_events]
    indices = [phase_events.index(phase) for phase in fired]
    assert indices == sorted(indices), (
        f"phase events out of order: {phase_events}"
    )

    # Sanity check — at minimum we expect socratic to fire.
    assert "socratic" in phase_events


def test_callback_fires_agent_and_verdict_events(single_node_dish):
    """At least one agent + verdict event is captured with non-empty fields."""
    captured: list[NodeProgressEvent] = []
    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
        slot_idx=0,
        on_event=captured.append,
    )

    agent_events = [event for event in captured if event.kind == "agent"]
    verdict_events = [event for event in captured if event.kind == "verdict"]

    assert agent_events, "no agent events captured"
    assert verdict_events, "no verdict events captured"

    for agent_event in agent_events:
        assert agent_event.agent, f"agent field empty: {agent_event}"
        assert agent_event.phase, f"phase field empty: {agent_event}"

    for verdict_event in verdict_events:
        assert verdict_event.agent, f"agent field empty: {verdict_event}"
        assert verdict_event.verdict, f"verdict field empty: {verdict_event}"
        assert verdict_event.phase, f"phase field empty: {verdict_event}"


def test_callback_exception_does_not_break_processing(single_node_dish):
    """Callback raises on every call — the engine still completes the node."""

    def raising_callback(event: NodeProgressEvent) -> None:
        raise RuntimeError("intentional UI failure")

    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    result = process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
        slot_idx=0,
        on_event=raising_callback,
    )

    assert result.node_id == node_id
    assert result.error is None, f"engine reported error: {result.error}"
    # The node must have advanced past the initial 'queued' state — i.e.
    # the loop ran and the callback exceptions were swallowed.
    queue_data = json.loads(
        (single_node_dish["petri_dir"] / "queue.json").read_text()
    )
    final_state = queue_data["entries"][node_id]["queue_state"]
    assert final_state in {
        "done", "deferred_open", "deferred_closed", "needs_human", "stalled",
    }, f"node did not reach a terminal state: {final_state}"


def test_slot_idx_stable_for_one_node(single_node_dish):
    """Every event for a given node carries the same slot_idx."""
    captured: list[NodeProgressEvent] = []
    provider = CallbackTestProvider()
    node_id = single_node_dish["node_ids"][0]

    process_node(
        node_id=node_id,
        petri_dir=single_node_dish["petri_dir"],
        provider=provider,
        slot_idx=7,
        on_event=captured.append,
    )

    slot_indices = {event.slot_idx for event in captured if event.node_id == node_id}
    assert slot_indices == {7}, f"slot_idx not stable: {slot_indices}"


# ── Concurrent slot uniqueness ──────────────────────────────────────────


class _ImmediateBalancer:
    """Stand-in for ``AdaptiveLoadBalancer`` that always recommends ``max``.

    The real balancer ramps from ``min_workers`` upward over multiple poll
    intervals; that's hostile to a unit test that wants to confirm three
    nodes occupy three distinct slots.  This shim short-circuits the ramp.
    """

    def __init__(self, max_workers: int = 1, min_workers: int = 1, **_kwargs):
        self.max_workers = max_workers
        self.min_workers = min_workers

    @property
    def recommended_workers(self) -> int:
        return self.max_workers

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_slot_idx_unique_for_concurrent_nodes(multi_node_dish, monkeypatch):
    """Three nodes processed in parallel get three distinct slot indices.

    Uses ``process_queue`` so the slot pool wrapper is exercised end-to-
    end.  ``find_eligible_nodes`` is fed an explicit ``node_ids`` list so
    we don't depend on colony.json deserialisation.

    The balancer is monkeypatched to immediately allow ``max_workers``
    concurrent submissions, otherwise the default ramp-from-1 strategy
    serialises the workers and only ever uses slot 0.
    """
    monkeypatch.setattr(
        "petri.engine.load_balancer.AdaptiveLoadBalancer", _ImmediateBalancer
    )

    captured: list[NodeProgressEvent] = []
    captured_lock = threading.Lock()

    def _record(event: NodeProgressEvent) -> None:
        with captured_lock:
            captured.append(event)

    # A small per-call delay holds each node in flight long enough for
    # the other two to be picked up — without this the first node would
    # often complete before the second is even submitted.
    provider = CallbackTestProvider(call_delay=0.01)

    result = process_queue(
        petri_dir=multi_node_dish["petri_dir"],
        provider=provider,
        max_concurrent=3,
        node_ids=multi_node_dish["node_ids"],
        all_nodes=True,
        on_event=_record,
    )

    assert result.processed == 3, f"expected 3 processed, got {result.processed}"

    # Build {node_id: {slot_idx, ...}} from the captured events.
    per_node_slots: dict[str, set[int]] = {}
    for event in captured:
        per_node_slots.setdefault(event.node_id, set()).add(event.slot_idx)

    # Each node must have used exactly one slot index throughout its life.
    for node_id, slots in per_node_slots.items():
        assert len(slots) == 1, (
            f"{node_id} switched slots mid-flight: {slots}"
        )

    # The three nodes occupied three DIFFERENT slot indices.
    occupied = {next(iter(slots)) for slots in per_node_slots.values()}
    assert len(occupied) == 3, (
        f"expected 3 distinct slot indices across 3 nodes, got {occupied}"
    )
    # Sanity — every slot is in [0, max_concurrent).
    for slot_index in occupied:
        assert 0 <= slot_index < 3, f"slot index out of bounds: {slot_index}"

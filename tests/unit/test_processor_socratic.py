"""Tests for the idempotent Socratic phase runner.

``_run_socratic_phase`` used to append a ``### Socratic Analysis`` block
to ``evidence.md`` unconditionally.  On a restart (``petri grow`` killed
mid-run) that produced duplicate Socratic sections.  The phase is now
guarded so a pre-existing ``socratic_*`` verdict in the event log causes
the runner to skip straight to ``research_active``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from petri.storage.event_log import append_event
from petri.storage.queue import update_state


DISH_ID = "test-dish"
COLONY_NAME = "mycolony"
CELL_ID = f"{DISH_ID}-{COLONY_NAME}-001-001"


class RecordingSocraticProvider:
    """Minimal InferenceProvider stub for Socratic phase tests.

    Records every ``assess_cell`` call so tests can assert that a skipped
    phase made zero provider calls.  Returns a deterministic
    AssessmentResult-shaped dict per step.
    """

    def __init__(self) -> None:
        self.assess_calls: list[dict] = []

    def assess_cell(
        self,
        cell_id: str,
        claim_text: str,
        context: dict,
        agent_role: str,
        *,
        on_progress=None,  # accepted but ignored by this stub
    ) -> dict:
        self.assess_calls.append(
            {
                "cell_id": cell_id,
                "agent_role": agent_role,
                "phase": context.get("phase", ""),
            }
        )
        phase = context.get("phase", "")
        if phase == "socratic_clarify":
            verdict_name = "CLARIFIED"
        elif phase == "socratic_challenge_assumptions":
            verdict_name = "ASSUMPTIONS_CHALLENGED"
        elif phase == "socratic_identify_evidence_needed":
            verdict_name = "EVIDENCE_MAPPED"
        else:
            verdict_name = "VERIFIED"
        return {
            "verdict": verdict_name,
            "summary": f"summary for {phase}",
            "arguments": f"arguments for {phase}",
            "confidence": "high",
            "sources_cited": [],
        }

    # Other InferenceProvider methods are unused by the Socratic phase.
    def assess_claim_substance(self, claim: str, on_progress=None) -> dict:
        return {"is_substantive": True, "reason": "", "suggested_rewrite": ""}

    def generate_clarifying_questions(
        self, claim: str, max_questions: int = 5, on_progress=None,
    ) -> list[dict]:
        return []

    def decompose_claim(
        self, claim, clarifications, guidance="", max_premises=5, on_progress=None,
    ) -> dict:
        return {"nodes": [], "edges": []}

    def decompose_why(
        self, premise, parent_level, parent_seq, max_premises=5, on_progress=None,
    ) -> list[dict]:
        return []

    def match_evidence(self, content: str, cells: list[dict]) -> list[dict]:
        return []


@pytest.fixture
def socratic_dish(tmp_path):
    """Build a minimal .petri/ layout for a single cell ready for Socratic phase.

    Creates:
      - petri_dir/queue.json with CELL_ID in socratic_active state
      - petri_dir/petri-dishes/<colony>/001-001/events.jsonl (empty)
      - petri_dir/petri-dishes/<colony>/001-001/evidence.md (with Status)
    """
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()

    # Queue with the cell pre-seeded in socratic_active so the transition
    # to research_active is valid.
    queue_data = {
        "version": 1,
        "last_updated": "2026-01-01T00:00:00+00:00",
        "entries": {
            CELL_ID: {
                "cell_id": CELL_ID,
                "queue_state": "socratic_active",
                "iteration": 0,
                "entered_at": "2026-01-01T00:00:00+00:00",
                "last_activity": "2026-01-01T00:00:00+00:00",
            }
        },
    }
    (petri_dir / "queue.json").write_text(json.dumps(queue_data, indent=2) + "\n")

    # Fallback cell layout: petri-dishes/<colony>/{level}-{seq}/
    cell_path = petri_dir / "petri-dishes" / COLONY_NAME / "001-001"
    cell_path.mkdir(parents=True)
    events_file = cell_path / "events.jsonl"
    events_file.touch()
    evidence_file = cell_path / "evidence.md"
    evidence_file.write_text(
        "# Cell evidence\n\n**Status:** pending\n\n"
    )

    return {
        "petri_dir": petri_dir,
        "cell_dir": cell_path,
        "events_file": events_file,
        "evidence_file": evidence_file,
    }


def _read_queue_state(petri_dir: Path) -> str:
    queue = json.loads((petri_dir / "queue.json").read_text())
    return queue["entries"][CELL_ID]["queue_state"]


def test_socratic_phase_runs_when_no_prior_events(socratic_dish):
    """Empty event log — phase runs, verdicts get written, state advances."""
    from petri.engine.processor import _run_socratic_phase

    provider = RecordingSocraticProvider()
    _run_socratic_phase(
        cell_id=CELL_ID,
        claim_text="claim under test",
        petri_dir=socratic_dish["petri_dir"],
        dish_id=DISH_ID,
        iteration=0,
        provider=provider,
    )

    # 3 socratic steps + 1 verification = 4 assess_cell calls.
    assert len(provider.assess_calls) == 4

    # At least one verdict_issued event per step now lives in the log.
    event_lines = socratic_dish["events_file"].read_text().strip().splitlines()
    verdict_events = [
        json.loads(line)
        for line in event_lines
        if json.loads(line).get("type") == "verdict_issued"
    ]
    socratic_verdicts = [
        event for event in verdict_events
        if str(event.get("agent", "")).startswith("socratic_")
    ]
    assert len(socratic_verdicts) >= 3

    # State advanced socratic_active -> research_active.
    assert _read_queue_state(socratic_dish["petri_dir"]) == "research_active"

    # Evidence file contains the Socratic Analysis block.
    evidence_body = socratic_dish["evidence_file"].read_text()
    assert evidence_body.count("### Socratic Analysis") == 1


def test_socratic_phase_skips_when_prior_socratic_events_exist(socratic_dish):
    """Pre-existing socratic_* verdict — phase is a no-op beyond advancing state."""
    from petri.engine.processor import _run_socratic_phase

    # Seed a prior socratic_clarify verdict so the idempotency guard fires.
    append_event(
        events_path=socratic_dish["events_file"],
        cell_id=CELL_ID,
        event_type="verdict_issued",
        agent="socratic_clarify",
        iteration=0,
        data={
            "verdict": "CLARIFIED",
            "summary": "already ran",
            "arguments": "",
            "evidence": "",
            "confidence": "high",
            "sources_cited": [],
        },
    )

    provider = RecordingSocraticProvider()
    _run_socratic_phase(
        cell_id=CELL_ID,
        claim_text="claim under test",
        petri_dir=socratic_dish["petri_dir"],
        dish_id=DISH_ID,
        iteration=0,
        provider=provider,
    )

    # Guard fired — provider was never called.
    assert provider.assess_calls == []

    # State still advanced so the pipeline can continue.
    assert _read_queue_state(socratic_dish["petri_dir"]) == "research_active"

    # Evidence file was left untouched (no new Socratic block).
    evidence_body = socratic_dish["evidence_file"].read_text()
    assert "### Socratic Analysis" not in evidence_body


def test_evidence_md_does_not_duplicate_socratic_block(socratic_dish):
    """Running the phase twice must produce exactly one Socratic block."""
    from petri.engine.processor import _run_socratic_phase

    provider = RecordingSocraticProvider()

    # First run — writes the block and advances state to research_active.
    _run_socratic_phase(
        cell_id=CELL_ID,
        claim_text="claim under test",
        petri_dir=socratic_dish["petri_dir"],
        dish_id=DISH_ID,
        iteration=0,
        provider=provider,
    )

    # Second run — must bail out via the idempotency guard.  Reset the
    # queue back to socratic_active so update_state's transition check
    # doesn't reject the call; the guard still sees the prior verdicts.
    update_state(
        socratic_dish["petri_dir"] / "queue.json",
        CELL_ID,
        "critique_active",
    )
    # Walk back to socratic_active via stalled -> queued -> socratic_active
    # using direct JSON edit (only needed for this test setup).
    queue_path = socratic_dish["petri_dir"] / "queue.json"
    queue_data = json.loads(queue_path.read_text())
    queue_data["entries"][CELL_ID]["queue_state"] = "socratic_active"
    queue_path.write_text(json.dumps(queue_data, indent=2) + "\n")

    first_run_call_count = len(provider.assess_calls)
    _run_socratic_phase(
        cell_id=CELL_ID,
        claim_text="claim under test",
        petri_dir=socratic_dish["petri_dir"],
        dish_id=DISH_ID,
        iteration=0,
        provider=provider,
    )

    # Second call made zero additional provider invocations.
    assert len(provider.assess_calls) == first_run_call_count

    # Evidence file still has exactly one Socratic Analysis heading.
    evidence_body = socratic_dish["evidence_file"].read_text()
    assert evidence_body.count("### Socratic Analysis") == 1

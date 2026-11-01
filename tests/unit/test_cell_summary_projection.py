"""Read terminal assessments from events without replacing legacy summaries."""

import pytest
from fastapi.testclient import TestClient

from petri import query
from petri.dashboard.api import create_app
from petri.execution.decisions import build_pipeline_dependencies
from petri.execution.models import CellPipelineState
from petri.execution.pipeline import run_pipeline
from petri.models import Event
from petri.storage import database
from tests.unit import test_research_phases

phase_services = test_research_phases.phase_services


@pytest.fixture
def cell_store(tmp_path):
    path = tmp_path / "petri-dishes/dish/petri.sqlite"
    key = "dish-alpha-000-000"
    database.put_cell(path, {"id": key, "colony": "dish-alpha", "seq": 0, "level": 0,
        "claim_text": "A measured claim", "status": "VALIDATED", "created_at": "2026-01-01"})
    return path, key


def assessment(path, key, identity, summary, *, agent="evidence_evaluator", verdict="EVIDENCE_CONFIRMS"):
    database.append_event(path, Event(id=identity, cell_id=key, timestamp="2026-01-01T00:00:00Z",
        type="verdict_issued", agent=agent, iteration=0, data={"verdict": verdict, "summary": summary}))


@pytest.mark.parametrize("status,verdict", [
    ("VALIDATED", "EVIDENCE_CONFIRMS"), ("DISPROVEN", "EVIDENCE_REFUTES"),
    ("DEFER_OPEN", "EVIDENCE_INCONCLUSIVE"), ("DEFER_CLOSED", "EVIDENCE_INCONCLUSIVE"),
])
def test_terminal_summary_uses_latest_recorded_assessment_without_domain_writes(cell_store, status, verdict):
    path, key = cell_store
    with database.write_connection(path) as connection:
        connection.execute("UPDATE cells SET status=? WHERE id=?", (status, key))
    assessment(path, key, "older", "An earlier assessment.")
    assessment(path, key, "terminal", "The actual terminal assessment.", verdict=verdict)
    assessment(path, key, "later-advisory", "An unrelated advisory.", agent="skeptic", verdict="CRITICAL_FLAW_FOUND")
    before = database.get_cell(path, key), database.read_events(path)
    assert query.get_cell(path, key)["summary_md"] == "The actual terminal assessment."
    assert query.list_cells(path, status=status)[0]["summary_md"] == "The actual terminal assessment."
    assert before == (database.get_cell(path, key), database.read_events(path))
    assert database.get_cell(path, key)["summary_md"] == ""


def test_existing_legacy_summary_is_preserved_verbatim(cell_store):
    path, key = cell_store
    legacy = "## Existing digest\n\nPreserve the reviewed wording."
    assessment(path, key, "legacy", legacy, agent="cell_lead", verdict="CONVERGED")
    assessment(path, key, "terminal", "A newer evaluator assessment.")
    assert query.get_cell(path, key)["summary_md"] == legacy


def test_reopened_or_unfinished_cells_do_not_reuse_old_terminal_assessments(cell_store):
    path, key = cell_store
    assessment(path, key, "older-round", "The previous round's conclusion.")
    database.append_event(path, Event(id="reopened", cell_id=key, timestamp="2026-01-02T00:00:00Z",
        type="cell_reopened", agent="execution_backend", iteration=0,
        data={"trigger": "human_revalidation", "prior_status": "VALIDATED"}))
    assert query.get_cell(path, key)["summary_md"] == ""
    assessment(path, key, "new-round", "The new round's assessment.")
    with database.write_connection(path) as connection:
        connection.execute("UPDATE cells SET status='NEW' WHERE id=?", (key,))
    assert query.get_cell(path, key)["summary_md"] == ""
    with database.write_connection(path) as connection:
        connection.execute("UPDATE cells SET status='VALIDATED' WHERE id=?", (key,))
    assert query.get_cell(path, key)["summary_md"] == "The new round's assessment."


def test_missing_or_empty_terminal_assessment_stays_empty(cell_store):
    path, key = cell_store
    assert query.get_cell(path, key)["summary_md"] == ""
    assessment(path, key, "advisory", "A skeptical finding.", agent="skeptic", verdict="CRITICAL_FLAW_FOUND")
    assert query.get_cell(path, key)["summary_md"] == ""
    assessment(path, key, "older", "An older evaluator finding.")
    assessment(path, key, "empty-terminal", "")
    assert query.get_cell(path, key)["summary_md"] == ""


def test_dashboard_detail_serves_the_projected_terminal_summary(cell_store, tmp_path):
    path, key = cell_store
    text = "The final measurement refutes the claim."
    assessment(path, key, "terminal", text, verdict="EVIDENCE_REFUTES")
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/cell/" + key)
    assert response.status_code == 200
    assert response.json()["summary_md"] == text
    assert response.json()["events"][-1]["data"]["verdict"] == "EVIDENCE_REFUTES"


@pytest.mark.asyncio
async def test_real_typed_pipeline_exposes_its_terminal_assessment(phase_services, canonical_colony):
    services, _ = phase_services
    key = canonical_colony["cell1"].id
    result = await run_pipeline(CellPipelineState(cell_id=key), build_pipeline_dependencies(services))
    assert result.status == "validated"
    assert query.get_cell(services.db_path, key)["summary_md"] == "evidence_evaluator reviewed the evidence."
    assert database.get_cell(services.db_path, key)["summary_md"] == ""

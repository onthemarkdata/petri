"""Interrupted seed calls recover from checkpoints and immutable output events."""

import json
from contextlib import closing

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from petri.cli import app
from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.reasoning.seed_checkpoint import SeedCheckpointStore, classify_cells
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.graph.colony import ColonyGraph
from petri.models import Colony

SETTINGS = DecompositionSettings(depth=3, per_layer=8, model="test")
INITIAL = {"nodes": [{"claim_text": "The meter is a defined unit"},
                     {"claim_text": "The second is a defined unit", "relation": "limits"}]}
ATOMIC = {"is_atomic": True, "bedrock_reason": "A unit definition", "sub_premises": []}


def run(path, payloads, *, resume=False, callback=None, store=None):
    successful = []
    sequence = iter(payloads)

    async def response(messages, info):
        payload = next(sequence)
        if isinstance(payload, Exception):
            raise payload
        successful.append(payload)
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    with HarnessInferenceProvider(FunctionModel(response)) as runner:
        from petri.reasoning.decomposition_runner import decompose_typed

        result = decompose_typed("Physical units", [], "dish", "units", runner,
            settings=SETTINGS, db_path=path, resume=resume,
            on_cell_created=callback, checkpoint_store=store)
    return result, successful


@pytest.mark.parametrize("damage", [None, "delete", "corrupt"])
def test_interrupt_resume_reuses_completed_calls(tmp_path, damage):
    path = tmp_path / "petri.sqlite"
    with pytest.raises(Exception, match="interrupted"):
        run(path, [INITIAL, ATOMIC, RuntimeError("interrupted")])
    store = SeedCheckpointStore(path)
    state = store.load("dish-units", 0)
    assert state and len(state.pending) == 1
    classes = classify_cells(path, "dish-units", 0)
    assert classes == {"dish-units-000-000": "expanded", "dish-units-001-001": "atomic",
                       "dish-units-001-002": "pending"}
    if damage == "delete":
        store.delete("dish-units")
    elif damage == "corrupt":
        with closing(database.connect(path)) as conn:
            conn.execute("UPDATE seed_checkpoints SET state='broken'")
    result, completed = run(path, [ATOMIC], resume=True)
    baseline, baseline_calls = run(tmp_path / "baseline.sqlite", [INITIAL, ATOMIC, ATOMIC])
    assert len(completed) + 2 == len(baseline_calls) == 3
    assert [(c.id, c.claim_text, c.is_atomic) for c in result.cells] == [
        (c.id, c.claim_text, c.is_atomic) for c in baseline.cells]
    assert result.edges == baseline.edges
    assert store.load("dish-units", 0).complete
    assert list(classify_cells(path, "dish-units", 0).values()).count("atomic") == 2


def test_crash_during_materialization_reuses_recorded_response(tmp_path):
    path = tmp_path / "petri.sqlite"

    def interrupt(cell, edges):
        raise RuntimeError("after first cell")

    with pytest.raises(RuntimeError, match="after first cell"):
        run(path, [INITIAL], callback=interrupt)
    result, completed = run(path, [ATOMIC, ATOMIC], resume=True)
    assert len(completed) == 2
    assert len(result.cells) == len(database.list_cells(path)) == 3
    assert len(database.read_edges(path)) == 2
    assert len(database.read_events(path, event_type="cell_expansion_completed")) == 3


@pytest.mark.parametrize("files", [False, True])
def test_store_round_trip_and_completed_resume_without_calls(tmp_path, files):
    path = tmp_path / "petri.sqlite"
    store = SeedCheckpointStore(path, file_dir=tmp_path / "checkpoints" if files else None)
    result, _ = run(path, [INITIAL, ATOMIC, ATOMIC], store=store)
    saved = store.load("dish-units", 0)
    assert saved and saved.complete
    restored, calls = run(path, [], resume=True, store=store)
    assert restored == result and calls == []
    assert store.load("dish-units", 1) is None
    with closing(database.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == database.SCHEMA_VERSION


def test_completed_cli_resume_does_not_resolve_provider(tmp_path, monkeypatch):
    from petri.cli import seed as seed_module

    petri = tmp_path / ".petri"
    path = petri / "petri-dishes" / "dish" / "petri.sqlite"
    result, _ = run(path, [INITIAL, ATOMIC, ATOMIC])
    graph = ColonyGraph(colony_id="dish-units")
    for cell in result.cells:
        graph.add_cell(cell)
    for edge in result.edges:
        graph.add_edge(edge)
    save_colony(path, graph, Colony(id="dish-units", dish="dish", center_claim="Physical units",
                                    center_cell_id="dish-units-000-000"))
    monkeypatch.setattr(seed_module, "find_petri_dir", lambda: petri)
    monkeypatch.setattr(seed_module, "get_dish_id", lambda _: "dish")
    monkeypatch.setattr(seed_module, "resolve_provider", lambda _: pytest.fail("resolved a provider"))
    runner = CliRunner()
    first = runner.invoke(app, ["seed", "--resume", "--colony", "units", "--no-questions"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["seed", "--resume", "--colony", "units"])
    assert second.exit_code == 0 and "nothing to resume" in second.output

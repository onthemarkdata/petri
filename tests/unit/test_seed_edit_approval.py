"""Draft edits validate before writes and survive an interrupted materialization."""

from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from petri.cli import app
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony, DecompositionResult, Edge
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.reasoning.seed_edit import dump_draft, save_draft_edit, validate_draft
from petri.storage import database
from petri.storage.colonies import save_colony

SETTINGS = DecompositionSettings(
    depth=3, per_layer=8, per_parent_cap=2, level_one_cap=3, model="test"
)


def draft():
    texts = [
        "Solar panels reduce imported electricity",
        "Incident sunlight exceeds the operating minimum",
        "The installed panel area meets demand",
        "The grid connection permits generation exports",
    ]
    cells = [
        Cell(
            id=f"dish-energy-{0 if i == 0 else 1:03d}-{i:03d}",
            colony_id="dish-energy",
            claim_text=text,
            level=0 if i == 0 else 1,
            relation="limits" if i == 3 else "supports",
        )
        for i, text in enumerate(texts)
    ]
    edges = [Edge(from_cell=cells[0].id, to_cell=cell.id) for cell in cells[1:]]
    return DecompositionResult(
        cells=cells, edges=edges, center_claim=texts[0], colony_name="energy"
    )


def persist(path, result):
    graph = ColonyGraph(colony_id="dish-energy")
    for cell in result.cells:
        graph.add_cell(cell)
    for edge in result.edges:
        graph.add_edge(edge)
    save_colony(
        path,
        graph,
        Colony(
            id="dish-energy",
            dish="dish",
            center_claim=result.center_claim,
            center_cell_id=result.cells[0].id,
        ),
    )


def modify(result, operation):
    value = yaml.safe_load(dump_draft(result))
    operation(value["premises"])
    return yaml.safe_dump(value)


def test_edit_text_and_audit_event(tmp_path):
    path = tmp_path / "petri.sqlite"
    original = draft()
    persist(path, original)
    text = modify(
        original,
        lambda rows: rows[0].update(
            claim_text="Measured sunlight exceeds the module startup threshold"
        ),
    )
    edited = validate_draft(text, original, SETTINGS)
    backups = []
    save_draft_edit(
        path, original, edited, SETTINGS, dump_draft(original), text, lambda: backups.append(True)
    )
    assert backups == [True]
    assert database.get_cell(path, original.cells[1].id)["claim_text"] == edited.cells[1].claim_text
    event = database.read_events(path, event_type="decomposition_edited")[0]
    assert original.cells[1].claim_text in event["data"]["original_text"]
    assert edited.cells[1].claim_text in event["data"]["edited_text"]


def test_drop_removes_only_the_selected_leaf():
    original = draft()
    edited = validate_draft(modify(original, lambda rows: rows.pop(1)), original, SETTINGS)
    assert [cell.claim_text for cell in edited.cells] == [
        original.cells[i].claim_text for i in [0, 1, 3]
    ]
    assert len(edited.edges) == 2


@pytest.mark.parametrize(
    "operation,match",
    [
        (lambda rows: rows[0].update(relation="invalid"), "relation"),
        (
            lambda rows: rows[0].update(claim_text="Solar panels reduce imported electricity"),
            "restates",
        ),
        (lambda rows: rows[0].update(is_atomic=True), "bedrock_reason"),
    ],
)
def test_invalid_edit_rejected_before_persistence(operation, match):
    original = draft()
    with pytest.raises(ValueError, match=match):
        validate_draft(modify(original, operation), original, SETTINGS)
    assert original.cells[1].claim_text == "Incident sunlight exceeds the operating minimum"


def test_caps_and_reference_reasons_are_preserved():
    original = draft()
    original.edges.append(
        Edge(
            from_cell=original.cells[1].id,
            to_cell="dish-foundation-001-001",
            edge_type="cross_colony",
            reason="Same measured premise",
        )
    )
    edited = validate_draft(
        dump_draft(original), original, SETTINGS.model_copy(update={"level_one_cap": 2})
    )
    assert len(edited.cells) == 3
    assert edited.edges[-1] == original.edges[-1]


def test_atomic_edit_prunes_descendants():
    original = draft()
    child = Cell(
        id="dish-energy-002-001",
        colony_id="dish-energy",
        level=2,
        claim_text="A separate measured condition",
    )
    original.cells.append(child)
    original.edges.append(Edge(from_cell=original.cells[1].id, to_cell=child.id))
    text = modify(
        original,
        lambda rows: rows[0].update(is_atomic=True, bedrock_reason="Measured directly at the site"),
    )
    edited = validate_draft(text, original, SETTINGS)
    assert edited.cells[1].is_atomic
    assert child.id not in {cell.id for cell in edited.cells}


@pytest.mark.parametrize("phase", ["after_journal", "after_clear"])
def test_interrupted_edit_replays_human_result_instead_of_old_output(tmp_path, monkeypatch, phase):
    from petri.reasoning import seed_edit
    from petri.reasoning.decomposition_runner import decompose_typed

    original = draft()
    path = tmp_path / "petri.sqlite"
    persist(path, original)
    text = modify(
        original,
        lambda rows: (
            rows[0].update(claim_text="Site measurements exceed the startup threshold"),
            rows.pop(1),
        ),
    )
    edited = validate_draft(text, original, SETTINGS)
    def interrupt(*args):
        raise RuntimeError("Interrupted after edit journal")

    with monkeypatch.context() as patch:
        if phase == "after_journal":
            patch.setattr(seed_edit, "materialize_edited_state", interrupt)
        else:
            patch.setattr(seed_edit.database, "put_cell", lambda *a, **kw: interrupt())
        with pytest.raises(RuntimeError, match="after edit journal"):
            save_draft_edit(path, original, edited, SETTINGS, dump_draft(original), text, lambda: None)
    assert len(database.list_cells(path)) == 4

    def unexpected(*args):
        pytest.fail("Resume requested model output after a recorded human edit")

    result = decompose_typed(
        original.center_claim,
        [],
        "dish",
        "energy",
        SimpleNamespace(initial_decomposition=unexpected, expand_why=unexpected),
        settings=SETTINGS,
        db_path=path,
        resume=True,
    )
    assert result == edited
    assert {row["id"] for row in database.list_cells(path)} == {cell.id for cell in edited.cells}


def test_editor_reprompts_on_invalid_yaml_before_any_writes(tmp_path, monkeypatch, capsys):
    from petri.cli import seed_edit

    original = draft()
    path = tmp_path / "petri.sqlite"
    persist(path, original)
    calls = []
    valid = modify(
        original,
        lambda rows: rows[0].update(claim_text="Site measurements exceed the startup threshold"),
    )
    responses = iter(
        ["premises: [", modify(original, lambda rows: rows[0].update(relation="wrong")), valid]
    )

    def editor(*args, **kwargs):
        assert (
            database.get_cell(path, original.cells[1].id)["claim_text"]
            == original.cells[1].claim_text
        )
        return next(responses)

    monkeypatch.setenv("EDITOR", "scripted-editor")
    monkeypatch.setattr(seed_edit.typer, "edit", editor)
    monkeypatch.setattr(
        seed_edit, "snapshot_before_destructive", lambda *args: calls.append("backup")
    )
    result = seed_edit.edit_pending_result(original, SETTINGS, path, tmp_path, "dish", None)
    assert calls == ["backup"]
    assert "Invalid edit" in capsys.readouterr().out
    assert database.get_cell(path, original.cells[1].id)["claim_text"] == result.cells[1].claim_text


def test_cli_edit_choice_uses_editor_without_regeneration(seeded_petri_dir, monkeypatch):
    from click.testing import _NamedTextIOWrapper
    from tests.conftest import TypedFakeProvider as FakeProvider

    provider = FakeProvider()
    provider.questions_response = []
    actions = iter(["Edit", "Accept"])
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda _: True)
    monkeypatch.setattr(
        "questionary.select", lambda *a, **kw: SimpleNamespace(ask=lambda: next(actions))
    )
    monkeypatch.setenv("EDITOR", "scripted-editor")

    def editor(text, **kwargs):
        data = yaml.safe_load(text)
        data["premises"][0]["claim_text"] = "An independently testable edited premise"
        return yaml.safe_dump(data)

    monkeypatch.setattr("petri.cli.seed_edit.typer.edit", editor)
    result = CliRunner().invoke(app, ["seed", "An energy claim", "--colony", "edited"])
    assert result.exit_code == 0, result.output
    assert len(provider.decompose_calls) == 1
    assert "An independently testable edited premise" in result.output


def test_inline_drop_and_edit(monkeypatch):
    from petri.cli.seed_edit import _inline_edit

    answers = iter(["edit 1 Updated measured condition", "drop 2", "relation 2 rebuts", "done"])
    questionary = SimpleNamespace(text=lambda *args: SimpleNamespace(ask=lambda: next(answers)))
    result = yaml.safe_load(_inline_edit(dump_draft(draft()), questionary))
    assert len(result["premises"]) == 2
    assert result["premises"][0]["claim_text"] == "Updated measured condition"
    assert result["premises"][1]["relation"] == "rebuts"


def test_actual_editor_round_trip(tmp_path, monkeypatch):
    import shlex
    import sys

    from petri.cli import seed_edit

    original = draft()
    path = tmp_path / "petri.sqlite"
    persist(path, original)
    script = tmp_path / "editor.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1])\n"
        "p.write_text(p.read_text().replace('Incident sunlight exceeds the operating minimum', "
        "'Measured irradiance exceeds the startup threshold'))\n"
    )
    monkeypatch.setenv("EDITOR", f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    monkeypatch.setattr(seed_edit, "snapshot_before_destructive", lambda *args: None)
    edited = seed_edit.edit_pending_result(original, SETTINGS, path, tmp_path, "dish", None)
    assert edited.cells[1].claim_text == "Measured irradiance exceeds the startup threshold"


def test_approved_colony_rejected_before_backup_or_journal(tmp_path):
    from petri.storage.event_log import append_event

    original = draft()
    path = tmp_path / "petri.sqlite"
    persist(path, original)
    append_event(
        database.SQLiteEventLog(path), original.cells[0].id, "colony_approved", "human", 0, {}
    )
    with pytest.raises(ValueError, match="unapproved"):
        save_draft_edit(
            path,
            original,
            original,
            SETTINGS,
            "original",
            "edited",
            lambda: pytest.fail("Backed up an approved colony for editing"),
        )
    assert not database.read_events(path, event_type="decomposition_edited")


def test_per_parent_cap_retains_a_counterargument():
    original = draft()
    for i, relation in enumerate(["supports", "supports", "rebuts"], 1):
        cell = Cell(
            id=f"dish-energy-002-{i:03d}",
            colony_id="dish-energy",
            level=2,
            claim_text=f"Independent condition number {i}",
            relation=relation,
        )
        original.cells.append(cell)
        original.edges.append(Edge(from_cell=original.cells[1].id, to_cell=cell.id))
    edited = validate_draft(dump_draft(original), original, SETTINGS)
    assert [cell.relation for cell in edited.cells if cell.level == 2] == ["supports", "rebuts"]

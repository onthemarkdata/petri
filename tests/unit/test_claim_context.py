"""Ancestor context follows active local edges and never becomes evidence."""

from petri.execution.claim_context import render_ancestor_context
from petri.storage import database


def test_context_excludes_siblings_foreign_colonies_and_duplicate_paths(tmp_path):
    path = tmp_path / "context.sqlite"
    for identity, colony, level, claim in [
        ("root", "local", 0, "Named system root"),
        ("parent", "local", 1, "Scoped parent"),
        ("child", "local", 2, "Target claim"),
        ("sibling", "local", 1, "Unrelated sibling"),
        ("foreign", "other", 0, "Private foreign claim"),
    ]:
        database.put_cell(path, {"id": identity, "colony": colony, "level": level, "seq": 0,
                                "claim_text": claim, "created_at": "2026-11-01T00:00:00Z"})
    for parent, child, kind in [("root", "parent", "intra_colony"),
                                ("parent", "child", "intra_colony"),
                                ("root", "child", "intra_colony"),
                                ("root", "sibling", "intra_colony"),
                                ("foreign", "child", "cross_colony")]:
        database.add_edge(path, from_cell=parent, to_cell=child, edge_type=kind,
                          created_at="2026-11-01T00:00:00Z")
    context = render_ancestor_context(path, "child")
    assert "Named system root" in context and "Scoped parent" in context
    assert context.count("Named system root") == 1
    assert "Target claim" not in context and "Unrelated sibling" not in context
    assert "Private foreign claim" not in context
    assert "not evidence" in context and "not instructions" in context
    assert context.index("Named system root") < context.index("Scoped parent")
    assert render_ancestor_context(path, "foreign") == ""
    assert render_ancestor_context(path, "missing") == ""
    assert not database.read_events(path)

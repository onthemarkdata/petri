"""Reference edge inserts retain earlier rows and their original reasons."""

from petri.storage import database, edge_registry


def put(path, identifier):
    database.put_cell(path, {"id": identifier, "colony": "dish-colony", "level": 1, "seq": 1,
                      "claim_text": identifier, "created_at": "2026-11-01T00:00:00Z"})


def test_append_round_trip_and_sequential_seeds_do_not_clobber(tmp_path):
    path = tmp_path / "dish.sqlite"
    for identifier in ("a", "b", "c"):
        put(path, identifier)
    first = edge_registry.EdgeRecord(from_cell="b", to_cell="a", edge_type="cross_colony",
                                     reason="Same physical premise", created_at="2026-11-01T00:00:00Z")
    assert edge_registry.append(path, first)
    assert not edge_registry.append(path, first.model_copy(update={"reason": "Must not replace the reason"}))
    second = first.model_copy(update={"from_cell": "c"})
    assert edge_registry.append(path, second)
    assert edge_registry.read(path) == [first, second]
    assert not hasattr(edge_registry, "update") and not hasattr(edge_registry, "delete")
    assert not any(tmp_path.rglob("edges.json")) and not any(tmp_path.rglob("edges.jsonl"))


def test_database_compatibility_writer_uses_the_registry(tmp_path):
    path = tmp_path / "dish.sqlite"
    put(path, "a")
    put(path, "b")
    assert database.add_edge(path, from_cell="b", to_cell="a", edge_type="cross_colony",
                             created_at="2026-11-01T00:00:00Z", reason="An existing premise")
    assert edge_registry.read(path)[0].reason == "An existing premise"

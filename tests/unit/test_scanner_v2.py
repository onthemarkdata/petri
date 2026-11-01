"""Corruption remains visible without rewriting domain records."""

from contextlib import closing

from petri.analysis.scanner import scan, scan_database
from petri.storage import database


def seed(tmp_path):
    path = tmp_path / "petri-dishes/dish/petri.sqlite"
    database.put_cell(path, {"id": "dish-colony-001-000", "colony": "dish-colony",
        "level": 1, "seq": 0, "claim_text": "A test claim", "created_at": "2026-01-01"})
    return path


def corrupt(path):
    with closing(database.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("INSERT INTO edges VALUES ('dish-colony-001-000','missing','intra_colony','','2026-01-01')")
        connection.execute("INSERT INTO events VALUES ('bad','cell','search_executed','agent',0,'{}','2026-01-01')")
        connection.execute("UPDATE cells SET status='UNKNOWN'")
        connection.commit()


def test_phantom_edge_and_registered_event_payload(tmp_path):
    path = seed(tmp_path)
    assert scan(tmp_path) == []
    corrupt(path)
    before = path.read_bytes()
    issues = scan_database(path)
    assert {issue.category for issue in issues} == {"phantom_edge", "event_payload", "cell_status"}
    assert "missing" in next(issue.description for issue in issues if issue.category == "phantom_edge")
    assert issues == scan(tmp_path)
    assert path.read_bytes() == before


def test_no_database_created_and_generated_files_ignored(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    file = generated / "agent.md"
    file.write_text("Unrecognized agent")
    assert scan(tmp_path, generated) == []
    assert file.read_text() == "Unrecognized agent"
    assert not (tmp_path / "petri-dishes").exists()

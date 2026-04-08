"""Unit tests for :mod:`petri.storage.paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from petri.storage.paths import (
    cell_dir,
    colony_dir,
    events_path,
    iter_events_files,
    metadata_path,
    parse_cell_id,
)


# ── parse_cell_id ────────────────────────────────────────────────────────


def test_parse_cell_id_simple():
    """A four-part cell ID parses cleanly."""
    dish, colony_slug, level, seq = parse_cell_id("dish-colony-001-002")
    assert dish == "dish"
    assert colony_slug == "colony"
    assert level == 1
    assert seq == 2


def test_parse_cell_id_zero_level_and_seq():
    """Center cells use level=0, seq=0."""
    dish, colony_slug, level, seq = parse_cell_id("mydish-mycol-000-000")
    assert dish == "mydish"
    assert colony_slug == "mycol"
    assert level == 0
    assert seq == 0


def test_parse_cell_id_with_hyphens_in_colony_slug():
    """A multi-hyphen colony slug survives parsing.

    With more than four parts, the first segment is treated as the
    dish and everything between the dish and the final two (level,
    seq) segments is rejoined as the colony slug. This matches the
    right-to-left parsing convention used in
    :func:`petri.models.parse_key`.
    """
    dish, colony_slug, level, seq = parse_cell_id(
        "my-dish-my-colony-003-004"
    )
    # Lossy heuristic: without knowing the dish_id we cannot
    # disambiguate a multi-hyphen dish from a multi-hyphen colony.
    # First segment becomes the dish; the remainder (before level+seq)
    # becomes the colony slug.
    assert dish == "my"
    assert colony_slug == "dish-my-colony"
    assert level == 3
    assert seq == 4


def test_parse_cell_id_five_parts():
    """A five-part ID (single-word dish, two-word colony) parses."""
    dish, colony_slug, level, seq = parse_cell_id("dish-foo-bar-010-099")
    assert dish == "dish"
    assert colony_slug == "foo-bar"
    assert level == 10
    assert seq == 99


def test_parse_cell_id_raises_on_too_few_parts():
    """Fewer than four hyphen-separated parts is invalid."""
    with pytest.raises(ValueError, match="fewer than 4"):
        parse_cell_id("foo-bar")


def test_parse_cell_id_raises_on_three_parts():
    """Exactly three parts still fails the 4-part minimum."""
    with pytest.raises(ValueError, match="fewer than 4"):
        parse_cell_id("dish-colony-001")


def test_parse_cell_id_raises_on_non_integer_level():
    """Non-integer level segment is invalid."""
    with pytest.raises(ValueError, match="level segment"):
        parse_cell_id("dish-colony-abc-002")


def test_parse_cell_id_raises_on_non_integer_seq():
    """Non-integer seq segment is invalid."""
    with pytest.raises(ValueError, match="seq segment"):
        parse_cell_id("dish-colony-001-xyz")


def test_parse_cell_id_raises_on_non_string():
    """Non-string input raises ValueError."""
    with pytest.raises(ValueError, match="must be a string"):
        parse_cell_id(12345)  # type: ignore[arg-type]


# ── colony_dir ───────────────────────────────────────────────────────────


def test_colony_dir_strips_dish_prefix():
    """colony_id with dish prefix is stripped to produce the right dir."""
    result = colony_dir(Path("/p"), "dish1", "dish1-mycolony")
    assert result == Path("/p/petri-dishes/mycolony")


def test_colony_dir_without_dish_prefix():
    """Bare colony slug is used verbatim."""
    result = colony_dir(Path("/p"), "dish1", "mycolony")
    assert result == Path("/p/petri-dishes/mycolony")


def test_colony_dir_only_strips_first_matching_prefix():
    """A colony slug that merely contains the dish ID is unchanged."""
    # 'otherdish-col' does not start with 'dish1-', so no strip.
    result = colony_dir(Path("/p"), "dish1", "otherdish-col")
    assert result == Path("/p/petri-dishes/otherdish-col")


def test_colony_dir_with_multihyphen_dish_id():
    """A multi-hyphen dish ID is stripped as a single prefix."""
    result = colony_dir(Path("/p"), "my-dish", "my-dish-mycol")
    assert result == Path("/p/petri-dishes/mycol")


def test_colony_dir_with_multihyphen_colony_slug():
    """A multi-hyphen colony slug is preserved after strip."""
    result = colony_dir(Path("/p"), "dish1", "dish1-foo-bar-baz")
    assert result == Path("/p/petri-dishes/foo-bar-baz")


# ── cell_dir ─────────────────────────────────────────────────────────────


def test_cell_dir_zero_pads_level_and_seq():
    """level=0, seq=5 produces '000-005'."""
    assert cell_dir(Path("/c"), 0, 5) == Path("/c/000-005")


def test_cell_dir_pads_three_digits():
    """Single-digit level and seq both get zero-padded to 3 digits."""
    assert cell_dir(Path("/colony"), 1, 2) == Path("/colony/001-002")


def test_cell_dir_does_not_truncate_large_values():
    """Values requiring more than 3 digits are not truncated."""
    assert cell_dir(Path("/c"), 10, 999) == Path("/c/010-999")
    assert cell_dir(Path("/c"), 100, 1000) == Path("/c/100-1000")


# ── events_path / metadata_path ──────────────────────────────────────────


def test_events_path():
    """events_path simply appends 'events.jsonl'."""
    assert events_path(Path("/c/001-002")) == Path("/c/001-002/events.jsonl")


def test_metadata_path():
    """metadata_path simply appends 'metadata.json'."""
    assert metadata_path(Path("/c/001-002")) == Path("/c/001-002/metadata.json")


def test_events_and_metadata_compose_with_cell_dir():
    """The helpers compose with cell_dir to rebuild full paths."""
    cell_path = cell_dir(Path("/p/petri-dishes/col"), 2, 7)
    assert events_path(cell_path) == Path(
        "/p/petri-dishes/col/002-007/events.jsonl"
    )
    assert metadata_path(cell_path) == Path(
        "/p/petri-dishes/col/002-007/metadata.json"
    )


# ── iter_events_files ────────────────────────────────────────────────────


def test_iter_events_files_yields_only_events_jsonl(tmp_path):
    """Only files literally named 'events.jsonl' are yielded."""
    dishes_dir = tmp_path / "petri-dishes"
    colony_a = dishes_dir / "colony-a"
    colony_b = dishes_dir / "colony-b"
    cell_a = colony_a / "000-000"
    cell_b1 = colony_b / "001-001"
    cell_b2 = colony_b / "001-002"
    for directory in (cell_a, cell_b1, cell_b2):
        directory.mkdir(parents=True)

    (cell_a / "events.jsonl").write_text("")
    (cell_a / "metadata.json").write_text("{}")
    (cell_a / "evidence.md").write_text("")
    (cell_b1 / "events.jsonl").write_text("")
    (cell_b2 / "events.jsonl").write_text("")
    (cell_b2 / "notes.txt").write_text("ignored")
    (colony_a / "colony.json").write_text("{}")

    found = sorted(iter_events_files(tmp_path))
    expected = sorted(
        [
            cell_a / "events.jsonl",
            cell_b1 / "events.jsonl",
            cell_b2 / "events.jsonl",
        ]
    )
    assert found == expected


def test_iter_events_files_handles_missing_dishes_dir(tmp_path):
    """A petri_dir without petri-dishes/ yields nothing, does not raise."""
    # tmp_path exists but has no petri-dishes subdir.
    result = list(iter_events_files(tmp_path))
    assert result == []


def test_iter_events_files_handles_missing_petri_dir(tmp_path):
    """A petri_dir that does not exist yields nothing, does not raise."""
    nonexistent = tmp_path / "does-not-exist"
    result = list(iter_events_files(nonexistent))
    assert result == []


def test_iter_events_files_handles_empty_dishes_dir(tmp_path):
    """An empty petri-dishes/ directory yields nothing."""
    (tmp_path / "petri-dishes").mkdir()
    result = list(iter_events_files(tmp_path))
    assert result == []


def test_iter_events_files_file_instead_of_dir(tmp_path):
    """If petri-dishes exists as a file (not a dir), nothing is yielded."""
    (tmp_path / "petri-dishes").write_text("not a directory")
    result = list(iter_events_files(tmp_path))
    assert result == []

"""Regression tests for ``_log_sources_from_result``.

The first parameter was previously named ``events_p`` while the body
referenced ``events_file``, causing a ``NameError`` the first time any
agent returned a ``sources_cited`` payload.  These tests lock the call
signature + runtime behaviour.
"""

from __future__ import annotations

import json


def test_log_sources_from_result_does_not_raise_name_error(tmp_path):
    """Regression — events_file param was misnamed events_p."""
    from petri.engine.processor import _log_sources_from_result

    events_file = tmp_path / "events.jsonl"
    fake_result = type("R", (), {
        "sources_cited": [
            {
                "url": "https://example.com",
                "title": "T",
                "hierarchy_level": 2,
                "finding": "f",
                "supports_or_contradicts": "supports",
            }
        ]
    })()
    _log_sources_from_result(
        events_file, "dish-colony-001-001", "investigator", 1, fake_result,
    )
    assert events_file.exists()


def test_log_sources_from_result_writes_source_reviewed_event(tmp_path):
    """Happy path — one source in, one source_reviewed event out."""
    from petri.engine.processor import _log_sources_from_result

    events_file = tmp_path / "events.jsonl"
    fake_result = {
        "sources_cited": [
            {
                "url": "https://example.org/paper",
                "title": "Paper Title",
                "hierarchy_level": 1,
                "finding": "direct measurement",
                "supports_or_contradicts": "supports",
            }
        ]
    }
    _log_sources_from_result(
        events_file, "dish-colony-002-003", "red_team_lead", 2, fake_result,
    )
    lines = events_file.read_text().strip().splitlines()
    assert len(lines) == 1
    event_record = json.loads(lines[0])
    assert event_record["type"] == "source_reviewed"
    assert event_record["agent"] == "red_team_lead"
    assert event_record["iteration"] == 2
    assert event_record["data"]["url"] == "https://example.org/paper"

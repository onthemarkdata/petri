"""Keep the contributor lifecycle map and generated diagram tied to code."""

import re
from pathlib import Path

from tests.unit.test_pipeline_conformance import LEGACY_TRANSITIONS
from tests.generate_pipeline_diagram import diagram_block, diagram_text, update_docs

ROOT = Path(__file__).resolve().parents[2]


def test_generated_diagram_is_current():
    assert (ROOT / "docs/pipeline-graph.mmd").read_text() == diagram_text()
    assert diagram_block() in (ROOT / "docs/pipeline-graph.md").read_text()


def test_mapping_covers_every_v1_state_once():
    page = (ROOT / "docs/pipeline-graph.md").read_text()
    rows = re.findall(r"^\| `([^`]+)` \|", page, flags=re.MULTILINE)
    assert len(rows) == len(set(rows)) == 14
    assert set(rows) == set(LEGACY_TRANSITIONS)


def test_regenerator_preserves_surrounding_prose(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    original = (ROOT / "docs/pipeline-graph.md").read_text()
    (docs / "pipeline-graph.md").write_text(original)
    update_docs(tmp_path)
    assert (docs / "pipeline-graph.md").read_text() == original
    assert (docs / "pipeline-graph.mmd").read_text() == diagram_text()

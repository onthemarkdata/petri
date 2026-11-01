"""Keep the documented telemetry configuration and attributes in sync."""

import re
from pathlib import Path

import yaml

from petri.telemetry.config import TelemetryConfig

ROOT = Path(__file__).parents[2]


def test_complete_reference_matches_all_config_fields_and_defaults():
    text = (ROOT / 'docs/observability.md').read_text()
    snippet = re.search(r'<!-- telemetry-config-reference -->\s*```yaml\n(.*?)```', text, re.S)[1]
    raw = yaml.safe_load(snippet)['telemetry']
    actual = TelemetryConfig.model_validate(raw)
    assert actual == TelemetryConfig()
    assert raw == actual.model_dump()


def test_attribute_table_matches_architecture():
    tables = []
    for filename in ['docs/observability.md', 'docs/ARCHITECTURE-V2.md']:
        text = (ROOT / filename).read_text()
        start = text.index('| Attribute | Type | Present when |')
        tables.append(text[start:].split('\n\n', 1)[0])
    assert tables[0] == tables[1]


def test_docs_link_and_architecture_tree():
    assert '(docs/observability.md)' in (ROOT / 'README.md').read_text()
    assert '├── telemetry/' in (ROOT / 'CLAUDE.md').read_text()

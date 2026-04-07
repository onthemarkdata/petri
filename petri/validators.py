"""Source hierarchy enforcement for terminal decisions.

Ensures that nodes can only reach a terminal verdict (VALIDATED or DISPROVEN)
when backed by sufficiently strong evidence -- at least one source at
hierarchy Level 1-4 (direct measurement, authoritative docs, derived
calculation, or corroborated expert consensus).
"""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from petri.event_log import load_events


def load_source_hierarchy(config_path: Path) -> dict:
    """Load source hierarchy from petri.yaml.

    *config_path* should point to a directory containing ``petri.yaml``
    (or legacy ``source_hierarchy.yaml``).  Returns the ``source_hierarchy``
    section.  If PyYAML is not installed or the file does not exist, returns
    sensible defaults.
    """
    if yaml is None:
        from petri.config import get_minimum_terminal_level
        return {"minimum_terminal_level": get_minimum_terminal_level(), "levels": {}}

    # Try consolidated petri.yaml first
    petri_yaml = config_path / "petri.yaml"
    if petri_yaml.exists():
        with open(petri_yaml) as f:
            raw = yaml.safe_load(f) or {}
        if "source_hierarchy" in raw:
            return raw["source_hierarchy"]

    # Legacy fallback
    hierarchy_file = config_path / "source_hierarchy.yaml"
    if not hierarchy_file.exists():
        from petri.config import get_minimum_terminal_level
        return {"minimum_terminal_level": get_minimum_terminal_level(), "levels": {}}

    with open(hierarchy_file) as f:
        return yaml.safe_load(f) or {}


def validate_terminal_sources(
    events_path: Path,
    node_id: str,
    min_level: int | None = None,
) -> dict:
    """Validate that a node has Level 1-4 sources for terminal decisions.

    Scans ``source_reviewed`` events for the given *node_id* in the JSONL file
    at *events_path*.  A terminal decision (VALIDATED or DISPROVEN) requires at
    least one source with ``hierarchy_level`` between 1 and *min_level*
    inclusive.

    Returns a dict with:
    - **pass** (``bool``): Whether the threshold is met.
    - **details** (``str``): Human-readable explanation.
    - **sources** (``list``): The matching ``source_reviewed`` events.
    - **highest_level** (``int | None``): The numerically lowest (strongest)
      hierarchy level found, or ``None`` if no levels are recorded.
    """
    events = load_events(events_path)
    sources = [
        event
        for event in events
        if event.get("type") == "source_reviewed" and event.get("node_id") == node_id
    ]

    if not sources:
        return {
            "pass": False,
            "details": "No source_reviewed events found for node",
            "sources": [],
            "highest_level": None,
        }

    levels: list[int] = []
    for source in sources:
        data = source.get("data", {})
        hl = data.get("hierarchy_level")
        if hl is not None:
            levels.append(int(hl))

    if not levels:
        return {
            "pass": False,
            "details": "Sources found but none have hierarchy_level assigned",
            "sources": sources,
            "highest_level": None,
        }

    if min_level is None:
        from petri.config import get_minimum_terminal_level
        min_level = get_minimum_terminal_level()

    highest = min(levels)  # Lower number = higher quality
    passes = highest <= min_level

    return {
        "pass": passes,
        "details": (
            f"Highest source level: {highest} "
            f"({'sufficient' if passes else 'insufficient'} for terminal decision)"
        ),
        "sources": sources,
        "highest_level": highest,
    }

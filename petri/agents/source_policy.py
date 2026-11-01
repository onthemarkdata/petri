"""Pure evidence-strength policy for terminal research verdicts."""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


class SourcePolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passes: bool
    strongest_level: int | None
    detail: str


def check_terminal_sources(sources: Iterable[Any], min_level: int) -> SourcePolicyResult:
    """Require at least one source at or above the configured credibility threshold."""
    if isinstance(min_level, bool) or not isinstance(min_level, int) or not 1 <= min_level <= 6:
        raise ValueError("minimum_terminal_level must be an integer from 1 through 6.")
    levels = []
    for source in sources:
        value = (
            source.get("hierarchy_level")
            if isinstance(source, Mapping)
            else getattr(source, "hierarchy_level", None)
        )
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 6:
            levels.append(value)
    strongest = min(levels, default=None)
    passes = strongest is not None and strongest <= min_level
    detail = (
        f"Source hierarchy policy: cite at least one level 1–{min_level} source for a terminal verdict. "
        f"Strongest supplied level: {strongest if strongest is not None else 'none'}."
    )
    return SourcePolicyResult(passes=passes, strongest_level=strongest, detail=detail)

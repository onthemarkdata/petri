"""Deterministic lexical search over the current dish's persisted cells."""

import re
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from petri.models import CellStatus
from petri.storage import database


class CellSearchCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    claim_text: str
    colony: str
    level: int = Field(ge=0)
    status: CellStatus
    score: float = Field(ge=0, le=1)


def search(db_path: Path, query: str, top_k: int = 5, *, exclude_colony: str | None = None) -> list[CellSearchCandidate]:
    """Rank positive word-token overlap by score, then exact phrase, then cell ID."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    tokens = set(re.findall(r"\w+", query.casefold()))
    if not tokens or not Path(db_path).is_file():
        return []
    found = []
    with closing(database.connect(db_path)) as conn:
        for row in conn.execute(
                "SELECT id, claim_text, colony, level, status FROM cells WHERE (? IS NULL OR colony <> ?)",
                (exclude_colony, exclude_colony)):
            words = set(re.findall(r"\w+", row["claim_text"].casefold()))
            overlap = len(tokens & words) / len(tokens | words)
            if overlap:
                found.append(CellSearchCandidate(**dict(row), score=overlap))
    normalized = " ".join(query.casefold().split())
    found.sort(key=lambda cell: (-cell.score, " ".join(cell.claim_text.casefold().split()) != normalized, cell.id))
    return found[:top_k]

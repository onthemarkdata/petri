"""Append and read typed dependency edges through one storage seam.

The registry does not detect cycles across colonies. Dish-wide cycle detection
and tombstones belong to the lifecycle layer; no update/delete API is exposed.
"""

from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from petri.storage import database


class EdgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    from_cell: str = Field(min_length=1)
    to_cell: str = Field(min_length=1)
    edge_type: Literal["intra_colony", "cross_colony"] = "intra_colony"
    reason: str = ""
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_self_reference(self):
        if self.from_cell == self.to_cell:
            raise ValueError("A cell cannot reference itself")
        return self


def append(db_path: Path, edge: EdgeRecord) -> bool:
    """Append once; replaying the same dependency leaves its original row intact."""
    with database._write(db_path) as conn:
        result = conn.execute(
            "INSERT INTO edges VALUES (?,?,?,?,?) ON CONFLICT(from_cell,to_cell,edge_type) DO NOTHING",
            (edge.from_cell, edge.to_cell, edge.edge_type, edge.reason, edge.created_at),
        )
        return result.rowcount == 1


def read(db_path: Path) -> list[EdgeRecord]:
    """Read dependency records in a stable order."""
    with closing(database.connect(db_path)) as conn:
        return [EdgeRecord.model_validate(dict(row)) for row in conn.execute(
            "SELECT * FROM edges ORDER BY from_cell,to_cell,edge_type",
        )]

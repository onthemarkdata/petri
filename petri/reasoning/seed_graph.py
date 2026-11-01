"""Serializable seed nodes and state, independent of the execution backend."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from petri.models import Cell, Edge
from petri.reasoning.decomposition_settings import DecompositionSettings


class InitialNode(BaseModel):
    kind: Literal["initial"] = "initial"


class ExpansionNode(BaseModel):
    kind: Literal["expand"] = "expand"
    cell_id: str


def _initial_nodes() -> list[InitialNode | ExpansionNode]:
    return [InitialNode()]


class SeedState(BaseModel):
    """The in-memory worklist is rebuilt from committed output events on resume."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    colony_id: str
    attempt: int = Field(ge=0)
    claim: str
    guidance: str = ""
    settings: DecompositionSettings
    cells: list[Cell]
    edges: list[Edge] = Field(default_factory=list)
    pending: list[InitialNode | ExpansionNode] = Field(default_factory=_initial_nodes)

    @property
    def complete(self) -> bool:
        return not self.pending

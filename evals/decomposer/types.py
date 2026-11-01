"""Typed inputs and outputs shared by decomposition cases and evaluators."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from petri.models import DecompositionResult


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: Literal["organic", "axiom", "reference", "negative"]
    claims: list[str] = Field(min_length=1)
    layer_sizes: list[list[PositiveInt]] = Field(default_factory=list)
    layer_ceiling: int = Field(default=13, gt=0)
    axiom_texts: list[str] = Field(default_factory=list)
    fixture: str | None = None

    @model_validator(mode="after")
    def validate_case_shape(self):
        if self.kind != "negative" and (len(self.layer_sizes) != len(self.claims) or not all(self.layer_sizes)):
            raise ValueError("Each positive claim requires a nonempty layer-size plan")
        if self.kind == "axiom" and not self.axiom_texts:
            raise ValueError("An axiom case requires the expected axiom text")
        if self.kind == "reference" and len(self.claims) != 2:
            raise ValueError("A reference case requires two claims")
        return self


class EvalBatch(BaseModel):
    results: list[DecompositionResult]
    expected_reference_ids: set[str] = Field(default_factory=set)
    trivial_cell_ids: set[str] = Field(default_factory=set)

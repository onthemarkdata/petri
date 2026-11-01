"""Validated decomposition results, distinct from persisted cell identities."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, StringConstraints, Tag, model_validator

from petri.structured_output import NamedOutput

ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PremiseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: ClaimText
    relation: Literal["supports", "limits", "rebuts"] = "supports"


class ReferencePremise(BaseModel):
    model_config = ConfigDict(extra="forbid")
    existing_cell_id: ClaimText
    reason: ClaimText


def premise_kind(value) -> str:
    if isinstance(value, ReferencePremise) or isinstance(value, dict) and "existing_cell_id" in value:
        return "reference"
    return "new"


Premise = Annotated[
    Annotated[PremiseCandidate, Tag("new")] | Annotated[ReferencePremise, Tag("reference")],
    Discriminator(premise_kind),
]


class InitialDecomposition(NamedOutput):
    model_config = ConfigDict(extra="forbid")
    nodes: list[Premise] = Field(min_length=1)


class WhyExpansion(NamedOutput):
    model_config = ConfigDict(extra="forbid")
    is_atomic: bool = Field(strict=True)
    bedrock_reason: ClaimText | None = None
    sub_premises: list[Premise]

    @model_validator(mode="after")
    def require_bedrock_reason(self):
        if self.is_atomic:
            if self.bedrock_reason is None:
                raise ValueError("An atomic premise requires bedrock_reason.")
            if self.sub_premises:
                raise ValueError("An atomic premise cannot have sub-premises.")
        return self

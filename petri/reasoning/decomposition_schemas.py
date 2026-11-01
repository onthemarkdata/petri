"""Validated decomposition results, distinct from persisted cell identities."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PremiseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: ClaimText
    relation: str = "supports"


class InitialDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[PremiseCandidate] = Field(min_length=1)


class WhyExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_premises: list[PremiseCandidate]
    is_atomic: bool = Field(strict=True)
    bedrock_reason: ClaimText | None = None

    @model_validator(mode="after")
    def require_bedrock_reason(self):
        if self.is_atomic:
            if self.bedrock_reason is None:
                raise ValueError("An atomic premise requires bedrock_reason.")
            if self.sub_premises:
                raise ValueError("An atomic premise cannot have sub-premises.")
        return self

"""Public retrieval is disabled until the dish explicitly enables it."""

from pydantic import BaseModel, ConfigDict


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    public_web: bool = False

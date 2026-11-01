"""Limits for opt-in verdict-driven decomposition."""

from pydantic import BaseModel, ConfigDict, Field


class LifecycleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_rounds_per_cell: int = Field(default=2, ge=0)
    per_parent_cap: int = Field(default=4, ge=1)
    stop_at_bedrock: bool = True
    dish_proposal_budget: int = Field(default=100, ge=0)
    auto_approve: bool = False

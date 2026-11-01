"""Typed dashboard submissions and durable run status at the backend boundary."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from petri.models import ClarifyingQuestion


class SeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = Field(min_length=1, max_length=20_000)
    colony: str | None = Field(default=None, max_length=100)
    regenerate: bool = False
    guidance: str = Field(default="", max_length=20_000)
    clarifications: list[dict[str, Any]] = Field(default_factory=list, max_length=100)

    @field_validator("claim")
    @classmethod
    def nonblank_claim(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Provide a nonempty claim.")
        return value.strip()

    @field_validator("clarifications")
    @classmethod
    def valid_clarifications(cls, values):
        return [
            ClarifyingQuestion.model_validate(value).model_dump(mode="json") for value in values
        ]

    @model_validator(mode="after")
    def colony_name(self):
        name = self.colony
        if name is None:
            words = re.findall(r"[a-z0-9]+", self.claim.lower())[:6]
            name = "-".join(words)[:80].strip("-") or "research"
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
            raise ValueError("Use lowercase words, numbers, and hyphens for the colony name.")
        object.__setattr__(self, "colony", name)
        return self


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: Literal["grow", "cell", "revalidate"] = "grow"
    colony: str | None = None
    cell_id: str | None = None
    cell_ids: list[str] = Field(default_factory=list)
    today: date = Field(default_factory=date.today)
    round_number: int = Field(default=1, ge=1)
    request_id: str | None = None

    @model_validator(mode="after")
    def require_cell(self):
        if self.command in {"cell", "revalidate"} and not self.cell_id:
            raise ValueError("A cell ID is required for this command.")
        return self


class RunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    command: str
    status: str
    created_at: int | None = None
    updated_at: int | None = None
    result: Any = None
    error: str | None = None


class RunReceipt(BaseModel):
    id: str
    workflow_id: str
    status: str
    status_url: str
    colony_id: str | None = None

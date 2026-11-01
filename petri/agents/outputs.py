"""Agent-specific output schemas independent of the legacy configuration loader."""

from typing import Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
    field_validator,
)

from petri.agents.config import AgentContent
from petri.agents.contract import AgentName


class SourceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str = ""
    hierarchy_level: int = Field(default=6, ge=1, le=6)
    finding: str = ""
    supports_or_contradicts: Literal["supports", "contradicts"] = "supports"
    confidence: str | None = None
    pub_date: str = ""
    pub_year: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        TypeAdapter(AnyHttpUrl).validate_python(value)
        return value


class AgentAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    verdict: str
    summary: str = ""
    arguments: str = ""
    evidence: str = ""
    confidence: str = ""
    sources_cited: list[SourceCitation] = Field(default_factory=list)


def output_model(name: AgentName, content: AgentContent) -> type[AgentAssessment]:
    """Constrain agent identity and verdict to this validated content's vocabulary."""
    vocabulary = tuple(content.verdicts_pass + content.verdicts_block)
    return create_model(
        "".join(part.title() for part in name.value.split("_")) + "Assessment",
        __base__=AgentAssessment,
        agent=(Literal[name.value], name),  # type: ignore[valid-type]
        verdict=(Literal[vocabulary], ...),  # type: ignore[valid-type]
    )

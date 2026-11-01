"""Strict v2 content loading with explicit paths and no import-time configuration."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from petri.execution.config import ExecutionConfig

from petri.agents.contract import AGENT_SPECS, AgentName, BlockingMode, DebateProtocol
from petri.telemetry.config import ModelPrice, TelemetryConfig

Nonempty = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class AgentConfigError(ValueError):
    """The requested v2 configuration cannot be loaded or validated."""


class StrictContent(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentContent(StrictContent):
    instruction: str
    verdicts_pass: list[Nonempty]
    verdicts_block: list[Nonempty]
    model: Nonempty | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_structure(cls, data):
        if isinstance(data, dict):
            moved = sorted(
                set(data)
                & {
                    "is_lead",
                    "phase",
                    "blocking",
                    "blocking_mode",
                    "scope",
                    "redirect_on_block",
                    "display_name",
                }
            )
            if moved:
                raise ValueError(
                    f"Structural keys {moved} moved to code; see "
                    "docs/ARCHITECTURE-V2.md#v2-agent-content-schema."
                )
        return data

    @model_validator(mode="after")
    def validate_vocabulary(self):
        if not self.verdicts_pass:
            raise ValueError("verdicts_pass must contain at least one verdict.")
        for key in ("verdicts_pass", "verdicts_block"):
            values = getattr(self, key)
            if len(values) != len(set(values)):
                raise ValueError(f"{key} contains duplicate verdicts.")
        if set(self.verdicts_pass) & set(self.verdicts_block):
            raise ValueError("Pass and block verdict vocabularies must not overlap.")
        return self


class UsageLimitConfig(StrictContent):
    request_limit: PositiveInt
    total_tokens_limit: PositiveInt
    tool_calls_limit: PositiveInt | None = None

    def usage_kwargs(self) -> dict[str, int]:
        """Return keyword arguments accepted by pydantic-ai UsageLimits."""
        return self.model_dump(exclude_none=True)


class LimitsConfig(StrictContent):
    run: UsageLimitConfig
    debate: UsageLimitConfig


class HarnessContent(StrictContent):
    backend: Literal["pi", "claude-code", "provider"]
    model: Nonempty | None = None
    provider: Nonempty | None = None
    tools: list[Nonempty] = Field(default_factory=list)


class DebateContent(StrictContent):
    pair: tuple[AgentName, AgentName]
    rounds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    purpose: Nonempty

    @model_validator(mode="after")
    def validate_pairing(self):
        if self.pair[0] == self.pair[1]:
            raise ValueError("Debate participants must be distinct.")
        if self.rounds * 2 != int(self.rounds * 2):
            raise ValueError("Debate rounds must be whole or half rounds.")
        return self

    @property
    def protocol(self) -> DebateProtocol:
        return DebateProtocol(
            pairing=self.pair,
            rounds=ceil(self.rounds),
            final_word=self.pair[0] if self.rounds % 1 else None,
            purpose=self.purpose,
        )


class SourceLevel(StrictContent):
    name: Nonempty
    description: Nonempty
    terminal_eligible: bool


class SourceHierarchy(StrictContent):
    minimum_terminal_level: Annotated[int, Field(strict=True, ge=1, le=6)]
    levels: dict[int, SourceLevel]

    @model_validator(mode="after")
    def validate_levels(self):
        if set(self.levels) != set(range(1, 7)):
            raise ValueError("source_hierarchy.levels must define levels 1 through 6.")
        for level, content in self.levels.items():
            if content.terminal_eligible != (level <= self.minimum_terminal_level):
                raise ValueError(
                    f"source_hierarchy.levels.{level}.terminal_eligible contradicts "
                    "minimum_terminal_level."
                )
        return self


class PrePipelineContent(StrictContent):
    socratic_questioner: AgentContent


class AgentConfig(StrictContent):
    auto_export: bool = False
    schema_version: Literal[2]
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    name: Nonempty = "petri"
    model: Nonempty
    harness: HarnessContent
    agents: dict[AgentName, AgentContent]
    pre_pipeline: PrePipelineContent
    debates: list[DebateContent]
    source_hierarchy: SourceHierarchy
    agent_tools: list[Nonempty]
    max_iterations: PositiveInt
    max_decomposition_depth: PositiveInt = 3
    max_nodes_per_layer: PositiveInt = 5
    per_parent_cap: PositiveInt = 4
    level_one_cap: PositiveInt = 10
    limits: LimitsConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    prices: dict[str, ModelPrice] = Field(default_factory=dict)

    @field_validator("agents", mode="before")
    @classmethod
    def validate_roster_keys(cls, data):
        if not isinstance(data, dict):
            raise ValueError("agents must be a mapping keyed by AgentName.")  # noqa: TRY004
        expected = {name.value for name in AgentName}
        unknown = sorted(set(data) - expected)
        missing = sorted(expected - set(data))
        if unknown or missing:
            raise ValueError(
                f"Agent roster mismatch: unknown keys={unknown}; missing keys={missing}."
            )
        return data

    @field_validator("agents")
    @classmethod
    def validate_roster_vocabulary(cls, agents):
        for name, content in agents.items():
            spec = AGENT_SPECS[name]
            for partition in ("verdicts_pass", "verdicts_block"):
                extra = sorted(set(getattr(content, partition)) - set(getattr(spec, partition)))
                if extra:
                    raise ValueError(
                        f"agents.{name}.{partition}: absent from code partition={extra}; "
                        f"allowed={list(getattr(spec, partition))}."
                    )
            if spec.blocking_mode != BlockingMode.NON_BLOCKING and not content.verdicts_block:
                raise ValueError(f"agents.{name}.verdicts_block must not be empty.")
        return agents


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently replacing prior content."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key!r}.")
                result[key] = self.construct_object(value_node, deep=deep)
            except TypeError as error:
                raise ValueError("YAML mapping keys must be scalar values.") from error
        return result


def load_agent_config(path: Path) -> AgentConfig:
    """Load exactly the requested file; missing or invalid files never fall back."""
    path = Path(path)
    try:
        data = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        return AgentConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as error:
        raise AgentConfigError(f"Cannot load v2 agent config {path}: {error}") from error

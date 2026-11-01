"""Load decomposition limits from the current dish for each seed operation."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from petri import config


class DecompositionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int = Field(gt=0)
    per_layer: int = Field(gt=0)
    per_parent_cap: int = Field(default=4, gt=0, validate_default=True)
    level_one_cap: int = Field(default=10, gt=0, validate_default=True)
    model: str = Field(min_length=1)

    @field_validator("per_parent_cap", "level_one_cap")
    @classmethod
    def fit_layer_ceiling(cls, value: int, info: ValidationInfo) -> int:
        return min(value, info.data.get("per_layer", value))

    @classmethod
    def defaults(cls) -> "DecompositionSettings":
        """Resolve packaged defaults at call time for callers without a dish."""
        return cls.from_config(config.load_config())

    @classmethod
    def from_config(cls, values):
        return cls(
            depth=values.max_decomposition_depth,
            per_layer=values.max_nodes_per_layer,
            per_parent_cap=values.per_parent_cap,
            level_one_cap=values.level_one_cap,
            model=values.model,
        )

    @classmethod
    def from_dish(cls, petri_dir: Path) -> "DecompositionSettings":
        """Read this dish afresh; omitted settings inherit packaged values."""
        return cls.from_config(config.load_dish_config(petri_dir))

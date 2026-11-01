"""Load decomposition limits from the current dish for each seed operation."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from petri import config


class DecompositionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int = Field(gt=0)
    per_layer: int = Field(gt=0)
    per_parent_cap: int | None = Field(default=None, gt=0)
    model: str = Field(min_length=1)

    @classmethod
    def defaults(cls) -> "DecompositionSettings":
        """Resolve packaged defaults at call time for callers without a dish."""
        return cls(
            depth=config.get_max_decomposition_depth(),
            per_layer=config.get_max_nodes_per_layer(),
            model=config.get_model_name(),
        )

    @classmethod
    def from_dish(cls, petri_dir: Path) -> "DecompositionSettings":
        """Read this dish afresh; omitted settings inherit packaged values."""
        values = {**config.load_config(), **config.load_dish_config(petri_dir)}
        return cls(
            depth=values["max_decomposition_depth"],
            per_layer=values["max_nodes_per_layer"],
            per_parent_cap=values.get("per_parent_cap"),
            model=config.get_model_name(values),
        )

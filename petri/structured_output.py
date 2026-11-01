"""Normalize one known model-name envelope before strict field validation."""

from pydantic import BaseModel, ConfigDict, model_validator


class NamedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def unwrap_named_result(cls, value):
        if isinstance(value, dict) and len(value) == 1:
            name = cls.__name__
            for key in (name, name[:1].lower() + name[1:]):
                if key in value and isinstance(value[key], dict):
                    return value[key]
        return value

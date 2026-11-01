"""Explicit browser origins allowed to access the local dashboard API."""

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DashboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:8090",
            "http://localhost:8090",
        ]
    )

    @field_validator("allowed_origins")
    @classmethod
    def explicit_origins(cls, values: list[str]) -> list[str]:
        for value in values:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or "*" in value
            ):
                raise ValueError("Dashboard CORS origins must be explicit HTTP or HTTPS origins.")
        return [value.rstrip("/") for value in values]

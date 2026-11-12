"""Resolve an explicitly supplied dish configuration without reading defaults."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel
from pydantic_ai.models import Model

from petri.agents.config import HarnessContent


class _HarnessConfig(BaseModel):
    harness: HarnessContent


def resolve_model(dish_config: Mapping[str, object]) -> Model | str:
    """Build the selected adapter, or preserve a Pydantic AI provider string.

    Legacy callers without a harness block continue to use resolve_provider.
    This function neither loads configuration nor caches model instances.
    """
    settings = _HarnessConfig.model_validate(dish_config).harness
    for key in ("model", "provider"):
        value = getattr(settings, key)
        if value is not None and not value.strip():
            raise ValueError(f"harness.{key} must not be blank.")
    if any(not tool.strip() for tool in settings.tools):
        raise ValueError("harness.tools must contain nonblank tool names.")
    model = settings.model if settings.model is not None else dish_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("harness.model or top-level model must be a nonempty string.")
    settings.validate_openai_chat_model(model)

    if settings.backend == "provider":
        if settings.provider is not None:
            raise ValueError("harness.provider is only used by pi; put the provider in harness.model.")
        if settings.tools:
            raise ValueError("harness.tools configures CLI tools; provider tools belong to the agent.")
        if settings.openai_chat_supports_multiple_system_messages is not None:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.profiles.openai import OpenAIModelProfile

            return OpenAIChatModel(model.split(":", 1)[1], provider="openai", profile=OpenAIModelProfile(
                openai_chat_supports_multiple_system_messages=settings.openai_chat_supports_multiple_system_messages,
            ))
        return model

    provider = settings.provider
    # Top-level v2 models use Pydantic AI's provider:model notation. An explicit
    # harness.model is already a backend-native identifier and stays untouched.
    if settings.model is None and ":" in model:
        inherited_provider, model = model.split(":", 1)
        if provider is not None and provider != inherited_provider:
            raise ValueError("harness.provider conflicts with the top-level model provider.")
        provider = inherited_provider
    if not model.strip():
        raise ValueError("harness.model must include a model identifier after the provider.")
    if settings.backend == "pi":
        from petri.harness.pi.model import PiModel

        return PiModel(model, provider=provider or "anthropic", tools=settings.tools)

    if provider not in (None, "anthropic"):
        raise ValueError("harness.provider must be anthropic for the claude-code backend.")
    from petri.harness.claude_code.model import ClaudeCodeModel

    return ClaudeCodeModel(model, allowed_tools=settings.tools)

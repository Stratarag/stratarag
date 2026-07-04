"""LLM providers and the `resolve_provider` factory."""
from __future__ import annotations

from typing import Any, Union

from ..errors import ConfigurationError
from .base import FunctionProvider, LLMProvider, LLMResponse
from .echo import EchoProvider

ModelLike = Union[str, LLMProvider, Any]


def resolve_provider(model: ModelLike) -> LLMProvider:
    """Turn a model spec into a provider.

    Accepted specs:
    - an LLMProvider instance (returned as-is)
    - a callable ``(messages, tools) -> str | LLMResponse``
    - ``"echo"`` — deterministic offline provider
    - ``"claude-*"`` or ``"anthropic:<model>"`` — Anthropic API
    """
    if isinstance(model, LLMProvider):
        return model
    if callable(model):
        return FunctionProvider(model)
    if isinstance(model, str):
        if model == "echo":
            return EchoProvider()
        if model.startswith("anthropic:"):
            from .anthropic import AnthropicProvider
            return AnthropicProvider(model.split(":", 1)[1])
        if model.startswith("claude"):
            from .anthropic import AnthropicProvider
            return AnthropicProvider(model)
    raise ConfigurationError(
        f"Unknown model spec: {model!r}. Pass an LLMProvider, a callable, "
        "'echo', 'claude-*', or 'anthropic:<model>'."
    )


__all__ = ["LLMProvider", "LLMResponse", "EchoProvider", "FunctionProvider", "resolve_provider"]

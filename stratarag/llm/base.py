"""Provider interface. Any LLM backend implements LLMProvider."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..types import Message, ToolCall


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract LLM provider. Implement `complete`; the async and streaming
    variants have sensible defaults so a minimal provider is one method."""

    name: str = "provider"

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await asyncio.to_thread(self.complete, messages, tools, **kwargs)

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Default streaming: complete, then yield word-by-word."""
        resp = self.complete(messages, tools, **kwargs)
        self._last_stream_response = resp
        words = resp.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    async def astream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        resp = await self.acomplete(messages, tools, **kwargs)
        self._last_stream_response = resp
        words = resp.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word


class FunctionProvider(LLMProvider):
    """Wraps a plain callable `(messages, tools) -> str | LLMResponse`."""

    name = "function"

    def __init__(self, fn):
        self._fn = fn

    def complete(self, messages, tools=None, **kwargs):
        out = self._fn(messages, tools)
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(text=str(out))

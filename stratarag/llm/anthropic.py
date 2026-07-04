"""Anthropic provider (optional dependency: `anthropic`)."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..errors import GenerationError, MissingDependencyError
from ..types import Message, ToolCall
from .base import LLMProvider, LLMResponse


def _client():
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise MissingDependencyError("anthropic", "anthropic", "The Anthropic provider") from e
    return anthropic


def _split(messages: List[Message]):
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }],
            })
        elif m.role == "assistant" and m.tool_calls:
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    return system, out


def _tool_schema(tools: Optional[List[Dict[str, Any]]]):
    if not tools:
        return None
    return [
        {"name": t["name"], "description": t.get("description", ""),
         "input_schema": t.get("parameters", {"type": "object", "properties": {}})}
        for t in tools
    ]


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        anthropic = _client()
        self._sync = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._async = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    def _parse(self, resp: Any) -> LLMResponse:
        text_parts: List[str] = []
        calls: List[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, args=dict(block.input), id=block.id))
        return LLMResponse(text="".join(text_parts), tool_calls=calls, raw=resp)

    def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
        system, msgs = _split(messages)
        try:
            resp = self._sync.messages.create(
                model=self.model, max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                system=system or "", messages=msgs,
                tools=_tool_schema(tools) or [], **kwargs)
        except Exception as e:  # pragma: no cover - network
            raise GenerationError(str(e)) from e
        return self._parse(resp)

    async def acomplete(self, messages, tools=None, **kwargs) -> LLMResponse:
        system, msgs = _split(messages)
        try:
            resp = await self._async.messages.create(
                model=self.model, max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                system=system or "", messages=msgs,
                tools=_tool_schema(tools) or [], **kwargs)
        except Exception as e:  # pragma: no cover - network
            raise GenerationError(str(e)) from e
        return self._parse(resp)

    def stream(self, messages, tools=None, **kwargs) -> Iterator[str]:  # pragma: no cover - network
        system, msgs = _split(messages)
        with self._sync.messages.stream(
            model=self.model, max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            system=system or "", messages=msgs, **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text
            self._last_stream_response = self._parse(stream.get_final_message())

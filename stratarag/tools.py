"""Tools: plain functions, decorated. Schemas are derived from type hints and
docstrings — no schema classes to learn.

    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        '''Return current weather for a city.'''
        ...
"""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .errors import ToolError

_JSON_TYPES = {str: "string", int: "integer", float: "number",
               bool: "boolean", list: "array", dict: "object"}


def _schema_from_signature(fn: Callable) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
        props[name] = {"type": _JSON_TYPES.get(ann, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            props[name]["default"] = param.default
    return {"type": "object", "properties": props, "required": required}


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable = field(repr=False, default=None)  # type: ignore

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)

    def run(self, args: Dict[str, Any]) -> str:
        try:
            result = self.fn(**args)
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
        except TypeError as e:
            raise ToolError(f"Bad arguments for tool '{self.name}': {e}") from e
        except Exception as e:
            raise ToolError(f"Tool '{self.name}' failed: {e}") from e
        return result if isinstance(result, str) else json.dumps(result, default=str)

    async def arun(self, args: Dict[str, Any]) -> str:
        try:
            result = self.fn(**args)
            if inspect.iscoroutine(result):
                result = await result
        except TypeError as e:
            raise ToolError(f"Bad arguments for tool '{self.name}': {e}") from e
        except Exception as e:
            raise ToolError(f"Tool '{self.name}' failed: {e}") from e
        return result if isinstance(result, str) else json.dumps(result, default=str)


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None,
         description: Optional[str] = None):
    """Decorator turning a function into a Tool. Usable bare or with args."""
    def wrap(f: Callable) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=description or inspect.getdoc(f) or f.__name__,
            parameters=_schema_from_signature(f),
            fn=f,
        )
    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    def __init__(self, tools: Optional[List[Any]] = None):
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Any) -> None:
        if not isinstance(t, Tool):
            if callable(t):
                t = tool(t)
            else:
                raise ToolError(f"Not a tool or callable: {t!r}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"Unknown tool '{name}'. Registered: {sorted(self._tools)}")
        return self._tools[name]

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

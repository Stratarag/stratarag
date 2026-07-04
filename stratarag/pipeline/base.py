"""Pipeline: an ordered list of stages passing one Context object along.
Every stage is a plain class with `run(ctx) -> ctx` — subclass anything."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..types import Message, MemoryRecord, ScoredChunk, TraceEvent


@dataclass
class Context:
    query: str
    user_id: str = "default"
    rewritten_query: Optional[str] = None
    candidates: List[ScoredChunk] = field(default_factory=list)
    memory: Any = None  # MemoryContext
    messages: List[Message] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    gated: bool = False
    trace: List[TraceEvent] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.query


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, ctx: Context) -> Context: ...

    async def arun(self, ctx: Context) -> Context:
        return self.run(ctx)

    def _trace(self, ctx: Context, started: float, **detail: Any) -> None:
        ctx.trace.append(TraceEvent(
            stage=self.name, detail=detail,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3)))


class Pipeline:
    def __init__(self, *stages: Stage):
        flat: List[Stage] = []
        for s in stages:
            if isinstance(s, (list, tuple)):
                flat.extend(s)
            else:
                flat.append(s)
        self.stages = flat

    def run(self, query: str, user_id: str = "default", **meta: Any) -> Context:
        ctx = Context(query=query, user_id=user_id, meta=meta)
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx

    async def arun(self, query: str, user_id: str = "default", **meta: Any) -> Context:
        ctx = Context(query=query, user_id=user_id, meta=meta)
        for stage in self.stages:
            ctx = await stage.arun(ctx)
        return ctx

    def then(self, stage: Stage) -> "Pipeline":
        return Pipeline(*self.stages, stage)

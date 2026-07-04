"""Agent: model + tools + memory + knowledge, with sync/async/streaming runs.

    agent = Agent(model="claude-sonnet-4-6", tools=[search], memory=Memory(),
                  knowledge=kb)
    result = agent.run("What changed in the refund policy?", user_id="u42")
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .errors import ToolError
from .knowledge import Knowledge
from .llm import LLMProvider, resolve_provider
from .memory import Memory
from .pipeline.stages import (build_grounded_messages, confidence_score,
                              grounding_score)
from .pipeline.base import Context
from .tools import Tool, ToolRegistry
from .types import AgentResult, Message, TraceEvent


class Agent:
    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        memory: Optional[Memory] = None,
        knowledge: Optional[Knowledge] = None,
        system: Optional[str] = None,
        retrieval_k: int = 4,
        retrieval: str = "hybrid",          # "hybrid" | "vector" | "keyword" | "graph"
        max_tool_rounds: int = 5,
        confidence_threshold: float = 0.0,  # 0 disables gating
        confidence_fallback: str = ("I'm not confident enough in the available "
                                    "sources to answer that reliably."),
    ):
        self.provider: LLMProvider = resolve_provider(model)
        self.tools = ToolRegistry(tools or [])
        self.memory = memory
        self.knowledge = knowledge
        self.system = system
        self.retrieval_k = retrieval_k
        self.retrieval = retrieval
        self.max_tool_rounds = max_tool_rounds
        self.confidence_threshold = confidence_threshold
        self.confidence_fallback = confidence_fallback

    # ---------------------------------------------------------------- context
    def _prepare(self, prompt: str, user_id: str) -> Context:
        ctx = Context(query=prompt, user_id=user_id)
        t0 = time.perf_counter()
        if self.knowledge is not None:
            if self.retrieval == "vector":
                ctx.candidates = self.knowledge.search(prompt, k=self.retrieval_k)
            elif self.retrieval == "keyword":
                ctx.candidates = self.knowledge.keyword_search(prompt, k=self.retrieval_k)
            elif self.retrieval == "graph":
                ctx.candidates = self.knowledge.graph_search(prompt, k=self.retrieval_k)
            else:
                ctx.candidates = self.knowledge.hybrid_search(prompt, k=self.retrieval_k)
            ctx.trace.append(TraceEvent(stage="retrieve",
                detail={"candidates": len(ctx.candidates)},
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))
        if self.memory is not None:
            t1 = time.perf_counter()
            ctx.memory = self.memory.read(prompt, user_id=user_id)
            ctx.trace.append(TraceEvent(stage="memory_read",
                detail={"facts": len(ctx.memory.facts)},
                elapsed_ms=round((time.perf_counter() - t1) * 1000, 3)))
        ctx.messages = build_grounded_messages(ctx, self.system)
        return ctx

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            return self.tools.get(name).run(args)
        except ToolError as e:
            return f"[tool error] {e}"

    async def _aexecute_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            return await self.tools.get(name).arun(args)
        except ToolError as e:
            return f"[tool error] {e}"

    def _finish(self, ctx: Context, answer: str, user_id: str) -> AgentResult:
        confidence = 1.0
        gated = False
        if ctx.candidates:
            confidence = confidence_score(
                ctx.query, answer, [sc.chunk.text for sc in ctx.candidates])
            if self.confidence_threshold and confidence < self.confidence_threshold:
                ctx.meta["ungated_answer"] = answer
                answer, gated = self.confidence_fallback, True
        if self.memory is not None:
            t0 = time.perf_counter()
            new_facts = self.memory.write_turn(ctx.query, answer, user_id=user_id,
                                               success=not gated)
            ctx.trace.append(TraceEvent(stage="memory_write",
                detail={"new_facts": new_facts},
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))
        return AgentResult(
            output=answer, confidence=confidence, gated=gated,
            sources=ctx.candidates,
            memory_used=ctx.memory.as_dict() if ctx.memory is not None else {},
            messages=ctx.messages, trace=ctx.trace)

    # -------------------------------------------------------------------- run
    def run(self, prompt: str, user_id: str = "default") -> AgentResult:
        ctx = self._prepare(prompt, user_id)
        messages = ctx.messages
        specs = self.tools.specs() or None
        for _ in range(self.max_tool_rounds + 1):
            t0 = time.perf_counter()
            resp = self.provider.complete(messages, tools=specs)
            ctx.trace.append(TraceEvent(stage="llm",
                detail={"tool_calls": [t.name for t in resp.tool_calls]},
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))
            if not resp.tool_calls:
                return self._finish(ctx, resp.text, user_id)
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls))
            for call in resp.tool_calls:
                result = self._execute_tool(call.name, call.args)
                ctx.trace.append(TraceEvent(stage=f"tool:{call.name}",
                                            detail={"args": call.args,
                                                    "result": result[:200]}))
                messages.append(Message(role="tool", content=result,
                                        name=call.name, tool_call_id=call.id))
        return self._finish(
            ctx, "I couldn't complete the task within the tool-call limit.", user_id)

    async def arun(self, prompt: str, user_id: str = "default") -> AgentResult:
        ctx = self._prepare(prompt, user_id)
        messages = ctx.messages
        specs = self.tools.specs() or None
        for _ in range(self.max_tool_rounds + 1):
            t0 = time.perf_counter()
            resp = await self.provider.acomplete(messages, tools=specs)
            ctx.trace.append(TraceEvent(stage="llm",
                detail={"tool_calls": [t.name for t in resp.tool_calls]},
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))
            if not resp.tool_calls:
                return self._finish(ctx, resp.text, user_id)
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls))
            results = await asyncio.gather(*[
                self._aexecute_tool(c.name, c.args) for c in resp.tool_calls])
            for call, result in zip(resp.tool_calls, results):
                ctx.trace.append(TraceEvent(stage=f"tool:{call.name}",
                                            detail={"args": call.args,
                                                    "result": result[:200]}))
                messages.append(Message(role="tool", content=result,
                                        name=call.name, tool_call_id=call.id))
        return self._finish(
            ctx, "I couldn't complete the task within the tool-call limit.", user_id)

    # ---------------------------------------------------------------- stream
    def stream(self, prompt: str, user_id: str = "default") -> Iterator[Dict[str, Any]]:
        """Yields events: {"type": "token", "text": ...} then
        {"type": "result", "result": AgentResult}. Tool rounds are resolved
        non-streamed; only the final answer streams."""
        ctx = self._prepare(prompt, user_id)
        messages = ctx.messages
        specs = self.tools.specs() or None
        rounds = 0
        while rounds < self.max_tool_rounds:
            resp = self.provider.complete(messages, tools=specs)
            if not resp.tool_calls:
                # replay this final answer as a stream
                break
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls))
            for call in resp.tool_calls:
                result = self._execute_tool(call.name, call.args)
                yield {"type": "tool", "name": call.name, "result": result}
                messages.append(Message(role="tool", content=result,
                                        name=call.name, tool_call_id=call.id))
            rounds += 1
        chunks: List[str] = []
        for token in self.provider.stream(messages, tools=None):
            chunks.append(token)
            yield {"type": "token", "text": token}
        answer = "".join(chunks)
        yield {"type": "result", "result": self._finish(ctx, answer, user_id)}

    async def astream(self, prompt: str, user_id: str = "default") -> AsyncIterator[Dict[str, Any]]:
        ctx = self._prepare(prompt, user_id)
        messages = ctx.messages
        specs = self.tools.specs() or None
        rounds = 0
        while rounds < self.max_tool_rounds:
            resp = await self.provider.acomplete(messages, tools=specs)
            if not resp.tool_calls:
                break
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls))
            for call in resp.tool_calls:
                result = await self._aexecute_tool(call.name, call.args)
                yield {"type": "tool", "name": call.name, "result": result}
                messages.append(Message(role="tool", content=result,
                                        name=call.name, tool_call_id=call.id))
            rounds += 1
        chunks: List[str] = []
        async for token in self.provider.astream(messages, tools=None):
            chunks.append(token)
            yield {"type": "token", "text": token}
        answer = "".join(chunks)
        yield {"type": "result", "result": self._finish(ctx, answer, user_id)}

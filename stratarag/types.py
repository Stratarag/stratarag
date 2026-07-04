"""Core datatypes shared across stratarag."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Document:
    text: str
    id: str = field(default_factory=lambda: new_id("doc"))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    doc_id: str = ""
    id: str = field(default_factory=lambda: new_id("chk"))
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [t.to_dict() for t in self.tool_calls]
        return d


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args}


@dataclass
class MemoryRecord:
    kind: str  # semantic | episodic | procedural | prospective
    content: str
    id: str = field(default_factory=lambda: new_id("mem"))
    user_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    score: float = 0.0


@dataclass
class TraceEvent:
    stage: str
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class AgentResult:
    output: str
    confidence: float = 1.0
    gated: bool = False
    sources: List[ScoredChunk] = field(default_factory=list)
    memory_used: Dict[str, List[MemoryRecord]] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    trace: List[TraceEvent] = field(default_factory=list)

    def __str__(self) -> str:
        return self.output

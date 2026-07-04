"""Memory: what the agent has learned — kept strictly separate from Knowledge.

    memory = Memory(semantic=True, episodic=True)          # in-process
    memory = Memory(backend="sqlite:./mem.db")             # persistent
    memory = Memory(extractor="llm", model="claude-...")   # LLM fact extraction

`read()` gathers everything relevant for a query; `write_turn()` learns from
a completed exchange. Agent calls both automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..embeddings import EmbedderLike, resolve_embedder
from ..stores import StoreLike, resolve_store
from ..types import MemoryRecord, Message
from .extractors import HeuristicExtractor, LLMExtractor, resolve_extractor
from .modules import (
    EpisodicMemory,
    ProceduralMemory,
    ProspectiveMemory,
    SemanticMemory,
    WorkingMemory,
)


@dataclass
class MemoryContext:
    facts: List[MemoryRecord] = field(default_factory=list)
    episodes: List[MemoryRecord] = field(default_factory=list)
    skills: List[MemoryRecord] = field(default_factory=list)
    intents: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[MemoryRecord]]:
        return {"semantic": self.facts, "episodic": self.episodes,
                "procedural": self.skills}

    def render(self) -> str:
        lines: List[str] = []
        if self.facts:
            lines.append("Known about this user:")
            lines += [f"- {r.content}" for r in self.facts]
        if self.episodes:
            lines.append("Relevant past experience:")
            lines += [f"- {r.content}" for r in self.episodes]
        if self.skills:
            lines.append("Relevant skills:")
            lines += [f"- {r.content}" for r in self.skills]
        if self.intents:
            lines.append("Pending intents now due:")
            lines += [f"- {i['intent']}" for i in self.intents]
        return "\n".join(lines)


class Memory:
    def __init__(
        self,
        semantic: bool = True,
        episodic: bool = False,
        procedural: bool = False,
        prospective: bool = False,
        working: bool = True,
        backend: StoreLike = "memory",
        embedder: EmbedderLike = "hashing",
        extractor: Any = "heuristic",
        model: Any = None,
        max_working_words: int = 2000,
    ):
        emb = resolve_embedder(embedder)
        store = resolve_store(backend, dim=emb.dim)
        self.semantic = SemanticMemory(store, emb) if semantic else None
        self.episodic = EpisodicMemory(store, emb) if episodic else None
        self.procedural = ProceduralMemory(store, emb) if procedural else None
        self.prospective = ProspectiveMemory() if prospective else None
        self.working = WorkingMemory(max_words=max_working_words) if working else None
        self.extractor = resolve_extractor(extractor, model=model)

    # --------------------------------------------------------------- reading
    def read(self, query: str, user_id: str = "default", k: int = 3) -> MemoryContext:
        ctx = MemoryContext()
        if self.semantic:
            ctx.facts = self.semantic.search(query, user_id=user_id, k=k)
        if self.episodic:
            ctx.episodes = self.episodic.search(query, user_id=user_id, k=k)
        if self.procedural:
            ctx.skills = self.procedural.lookup(query, user_id=user_id, k=k)
        if self.prospective:
            ctx.intents = self.prospective.due(query, user_id=user_id)
        if self.working:
            ctx.history = self.working.messages(user_id=user_id)
        return ctx

    # --------------------------------------------------------------- writing
    def write_turn(self, user_text: str, assistant_text: str,
                   user_id: str = "default", success: bool = True) -> List[str]:
        """Learn from one exchange: extract facts, log the episode, extend
        working memory. Returns the list of new facts stored."""
        new_facts: List[str] = []
        if self.working:
            self.working.append(Message(role="user", content=user_text), user_id)
            self.working.append(Message(role="assistant", content=assistant_text), user_id)
        if self.semantic:
            for fact in self.extractor.extract(user_text, assistant_text):
                self.semantic.add(fact, user_id=user_id)
                new_facts.append(fact)
        if self.episodic:
            outcome = assistant_text[:160]
            self.episodic.log(task=user_text[:120], outcome=outcome,
                              success=success, user_id=user_id)
        return new_facts

    def remember(self, fact: str, user_id: str = "default") -> None:
        """Explicitly store a fact."""
        if not self.semantic:
            raise ValueError("semantic memory is disabled")
        self.semantic.add(fact, user_id=user_id)


__all__ = ["Memory", "MemoryContext", "SemanticMemory", "EpisodicMemory",
           "ProceduralMemory", "ProspectiveMemory", "WorkingMemory",
           "HeuristicExtractor", "LLMExtractor"]

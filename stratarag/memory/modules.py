"""The five memory types, each a small module over a VectorStore."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..embeddings import Embedder
from ..stores import VectorStore
from ..types import MemoryRecord, Message, new_id


class _VectorMemory:
    """Shared implementation: records embedded + filtered by kind/user."""

    kind = "generic"

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def _payload(self, rec: MemoryRecord) -> Dict[str, Any]:
        return {"kind": rec.kind, "content": rec.content, "user_id": rec.user_id,
                "metadata": rec.metadata, "created_at": rec.created_at}

    def add(self, content: str, user_id: str = "default",
            metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        rec = MemoryRecord(kind=self.kind, content=content, user_id=user_id,
                           metadata=metadata or {})
        self.store.add([rec.id], [self.embedder.embed_one(content)], [self._payload(rec)])
        return rec

    def search(self, query: str, user_id: str = "default", k: int = 3) -> List[MemoryRecord]:
        hits = self.store.query(self.embedder.embed_one(query), k=k,
                                where={"kind": self.kind, "user_id": user_id})
        return [
            MemoryRecord(id=h.id, kind=self.kind, content=h.payload["content"],
                         user_id=user_id, metadata=h.payload.get("metadata", {}),
                         created_at=h.payload.get("created_at", 0.0), score=h.score)
            for h in hits
        ]

    def all(self, user_id: str = "default") -> List[MemoryRecord]:
        payloads = self.store.all_payloads(where={"kind": self.kind, "user_id": user_id})
        return [MemoryRecord(kind=self.kind, content=p["content"], user_id=user_id,
                             metadata=p.get("metadata", {}),
                             created_at=p.get("created_at", 0.0))
                for p in payloads]


class SemanticMemory(_VectorMemory):
    """Durable facts and preferences: 'knowledge that lasts'."""
    kind = "semantic"

    def add(self, content, user_id="default", metadata=None):
        # de-dupe: skip if an almost-identical fact already exists
        existing = self.search(content, user_id=user_id, k=1)
        if existing and existing[0].score > 0.95:
            return existing[0]
        return super().add(content, user_id, metadata)


class EpisodicMemory(_VectorMemory):
    """Past runs and their outcomes: 'experiences that teach'."""
    kind = "episodic"

    def log(self, task: str, outcome: str, success: bool = True,
            reflection: str = "", user_id: str = "default") -> MemoryRecord:
        content = f"Task: {task} | Outcome: {outcome}"
        if reflection:
            content += f" | Reflection: {reflection}"
        return self.add(content, user_id=user_id,
                        metadata={"success": success, "task": task,
                                  "reflection": reflection})


class ProceduralMemory(_VectorMemory):
    """Named, reusable skills and workflows: 'how to do things'."""
    kind = "procedural"

    def register(self, name: str, steps: List[str], user_id: str = "default") -> MemoryRecord:
        content = f"Skill '{name}': " + " -> ".join(steps)
        return self.add(content, user_id=user_id,
                        metadata={"name": name, "steps": steps})

    def lookup(self, task: str, user_id: str = "default", k: int = 1) -> List[MemoryRecord]:
        return self.search(task, user_id=user_id, k=k)


class ProspectiveMemory:
    """Future intentions: 'what you plan to do next'. Triggers are either a
    due timestamp or a keyword that appears in a later query."""

    kind = "prospective"

    def __init__(self):
        self._intents: List[Dict[str, Any]] = []

    def add(self, intent: str, due_at: Optional[float] = None,
            trigger: Optional[str] = None, user_id: str = "default") -> str:
        iid = new_id("intent")
        self._intents.append({"id": iid, "intent": intent, "due_at": due_at,
                              "trigger": (trigger or "").lower(), "user_id": user_id,
                              "done": False})
        return iid

    def due(self, query: str = "", user_id: str = "default",
            now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = now if now is not None else time.time()
        fired = []
        for it in self._intents:
            if it["done"] or it["user_id"] != user_id:
                continue
            time_hit = it["due_at"] is not None and now >= it["due_at"]
            word_hit = bool(it["trigger"]) and it["trigger"] in query.lower()
            if time_hit or word_hit:
                fired.append(it)
        return fired

    def complete(self, intent_id: str) -> None:
        for it in self._intents:
            if it["id"] == intent_id:
                it["done"] = True

    def pending(self, user_id: str = "default") -> List[Dict[str, Any]]:
        return [it for it in self._intents if not it["done"] and it["user_id"] == user_id]


class WorkingMemory:
    """The rolling conversation buffer, trimmed to a word budget. Optionally
    pass `summarizer(messages) -> str` to compress dropped turns instead of
    losing them."""

    def __init__(self, max_words: int = 2000,
                 summarizer: Optional[Callable[[List[Message]], str]] = None):
        self.max_words = max_words
        self.summarizer = summarizer
        self._turns: Dict[str, List[Message]] = {}
        self._summaries: Dict[str, str] = {}

    def append(self, message: Message, user_id: str = "default") -> None:
        self._turns.setdefault(user_id, []).append(message)
        self._trim(user_id)

    def messages(self, user_id: str = "default") -> List[Message]:
        out: List[Message] = []
        summary = self._summaries.get(user_id)
        if summary:
            out.append(Message(role="system", content=f"Conversation so far: {summary}"))
        out.extend(self._turns.get(user_id, []))
        return out

    def clear(self, user_id: str = "default") -> None:
        self._turns.pop(user_id, None)
        self._summaries.pop(user_id, None)

    def _trim(self, user_id: str) -> None:
        turns = self._turns[user_id]
        def total(ms): return sum(len(m.content.split()) for m in ms)
        dropped: List[Message] = []
        while len(turns) > 2 and total(turns) > self.max_words:
            dropped.append(turns.pop(0))
        if dropped and self.summarizer:
            prev = self._summaries.get(user_id, "")
            new = self.summarizer(dropped)
            self._summaries[user_id] = (prev + " " + new).strip()

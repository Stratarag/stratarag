"""Built-in pipeline stages — the production RAG architecture as legos:

    QueryRewrite -> HybridRetrieve -> Rerank -> ContextFilter
        -> MemoryRead -> Generate -> ConfidenceGate
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

from ..embeddings import tokenize
from ..knowledge import Knowledge
from ..llm import resolve_provider
from ..memory import Memory
from ..types import Message, ScoredChunk
from .base import Context, Stage
from .rerankers import resolve_reranker
from .rewrite import resolve_rewriter

_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
         "and", "or", "for", "it", "this", "that", "with", "as", "at", "by",
         "be", "from", "i", "you", "we", "they", "he", "she", "do", "does",
         "what", "who", "how", "when", "where", "why", "which", "can", "could",
         "should", "would", "will", "my", "our", "your", "me", "us", "much",
         "many", "get", "about", "hi", "hello", "please", "there", "if", "no",
         "tell", "know", "want", "need", "just"}


def _content_words(text: str) -> set:
    return {t for t in tokenize(text) if t not in _STOP}


def grounding_score(answer: str, context_texts: List[str]) -> float:
    """Fraction of the answer's content words that appear in the context —
    a cheap faithfulness proxy used by ConfidenceGate and the eval harness."""
    words = _content_words(answer)
    if not words:
        return 0.0
    ctx_words = set()
    for t in context_texts:
        ctx_words |= _content_words(t)
    return len(words & ctx_words) / len(words)


def confidence_score(query: str, answer: str,
                     context_texts: List[str]) -> float:
    """Gate score = geometric mean of faithfulness (is the answer grounded
    in the context?) and relevance (is the context actually about the
    query?). Faithfulness alone is gameable: a model that parrots retrieved
    text scores 1.0 even when the retrieval was irrelevant to the question."""
    faithfulness = grounding_score(answer, context_texts)
    relevance = grounding_score(query, context_texts)
    return (faithfulness * relevance) ** 0.5


class QueryRewrite(Stage):
    name = "query_rewrite"

    def __init__(self, rewriter: Any = "heuristic", model: Any = None):
        self.rewriter = resolve_rewriter(rewriter, model=model)

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        ctx.rewritten_query = self.rewriter.rewrite(ctx.query)
        self._trace(ctx, t0, rewritten=ctx.rewritten_query)
        return ctx


class Retrieve(Stage):
    name = "retrieve"

    def __init__(self, knowledge: Knowledge, k: int = 10, where=None):
        self.knowledge, self.k, self.where = knowledge, k, where

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        where = ctx.meta.get("where", self.where)
        ctx.candidates = self.knowledge.search(ctx.effective_query, k=self.k,
                                               where=where)
        self._trace(ctx, t0, candidates=len(ctx.candidates), where=where)
        return ctx


class HybridRetrieve(Stage):
    name = "hybrid_retrieve"

    def __init__(self, knowledge: Knowledge, k: int = 10, alpha: float = 0.5,
                 where=None):
        self.knowledge, self.k, self.alpha, self.where = knowledge, k, alpha, where

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        where = ctx.meta.get("where", self.where)
        ctx.candidates = self.knowledge.hybrid_search(
            ctx.effective_query, k=self.k, alpha=self.alpha, where=where)
        self._trace(ctx, t0, candidates=len(ctx.candidates), where=where)
        return ctx


class GraphRetrieve(Stage):
    """Cross-modal graph-augmented retrieval; requires Knowledge(graph=True)."""
    name = "graph_retrieve"

    def __init__(self, knowledge: Knowledge, k: int = 10, hops: int = 1,
                 alpha: float = 0.6):
        self.knowledge, self.k, self.hops, self.alpha = knowledge, k, hops, alpha

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        ctx.candidates = self.knowledge.graph_search(
            ctx.effective_query, k=self.k, hops=self.hops, alpha=self.alpha)
        self._trace(ctx, t0, candidates=len(ctx.candidates))
        return ctx


class Rerank(Stage):
    name = "rerank"

    def __init__(self, reranker: Any = "lexical", top_n: int = 5):
        self.reranker = resolve_reranker(reranker)
        self.top_n = top_n

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        ctx.candidates = self.reranker.rerank(
            ctx.effective_query, ctx.candidates)[: self.top_n]
        self._trace(ctx, t0, kept=len(ctx.candidates))
        return ctx


class ContextFilter(Stage):
    """Drop low-signal and duplicate chunks."""
    name = "context_filter"

    def __init__(self, min_score: float = 0.0, max_chunks: int = 5):
        self.min_score, self.max_chunks = min_score, max_chunks

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        seen, kept = set(), []
        for sc in ctx.candidates:
            key = sc.chunk.text[:120]
            if sc.score < self.min_score or key in seen:
                continue
            seen.add(key)
            kept.append(sc)
            if len(kept) >= self.max_chunks:
                break
        ctx.candidates = kept
        self._trace(ctx, t0, kept=len(kept))
        return ctx


class MemoryRead(Stage):
    name = "memory_read"

    def __init__(self, memory: Memory, k: int = 3):
        self.memory, self.k = memory, k

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        ctx.memory = self.memory.read(ctx.query, user_id=ctx.user_id, k=self.k)
        self._trace(ctx, t0,
                    facts=len(ctx.memory.facts), episodes=len(ctx.memory.episodes))
        return ctx


def build_grounded_messages(ctx: Context, system: Optional[str]) -> List[Message]:
    parts: List[str] = []
    if system:
        parts.append(system)
    if ctx.candidates:
        src = "\n".join(f"[source {i+1}] {sc.chunk.text}"
                        for i, sc in enumerate(ctx.candidates))
        parts.append(
            "Answer ONLY from the sources below. If they don't contain the "
            "answer, say you don't know.\n" + src)
    if ctx.memory is not None:
        rendered = ctx.memory.render()
        if rendered:
            parts.append(rendered)
    msgs: List[Message] = [Message(role="system", content="\n\n".join(parts))] \
        if parts else []
    history = list(ctx.memory.history) if ctx.memory is not None else []
    msgs.extend(m for m in history if m.role != "system")
    msgs.append(Message(role="user", content=ctx.query))
    return msgs


class Generate(Stage):
    name = "generate"

    def __init__(self, model: Any, system: Optional[str] = None, grounded: bool = True):
        self.provider = resolve_provider(model)
        self.system = system
        self.grounded = grounded

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        if self.grounded:
            ctx.messages = build_grounded_messages(ctx, self.system)
        else:
            ctx.messages = ([Message(role="system", content=self.system)]
                            if self.system else [])
            ctx.messages.append(Message(role="user", content=ctx.query))
        resp = self.provider.complete(ctx.messages)
        ctx.answer = resp.text
        self._trace(ctx, t0, chars=len(ctx.answer))
        return ctx

    async def arun(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        if self.grounded:
            ctx.messages = build_grounded_messages(ctx, self.system)
        else:
            ctx.messages = ([Message(role="system", content=self.system)]
                            if self.system else [])
            ctx.messages.append(Message(role="user", content=ctx.query))
        resp = await self.provider.acomplete(ctx.messages)
        ctx.answer = resp.text
        self._trace(ctx, t0, chars=len(ctx.answer))
        return ctx


class ConfidenceGate(Stage):
    """Score how grounded the answer is in the retrieved context; below the
    threshold, replace it with a safe fallback and mark the context gated."""
    name = "confidence_gate"

    def __init__(self, threshold: float = 0.5,
                 fallback: str = ("I'm not confident enough in the available "
                                  "sources to answer that reliably.")):
        self.threshold, self.fallback = threshold, fallback

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        ctx.confidence = confidence_score(
            ctx.effective_query, ctx.answer,
            [sc.chunk.text for sc in ctx.candidates])
        if ctx.confidence < self.threshold:
            ctx.meta["ungated_answer"] = ctx.answer
            ctx.answer = self.fallback
            ctx.gated = True
        self._trace(ctx, t0, confidence=round(ctx.confidence, 3), gated=ctx.gated)
        return ctx

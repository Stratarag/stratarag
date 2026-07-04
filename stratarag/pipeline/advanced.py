"""Advanced retrieval stages completing the ten classic RAG architectures:

CorrectiveRetrieve   (CRAG)  - score relevance, fallback search when weak
MultiHopRetrieve             - decompose into sub-questions, retrieve per hop
IterativeRetrieve    (IRAG)  - refine the query across bounded loops
Compress                     - keep only query-relevant sentences per chunk
SelfRAGGenerate      (Self-RAG) - draft, self-critique, re-retrieve, redo
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, List, Optional

from ..chunking import split_sentences
from ..knowledge import Knowledge
from ..llm import resolve_provider
from ..types import Message, ScoredChunk
from .base import Context, Stage
from .stages import Generate, _content_words, grounding_score

_SUBQ_SPLIT = re.compile(r"\s+(?:and|then|also|plus)\s+|[;,]\s+|\?\s+", re.I)

_DECOMPOSE_PROMPT = """Break this question into 2-4 independent sub-questions
that must each be answered to answer the whole. Return ONLY a JSON array of
strings. If it is already a single simple question, return a one-item array.

Question: {query}"""

_CRITIQUE_PROMPT = """You are grading a draft answer against its sources.
Reply with ONLY one word: PASS if the draft is supported by the sources and
answers the question, or FAIL otherwise.

Question: {query}
Sources:
{sources}
Draft: {draft}"""


class CorrectiveRetrieve(Stage):
    """CRAG: retrieve, score relevance of what came back, and when it is
    weak, run a fallback retrieval (keyword search + a de-filled query) and
    keep the better set. Deterministic, no extra LLM calls."""

    name = "corrective_retrieve"

    def __init__(self, knowledge: Knowledge, k: int = 10,
                 min_relevance: float = 0.34, where=None):
        self.knowledge, self.k = knowledge, k
        self.min_relevance = min_relevance
        self.where = where

    def _relevance(self, query: str, cands: List[ScoredChunk]) -> float:
        if not cands:
            return 0.0
        return grounding_score(query, [sc.chunk.text for sc in cands])

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        where = ctx.meta.get("where", self.where)
        primary = self.knowledge.hybrid_search(ctx.effective_query, k=self.k,
                                               where=where)
        score = self._relevance(ctx.effective_query, primary)
        corrected = False
        if score < self.min_relevance:
            # fallback 1: pure keyword; fallback 2: content-words-only query
            fallback = self.knowledge.keyword_search(ctx.effective_query,
                                                     k=self.k, where=where)
            fb_score = self._relevance(ctx.effective_query, fallback)
            stripped = " ".join(sorted(_content_words(ctx.effective_query)))
            if stripped:
                fb2 = self.knowledge.hybrid_search(stripped, k=self.k, where=where)
                if self._relevance(ctx.effective_query, fb2) > fb_score:
                    fallback, fb_score = fb2, self._relevance(
                        ctx.effective_query, fb2)
            if fb_score > score:
                primary, score, corrected = fallback, fb_score, True
        ctx.candidates = primary
        ctx.meta["retrieval_relevance"] = round(score, 3)
        self._trace(ctx, t0, relevance=round(score, 3), corrected=corrected)
        return ctx


class MultiHopRetrieve(Stage):
    """Decompose the query into sub-questions (heuristic split on
    conjunctions, or an LLM decomposer), retrieve for each hop, and merge
    the candidate pools with max-score de-duplication."""

    name = "multi_hop_retrieve"

    def __init__(self, knowledge: Knowledge, k_per_hop: int = 4,
                 decomposer: Any = "heuristic", model: Any = None,
                 max_hops: int = 4, where=None):
        self.knowledge, self.k_per_hop, self.max_hops = knowledge, k_per_hop, max_hops
        self.where = where
        self._provider = None
        if decomposer == "llm":
            if model is None:
                raise ValueError("decomposer='llm' requires a model")
            self._provider = resolve_provider(model)

    def _decompose(self, query: str) -> List[str]:
        if self._provider is not None:
            try:
                resp = self._provider.complete([Message(
                    role="user", content=_DECOMPOSE_PROMPT.format(query=query))])
                raw = resp.text
                subs = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
                subs = [str(s).strip() for s in subs if str(s).strip()]
                if subs:
                    return subs[: self.max_hops]
            except Exception:
                pass
        parts = [p.strip(" ?.") for p in _SUBQ_SPLIT.split(query) if p.strip(" ?.")]
        return (parts or [query])[: self.max_hops]

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        where = ctx.meta.get("where", self.where)
        hops = self._decompose(ctx.effective_query)
        best: dict = {}
        for hop in hops:
            for sc in self.knowledge.hybrid_search(hop, k=self.k_per_hop,
                                                   where=where):
                prev = best.get(sc.chunk.id)
                if prev is None or sc.score > prev.score:
                    best[sc.chunk.id] = sc
        ctx.candidates = sorted(best.values(), key=lambda s: s.score, reverse=True)
        ctx.meta["hops"] = hops
        self._trace(ctx, t0, hops=hops, candidates=len(ctx.candidates))
        return ctx


class IterativeRetrieve(Stage):
    """IRAG: retrieve, then refine the query with salient terms from the top
    results and retrieve again, for up to `loops` rounds or until the
    candidate set stops improving."""

    name = "iterative_retrieve"

    def __init__(self, knowledge: Knowledge, k: int = 8, loops: int = 2,
                 where=None):
        self.knowledge, self.k, self.loops, self.where = knowledge, k, loops, where

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        where = ctx.meta.get("where", self.where)
        query = ctx.effective_query
        cands = self.knowledge.hybrid_search(query, k=self.k, where=where)
        history = [query]
        for _ in range(max(0, self.loops - 1)):
            if not cands:
                break
            top_words = _content_words(cands[0].chunk.text)
            expansion = " ".join(sorted(top_words)[:6])
            refined = f"{ctx.effective_query} {expansion}".strip()
            if refined in history:
                break
            history.append(refined)
            new = self.knowledge.hybrid_search(refined, k=self.k, where=where)
            merged = {sc.chunk.id: sc for sc in cands}
            for sc in new:
                if sc.chunk.id not in merged or sc.score > merged[sc.chunk.id].score:
                    merged[sc.chunk.id] = sc
            improved = len(merged) > len(cands)
            cands = sorted(merged.values(), key=lambda s: s.score, reverse=True)[: self.k]
            if not improved:
                break
        ctx.candidates = cands
        ctx.meta["query_history"] = history
        self._trace(ctx, t0, rounds=len(history), candidates=len(cands))
        return ctx


class Compress(Stage):
    """Contextual compression: within each retrieved chunk, keep only the
    sentences that share content words with the query (plus one neighbor for
    coherence). Cuts token cost without dropping whole chunks."""

    name = "compress"

    def __init__(self, min_sentence_overlap: int = 1, max_sentences: int = 4):
        self.min_overlap = min_sentence_overlap
        self.max_sentences = max_sentences

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        q_words = _content_words(ctx.effective_query)
        before = sum(len(sc.chunk.text) for sc in ctx.candidates)
        compressed: List[ScoredChunk] = []
        for sc in ctx.candidates:
            sentences = split_sentences(sc.chunk.text)
            if len(sentences) <= 1:
                compressed.append(sc)
                continue
            keep_idx = set()
            for i, s in enumerate(sentences):
                if len(q_words & _content_words(s)) >= self.min_overlap:
                    keep_idx.add(i)
                    if i + 1 < len(sentences):
                        keep_idx.add(i + 1)   # neighbor for coherence
            kept = [sentences[i] for i in sorted(keep_idx)][: self.max_sentences]
            if kept:
                new_chunk = type(sc.chunk)(
                    text=" ".join(kept), id=sc.chunk.id, doc_id=sc.chunk.doc_id,
                    parent_id=sc.chunk.parent_id,
                    metadata={**(sc.chunk.metadata or {}), "compressed": True})
                compressed.append(ScoredChunk(chunk=new_chunk, score=sc.score))
        ctx.candidates = compressed or ctx.candidates
        after = sum(len(sc.chunk.text) for sc in ctx.candidates)
        ratio = round(after / before, 3) if before else 1.0
        ctx.meta["compression_ratio"] = ratio
        self._trace(ctx, t0, ratio=ratio, kept=len(ctx.candidates))
        return ctx


class SelfRAGGenerate(Stage):
    """Self-RAG: generate a draft, self-critique it (LLM verdict when
    available, grounding heuristic otherwise), and on failure re-retrieve
    with a refined query and regenerate — bounded by `max_rounds`."""

    name = "self_rag_generate"

    def __init__(self, model: Any, knowledge: Optional[Knowledge] = None,
                 system: Optional[str] = None, max_rounds: int = 2,
                 pass_threshold: float = 0.45, llm_critic: bool = True):
        self.provider = resolve_provider(model)
        self.knowledge = knowledge
        self.max_rounds = max(1, max_rounds)
        self.pass_threshold = pass_threshold
        self.llm_critic = llm_critic
        self._generate = Generate(model, system=system, grounded=True)

    def _critique(self, ctx: Context, draft: str) -> bool:
        texts = [sc.chunk.text for sc in ctx.candidates]
        heuristic_ok = (grounding_score(draft, texts) *
                        grounding_score(ctx.effective_query, texts)) ** 0.5 \
            >= self.pass_threshold if texts else False
        if not self.llm_critic:
            return heuristic_ok
        try:
            sources = "\n".join(f"- {t[:200]}" for t in texts[:5])
            resp = self.provider.complete([Message(
                role="user",
                content=_CRITIQUE_PROMPT.format(query=ctx.query,
                                                sources=sources, draft=draft))])
            verdict = resp.text.strip().upper()
            if "PASS" in verdict[:12]:
                return True
            if "FAIL" in verdict[:12]:
                return False
        except Exception:
            pass
        return heuristic_ok

    def run(self, ctx: Context) -> Context:
        t0 = time.perf_counter()
        rounds = 0
        for attempt in range(self.max_rounds):
            rounds += 1
            ctx = self._generate.run(ctx)
            if self._critique(ctx, ctx.answer):
                break
            if self.knowledge is not None and attempt + 1 < self.max_rounds:
                refined = " ".join(sorted(_content_words(ctx.query))) or ctx.query
                ctx.candidates = self.knowledge.hybrid_search(refined, k=8)
                ctx.rewritten_query = refined
        ctx.meta["self_rag_rounds"] = rounds
        self._trace(ctx, t0, rounds=rounds)
        return ctx

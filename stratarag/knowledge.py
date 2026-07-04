"""Knowledge: the source-of-truth layer. Chunking + embedding + vector store
behind one interface, with vector, keyword (BM25) and hybrid search.

Deliberately separate from Memory: Knowledge is what the *organization* knows;
Memory is what the agent has learned about users and past sessions. They never
share a store, so user context can't pollute the source of truth.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Union

from .chunking import ChunkerLike, ParentChildChunker, resolve_chunker
from .embeddings import EmbedderLike, resolve_embedder, tokenize
from .graph import EntityGraph
from .stores import StoreLike, resolve_store
from .types import Chunk, Document, ScoredChunk

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".html"}


class Knowledge:
    def __init__(
        self,
        store: StoreLike = "memory",
        embedder: EmbedderLike = "hashing",
        chunking: ChunkerLike = "recursive",
        graph: Any = False,          # False | True ("heuristic") | "llm"
        graph_model: Any = None,     # model for graph="llm"
        **chunker_kwargs,
    ):
        self.embedder = resolve_embedder(embedder)
        self.store = resolve_store(store, dim=self.embedder.dim)
        self.chunker = resolve_chunker(chunking, **chunker_kwargs)
        self._parents: Dict[str, Chunk] = {}
        self._chunks: Dict[str, Chunk] = {}   # keyword index lives client-side
        self._df: Counter = Counter()
        self._hashes: set = set()             # incremental ingest de-dup
        self.graph: Optional[EntityGraph] = None
        if graph:
            self.graph = EntityGraph(
                extractor="llm" if graph == "llm" else "heuristic",
                model=graph_model)

    # ------------------------------------------------------------- ingestion
    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs) -> "Knowledge":
        kb = cls(**kwargs)
        kb.add([Document(text=t) for t in texts])
        return kb

    @classmethod
    def from_docs(cls, path: str, **kwargs) -> "Knowledge":
        """Load every text-like file under `path` (file or directory)."""
        docs: List[Document] = []
        if os.path.isfile(path):
            paths = [path]
        else:
            paths = [
                os.path.join(root, f)
                for root, _, files in os.walk(path)
                for f in files
                if os.path.splitext(f)[1].lower() in _TEXT_EXTS
            ]
        for p in sorted(paths):
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                docs.append(Document(text=fh.read(), metadata={"source": p}))
        kb = cls(**kwargs)
        kb.add(docs)
        return kb

    def add(self, docs: Union[Document, str, List[Union[Document, str]]]) -> int:
        if isinstance(docs, (Document, str)):
            docs = [docs]
        documents = [d if isinstance(d, Document) else Document(text=d) for d in docs]
        chunks = self.chunker.chunk_all(documents)
        if isinstance(self.chunker, ParentChildChunker):
            for parent in self.chunker.parents:
                self._parents[parent.id] = parent
            self.chunker.parents = []
        # incremental ingest: skip chunks whose exact content is already indexed
        fresh = []
        for c in chunks:
            h = hashlib.sha1(c.text.strip().encode("utf-8")).hexdigest()
            if h in self._hashes:
                continue
            self._hashes.add(h)
            fresh.append(c)
        chunks = fresh
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        payloads = []
        for c in chunks:
            self._chunks[c.id] = c
            self._df.update(set(tokenize(c.text)))
            payloads.append({
                "text": c.text, "doc_id": c.doc_id,
                "parent_id": c.parent_id, "metadata": c.metadata,
            })
            if self.graph is not None:
                self.graph.index(c)
        self.store.add([c.id for c in chunks], vectors, payloads)
        return len(chunks)

    # ------------------------------------------------------------- retrieval
    @staticmethod
    def _meta_match(chunk_meta: Dict, where: Optional[Dict]) -> bool:
        """Equality metadata filter; a list value means 'any of'."""
        if not where:
            return True
        for key, want in where.items():
            got = (chunk_meta or {}).get(key)
            if isinstance(want, (list, tuple, set)):
                if got not in want:
                    return False
            elif got != want:
                return False
        return True

    def search(self, query: str, k: int = 5,
               where: Optional[Dict] = None) -> List[ScoredChunk]:
        """Dense (vector) search, with parent resolution for parent_child and
        optional metadata filtering, e.g. where={"source": "handbook.md"}."""
        overfetch = k * 4 if where else k
        hits = self.store.query(self.embedder.embed_one(query), k=overfetch)
        if where:
            hits = [h for h in hits
                    if self._meta_match((h.payload or {}).get("metadata", {}),
                                        where)][:k]
        return self._to_scored(
            [(h.id, h.score, h.payload) for h in hits]
        )

    def keyword_search(self, query: str, k: int = 5,
                       where: Optional[Dict] = None) -> List[ScoredChunk]:
        """BM25 scoring over the ingested chunks (optionally filtered)."""
        q_tokens = tokenize(query)
        n = max(1, len(self._chunks))
        avgdl = (sum(len(tokenize(c.text)) for c in self._chunks.values()) / n) or 1.0
        k1, b = 1.5, 0.75
        scored = []
        for c in self._chunks.values():
            if where and not self._meta_match(c.metadata, where):
                continue
            tokens = tokenize(c.text)
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
            if score > 0:
                scored.append((c.id, score, None))
        scored.sort(key=lambda x: x[1], reverse=True)
        return self._to_scored(scored[:k])

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5,
                      where: Optional[Dict] = None) -> List[ScoredChunk]:
        """Weighted reciprocal-rank fusion of dense + keyword results.
        alpha=1 is pure vector, alpha=0 is pure keyword."""
        dense = self.search(query, k=k * 2, where=where)
        sparse = self.keyword_search(query, k=k * 2, where=where)
        rrf: Dict[str, float] = {}
        lookup: Dict[str, ScoredChunk] = {}
        for rank, sc in enumerate(dense):
            rrf[sc.chunk.id] = rrf.get(sc.chunk.id, 0.0) + alpha / (60 + rank)
            lookup[sc.chunk.id] = sc
        for rank, sc in enumerate(sparse):
            rrf[sc.chunk.id] = rrf.get(sc.chunk.id, 0.0) + (1 - alpha) / (60 + rank)
            lookup.setdefault(sc.chunk.id, sc)
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [ScoredChunk(chunk=lookup[cid].chunk, score=score) for cid, score in ranked]

    def graph_search(self, query: str, k: int = 5, hops: int = 1,
                     alpha: float = 0.6) -> List[ScoredChunk]:
        """Cross-modal hybrid retrieval (RAG-Anything style): semantic seeds
        from hybrid search, expanded through the entity graph so evidence
        connected by shared entities is pulled in even without lexical
        overlap. alpha weights semantic vs graph score. Requires
        Knowledge(graph=True)."""
        if self.graph is None:
            raise ValueError("graph_search requires Knowledge(graph=True)")
        seeds = self.hybrid_search(query, k=k)
        seed_scores = {sc.chunk.id: sc.score for sc in seeds}
        graph_scores = self.graph.expand(list(seed_scores), query=query, hops=hops)
        combined: Dict[str, float] = {}
        max_seed = max(seed_scores.values(), default=1.0) or 1.0
        for cid, s in seed_scores.items():
            combined[cid] = alpha * (s / max_seed)
        for cid, g in graph_scores.items():
            combined[cid] = combined.get(cid, 0.0) + (1 - alpha) * g
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        out: List[ScoredChunk] = []
        for cid, score in ranked:
            chunk = self._chunks.get(cid)
            if chunk is not None:
                out.append(ScoredChunk(chunk=chunk, score=score))
        return out

    def by_modality(self, results: List[ScoredChunk],
                    modality: str) -> List[ScoredChunk]:
        """Filter retrieval results to one modality (text/table/code/
        equation/image) — pairs with chunking='modality'."""
        return [sc for sc in results
                if (sc.chunk.metadata or {}).get("modality") == modality]

    # -------------------------------------------------------------- internal
    def _to_scored(self, rows) -> List[ScoredChunk]:
        out: List[ScoredChunk] = []
        seen_parents = set()
        for cid, score, payload in rows:
            chunk = self._chunks.get(cid)
            if chunk is None and payload:
                chunk = Chunk(text=payload.get("text", ""), id=cid,
                              doc_id=payload.get("doc_id", ""),
                              parent_id=payload.get("parent_id"),
                              metadata=payload.get("metadata", {}) or {})
            if chunk is None:
                continue
            if chunk.parent_id and chunk.parent_id in self._parents:
                if chunk.parent_id in seen_parents:
                    continue
                seen_parents.add(chunk.parent_id)
                chunk = self._parents[chunk.parent_id]
            out.append(ScoredChunk(chunk=chunk, score=float(score)))
        return out

    def __len__(self) -> int:
        return self.store.count()

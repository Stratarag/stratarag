"""Rerankers: reorder retrieved candidates by relevance to the query."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, List

from ..embeddings import tokenize
from ..errors import MissingDependencyError
from ..types import ScoredChunk


class LexicalOverlapReranker:
    """Dependency-free reranker: cosine over token counts (query vs chunk),
    blended with the retriever's own score."""

    def __init__(self, blend: float = 0.5):
        self.blend = blend

    def rerank(self, query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        q = Counter(tokenize(query))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        rescored: List[ScoredChunk] = []
        for sc in candidates:
            c = Counter(tokenize(sc.chunk.text))
            cn = math.sqrt(sum(v * v for v in c.values())) or 1.0
            overlap = sum(q[t] * c[t] for t in q) / (qn * cn)
            score = self.blend * overlap + (1 - self.blend) * sc.score
            rescored.append(ScoredChunk(chunk=sc.chunk, score=score))
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder adapter (optional dependency).
    e.g. CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L-6-v2')"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            raise MissingDependencyError(
                "sentence-transformers", "st", "CrossEncoderReranker") from e
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:  # pragma: no cover - heavy dep
        pairs = [(query, sc.chunk.text) for sc in candidates]
        scores = self._model.predict(pairs)
        rescored = [ScoredChunk(chunk=sc.chunk, score=float(s))
                    for sc, s in zip(candidates, scores)]
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored


def resolve_reranker(spec: Any):
    if spec is None or spec == "lexical":
        return LexicalOverlapReranker()
    if isinstance(spec, str) and spec.startswith("cross-encoder"):
        name = spec.split(":", 1)[1] if ":" in spec else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(name)
    if hasattr(spec, "rerank"):
        return spec
    raise ValueError(f"Unknown reranker spec: {spec!r}")

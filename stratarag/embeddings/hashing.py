"""HashingEmbedder: deterministic, dependency-free embeddings.

Feature-hashed unigram + bigram counts, L2-normalized. Not a semantic model —
it's a lexical embedding — but it is fast, offline, and good enough for local
development, tests, and small corpora. Swap in a real model for production:
``embedder="sentence-transformers:all-MiniLM-L6-v2"``.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from .base import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Very light suffix stripping so 'refunds'/'refund' and
    'shipping'/'ship' land in the same slot. Not a real stemmer; just enough
    for lexical matching."""
    for suffix in ("ing", "edly", "es", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def _slot(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = tokenize(text)
            for tok in tokens:
                vec[self._slot(tok)] += 1.0
            for a, b in zip(tokens, tokens[1:]):
                vec[self._slot(a + "_" + b)] += 0.5
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

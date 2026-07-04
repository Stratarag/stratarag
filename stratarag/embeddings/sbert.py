"""SentenceTransformers embedder (optional dependency)."""
from __future__ import annotations

from typing import List

from ..errors import MissingDependencyError
from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise MissingDependencyError(
                "sentence-transformers", "st", "SentenceTransformerEmbedder"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - heavy dep
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]

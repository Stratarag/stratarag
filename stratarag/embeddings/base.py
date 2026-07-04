"""Embedder interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    dim: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]: ...

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

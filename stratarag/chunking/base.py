"""Chunker interface."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List

from ..types import Chunk, Document

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]: ...

    def chunk_all(self, docs: List[Document]) -> List[Chunk]:
        out: List[Chunk] = []
        for d in docs:
            out.extend(self.chunk(d))
        return out

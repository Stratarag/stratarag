"""Built-in chunking strategies.

fixed        - fixed-size word windows with overlap
recursive    - split on paragraphs, then sentences, then words, until it fits
markdown     - structure-aware: split on markdown headers, keep section path
semantic     - embedding-based topic-shift splitting
parent_child - small child chunks for precise search, linked to large parents
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..embeddings import Embedder, HashingEmbedder
from ..types import Chunk, Document
from .base import Chunker, split_sentences, word_count

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class FixedSizeChunker(Chunker):
    def __init__(self, size: int = 200, overlap: float = 0.1):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.size = max(1, size)
        self.overlap = overlap

    def chunk(self, doc: Document) -> List[Chunk]:
        words = doc.text.split()
        if not words:
            return []
        step = max(1, int(self.size * (1 - self.overlap)))
        chunks: List[Chunk] = []
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + self.size])
            chunks.append(Chunk(text=piece, doc_id=doc.id, metadata=dict(doc.metadata)))
            if start + self.size >= len(words):
                break
        return chunks


class RecursiveChunker(Chunker):
    def __init__(self, max_words: int = 200):
        self.max_words = max(1, max_words)

    def chunk(self, doc: Document) -> List[Chunk]:
        pieces = self._split(doc.text)
        return [
            Chunk(text=p, doc_id=doc.id, metadata=dict(doc.metadata))
            for p in pieces
            if p.strip()
        ]

    def _split(self, text: str) -> List[str]:
        if word_count(text) <= self.max_words:
            return [text.strip()]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return self._merge([s for p in paragraphs for s in self._split(p)])
        sentences = split_sentences(text)
        if len(sentences) > 1:
            return self._merge(sentences)
        words = text.split()
        return [
            " ".join(words[i : i + self.max_words])
            for i in range(0, len(words), self.max_words)
        ]

    def _merge(self, pieces: List[str]) -> List[str]:
        merged: List[str] = []
        buf = ""
        for p in pieces:
            candidate = (buf + " " + p).strip() if buf else p
            if word_count(candidate) <= self.max_words:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        return merged


class MarkdownChunker(Chunker):
    """Structure-aware chunking: one chunk per header section, with the header
    path stored in metadata['section']."""

    def __init__(self, max_words: int = 400):
        self.max_words = max_words
        self._fallback = RecursiveChunker(max_words=max_words)

    def chunk(self, doc: Document) -> List[Chunk]:
        matches = list(_HEADER_RE.finditer(doc.text))
        if not matches:
            return self._fallback.chunk(doc)
        chunks: List[Chunk] = []
        path: List[str] = []
        preamble = doc.text[: matches[0].start()].strip()
        if preamble:
            chunks.append(Chunk(text=preamble, doc_id=doc.id, metadata=dict(doc.metadata)))
        for i, m in enumerate(matches):
            level, title = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [title]
            end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
            body = doc.text[m.end() : end].strip()
            section = " > ".join(path)
            text = f"{title}\n{body}" if body else title
            for piece in self._fallback._split(text):
                md = dict(doc.metadata)
                md["section"] = section
                chunks.append(Chunk(text=piece, doc_id=doc.id, metadata=md))
        return chunks


class SemanticChunker(Chunker):
    """Split where the topic shifts: cosine similarity between adjacent
    sentences drops below `threshold`."""

    def __init__(self, embedder: Optional[Embedder] = None, threshold: float = 0.25,
                 max_words: int = 300):
        self.embedder = embedder or HashingEmbedder(dim=256)
        self.threshold = threshold
        self.max_words = max_words

    def chunk(self, doc: Document) -> List[Chunk]:
        sentences = split_sentences(doc.text)
        if len(sentences) <= 1:
            return [Chunk(text=doc.text.strip(), doc_id=doc.id,
                          metadata=dict(doc.metadata))] if doc.text.strip() else []
        vecs = self.embedder.embed(sentences)
        groups: List[List[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = sum(a * b for a, b in zip(vecs[i - 1], vecs[i]))
            current = " ".join(groups[-1])
            if sim < self.threshold or word_count(current) >= self.max_words:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        return [
            Chunk(text=" ".join(g), doc_id=doc.id, metadata=dict(doc.metadata))
            for g in groups
        ]


class ParentChildChunker(Chunker):
    """Index small child chunks for precision; each child links to a large
    parent chunk via `parent_id`. `Knowledge` swaps children for parents at
    retrieval time, so the LLM sees rich context."""

    def __init__(self, parent_words: int = 800, child_words: int = 100):
        if child_words >= parent_words:
            raise ValueError("child_words must be smaller than parent_words")
        self._parents = RecursiveChunker(max_words=parent_words)
        self._children = RecursiveChunker(max_words=child_words)
        self.parents: List[Chunk] = []  # populated on each chunk() call

    def chunk(self, doc: Document) -> List[Chunk]:
        children: List[Chunk] = []
        for parent in self._parents.chunk(doc):
            parent.metadata["role"] = "parent"
            self.parents.append(parent)
            for child in self._children.chunk(
                Document(text=parent.text, id=doc.id, metadata=dict(doc.metadata))
            ):
                child.parent_id = parent.id
                child.metadata["role"] = "child"
                children.append(child)
        return children

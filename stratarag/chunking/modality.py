"""ModalityChunker: RAG-Anything-style modality-aware parsing.

Splits a document into *typed* blocks — text, table, code, equation, image —
so each modality gets specialized handling instead of being flattened:

- tables    : kept intact (never split mid-table) + linearized row-by-row
              rendering stored for better embedding recall
- equations : LaTeX preserved verbatim ($$...$$, \\[...\\], inline $...$ blocks)
- code      : fenced blocks kept whole, language tagged
- images    : ![alt](src) — alt text embedded; optional `captioner(src, alt)`
              hook (e.g. a VLM call) enriches the searchable text
- text      : falls through to a nested text chunker (recursive by default)

Every chunk carries metadata["modality"], enabling modality-aware retrieval
filters downstream.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from ..types import Chunk, Document
from .base import Chunker
from .strategies import RecursiveChunker

_FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```\s*$", re.M | re.S)
_BLOCK_EQ_RE = re.compile(r"(\$\$.+?\$\$|\\\[.+?\\\])", re.S)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_TABLE_RE = re.compile(
    r"((?:^\|[^\n]*\|\s*\n)(?:^\|[\s:|-]+\|\s*\n)(?:^\|[^\n]*\|\s*\n?)+)", re.M)


def linearize_table(md_table: str) -> str:
    """Render a markdown table as 'Header: value' sentences — far better
    embedding recall than raw pipe syntax."""
    rows = [r.strip() for r in md_table.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        return md_table
    def cells(row: str) -> List[str]:
        return [c.strip() for c in row.strip("|").split("|")]
    headers = cells(rows[0])
    out: List[str] = []
    for row in rows[2:]:  # skip separator row
        vals = cells(row)
        pairs = [f"{h}: {v}" for h, v in zip(headers, vals) if v]
        if pairs:
            out.append("; ".join(pairs))
    return ". ".join(out)


class ModalityChunker(Chunker):
    def __init__(self, text_chunker: Optional[Chunker] = None,
                 captioner: Optional[Callable[[str, str], str]] = None,
                 max_words: int = 200):
        self.text_chunker = text_chunker or RecursiveChunker(max_words=max_words)
        self.captioner = captioner

    def chunk(self, doc: Document) -> List[Chunk]:
        blocks = self._parse(doc.text)
        chunks: List[Chunk] = []
        for modality, raw, extra in blocks:
            md = dict(doc.metadata)
            md["modality"] = modality
            md.update(extra)
            if modality == "text":
                for piece in self.text_chunker.chunk(
                        Document(text=raw, id=doc.id, metadata=md)):
                    piece.metadata["modality"] = "text"
                    chunks.append(piece)
            elif modality == "table":
                text = linearize_table(raw)
                md["raw"] = raw.strip()
                chunks.append(Chunk(text=text, doc_id=doc.id, metadata=md))
            elif modality == "image":
                alt, src = extra.get("alt", ""), extra.get("src", "")
                caption = ""
                if self.captioner:
                    try:
                        caption = self.captioner(src, alt) or ""
                    except Exception:
                        caption = ""
                text = " ".join(p for p in [f"Image: {alt}" if alt else "Image",
                                            caption] if p)
                chunks.append(Chunk(text=text, doc_id=doc.id, metadata=md))
            else:  # code / equation kept verbatim
                chunks.append(Chunk(text=raw.strip(), doc_id=doc.id, metadata=md))
        return [c for c in chunks if c.text.strip()]

    # -------------------------------------------------------------- parsing
    def _parse(self, text: str):
        """Return ordered (modality, raw, extra) blocks."""
        blocks = []
        # 1) lift fenced code out first so nothing inside is re-parsed
        cursor = 0
        segments = []
        for m in _FENCE_RE.finditer(text):
            segments.append(("text", text[cursor:m.start()]))
            segments.append(("code", m.group(2), {"language": m.group(1) or "plain"}))
            cursor = m.end()
        segments.append(("text", text[cursor:]))
        # 2) within remaining text: tables, block equations, images
        for seg in segments:
            if seg[0] == "code":
                blocks.append(("code", seg[1], seg[2]))
                continue
            blocks.extend(self._parse_inline(seg[1]))
        return blocks

    def _parse_inline(self, text: str):
        blocks = []
        pattern = re.compile(
            "|".join([_TABLE_RE.pattern, _BLOCK_EQ_RE.pattern, _IMAGE_RE.pattern]),
            re.M | re.S)
        cursor = 0
        for m in pattern.finditer(text):
            before = text[cursor:m.start()]
            if before.strip():
                blocks.append(("text", before, {}))
            piece = m.group(0)
            if _TABLE_RE.match(piece):
                blocks.append(("table", piece, {}))
            elif piece.startswith("!["):
                im = _IMAGE_RE.match(piece)
                blocks.append(("image", piece, {"alt": im.group(1),
                                                "src": im.group(2)}))
            else:
                blocks.append(("equation", piece, {"format": "latex"}))
            cursor = m.end()
        tail = text[cursor:]
        if tail.strip():
            blocks.append(("text", tail, {}))
        return blocks

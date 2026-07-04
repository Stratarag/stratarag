"""EntityGraph: lightweight knowledge-graph indexing (GraphRAG, RAG-Anything
style) with zero dependencies.

Entities are extracted from every chunk (heuristic by default, or an LLM
extractor). Chunks sharing entities become graph-connected — across
modalities, since a table and a paragraph mentioning "Q3 Revenue" link to the
same node. Retrieval then combines semantic similarity with graph expansion:
seed chunks come from hybrid search, and their entity-neighbors are pulled in
with per-hop score decay, so evidence scattered across a document (or across
modalities) is assembled even when it doesn't lexically match the query.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .embeddings import tokenize
from .llm import resolve_provider
from .types import Chunk, Message

_STOP_TITLE = {"The", "A", "An", "This", "That", "These", "Those", "It", "In",
               "On", "At", "For", "And", "But", "Or", "If", "When", "What",
               "How", "Why", "Who", "Image", "Table", "Figure"}
_CAP_RE = re.compile(r"\b([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3})\b")
_ACRO_RE = re.compile(r"\b([A-Z]{2,6}\d?)\b")

_LLM_PROMPT = """Extract the named entities (people, organizations, products,
metrics, concepts) from this text. Return ONLY a JSON array of strings.

Text:
{text}"""


class HeuristicEntityExtractor:
    def extract(self, text: str) -> List[str]:
        found: List[str] = []
        for m in _CAP_RE.finditer(text):
            phrase = m.group(1).strip()
            first = phrase.split()[0]
            if first in _STOP_TITLE and len(phrase.split()) == 1:
                continue
            if first in _STOP_TITLE:
                phrase = " ".join(phrase.split()[1:])
            if len(phrase) >= 2:
                found.append(phrase)
        found.extend(_ACRO_RE.findall(text))
        seen, out = set(), []
        for f in found:
            key = f.lower()
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out


class LLMEntityExtractor:
    def __init__(self, model: Any):
        self.provider = resolve_provider(model)
        self._fallback = HeuristicEntityExtractor()

    def extract(self, text: str) -> List[str]:
        try:
            resp = self.provider.complete(
                [Message(role="user", content=_LLM_PROMPT.format(text=text[:2000]))])
            raw = resp.text
            start, end = raw.find("["), raw.rfind("]")
            ents = json.loads(raw[start:end + 1])
            out = [str(e).strip() for e in ents if str(e).strip()]
            if out:
                return out
        except Exception:
            pass
        return self._fallback.extract(text)


class EntityGraph:
    def __init__(self, extractor: Any = "heuristic", model: Any = None):
        if extractor == "heuristic" or extractor is None:
            self.extractor = HeuristicEntityExtractor()
        elif extractor == "llm":
            self.extractor = LLMEntityExtractor(model)
        elif hasattr(extractor, "extract"):
            self.extractor = extractor
        else:
            raise ValueError(f"Unknown entity extractor: {extractor!r}")
        self.entity_chunks: Dict[str, Set[str]] = defaultdict(set)  # entity -> chunk ids
        self.chunk_entities: Dict[str, Set[str]] = defaultdict(set)
        self.edges: Counter = Counter()          # (e1, e2) co-occurrence weight
        self.entity_names: Dict[str, str] = {}   # canonical key -> display name

    # ---------------------------------------------------------------- build
    def index(self, chunk: Chunk) -> List[str]:
        entities = self.extractor.extract(chunk.text)
        keys = []
        for ent in entities:
            key = ent.lower()
            self.entity_names.setdefault(key, ent)
            self.entity_chunks[key].add(chunk.id)
            self.chunk_entities[chunk.id].add(key)
            keys.append(key)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a != b:
                    self.edges[tuple(sorted((a, b)))] += 1
        return keys

    # ------------------------------------------------------------ traversal
    def neighbors(self, entity_key: str) -> List[Tuple[str, int]]:
        out = []
        for (a, b), w in self.edges.items():
            if a == entity_key:
                out.append((b, w))
            elif b == entity_key:
                out.append((a, w))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def entities_in(self, text: str) -> List[str]:
        """Entity keys mentioned in free text (query linking)."""
        toks = set(tokenize(text))
        hits = []
        for key in self.entity_chunks:
            if all(t in toks for t in tokenize(key)):
                hits.append(key)
        return hits

    def expand(self, seed_chunk_ids: List[str], query: str = "",
               hops: int = 1, decay: float = 0.5,
               limit: int = 20) -> Dict[str, float]:
        """Graph expansion: from seed chunks (and entities named in the
        query), walk entity links outward. Returns chunk_id -> graph score."""
        frontier: Set[str] = set()
        for cid in seed_chunk_ids:
            frontier |= self.chunk_entities.get(cid, set())
        frontier |= set(self.entities_in(query))
        scores: Dict[str, float] = {}
        weight = 1.0
        visited_entities: Set[str] = set()
        for _ in range(max(1, hops)):
            next_frontier: Set[str] = set()
            for ent in frontier:
                if ent in visited_entities:
                    continue
                visited_entities.add(ent)
                for cid in self.entity_chunks.get(ent, ()):
                    scores[cid] = max(scores.get(cid, 0.0), weight)
                for nbr, _w in self.neighbors(ent)[:5]:
                    next_frontier.add(nbr)
            frontier = next_frontier
            weight *= decay
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return dict(ranked)

    def stats(self) -> Dict[str, int]:
        return {"entities": len(self.entity_chunks),
                "edges": len(self.edges),
                "indexed_chunks": len(self.chunk_entities)}

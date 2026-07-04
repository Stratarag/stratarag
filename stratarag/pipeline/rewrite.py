"""Query rewriters for retrieval."""
from __future__ import annotations

import re
from typing import Any

from ..llm import resolve_provider
from ..types import Message

_FILLER = re.compile(
    r"^(please|hey|hi|hello|can you|could you|would you|tell me|i want to know"
    r"|i'd like to know|what about|so)\b[\s,]*", re.I)

_REWRITE_PROMPT = """Rewrite this user question as a short, keyword-rich search
query for a document retrieval system. Return ONLY the rewritten query, nothing
else.

Question: {query}"""


class HeuristicRewriter:
    """Strips conversational filler and normalizes whitespace. Cheap, safe."""

    def rewrite(self, query: str) -> str:
        q = query.strip()
        prev = None
        while prev != q:
            prev = q
            q = _FILLER.sub("", q).strip()
        q = re.sub(r"\s+", " ", q).strip(" ?!.")
        return q or query


class LLMRewriter:
    """Asks a model to rewrite the query; falls back to the heuristic on any
    failure or empty output."""

    def __init__(self, model: Any):
        self.provider = resolve_provider(model)
        self._fallback = HeuristicRewriter()

    def rewrite(self, query: str) -> str:
        try:
            resp = self.provider.complete(
                [Message(role="user", content=_REWRITE_PROMPT.format(query=query))])
            out = resp.text.strip().strip('"').splitlines()[0].strip()
            if 0 < len(out) <= 300:
                return out
        except Exception:
            pass
        return self._fallback.rewrite(query)


def resolve_rewriter(spec: Any, model: Any = None):
    if spec is None or spec == "heuristic":
        return HeuristicRewriter()
    if spec == "llm":
        if model is None:
            raise ValueError("rewriter='llm' requires a model")
        return LLMRewriter(model)
    if hasattr(spec, "rewrite"):
        return spec
    if callable(spec):
        class _Fn:
            def rewrite(self, q, _fn=spec):
                return _fn(q)
        return _Fn()
    raise ValueError(f"Unknown rewriter spec: {spec!r}")

"""EchoProvider: a deterministic, offline provider.

Two modes:
- scripted: pass ``script=[...]``; each call pops the next item. A string item
  becomes a text response; a dict like ``{"tool": "search", "args": {...}}``
  becomes a tool call. Ideal for tests.
- synthetic: with no script, it behaves like an instructed grounded model:
  it ranks the [source N] blocks in the system message by overlap with the
  user question, answers from the most relevant ones, and says it does not
  know when nothing relevant was retrieved.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from ..embeddings import tokenize
from ..types import Message, ToolCall
from .base import LLMProvider, LLMResponse

_SOURCE_SPLIT = re.compile(r"\[source \d+\]\s*")
_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
         "and", "or", "for", "it", "this", "that", "with", "as", "at", "by",
         "be", "from", "i", "you", "we", "they", "he", "she", "do", "does",
         "what", "who", "how", "when", "where", "why", "which", "can", "could",
         "should", "would", "will", "my", "our", "your", "me", "us", "much",
         "many", "get", "about", "hi", "hello", "hey", "name", "please", "am",
         "m", "s", "t", "there", "if", "not", "no", "its", "their", "tell",
         "know", "want", "need", "just", "lets", "let", "ok", "thanks",
         "thank", "till", "until", "lol"}

_DONT_KNOW = ("I don't have enough information in the provided sources to "
              "answer that.")


def _content_words(text: str) -> set:
    return {t for t in tokenize(text) if t not in _STOP}


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, script: Optional[List[Any]] = None,
                 relevance_floor: float = 0.3):
        self._script = list(script) if script else []
        self.relevance_floor = relevance_floor
        self.calls: List[List[Message]] = []

    def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, dict) and "tool" in item:
                return LLMResponse(
                    tool_calls=[ToolCall(name=item["tool"], args=item.get("args", {}))]
                )
            if isinstance(item, LLMResponse):
                return item
            return LLMResponse(text=str(item))
        return LLMResponse(text=self._synthesize(messages))

    def _sources(self, system: str) -> List[str]:
        blocks = _SOURCE_SPLIT.split(system)[1:]
        out = []
        for b in blocks:
            # a source block ends at the first blank line (next prompt part)
            out.append(b.split("\n\n", 1)[0].strip())
        return out

    def _synthesize(self, messages: List[Message]) -> str:
        system = "\n".join(m.content for m in messages if m.role == "system")
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        tool_notes = [m.content for m in messages if m.role == "tool"]
        snippets = self._sources(system)
        parts: List[str] = []
        if snippets:
            q_words = _content_words(user)
            def matched(snippet: str) -> set:
                return q_words & _content_words(snippet)
            def rel(snippet: str) -> float:
                if not q_words:
                    return 1.0
                return len(matched(snippet)) / len(q_words)
            ranked = sorted(snippets, key=rel, reverse=True)
            # compound questions: judge coverage over the top snippets
            # combined, not any single chunk alone
            covered = matched(ranked[0]) | (matched(ranked[1])
                                            if len(ranked) > 1 else set())
            coverage = len(covered) / len(q_words) if q_words else 1.0
            if coverage < self.relevance_floor:
                parts.append(_DONT_KNOW)
            else:
                parts.append(" ".join(ranked[0].split()))
                if len(ranked) > 1 and (
                        rel(ranked[1]) >= self.relevance_floor
                        or matched(ranked[1]) - matched(ranked[0])):
                    parts.append(" ".join(ranked[1].split()))
        if tool_notes:
            parts.append(f"Tool result: {tool_notes[-1]}")
        if not parts:
            return ("I don't have enough information in the provided context "
                    f"to answer: {user}")
        return " ".join(parts)

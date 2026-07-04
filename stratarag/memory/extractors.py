"""Fact extractors for semantic memory.

- HeuristicExtractor: dependency-free pattern matching (offline default).
- LLMExtractor: asks a model to pull durable facts out of a conversation
  turn and return them as a JSON array. Falls back to the heuristic if the
  model output isn't parseable.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from ..llm import LLMProvider, resolve_provider
from ..types import Message

_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)", re.I),
    re.compile(r"\bI(?:'m| am)\s+([A-Z][\w'-]+)"),          # "I'm Rohan" (capitalized -> a name)
    re.compile(r"\bfrom the\s+([\w -]{2,30}?)\s+team\b", re.I),
    re.compile(r"\bi work (?:in|on|for)\s+([\w .,'&/-]{2,50})", re.I),
    re.compile(r"\bi(?:'m| am)\s+(?:a|an)\s+([\w -]{3,40})", re.I),
    re.compile(r"\bi (?:prefer|like|love|enjoy|use|work (?:at|on|with))\s+([\w .,'&/-]{3,60})", re.I),
    re.compile(r"\bi (?:don't|do not|dislike|hate)\s+(?:like\s+)?([\w .,'&/-]{3,60})", re.I),
    re.compile(r"\bcall me\s+([\w'-]{2,30})", re.I),
    re.compile(r"\bi live in\s+([\w .,'-]{2,40})", re.I),
]

_EXTRACT_PROMPT = """You extract durable user facts from a conversation turn.
Return ONLY a JSON array of short fact strings written in third person
(e.g. ["User's name is Priya", "User prefers TypeScript"]).
Return [] if there is nothing durable (no small talk, no one-off requests).

Turn:
{turn}"""


class HeuristicExtractor:
    def extract(self, user_text: str, assistant_text: str = "") -> List[str]:
        facts: List[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", user_text):
            for pat in _PATTERNS:
                m = pat.search(sentence)
                if m:
                    fact = re.sub(r"\s+", " ", sentence.strip().rstrip(".!?"))
                    if fact and fact not in facts:
                        facts.append(fact)
                    break
        return facts


class LLMExtractor:
    def __init__(self, model, fallback: bool = True):
        self.provider: LLMProvider = resolve_provider(model)
        self._fallback = HeuristicExtractor() if fallback else None

    def extract(self, user_text: str, assistant_text: str = "") -> List[str]:
        turn = f"user: {user_text}\nassistant: {assistant_text}".strip()
        try:
            resp = self.provider.complete(
                [Message(role="user", content=_EXTRACT_PROMPT.format(turn=turn))])
            raw = resp.text.strip()
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError("no JSON array in output")
            facts = json.loads(raw[start : end + 1])
            return [str(f).strip() for f in facts if str(f).strip()]
        except Exception:
            if self._fallback:
                return self._fallback.extract(user_text, assistant_text)
            return []


def resolve_extractor(spec, model=None):
    if spec is None or spec == "heuristic":
        return HeuristicExtractor()
    if spec == "llm":
        if model is None:
            raise ValueError("extractor='llm' requires a model")
        return LLMExtractor(model)
    if hasattr(spec, "extract"):
        return spec
    raise ValueError(f"Unknown extractor spec: {spec!r}")

"""VectorStore interface. All backends — local or cloud — implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Hit:
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


FilterFn = Optional[Callable[[Dict[str, Any]], bool]]


def payload_filter(where: Optional[Dict[str, Any]]) -> FilterFn:
    """Build a filter callable from an equality dict, e.g. {'user_id': 'u1'}."""
    if not where:
        return None
    def fn(payload: Dict[str, Any]) -> bool:
        return all(payload.get(k) == v for k, v in where.items())
    return fn


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[str], vectors: List[List[float]],
            payloads: List[Dict[str, Any]]) -> None: ...

    @abstractmethod
    def query(self, vector: List[float], k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[Hit]: ...

    @abstractmethod
    def get(self, ids: List[str]) -> List[Optional[Dict[str, Any]]]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    def all_payloads(self, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

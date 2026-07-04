from __future__ import annotations

from typing import Any, Union

from ..errors import ConfigurationError
from .base import Chunker, split_sentences, word_count
from .modality import ModalityChunker, linearize_table
from .strategies import (
    FixedSizeChunker,
    MarkdownChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
)

_REGISTRY = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "markdown": MarkdownChunker,
    "semantic": SemanticChunker,
    "parent_child": ParentChildChunker,
    "modality": ModalityChunker,
}

ChunkerLike = Union[str, Chunker]


def resolve_chunker(spec: ChunkerLike, **kwargs: Any) -> Chunker:
    if isinstance(spec, Chunker):
        return spec
    if isinstance(spec, str) and spec in _REGISTRY:
        return _REGISTRY[spec](**kwargs)
    raise ConfigurationError(
        f"Unknown chunking strategy: {spec!r}. Available: {sorted(_REGISTRY)}"
    )


__all__ = [
    "Chunker", "FixedSizeChunker", "RecursiveChunker", "MarkdownChunker",
    "SemanticChunker", "ParentChildChunker", "ModalityChunker",
    "linearize_table", "resolve_chunker",
    "split_sentences", "word_count",
]

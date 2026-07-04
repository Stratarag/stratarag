from __future__ import annotations

from typing import Union

from ..errors import ConfigurationError
from .base import Embedder
from .hashing import HashingEmbedder, tokenize

EmbedderLike = Union[str, Embedder]


def resolve_embedder(spec: EmbedderLike) -> Embedder:
    if isinstance(spec, Embedder):
        return spec
    if isinstance(spec, str):
        if spec == "hashing":
            return HashingEmbedder()
        if spec.startswith("hashing:"):
            return HashingEmbedder(dim=int(spec.split(":", 1)[1]))
        if spec.startswith("sentence-transformers:"):
            from .sbert import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(spec.split(":", 1)[1])
        if spec == "openai" or spec.startswith("openai:"):
            from .providers import OpenAIEmbedder
            model = spec.split(":", 1)[1] if ":" in spec else "text-embedding-3-small"
            return OpenAIEmbedder(model)
        if spec.startswith("azure-openai:"):
            from .providers import AzureOpenAIEmbedder
            return AzureOpenAIEmbedder(spec.split(":", 1)[1])
        if spec == "cohere" or spec.startswith("cohere:"):
            from .providers import CohereEmbedder
            model = spec.split(":", 1)[1] if ":" in spec else "embed-english-v3.0"
            return CohereEmbedder(model)
        if spec == "vertex" or spec.startswith("vertex:"):
            from .providers import VertexEmbedder
            model = spec.split(":", 1)[1] if ":" in spec else "text-embedding-004"
            return VertexEmbedder(model)
    raise ConfigurationError(
        f"Unknown embedder spec: {spec!r}. Use 'hashing', 'hashing:<dim>', "
        "'sentence-transformers:<model>', 'openai:<model>', "
        "'azure-openai:<deployment>', 'cohere:<model>', 'vertex:<model>', "
        "or an Embedder instance."
    )


__all__ = ["Embedder", "HashingEmbedder", "resolve_embedder", "tokenize"]

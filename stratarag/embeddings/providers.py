"""API-backed embedding providers (all optional dependencies):
OpenAI, Azure OpenAI, Cohere, Google Vertex AI.

Every provider L2-normalizes output so cosine scoring stays consistent with
the local stores, and batches requests to respect API limits.
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..errors import MissingDependencyError
from .base import Embedder

_DIMS = {
    "text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "embed-english-v3.0": 1024, "embed-multilingual-v3.0": 1024,
    "text-embedding-004": 768, "text-embedding-005": 768,
    "gemini-embedding-001": 3072,
}


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _batched(texts: List[str], size: int):
    for i in range(0, len(texts), size):
        yield texts[i:i + size]


class OpenAIEmbedder(Embedder):
    """spec: "openai:text-embedding-3-small"  (OPENAI_API_KEY env)"""

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None, batch_size: int = 512):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("openai", "openai", "OpenAIEmbedder") from e
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.dim = _DIMS.get(model, 1536)
        self._batch = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - network
        out: List[List[float]] = []
        for batch in _batched(texts, self._batch):
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(_normalize(list(d.embedding)) for d in resp.data)
        return out


class AzureOpenAIEmbedder(Embedder):
    """spec: "azure-openai:<deployment-name>"
    (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT env)"""

    def __init__(self, deployment: str, api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_version: str = "2024-06-01", dim: int = 1536,
                 batch_size: int = 512):
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("openai", "openai", "AzureOpenAIEmbedder") from e
        import os
        self._client = AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=api_version)
        self.deployment = deployment
        self.dim = dim
        self._batch = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - network
        out: List[List[float]] = []
        for batch in _batched(texts, self._batch):
            resp = self._client.embeddings.create(model=self.deployment, input=batch)
            out.extend(_normalize(list(d.embedding)) for d in resp.data)
        return out


class CohereEmbedder(Embedder):
    """spec: "cohere:embed-english-v3.0"  (CO_API_KEY env)"""

    def __init__(self, model: str = "embed-english-v3.0",
                 api_key: Optional[str] = None, batch_size: int = 96):
        try:
            import cohere  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("cohere", "cohere", "CohereEmbedder") from e
        self._client = cohere.Client(api_key) if api_key else cohere.Client()
        self.model = model
        self.dim = _DIMS.get(model, 1024)
        self._batch = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - network
        out: List[List[float]] = []
        for batch in _batched(texts, self._batch):
            resp = self._client.embed(texts=list(batch), model=self.model,
                                      input_type="search_document")
            out.extend(_normalize(list(v)) for v in resp.embeddings)
        return out


class VertexEmbedder(Embedder):
    """spec: "vertex:text-embedding-004"
    (uses google-cloud-aiplatform; GOOGLE_APPLICATION_CREDENTIALS env,
    project/location via GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_REGION)"""

    def __init__(self, model: str = "text-embedding-004",
                 project: Optional[str] = None, location: Optional[str] = None,
                 batch_size: int = 64):
        try:
            import vertexai  # type: ignore
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("google-cloud-aiplatform", "vertex",
                                         "VertexEmbedder") from e
        import os
        vertexai.init(
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"))
        self._model = TextEmbeddingModel.from_pretrained(model)
        self.dim = _DIMS.get(model, 768)
        self._batch = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - network
        out: List[List[float]] = []
        for batch in _batched(texts, self._batch):
            for emb in self._model.get_embeddings(list(batch)):
                out.append(_normalize(list(emb.values)))
        return out

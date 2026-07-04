from __future__ import annotations

from typing import Union

from ..errors import ConfigurationError
from .base import Hit, VectorStore, cosine
from .local import InMemoryVectorStore, SQLiteVectorStore

StoreLike = Union[str, VectorStore]


def resolve_store(spec: StoreLike, dim: int = 512) -> VectorStore:
    """Resolve a store spec:

    - VectorStore instance -> returned as-is
    - "memory"             -> InMemoryVectorStore
    - "sqlite:<path>"      -> SQLiteVectorStore
    - "chroma[:<path>]"    -> ChromaVectorStore        (pip install stratarag[chroma])
    - "qdrant:<url>"       -> QdrantVectorStore        (pip install stratarag[qdrant])
    - "pgvector:<dsn>"     -> PgVectorStore            (pip install stratarag[pgvector])
    """
    if isinstance(spec, VectorStore):
        return spec
    if isinstance(spec, str):
        if spec == "memory":
            return InMemoryVectorStore()
        if spec == "sqlite" or spec.startswith("sqlite:"):
            path = spec.split(":", 1)[1] if ":" in spec else ":memory:"
            return SQLiteVectorStore(path=path or ":memory:")
        if spec == "chroma" or spec.startswith("chroma:"):
            from .adapters import ChromaVectorStore
            path = spec.split(":", 1)[1] if ":" in spec else None
            return ChromaVectorStore(path=path or None)
        if spec.startswith("qdrant:"):
            from .adapters import QdrantVectorStore
            return QdrantVectorStore(url=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("pgvector:"):
            from .adapters import PgVectorStore
            return PgVectorStore(dsn=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("pinecone:"):
            from .adapters2 import PineconeVectorStore
            return PineconeVectorStore(index=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("weaviate:"):
            from .adapters2 import WeaviateVectorStore
            return WeaviateVectorStore(url=spec.split(":", 1)[1])
        if spec.startswith("milvus:"):
            from .adapters2 import MilvusVectorStore
            return MilvusVectorStore(uri=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("elasticsearch:"):
            from .adapters2 import ElasticsearchVectorStore
            return ElasticsearchVectorStore(url=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("redis:"):
            from .adapters2 import RedisVectorStore
            return RedisVectorStore(url=spec.split(":", 1)[1], dim=dim)
        if spec.startswith("mongodb:"):
            from .adapters2 import MongoDBVectorStore
            rest = spec.split(":", 1)[1]
            parts = rest.split("#")
            uri = parts[0]
            db, col = (parts[1].split(".", 1) if len(parts) > 1 and "." in parts[1]
                       else ("stratarag", "vectors"))
            idx = parts[2] if len(parts) > 2 else "vector_index"
            return MongoDBVectorStore(uri=uri, db=db, collection=col,
                                      search_index=idx, dim=dim)
    raise ConfigurationError(
        f"Unknown store spec: {spec!r}. Use 'memory', 'sqlite:<path>', "
        "'chroma[:<path>]', 'qdrant:<url>', 'pgvector:<dsn>', "
        "'pinecone:<index>', 'weaviate:<url>', 'milvus:<uri>', "
        "'elasticsearch:<url>', 'redis:<url>', "
        "'mongodb:<uri>#<db>.<col>#<index>', or a VectorStore instance."
    )


__all__ = ["VectorStore", "Hit", "cosine", "InMemoryVectorStore",
           "SQLiteVectorStore", "resolve_store"]

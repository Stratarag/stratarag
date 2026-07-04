"""Adapters for external vector databases (all optional dependencies).

Each adapter implements the same `VectorStore` interface, so switching
backends is a one-line change:

    Knowledge(store="memory")                     # dev
    Knowledge(store="chroma:./chroma_dir")        # local persistent
    Knowledge(store="qdrant:http://localhost:6333")
    Knowledge(store="pgvector:postgresql://...")
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..errors import MissingDependencyError, StoreError
from .base import Hit, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Optional[str] = None, collection: str = "stratarag"):
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("chromadb", "chroma", "ChromaVectorStore") from e
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _flatten(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"_json": json.dumps(payload)}

    @staticmethod
    def _restore(meta: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(meta.get("_json", "{}"))

    def add(self, ids, vectors, payloads):
        self._col.upsert(ids=ids, embeddings=vectors,
                         metadatas=[self._flatten(p) for p in payloads])

    def query(self, vector, k=5, where=None):
        res = self._col.query(query_embeddings=[vector], n_results=k)
        hits: List[Hit] = []
        for i, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
            payload = self._restore(meta or {})
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=i, score=1.0 - float(dist), payload=payload))
        return hits

    def get(self, ids):
        res = self._col.get(ids=ids)
        found = {i: self._restore(m or {}) for i, m in zip(res["ids"], res["metadatas"])}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._col.delete(ids=ids)

    def count(self):
        return self._col.count()


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str = "http://localhost:6333", collection: str = "stratarag",
                 dim: int = 512, api_key: Optional[str] = None):
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as qm  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("qdrant-client", "qdrant", "QdrantVectorStore") from e
        self._qm = qm
        self._client = QdrantClient(url=url, api_key=api_key) if url != ":memory:" \
            else QdrantClient(":memory:")
        self._collection = collection
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))

    @staticmethod
    def _uuid(id_: str) -> str:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, id_))

    def add(self, ids, vectors, payloads):
        points = [
            self._qm.PointStruct(id=self._uuid(i), vector=v,
                                 payload={**p, "_stratarag_id": i})
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def _where(self, where: Optional[Dict[str, Any]]):
        if not where:
            return None
        return self._qm.Filter(must=[
            self._qm.FieldCondition(key=k, match=self._qm.MatchValue(value=v))
            for k, v in where.items()
        ])

    def query(self, vector, k=5, where=None):
        res = self._client.search(
            collection_name=self._collection, query_vector=vector,
            limit=k, query_filter=self._where(where), with_payload=True)
        hits = []
        for p in res:
            payload = dict(p.payload or {})
            mid = payload.pop("_stratarag_id", str(p.id))
            hits.append(Hit(id=mid, score=float(p.score), payload=payload))
        return hits

    def get(self, ids):
        res = self._client.retrieve(
            collection_name=self._collection,
            ids=[self._uuid(i) for i in ids], with_payload=True)
        found = {}
        for p in res:
            payload = dict(p.payload or {})
            found[payload.pop("_stratarag_id", str(p.id))] = payload
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._client.delete(collection_name=self._collection,
                            points_selector=[self._uuid(i) for i in ids])

    def count(self):
        return self._client.count(collection_name=self._collection).count


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector. Requires `CREATE EXTENSION vector;` on the DB."""

    def __init__(self, dsn: str, table: str = "stratarag_vectors", dim: int = 512):
        try:
            import psycopg  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("psycopg[binary]", "pgvector", "PgVectorStore") from e
        if not table.replace("_", "").isalnum():
            raise StoreError("table must be alphanumeric/underscore")
        self._psycopg = psycopg
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                f"(id TEXT PRIMARY KEY, embedding vector({dim}), payload JSONB)")

    @staticmethod
    def _vec(v: List[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

    def add(self, ids, vectors, payloads):
        with self._conn.cursor() as cur:
            for i, v, p in zip(ids, vectors, payloads):
                cur.execute(
                    f"INSERT INTO {self._table} (id, embedding, payload) "
                    "VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE "
                    "SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
                    (i, self._vec(v), json.dumps(p)))

    def query(self, vector, k=5, where=None):
        sql = (f"SELECT id, payload, 1 - (embedding <=> %s::vector) AS score "
               f"FROM {self._table}")
        params: List[Any] = [self._vec(vector)]
        if where:
            conds = []
            for kk, vv in where.items():
                conds.append("payload ->> %s = %s")
                params.extend([kk, str(vv)])
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([self._vec(vector), k])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [Hit(id=r[0], payload=r[1], score=float(r[2])) for r in cur.fetchall()]

    def get(self, ids):
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {self._table} WHERE id = ANY(%s)", (ids,))
            found = {r[0]: r[1] for r in cur.fetchall()}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE id = ANY(%s)", (ids,))

    def count(self):
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

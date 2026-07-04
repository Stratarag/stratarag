"""Cloud & enterprise vector DB adapters (all optional dependencies):
Pinecone, Weaviate, Milvus, Elasticsearch, Redis, MongoDB Atlas.
All implement the same `VectorStore` contract as the local backends."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..errors import MissingDependencyError, StoreError
from .base import Hit, VectorStore


class PineconeVectorStore(VectorStore):
    """spec: "pinecone:<index-name>"  (PINECONE_API_KEY env or api_key=...)"""

    def __init__(self, index: str, api_key: Optional[str] = None,
                 namespace: str = "", dim: int = 512):
        try:
            from pinecone import Pinecone, ServerlessSpec  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("pinecone", "pinecone", "PineconeVectorStore") from e
        import os
        self._pc = Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY"))
        names = [i["name"] for i in self._pc.list_indexes()]
        if index not in names:
            self._pc.create_index(name=index, dimension=dim, metric="cosine",
                                  spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self._index = self._pc.Index(index)
        self._ns = namespace

    @staticmethod
    def _meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"_json": json.dumps(payload)}

    @staticmethod
    def _payload(meta: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(meta.get("_json", "{}"))

    def add(self, ids, vectors, payloads):
        self._index.upsert(
            vectors=[{"id": i, "values": v, "metadata": self._meta(p)}
                     for i, v, p in zip(ids, vectors, payloads)],
            namespace=self._ns)

    def query(self, vector, k=5, where=None):
        res = self._index.query(vector=vector, top_k=k, namespace=self._ns,
                                include_metadata=True)
        hits = []
        for m in res.get("matches", []):
            payload = self._payload(m.get("metadata") or {})
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=m["id"], score=float(m["score"]), payload=payload))
        return hits

    def get(self, ids):
        res = self._index.fetch(ids=ids, namespace=self._ns)
        found = {i: self._payload(rec.get("metadata") or {})
                 for i, rec in (res.get("vectors") or {}).items()}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._index.delete(ids=ids, namespace=self._ns)

    def count(self):
        stats = self._index.describe_index_stats()
        if self._ns:
            return stats.get("namespaces", {}).get(self._ns, {}).get("vector_count", 0)
        return stats.get("total_vector_count", 0)


class WeaviateVectorStore(VectorStore):
    """spec: "weaviate:<url>"  (or url="embedded" for embedded weaviate)"""

    def __init__(self, url: str = "http://localhost:8080",
                 collection: str = "StrataRAG", api_key: Optional[str] = None):
        try:
            import weaviate  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("weaviate-client", "weaviate", "WeaviateVectorStore") from e
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        if url in ("embedded", ""):
            self._client = weaviate.connect_to_embedded()
        elif "localhost" in url or "127.0.0.1" in url:
            port = int(url.rsplit(":", 1)[1]) if url.count(":") >= 2 else 8080
            self._client = weaviate.connect_to_local(port=port, auth_credentials=auth)
        else:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url, auth_credentials=auth)
        self._name = collection
        if not self._client.collections.exists(collection):
            self._client.collections.create(collection)
        self._col = self._client.collections.get(collection)

    @staticmethod
    def _uuid(id_: str) -> str:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, id_))

    def add(self, ids, vectors, payloads):
        with self._col.batch.dynamic() as batch:
            for i, v, p in zip(ids, vectors, payloads):
                batch.add_object(
                    properties={"stratarag_id": i, "payload_json": json.dumps(p)},
                    uuid=self._uuid(i), vector=v)

    def query(self, vector, k=5, where=None):
        res = self._col.query.near_vector(near_vector=vector, limit=k,
                                          return_metadata=["distance"])
        hits = []
        for o in res.objects:
            payload = json.loads(o.properties.get("payload_json", "{}"))
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            dist = getattr(o.metadata, "distance", 0.0) or 0.0
            hits.append(Hit(id=o.properties.get("stratarag_id", str(o.uuid)),
                            score=1.0 - float(dist), payload=payload))
        return hits

    def get(self, ids):
        out = []
        for i in ids:
            try:
                o = self._col.query.fetch_object_by_id(self._uuid(i))
                out.append(json.loads(o.properties["payload_json"]) if o else None)
            except Exception:
                out.append(None)
        return out

    def delete(self, ids):
        for i in ids:
            self._col.data.delete_by_id(self._uuid(i))

    def count(self):
        return self._col.aggregate.over_all(total_count=True).total_count


class MilvusVectorStore(VectorStore):
    """spec: "milvus:<uri>"  e.g. milvus:http://localhost:19530 or a local
    'milvus:./milvus.db' file via Milvus Lite."""

    def __init__(self, uri: str, collection: str = "stratarag", dim: int = 512,
                 token: Optional[str] = None):
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("pymilvus", "milvus", "MilvusVectorStore") from e
        self._client = MilvusClient(uri=uri, token=token or "")
        self._name = collection
        if not self._client.has_collection(collection):
            self._client.create_collection(
                collection_name=collection, dimension=dim, metric_type="COSINE",
                auto_id=False, id_type="string", max_length=64)

    def add(self, ids, vectors, payloads):
        self._client.upsert(collection_name=self._name, data=[
            {"id": i, "vector": v, "payload_json": json.dumps(p)}
            for i, v, p in zip(ids, vectors, payloads)])

    def query(self, vector, k=5, where=None):
        res = self._client.search(collection_name=self._name, data=[vector],
                                  limit=k, output_fields=["payload_json"])
        hits = []
        for m in (res[0] if res else []):
            payload = json.loads(m["entity"].get("payload_json", "{}"))
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=str(m["id"]), score=float(m["distance"]),
                            payload=payload))
        return hits

    def get(self, ids):
        res = self._client.get(collection_name=self._name, ids=ids,
                               output_fields=["payload_json"])
        found = {str(r["id"]): json.loads(r.get("payload_json", "{}")) for r in res}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._client.delete(collection_name=self._name, ids=ids)

    def count(self):
        stats = self._client.get_collection_stats(self._name)
        return int(stats.get("row_count", 0))


class ElasticsearchVectorStore(VectorStore):
    """spec: "elasticsearch:<url>"  (ELASTIC_API_KEY env optional). Uses a
    dense_vector field with cosine similarity + script_score fallback."""

    def __init__(self, url: str = "http://localhost:9200",
                 index: str = "stratarag", dim: int = 512,
                 api_key: Optional[str] = None):
        try:
            from elasticsearch import Elasticsearch  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("elasticsearch", "elasticsearch",
                                         "ElasticsearchVectorStore") from e
        import os
        key = api_key or os.environ.get("ELASTIC_API_KEY")
        self._es = Elasticsearch(url, api_key=key) if key else Elasticsearch(url)
        self._index = index
        if not self._es.indices.exists(index=index):
            self._es.indices.create(index=index, mappings={"properties": {
                "vector": {"type": "dense_vector", "dims": dim,
                           "index": True, "similarity": "cosine"},
                "payload_json": {"type": "keyword", "index": False},
            }})

    def add(self, ids, vectors, payloads):
        from elasticsearch import helpers  # type: ignore
        helpers.bulk(self._es, [
            {"_index": self._index, "_id": i,
             "_source": {"vector": v, "payload_json": json.dumps(p)}}
            for i, v, p in zip(ids, vectors, payloads)])
        self._es.indices.refresh(index=self._index)

    def query(self, vector, k=5, where=None):
        res = self._es.search(index=self._index, knn={
            "field": "vector", "query_vector": vector,
            "k": k, "num_candidates": max(50, k * 5)})
        hits = []
        for h in res["hits"]["hits"]:
            payload = json.loads(h["_source"].get("payload_json", "{}"))
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=h["_id"], score=float(h["_score"]), payload=payload))
        return hits

    def get(self, ids):
        res = self._es.mget(index=self._index, ids=ids)
        return [json.loads(d["_source"]["payload_json"]) if d.get("found") else None
                for d in res["docs"]]

    def delete(self, ids):
        for i in ids:
            self._es.delete(index=self._index, id=i, ignore=[404])
        self._es.indices.refresh(index=self._index)

    def count(self):
        return int(self._es.count(index=self._index)["count"])


class RedisVectorStore(VectorStore):
    """spec: "redis:<url>"  e.g. redis:redis://localhost:6379. Uses RediSearch
    (Redis Stack) HNSW index over hashes."""

    def __init__(self, url: str = "redis://localhost:6379",
                 index: str = "stratarag", dim: int = 512):
        try:
            import redis  # type: ignore
            from redis.commands.search.field import TextField, VectorField  # type: ignore
            from redis.commands.search.indexDefinition import (IndexDefinition,  # type: ignore
                                                               IndexType)
        except ImportError as e:
            raise MissingDependencyError("redis", "redis", "RedisVectorStore") from e
        self._r = redis.from_url(url)
        self._index = index
        self._prefix = f"{index}:"
        self._dim = dim
        try:
            self._r.ft(index).info()
        except Exception:
            self._r.ft(index).create_index(
                fields=[TextField("payload_json"),
                        VectorField("vector", "HNSW",
                                    {"TYPE": "FLOAT32", "DIM": dim,
                                     "DISTANCE_METRIC": "COSINE"})],
                definition=IndexDefinition(prefix=[self._prefix],
                                           index_type=IndexType.HASH))

    @staticmethod
    def _pack(v: List[float]) -> bytes:
        import struct
        return struct.pack(f"{len(v)}f", *v)

    def add(self, ids, vectors, payloads):
        pipe = self._r.pipeline()
        for i, v, p in zip(ids, vectors, payloads):
            pipe.hset(self._prefix + i, mapping={
                "vector": self._pack(v), "payload_json": json.dumps(p)})
        pipe.execute()

    def query(self, vector, k=5, where=None):
        from redis.commands.search.query import Query  # type: ignore
        q = (Query(f"*=>[KNN {k} @vector $vec AS dist]")
             .sort_by("dist").return_fields("payload_json", "dist").dialect(2))
        res = self._r.ft(self._index).search(q, {"vec": self._pack(vector)})
        hits = []
        for doc in res.docs:
            payload = json.loads(doc.payload_json)
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=doc.id[len(self._prefix):],
                            score=1.0 - float(doc.dist), payload=payload))
        return hits

    def get(self, ids):
        out = []
        for i in ids:
            raw = self._r.hget(self._prefix + i, "payload_json")
            out.append(json.loads(raw) if raw else None)
        return out

    def delete(self, ids):
        self._r.delete(*[self._prefix + i for i in ids])

    def count(self):
        return int(self._r.ft(self._index).info()["num_docs"])


class MongoDBVectorStore(VectorStore):
    """spec: "mongodb:<uri>#<db>.<collection>#<search-index>"
    e.g. mongodb:mongodb+srv://user:pw@cluster.mongodb.net#rag.chunks#vector_index
    Requires an Atlas Vector Search index on the 'vector' field."""

    def __init__(self, uri: str, db: str = "stratarag", collection: str = "vectors",
                 search_index: str = "vector_index", dim: int = 512):
        try:
            from pymongo import MongoClient  # type: ignore
        except ImportError as e:
            raise MissingDependencyError("pymongo", "mongodb", "MongoDBVectorStore") from e
        self._col = MongoClient(uri)[db][collection]
        self._search_index = search_index
        self._dim = dim

    def add(self, ids, vectors, payloads):
        from pymongo import ReplaceOne  # type: ignore
        self._col.bulk_write([
            ReplaceOne({"_id": i},
                       {"_id": i, "vector": v, "payload": p}, upsert=True)
            for i, v, p in zip(ids, vectors, payloads)])

    def query(self, vector, k=5, where=None):
        pipeline = [{"$vectorSearch": {
            "index": self._search_index, "path": "vector",
            "queryVector": vector, "numCandidates": max(100, k * 10),
            "limit": k}},
            {"$project": {"payload": 1,
                          "score": {"$meta": "vectorSearchScore"}}}]
        hits = []
        for docu in self._col.aggregate(pipeline):
            payload = docu.get("payload", {})
            if where and any(payload.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(id=str(docu["_id"]), score=float(docu["score"]),
                            payload=payload))
        return hits

    def get(self, ids):
        found = {d["_id"]: d.get("payload")
                 for d in self._col.find({"_id": {"$in": ids}})}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._col.delete_many({"_id": {"$in": ids}})

    def count(self):
        return self._col.count_documents({})

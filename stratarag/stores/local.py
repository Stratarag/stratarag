"""Local stores: in-memory (dev/tests) and SQLite (persistent, zero-infra)."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from .base import Hit, VectorStore, cosine, payload_filter


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}

    def add(self, ids, vectors, payloads):
        for i, v, p in zip(ids, vectors, payloads):
            self._vectors[i] = v
            self._payloads[i] = p

    def query(self, vector, k=5, where=None):
        flt = payload_filter(where)
        hits = [
            Hit(id=i, score=cosine(vector, v), payload=self._payloads[i])
            for i, v in self._vectors.items()
            if flt is None or flt(self._payloads[i])
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def get(self, ids):
        return [self._payloads.get(i) for i in ids]

    def delete(self, ids):
        for i in ids:
            self._vectors.pop(i, None)
            self._payloads.pop(i, None)

    def count(self):
        return len(self._vectors)

    def all_payloads(self, where=None):
        flt = payload_filter(where)
        return [p for p in self._payloads.values() if flt is None or flt(p)]


class SQLiteVectorStore(VectorStore):
    """Persistent brute-force store. Fine up to ~100k vectors; beyond that,
    use the qdrant / pgvector / chroma adapters."""

    def __init__(self, path: str = ":memory:", table: str = "vectors"):
        if not table.replace("_", "").isalnum():
            raise ValueError("table must be alphanumeric/underscore")
        self._table = table
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(id TEXT PRIMARY KEY, vector TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            self._conn.commit()

    def add(self, ids, vectors, payloads):
        rows = [(i, json.dumps(v), json.dumps(p)) for i, v, p in zip(ids, vectors, payloads)]
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self._table} VALUES (?, ?, ?)", rows)
            self._conn.commit()

    def query(self, vector, k=5, where=None):
        flt = payload_filter(where)
        hits: List[Hit] = []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, vector, payload FROM {self._table}").fetchall()
        for i, v_json, p_json in rows:
            payload = json.loads(p_json)
            if flt is not None and not flt(payload):
                continue
            hits.append(Hit(id=i, score=cosine(vector, json.loads(v_json)), payload=payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def get(self, ids):
        out: List[Optional[Dict[str, Any]]] = []
        with self._lock:
            for i in ids:
                row = self._conn.execute(
                    f"SELECT payload FROM {self._table} WHERE id = ?", (i,)).fetchone()
                out.append(json.loads(row[0]) if row else None)
        return out

    def delete(self, ids):
        with self._lock:
            self._conn.executemany(
                f"DELETE FROM {self._table} WHERE id = ?", [(i,) for i in ids])
            self._conn.commit()

    def count(self):
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def all_payloads(self, where=None):
        flt = payload_filter(where)
        with self._lock:
            rows = self._conn.execute(f"SELECT payload FROM {self._table}").fetchall()
        payloads = [json.loads(r[0]) for r in rows]
        return [p for p in payloads if flt is None or flt(p)]
    
    def close(self):
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
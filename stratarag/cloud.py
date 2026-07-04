"""StrataRAG Cloud client (paid tier) — same interfaces, hosted backends.

The open-source library is complete on its own. Cloud adds hosted, durable
memory and knowledge stores plus trace collection, behind the exact same
`VectorStore` interface, so upgrading is one line:

    Memory(backend=CloudStore(api_key="mk_..."))

This module ships as a thin, honest stub: it implements the wire protocol but
requires STRATARAG_API_KEY / api_key to be configured, and raises clear errors
otherwise. No hidden phoning home: nothing here runs unless you construct it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .errors import ConfigurationError, StoreError
from .stores.base import Hit, VectorStore

DEFAULT_BASE_URL = "https://api.stratarag.dev/v1"


class CloudStore(VectorStore):
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 namespace: str = "default", timeout: float = 15.0):
        self.api_key = api_key or os.environ.get("STRATARAG_API_KEY", "")
        if not self.api_key:
            raise ConfigurationError(
                "CloudStore needs an API key: pass api_key=... or set "
                "STRATARAG_API_KEY. Get one at https://stratarag.dev (or keep using "
                "the free local backends — they're the same interface).")
        self.base_url = base_url.rstrip("/")
        self.namespace = namespace
        self.timeout = timeout

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps({"namespace": self.namespace, **body}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise StoreError(f"StrataRAG Cloud unreachable: {e}") from e

    def add(self, ids, vectors, payloads):
        self._post("/vectors/add",
                   {"ids": ids, "vectors": vectors, "payloads": payloads})

    def query(self, vector, k=5, where=None):
        data = self._post("/vectors/query",
                          {"vector": vector, "k": k, "where": where or {}})
        return [Hit(id=h["id"], score=h["score"], payload=h.get("payload", {}))
                for h in data.get("hits", [])]

    def get(self, ids):
        data = self._post("/vectors/get", {"ids": ids})
        found = {h["id"]: h.get("payload") for h in data.get("items", [])}
        return [found.get(i) for i in ids]

    def delete(self, ids):
        self._post("/vectors/delete", {"ids": ids})

    def count(self):
        return int(self._post("/vectors/count", {}).get("count", 0))

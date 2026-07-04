import os
import tempfile
import unittest

from stratarag.embeddings import HashingEmbedder, resolve_embedder
from stratarag.errors import ConfigurationError
from stratarag.knowledge import Knowledge
from stratarag.stores import InMemoryVectorStore, SQLiteVectorStore, resolve_store

DOCS = [
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Shipping takes 5 to 7 business days for standard orders.",
    "Premium support is available around the clock for enterprise plans.",
]


class TestEmbedder(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        e = HashingEmbedder(dim=64)
        v1, v2 = e.embed(["hello world"])[0], e.embed(["hello world"])[0]
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(sum(x * x for x in v1), 1.0, places=5)

    def test_resolve_specs(self):
        self.assertEqual(resolve_embedder("hashing:128").dim, 128)
        with self.assertRaises(ConfigurationError):
            resolve_embedder("bogus")


class _StoreContract:
    def make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def test_add_query_filter_delete(self):
        store = self.make()
        e = HashingEmbedder(dim=32)
        texts = ["alpha beta", "beta gamma", "delta epsilon"]
        vecs = e.embed(texts)
        store.add(["a", "b", "c"], vecs,
                  [{"text": t, "user_id": "u1" if i < 2 else "u2"}
                   for i, t in enumerate(texts)])
        self.assertEqual(store.count(), 3)
        hits = store.query(e.embed_one("alpha beta"), k=2)
        self.assertEqual(hits[0].id, "a")
        hits = store.query(e.embed_one("delta"), k=5, where={"user_id": "u1"})
        self.assertTrue(all(h.payload["user_id"] == "u1" for h in hits))
        self.assertEqual(store.get(["b"])[0]["text"], "beta gamma")
        self.assertIsNone(store.get(["zz"])[0])
        store.delete(["a"])
        self.assertEqual(store.count(), 2)

    def test_upsert_overwrites(self):
        store = self.make()
        e = HashingEmbedder(dim=32)
        store.add(["x"], e.embed(["one"]), [{"text": "one"}])
        store.add(["x"], e.embed(["two"]), [{"text": "two"}])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get(["x"])[0]["text"], "two")


class TestInMemoryStore(_StoreContract, unittest.TestCase):
    def make(self):
        return InMemoryVectorStore()


class TestSQLiteStore(_StoreContract, unittest.TestCase):
    def make(self):
        return SQLiteVectorStore(":memory:")

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.db")
            e = HashingEmbedder(dim=16)
            with SQLiteVectorStore(path) as s1:
                s1.add(["k"], e.embed(["persist me"]), [{"text": "persist me"}])
            with SQLiteVectorStore(path) as s2:
                self.assertEqual(s2.count(), 1)

    def test_bad_table_name(self):
        with self.assertRaises(ValueError):
            SQLiteVectorStore(":memory:", table="bad; drop")


class TestResolveStore(unittest.TestCase):
    def test_specs(self):
        self.assertIsInstance(resolve_store("memory"), InMemoryVectorStore)
        self.assertIsInstance(resolve_store("sqlite"), SQLiteVectorStore)
        with self.assertRaises(ConfigurationError):
            resolve_store("wat")


class TestKnowledge(unittest.TestCase):
    def test_vector_keyword_hybrid(self):
        kb = Knowledge.from_texts(DOCS)
        self.assertIn("Refunds", kb.search("refund window", k=1)[0].chunk.text)
        self.assertIn("Shipping", kb.keyword_search("shipping days", k=1)[0].chunk.text)
        self.assertIn("Premium", kb.hybrid_search("enterprise support", k=1)[0].chunk.text)

    def test_from_docs_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.md"), "w") as f:
                f.write("# Policy\nRefunds within 30 days.")
            with open(os.path.join(d, "b.txt"), "w") as f:
                f.write("Shipping takes a week.")
            with open(os.path.join(d, "c.bin"), "w") as f:
                f.write("ignored")
            kb = Knowledge.from_docs(d)
            self.assertGreaterEqual(len(kb), 2)
            top = kb.search("refund policy", k=1)[0]
            self.assertTrue(top.chunk.metadata["source"].endswith("a.md"))

    def test_parent_child_returns_parent(self):
        text = " ".join(f"the refund clause number {i} applies here." for i in range(80))
        kb = Knowledge(chunking="parent_child", parent_words=120, child_words=15)
        kb.add(text)
        top = kb.search("refund clause", k=3)
        self.assertTrue(all(sc.chunk.metadata.get("role") == "parent" for sc in top))
        # dedupe: no parent twice
        ids = [sc.chunk.id for sc in top]
        self.assertEqual(len(ids), len(set(ids)))

    def test_add_plain_strings(self):
        kb = Knowledge()
        n = kb.add("just one string document")
        self.assertEqual(n, 1)
        self.assertEqual(len(kb), 1)

    def test_sqlite_backend_roundtrip(self):
        kb = Knowledge(store="sqlite")
        kb.add(DOCS)
        self.assertIn("Refunds", kb.search("refund", k=1)[0].chunk.text)


if __name__ == "__main__":
    unittest.main()

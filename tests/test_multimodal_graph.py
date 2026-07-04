import unittest

from stratarag import Agent, EntityGraph, Knowledge, Pipeline
from stratarag.chunking import ModalityChunker, linearize_table
from stratarag.graph import HeuristicEntityExtractor, LLMEntityExtractor
from stratarag.llm.echo import EchoProvider
from stratarag.pipeline import Generate, GraphRetrieve
from stratarag.types import Document

RICH_DOC = """# Q3 Financials
Acme Corp revenue grew 40% driven by the Falcon platform.

| Product | Revenue | Growth |
|---------|---------|--------|
| Falcon  | $4M     | 62%    |
| Sparrow | $1M     | 8%     |

The Falcon roadmap is owned by Dana Weiss in the Berlin office.

```python
def revenue(units, price):
    return units * price
```

The growth model follows $$R_t = R_0 e^{kt}$$ as documented.

![Q3 revenue chart by product](charts/q3.png)
"""


class TestModalityChunker(unittest.TestCase):
    def setUp(self):
        self.chunks = ModalityChunker().chunk(Document(text=RICH_DOC))
        self.by_mod = {}
        for c in self.chunks:
            self.by_mod.setdefault(c.metadata["modality"], []).append(c)

    def test_all_modalities_detected(self):
        self.assertEqual(set(self.by_mod),
                         {"text", "table", "code", "equation", "image"})

    def test_table_is_linearized_and_kept_whole(self):
        table = self.by_mod["table"][0]
        self.assertIn("Product: Falcon", table.text)
        self.assertIn("Revenue: $4M", table.text)
        self.assertIn("|", table.metadata["raw"])  # original preserved

    def test_code_block_kept_verbatim_with_language(self):
        code = self.by_mod["code"][0]
        self.assertIn("def revenue", code.text)
        self.assertEqual(code.metadata["language"], "python")

    def test_equation_latex_preserved(self):
        eq = self.by_mod["equation"][0]
        self.assertIn("R_t = R_0", eq.text)
        self.assertEqual(eq.metadata["format"], "latex")

    def test_image_alt_text_and_captioner_hook(self):
        img = self.by_mod["image"][0]
        self.assertIn("revenue chart", img.text)
        self.assertEqual(img.metadata["src"], "charts/q3.png")
        chunks = ModalityChunker(
            captioner=lambda src, alt: f"caption of {src}"
        ).chunk(Document(text="![x](a.png)"))
        self.assertIn("caption of a.png", chunks[0].text)

    def test_broken_captioner_does_not_crash(self):
        def boom(src, alt):
            raise RuntimeError("vlm down")
        chunks = ModalityChunker(captioner=boom).chunk(Document(text="![x](a.png)"))
        self.assertTrue(chunks[0].text.startswith("Image"))

    def test_linearize_degenerate_table(self):
        self.assertEqual(linearize_table("| just one row |"), "| just one row |")


class TestEntityGraph(unittest.TestCase):
    def test_heuristic_extraction(self):
        ents = HeuristicEntityExtractor().extract(
            "Dana Weiss leads the Falcon roadmap at Acme Corp using GPU clusters.")
        joined = " ".join(ents)
        self.assertIn("Dana Weiss", joined)
        self.assertIn("Acme Corp", joined)
        self.assertIn("GPU", joined)

    def test_llm_extractor_with_fallback(self):
        ex = LLMEntityExtractor(EchoProvider(script=['["Falcon", "Acme"]']))
        self.assertEqual(ex.extract("whatever"), ["Falcon", "Acme"])
        ex2 = LLMEntityExtractor(EchoProvider(script=["garbage"]))
        self.assertTrue(any("Dana" in e for e in ex2.extract("Dana Weiss shipped it.")))

    def test_edges_and_neighbors(self):
        g = EntityGraph()
        from stratarag.types import Chunk
        g.index(Chunk(text="Falcon is built by Acme Corp.", id="c1"))
        g.index(Chunk(text="Acme Corp hired Dana Weiss.", id="c2"))
        nbrs = dict(g.neighbors("acme corp"))
        self.assertIn("falcon", nbrs)
        self.assertIn("dana weiss", nbrs)
        self.assertEqual(g.stats()["indexed_chunks"], 2)


class TestGraphSearch(unittest.TestCase):
    def setUp(self):
        self.kb = Knowledge(chunking="modality", graph=True)
        self.kb.add(RICH_DOC)

    def test_cross_modal_multihop(self):
        # "who owns the top product" needs table (revenue) + text (owner),
        # linked through the shared Falcon entity
        res = self.kb.graph_search("Who owns the Falcon roadmap and its revenue?", k=5)
        mods = {sc.chunk.metadata.get("modality") for sc in res}
        self.assertIn("table", mods)
        texts = " ".join(sc.chunk.text for sc in res)
        self.assertIn("Dana Weiss", texts)

    def test_graph_search_requires_graph(self):
        kb = Knowledge()
        kb.add("plain")
        with self.assertRaises(ValueError):
            kb.graph_search("q")

    def test_modality_filter(self):
        res = self.kb.graph_search("Falcon revenue", k=6)
        tables = self.kb.by_modality(res, "table")
        self.assertTrue(all(sc.chunk.metadata["modality"] == "table" for sc in tables))

    def test_incremental_ingest_dedupes(self):
        before = len(self.kb)
        added = self.kb.add(RICH_DOC)  # identical content
        self.assertEqual(added, 0)
        self.assertEqual(len(self.kb), before)
        self.assertEqual(self.kb.add("Genuinely new sentence here."), 1)

    def test_graph_retrieve_stage_and_agent_mode(self):
        pipe = Pipeline(GraphRetrieve(self.kb, k=4), Generate("echo"))
        ctx = pipe.run("Falcon revenue owner?")
        self.assertTrue(ctx.candidates)
        self.assertEqual(ctx.trace[0].stage, "graph_retrieve")
        agent = Agent(model="echo", knowledge=self.kb, retrieval="graph")
        res = agent.run("Dana Weiss Falcon?")
        self.assertTrue(res.sources)


def _importable(module: str) -> bool:
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


class TestNewAdapterHints(unittest.TestCase):
    def test_all_new_stores_raise_actionable_hints(self):
        from stratarag.errors import MissingDependencyError
        from stratarag.stores import resolve_store
        specs = {
            "pinecone:myindex": "pinecone",
            "weaviate:http://localhost:8080": "weaviate-client",
            "milvus:./milvus.db": "pymilvus",
            "elasticsearch:http://localhost:9200": "elasticsearch",
            "redis:redis://localhost:6379": "redis",
            "mongodb:mongodb://localhost#db.col#idx": "pymongo",
        }
        modules = {"pinecone": "pinecone", "weaviate-client": "weaviate",
                   "pymilvus": "pymilvus", "elasticsearch": "elasticsearch",
                   "redis": "redis", "pymongo": "pymongo"}
        for spec, pkg in specs.items():
            if _importable(modules[pkg]):
                continue  # installed here: hint path not applicable
            with self.assertRaises(MissingDependencyError, msg=spec) as cm:
                resolve_store(spec)
            self.assertIn(pkg, str(cm.exception))
            self.assertIn("pip install stratarag[", str(cm.exception))

    def test_all_new_embedders_raise_actionable_hints(self):
        from stratarag.errors import MissingDependencyError
        from stratarag.embeddings import resolve_embedder
        specs = {
            "openai:text-embedding-3-small": "openai",
            "azure-openai:my-deployment": "openai",
            "cohere:embed-english-v3.0": "cohere",
            "vertex:text-embedding-004": "google-cloud-aiplatform",
        }
        modules = {"openai": "openai", "cohere": "cohere",
                   "google-cloud-aiplatform": "vertexai"}
        for spec, pkg in specs.items():
            if _importable(modules[pkg]):
                continue
            with self.assertRaises(MissingDependencyError, msg=spec) as cm:
                resolve_embedder(spec)
            self.assertIn(pkg, str(cm.exception))


if __name__ == "__main__":
    unittest.main()

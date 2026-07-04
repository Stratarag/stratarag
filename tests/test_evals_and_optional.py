import unittest

from stratarag import Agent, EvalCase, EvalSuite, Knowledge, default_rag
from stratarag.errors import ConfigurationError, MissingDependencyError

DOCS = [
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Shipping takes 5 to 7 business days for standard orders.",
]


class TestEvalHarness(unittest.TestCase):
    def setUp(self):
        self.kb = Knowledge.from_texts(DOCS)
        self.cases = [
            EvalCase("what is the refund window?", expected_contains=["30 days"]),
            EvalCase("how long is shipping?", expected_contains=["5 to 7"]),
            EvalCase("what color is the CEO's car?", expected_contains=["purple"]),
        ]

    def test_runs_against_agent(self):
        report = EvalSuite(self.cases).run(Agent(model="echo", knowledge=self.kb))
        s = report.summary()
        self.assertEqual(s["cases"], 3)
        self.assertAlmostEqual(report.pass_rate, 2 / 3, places=3)
        failed = [r for r in report.results if not r.passed]
        self.assertIn("missing", failed[0].failure_reason)
        self.assertGreater(s["avg_faithfulness"], 0.5)

    def test_runs_against_pipeline_and_callable(self):
        pipe = default_rag(self.kb, model="echo", confidence_threshold=0.2)
        report = EvalSuite(self.cases[:2]).run(pipe)
        self.assertEqual(report.pass_rate, 1.0)
        report2 = EvalSuite([EvalCase("hi", expected_exact="hi back")]).run(
            lambda q: "hi back")
        self.assertEqual(report2.pass_rate, 1.0)

    def test_report_formats(self):
        report = EvalSuite(self.cases[:1]).run(Agent(model="echo", knowledge=self.kb))
        md = report.to_markdown()
        self.assertIn("Eval report", md)
        self.assertIn("| 1 |", md)
        import json
        data = json.loads(report.to_json())
        self.assertIn("summary", data)
        self.assertEqual(len(data["results"]), 1)


class TestOptionalDependencies(unittest.TestCase):
    """Missing optional deps must fail with actionable install hints,
    never bare ImportErrors."""

    def _assert_hint(self, fn, package):
        try:
            fn()
        except MissingDependencyError as e:
            self.assertIn("pip install stratarag[", str(e))
            self.assertIn(package, str(e))
        except ConfigurationError:
            pass  # also acceptable for cloud (no key)
        else:
            self.fail("expected MissingDependencyError")

    def test_chroma_qdrant_pgvector_cross_encoder_sbert(self):
        from stratarag.stores import resolve_store
        self._assert_hint(lambda: resolve_store("chroma"), "chromadb")
        self._assert_hint(lambda: resolve_store("qdrant:http://localhost:6333"),
                          "qdrant-client")
        self._assert_hint(lambda: resolve_store("pgvector:postgresql://x"),
                          "psycopg")
        from stratarag.pipeline.rerankers import resolve_reranker
        self._assert_hint(lambda: resolve_reranker("cross-encoder"),
                          "sentence-transformers")
        from stratarag.embeddings import resolve_embedder
        self._assert_hint(
            lambda: resolve_embedder("sentence-transformers:all-MiniLM-L6-v2"),
            "sentence-transformers")

    def test_anthropic_provider_hint(self):
        try:
            import anthropic  # noqa: F401
            self.skipTest("anthropic installed")
        except ImportError:
            pass
        from stratarag.llm import resolve_provider
        self._assert_hint(lambda: resolve_provider("claude-sonnet-4-6"), "anthropic")

    def test_cloud_requires_key(self):
        import os
        from stratarag.cloud import CloudStore
        os.environ.pop("STRATARAG_API_KEY", None)
        with self.assertRaises(ConfigurationError) as cm:
            CloudStore()
        self.assertIn("STRATARAG_API_KEY", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

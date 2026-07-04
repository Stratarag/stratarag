"""Tests for the ten RAG architectures: metadata filtering, the five advanced
stages, and every recipe builder."""
import unittest

import stratarag as sr
from stratarag.llm.echo import EchoProvider
from stratarag.pipeline.advanced import (Compress, CorrectiveRetrieve,
                                         IterativeRetrieve, MultiHopRetrieve,
                                         SelfRAGGenerate)
from stratarag.types import Document

DOCS = [
    Document(text="Refunds are accepted within 14 days of purchase.",
             metadata={"source": "policy.md", "year": 2026}),
    Document(text="Standard shipping takes 5 to 7 business days.",
             metadata={"source": "faq.md", "year": 2026}),
    Document(text="Refunds used to take 30 days before the 2024 update.",
             metadata={"source": "archive.md", "year": 2024}),
]


def make_kb(**kw):
    kb = sr.Knowledge(**kw)
    kb.add(list(DOCS))
    return kb


class TestMetadataFiltering(unittest.TestCase):
    def setUp(self):
        self.kb = make_kb()

    def test_where_filters_dense_keyword_hybrid(self):
        for fn in (self.kb.search, self.kb.keyword_search, self.kb.hybrid_search):
            res = fn("refund days", k=5, where={"source": "policy.md"})
            self.assertTrue(res)
            self.assertTrue(all(sc.chunk.metadata["source"] == "policy.md"
                                for sc in res), fn.__name__)

    def test_where_list_means_any_of(self):
        res = self.kb.hybrid_search("refund", k=5,
                                    where={"source": ["policy.md", "faq.md"]})
        self.assertTrue(all(sc.chunk.metadata["source"] != "archive.md"
                            for sc in res))

    def test_where_no_match_returns_empty(self):
        self.assertEqual(self.kb.search("refund", where={"source": "nope.md"}), [])

    def test_stage_and_runtime_where(self):
        pipe = sr.Pipeline(sr.HybridRetrieve(self.kb, k=5,
                                             where={"year": 2024}),
                           sr.Generate("echo"))
        ctx = pipe.run("refund days?")
        self.assertIn("30 days", ctx.answer)
        # runtime override via pipeline meta beats the stage default
        ctx2 = pipe.run("refund days?", where={"year": 2026})
        self.assertIn("14 days", ctx2.answer)


class TestCorrectiveRetrieve(unittest.TestCase):
    def test_good_retrieval_untouched(self):
        kb = make_kb()
        ctx = sr.Pipeline(CorrectiveRetrieve(kb, k=3)).run("refund window days?")
        self.assertGreaterEqual(ctx.meta["retrieval_relevance"], 0.34)
        self.assertFalse(ctx.trace[0].detail["corrected"])

    def test_weak_retrieval_triggers_fallback(self):
        kb = make_kb()
        stage = CorrectiveRetrieve(kb, k=3, min_relevance=0.99)  # force fallback
        ctx = sr.Pipeline(stage).run("refund days")
        self.assertTrue(ctx.candidates)  # still returns best available
        self.assertIn("corrected", ctx.trace[0].detail)


class TestMultiHop(unittest.TestCase):
    def test_heuristic_decomposition_covers_both_topics(self):
        kb = make_kb()
        stage = MultiHopRetrieve(kb, k_per_hop=1)
        ctx = sr.Pipeline(stage).run(
            "what is the refund window and how long does shipping take")
        self.assertGreaterEqual(len(ctx.meta["hops"]), 2)
        texts = " ".join(sc.chunk.text for sc in ctx.candidates).lower()
        self.assertIn("refunds", texts)
        self.assertIn("shipping", texts)

    def test_llm_decomposer_with_fallback(self):
        kb = make_kb()
        good = MultiHopRetrieve(kb, decomposer="llm",
                                model=EchoProvider(script=['["refund window", "shipping time"]']))
        ctx = sr.Pipeline(good).run("refunds and shipping?")
        self.assertEqual(ctx.meta["hops"], ["refund window", "shipping time"])
        bad = MultiHopRetrieve(kb, decomposer="llm",
                               model=EchoProvider(script=["not json"]))
        ctx2 = sr.Pipeline(bad).run("refunds and shipping?")
        self.assertTrue(ctx2.meta["hops"])  # heuristic fallback

    def test_requires_model_for_llm_mode(self):
        with self.assertRaises(ValueError):
            MultiHopRetrieve(make_kb(), decomposer="llm")


class TestIterative(unittest.TestCase):
    def test_bounded_loops_and_history(self):
        kb = make_kb()
        ctx = sr.Pipeline(IterativeRetrieve(kb, k=3, loops=3)).run("refund")
        self.assertLessEqual(len(ctx.meta["query_history"]), 3)
        self.assertTrue(ctx.candidates)
        self.assertEqual(ctx.meta["query_history"][0], "refund")


class TestCompress(unittest.TestCase):
    def test_keeps_relevant_sentence_drops_noise(self):
        kb = sr.Knowledge()
        kb.add("The office cafeteria serves lunch at noon. Refunds are "
               "accepted within 14 days. Parking permits renew annually. "
               "The gym opens at six.")
        pipe = sr.Pipeline(sr.Retrieve(kb, k=1), Compress(),
                           sr.Generate("echo"))
        ctx = pipe.run("what is the refund window?")
        kept = ctx.candidates[0].chunk.text
        self.assertIn("14 days", kept)
        self.assertNotIn("gym", kept)
        self.assertLess(ctx.meta["compression_ratio"], 1.0)
        self.assertTrue(ctx.candidates[0].chunk.metadata.get("compressed"))

    def test_single_sentence_chunks_pass_through(self):
        kb = make_kb()
        ctx = sr.Pipeline(sr.Retrieve(kb, k=2), Compress()).run("refund days")
        self.assertTrue(ctx.candidates)


class TestSelfRAG(unittest.TestCase):
    def test_pass_on_first_round(self):
        kb = make_kb()
        pipe = sr.recipes.self_rag(kb, model="echo", max_rounds=3)
        ctx = pipe.run("what is the refund window?")
        self.assertEqual(ctx.meta["self_rag_rounds"], 1)
        self.assertIn("14 days", ctx.answer)

    def test_fail_verdict_triggers_second_round(self):
        kb = make_kb()
        provider = EchoProvider(script=[
            "totally ungrounded nonsense",   # round 1 draft
            "FAIL",                          # critic verdict
            "Refunds are accepted within 14 days of purchase.",  # round 2
            "PASS",
        ])
        stage = SelfRAGGenerate(provider, knowledge=kb, max_rounds=2)
        ctx = sr.Pipeline(sr.HybridRetrieve(kb, k=3), stage).run("refund window?")
        self.assertEqual(ctx.meta["self_rag_rounds"], 2)
        self.assertIn("14 days", ctx.answer)

    def test_heuristic_critic_when_llm_disabled(self):
        kb = make_kb()
        stage = SelfRAGGenerate("echo", knowledge=kb, max_rounds=2,
                                llm_critic=False)
        ctx = sr.Pipeline(sr.HybridRetrieve(kb, k=3), stage).run("refund window?")
        self.assertIn("14 days", ctx.answer)


class TestAllRecipes(unittest.TestCase):
    def test_every_recipe_answers_the_refund_question(self):
        kb = make_kb()
        gkb = sr.Knowledge(graph=True)
        gkb.add(list(DOCS))
        for name in sr.recipes.ALL:
            kwargs = {}
            if name == "metadata":
                kwargs["where"] = {"source": "policy.md"}
            target_kb = gkb if name == "graph" else kb
            pipe = sr.recipes.build(name, target_kb, "echo", **kwargs)
            ctx = pipe.run("what is the refund window?")
            self.assertIn("14 days", ctx.answer, f"recipe {name}")

    def test_metadata_recipe_excludes_other_sources(self):
        kb = make_kb()
        pipe = sr.recipes.metadata_rag(kb, "echo", where={"year": 2024})
        ctx = pipe.run("refund days?")
        self.assertIn("30 days", ctx.answer)

    def test_unknown_recipe_raises(self):
        with self.assertRaises(ValueError):
            sr.recipes.build("nope", make_kb(), "echo")


if __name__ == "__main__":
    unittest.main()

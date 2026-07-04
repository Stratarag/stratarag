import unittest

from stratarag import Knowledge, Memory, Pipeline, default_rag, tool
from stratarag.errors import ToolError
from stratarag.llm.echo import EchoProvider
from stratarag.pipeline import (ConfidenceGate, ContextFilter, Generate,
                            HybridRetrieve, MemoryRead, QueryRewrite, Rerank,
                            Retrieve, grounding_score)
from stratarag.pipeline.rewrite import HeuristicRewriter, LLMRewriter
from stratarag.pipeline.rerankers import LexicalOverlapReranker
from stratarag.tools import ToolRegistry
from stratarag.types import Chunk, ScoredChunk

DOCS = [
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Shipping takes 5 to 7 business days for standard orders.",
    "Premium support is available around the clock for enterprise plans.",
]


class TestTools(unittest.TestCase):
    def test_schema_from_hints_and_defaults(self):
        @tool
        def add(a: int, b: int = 2) -> int:
            """Add two integers."""
            return a + b
        spec = add.spec()
        self.assertEqual(spec["name"], "add")
        self.assertEqual(spec["description"], "Add two integers.")
        self.assertEqual(spec["parameters"]["properties"]["a"]["type"], "integer")
        self.assertEqual(spec["parameters"]["required"], ["a"])
        self.assertEqual(add.run({"a": 3}), "5")

    def test_decorator_with_args_and_json_result(self):
        @tool(name="lookup", description="Find a user.")
        def fn(user: str) -> dict:
            return {"user": user, "ok": True}
        self.assertEqual(fn.name, "lookup")
        self.assertIn('"ok": true', fn.run({"user": "amy"}))

    def test_bad_args_and_failures_raise_tool_error(self):
        @tool
        def strict(x: int) -> int:
            return x
        with self.assertRaises(ToolError):
            strict.run({"nope": 1})

        @tool
        def boom() -> str:
            raise RuntimeError("kaboom")
        with self.assertRaises(ToolError):
            boom.run({})

    def test_registry(self):
        reg = ToolRegistry([lambda: "hi"])
        self.assertEqual(len(reg), 1)
        with self.assertRaises(ToolError):
            reg.get("missing")

    def test_async_tool_via_sync_run(self):
        @tool
        async def afetch(x: int) -> int:
            """double"""
            return x * 2
        self.assertEqual(afetch.run({"x": 4}), "8")


class TestRewriters(unittest.TestCase):
    def test_heuristic_strips_filler(self):
        r = HeuristicRewriter()
        self.assertEqual(r.rewrite("Hey can you please tell me the refund policy?"),
                         "the refund policy")
        self.assertEqual(r.rewrite("   "), "   ")  # never returns empty

    def test_llm_rewriter_with_fallback(self):
        r = LLMRewriter(EchoProvider(script=["refund policy window"]))
        self.assertEqual(r.rewrite("what about refunds?"), "refund policy window")
        r2 = LLMRewriter(EchoProvider(script=[""]))  # empty -> fallback
        self.assertTrue(r2.rewrite("please tell me about refunds"))


class TestReranker(unittest.TestCase):
    def test_lexical_moves_relevant_up(self):
        cands = [
            ScoredChunk(Chunk(text="bananas are yellow fruit"), 0.9),
            ScoredChunk(Chunk(text="the refund window is thirty days"), 0.5),
        ]
        out = LexicalOverlapReranker(blend=0.9).rerank("refund window days", cands)
        self.assertIn("refund", out[0].chunk.text)


class TestGroundingScore(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(grounding_score("", ["ctx"]), 0.0)
        self.assertAlmostEqual(
            grounding_score("refund window", ["the refund window is 30 days"]), 1.0)
        self.assertLess(grounding_score("unicorns fly high", ["refund policy"]), 0.5)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.kb = Knowledge.from_texts(DOCS)

    def test_default_rag_answers_and_traces(self):
        pipe = default_rag(self.kb, model="echo", confidence_threshold=0.3)
        ctx = pipe.run("please tell me what is the refund window?")
        self.assertIn("Refunds", ctx.answer)
        self.assertFalse(ctx.gated)
        stages = [t.stage for t in ctx.trace]
        self.assertEqual(stages, ["query_rewrite", "hybrid_retrieve", "rerank",
                                  "context_filter", "generate", "confidence_gate"])
        self.assertTrue(all(t.elapsed_ms >= 0 for t in ctx.trace))

    def test_gate_blocks_ungrounded_answer(self):
        pipe = Pipeline(
            Retrieve(self.kb, k=2),
            Generate(EchoProvider(script=["unicorns invented the moon rocket"]),
                     grounded=True),
            ConfidenceGate(threshold=0.6),
        )
        ctx = pipe.run("refund window?")
        self.assertTrue(ctx.gated)
        self.assertIn("not confident", ctx.answer)
        self.assertIn("unicorns", ctx.meta["ungated_answer"])

    def test_memory_read_stage_injects_facts(self):
        mem = Memory(semantic=True)
        mem.remember("User prefers concise answers", user_id="u7")
        pipe = Pipeline(
            Retrieve(self.kb, k=2),
            MemoryRead(mem),
            Generate("echo", grounded=True),
        )
        ctx = pipe.run("refund window", user_id="u7")
        self.assertTrue(any("concise" in f.content for f in ctx.memory.facts))
        self.assertIn("Known about this user", ctx.messages[0].content)

    def test_context_filter_dedupes_and_caps(self):
        dup = ScoredChunk(Chunk(text="same text"), 0.9)
        ctx_stage = ContextFilter(max_chunks=2)
        from stratarag.pipeline.base import Context
        ctx = Context(query="q")
        ctx.candidates = [dup, ScoredChunk(Chunk(text="same text"), 0.8),
                          ScoredChunk(Chunk(text="other"), 0.7),
                          ScoredChunk(Chunk(text="third"), 0.6)]
        out = ctx_stage.run(ctx)
        texts = [sc.chunk.text for sc in out.candidates]
        self.assertEqual(texts, ["same text", "other"])

    def test_async_pipeline(self):
        import asyncio
        pipe = default_rag(self.kb, model="echo", confidence_threshold=0.2)
        ctx = asyncio.run(pipe.arun("shipping time?"))
        self.assertIn("Shipping", ctx.answer)

    def test_llm_rewrite_stage_in_pipeline(self):
        pipe = Pipeline(
            QueryRewrite(rewriter="llm",
                         model=EchoProvider(script=["shipping duration days"])),
            HybridRetrieve(self.kb, k=2),
            Rerank(top_n=1),
            # floor disabled: this test verifies rewrite plumbing; the user
            # phrasing ("stuff arrives lol") shares no tokens with the doc
            Generate(EchoProvider(relevance_floor=0.0)),
        )
        ctx = pipe.run("how long till my stuff arrives lol")
        self.assertEqual(ctx.rewritten_query, "shipping duration days")
        self.assertIn("Shipping", ctx.answer)


if __name__ == "__main__":
    unittest.main()

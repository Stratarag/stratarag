import asyncio
import unittest

from stratarag import Agent, Knowledge, Memory, tool
from stratarag.llm.echo import EchoProvider
from stratarag.types import AgentResult

DOCS = [
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Shipping takes 5 to 7 business days for standard orders.",
]


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id} shipped yesterday."


@tool
async def aget_stock(sku: str) -> str:
    """Async stock check."""
    await asyncio.sleep(0)
    return f"{sku}: 12 units in stock"


class TestAgentBasics(unittest.TestCase):
    def test_plain_run_returns_result(self):
        agent = Agent(model=EchoProvider(script=["hello there"]))
        res = agent.run("hi")
        self.assertIsInstance(res, AgentResult)
        self.assertEqual(res.output, "hello there")
        self.assertEqual(str(res), "hello there")

    def test_grounded_run_uses_knowledge(self):
        agent = Agent(model="echo", knowledge=Knowledge.from_texts(DOCS))
        res = agent.run("what is the refund window?")
        self.assertIn("Refunds", res.output)
        self.assertTrue(res.sources)
        self.assertGreater(res.confidence, 0.5)

    def test_confidence_gate_on_agent(self):
        agent = Agent(model=EchoProvider(script=["moon cheese conspiracy"]),
                      knowledge=Knowledge.from_texts(DOCS),
                      confidence_threshold=0.6)
        res = agent.run("refund window?")
        self.assertTrue(res.gated)
        self.assertIn("not confident", res.output)

    def test_unknown_model_spec_raises(self):
        from stratarag.errors import ConfigurationError
        with self.assertRaises(ConfigurationError):
            Agent(model="gpt-nonexistent-99")


class TestToolLoop(unittest.TestCase):
    def test_single_tool_round(self):
        provider = EchoProvider(script=[
            {"tool": "get_order_status", "args": {"order_id": "A1"}},
            "Your order A1 shipped yesterday.",
        ])
        agent = Agent(model=provider, tools=[get_order_status])
        res = agent.run("where is my order A1?")
        self.assertIn("shipped", res.output)
        tool_events = [t for t in res.trace if t.stage == "tool:get_order_status"]
        self.assertEqual(len(tool_events), 1)
        self.assertIn("A1", tool_events[0].detail["result"])

    def test_unknown_tool_reported_not_crashed(self):
        provider = EchoProvider(script=[
            {"tool": "nope", "args": {}},
            "recovered gracefully",
        ])
        agent = Agent(model=provider, tools=[get_order_status])
        res = agent.run("do something")
        self.assertEqual(res.output, "recovered gracefully")
        tool_events = [t for t in res.trace if t.stage.startswith("tool:")]
        self.assertIn("tool error", tool_events[0].detail["result"])

    def test_max_rounds_cutoff(self):
        provider = EchoProvider(
            script=[{"tool": "get_order_status", "args": {"order_id": "X"}}] * 10)
        agent = Agent(model=provider, tools=[get_order_status], max_tool_rounds=2)
        res = agent.run("loop forever")
        self.assertIn("tool-call limit", res.output)


class TestMemoryIntegration(unittest.TestCase):
    def test_agent_learns_across_turns(self):
        mem = Memory(semantic=True, episodic=True)
        agent = Agent(model="echo", memory=mem,
                      knowledge=Knowledge.from_texts(DOCS))
        agent.run("My name is Dev and I prefer email updates. refund info?",
                  user_id="u1")
        res2 = agent.run("how should you contact me about refunds?", user_id="u1")
        self.assertTrue(any("email" in f.content
                            for f in res2.memory_used.get("semantic", [])))
        writes = [t for t in res2.trace if t.stage == "memory_write"]
        self.assertEqual(len(writes), 1)


class TestAsyncAndStreaming(unittest.TestCase):
    def test_arun_with_async_tool(self):
        provider = EchoProvider(script=[
            {"tool": "aget_stock", "args": {"sku": "SKU9"}},
            "SKU9 has 12 units.",
        ])
        agent = Agent(model=provider, tools=[aget_stock])
        res = asyncio.run(agent.arun("stock for SKU9?"))
        self.assertIn("12 units", res.output)

    def test_stream_yields_tokens_then_result(self):
        agent = Agent(model="echo", knowledge=Knowledge.from_texts(DOCS))
        events = list(agent.stream("refund window?"))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[-1], "result")
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertEqual(tokens, events[-1]["result"].output)
        self.assertIn("Refunds", tokens)

    def test_stream_surfaces_tool_events(self):
        provider = EchoProvider(script=[
            {"tool": "get_order_status", "args": {"order_id": "Z3"}},
            "Order Z3 shipped yesterday.",
        ])
        agent = Agent(model=provider, tools=[get_order_status])
        events = list(agent.stream("track Z3"))
        self.assertEqual(events[0]["type"], "tool")
        self.assertIn("Z3", events[0]["result"])

    def test_astream(self):
        async def collect():
            agent = Agent(model="echo", knowledge=Knowledge.from_texts(DOCS))
            return [e async for e in agent.astream("shipping time?")]
        events = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "result")
        self.assertIn("Shipping", events[-1]["result"].output)


if __name__ == "__main__":
    unittest.main()

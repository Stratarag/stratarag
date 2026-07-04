"""Evaluate an agent or pipeline before you ship it."""
import stratarag as mn

kb = mn.Knowledge.from_texts([
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Standard shipping takes 5 to 7 business days.",
])
agent = mn.Agent(model="echo", knowledge=kb, confidence_threshold=0.3)

suite = mn.EvalSuite([
    mn.EvalCase("what is the refund window?", expected_contains=["30 days"]),
    mn.EvalCase("how long is shipping?", expected_contains=["5 to 7"]),
    mn.EvalCase("who is the CEO?", expected_contains=["Jane"]),  # should fail/gate
])
report = suite.run(agent)
print(report.to_markdown())

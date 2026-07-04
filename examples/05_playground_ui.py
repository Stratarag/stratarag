"""Launch the local playground UI at http://localhost:7327 — chat with the
agent and watch memory, sources, confidence, and the stage trace live."""
import stratarag as mn
from stratarag.dashboard import serve

kb = mn.Knowledge.from_texts([
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Standard shipping takes 5 to 7 business days.",
    "Premium support is available 24/7 on enterprise plans.",
])
agent = mn.Agent(model="echo", knowledge=kb,
                 memory=mn.Memory(semantic=True, episodic=True),
                 confidence_threshold=0.35)
serve(agent, port=7327)

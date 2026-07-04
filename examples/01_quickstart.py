"""Five-minute quickstart: knowledge + memory + agent. Runs fully offline
with the deterministic `echo` model; swap model="claude-sonnet-4-6" (and
`pip install stratarag[anthropic]`, set ANTHROPIC_API_KEY) for a real LLM."""
import stratarag as mn

kb = mn.Knowledge.from_texts([
    "Refunds are accepted within 30 days of purchase with a receipt.",
    "Standard shipping takes 5 to 7 business days.",
    "Premium support is available 24/7 on enterprise plans.",
])

agent = mn.Agent(
    model="echo",
    knowledge=kb,
    memory=mn.Memory(semantic=True, episodic=True),
    confidence_threshold=0.4,
)

print(agent.run("Hi, my name is Priya. What's the refund window?", user_id="u1"))
print(agent.run("How long does shipping take?", user_id="u1"))
print("Learned:", [f.content for f in agent.memory.semantic.all("u1")])

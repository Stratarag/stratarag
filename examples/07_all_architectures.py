"""All ten RAG architectures, one corpus, one line each."""
import stratarag as sr
from stratarag.types import Document

docs = [
    Document(text="Refunds are accepted within 14 days of purchase.",
             metadata={"source": "policy.md", "year": 2026}),
    Document(text="Standard shipping takes 5 to 7 business days.",
             metadata={"source": "faq.md", "year": 2026}),
    Document(text="Refunds used to take 30 days before the 2024 update.",
             metadata={"source": "archive.md", "year": 2024}),
]
kb = sr.Knowledge(graph=True); kb.add(docs)

Q = "what is the refund window?"
print("1 simple     :", sr.recipes.simple_rag(kb, "echo").run(Q).answer[:60])
print("2 hybrid     :", sr.recipes.hybrid_rag(kb, "echo").run(Q).answer[:60])
print("3 corrective :", sr.recipes.corrective_rag(kb, "echo").run(Q).answer[:60])
print("4 self-rag   :", sr.recipes.self_rag(kb, "echo").run(Q).answer[:60])
print("5 graph      :", sr.recipes.graph_rag(kb, "echo").run(Q).answer[:60])
print("6 agentic    :", sr.Agent(model="echo", knowledge=kb).run(Q).output[:60])
print("7 multi-hop  :", sr.recipes.multi_hop_rag(kb, "echo").run(
    "refund window and shipping time?").answer[:60])
print("8 iterative  :", sr.recipes.iterative_rag(kb, "echo").run(Q).answer[:60])
print("9 compression:", sr.recipes.compression_rag(kb, "echo").run(Q).answer[:60])
print("10 metadata  :", sr.recipes.metadata_rag(
    kb, "echo", where={"year": 2024}).run("refund days?").answer[:60])

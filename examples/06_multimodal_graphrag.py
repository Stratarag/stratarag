"""Multimodal GraphRAG (RAG-Anything style): typed parsing of tables, code,
equations and images + entity-graph retrieval that connects evidence across
modalities."""
import stratarag as mn

DOC = """# Q3 Financials
Acme Corp revenue grew 40% driven by the Falcon platform.

| Product | Revenue | Growth |
|---------|---------|--------|
| Falcon  | $4M     | 62%    |
| Sparrow | $1M     | 8%     |

The Falcon roadmap is owned by Dana Weiss in the Berlin office.

The growth model follows $$R_t = R_0 e^{kt}$$ as documented.

![Q3 revenue chart by product](charts/q3.png)
"""

kb = mn.Knowledge(chunking="modality", graph=True)
# optional hooks: chunking="modality" accepts captioner=lambda src, alt: "..." (e.g. a VLM call)
# and graph="llm", graph_model="claude-sonnet-4-6" for LLM entity extraction
kb.add(DOC)
print("graph:", kb.graph.stats())

res = kb.graph_search("Who owns the product driving Acme's revenue?", k=4)
for sc in res:
    print(f"  [{sc.chunk.metadata['modality']:^8}] {sc.score:.2f}  {sc.chunk.text[:70]}")

print("\ntables only:", [sc.chunk.text[:50] for sc in kb.by_modality(res, "table")])

agent = mn.Agent(model="echo", knowledge=kb, retrieval="graph")
print("\nagent:", agent.run("Falcon revenue and owner?"))

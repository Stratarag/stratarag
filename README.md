<div align="center">

# 🧠 StrataRAG

### Agents that remember. RAG in every shape. Zero required dependencies.

<p>
  <img src="https://img.shields.io/badge/python-3.9%2B-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/license-Apache--2.0-00d9ff?style=for-the-badge&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/core%20dependencies-0-ff6b6b?style=for-the-badge&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/tests-135%20passing-2ea44f?style=for-the-badge&labelColor=1a1a2e">
</p>

<p>
  <a href="#-quick-start"><b>Quick Start</b></a> ·
  <a href="#%EF%B8%8F-the-ten-rag-architectures"><b>10 RAG Architectures</b></a> ·
  <a href="#-memory-types"><b>Memory</b></a> ·
  <a href="#-multi-agent-orchestration"><b>Multi-Agent</b></a> ·
  <a href="#%EF%B8%8F-backend-matrix"><b>Backends</b></a> ·
  <a href="#-playground-ui"><b>Playground</b></a>
</p>

</div>

---

## 🌟 Why StrataRAG

Modern AI applications need three things existing frameworks bolt on as afterthoughts: **retrieval in the right shape** (there is no one-size RAG), **memory that persists and learns**, and **orchestration across agents**. StrataRAG makes all three first-class — in a core that runs on the Python standard library alone, so your tests and CI never need a network, an API key, or a GPU.

- ✅ **Five primitives** — `Knowledge`, `Memory`, `Tool`, `Agent`, `Pipeline`. Everything composes from them.
- ✅ **All ten classic RAG architectures** as one-line recipes — and every recipe is an open `Pipeline` you can rearrange or subclass.
- ✅ **Typed memory** — semantic, episodic, procedural, prospective, working — read/written automatically on every agent turn.
- ✅ **Multimodal GraphRAG** — tables, equations, code, and images parsed as typed chunks; entity graph links evidence across modalities.
- ✅ **Multi-agent orchestration** — sequential workflows, hub-and-spoke routing, collaborative teams with critique rounds.
- ✅ **10 vector stores · 6 embedding providers** behind two interfaces — migrate with a string change.
- ✅ **Production honesty** — confidence gating, per-stage tracing, eval harness, incremental ingestion, actionable errors.

## 🚀 Quick Start

```bash
pip install stratarag          # core: nothing else needed
pip install stratarag[all]     # every optional backend
```

```python
import stratarag as sr

kb = sr.Knowledge.from_docs("docs/", chunking="markdown", graph=True)

agent = sr.Agent(
    model="claude-sonnet-4-6",            # or "echo" for offline dev
    knowledge=kb,
    memory=sr.Memory(semantic=True, episodic=True, backend="sqlite:./mem.db"),
    confidence_threshold=0.35,            # ungrounded answers get gated
)

result = agent.run("What changed in the refund policy?", user_id="u42")
print(result.output, result.confidence, result.sources)
```

Run `python examples/05_playground_ui.py` → http://localhost:7327 for a zero-dependency local playground: chat, recalled memories, sources, confidence gauge, and the stage-by-stage trace.

## 🏗️ The Ten RAG Architectures

Every pattern from the canonical taxonomy, each a one-liner returning an open `Pipeline`:

| # | Architecture | Recipe | What it adds |
|---|---|---|---|
| 1 | Simple RAG | `sr.recipes.simple_rag(kb, model)` | top-k retrieve → grounded generate |
| 2 | Hybrid RAG | `sr.recipes.hybrid_rag(kb, model)` | BM25 + dense fusion (RRF) → rerank |
| 3 | Corrective RAG (CRAG) | `sr.recipes.corrective_rag(kb, model)` | relevance-scored retrieval, fallback search when weak |
| 4 | Self-RAG | `sr.recipes.self_rag(kb, model)` | draft → self-critique → re-retrieve → regenerate |
| 5 | Graph RAG | `sr.recipes.graph_rag(kb, model)` | entity-graph expansion, multi-hop, cross-modal |
| 6 | Agentic RAG | `sr.Agent(model, tools=[...], knowledge=kb)` | plans, calls tools, iterates |
| 7 | Multi-Hop RAG | `sr.recipes.multi_hop_rag(kb, model)` | sub-question decomposition, retrieve per hop |
| 8 | Iterative RAG (IRAG) | `sr.recipes.iterative_rag(kb, model)` | bounded query-refinement loops |
| 9 | Contextual Compression | `sr.recipes.compression_rag(kb, model)` | keep only query-relevant sentences |
| 10 | Metadata-Driven RAG | `sr.recipes.metadata_rag(kb, model, where={...})` | hard filters by tag/source/date |

Or compose your own from the stage library — `QueryRewrite`, `HybridRetrieve`, `GraphRetrieve`, `CorrectiveRetrieve`, `MultiHopRetrieve`, `IterativeRetrieve`, `Rerank`, `Compress`, `ContextFilter`, `MemoryRead`, `Generate`, `SelfRAGGenerate`, `ConfidenceGate` — every stage is a plain class with `run(ctx) -> ctx`.

Metadata filtering works everywhere: `kb.search(q, where={"source": "policy.md", "year": 2026})`, per-stage defaults, or per-run overrides (`pipe.run(q, where={...})`). List values mean *any of*.

## 🧠 Memory Types

```python
memory = sr.Memory(
    semantic=True,     # durable facts — "user prefers metric units"
    episodic=True,     # past runs & outcomes — learn from failures
    procedural=True,   # registered, reusable skills
    prospective=True,  # future intents that fire on time or keyword triggers
    working=True,      # rolling conversation buffer with word budget
    backend="sqlite:./mem.db",                     # or any VectorStore
    extractor="llm", model="claude-sonnet-4-6",    # LLM fact extraction
)
```

`agent.run()` calls `memory.read()` before answering and `memory.write_turn()` after. Knowledge and Memory never share a store — user context cannot pollute your source of truth.

## 🖼️ Multimodal GraphRAG

`chunking="modality"` parses **tables** (kept whole + linearized row-by-row), **LaTeX equations**, **fenced code**, and **images** (alt text + optional VLM `captioner=` hook) into typed chunks. `graph=True` builds an entity graph across all of them, so a table row and a paragraph about the same entity are graph-connected. Ingestion is **incremental** — re-adding a document skips unchanged chunks.

## 🤝 Multi-Agent Orchestration

The three enterprise deployment archetypes, with agents, pipelines, or plain callables as units:

```python
from stratarag.orchestration import Workflow, Orchestrator, Team

# Sequential — deterministic chains (AP auditing, tax filing, underwriting)
Workflow([("ingest", extractor), ("reconcile", agent), ("comply", checker)]).run(task)

# Hub-and-spoke — a router dispatches to specialists (onboarding, maintenance)
Orchestrator({"billing": ("refunds invoices", billing_agent),
              "it": ("laptops access", it_agent)}, router=model).run(task)

# Collaborative — contribute, optionally critique each other, synthesize
Team({"siem": siem_agent, "forensics": forensics_agent},
     synthesizer=model, critique=True).run(task)
```

Every run returns an `OrchestrationResult` with a full step-by-step trace and shared state.

## 🗄️ Backend Matrix

| Vector stores | Embeddings | LLM providers |
|---|---|---|
| In-memory, SQLite *(built in)* | Hashing *(built in, offline)* | Echo *(built in, deterministic)* |
| Chroma · Qdrant · pgvector | Sentence-Transformers | Anthropic (`claude-*`) |
| Pinecone · Weaviate · Milvus | OpenAI · Azure OpenAI | any callable `(messages, tools) -> str` |
| Elasticsearch · Redis · MongoDB Atlas | Cohere · Vertex AI | custom `LLMProvider` subclass |

One `VectorStore` interface, one `Embedder` interface, one `LLMProvider` interface. Specs are strings: `store="qdrant:http://localhost:6333"`, `embedder="openai:text-embedding-3-small"`. Missing optional packages raise `MissingDependencyError` with the exact `pip install stratarag[extra]` to run.

## 📊 Evals Before You Ship

```python
report = sr.EvalSuite([
    sr.EvalCase("refund window?", expected_contains=["14 days"]),
]).run(agent)                       # Agent, Pipeline, or any callable
print(report.to_markdown())         # pass rate, faithfulness, relevance, latency, gating
```

## 🎨 Playground UI

A zero-dependency local dashboard (`stratarag.dashboard.serve(agent)`): chat panel, recall strip showing what the agent remembered, source cards, a confidence gauge with gated-answer styling, teach-it-a-fact input, and per-stage timing trace.

## 🧭 Design Principles

1. **Tiny primitive set** — five nouns; the ten architectures are arrangements, not new machinery.
2. **Your code stays normal Python** — tools are functions, stages are classes, debugging is `print()`.
3. **Layered with escape hatches** — recipe → rearranged stages → subclassed stage. Moving down never requires a rewrite.
4. **Offline-first** — the echo model, hashing embedder, and local stores mean CI needs no network and no keys.
5. **Honest failures** — every boundary fails loudly with the fix in the message.

## 🧪 Development & Testing

```bash
python -m unittest discover -s tests    # 135 tests, no network, < 1s
```

The suite covers chunking edge cases, store contracts, all ten architecture recipes (behavioral assertions, not just smoke), memory types, tool failures, gating, streaming, async, orchestration archetypes, multimodal parsing, graph traversal, missing-dependency paths — plus regression tests for every bug found by dogfooding.

## 📄 License

Apache-2.0. See `CHANGELOG.md` for version history.

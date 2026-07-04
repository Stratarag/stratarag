# StrataRAG Launch Announcement — three versions

Pick per channel. All claims below are true of the shipped code; nothing is
aspirational. Fill in the links once the repo/PyPI pages are live.

---

## 1) Short version — X/Twitter, Mastodon, Bluesky (~280 chars)

> Shipping StrataRAG 🧠 — an open-source agent & RAG framework with ZERO required dependencies.
>
> All 10 classic RAG architectures as one-liners. Typed agent memory. Multi-agent orchestration. 10 vector DBs, 6 embedding providers — one interface each.
>
> pip install stratarag

---

## 2) Medium version — LinkedIn / Reddit (r/LocalLLaMA, r/MachineLearning)

**StrataRAG: agents that remember, RAG in every shape — with zero required dependencies**

I just open-sourced StrataRAG, a Python framework built around one frustration: every RAG/agent framework treats *memory* as an afterthought and locks you into one retrieval pattern.

What's different:

**Retrieval in every shape.** All ten classic RAG architectures — Simple, Hybrid, Corrective (CRAG), Self-RAG, Graph, Agentic, Multi-Hop, Iterative, Contextual Compression, Metadata-Driven — each a one-line recipe that returns an open pipeline. When the recipe stops fitting, you rearrange stages or subclass one; you never rewrite.

**Memory as a first-class primitive.** Semantic facts, episodic outcomes, procedural skills, prospective intents, and working memory — typed, per-user, persisted, and read/written automatically on every agent turn. Kept strictly separate from your knowledge base, so user context never pollutes the source of truth.

**Zero required dependencies.** The entire core runs on the Python standard library — including a deterministic offline model and a hashing embedder — so your tests and CI need no network, no API keys, no GPU. Real backends (Pinecone, Qdrant, Weaviate, Milvus, pgvector, Elasticsearch, Redis, MongoDB Atlas, Chroma; OpenAI/Azure/Cohere/Vertex/sentence-transformers embeddings; Anthropic models) are each one string away, behind one interface.

Also in the box: multimodal parsing (tables, equations, code, images) with entity-graph retrieval across modalities, multi-agent orchestration (sequential / hub-and-spoke / collaborative-with-critique), confidence gating, per-stage tracing, an eval harness, and a zero-dependency local playground UI.

137 tests, all offline, under a second. Apache-2.0.

`pip install stratarag` — repo: <link>. Feedback and PRs very welcome; CONTRIBUTING.md has recipes for adding adapters in ~50 lines.

---

## 3) Long version — Hacker News "Show HN" / blog post

**Show HN: StrataRAG — a zero-dependency agent & RAG framework where memory is the point**

Hi HN! I built StrataRAG after noticing two things about the current agent-framework landscape.

First, "memory" in most frameworks means stuffing chat history into the prompt. But the research taxonomy (CoALA and friends) distinguishes semantic memory (durable facts), episodic memory (what happened and whether it worked), procedural memory (reusable skills), prospective memory (future intents), and working memory (the current window). StrataRAG implements all of them as typed, per-user, persistable modules that the agent reads before answering and writes after — with fact extraction, deduplication, and a hard architectural wall between *memory* (what the agent learned) and *knowledge* (your source of truth), so one can never contaminate the other.

Second, there is no one-size RAG. Naive top-k works for FAQs and falls apart on everything else, and each failure mode has a known fix: hybrid search for keyword-heavy corpora, corrective retrieval with fallback for weak recall, self-critique loops for hallucination control, query decomposition for multi-hop questions, entity graphs for evidence scattered across documents, sentence-level compression for token budgets, metadata filters for compliance. StrataRAG ships all ten canonical architectures as recipes — but each recipe just returns a plain `Pipeline` of plain stage classes, so the one-liner and the fully customized version are the same code at different zoom levels.

The design constraint I'm proudest of: **the core has zero dependencies.** Not "lightweight" — literally none. It includes a deterministic offline model (which ranks provided sources by relevance and refuses when nothing relevant was retrieved, like an instructed real model), a hashing embedder, and in-memory/SQLite vector stores. That means the 137-test suite runs in under a second with no network and no keys, and a CI job proves it on every commit by installing nothing first. Real backends — nine vector databases, four embedding APIs, Anthropic models — are lazy adapters behind single interfaces; a missing client fails with the exact `pip install stratarag[extra]` to run.

Honest limitations, because they matter: the built-in embedder is lexical (feature hashing + light stemming), great for dev/CI and small corpora, not a semantic model — production wants sentence-transformers or an API embedder, one string away. The network-backed adapters are written to each vendor's current client API and contract-shaped, but I could only integration-test the local backends; there's a 30-second smoke-test pattern in the docs for verifying yours. And the confidence gate is a cheap lexical proxy (geometric mean of faithfulness and query-context relevance), not an oracle — it caught real bugs in my own dogfooding, but tune the threshold for your domain.

Other things in the box: multimodal parsing that keeps tables whole (and linearizes them for embedding recall), preserves LaTeX and code, and hooks a VLM captioner for images; an entity graph built at ingest that connects a table row and a paragraph about the same entity for cross-modal multi-hop retrieval; multi-agent orchestration in the three enterprise archetypes (sequential workflow, hub-and-spoke router, collaborative team with a critique round); an eval harness that scores pass rate, faithfulness, relevance, and latency against any agent, pipeline, or callable; and a local playground UI served by the stdlib http server.

Apache-2.0. `pip install stratarag`. Repo: <link>. I'd love feedback on the memory API design especially — that's the part I think the ecosystem hasn't settled yet.

---

## Posting notes

- HN: post the long version's first two paragraphs as the text, lead comment
  with the limitations paragraph — HN rewards stated limitations.
- Fill every `<link>` before posting; verify `pip install stratarag` works
  from a clean machine first.
- Do not add star/download badges to the README until the numbers exist.

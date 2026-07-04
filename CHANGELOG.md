# Changelog

## 0.5.0
- The ten classic RAG architectures complete: new stages CorrectiveRetrieve
  (CRAG), MultiHopRetrieve, IterativeRetrieve (IRAG), Compress (contextual
  compression), SelfRAGGenerate (Self-RAG) — plus `stratarag.recipes` with a
  one-line builder per architecture and string-addressable `recipes.build()`
- Metadata-driven retrieval: `where=` equality filters (list = any-of) across
  dense / keyword / hybrid search, stage defaults, and per-run overrides
- Multi-agent orchestration: Workflow (sequential), Orchestrator
  (hub-and-spoke routing with LLM or keyword router), Team (collaborative
  with optional critique round and LLM/callable synthesizer), all with
  step-level tracing and shared state
- EchoProvider: compound-question coverage judged across top sources

## 0.4.0
- Multimodal parsing (`chunking="modality"`): typed chunks for text, tables
  (linearized + raw preserved), code, LaTeX equations, images (alt text +
  optional VLM captioner hook); modality-aware filtering via `by_modality`
- Knowledge graph indexing (`Knowledge(graph=True)` / `graph="llm"`): entity
  extraction, co-occurrence edges, cross-modal chunk linking; `graph_search`,
  `GraphRetrieve` stage, `Agent(retrieval="graph")`
- Incremental ingestion: content-hash de-dup on `Knowledge.add`
- New vector store adapters: Pinecone, Weaviate, Milvus, Elasticsearch,
  Redis (RediSearch), MongoDB Atlas Vector Search
- New embedding providers: OpenAI, Azure OpenAI, Cohere, Google Vertex AI
  (batched, L2-normalized)

## 0.3.0
- Vector store adapters: Chroma (`chroma[:path]`), Qdrant (`qdrant:<url>`), pgvector (`pgvector:<dsn>`), all behind the same `VectorStore` interface
- LLM-backed query rewriting (`QueryRewrite(rewriter="llm", model=...)`) with heuristic fallback
- Eval harness: `EvalSuite` / `EvalCase` / `EvalReport` with faithfulness, relevance, latency, gating metrics; markdown + JSON reports
- Playground UI (`stratarag.dashboard.serve`) — zero-dependency local dev dashboard

## 0.2.0
- Async agent runs (`Agent.arun`) with parallel async tool execution
- Streaming (`Agent.stream` / `Agent.astream`): token, tool, and result events
- LLM-backed semantic fact extraction (`Memory(extractor="llm", model=...)`) with heuristic fallback
- Cross-encoder reranker adapter (`Rerank(reranker="cross-encoder:<model>")`)

## 0.1.0
- Five primitives: Knowledge, Memory, Tool, Agent, Pipeline
- Chunking: fixed, recursive, markdown, semantic, parent_child
- Local stores: in-memory, SQLite; hashing embedder (offline default)
- Typed memory: semantic, episodic, procedural, prospective, working
- Production RAG stages incl. hybrid retrieval (BM25 + dense, RRF) and confidence gating
- Anthropic provider; deterministic `echo` provider for tests/CI

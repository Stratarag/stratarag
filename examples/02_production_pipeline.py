"""The production RAG architecture as a pipeline: rewrite -> hybrid retrieve
-> rerank -> filter -> grounded generate -> confidence gate."""
import stratarag as mn
from stratarag.pipeline import (QueryRewrite, HybridRetrieve, Rerank,
                            ContextFilter, Generate, ConfidenceGate, Pipeline)

kb = mn.Knowledge.from_texts([
    "The API rate limit is 100 requests per minute per key.",
    "Webhooks retry three times with exponential backoff.",
    "Enterprise plans include single sign-on and audit logs.",
], chunking="recursive", max_words=60)

pipe = Pipeline(
    QueryRewrite(rewriter="heuristic"),          # or rewriter="llm", model=...
    HybridRetrieve(kb, k=10, alpha=0.5),
    Rerank(reranker="lexical", top_n=3),         # or "cross-encoder:<model>"
    ContextFilter(max_chunks=3),
    Generate("echo", grounded=True),
    ConfidenceGate(threshold=0.5),
)

ctx = pipe.run("hey can you tell me the api rate limits?")
print("answer   :", ctx.answer)
print("confidence:", round(ctx.confidence, 2), "gated:", ctx.gated)
for t in ctx.trace:
    print(f"  {t.stage:<16} {t.elapsed_ms:>7.2f} ms  {t.detail}")

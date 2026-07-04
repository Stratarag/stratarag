from __future__ import annotations

from typing import Any, Optional

from ..knowledge import Knowledge
from ..memory import Memory
from .base import Context, Pipeline, Stage
from .rerankers import CrossEncoderReranker, LexicalOverlapReranker
from .rewrite import HeuristicRewriter, LLMRewriter
from .stages import (
    ConfidenceGate,
    GraphRetrieve,
    ContextFilter,
    Generate,
    HybridRetrieve,
    MemoryRead,
    QueryRewrite,
    Rerank,
    Retrieve,
    grounding_score,
    confidence_score,
)


def default_rag(
    knowledge: Knowledge,
    model: Any,
    memory: Optional[Memory] = None,
    system: Optional[str] = None,
    k: int = 10,
    top_n: int = 5,
    confidence_threshold: float = 0.5,
    rewriter: Any = "heuristic",
    reranker: Any = "lexical",
) -> Pipeline:
    """The production RAG pipeline: rewrite -> hybrid retrieve -> rerank ->
    filter -> (memory) -> grounded generate -> confidence gate."""
    stages = [
        QueryRewrite(rewriter=rewriter, model=model),
        HybridRetrieve(knowledge, k=k),
        Rerank(reranker=reranker, top_n=top_n),
        ContextFilter(max_chunks=top_n),
    ]
    if memory is not None:
        stages.append(MemoryRead(memory))
    stages.append(Generate(model, system=system, grounded=True))
    stages.append(ConfidenceGate(threshold=confidence_threshold))
    return Pipeline(*stages)


__all__ = ["Pipeline", "Stage", "Context", "QueryRewrite", "Retrieve",
           "HybridRetrieve", "GraphRetrieve", "Rerank", "ContextFilter", "MemoryRead",
           "Generate", "ConfidenceGate", "default_rag", "grounding_score", "confidence_score",
           "LexicalOverlapReranker", "CrossEncoderReranker",
           "HeuristicRewriter", "LLMRewriter"]

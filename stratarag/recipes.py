"""Recipes: the ten classic RAG architectures as one-line pipeline builders.

Every recipe returns a plain Pipeline — crack it open, rearrange stages, or
subclass any stage when your use case outgrows the default.

    import stratarag as sr
    pipe = sr.recipes.corrective_rag(kb, model="claude-sonnet-4-6")
    ctx = pipe.run("what changed in the refund policy?")
"""
from __future__ import annotations

from typing import Any, Optional

from .knowledge import Knowledge
from .memory import Memory
from .pipeline.advanced import (CorrectiveRetrieve, Compress, IterativeRetrieve,
                                MultiHopRetrieve, SelfRAGGenerate)
from .pipeline.base import Pipeline
from .pipeline.stages import (ConfidenceGate, ContextFilter, Generate,
                              GraphRetrieve, HybridRetrieve, MemoryRead,
                              QueryRewrite, Rerank, Retrieve)


def _finish(model: Any, system: Optional[str], memory: Optional[Memory],
            gate: float):
    tail = []
    if memory is not None:
        tail.append(MemoryRead(memory))
    tail.append(Generate(model, system=system, grounded=True))
    if gate:
        tail.append(ConfidenceGate(threshold=gate))
    return tail


def simple_rag(kb: Knowledge, model: Any, k: int = 5, system: str = None,
               memory: Memory = None, gate: float = 0.0) -> Pipeline:
    """#1 Simple RAG: top-k dense retrieval -> grounded generation."""
    return Pipeline(Retrieve(kb, k=k), *_finish(model, system, memory, gate))


def hybrid_rag(kb: Knowledge, model: Any, k: int = 10, top_n: int = 5,
               alpha: float = 0.5, reranker: Any = "lexical",
               system: str = None, memory: Memory = None,
               gate: float = 0.0) -> Pipeline:
    """#2 Hybrid RAG: semantic + keyword fusion, then rerank."""
    return Pipeline(HybridRetrieve(kb, k=k, alpha=alpha),
                    Rerank(reranker=reranker, top_n=top_n),
                    *_finish(model, system, memory, gate))


def corrective_rag(kb: Knowledge, model: Any, k: int = 10,
                   min_relevance: float = 0.34, system: str = None,
                   memory: Memory = None, gate: float = 0.35) -> Pipeline:
    """#3 CRAG: relevance-scored retrieval with fallback search, gated."""
    return Pipeline(CorrectiveRetrieve(kb, k=k, min_relevance=min_relevance),
                    ContextFilter(max_chunks=5),
                    *_finish(model, system, memory, gate))


def self_rag(kb: Knowledge, model: Any, k: int = 8, max_rounds: int = 2,
             system: str = None, memory: Memory = None,
             gate: float = 0.0) -> Pipeline:
    """#4 Self-RAG: draft -> self-critique -> re-retrieve -> regenerate."""
    stages = [HybridRetrieve(kb, k=k)]
    if memory is not None:
        stages.append(MemoryRead(memory))
    stages.append(SelfRAGGenerate(model, knowledge=kb, system=system,
                                  max_rounds=max_rounds))
    if gate:
        stages.append(ConfidenceGate(threshold=gate))
    return Pipeline(*stages)


def graph_rag(kb: Knowledge, model: Any, k: int = 8, hops: int = 1,
              system: str = None, memory: Memory = None,
              gate: float = 0.0) -> Pipeline:
    """#5 Graph RAG: entity-graph expansion (requires Knowledge(graph=True))."""
    return Pipeline(GraphRetrieve(kb, k=k, hops=hops),
                    *_finish(model, system, memory, gate))


def multi_hop_rag(kb: Knowledge, model: Any, k_per_hop: int = 4,
                  decomposer: Any = "heuristic", system: str = None,
                  memory: Memory = None, gate: float = 0.0) -> Pipeline:
    """#7 Multi-Hop RAG: sub-question decomposition, retrieve per hop."""
    return Pipeline(
        MultiHopRetrieve(kb, k_per_hop=k_per_hop, decomposer=decomposer,
                         model=model if decomposer == "llm" else None),
        Rerank(top_n=6), *_finish(model, system, memory, gate))


def iterative_rag(kb: Knowledge, model: Any, k: int = 8, loops: int = 2,
                  system: str = None, memory: Memory = None,
                  gate: float = 0.0) -> Pipeline:
    """#8 Iterative RAG: bounded query-refinement loops."""
    return Pipeline(IterativeRetrieve(kb, k=k, loops=loops),
                    *_finish(model, system, memory, gate))


def compression_rag(kb: Knowledge, model: Any, k: int = 12, top_n: int = 6,
                    system: str = None, memory: Memory = None,
                    gate: float = 0.0) -> Pipeline:
    """#9 Contextual Compression RAG: retrieve wide, compress to the
    query-relevant sentences before generation."""
    return Pipeline(HybridRetrieve(kb, k=k), Rerank(top_n=top_n), Compress(),
                    *_finish(model, system, memory, gate))


def metadata_rag(kb: Knowledge, model: Any, where: dict, k: int = 8,
                 system: str = None, memory: Memory = None,
                 gate: float = 0.0) -> Pipeline:
    """#10 Metadata-Driven RAG: hard-filter retrieval by tags/source/date
    before ranking, e.g. where={"source": "approved_policies.md"}."""
    return Pipeline(HybridRetrieve(kb, k=k, where=where),
                    *_finish(model, system, memory, gate))


# #6 Agentic RAG is the Agent itself: stratarag.Agent(model, tools=[...],
# knowledge=kb) — plans, calls tools, retrieves, and iterates.

ALL = {
    "simple": simple_rag, "hybrid": hybrid_rag, "corrective": corrective_rag,
    "self": self_rag, "graph": graph_rag, "multi_hop": multi_hop_rag,
    "iterative": iterative_rag, "compression": compression_rag,
    "metadata": metadata_rag,
}


def build(name: str, kb: Knowledge, model: Any, **kwargs) -> Pipeline:
    """recipes.build("corrective", kb, model, ...) — string-addressable."""
    if name not in ALL:
        raise ValueError(f"Unknown recipe {name!r}. Available: {sorted(ALL)}")
    return ALL[name](kb, model, **kwargs)

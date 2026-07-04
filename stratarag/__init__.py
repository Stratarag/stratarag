"""stratarag — agents that remember.

Five primitives: Knowledge, Memory, Tool, Agent, Pipeline.

    import stratarag as mn

    kb = mn.Knowledge.from_texts(["Refunds are accepted within 30 days."])
    agent = mn.Agent(model="echo", knowledge=kb, memory=mn.Memory())
    print(agent.run("What's the refund window?"))
"""
from . import recipes
from .agent import Agent
from .errors import (
    ConfigurationError,
    GenerationError,
    MissingDependencyError,
    StrataRAGError,
    StoreError,
    ToolError,
)
from .evals import EvalCase, EvalReport, EvalSuite
from .graph import EntityGraph
from .knowledge import Knowledge
from .memory import Memory, MemoryContext
from .pipeline import (
    ConfidenceGate,
    Context,
    ContextFilter,
    Generate,
    GraphRetrieve,
    HybridRetrieve,
    MemoryRead,
    Pipeline,
    QueryRewrite,
    Rerank,
    Retrieve,
    Stage,
    default_rag,
)
from .orchestration import OrchestrationResult, Orchestrator, Team, Workflow
from .pipeline.advanced import (
    Compress,
    CorrectiveRetrieve,
    IterativeRetrieve,
    MultiHopRetrieve,
    SelfRAGGenerate,
)
from .tools import Tool, ToolRegistry, tool
from .types import AgentResult, Chunk, Document, Message, ScoredChunk

__version__ = "0.5.0"

__all__ = [
    "Agent", "Knowledge", "Memory", "MemoryContext", "Pipeline", "Stage",
    "Context", "tool", "Tool", "ToolRegistry", "default_rag",
    "QueryRewrite", "Retrieve", "HybridRetrieve", "GraphRetrieve", "Rerank", "ContextFilter",
    "MemoryRead", "Generate", "ConfidenceGate",
    "CorrectiveRetrieve", "MultiHopRetrieve", "IterativeRetrieve",
    "Compress", "SelfRAGGenerate", "recipes",
    "Workflow", "Orchestrator", "Team", "OrchestrationResult",
    "EvalCase", "EvalSuite", "EvalReport", "EntityGraph",
    "Document", "Chunk", "ScoredChunk", "Message", "AgentResult",
    "StrataRAGError", "ConfigurationError", "MissingDependencyError",
    "StoreError", "ToolError", "GenerationError",
    "__version__",
]

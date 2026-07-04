"""Multi-agent orchestration: the three enterprise deployment archetypes.

Workflow      (sequential)    - deterministic step chains: legal review, tax
                                filing, underwriting. Each step reads/writes a
                                shared state dict; steps are agents, pipelines,
                                or plain callables.
Orchestrator  (hub-and-spoke) - a router dispatches each task to the best
                                specialist: onboarding, predictive maintenance.
Team          (collaborative) - specialists each contribute, then a
                                synthesizer merges; optional critique round:
                                cybersecurity triage, AP auditing, ABM.

All three return an OrchestrationResult with a full step-by-step trace, and
every unit is just an Agent/Pipeline/callable — compose them freely.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .agent import Agent
from .errors import ConfigurationError
from .llm import resolve_provider
from .pipeline.base import Pipeline
from .types import AgentResult, Message

Unit = Union[Agent, Pipeline, Callable[..., Any]]


@dataclass
class StepRecord:
    name: str
    input: str
    output: str
    elapsed_ms: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    output: str
    steps: List[StepRecord] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output


def _run_unit(unit: Unit, task: str, state: Dict[str, Any],
              user_id: str = "default") -> Tuple[str, Dict[str, Any]]:
    """Execute any supported unit type and normalize its output to text."""
    if isinstance(unit, Agent):
        res: AgentResult = unit.run(task, user_id=user_id)
        return res.output, {"confidence": res.confidence, "gated": res.gated}
    if isinstance(unit, Pipeline):
        ctx = unit.run(task, user_id=user_id)
        return ctx.answer, {"confidence": ctx.confidence, "gated": ctx.gated}
    if callable(unit):
        try:
            out = unit(task, state)          # step(task, state) preferred
        except TypeError:
            out = unit(task)                 # plain fn(task) also fine
        return ("" if out is None else str(out)), {}
    raise ConfigurationError(f"Unsupported step unit: {unit!r}")


class Workflow:
    """Sequential archetype. Steps are (name, unit) pairs; each step receives
    the previous step's output as its task (or a template over state)."""

    def __init__(self, steps: List[Tuple[str, Unit]],
                 stop_on: Optional[Callable[[str, Dict], bool]] = None):
        if not steps:
            raise ConfigurationError("Workflow needs at least one step")
        self.steps = steps
        self.stop_on = stop_on

    def run(self, task: str, user_id: str = "default",
            state: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        state = dict(state or {})
        state.setdefault("task", task)
        records: List[StepRecord] = []
        current = task
        for name, unit in self.steps:
            t0 = time.perf_counter()
            out, meta = _run_unit(unit, current, state, user_id)
            records.append(StepRecord(
                name=name, input=current, output=out,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                meta=meta))
            state[name] = out
            current = out or current
            if self.stop_on and self.stop_on(out, state):
                break
        return OrchestrationResult(output=current, steps=records, state=state)


_ROUTE_PROMPT = """Route this task to exactly one specialist. Reply with ONLY
the specialist's name, nothing else.

Specialists:
{roster}

Task: {task}"""


class Orchestrator:
    """Hub-and-spoke archetype. Registered specialists carry descriptions;
    a router (LLM, or keyword fallback) dispatches each task."""

    def __init__(self, specialists: Dict[str, Tuple[str, Unit]],
                 router: Any = None):
        """specialists: {name: (description, unit)}; router: model spec or
        None for deterministic keyword routing."""
        if not specialists:
            raise ConfigurationError("Orchestrator needs at least one specialist")
        self.specialists = specialists
        self._router = resolve_provider(router) if router is not None else None

    def route(self, task: str) -> str:
        names = list(self.specialists)
        if self._router is not None:
            roster = "\n".join(f"- {n}: {d}" for n, (d, _) in self.specialists.items())
            try:
                resp = self._router.complete([Message(
                    role="user",
                    content=_ROUTE_PROMPT.format(roster=roster, task=task))])
                choice = resp.text.strip().splitlines()[0].strip(" .:`\"'").lower()
                for n in names:
                    if n.lower() == choice or n.lower() in choice:
                        return n
            except Exception:
                pass
        # deterministic fallback: description keyword overlap
        task_words = set(task.lower().split())
        best, best_score = names[0], -1
        for n, (desc, _) in self.specialists.items():
            words = set(desc.lower().split()) | set(n.lower().replace("_", " ").split())
            score = len(task_words & words)
            if score > best_score:
                best, best_score = n, score
        return best

    def run(self, task: str, user_id: str = "default") -> OrchestrationResult:
        name = self.route(task)
        _, unit = self.specialists[name]
        t0 = time.perf_counter()
        out, meta = _run_unit(unit, task, {}, user_id)
        rec = StepRecord(name=f"route->{name}", input=task, output=out,
                         elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                         meta=meta)
        return OrchestrationResult(output=out, steps=[rec],
                                   state={"routed_to": name})


_SYNTH_PROMPT = """You are the lead of a team. Merge the specialists'
contributions below into one final, coherent answer to the task. Be concise
and do not mention the specialists.

Task: {task}

Contributions:
{contributions}"""


class Team:
    """Collaborative archetype. Every member contributes on the task; a
    synthesizer (LLM or callable) merges. With critique=True, each member
    sees the others' first-round contributions and may revise once —
    the challenge/validate pattern from multi-agent systems."""

    def __init__(self, members: Dict[str, Unit], synthesizer: Any = None,
                 critique: bool = False):
        if not members:
            raise ConfigurationError("Team needs at least one member")
        self.members = members
        self.critique = critique
        self._synth_provider = None
        self._synth_fn = None
        if callable(synthesizer) and not isinstance(synthesizer, str):
            self._synth_fn = synthesizer
        elif synthesizer is not None:
            self._synth_provider = resolve_provider(synthesizer)

    def _synthesize(self, task: str, contributions: Dict[str, str]) -> str:
        if self._synth_fn is not None:
            return str(self._synth_fn(task, contributions))
        if self._synth_provider is not None:
            body = "\n".join(f"[{n}] {c}" for n, c in contributions.items())
            resp = self._synth_provider.complete([Message(
                role="user",
                content=_SYNTH_PROMPT.format(task=task, contributions=body))])
            if resp.text.strip():
                return resp.text
        return "\n".join(f"{n}: {c}" for n, c in contributions.items())

    def run(self, task: str, user_id: str = "default") -> OrchestrationResult:
        records: List[StepRecord] = []
        contributions: Dict[str, str] = {}
        for name, unit in self.members.items():
            t0 = time.perf_counter()
            out, meta = _run_unit(unit, task, {}, user_id)
            contributions[name] = out
            records.append(StepRecord(
                name=name, input=task, output=out,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                meta=meta))
        if self.critique:
            peer_view = json.dumps(contributions)
            for name, unit in self.members.items():
                revise_task = (f"{task}\n\nPeer contributions: {peer_view}\n"
                               "Revise or confirm your contribution.")
                t0 = time.perf_counter()
                out, meta = _run_unit(unit, revise_task, {}, user_id)
                if out.strip():
                    contributions[name] = out
                records.append(StepRecord(
                    name=f"{name}:revise", input=revise_task[:120], output=out,
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
                    meta=meta))
        t0 = time.perf_counter()
        final = self._synthesize(task, contributions)
        records.append(StepRecord(
            name="synthesize", input=task, output=final,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))
        return OrchestrationResult(output=final, steps=records,
                                   state={"contributions": contributions})

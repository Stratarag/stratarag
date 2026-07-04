"""Eval harness: run cases against an Agent, Pipeline, or plain callable and
score answers on faithfulness, relevance, and expectation checks.

    suite = EvalSuite([
        EvalCase("What is the refund window?", expected_contains=["30 days"]),
    ])
    report = suite.run(agent)
    print(report.to_markdown())
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..agent import Agent
from ..pipeline.base import Pipeline
from ..pipeline.stages import grounding_score
from ..types import AgentResult


@dataclass
class EvalCase:
    query: str
    expected_contains: Optional[List[str]] = None
    expected_exact: Optional[str] = None
    user_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case: EvalCase
    answer: str
    passed: bool
    faithfulness: float
    relevance: float
    confidence: float
    gated: bool
    latency_ms: float
    failure_reason: str = ""


@dataclass
class EvalReport:
    results: List[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def _mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results]
        return statistics.fmean(vals) if vals else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "cases": len(self.results),
            "pass_rate": round(self.pass_rate, 3),
            "avg_faithfulness": round(self._mean("faithfulness"), 3),
            "avg_relevance": round(self._mean("relevance"), 3),
            "avg_confidence": round(self._mean("confidence"), 3),
            "avg_latency_ms": round(self._mean("latency_ms"), 2),
            "gated": sum(r.gated for r in self.results),
        }

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            "# Eval report",
            "",
            f"- cases: {s['cases']}  |  pass rate: {s['pass_rate']:.0%}  "
            f"|  gated: {s['gated']}",
            f"- avg faithfulness: {s['avg_faithfulness']}  "
            f"|  avg relevance: {s['avg_relevance']}  "
            f"|  avg latency: {s['avg_latency_ms']} ms",
            "",
            "| # | query | pass | faith | rel | ms | note |",
            "|---|-------|------|-------|-----|----|------|",
        ]
        for i, r in enumerate(self.results, 1):
            q = r.case.query[:48].replace("|", "/")
            lines.append(
                f"| {i} | {q} | {'✅' if r.passed else '❌'} "
                f"| {r.faithfulness:.2f} | {r.relevance:.2f} "
                f"| {r.latency_ms:.0f} | {r.failure_reason[:40]} |")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "summary": self.summary(),
            "results": [{
                "query": r.case.query, "answer": r.answer, "passed": r.passed,
                "faithfulness": r.faithfulness, "relevance": r.relevance,
                "confidence": r.confidence, "gated": r.gated,
                "latency_ms": r.latency_ms, "failure_reason": r.failure_reason,
            } for r in self.results],
        }, indent=2)


Target = Union[Agent, Pipeline, Callable[[str], Any]]


class EvalSuite:
    def __init__(self, cases: List[EvalCase]):
        self.cases = cases

    def _invoke(self, target: Target, case: EvalCase):
        if isinstance(target, Agent):
            res = target.run(case.query, user_id=case.user_id)
            return res.output, [sc.chunk.text for sc in res.sources], \
                res.confidence, res.gated
        if isinstance(target, Pipeline):
            ctx = target.run(case.query, user_id=case.user_id)
            return ctx.answer, [sc.chunk.text for sc in ctx.candidates], \
                ctx.confidence, ctx.gated
        out = target(case.query)
        if isinstance(out, AgentResult):
            return out.output, [sc.chunk.text for sc in out.sources], \
                out.confidence, out.gated
        return str(out), [], 1.0, False

    def run(self, target: Target) -> EvalReport:
        results: List[CaseResult] = []
        for case in self.cases:
            t0 = time.perf_counter()
            answer, ctx_texts, confidence, gated = self._invoke(target, case)
            latency = (time.perf_counter() - t0) * 1000
            passed, reason = True, ""
            if case.expected_exact is not None and answer.strip() != case.expected_exact.strip():
                passed, reason = False, "exact mismatch"
            if case.expected_contains:
                missing = [e for e in case.expected_contains
                           if e.lower() not in answer.lower()]
                if missing:
                    passed, reason = False, f"missing: {missing}"
            results.append(CaseResult(
                case=case, answer=answer, passed=passed,
                faithfulness=grounding_score(answer, ctx_texts) if ctx_texts else 0.0,
                relevance=grounding_score(case.query, ctx_texts) if ctx_texts else 0.0,
                confidence=confidence, gated=gated,
                latency_ms=latency, failure_reason=reason))
        return EvalReport(results)

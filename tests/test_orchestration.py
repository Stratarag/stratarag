"""Tests for the three multi-agent archetypes."""
import unittest

import stratarag as sr
from stratarag.errors import ConfigurationError
from stratarag.llm.echo import EchoProvider
from stratarag.orchestration import Orchestrator, Team, Workflow


def upper_step(task, state):
    state["saw"] = task
    return task.upper()


class TestWorkflow(unittest.TestCase):
    def test_sequential_chain_threads_output(self):
        wf = Workflow([
            ("extract", lambda t: f"extracted:{t}"),
            ("validate", upper_step),
            ("file", lambda t, s: f"filed[{t}] steps={len(s)}"),
        ])
        res = wf.run("invoice-42")
        self.assertEqual([s.name for s in res.steps], ["extract", "validate", "file"])
        self.assertIn("FILED", res.output.upper())
        self.assertEqual(res.state["extract"], "extracted:invoice-42")
        self.assertEqual(res.state["saw"], "extracted:invoice-42")
        self.assertTrue(all(s.elapsed_ms >= 0 for s in res.steps))

    def test_agents_and_pipelines_as_steps(self):
        kb = sr.Knowledge.from_texts(["Refund window is 14 days."])
        drafter = sr.Agent(model=EchoProvider(script=["draft: refund is 14 days"]))
        checker = sr.recipes.simple_rag(kb, "echo")
        res = Workflow([("draft", drafter), ("check", checker)]).run("refund policy?")
        self.assertIn("14 days", res.output)
        self.assertIn("confidence", res.steps[0].meta)

    def test_stop_on_halts_early(self):
        wf = Workflow([("a", lambda t: "REJECTED"), ("b", lambda t: "never")],
                      stop_on=lambda out, state: "REJECTED" in out)
        res = wf.run("x")
        self.assertEqual(len(res.steps), 1)
        self.assertEqual(res.output, "REJECTED")

    def test_empty_workflow_rejected(self):
        with self.assertRaises(ConfigurationError):
            Workflow([])


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.hub = Orchestrator({
            "billing": ("handles invoices refunds payments billing",
                        lambda t: "billing handled"),
            "it_support": ("handles laptop hardware software access",
                           lambda t: "it handled"),
        })

    def test_keyword_routing(self):
        self.assertEqual(self.hub.route("I need a refund on my invoice"), "billing")
        self.assertEqual(self.hub.route("my laptop needs software access"), "it_support")

    def test_run_records_route(self):
        res = self.hub.run("refund my payment")
        self.assertEqual(res.state["routed_to"], "billing")
        self.assertEqual(res.output, "billing handled")
        self.assertTrue(res.steps[0].name.startswith("route->"))

    def test_llm_router_and_fallback(self):
        hub = Orchestrator(
            {"billing": ("money", lambda t: "b"), "it": ("tech", lambda t: "i")},
            router=EchoProvider(script=["it"]))
        self.assertEqual(hub.route("anything"), "it")
        hub2 = Orchestrator(
            {"billing": ("invoice refund", lambda t: "b"),
             "it": ("laptop", lambda t: "i")},
            router=EchoProvider(script=["no-such-specialist"]))
        self.assertEqual(hub2.route("refund the invoice"), "billing")  # fallback

    def test_empty_orchestrator_rejected(self):
        with self.assertRaises(ConfigurationError):
            Orchestrator({})


class TestTeam(unittest.TestCase):
    def test_contributions_and_callable_synthesizer(self):
        team = Team(
            {"researcher": lambda t: "found A",
             "analyst": lambda t: "risk low"},
            synthesizer=lambda task, c: f"FINAL({len(c)} inputs)")
        res = team.run("assess vendor")
        self.assertEqual(res.output, "FINAL(2 inputs)")
        self.assertEqual(res.state["contributions"]["researcher"], "found A")
        self.assertEqual(res.steps[-1].name, "synthesize")

    def test_llm_synthesizer(self):
        team = Team({"a": lambda t: "x", "b": lambda t: "y"},
                     synthesizer=EchoProvider(script=["merged answer"]))
        self.assertEqual(team.run("t").output, "merged answer")

    def test_default_synthesis_without_synthesizer(self):
        res = Team({"a": lambda t: "alpha"}).run("t")
        self.assertIn("alpha", res.output)

    def test_critique_round_revises(self):
        calls = {"n": 0}
        def flaky(task):
            calls["n"] += 1
            return "revised!" if "Peer contributions" in task else "first pass"
        res = Team({"m": flaky}, critique=True).run("t")
        self.assertEqual(res.state["contributions"]["m"], "revised!")
        names = [s.name for s in res.steps]
        self.assertIn("m:revise", names)

    def test_agents_as_members(self):
        team = Team({
            "triage": sr.Agent(model=EchoProvider(script=["sev2 incident"])),
            "forensics": sr.Agent(model=EchoProvider(script=["matches T1078"])),
        })
        res = team.run("suspicious login burst")
        self.assertIn("sev2", res.state["contributions"]["triage"])
        self.assertIn("T1078", res.state["contributions"]["forensics"])


if __name__ == "__main__":
    unittest.main()

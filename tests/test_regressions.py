"""Regression tests for bugs found by dogfooding a document-QA agent.
Each test reproduces the original failure before the fix."""
import unittest

from stratarag import Agent, Knowledge, Memory
from stratarag.llm.echo import EchoProvider
from stratarag.memory.extractors import HeuristicExtractor
from stratarag.pipeline.stages import confidence_score
from stratarag.types import Message

HANDBOOK = """# HR Handbook

## Leave Policy
Employees receive 24 days of paid annual leave per year. Unused leave up to
5 days can be carried over to the next calendar year.

## Remote Work
Remote work is allowed up to 3 days per week.
"""


class TestBug1MultilineChunks(unittest.TestCase):
    """Answers were built from only the first line of multi-line chunks
    ('Leave Policy' instead of the body with '24 days')."""

    def test_full_chunk_text_reaches_the_answer(self):
        kb = Knowledge.from_texts([HANDBOOK], chunking="markdown")
        agent = Agent(model="echo", knowledge=kb)
        res = agent.run("how many days of annual leave do employees get?")
        self.assertIn("24 days", res.output)

    def test_echo_parses_multiline_source_blocks(self):
        e = EchoProvider()
        sys_msg = ("Answer ONLY from the sources below.\n"
                   "[source 1] Title Line\nBody with the real answer inside.\n"
                   "[source 2] Other chunk here.")
        blocks = e._sources(sys_msg)
        self.assertIn("Body with the real answer inside.", blocks[0])


class TestBug2IrrelevantContextGating(unittest.TestCase):
    """A parroting model scored confidence 1.0 on questions the docs can't
    answer, because faithfulness alone ignores query-context relevance."""

    def test_unanswerable_question_is_gated(self):
        kb = Knowledge.from_texts([HANDBOOK], chunking="markdown")
        agent = Agent(model="echo", knowledge=kb, confidence_threshold=0.35)
        res = agent.run("What is the company's policy on office pets?")
        self.assertTrue(res.gated, f"should gate, got: {res.output!r}")

    def test_answerable_question_is_not_gated(self):
        kb = Knowledge.from_texts([HANDBOOK], chunking="markdown")
        agent = Agent(model="echo", knowledge=kb, confidence_threshold=0.35)
        res = agent.run("How many remote days per week are allowed?")
        self.assertFalse(res.gated)
        self.assertIn("3 days", res.output)

    def test_confidence_score_punishes_irrelevant_context(self):
        ctx = ["Employees receive 24 days of paid annual leave per year."]
        parroted = ctx[0]
        relevant = confidence_score("how much annual leave?", parroted, ctx)
        irrelevant = confidence_score("office pets policy?", parroted, ctx)
        self.assertGreater(relevant, 0.6)
        self.assertLess(irrelevant, 0.35)


class TestBug3IntroductionFacts(unittest.TestCase):
    """'Hi, I'm Rohan from the design team' produced zero memory facts."""

    def test_im_name_and_team_are_extracted(self):
        facts = " ".join(HeuristicExtractor().extract(
            "Hi, I'm Rohan from the design team. How many leave days?"))
        self.assertIn("Rohan", facts)
        self.assertIn("design", facts)

    def test_lowercase_im_is_not_a_name(self):
        facts = HeuristicExtractor().extract("i'm tired of meetings")
        self.assertFalse(any("tired" in f and "I'm" in f for f in facts))

    def test_agent_remembers_introduction(self):
        kb = Knowledge.from_texts([HANDBOOK], chunking="markdown")
        mem = Memory(semantic=True)
        agent = Agent(model="echo", knowledge=kb, memory=mem)
        agent.run("Hi, I'm Rohan from the design team. Leave policy?",
                  user_id="rohan")
        facts = " ".join(f.content for f in mem.semantic.all("rohan"))
        self.assertIn("Rohan", facts)


if __name__ == "__main__":
    unittest.main()


class TestReleaseHygiene(unittest.TestCase):
    """Guards the release pipeline's assumptions."""

    def test_package_version_matches_pyproject(self):
        import os
        import stratarag
        path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        if not os.path.exists(path):  # not present in installed wheels
            self.skipTest("source tree only")
        try:
            import tomllib
        except ImportError:
            self.skipTest("py<3.11")
        with open(path, "rb") as fh:
            declared = tomllib.load(fh)["project"]["version"]
        self.assertEqual(declared, stratarag.__version__)

    def test_core_has_no_module_level_third_party_imports(self):
        import ast, os, sys
        if hasattr(sys, "stdlib_module_names"):
            stdlib = set(sys.stdlib_module_names)
        else:
            # Python 3.9 compatibility
            import sysconfig

            stdlib = set()
            stdlib_path = sysconfig.get_paths()["stdlib"]

            for _, module_name, _ in __import__("pkgutil").iter_modules([stdlib_path]):
                stdlib.add(module_name)

            # Builtins aren't returned by iter_modules()
            stdlib.update({
                "math",
                "cmath",
                "array",
                "binascii",
                "errno",
                "fcntl",
                "grp",
                "mmap",
                "resource",
                "select",
                "socket",
                "ssl",
                "termios",
                "unicodedata",
                "zlib",
            })
        root = os.path.join(os.path.dirname(__file__), "..", "stratarag")
        if not os.path.isdir(root):
            self.skipTest("source tree only")
        offenders = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=path)
                for node in tree.body:   # module level only; lazy is fine
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                            and node.module:
                        names = [node.module.split(".")[0]]
                    offenders += [f"{f}:{n}" for n in names
                                  if n not in stdlib and n != "stratarag"]
        self.assertEqual(offenders, [])

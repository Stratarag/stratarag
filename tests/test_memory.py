import time
import unittest

from stratarag.llm.echo import EchoProvider
from stratarag.memory import Memory
from stratarag.memory.extractors import HeuristicExtractor, LLMExtractor
from stratarag.types import Message


class TestHeuristicExtractor(unittest.TestCase):
    def test_extracts_name_and_preferences(self):
        facts = HeuristicExtractor().extract(
            "Hi! My name is Priya. I prefer TypeScript over Python. What's up?")
        joined = " ".join(facts).lower()
        self.assertIn("priya", joined)
        self.assertIn("typescript", joined)

    def test_no_facts_in_small_talk(self):
        self.assertEqual(HeuristicExtractor().extract("what's the weather like?"), [])


class TestLLMExtractor(unittest.TestCase):
    def test_parses_json_facts(self):
        ex = LLMExtractor(EchoProvider(script=['["User prefers dark mode"]']))
        self.assertEqual(ex.extract("i love dark mode", ""), ["User prefers dark mode"])

    def test_falls_back_on_garbage(self):
        ex = LLMExtractor(EchoProvider(script=["not json at all"]))
        facts = ex.extract("My name is Ravi.", "")
        self.assertTrue(any("Ravi" in f for f in facts))


class TestSemanticMemory(unittest.TestCase):
    def test_store_recall_scoped_by_user(self):
        mem = Memory(semantic=True)
        mem.remember("User prefers metric units", user_id="u1")
        mem.remember("User is allergic to peanuts", user_id="u2")
        got = mem.semantic.search("what units should I use", user_id="u1", k=2)
        self.assertTrue(any("metric" in r.content for r in got))
        self.assertFalse(any("peanuts" in r.content for r in got))

    def test_dedupes_identical_facts(self):
        mem = Memory(semantic=True)
        mem.remember("User prefers metric units", user_id="u1")
        mem.remember("User prefers metric units", user_id="u1")
        self.assertEqual(len(mem.semantic.all("u1")), 1)


class TestEpisodicProcedural(unittest.TestCase):
    def test_episode_logging_and_recall(self):
        mem = Memory(semantic=False, episodic=True)
        mem.episodic.log("deploy the api", "failed: missing env var",
                         success=False, reflection="check env first", user_id="u1")
        got = mem.episodic.search("deploy api", user_id="u1", k=1)
        self.assertEqual(len(got), 1)
        self.assertFalse(got[0].metadata["success"])
        self.assertIn("env", got[0].metadata["reflection"])

    def test_procedural_register_lookup(self):
        mem = Memory(semantic=False, procedural=True)
        mem.procedural.register("release", ["run tests", "tag version", "publish"],
                                user_id="u1")
        got = mem.procedural.lookup("how do I do a release", user_id="u1")
        self.assertIn("release", got[0].metadata["name"])
        self.assertEqual(got[0].metadata["steps"][0], "run tests")


class TestProspective(unittest.TestCase):
    def test_time_trigger(self):
        mem = Memory(semantic=False, prospective=True)
        mem.prospective.add("send weekly report", due_at=time.time() - 1, user_id="u1")
        due = mem.prospective.due(user_id="u1")
        self.assertEqual(len(due), 1)
        mem.prospective.complete(due[0]["id"])
        self.assertEqual(mem.prospective.due(user_id="u1"), [])

    def test_keyword_trigger(self):
        mem = Memory(semantic=False, prospective=True)
        mem.prospective.add("mention the migration", trigger="database", user_id="u1")
        self.assertEqual(len(mem.prospective.due("about our database", user_id="u1")), 1)
        self.assertEqual(mem.prospective.due("about lunch", user_id="u1"), [])


class TestWorkingMemory(unittest.TestCase):
    def test_trims_to_budget(self):
        mem = Memory(semantic=False, working=True, max_working_words=20)
        for i in range(10):
            mem.working.append(Message(role="user", content=f"message {i} has four words"),
                               user_id="u1")
        msgs = mem.working.messages("u1")
        self.assertLessEqual(sum(len(m.content.split()) for m in msgs), 25)
        self.assertIn("message 9", msgs[-1].content)

    def test_summarizer_compresses_dropped_turns(self):
        mem = Memory(semantic=False, working=True, max_working_words=15)
        mem.working.summarizer = lambda dropped: f"({len(dropped)} turns dropped)"
        for i in range(8):
            mem.working.append(Message(role="user", content=f"turn {i} words words words"),
                               user_id="u1")
        first = mem.working.messages("u1")[0]
        self.assertEqual(first.role, "system")
        self.assertIn("dropped", first.content)


class TestWriteTurnAndRead(unittest.TestCase):
    def test_full_cycle(self):
        mem = Memory(semantic=True, episodic=True, working=True)
        new = mem.write_turn("My name is Asha and I use vim.",
                             "Nice to meet you, Asha!", user_id="u9")
        self.assertTrue(new)
        ctx = mem.read("what editor do I use", user_id="u9")
        self.assertTrue(any("vim" in f.content for f in ctx.facts))
        self.assertTrue(ctx.history)
        rendered = ctx.render()
        self.assertIn("Known about this user", rendered)

    def test_sqlite_backend_persistence_shape(self):
        mem = Memory(semantic=True, backend="sqlite")
        mem.remember("User works at Acme", user_id="u1")
        got = mem.read("where does the user work", user_id="u1")
        self.assertTrue(any("Acme" in f.content for f in got.facts))


if __name__ == "__main__":
    unittest.main()

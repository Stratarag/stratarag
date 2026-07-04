import unittest

from stratarag.chunking import (FixedSizeChunker, MarkdownChunker, ParentChildChunker,
                            RecursiveChunker, SemanticChunker, resolve_chunker)
from stratarag.errors import ConfigurationError
from stratarag.types import Document


class TestFixedSize(unittest.TestCase):
    def test_splits_with_overlap(self):
        doc = Document(text=" ".join(f"w{i}" for i in range(100)))
        chunks = FixedSizeChunker(size=40, overlap=0.25).chunk(doc)
        self.assertGreater(len(chunks), 2)
        # overlap: last words of chunk 0 appear in chunk 1
        tail = chunks[0].text.split()[-5:]
        self.assertTrue(all(w in chunks[1].text.split() for w in tail))

    def test_empty_doc(self):
        self.assertEqual(FixedSizeChunker().chunk(Document(text="")), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            FixedSizeChunker(overlap=1.0)

    def test_short_doc_single_chunk(self):
        chunks = FixedSizeChunker(size=100).chunk(Document(text="just five words right here"))
        self.assertEqual(len(chunks), 1)


class TestRecursive(unittest.TestCase):
    def test_respects_max_words(self):
        paras = "\n\n".join("sentence one. sentence two. sentence three." for _ in range(10))
        chunks = RecursiveChunker(max_words=12).chunk(Document(text=paras))
        self.assertTrue(all(len(c.text.split()) <= 12 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_giant_word_is_hard_split(self):
        doc = Document(text=" ".join(["word"] * 50))
        chunks = RecursiveChunker(max_words=10).chunk(doc)
        self.assertTrue(all(len(c.text.split()) <= 10 for c in chunks))


class TestMarkdown(unittest.TestCase):
    def test_sections_and_paths(self):
        text = "intro line\n\n# Setup\ninstall it\n## Linux\napt install\n# Usage\nrun it"
        chunks = MarkdownChunker().chunk(Document(text=text))
        sections = [c.metadata.get("section") for c in chunks]
        self.assertIn("Setup", sections)
        self.assertIn("Setup > Linux", sections)
        self.assertIn("Usage", sections)
        self.assertEqual(chunks[0].text, "intro line")  # preamble preserved

    def test_no_headers_falls_back(self):
        chunks = MarkdownChunker().chunk(Document(text="plain text no headers"))
        self.assertEqual(len(chunks), 1)


class TestSemantic(unittest.TestCase):
    def test_splits_on_topic_shift(self):
        text = ("Cats are small felines. Cats enjoy sleeping all day. "
                "Quantum computers use superposition qubits. "
                "Quantum computers factor integers quickly.")
        chunks = SemanticChunker(threshold=0.2).chunk(Document(text=text))
        self.assertGreaterEqual(len(chunks), 2)

    def test_single_sentence(self):
        chunks = SemanticChunker().chunk(Document(text="One sentence only"))
        self.assertEqual(len(chunks), 1)


class TestParentChild(unittest.TestCase):
    def test_children_link_to_parents(self):
        text = " ".join(f"sentence number {i} is here." for i in range(120))
        chunker = ParentChildChunker(parent_words=100, child_words=20)
        children = chunker.chunk(Document(text=text))
        self.assertTrue(all(c.parent_id for c in children))
        parent_ids = {p.id for p in chunker.parents}
        self.assertTrue(all(c.parent_id in parent_ids for c in children))

    def test_invalid_sizes(self):
        with self.assertRaises(ValueError):
            ParentChildChunker(parent_words=50, child_words=100)


class TestRegistry(unittest.TestCase):
    def test_resolve_by_name(self):
        self.assertIsInstance(resolve_chunker("fixed", size=10), FixedSizeChunker)

    def test_unknown_raises(self):
        with self.assertRaises(ConfigurationError):
            resolve_chunker("nope")


if __name__ == "__main__":
    unittest.main()

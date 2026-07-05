"""Unit tests: everything imports the *real code* under test (no logic copied into the tests).

Core principle: the point of a test is to guard the real code. If you copy a piece of logic into
the test to test it, then when the real code breaks the test still passes green = false sense of
safety. So we always import the real functions/regexes.

These are "fast unit tests": no big models loaded, no network, they run in seconds.
(End-to-end verification involving VL/embedding/retrieval is a "slow integration test" -- costly,
done/kept separately.)

Run:  python test_rag.py       or      pytest
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestConfig(unittest.TestCase):

    def test_values_are_sensible(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP, CANDIDATE_K, BATCH_SIZE
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreater(CHUNK_OVERLAP, 0)
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)
        self.assertGreater(CANDIDATE_K, 0)
        self.assertGreater(BATCH_SIZE, 0)

    def test_model_paths_exist(self):
        # Note: this is an "environment check" that depends on local model paths; on another
        # machine without the models downloaded it will fail (as expected).
        from config import EMBED_MODEL, RERANKER_MODEL
        self.assertTrue(Path(EMBED_MODEL).exists(), f"Embed model missing: {EMBED_MODEL}")
        self.assertTrue(Path(RERANKER_MODEL).exists(), f"Reranker missing: {RERANKER_MODEL}")


class TestBboxOverlap(unittest.TestCase):
    """Tests the real loaders.pdf.bbox_overlap."""

    def test_separated(self):
        from loaders.pdf import bbox_overlap
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (20, 0, 30, 10)))    # separated left/right
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (0, 20, 10, 30)))    # separated top/bottom

    def test_overlap(self):
        from loaders.pdf import bbox_overlap
        self.assertTrue(bbox_overlap((0, 0, 20, 20), (10, 10, 30, 30)))    # partial overlap
        self.assertTrue(bbox_overlap((0, 0, 100, 100), (10, 10, 20, 20)))  # fully contained

    def test_touching_edge_not_overlap(self):
        from loaders.pdf import bbox_overlap
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (10, 0, 20, 10)))    # merely touching edges is not overlap


class TestTableToMarkdown(unittest.TestCase):
    """Tests the real loaders.pdf.table_to_markdown.

    It takes a "table object" (which must have .extract() returning a 2-D list), so we feed it a
    *fake table stub* -- that's a "test double": it exercises the real function without depending
    on a real PDF.
    """

    class _FakeTable:
        def __init__(self, rows): self._rows = rows
        def extract(self): return self._rows

    def test_basic(self):
        from loaders.pdf import table_to_markdown
        md = table_to_markdown(self._FakeTable([["Gene", "Score"], ["TP53", "0.87"]]))
        self.assertIn("| Gene | Score |", md)
        self.assertIn("| --- | --- |", md)
        self.assertIn("| TP53 | 0.87 |", md)

    def test_empty(self):
        from loaders.pdf import table_to_markdown
        self.assertEqual(table_to_markdown(self._FakeTable([])), "")

    def test_none_cells(self):
        from loaders.pdf import table_to_markdown
        md = table_to_markdown(self._FakeTable([["Gene", None], ["TP53", None]]))
        self.assertIn("| Gene |  |", md)


class TestSectionRE(unittest.TestCase):
    """Tests the real loaders.pdf.SECTION_RE."""

    def test_matches(self):
        from loaders.pdf import SECTION_RE
        for s in ["Introduction", "2. Methods", "RESULTS", "Discussion",
                  "Supplementary", "References", "1. Abstract"]:
            with self.subTest(s=s):
                self.assertTrue(SECTION_RE.match(s))

    def test_non_matches(self):
        from loaders.pdf import SECTION_RE
        for s in ["The results showed", "In this study", "Figure 1"]:
            with self.subTest(s=s):
                self.assertFalse(SECTION_RE.match(s))


class TestFigureRE(unittest.TestCase):
    """Tests the real loaders.pdf.FIGURE_RE."""

    def test_matches(self):
        from loaders.pdf import FIGURE_RE
        for s in ["Figure 1", "Fig. 2A", "Supplementary Figure 3",
                  "Extended Data Figure 1", "Fig 4B"]:
            with self.subTest(s=s):
                self.assertTrue(FIGURE_RE.match(s))

    def test_non_matches(self):
        from loaders.pdf import FIGURE_RE
        for s in ["The figure shows", "In Figure", "Table 1"]:
            with self.subTest(s=s):
                self.assertFalse(FIGURE_RE.match(s))


class TestHGNC(unittest.TestCase):
    """Gene-alias expansion. Uses hgnc's real CACHE_FILE path to check whether the cache exists
    (no more guessing directories)."""

    def setUp(self):
        from hgnc import CACHE_FILE
        if not CACHE_FILE.exists():
            self.skipTest(f"Gene cache not found ({CACHE_FILE}) -- put gene_aliases.json in place first")

    def test_known_gene_expanded(self):
        from hgnc import expand_query
        result = expand_query("What does TP53 do in apoptosis?")
        self.assertIn("Gene aliases", result)
        self.assertIn("TP53", result)

    def test_plain_text_unchanged(self):
        from hgnc import expand_query
        self.assertEqual(expand_query("what is the weather today"),
                         "what is the weather today")


if __name__ == "__main__":
    unittest.main(verbosity=2)

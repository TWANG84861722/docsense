import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestConfig(unittest.TestCase):

    def test_values_are_sensible(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP, CANDIDATE_K, MIN_K, BATCH_SIZE
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreater(CHUNK_OVERLAP, 0)
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)
        self.assertGreater(CANDIDATE_K, MIN_K)
        self.assertGreater(BATCH_SIZE, 0)

    def test_model_paths_exist(self):
        from config import EMBED_MODEL, RERANKER_MODEL
        self.assertTrue(Path(EMBED_MODEL).exists(), f"Embed model missing: {EMBED_MODEL}")
        self.assertTrue(Path(RERANKER_MODEL).exists(), f"Reranker missing: {RERANKER_MODEL}")


class TestBboxOverlap(unittest.TestCase):

    def _overlap(self, a, b):
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def test_left_of(self):
        self.assertFalse(self._overlap((0, 0, 10, 10), (20, 0, 30, 10)))

    def test_right_of(self):
        self.assertFalse(self._overlap((20, 0, 30, 10), (0, 0, 10, 10)))

    def test_above(self):
        self.assertFalse(self._overlap((0, 0, 10, 10), (0, 20, 10, 30)))

    def test_below(self):
        self.assertFalse(self._overlap((0, 20, 10, 30), (0, 0, 10, 10)))

    def test_partial_overlap(self):
        self.assertTrue(self._overlap((0, 0, 20, 20), (10, 10, 30, 30)))

    def test_fully_contained(self):
        self.assertTrue(self._overlap((0, 0, 100, 100), (10, 10, 20, 20)))

    def test_touching_edge_not_overlap(self):
        self.assertFalse(self._overlap((0, 0, 10, 10), (10, 0, 20, 10)))


class TestTableToMarkdown(unittest.TestCase):

    def _to_md(self, rows):
        if not rows:
            return ""
        lines = []
        header = [str(c or "").strip() for c in rows[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            cells = [str(c or "").strip() for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def test_basic(self):
        rows = [["Gene", "Score"], ["TP53", "0.87"], ["BRCA1", "0.92"]]
        result = self._to_md(rows)
        self.assertIn("| Gene | Score |", result)
        self.assertIn("| --- | --- |", result)
        self.assertIn("| TP53 | 0.87 |", result)

    def test_empty(self):
        self.assertEqual(self._to_md([]), "")

    def test_none_cells(self):
        rows = [["Gene", None], ["TP53", None]]
        result = self._to_md(rows)
        self.assertIn("| Gene |  |", result)


class TestSectionRE(unittest.TestCase):

    SECTION_RE = re.compile(
        r"^\d{0,2}\.?\s*"
        r"(abstract|introduction|background|related\s+work|"
        r"methods?|materials?\s*(and\s+)?methods?|experimental|"
        r"results?(\s+and\s+discussion)?|discussion|conclusions?|"
        r"acknowledgements?|references?|supplementary|appendix|"
        r"funding|ethics|data\s+availability)\b",
        re.IGNORECASE
    )

    def test_matches(self):
        for s in ["Introduction", "2. Methods", "RESULTS", "Discussion",
                  "Supplementary", "References", "1. Abstract"]:
            with self.subTest(s=s):
                self.assertTrue(self.SECTION_RE.match(s))

    def test_non_matches(self):
        for s in ["The results showed", "In this study", "Figure 1"]:
            with self.subTest(s=s):
                self.assertFalse(self.SECTION_RE.match(s))


class TestFigureRE(unittest.TestCase):

    FIGURE_RE = re.compile(
        r'^((?:Extended\s+Data\s+|Supplementary\s+)?Fig(?:ure)?\.?\s*\d+[A-Za-z]?)\b',
        re.IGNORECASE
    )

    def test_matches(self):
        for s in ["Figure 1", "Fig. 2A", "Supplementary Figure 3",
                  "Extended Data Figure 1", "Fig 4B"]:
            with self.subTest(s=s):
                self.assertTrue(self.FIGURE_RE.match(s))

    def test_non_matches(self):
        for s in ["The figure shows", "In Figure", "Table 1"]:
            with self.subTest(s=s):
                self.assertFalse(self.FIGURE_RE.match(s))


class TestHGNC(unittest.TestCase):

    def setUp(self):
        from config import DB_DIR
        self.cache_exists = (DB_DIR / "gene_aliases.json").exists()

    def test_known_gene_expanded(self):
        if not self.cache_exists:
            self.skipTest("Gene alias cache not built yet — run chat.py once first")
        from hgnc import expand_query
        result = expand_query("What does TP53 do in apoptosis?")
        self.assertIn("Gene aliases", result)
        self.assertIn("TP53", result)

    def test_plain_text_unchanged(self):
        if not self.cache_exists:
            self.skipTest("Gene alias cache not built yet")
        from hgnc import expand_query
        text = "what is the weather today"
        self.assertEqual(expand_query(text), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

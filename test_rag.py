"""单元测试：全部 import【真实代码】来测（不在测试里复制逻辑）。

核心原则：测试的意义是守护真代码。若在测试里复制一份逻辑来测，
真代码改坏了测试却照样绿灯 = 假的安全感。所以一律 import 真的函数/正则。

这些是"快单元测试"：不加载大模型、不联网，秒级跑完。
（涉及 VL/嵌入/检索的端到端验证属于"慢集成测试"，成本高，另做/另放。）

运行：  python test_rag.py       或      pytest
"""
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
        # 注：这是"环境检查"，依赖本机模型路径；换机器没下模型会失败（符合预期）。
        from config import EMBED_MODEL, RERANKER_MODEL
        self.assertTrue(Path(EMBED_MODEL).exists(), f"Embed model missing: {EMBED_MODEL}")
        self.assertTrue(Path(RERANKER_MODEL).exists(), f"Reranker missing: {RERANKER_MODEL}")


class TestBboxOverlap(unittest.TestCase):
    """测真实的 loaders.pdf.bbox_overlap。"""

    def test_separated(self):
        from loaders.pdf import bbox_overlap
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (20, 0, 30, 10)))    # 左右分开
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (0, 20, 10, 30)))    # 上下分开

    def test_overlap(self):
        from loaders.pdf import bbox_overlap
        self.assertTrue(bbox_overlap((0, 0, 20, 20), (10, 10, 30, 30)))    # 部分重叠
        self.assertTrue(bbox_overlap((0, 0, 100, 100), (10, 10, 20, 20)))  # 完全包含

    def test_touching_edge_not_overlap(self):
        from loaders.pdf import bbox_overlap
        self.assertFalse(bbox_overlap((0, 0, 10, 10), (10, 0, 20, 10)))    # 仅贴边不算重叠


class TestTableToMarkdown(unittest.TestCase):
    """测真实的 loaders.pdf.table_to_markdown。

    它接收一个"表对象"(需有 .extract() 返回二维列表)，所以用一个【假表 stub】喂给它——
    这就是"测试替身"：既跑到了真函数，又不依赖真实 PDF。
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
    """测真实的 loaders.pdf.SECTION_RE。"""

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
    """测真实的 loaders.pdf.FIGURE_RE。"""

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
    """基因别名扩展。用 hgnc 真实的 CACHE_FILE 路径判断缓存在不在（不再猜目录）。"""

    def setUp(self):
        from hgnc import CACHE_FILE
        if not CACHE_FILE.exists():
            self.skipTest(f"基因缓存不存在({CACHE_FILE})——放好 gene_aliases.json 再测")

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

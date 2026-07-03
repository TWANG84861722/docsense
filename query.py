import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from hgnc import expand_query

import config
import embedder
from config import (
    DB_DIR, RERANKER_MODEL,
    CANDIDATE_K,
)

logger = logging.getLogger(__name__)

# ----------------------------
# Load models and index
# ----------------------------

index_path    = DB_DIR / "index.faiss"
metadata_path = DB_DIR / "metadata.json"

if not index_path.exists():
    raise FileNotFoundError(f"Index not found at {index_path}. Run ingest.py first.")
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run ingest.py first.")

logger.info(f"Embedder backend: {embedder.backend()}")

logger.info("Loading reranker...")
# max_length=512：把每个 (query, doc) 对截断到 512 token。交叉编码器的标准做法，
# 否则遇到超长 chunk(整页扫描/大图描述几千 token)，注意力矩阵 ~序列长² 会爆到几十 GiB、
# 在 MPS/GPU 上直接 OOM（"Invalid buffer size"）。截断不影响排序质量。
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

logger.info("Loading FAISS index...")
index = faiss.read_index(str(index_path))

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ----------------------------
# 小工具：文档标题 / 文本拼装 / 分词
# ----------------------------

def _doc_title(paper):
    return Path(paper).stem.replace("_", " ").replace("-", " ").strip()


def _tok(text):
    """简单分词：小写 ASCII 词 + 单个中文字。
    （英文/基因名按词；中文按单字切——BM25 够用，避免漏掉中文。想更精确可上 jieba。）"""
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


def _bm25_text(c):
    """BM25 索引用：标题 + section + 正文 → 这样能按"入职/离职"等标题词区分文档。
    （IDF 会自动给常见词低权、罕见词高权，所以同质语料的标题词无害。）"""
    return f"{_doc_title(c['paper'])} {c.get('section', '')} {c['text']}"


def _rerank_text(c):
    """rerank 输入用：给正文带上「标题 + section」上下文，帮 cross-encoder 区分近乎相同的段。"""
    sec = c.get("section", "")
    head = f"[{_doc_title(c['paper'])}]" + (f" [{sec}]" if sec else "")
    return f"{head}\n{c['text']}"


logger.info("Building BM25 index...")
bm25 = BM25Okapi([_tok(_bm25_text(c)) for c in metadata])

logger.info(f"Index ready — {index.ntotal} vectors, {len(metadata)} chunks")


# ----------------------------
# 混合检索：FAISS + BM25 → 完整并集 → rerank（逐轮加深）
# ----------------------------

MAX_ROUNDS = 5   # 最多加深几轮（每轮两路各再取 CANDIDATE_K）。安全上限，防止无限翻页。


def _ranked_indices(question, depth):
    """算出 FAISS 和 BM25 各自的前 depth 名 chunk 下标（各按自己的相关度排序）。"""
    expanded = expand_query(question)
    qv = embedder.embed([expanded])
    faiss.normalize_L2(qv)
    _, I = index.search(qv, depth)
    faiss_ranked = [int(i) for i in I[0] if i >= 0]
    bm25_scores = bm25.get_scores(_tok(expanded))
    bm25_ranked = [int(i) for i in np.argsort(bm25_scores)[::-1][:depth]]
    return faiss_ranked, bm25_ranked


def retrieve_rounds(question, k=None):
    """逐轮产出候选（生成器）。上层用 next() 决定要不要再加深一轮。

    第 r 轮 = FAISS[r*k:(r+1)*k] ∪ BM25[r*k:(r+1)*k] 的【完整并集】(去重，含跨轮去重)，
    整批 rerank 后按相关度产出（最多 2*k 个/轮）。
    不做 RRF 截断——两路各取的 k 个一个不扔，全交给 rerank 排序，这样 BM25(或 FAISS)
    深处的相关 chunk 不会被"融合后只取前 K"挤掉。
    """
    if k is None:
        k = CANDIDATE_K
    faiss_ranked, bm25_ranked = _ranked_indices(question, k * MAX_ROUNDS)
    seen = set()
    for r in range(MAX_ROUNDS):
        lo, hi = r * k, (r + 1) * k
        idxs = []
        for i in faiss_ranked[lo:hi] + bm25_ranked[lo:hi]:   # 两路本段的并集
            if i not in seen:                                 # 轮内 + 跨轮去重
                seen.add(i)
                idxs.append(i)
        if not idxs:
            return
        candidates = [metadata[i] for i in idxs]
        pairs = [[question, _rerank_text(c)] for c in candidates]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        logger.info(f"retrieve round {r + 1}: {len(ranked)} candidates (FAISS+BM25 union, reranked)")
        yield [{"rerank_score": float(s), **c} for s, c in ranked]

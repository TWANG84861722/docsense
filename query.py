import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from hgnc import expand_query

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
# max_length=512: truncate every (query, doc) pair to 512 tokens. Standard practice for a
# cross-encoder -- otherwise a very long chunk (a full scanned page / a big figure description
# of a few thousand tokens) blows the attention matrix (~ sequence_length²) up to tens of GiB
# and OOMs straight away on MPS/GPU ("Invalid buffer size"). Truncation does not hurt ranking
# quality.
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

logger.info("Loading FAISS index...")
index = faiss.read_index(str(index_path))

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ----------------------------
# Small helpers: document title / text assembly / tokenization
# ----------------------------

def _doc_title(paper):
    return Path(paper).stem.replace("_", " ").replace("-", " ").strip()


def _tok(text):
    """Simple tokenizer: lowercase ASCII words + individual CJK characters.
    (English / gene names split by word; CJK split per character -- good enough for BM25 and
    avoids dropping CJK text. For higher precision on CJK, swap in jieba.)"""
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


def _bm25_text(c):
    """For the BM25 index: title + section + body → lets it distinguish documents by title
    words (e.g. "onboarding" vs "offboarding").
    (IDF automatically down-weights common words and up-weights rare ones, so title words in a
    homogeneous corpus do no harm.)"""
    return f"{_doc_title(c['paper'])} {c.get('section', '')} {c['text']}"


def _rerank_text(c):
    """For the rerank input: prepend "title + section" context to the body, helping the
    cross-encoder tell nearly identical passages apart."""
    sec = c.get("section", "")
    head = f"[{_doc_title(c['paper'])}]" + (f" [{sec}]" if sec else "")
    return f"{head}\n{c['text']}"


logger.info("Building BM25 index...")
bm25 = BM25Okapi([_tok(_bm25_text(c)) for c in metadata])

logger.info(f"Index ready — {index.ntotal} vectors, {len(metadata)} chunks")


# ----------------------------
# Hybrid retrieval: FAISS + BM25 → full union → rerank (deepening round by round)
# ----------------------------

MAX_ROUNDS = 5   # Max number of deepening rounds (each round takes another CANDIDATE_K from
                 # both routes). A safety cap to prevent paging forever.


def _ranked_indices(question, depth):
    """Compute FAISS's and BM25's top-`depth` chunk indices (each ordered by its own relevance)."""
    expanded = expand_query(question)
    qv = embedder.embed([expanded])
    faiss.normalize_L2(qv)
    _, I = index.search(qv, depth)
    faiss_ranked = [int(i) for i in I[0] if i >= 0]
    bm25_scores = bm25.get_scores(_tok(expanded))
    bm25_ranked = [int(i) for i in np.argsort(bm25_scores)[::-1][:depth]]
    return faiss_ranked, bm25_ranked


def retrieve_rounds(question, k=None):
    """Yield candidates one round at a time (a generator). The caller uses next() to decide
    whether to deepen by another round.

    Round r = the *full union* of FAISS[r*k:(r+1)*k] ∪ BM25[r*k:(r+1)*k] (deduplicated,
    including across rounds), reranked as one batch and yielded by relevance (at most 2*k
    per round).
    No RRF truncation -- not one of the k items from either route is thrown away; they all go
    to the reranker to be ordered, so a relevant chunk sitting deep in BM25 (or FAISS) is not
    squeezed out by a "fuse then keep only top K" step.
    """
    if k is None:
        k = CANDIDATE_K
    faiss_ranked, bm25_ranked = _ranked_indices(question, k * MAX_ROUNDS)
    seen = set()
    for r in range(MAX_ROUNDS):
        lo, hi = r * k, (r + 1) * k
        idxs = []
        for i in faiss_ranked[lo:hi] + bm25_ranked[lo:hi]:   # union of this slice from both routes
            if i not in seen:                                 # dedup within-round + across-round
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

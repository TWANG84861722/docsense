"""Embedding-backend abstraction -- cross-platform in one place.

Embedding model (bge-m3):
- Apple Silicon → use **MLX** (native to Apple, saturates the GPU/ANE, fast and power-efficient);
- Windows / Linux / GPU server → use **sentence-transformers** (torch; uses the GPU when CUDA is
  available, otherwise CPU).

Both backends load the **same bge-m3** and output **same-dimension vectors**, so they are
interchangeable. The backend is auto-selected by platform by default (config.EMBED_BACKEND="auto"),
or forced to "mlx" / "st" via the EMBED_BACKEND environment variable.

It exposes a single function: embed(texts) -> np.ndarray(float32, **not normalized**).
Normalization (faiss.normalize_L2) is still done by the caller, keeping the original logic unchanged.

Note: after switching backend, it is best to re-run ingest on the target machine to rebuild the
index (building and querying must use the same backend for the vectors to line up).
"""
from __future__ import annotations

import logging
import platform

import numpy as np

import config

logger = logging.getLogger(__name__)

_backend = None      # "mlx" | "st"
_state = None        # mlx: (model, tokenizer); st: a SentenceTransformer instance


def _pick_backend() -> str:
    b = (config.EMBED_BACKEND or "auto").lower()
    if b != "auto":
        return b
    # Auto: Apple Silicon → mlx; everything else → sentence-transformers (torch)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "st"


def _ensure_loaded():
    global _backend, _state
    if _state is not None:
        return
    _backend = _pick_backend()
    if _backend == "mlx":
        from mlx_embeddings import load                     # only installed on Apple
        logger.info(f"Loading embedder: MLX / {config.EMBED_MODEL}")
        _state = load(config.EMBED_MODEL)                   # (model, tokenizer)
    else:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedder: sentence-transformers / {config.EMBED_MODEL}")
        _state = SentenceTransformer(config.EMBED_MODEL)    # torch, auto-picks CUDA/CPU


def embed(texts: list[str]) -> np.ndarray:
    """A batch of texts → float32 vector matrix (not normalized). Both backends output the
    same bge-m3 vectors."""
    _ensure_loaded()
    if _backend == "mlx":
        from mlx_embeddings import generate
        model, tokenizer = _state
        out = generate(model, tokenizer, texts)
        return np.array(out.text_embeds, dtype=np.float32)
    vecs = _state.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(vecs, dtype=np.float32)


def backend() -> str:
    """Return the name of the backend currently in effect (for debugging)."""
    _ensure_loaded()
    return _backend

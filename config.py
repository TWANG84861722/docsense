import logging
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load the project-root .env (DASHSCOPE_API_KEY / OPENAI_API_KEY, etc.) so that
# running ingest.py / chat.py never needs a manual `source`.
load_dotenv()

# ── Paths ──────────────────────────────────────────────────
DATA_DIR = Path("data")
# DB_DIR is named automatically below based on CHUNK_SIZE (see the Ingestion section).

# ── Local models (embedding + reranker; portable across platforms) ──
# Resolution priority: environment variable > an existing local path > HuggingFace
# model id (auto-downloaded).
# → On this Mac we keep using the local models under ~/models (no re-download);
#   on Windows/a server there is no local path → they are pulled from HF automatically.
def _resolve_model(env_name, local_default, hf_id):
    v = os.environ.get(env_name)
    if v:
        return v
    return local_default if Path(local_default).exists() else hf_id

EMBED_MODEL    = _resolve_model("EMBED_MODEL",    "/Users/taowang/models/bge-m3",              "BAAI/bge-m3")
RERANKER_MODEL = _resolve_model("RERANKER_MODEL", "/Users/taowang/models/bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3")

# Embedding backend: auto (Apple Silicon → MLX, everything else → sentence-transformers/torch)
# | mlx | st. The EMBED_BACKEND environment variable overrides this.
EMBED_BACKEND  = os.environ.get("EMBED_BACKEND", "auto")

# ── LLM / VL provider (swappable) ──────────────────────────
# Switch models in one place: edit `active` in models.yaml (openai/claude/gemini/qwen/local).
# Here we only read the yaml in; the actual calls live in model_client.py.
MODELS = yaml.safe_load((Path(__file__).parent / "models.yaml").read_text(encoding="utf-8"))

def _active(role: str) -> dict:
    """Return the provider config currently in effect for a role ('chat' or 'vl')."""
    override = MODELS.get("llm_override") if role == "chat" else MODELS.get("vl_override")
    name = override or MODELS["active"]
    return name, MODELS["providers"][name]

def ocr_model() -> str:
    """Name of the OCR model used to 'transcribe' scanned pages / tables.

    If the current VL provider configured an ocr_model (e.g. qwen's qwen-vl-ocr) we use it;
    otherwise we fall back to that provider's vl_model -- so switching to another provider
    (openai/claude...) never errors out for a missing ocr_model, and "swap provider in one
    place" still holds. Image *understanding* (describe_figure) does not go through here;
    it still uses vl_model.
    """
    _name, spec = _active("vl")
    return spec.get("ocr_model") or spec["vl_model"]

# ── Ingestion ───────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 150
BATCH_SIZE    = 64

# The index folder is separated automatically by chunk_size: 500 → index_chunk500/,
# 1000 → index_chunk1000/. Changing CHUNK_SIZE alone decides both "how big to split" and
# "which index to read/write". vl_cache/ is not here -- it is shared across all versions.
DB_DIR = Path(f"index_chunk{CHUNK_SIZE}")

# ── Retrieval ───────────────────────────────────────────────
CANDIDATE_K = 100      # How many each route (FAISS/BM25) takes per round → union then rerank;
                       # also the "keep going" threshold for fetching another round.

# ── Chat ────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
MAX_TOKENS        = 800

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

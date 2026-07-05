import bisect
import hashlib
import logging
import json

import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR, DB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
)
import embedder
import loaders

logger = logging.getLogger(__name__)

# ----------------------------
# Chunker (module-level: build_chunks needs it)
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ".", ""],
    add_start_index=True,
)

# ----------------------------
# Format dispatch table: file extension → parser function.
# To add a new format: implement parse_xxx in loaders, then add one line here;
# main() does not change.
# ----------------------------

PARSERS = {
    ".pdf":  loaders.parse_pdf,
    ".txt":  loaders.parse_txt,
    ".docx": loaders.parse_docx,
    ".pptx": loaders.parse_pptx,
    ".xlsx": loaders.parse_xlsx,
    ".xls":  loaders.parse_xlsx,
    ".csv":  loaders.parse_csv,
}


# ════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════

def main():
    """Scan data/ → parse each file → chunk → embed → build FAISS index → save to disk."""
    DB_DIR.mkdir(exist_ok=True)

    logger.info(f"Embedder backend: {embedder.backend()}")   # triggers load + prints mlx/st

    # Load an existing index / metadata (supports resuming an interrupted run).
    index_path    = DB_DIR / "index.faiss"
    metadata_path = DB_DIR / "metadata.json"
    if index_path.exists() and metadata_path.exists():
        logger.info("Loading existing index and metadata...")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
        processed_papers = {c["paper"] for c in all_chunks}
        logger.info(f"Resuming — {len(processed_papers)} files already done, {len(all_chunks)} chunks loaded")
    else:
        index = None
        all_chunks = []
        processed_papers = set()

    files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    if not files:
        logger.warning(f"No files found in {DATA_DIR}")

    for file_path in files:
        parser = PARSERS.get(file_path.suffix.lower())
        if parser is None:
            # Only warn for files with a real extension that are not hidden
            # (so .DS_Store / hidden files don't spam the log).
            if file_path.suffix and not file_path.name.startswith("."):
                logger.info(f"Skipping unsupported format: {file_path.name}")
            continue

        if file_path.name in processed_papers:
            logger.info(f"Skip {file_path.name} (already processed)")
            continue

        logger.info(f"Processing {file_path.name}")
        try:
            elements = parser(file_path)
        except Exception as e:
            logger.error(f"Parse failed for {file_path.name}: {e}")
            continue

        paper_chunks = build_chunks(file_path.name, elements)
        if not paper_chunks:
            logger.warning(f"No chunks extracted from {file_path.name}, skipping")
            continue

        # ── Embed ────────────────────────────────────────────
        logger.info(f"  Embedding {len(paper_chunks)} chunks...")
        # Embed the raw body text only: this keeps vector contrast clean (titles/section names
        # are topic words; stamping them onto every chunk would flatten the contrast).
        # "Distinguishing by document/section" is left to BM25 + rerank (they can carry the
        # titles without hurting contrast).
        texts = [c["text"] for c in paper_chunks]
        vecs_list = []
        for i in range(0, len(texts), BATCH_SIZE):
            vecs_list.append(embedder.embed(texts[i : i + BATCH_SIZE]))
        vecs = np.concatenate(vecs_list, axis=0)
        faiss.normalize_L2(vecs)

        # ── Add to index ─────────────────────────────────────
        if index is None:
            index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        all_chunks.extend(paper_chunks)

        # ── Save immediately ─────────────────────────────────
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved — {len(paper_chunks)} chunks | total {len(all_chunks)} | index {index.ntotal} vectors")

    logger.info(f"Done. Total chunks: {len(all_chunks)}")


# ════════════════════════════════════════════════════════════
#  Parts used by main()
# ════════════════════════════════════════════════════════════

def build_chunks(paper, elements):
    """elements → final chunks.

    table / figure elements are finished products, kept as-is; text elements are first
    concatenated and then split by chunk_size, using character offsets to map each chunk
    back to the page / section it came from.
    Finally, each chunk gets a doc_id / chunk_id (stable identity, handy for citing,
    deduplicating, and future incremental updates).
    """
    doc_id = hashlib.md5(paper.encode("utf-8")).hexdigest()[:10]   # doc id = hash of filename (compact & stable)
    chunks = []
    text_stream = []
    for el in elements:
        if el["type"] == "text":
            text_stream.append((el["page"], el["section"], el["text"]))
        else:  # table / figure: finished product, keep as-is with the source filename attached
            chunks.append({"paper": paper, **el})

    if text_stream:
        combined = ""
        boundaries = []
        for pn, sec, txt in text_stream:
            boundaries.append((len(combined), pn, sec))
            combined += txt + " "
        offsets = [b[0] for b in boundaries]

        for doc_chunk in splitter.create_documents([combined]):
            start = doc_chunk.metadata.get("start_index", 0)
            idx = bisect.bisect_right(offsets, start) - 1
            _, pn, sec = boundaries[max(0, idx)]
            chunks.append({
                "paper": paper,
                "page": pn,
                "section": sec,
                "type": "text",
                "text": doc_chunk.page_content,
            })

    for i, c in enumerate(chunks):
        c["doc_id"] = doc_id
        c["chunk_id"] = f"{doc_id}#{i:04d}"
    return chunks


if __name__ == "__main__":
    main()

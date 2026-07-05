"""Plain-text (.txt) parsing."""
from pathlib import Path


def parse_txt(path):
    """Parse plain text → a single text element (the whole file, left for downstream chunking)."""
    # Path(path).read_text(...)  Takes: a file path   Returns: the whole file's text (a string)
    #   encoding="utf-8" reads as UTF-8; errors="ignore" skips any undecodable bytes (so one bad
    #   byte doesn't crash the whole file); .strip() removes leading/trailing whitespace
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:                # empty file → return an empty list
        return []
    # Treat the whole file as one text element (page fixed at 1, section left empty); if it is
    # long, the downstream will split it by chunk_size.
    return [{"page": 1, "section": "", "type": "text", "text": text}]

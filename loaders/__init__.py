"""Parsing layer (the loaders package) -- turns a "file" into a uniform list of elements.

Every parse_xxx returns the same elements list:
  element = {"page": int, "section": str,
             "type": "text" | "table" | "figure", "text": str}
  · text elements   = raw body text, left for the downstream (ingest) to split by chunk_size
  · table/figure elements = finished products, kept as-is
  · no "paper" field (the source filename is stamped on later, uniformly, by ingest)

To add a new format: create loaders/xxx.py with a parse_xxx, import and expose it here, then
register one line in ingest.py's PARSERS. The downstream main loop does not change.
"""
# Lift each in-package module's parse_* (.pdf / .txt / ...) up to the package top level, so that
# after `import loaders` you can call loaders.parse_pdf(...) directly without caring which file
# it lives in. The leading dot = "within this package" (relative import): .pdf means
# loaders/pdf.py, not some installed third-party library.
from .pdf import parse_pdf
from .txt import parse_txt
from .docx import parse_docx
from .pptx import parse_pptx
from .excel import parse_xlsx, parse_csv

# __all__: declares the names this package makes public (also the export list for
# `from loaders import *`).
__all__ = ["parse_pdf", "parse_txt", "parse_docx", "parse_pptx", "parse_xlsx", "parse_csv"]

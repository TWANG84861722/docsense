"""Small helpers shared across formats. Reused by the docx / pptx / excel / csv parsers to
avoid duplication (DRY)."""
import re

# Table caption/label regex: matches text starting with "Table 1" / "Table 1:" / "Tab. 2" / "表 1".
#   ^\s*                 leading spaces allowed
#   (?:Table|Tab\.?|表)  accepts all three spellings (the dot after "Tab" is optional)
#   \s*\d+               followed by a number
#   [A-Za-z]?            allows "Table 1A" with a trailing letter
TABLE_RE = re.compile(r'^\s*(?:Table|Tab\.?|表)\s*\d+[A-Za-z]?\b', re.IGNORECASE)


def is_table_caption(text):
    # Takes: a piece of text (string)   Returns: True/False -- does it start with "Table N" / "表 N"
    # `text or ""`: guard against None so .match doesn't blow up
    return bool(TABLE_RE.match(text or ""))


def table_label(text):
    # Takes: a piece of text   Returns: the table label string (e.g. "Table 2"), or None if not a caption
    # Used to deduplicate "have we already grabbed this table"
    m = TABLE_RE.match(text or "")
    return m.group(0).strip() if m else None


def _rows_to_md(rows):
    # Takes: a 2-D list `rows` (a list of rows, each row a list of cells)
    # Returns: a single Markdown table string
    # Step 1: clean each cell (None→"", newline→space, strip whitespace); shape unchanged, still 2-D
    rows = [[str(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)            # how many cells the widest row has (all rows align to this)
    pad = lambda r: r + [""] * (width - len(r))  # tiny helper: pad a row with empty cells up to `width`
    # Header row + the separator line Markdown requires (| --- | --- | ...)
    lines = ["| " + " | ".join(pad(rows[0])) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:                           # remaining rows: pad → join into "| a | b | c |"
        lines.append("| " + " | ".join(pad(r)) + " |")
    return "\n".join(lines)                       # join rows with newlines → the full Markdown table

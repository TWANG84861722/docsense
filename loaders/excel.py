"""Excel (.xlsx/.xls) and CSV parsing (both are tables, handled with pandas)."""
from .common import _rows_to_md
from .office_images import image_elements


def _df_to_md(df):
    # Takes: a pandas DataFrame (a table)   Returns: a Markdown table string
    df = df.fillna("")                                          # NaN→"" (NaN is a "real value", so _rows_to_md's `or ""` won't catch it; fill first)
    rows = [list(df.columns)] + df.astype(str).values.tolist()  # header row + each data row (all as strings), into a 2-D list
    return _rows_to_md(rows)                                    # hand to the shared helper to build Markdown


def parse_xlsx(path):
    """Parse Excel (.xlsx/.xls) → elements. Each sheet → one text element (a Markdown table).

    Uses a text element (not a table element) because data tables can be large; leaving it for
    downstream chunking avoids a single chunk exceeding the embedding length.
    Note: pure data tables (mostly numbers) have limited semantic-search value and suit exact
    lookups better; this is just a generic ingest path.
    """
    import pandas as pd                                  # lazy import

    # pd.read_excel(..., sheet_name=None)  Takes: a path   Returns: a dict {sheet_name: DataFrame} (None = read all sheets)
    sheets = pd.read_excel(str(path), sheet_name=None)
    elements = []
    # enumerate(sheets.items(), start=1)  Returns: each (index i, (sheet name, DataFrame df))
    for i, (name, df) in enumerate(sheets.items(), start=1):
        md = _df_to_md(df)                               # this sheet → a Markdown table
        if md:
            # section = the sheet name (a natural title for this table)
            elements.append({"page": i, "section": name, "type": "text", "text": md})
    elements.extend(image_elements(path, "Sheet image"))   # images embedded in the workbook → VL understanding → figure
    return elements


def parse_csv(path):
    """Parse CSV → a single text element (a Markdown table; downstream will chunk it)."""
    import pandas as pd

    df = pd.read_csv(str(path))                          # Takes: a path   Returns: one DataFrame
    md = _df_to_md(df)                                   # → a Markdown table
    return [{"page": 1, "section": "", "type": "text", "text": md}] if md else []

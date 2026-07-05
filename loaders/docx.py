"""Word (.docx) parsing."""
from .common import _rows_to_md, is_table_caption
from .office_images import image_elements


def _docx_table_md(table):
    # Takes: a docx Table object        Returns: a Markdown table string
    # How: pull each row/cell's .text into a 2-D list → hand it to the shared _rows_to_md
    return _rows_to_md([[c.text for c in row.cells] for row in table.rows])


def parse_docx(path):
    """Parse Word (.docx) → elements.
    Paragraphs → text (heading styles become the section); tables → table (Markdown);
    embedded images → VL image understanding (image_elements).
    """
    from docx import Document
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    doc = Document(str(path))   # Takes: a file path (string)   Returns: the whole Word document object
    elements = []               # the element list we will return
    section = ""                # the current section (updated when we hit a heading)
    buf = []                    # buffers "body lines not yet filed"

    def flush():
        # Takes: nothing (uses buf / section / elements from the closure)
        # Returns: nothing (side effect: turn buffered text into one text element in `elements`, then clear buf)
        text = " ".join(buf).strip()
        if text:
            elements.append({"page": 1, "section": section, "type": "text", "text": text})
        buf.clear()

    # doc.element.body.iterchildren()
    #   Takes: nothing   Returns: yields each "child node" in the body (a paragraph/table's raw XML)
    #          in the document's original order
    for child in doc.element.body.iterchildren():

        # isinstance(child, CT_P)  Takes: (object, type)  Returns: True/False -- is this child a "paragraph"
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)   # Takes: (paragraph XML node, owning document)  Returns: a usable Paragraph object
            txt = para.text.strip()        # para.text → paragraph plain text (string); .strip() → trim whitespace
            if not txt:                    # empty paragraph → skip
                continue
            # para.style.name → the style name used (string, e.g. "Heading 1"/"Normal"); .lower() for easy comparison
            style = (para.style.name or "").lower() if para.style else ""
            if style.startswith("heading") or style == "title":
                flush()                    # first file the body buffered above
                section = txt              # this paragraph is a heading → start a new section
            else:
                buf.append(txt)            # ordinary body → buffer it, filed on the next flush

        # isinstance(child, CT_Tbl) → is this child a "table"
        elif isinstance(child, CT_Tbl):
            # A table caption is usually *above* the table: if buf's last line is "Table N...",
            # pop it out to use as the caption and bind it to this table.
            caption = buf.pop() if (buf and is_table_caption(buf[-1])) else ""
            flush()                                  # file the remaining body
            md = _docx_table_md(Table(child, doc))   # Table(...) → table object; _docx_table_md(...) → Markdown string
            if md:
                text = f"{caption}\n{md}" if caption else md   # if there's a caption, prepend it so it stays in the same chunk
                elements.append({"page": 1, "section": section, "type": "table", "text": text})

    flush()           # loop done: file the last buffered body too
    elements.extend(image_elements(path, "Figure"))   # embedded images → VL understanding → figure elements
    return elements   # Returns: the element list (text / table / figure)

"""PDF parsing: parse_pdf (the main entry, at the top) + the parts it uses (below).

Read top-down (high level first, details later, i.e. a stepdown / newspaper structure):
  parse_pdf(one PDF)          ← main entry: page-by-page segmentation + figure/table extraction
                                + image understanding, all merged into elements
     ↓ uses
  page_segments               ← cut a page by layout into (text/table, section, text)
  _bordered_tables            ← find_tables finds bordered tables (shared by page_segments/parse_pdf)
  parse_scanned_page          ← scanned page (no text layer): render the whole page → VL → markdown
  extract_figure_captions     ← find the "Figure N" captions
  describe_figure/_fullpage   ← render the image → VL understanding (Tier 0 crop above the caption /
                                fallback finds the figure on the whole page)
  extract_table_captions      ← find the "Table N" captions
  extract_table_via_vl        ← borderless table: VL reads it into Markdown from below/above the
                                caption (_table_region / _vl_read_table)
  table_to_markdown           ← a bordered table object → Markdown
  bbox_overlap / is_references ← small helpers
"""
from collections import Counter
import logging
import re

import fitz

import config
import model_client
from .common import is_table_caption, table_label, _rows_to_md

logger = logging.getLogger(__name__)

fitz.TOOLS.mupdf_display_errors(False)   # silence PyMuPDF's error spam (stay quiet on broken pages)

# Section-heading regex: matches common paper section names like "Introduction" / "2. Methods" /
# "Results and Discussion".
SECTION_RE = re.compile(
    r"^\d{0,2}\.?\s*"
    r"(abstract|introduction|background|related\s+work|"
    r"methods?|materials?\s*(and\s+)?methods?|experimental|"
    r"results?(\s+and\s+discussion)?|discussion|conclusions?|"
    r"acknowledgements?|references?|supplementary|appendix|"
    r"funding|ethics|data\s+availability)\b",
    re.IGNORECASE
)

# Figure-caption regex: matches openings like "Figure 1" / "Fig. 2A" / "Extended Data Fig 3".
FIGURE_RE = re.compile(
    r'^((?:Extended\s+Data\s+|Supplementary\s+)?Fig(?:ure)?\.?\s*\d+[A-Za-z]?)\b',
    re.IGNORECASE
)

# References section heading: matches "References" / "1. References" etc.
REFERENCES_RE = re.compile(r'^\d{0,2}\.?\s*references?\b', re.IGNORECASE)

# Scanned-page test: if the text layer has fewer than this many characters → treat the page as
# having no extractable text (scanned/image page) and route it through whole-page VL.
# A normal text page easily has hundreds/thousands of characters; a pure image/scanned page is
# usually 0, so a small threshold separates them safely.
_SCANNED_PAGE_MAX_CHARS = 50




# ════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════

def parse_pdf(path):
    """Parse a single PDF → elements.

    Takes: a PDF file path
    Returns: a list of elements, each {"page","section","type"∈{text,table,figure},"text"}
    """
    try:
        doc = fitz.open(path)                  # open the PDF (on failure, log and return empty)
    except Exception as e:
        logger.error(f"Cannot open {path}: {e}")
        return []

    elements = []
    current_section = ""          # the "current section", carried across pages
    in_references = False         # whether we've entered the references section (drop everything after)

    for page_num, page in enumerate(doc):       # page by page: page_num starts at 0 (+1 for the page number)
        try:
            # Scanned page (no text layer): get_text extracts nothing → hand the whole page to the VL
            # to turn into markdown, skipping the text-layer-based cropping flow. One whole-page VL
            # call yields body + tables + figure descriptions at once -- exactly how NotebookLM handles
            # scanned documents.
            if len(page.get_text().strip()) < _SCANNED_PAGE_MAX_CHARS:
                md = parse_scanned_page(page)
                if md:
                    elements.append({
                        "page": page_num + 1,
                        "section": current_section,   # scanned pages don't track sections; reuse the previous page's
                        "type": "text",
                        "text": md,
                    })
                continue                              # this page is done; skip the cropping flow below

            # This page's figure captions, table captions, and bordered tables (all used below;
            # bordered is reused so find_tables isn't run twice).
            fig_caps = extract_figure_captions(page)
            table_caps = extract_table_captions(page)
            bordered_tables, bordered_bboxes = _bordered_tables(page)

            # Borderless tables (find_tables can't catch them): first use the VL to extract clean
            # Markdown, and record their regions. Recording the region lets page_segments later
            # exclude that block from the body text → avoiding "the table smeared into the body" and
            # a duplicate of the VL version.
            vl_tables = []          # VL tables to add to elements
            table_excludes = []     # borderless-table regions (incl. caption), passed to page_segments to exclude from body
            for label, caption, cap_bbox in table_caps:
                cap_top, cap_bottom = cap_bbox[1], cap_bbox[3]
                # Table region top/bottom bounds = the nearest table/figure caption (below → its top,
                # above → its bottom), otherwise the page top/bottom.
                belows = ([c[2][1] for c in table_caps if c[2][1] > cap_bottom]
                          + [f[2][1] for f in fig_caps if f[2][1] > cap_bottom])
                bottom_y = min(belows) if belows else page.rect.y1
                aboves = ([c[2][3] for c in table_caps if c[2][3] < cap_top]
                          + [f[2][3] for f in fig_caps if f[2][3] < cap_top])
                top_y = max(aboves) if aboves else page.rect.y0
                # Already caught by find_tables (bordered) → page_segments will handle it, so skip
                # (a hit on either side of the caption, above or below, counts).
                below_r = _table_region(page, cap_bottom, bottom_y)
                above_r = _table_region(page, top_y, cap_top)
                if any(bbox_overlap(tuple(below_r), bb) or bbox_overlap(tuple(above_r), bb)
                       for bb in bordered_bboxes):
                    continue
                md, region = extract_table_via_vl(page, cap_bbox, bottom_y, top_y)
                if md:
                    vl_tables.append({
                        "page": page_num + 1,
                        "section": label,
                        "type": "table",
                        "text": f"{caption}\n{md}",   # original caption + the Markdown table the VL extracted
                    })
                    table_excludes.append(tuple(region | fitz.Rect(cap_bbox)))  # exclude the caption along with it

            # Body segmentation: exclude from the body both the bordered-table regions and the
            # borderless-table regions (already grabbed by the VL).
            segs = page_segments(page, current_section,
                                 bordered=(bordered_tables, bordered_bboxes),
                                 exclude_bboxes=table_excludes)
            if segs:
                current_section = segs[-1][1]             # use this page's last segment's section, carried to the next page

            for seg_type, section, seg_text in segs:
                # Once in the References section, stop collecting body/tables (references don't enter the index).
                if is_references(section):
                    in_references = True
                elif section:
                    in_references = False
                if in_references:
                    continue

                elements.append({
                    "page": page_num + 1,
                    "section": section,
                    "type": "table" if seg_type == "table" else "text",
                    "text": seg_text,
                })

            # Captions + image understanding: each "Figure N" caption → render its image → VL
            # description → one figure element. When cropping fails (VL says it sees no figure),
            # escalate through fallbacks tier by tier: whole current page (rescues side-captions /
            # wrong region) → previous page (rescues cross-page) → give up and keep only the caption.
            # The ladder is self-terminating: a rescuable figure stops at the tier that rescues it;
            # anything unrescuable naturally ends up as "broken/no such figure".
            for fig_label, caption, cap_bbox in fig_caps:
                description = describe_figure(page, cap_bbox)                   # Tier 0: crop the block above the caption
                if description and model_client.vl_found_nothing(description):
                    description = None
                if description is None:                                         # Tier 2a: whole current page
                    logger.info(f"  {fig_label} p{page_num+1}: crop found no figure → escalate to whole page (Tier 2a)")
                    description = describe_figure_fullpage(page, caption)
                if description is None and page_num > 0:                        # Tier 2b: previous page (cross-page)
                    logger.info(f"  {fig_label} p{page_num+1}: whole page still empty → try previous page (Tier 2b)")
                    description = describe_figure_fullpage(doc[page_num - 1], caption)
                if description is None:                                         # all tiers failed
                    logger.info(f"  {fig_label} p{page_num+1}: no tier found the figure → keep caption only (likely broken figure)")
                text = f"{description}\n{caption}" if description else caption  # all failed → keep only the caption
                elements.append({
                    "page": page_num + 1,
                    "section": fig_label,        # a figure's section uses the figure label, e.g. "Figure 1"
                    "type": "figure",
                    "text": text,
                })

            # Borderless tables: add the clean Markdown tables already extracted by the VL above
            # (the copy in the body was excluded, so no duplication).
            elements.extend(vl_tables)
        except Exception as e:
            logger.warning(f"  Page {page_num + 1} skipped: {e}")   # a single-page error only skips that page, not the whole doc

    doc.close()
    return elements


# ════════════════════════════════════════════════════════════
#  Parts used by parse_pdf
# ════════════════════════════════════════════════════════════

def page_segments(page, prev_section="", bordered=None, exclude_bboxes=()):
    """Cut a page into (kind, section, text) segments, top to bottom by layout.

    Takes: the page object `page`, the section in effect on entering this page `prev_section`,
           `bordered` (optional (list of table objects, list of bboxes); pass it in if parse_pdf
           already computed it, to avoid re-running find_tables),
           `exclude_bboxes` (optional, extra regions to exclude from the body -- i.e. the
           borderless-table block already grabbed by the VL)
    Returns: a list, each item (kind, section, text), kind ∈ {"text","table"}
             (figures are not here; parse_pdf handles them separately)

    Bordered tables → converted to Markdown as their own segment and excluded from the body;
    borderless tables → parse_pdf extracts them with the VL and passes their region in via
                        exclude_bboxes to remove from the body (otherwise the table's text gets
                        smeared into the body and duplicates the clean VL version).
    """
    # ── 1. Bordered tables (find_tables): reuse if parse_pdf already computed them, else compute ──
    if bordered is None:
        bordered_tables, bordered_bboxes = _bordered_tables(page)
    else:
        bordered_tables, bordered_bboxes = bordered
    # All regions to exclude from the body = bordered tables + borderless tables (already grabbed by VL)
    skip_bboxes = list(bordered_bboxes) + list(exclude_bboxes)

    blocks = page.get_text("dict")["blocks"]   # all "blocks" on the page (text blocks type=0 / image blocks type=1)

    # ── 2. Find the body font size: the most common size is the "body size"; lines bigger than it may be headings ──
    sizes = [
        round(s["size"], 1)
        for b in blocks if b.get("type") == 0                       # text blocks only
        if not any(bbox_overlap(b["bbox"], tb) for tb in skip_bboxes)  # and not inside a table region
        for line in b.get("lines", [])
        for s in line.get("spans", [])
        if s["text"].strip()
    ]

    if not sizes:                       # page has no body text at all (maybe pure image) → return the whole page's text as one segment
        raw = page.get_text()
        return [("text", prev_section, raw)] if raw.strip() else []

    body_size = Counter(sizes).most_common(1)[0][0]   # most frequent size = body font size

    # ── 3. Put "text blocks" and "tables" into one list `items`, sorted by vertical position (y), top to bottom ──
    items = []
    for b in blocks:
        if b.get("type") != 0:                                          # not a text block, skip
            continue
        if any(bbox_overlap(b["bbox"], tb) for tb in skip_bboxes):  # falls inside a table region, skip (tables handled separately)
            continue
        items.append(("text", b["bbox"][1], b))     # ("text", this block's top y, block content)
    for t in bordered_tables:
        items.append(("table", t.bbox[1], t))       # ("table", table's top y, table object)
    items.sort(key=lambda x: x[1])                  # sort by y (top edge) → restore top-to-bottom reading order

    # ── 4. Walk top to bottom, classifying as we go: heading → new section; body → accumulate; table → its own segment ──
    segs = []
    cur_section = prev_section
    cur_text = ""                 # buffers "body not yet filed"
    pending_caption = ""          # the last recognized "Table N..." caption, waiting for the next table to bind (caption above the table)
    last_table = None             # the most recently created table segment, for back-filling a caption when it's below the table

    for kind, _, content in items:

        if kind == "table":
            if cur_text.strip():                                   # before the table, file the buffered body
                segs.append(("text", cur_section, cur_text.strip()))
                cur_text = ""
            md = table_to_markdown(content)
            if md:
                # A caption is usually above the table: prepend the just-stored caption so it stays in the same chunk.
                text = f"{pending_caption}\n{md}" if pending_caption else md
                segs.append(("table", cur_section, text))
                last_table = {"idx": len(segs) - 1, "bbox": content.bbox,
                              "captioned": bool(pending_caption)}   # record this table, for back-fill when caption is below
            pending_caption = ""

        else:   # text block
            lines = content.get("lines", [])
            if lines:
                first_spans = [s["text"] for s in lines[0].get("spans", []) if s["text"].strip()]
                first_line = " ".join(first_spans).strip()         # this block's first line
                if FIGURE_RE.match(first_line):                    # it's a figure caption → skip (figures handled by parse_pdf)
                    continue
                # Is this block itself a table caption ("Table N ...")?
                if is_table_caption(first_line):
                    cap_text = " ".join(                            # the whole caption's text
                        " ".join(s["text"] for s in ln.get("spans", []))
                        for ln in lines
                    ).strip()
                    cap_bbox = content["bbox"]
                    # Caption *below* the table: if there's a just-above table still lacking a caption →
                    # back-fill the caption onto that table (don't drop it into the body).
                    if (last_table and not last_table["captioned"]
                            and 0 <= cap_bbox[1] - last_table["bbox"][3] < 30):
                        i = last_table["idx"]
                        _, sec, tbl_text = segs[i]
                        segs[i] = ("table", sec, f"{cap_text}\n{tbl_text}")
                        last_table["captioned"] = True
                        continue
                    # Otherwise the caption is *above* the table (the common case): store it, to bind to the next table.
                    if cur_text.strip():
                        segs.append(("text", cur_section, cur_text.strip()))
                        cur_text = ""
                    pending_caption = cap_text
                    continue

            # Ordinary text block: if we stored a caption but no table ever came, put it back into the body (don't drop it).
            if pending_caption:
                cur_text += pending_caption + " "
                pending_caption = ""

            # Process line by line: decide whether it's a "section heading", otherwise accumulate into the body.
            for line in lines:
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                line_text = " ".join(s["text"] for s in spans).strip()
                avg_size = sum(s["size"] for s in spans) / len(spans)        # this line's average font size
                is_bold = any(bool(s["flags"] & 16) for s in spans)         # is this line bold (bit 16 of flags = bold)
                is_larger = avg_size > body_size * 1.1                      # more than 10% larger than the body

                # Short + (bold or larger) + matches a section name → judged a "section heading".
                if (
                    len(line_text) < 80
                    and (is_bold or is_larger)
                    and SECTION_RE.match(line_text)
                ):
                    if cur_text.strip():                            # before opening a new section, file the previous section's body
                        segs.append(("text", cur_section, cur_text.strip()))
                    cur_section = line_text                         # switch to the new section
                    cur_text = ""
                else:
                    cur_text += line_text + " "                     # ordinary body → accumulate

    if pending_caption:           # loop ended with a stored caption (no table ever came) → put it back into the body, don't drop it
        cur_text += pending_caption + " "
    if cur_text.strip():          # file the last buffered body too
        segs.append(("text", cur_section, cur_text.strip()))

    return segs


def _bordered_tables(page):
    """This page's "bordered tables": returns (list of table objects, list of bboxes).

    find_tables(strategy="lines") only recognizes tables with vertical lines; filter out broken
    tables whose bbox can't be computed (an empty table makes t.bbox crash → the whole page gets
    skipped and its body is lost). Shared by page_segments and parse_pdf, to avoid detecting twice.
    """
    try:
        found = page.find_tables(strategy="lines").tables
    except Exception:
        found = []
    tables, bboxes = [], []
    for t in found:
        try:
            bbox = t.bbox              # try computing once: keep it only if it computes
        except Exception:
            continue
        tables.append(t)
        bboxes.append(bbox)
    return tables, bboxes


def extract_figure_captions(page):
    # Takes: a page object
    # Returns: a list, each item (figure label, full caption text, the caption block's bbox)
    #          e.g. ("Figure 1", "Figure 1. Editing efficiency...", (x0,y0,x1,y1))
    captions = []
    for block in page.get_text("dict")["blocks"]:   # iterate every "block" on the page
        if block.get("type") != 0:                  # type!=0 not a text block (maybe an image), skip
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        # Take this block's "first line" of text and see if it starts with "Figure N...".
        first_line = " ".join(
            s["text"] for s in lines[0].get("spans", [])
        ).strip()
        m = FIGURE_RE.match(first_line)
        if not m:                                   # not a caption → skip
            continue
        # It is a caption: join all lines of this block into the full caption text.
        full_text = " ".join(
            " ".join(s["text"] for s in line.get("spans", []))
            for line in lines
        ).strip()
        captions.append((m.group(1), full_text, block["bbox"]))   # m.group(1) = the "Figure 1" label
    return captions


def describe_figure(page, caption_bbox):
    """Render the figure image above the caption and return a VL description.

    Takes: the page object `page`, the caption's bbox `caption_bbox`
    Returns: this figure's text description (string), or None on failure

    The image-understanding budget is set generously and uniformly by describe_image (cap 4000):
    billing is by actual generation and the model stops on its own when done, so one generous cap
    suffices; "per-panel description" is driven by the prompt, not by max_tokens.
    """
    cap_top = caption_bbox[1]   # the caption box's "top" y (the figure is above it, so we look for things whose bottom is above this)

    # Find image blocks sitting above the caption
    image_blocks = [
        b for b in page.get_text("dict")["blocks"]  # get_text("dict")["blocks"] splits the page into "blocks"; each has type: 0=text block, 1=image block.
        if b.get("type") == 1 and b["bbox"][3] <= cap_top + 10  # keep: (1) image blocks (type==1) and (2) bottom (bbox[3]) ≤ caption top (cap_top) -- images sitting above the caption. +10 is tolerance.
    ]

    if image_blocks:
        best = max(image_blocks, key=lambda b: b["bbox"][3])
        clip = fitz.Rect(best["bbox"])  # pick the one with the largest bottom -- closest to, and directly above, the caption (most likely this caption's figure). clip = its bbox.
    else:
        # Fall back to the page region above the caption
        page_rect = page.rect
        clip = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, cap_top)
        # Fallback: a vector figure isn't an "image block", so none is found. Box the whole strip from "page top → caption top", since the figure is above the caption anyway.

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    # get_pixmap = render a region of the PDF into a pixel image. clip = only this region; matrix = 2x zoom for sharpness so the VL reads it more accurately.

    prompt = (
        "You are reading a figure from a molecular-biology research paper. "
        "Describe it thoroughly so it can be retrieved later. "
        "If the figure has multiple panels (A, B, C, ...), describe EACH panel "
        "separately: its purpose, what is plotted (axes, conditions/groups), and the "
        "key quantitative result or trend. Identify the figure type (e.g., western blot, "
        "bar graph, microscopy, survival curve). Include numbers, units, and statistical "
        "significance where visible. Do not omit any panel."
    )
    try:
        # pix.tobytes("png") → the image's PNG bytes; hand to the swappable VL (currently qwen-vl-max) for understanding
        return model_client.describe_image(pix.tobytes("png"), prompt)
    except Exception as e:
        logger.warning(f"VL description failed: {e}")
        return None


def parse_scanned_page(page):
    """Scanned page (no text layer): render the whole page → VL turns it into structured Markdown
    (body + tables + figure descriptions).

    Takes: the page object   Returns: a markdown string (None on failure)
    A scanned page's get_text extracts nothing, so we can only "look at the image". A higher
    resolution (2.5x) helps the VL read text. The prompt makes clear "only write [FIGURE] for a
    truly visible image, don't invent from the body" -- preventing hallucinated figure blocks on
    a pure-text page.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    prompt = (
        "You are transcribing a scanned page from a scientific journal article into clean Markdown "
        "so it can be indexed for retrieval:\n"
        "- Transcribe all body text in reading order (handle multi-column layout correctly).\n"
        "- Render any table as a Markdown table with all rows and columns.\n"
        "- For a figure/photo/gel/diagram that is ACTUALLY VISIBLE on this page, add a line starting "
        "with '[FIGURE]' followed by its caption (if any) and a thorough description of what it shows "
        "(figure type, panels, axes/labels, key findings).\n"
        "- Do NOT invent a [FIGURE] block from the text alone — only when a figure is truly present.\n"
        "- Preserve section headings with ##. Do not add commentary that isn't on the page."
    )
    try:
        # A scanned page is a "read the text" job → use the OCR-dedicated model (qwen-vl-ocr, accurate and cheap; falls back to vl_model if unset)
        return model_client.describe_image(pix.tobytes("png"), prompt, model=config.ocr_model())
    except Exception as e:
        logger.warning(f"VL scanned-page parse failed: {e}")
        return None


def describe_figure_fullpage(page, caption):
    """Fallback: render the whole page and let the VL find "the figure whose caption is `caption`"
    on it and describe it.

    Takes: the page object, the figure's full caption text `caption`   Returns: the figure
    description (None if not found / on failure)
    Used after describe_figure's crop fails -- giving the VL a whole-page view + the caption text,
    it can locate the corresponding figure itself (rescues side-captions / wrong region); calling
    it with the previous page's object rescues the cross-page case.
    The prompt gives an explicit "reply NO FIGURE HERE if absent", which together with
    vl_found_nothing triggers the next fallback tier.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    prompt = (
        "This is a full page from a research paper. On this page there should be a figure whose caption is:\n"
        f"\"{caption}\"\n"
        "Find THAT specific figure on the page and describe it thoroughly for retrieval: figure type, "
        "each panel (A, B, C, ...), axes/conditions/groups, and the key quantitative results or trends. "
        "If that figure is NOT visually present anywhere on this page, reply with exactly: NO FIGURE HERE."
    )
    try:
        desc = model_client.describe_image(pix.tobytes("png"), prompt)
    except Exception as e:
        logger.warning(f"VL full-page figure failed: {e}")
        return None
    if desc and model_client.vl_found_nothing(desc):
        return None      # VL says this page has no such figure → let the caller escalate to the next tier (previous page)
    return desc


def extract_table_captions(page):
    # Takes: a page object
    # Returns: a list, each item (table label, full caption, caption block bbox), e.g. ("Table 2", "Table 2. ...", (x0,y0,x1,y1))
    caps = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        first_line = " ".join(s["text"] for s in lines[0].get("spans", [])).strip()
        label = table_label(first_line)        # is the first line "Table N..."
        if not label:
            continue
        full = " ".join(
            " ".join(s["text"] for s in ln.get("spans", []))
            for ln in lines
        ).strip()
        caps.append((label, full, block["bbox"]))
    return caps


def _table_region(page, top_y, bottom_y):
    """The region (fitz.Rect) of the table within the vertical band [top_y, bottom_y].

    Left/right bounds are computed from "the text blocks falling inside that band" (a caption is
    often narrower than the table, so using the caption width would clip the rightmost columns).
    extract_table_via_vl uses it as the render clip; parse_pdf uses it to exclude the block from
    the body -- both share the same algorithm.
    """
    xs0, xs1 = [], []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        if top_y <= by0 < bottom_y:
            xs0.append(bx0)
            xs1.append(bx1)
    left = min(xs0) if xs0 else page.rect.x0     # no text block in the band → fall back to full page width
    right = max(xs1) if xs1 else page.rect.x1
    return fitz.Rect(left, top_y, right, bottom_y)


def _vl_read_table(page, clip):
    """Render a region → OCR-read it into a Markdown table. Returns None if the region has no table."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    prompt = (
        "This image is a region from a scientific paper. If it contains a table, extract it as a "
        "clean GitHub-flavored Markdown table — preserve all rows, columns, headers, numbers and units "
        "exactly, and output ONLY the markdown table (no caption or commentary). "
        "If the region does NOT contain a table (e.g. it is body text or a figure), reply with exactly: NO TABLE."
    )
    try:
        # A table is a "read the text" job → use the OCR-dedicated model (transcribes three-line tables more accurately and cheaply)
        md = model_client.describe_image(pix.tobytes("png"), prompt, model=config.ocr_model())
    except Exception as e:
        logger.warning(f"VL table extraction failed: {e}")
        return None
    if md and model_client.vl_found_nothing(md):
        return None      # VL says this block has no table (fake "Table" heading / wrong direction / it's body text) → skip
    return md


def extract_table_via_vl(page, caption_bbox, bottom_y, top_y):
    """Three-line / borderless tables: the caption may be *below* the table (common) or *above*,
    so try both directions.

    Takes: page, caption bbox, bottom bound bottom_y (next table/figure caption or page bottom),
           top bound top_y (previous table/figure caption or page top)
    Returns: (markdown, the Rect region used); (None, None) if no table can be read either way
    Tries "below the caption" first (the vast majority); if OCR says no table (NO TABLE), tries
    "above the caption" (caption-below-the-table case). We use the VL because three-line tables
    have no vertical lines and find_tables can't catch them; letting the VL "look and read the
    table" bypasses geometric detection.
    """
    cap_top, cap_bottom = caption_bbox[1], caption_bbox[3]
    for top, bot in [(cap_bottom, bottom_y), (top_y, cap_top)]:   # below first, then above
        if bot - top < 10:                       # this side's band is too thin → nothing there, skip
            continue
        region = _table_region(page, top, bot)
        md = _vl_read_table(page, region)
        if md:
            return md, region
    return None, None


def table_to_markdown(table):
    # Takes: a table object detected by PyMuPDF   Returns: a Markdown table string
    # .extract() → a 2-D list; delegated to common._rows_to_md (pads ragged rows, turns in-cell
    # newlines into spaces -- more robust than hand-writing, and shared with the office tables)
    return _rows_to_md(table.extract())


def bbox_overlap(a, b):
    # Takes: two rectangles a, b, each (x0 left, y0 top, x1 right, y1 bottom)
    # Returns: True/False -- whether the two boxes overlap
    # Idea: list the 4 "definitely not overlapping" cases; negate for "overlap".
    #   a[2] <= b[0]: a's right ≤ b's left → a is entirely left of b
    #   a[0] >= b[2]: a is right of b;  a[3] <= b[1]: a is above b;  a[1] >= b[3]: a is below b
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def is_references(section):
    """Takes: a section name (string)  Returns: True/False -- is it a "References" section heading."""
    return bool(REFERENCES_RE.match(section or ""))

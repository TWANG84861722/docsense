"""Unified image understanding for images embedded in Office files (shared by docx / pptx / xlsx).

All three formats are zip archives, and their images live in the archive's media/ folder
(word/media, ppt/media, xl/media). Reading media/ straight from the zip is the most robust
approach -- it catches images inside "groups/placeholders" that the shapes API easily misses.
Costs: (1) we lose position (which page/paragraph); (2) we also pick up decorations like logos,
icons, signatures. Countermeasure: first filter out small decorations by "byte size + pixel
dimensions", then let the VL make the final call (reply DECORATIVE → dropped).
"""
import io
import logging
import zipfile

import model_client

logger = logging.getLogger(__name__)

# Raster formats the VL can accept (vector formats like emf/wmf can't be accepted → skip them)
_RASTER = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
           "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp"}
_MIN_PX = 130        # any side < 130px → most likely an icon/decoration
_MIN_BYTES = 3000    # < 3KB → blank/placeholder image

_PROMPT = (
    "Describe this image from an office document thoroughly so it can be retrieved later. "
    "If it is a chart, diagram, figure, microscopy, gel, or photo, describe what it shows — "
    "axes, labels, conditions, key data and findings. "
    "If it is only a logo, icon, signature, or decoration, reply with exactly: DECORATIVE"
)


def _dimensions(data):
    # Takes: image bytes   Returns: (width, height) in pixels; None if unreadable
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def image_elements(path, label="Image", max_tokens=800):
    """Read images from media/ inside an Office file (zip) → VL description → list of figure elements.

    Takes: the file path, and a section prefix to name the images with
    Returns: a list of figure elements (decorations already filtered out)
    """
    elements = []
    try:
        z = zipfile.ZipFile(path)
    except Exception as e:
        logger.warning(f"Could not open {path}: {e}")
        return elements

    n = 0
    for name in sorted(z.namelist()):
        if "/media/" not in name:
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        mime = _RASTER.get(ext)
        if not mime:                          # vector etc. formats the VL can't accept
            continue
        data = z.read(name)
        if len(data) < _MIN_BYTES:            # too small → blank/placeholder
            continue
        dims = _dimensions(data)
        if dims and (dims[0] < _MIN_PX or dims[1] < _MIN_PX):
            continue                          # any side too small → icon/decoration
        try:
            desc = model_client.describe_image(data, _PROMPT, max_tokens=max_tokens, mime=mime)
        except Exception as e:
            logger.warning(f"Image understanding failed for {name}: {e}")
            continue
        if not desc or desc.strip().upper().startswith("DECORATIVE") or model_client.vl_found_nothing(desc):
            continue                          # decoration / VL says it sees no image → drop
        n += 1
        elements.append({"page": 1, "section": f"{label} {n}", "type": "figure", "text": desc})
    return elements

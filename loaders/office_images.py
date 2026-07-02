"""Office 文件内嵌图片的统一识图（docx / pptx / xlsx 共用）。

三种格式都是 zip，图片都放在压缩包的 media/ 文件夹（word/media、ppt/media、xl/media）。
直接从 zip 读 media/ 最稳——能抓到"组合/占位符"里、用 shapes API 容易漏掉的图。
代价：①丢了位置（不知第几页/段）；②会带上 logo/图标/签名等装饰图。
对策：先按"字节大小 + 像素尺寸"过滤掉小装饰，再让 VL 兜底判定（回 DECORATIVE 就丢）。
"""
import io
import logging
import zipfile

import model_client

logger = logging.getLogger(__name__)

# VL 能收的位图格式（emf/wmf 等矢量格式收不了，跳过）
_RASTER = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
           "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp"}
_MIN_PX = 130        # 任一边 < 130px → 多半是图标/装饰
_MIN_BYTES = 3000    # < 3KB → 空图/占位

_PROMPT = (
    "Describe this image from an office document thoroughly so it can be retrieved later. "
    "If it is a chart, diagram, figure, microscopy, gel, or photo, describe what it shows — "
    "axes, labels, conditions, key data and findings. "
    "If it is only a logo, icon, signature, or decoration, reply with exactly: DECORATIVE"
)


def _dimensions(data):
    # 接收: 图片字节   输出: (宽, 高) 像素；读不出就 None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def image_elements(path, label="Image", max_tokens=800):
    """读 Office 文件(zip)里 media/ 的图片 → VL 描述 → figure 元件列表。

    接收: 文件路径、给图片起的 section 前缀
    输出: figure 元件列表（已滤掉装饰图）
    """
    elements = []
    try:
        z = zipfile.ZipFile(path)
    except Exception as e:
        logger.warning(f"打不开 {path}: {e}")
        return elements

    n = 0
    for name in sorted(z.namelist()):
        if "/media/" not in name:
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        mime = _RASTER.get(ext)
        if not mime:                          # 矢量等格式 VL 收不了
            continue
        data = z.read(name)
        if len(data) < _MIN_BYTES:            # 太小 → 空图/占位
            continue
        dims = _dimensions(data)
        if dims and (dims[0] < _MIN_PX or dims[1] < _MIN_PX):
            continue                          # 任一边太小 → 图标/装饰
        try:
            desc = model_client.describe_image(data, _PROMPT, max_tokens=max_tokens, mime=mime)
        except Exception as e:
            logger.warning(f"识图失败 {name}: {e}")
            continue
        if not desc or desc.strip().upper().startswith("DECORATIVE") or model_client.vl_found_nothing(desc):
            continue                          # 装饰 / VL 说没看到图 → 丢
        n += 1
        elements.append({"page": 1, "section": f"{label} {n}", "type": "figure", "text": desc})
    return elements

"""PowerPoint(.pptx) 解析。"""
from .common import _rows_to_md
from .office_images import image_elements


def _pptx_table_md(table):
    # 接收: 一个 pptx 表格对象   输出: Markdown 表格字符串
    # 把每行每格的 .text 取成二维列表 → 交给共用的 _rows_to_md
    return _rows_to_md([[c.text for c in row.cells] for row in table.rows])


def parse_pptx(path):
    """解析 PPT(.pptx) → elements。每页文字→text（section='Slide N'，含备注）；表格→table。"""
    from pptx import Presentation        # 惰性 import：没装 python-pptx 时也不影响导入本模块

    prs = Presentation(str(path))        # 接收: 文件路径   输出: 整个演示文稿对象
    elements = []
    # enumerate(prs.slides, start=1)  输出: 逐个 (页码 i, 幻灯片对象 slide)，i 从 1 开始
    for i, slide in enumerate(prs.slides, start=1):
        section = f"Slide {i}"           # 这页的 section 就叫 "Slide 1/2/3..."
        texts = []                       # 攒这页所有文本框的文字
        for shape in slide.shapes:       # 遍历这页上的每个"形状"（文本框 / 表格 / 图片…）
            if shape.has_table:                       # 这个形状是表格
                md = _pptx_table_md(shape.table)      # → Markdown 表
                if md:
                    elements.append({"page": i, "section": section, "type": "table", "text": md})
            elif shape.has_text_frame:                # 这个形状是文本框
                t = shape.text_frame.text.strip()     # 取里面的文字
                if t:
                    texts.append(t)
        if slide.has_notes_slide:                     # 这页有"演讲者备注"的话也收进来
            note = slide.notes_slide.notes_text_frame.text.strip()
            if note:
                texts.append(f"[备注] {note}")
        if texts:                                     # 这页攒的文字合成一个 text 元件
            elements.append({"page": i, "section": section, "type": "text", "text": "\n".join(texts)})
    elements.extend(image_elements(path, "Slide image"))   # 幻灯片内嵌图片 → VL 识图 → figure
    return elements

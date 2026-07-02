"""Word(.docx) 解析。"""
from .common import _rows_to_md, is_table_caption
from .office_images import image_elements


def _docx_table_md(table):
    # 接收: 一个 docx Table 对象        输出: Markdown 表格字符串
    # 做法: 把每行每格的 .text 取成二维列表 → 交给共用的 _rows_to_md 拼成表
    return _rows_to_md([[c.text for c in row.cells] for row in table.rows])


def parse_docx(path):
    """解析 Word(.docx) → elements。
    段落→text（标题样式当 section）；表格→table（Markdown）。
    注：内嵌图片 v1 暂不识图，需要时再加 model_client.describe_image。
    """
    from docx import Document
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    doc = Document(str(path))   # 接收: 文件路径(字符串)   输出: 整个 Word 文档对象
    elements = []               # 最终要返回的元件清单
    section = ""                # 当前所在章节（遇到标题时更新）
    buf = []                    # 暂存"还没归档的正文行"

    def flush():
        # 接收: 无（用闭包里的 buf / section / elements）
        # 输出: 无（副作用：把 buf 攒的文字凑成一个 text 元件放进 elements，然后清空 buf）
        text = " ".join(buf).strip()
        if text:
            elements.append({"page": 1, "section": section, "type": "text", "text": text})
        buf.clear()

    # doc.element.body.iterchildren()
    #   接收: 无   输出: 逐个吐出 body 里的"子节点"(段落/表格的原始 XML)，按文档原始顺序
    for child in doc.element.body.iterchildren():

        # isinstance(child, CT_P)  接收:(对象, 类型)  输出: True/False —— 这个子节点是不是"段落"
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)   # 接收:(段落XML节点, 所属文档)  输出: 好用的 Paragraph 对象
            txt = para.text.strip()        # para.text → 段落纯文字(字符串); .strip() → 去首尾空白
            if not txt:                    # 空段落 → 跳过
                continue
            # para.style.name → 该段用的样式名(字符串，如 "Heading 1"/"Normal"); .lower() 转小写好比较
            style = (para.style.name or "").lower() if para.style else ""
            if style.startswith("heading") or style == "title":
                flush()                    # 先把上面攒的正文归档
                section = txt              # 这段是标题 → 开启新 section
            else:
                buf.append(txt)            # 普通正文 → 先攒进 buf，等下次 flush 时归档

        # isinstance(child, CT_Tbl) → 这个子节点是不是"表格"
        elif isinstance(child, CT_Tbl):
            # 表注通常在表「上方」：若 buf 末尾那段是 "Table N..."，取出来当表注，绑到这张表上
            caption = buf.pop() if (buf and is_table_caption(buf[-1])) else ""
            flush()                                  # 其余正文归档
            md = _docx_table_md(Table(child, doc))   # Table(...)→表格对象; _docx_table_md(...)→Markdown字符串
            if md:
                text = f"{caption}\n{md}" if caption else md   # 有表注就拼到表格前面，和表同进一个 chunk
                elements.append({"page": 1, "section": section, "type": "table", "text": text})

    flush()           # 循环结束，把最后攒的正文也归档
    elements.extend(image_elements(path, "Figure"))   # 内嵌图片 → VL 识图 → figure 元件
    return elements   # 输出: 元件列表（text / table / figure）

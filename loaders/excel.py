"""Excel(.xlsx/.xls) 和 CSV 解析（都是表格，用 pandas）。"""
from .common import _rows_to_md
from .office_images import image_elements


def _df_to_md(df):
    # 接收: 一个 pandas DataFrame（表格）   输出: Markdown 表格字符串
    df = df.fillna("")                                          # NaN→""（NaN 是"真值"，_rows_to_md 的 or "" 挡不住，要先填）
    rows = [list(df.columns)] + df.astype(str).values.tolist()  # 表头一行 + 数据各行（都转字符串），拼成二维列表
    return _rows_to_md(rows)                                    # 交给共用函数拼成 Markdown


def parse_xlsx(path):
    """解析 Excel(.xlsx/.xls) → elements。每个 sheet → 一个 text 元件（Markdown 表格）。

    用 text 型（而非 table 型）：数据表可能很大，留给下游切块，避免单个 chunk 超出嵌入长度。
    注：纯数据表（数字为主）语义检索效果有限，更适合精确查询；此处仅做通用接入。
    """
    import pandas as pd                                  # 惰性 import

    # pd.read_excel(..., sheet_name=None)  接收: 路径   输出: 字典 {sheet名: DataFrame}（None=读所有 sheet）
    sheets = pd.read_excel(str(path), sheet_name=None)
    elements = []
    # enumerate(sheets.items(), start=1)  输出: 逐个 (序号 i, (sheet名 name, 表 df))
    for i, (name, df) in enumerate(sheets.items(), start=1):
        md = _df_to_md(df)                               # 这个 sheet → Markdown 表
        if md:
            # section = sheet 名（天然就是这张表的标题）
            elements.append({"page": i, "section": name, "type": "text", "text": md})
    elements.extend(image_elements(path, "Sheet image"))   # 表里内嵌的图片 → VL 识图 → figure
    return elements


def parse_csv(path):
    """解析 CSV → 一个 text 元件（Markdown 表格，下游会切块）。"""
    import pandas as pd

    df = pd.read_csv(str(path))                          # 接收: 路径   输出: 一个 DataFrame
    md = _df_to_md(df)                                   # → Markdown 表
    return [{"page": 1, "section": "", "type": "text", "text": md}] if md else []

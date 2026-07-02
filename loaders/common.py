"""跨格式共用的小工具。docx / pptx / excel / csv 等多个 parser 共用，避免重复（DRY）。"""
import re

# 表格标题/注释正则：匹配以 "Table 1" / "Table 1:" / "Tab. 2" / "表 1" 开头的文字
#   ^\s*            开头可有空格
#   (?:Table|Tab\.?|表)  三种写法都认（Tab 后面的点可有可无）
#   \s*\d+          后面跟编号
#   [A-Za-z]?       容许 "Table 1A" 这种带字母的
TABLE_RE = re.compile(r'^\s*(?:Table|Tab\.?|表)\s*\d+[A-Za-z]?\b', re.IGNORECASE)


def is_table_caption(text):
    # 接收: 一段文字(字符串)   输出: True/False —— 它是不是以 "Table N" / "表 N" 开头
    # text or "" : 万一传进来 None，用空串兜底，免得 .match 报错
    return bool(TABLE_RE.match(text or ""))


def table_label(text):
    # 接收: 一段文字   输出: 表标签字符串（如 "Table 2"），不是表注则返回 None
    # 用于"这张表抓没抓过"的去重
    m = TABLE_RE.match(text or "")
    return m.group(0).strip() if m else None


def _rows_to_md(rows):
    # 接收: 二维列表 rows（行的列表，每行又是格子的列表）
    # 输出: 一个 Markdown 表格字符串
    # 第1步：逐格清洗（None→""、换行→空格、去首尾空白），形状不变、还是二维
    rows = [[str(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)            # 最宽的行有几格（所有行要对齐到这个列数）
    pad = lambda r: r + [""] * (width - len(r))  # 小函数：把某行补空格子，补齐到 width 列
    # 表头行 + Markdown 必需的分隔线（| --- | --- | ...）
    lines = ["| " + " | ".join(pad(rows[0])) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:                           # 其余每行：补齐 → join 成 "| a | b | c |"
        lines.append("| " + " | ".join(pad(r)) + " |")
    return "\n".join(lines)                       # 各行用换行连起来 → 完整的 Markdown 表

"""纯文本(.txt) 解析。"""
from pathlib import Path


def parse_txt(path):
    """解析纯文本 → 一个 text 元件（整篇，交给下游切块）。"""
    # Path(path).read_text(...)  接收: 文件路径   输出: 整个文件的文字(字符串)
    #   encoding="utf-8" 按 UTF-8 读；errors="ignore" 遇到无法解码的字节就跳过（别让整篇崩）
    #   .strip() 去掉首尾空白
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:                # 空文件 → 返回空清单
        return []
    # 整篇当一个 text 元件（page 固定 1、section 留空）；过长的话下游会按 chunk_size 切
    return [{"page": 1, "section": "", "type": "text", "text": text}]

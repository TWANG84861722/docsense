"""解析层（loaders 包）—— 把"文件"解析成统一的元件清单。

每个 parse_xxx 都返回统一的 elements 列表：
  element = {"page": int, "section": str,
             "type": "text" | "table" | "figure", "text": str}
  · text 型   = 原始正文，留给下游（ingest）按 chunk_size 切块
  · table/figure 型 = 成品，原样保留
  · 不含 "paper"（来源文件名由 ingest 统一打上）

加新格式：在 loaders/ 下新建 xxx.py 写 parse_xxx，在这里 import 暴露出来，
再去 ingest.py 的 PARSERS 注册一行即可。下游主循环不用动。
"""
# 把包内各模块（.pdf / .txt / ...）里的 parse_* 提到包顶层，
# 这样外部 `import loaders` 后就能直接 loaders.parse_pdf(...)，不用关心它具体在哪个文件。
# 开头的点 = "本包内"（相对导入）：.pdf 指 loaders/pdf.py，不是装的第三方库。
from .pdf import parse_pdf
from .txt import parse_txt
from .docx import parse_docx
from .pptx import parse_pptx
from .excel import parse_xlsx, parse_csv

# __all__：声明本包对外公开的名字（也是 from loaders import * 时会导出的清单）
__all__ = ["parse_pdf", "parse_txt", "parse_docx", "parse_pptx", "parse_xlsx", "parse_csv"]

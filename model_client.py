"""统一的模型调用层 —— 一键换 provider。

所有云厂商（OpenAI / Claude / Gemini / Qwen）和本地 server 都走 OpenAI 兼容接口，
所以这里只用 openai 这一个库。换 provider 只需改 models.yaml 的 active，
本文件和上层代码（chat.py / ingest.py）都不用动。

对外只暴露两个函数：
- chat(messages, max_tokens)      —— 文本对话（LLM）
- describe_image(image_bytes, prompt, max_tokens) —— 识图（VL）
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ── VL 识图结果缓存 ─────────────────────────────────────────
# 同一张图(+同一 prompt+模型)只调一次 VL，结果按"内容哈希"存盘。
# 这样反复改 parse/切块/组装都不再重付 VL 钱（图片没变就命中）。
# 放在 vl_cache/（不在 db/ 里 → "清空 db 重建索引"不会丢这份缓存）。
# 注：prompt/模型 进了哈希 key —— 改了 prompt 或换了模型会自动 miss、重新识图。
_VL_CACHE_DIR = Path("vl_cache")


def _vl_cache_key(image_bytes: bytes, prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(image_bytes); h.update(b"\x00")
    h.update(prompt.encode("utf-8")); h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _vl_cache_get(key: str) -> str | None:
    f = _VL_CACHE_DIR / f"{key}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else None


def _vl_cache_put(key: str, text: str) -> None:
    _VL_CACHE_DIR.mkdir(exist_ok=True)
    (_VL_CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")


@lru_cache(maxsize=8)
def _client(provider_name: str) -> OpenAI:
    """按 provider 名创建（并缓存）一个 OpenAI 兼容客户端。"""
    spec = config.MODELS["providers"][provider_name]
    key_env = spec.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(
            f"缺少 API key：请在 .env / 环境变量里设置 {key_env}（provider='{provider_name}'）"
        )
    return OpenAI(base_url=spec["base_url"], api_key=api_key or "not-needed")


def _resolve(role: str):
    """返回 (provider名, 该provider配置, 该角色用的模型名)。role: 'chat' | 'vl'。"""
    name, spec = config._active(role)
    model = spec["chat_model"] if role == "chat" else spec["vl_model"]
    return name, spec, model


def chat(messages: list[dict], max_tokens: int = 800, model: str | None = None) -> str:
    """文本对话。messages 用标准 OpenAI 格式：[{"role": "...", "content": "..."}]。"""
    name, _spec, default_model = _resolve("chat")
    resp = _client(name).chat.completions.create(
        model=model or default_model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _vl_call(
    provider_name: str,
    image_bytes: bytes,
    prompt: str,
    model_name: str,
    max_tokens: int,
    mime: str,
) -> tuple[str, str]:
    """发一次 VL 请求，返回 (文字, finish_reason)。

    finish_reason == 'length' 表示这次是被 max_tokens 上限**切断**的（没写完）。
    """
    b64 = base64.b64encode(image_bytes).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]
    resp = _client(provider_name).chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason


def describe_image(
    image_bytes: bytes,
    prompt: str,
    max_tokens: int = 4000,
    model: str | None = None,
    mime: str = "image/png",
) -> str:
    """识图：把图片 + 提示发给当前 VL provider，返回描述文字。

    - 磁盘缓存：同一张图(+同一 prompt+模型)只调一次 VL，之后直接取缓存。
    - 截断处理：max_tokens 只是上限、按实际生成计费、模型写完自停 → **一次给足封顶(默认 4000)**。
      正常内容(实测图~2000、满页表~3000)都能一次写完。万一 4000 还被切断(finish_reason=='length')，
      视为异常页(内容超长/模型打转)——**大声告警、且不写缓存**(绝不把半截结果当好货存，下次会重试)。
      不做"从低起步再重试"：按实付费下那是浪费(半截 token 白花 + 图片重发)，封顶一次到位更省。
      → 这也是 VL 识图预算的【唯一一处】设定，pdf.py 等调用方都继承它。
    """
    name, _spec, default_model = _resolve("vl")
    model_name = model or default_model

    key = _vl_cache_key(image_bytes, prompt, model_name)
    cached = _vl_cache_get(key)
    if cached is not None:
        return cached                       # 命中缓存 → 不调 VL

    result, reason = _vl_call(name, image_bytes, prompt, model_name, max_tokens, mime)

    if reason == "length":                  # 封顶还被切断 → 异常页：告警、返回半截、但不缓存
        logger.error(
            f"VL 输出在 max_tokens={max_tokens} 被截断，疑似内容超长/模型打转 → 本次不缓存(下次重试)"
        )
        return result

    if result:
        _vl_cache_put(key, result)          # 只有【完整】结果才存盘 → 缓存里都是"已知好货"
    return result


_VL_NOTHING = (
    "no table", "no figure", "no image", "no chart", "no visual data", "no visible",
    "does not contain", "there is no", "only displays the", "only contains the",
    "only the text", "no discernible", "appears to be blank", "unable to",
    "cannot see", "cannot describe", "can't describe", "no data or figure",
    "not visible", "placeholder", "please upload", "please provide", "no visual content",
)


def vl_found_nothing(text: str) -> bool:
    """判断 VL 回答是不是"这张图里没图/没表"（区域取错时会这么回）。"""
    t = (text or "").lower()[:200]
    return any(p in t for p in _VL_NOTHING)


def active_summary() -> str:
    """调试用：打印当前生效的 provider/模型。"""
    cn, _, cm = _resolve("chat")
    vn, _, vm = _resolve("vl")
    return f"chat → {cn}:{cm} | vl → {vn}:{vm}"

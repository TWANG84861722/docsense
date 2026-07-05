"""Unified model-call layer -- switch provider in one place.

Every cloud vendor (OpenAI / Claude / Gemini / Qwen) and the local server speaks the
OpenAI-compatible API, so this file uses the single `openai` library only. Switching provider
means changing `active` in models.yaml; neither this file nor the callers (chat.py / ingest.py)
need to change.

It exposes two public functions:
- chat(messages, max_tokens)                       -- text conversation (LLM)
- describe_image(image_bytes, prompt, max_tokens)  -- image understanding (VL)
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

# ── VL image-description cache ──────────────────────────────
# The same image (+ same prompt + model) calls the VL only once; the result is stored on disk
# keyed by a "content hash". This means repeatedly reworking parse/chunking/assembly never
# pays for VL again (unchanged image → cache hit).
# Kept under vl_cache/ (not inside db/ → "wipe db and rebuild the index" does not lose this cache).
# Note: the prompt/model are part of the hash key -- changing the prompt or switching the model
# automatically misses and re-runs the VL.
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
    """Create (and cache) an OpenAI-compatible client for a provider name."""
    spec = config.MODELS["providers"][provider_name]
    key_env = spec.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(
            f"Missing API key: please set {key_env} in .env / the environment (provider='{provider_name}')"
        )
    return OpenAI(base_url=spec["base_url"], api_key=api_key or "not-needed")


def _resolve(role: str):
    """Return (provider name, that provider's config, the model name for this role). role: 'chat' | 'vl'."""
    name, spec = config._active(role)
    model = spec["chat_model"] if role == "chat" else spec["vl_model"]
    return name, spec, model


def chat(messages: list[dict], max_tokens: int = 800, model: str | None = None) -> str:
    """Text conversation. `messages` uses the standard OpenAI format: [{"role": "...", "content": "..."}]."""
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
    """Send one VL request, returning (text, finish_reason).

    finish_reason == 'length' means this response was **cut off** by the max_tokens cap
    (it did not finish writing).
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
    """Image understanding: send image + prompt to the current VL provider, return description text.

    - Disk cache: the same image (+ same prompt + model) calls the VL only once, then serves the cache.
    - Truncation handling: max_tokens is only a cap, billing is by actual generation, and the model
      stops on its own when done → **give a generous cap once (default 4000)**.
      Normal content (measured: figures ~2000, full-page tables ~3000) finishes in one shot. If 4000
      is still cut off (finish_reason == 'length'), treat it as an abnormal page (content too long /
      the model looping) -- **warn loudly AND do not cache** (never store a half-finished result as if
      it were good; the next run will retry).
      No "start low then retry": under pay-per-use that is wasteful (the half-written tokens are burned
      + the image is re-sent); a single generous cap is cheaper.
      → This is also the *single place* the VL budget is set; callers like pdf.py all inherit it.
    """
    name, _spec, default_model = _resolve("vl")
    model_name = model or default_model

    key = _vl_cache_key(image_bytes, prompt, model_name)
    cached = _vl_cache_get(key)
    if cached is not None:
        return cached                       # cache hit → do not call the VL

    result, reason = _vl_call(name, image_bytes, prompt, model_name, max_tokens, mime)

    if reason == "length":                  # capped yet still cut off → abnormal page: warn, return the partial, but don't cache
        logger.error(
            f"VL output truncated at max_tokens={max_tokens}, likely content too long / model looping "
            f"→ not caching this one (will retry next time)"
        )
        return result

    if result:
        _vl_cache_put(key, result)          # only *complete* results get saved → the cache holds "known-good" only
    return result


_VL_NOTHING = (
    "no table", "no figure", "no image", "no chart", "no visual data", "no visible",
    "does not contain", "there is no", "only displays the", "only contains the",
    "only the text", "no discernible", "appears to be blank", "unable to",
    "cannot see", "cannot describe", "can't describe", "no data or figure",
    "not visible", "placeholder", "please upload", "please provide", "no visual content",
)


def vl_found_nothing(text: str) -> bool:
    """Decide whether the VL's answer means "no figure/no table in this image" (which it says
    when the cropped region was picked wrong)."""
    t = (text or "").lower()[:200]
    return any(p in t for p in _VL_NOTHING)

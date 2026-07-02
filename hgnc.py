import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILE = Path("db/gene_aliases.json")

# Gene-like tokens: starts with uppercase, has at least one more alphanumeric/hyphen char
# Matches: TP53, BRCA1, IL-1B, Trp53   Skips: "the", "in", "results"
_GENE_LIKE = re.compile(r'^[A-Z][A-Za-z0-9\-]+$')

_alias_to_canonical: dict[str, str] = {}
_canonical_to_all: dict[str, frozenset[str]] = {}
_load_failed = False   # 缓存缺失且 mygene/联网不可用时置 True，之后直接跳过、不再重试


def _build_cache():
    import mygene
    logger.info("Building gene alias cache from MyGene.info (one-time)...")
    mg = mygene.MyGeneInfo()

    alias_to_canonical: dict[str, str] = {}
    canonical_to_all: dict[str, list[str]] = {}

    hits = mg.query(
        "taxid:9606",
        fields="symbol,alias",
        size=1000,
        fetch_all=True,
        verbose=False,
    )

    for gene in hits:
        symbol = gene.get("symbol", "").strip()
        if not symbol:
            continue

        aliases = gene.get("alias", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        all_names = {symbol} | set(aliases)
        canonical_to_all[symbol] = sorted(all_names)

        for name in all_names:
            alias_to_canonical.setdefault(name.lower(), symbol)

    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"alias_to_canonical": alias_to_canonical,
             "canonical_to_all": canonical_to_all},
            f
        )

    logger.info(f"Cached {len(canonical_to_all):,} genes → {CACHE_FILE}")
    return alias_to_canonical, canonical_to_all


def _load():
    global _alias_to_canonical, _canonical_to_all

    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _alias_to_canonical = data["alias_to_canonical"]
        _canonical_to_all = {
            k: frozenset(v) for k, v in data["canonical_to_all"].items()
        }
        logger.info(f"Gene alias cache loaded ({len(_canonical_to_all):,} genes)")
    else:
        a2c, c2a = _build_cache()
        _alias_to_canonical = a2c
        _canonical_to_all = {k: frozenset(v) for k, v in c2a.items()}


def _ensure_loaded():
    global _load_failed
    if _alias_to_canonical or _load_failed:
        return
    try:
        _load()
    except Exception as e:
        _load_failed = True    # 建/读缓存失败 → 标记，避免每次查询重试
        logger.warning(
            f"基因别名扩展不可用，已跳过（检索照常，只是不扩展基因同义词）。"
            f"原因: {e}。想启用：`pip install mygene`（首次会联网建缓存到 {CACHE_FILE}）。"
        )


def expand_query(text: str) -> str:
    """
    Detect gene symbols in text and append all known aliases.

    "role of p53 in apoptosis"
    → "role of p53 in apoptosis
       Gene aliases: p53 [BRCC3, TP53, TRP53, LFS1, ...]"
    """
    _ensure_loaded()

    tokens = set(re.findall(r'\b[A-Za-z][A-Za-z0-9\-]*\b', text))
    expansions = []
    seen_canonical: set[str] = set()

    for token in tokens:
        if not _GENE_LIKE.match(token):
            continue

        canonical = _alias_to_canonical.get(token.lower())
        if canonical is None or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        synonyms = sorted(_canonical_to_all.get(canonical, frozenset()) - {token})[:8]
        if synonyms:
            expansions.append(f"{token} [{', '.join(synonyms)}]")

    if not expansions:
        return text

    return text + "\nGene aliases: " + "; ".join(expansions)

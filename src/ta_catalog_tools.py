# -*- coding: utf-8 -*-
"""
Tools to filter and annotate a pandas_ta catalog JSON produced by ta_introspection.

Goals:
- Keep only predictive indicator categories (e.g., momentum, overlap, trend, volume, statistics, candles).
- Flag likely look-ahead/repainting indicators using conservative heuristics and a small blacklist.

Usage:
    from src.ta_catalog_tools import load_catalog, filter_catalog, save_catalog
    cat = load_catalog('data/pandas_ta_catalog.json')
    filtered = filter_catalog(cat)
    save_catalog(filtered, 'data/pandas_ta_catalog_filtered.json')
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


KEEP_CATEGORIES = {
    "momentum",
    "overlap",
    "trend",
    "volume",
    "statistics",
    "candles",
    # "volatility",  # include if present in your install
}


# Known indicators that are prone to repainting or use forward-shifts/centered windows
BLACKLIST_NAMES = {
    # common repaint/forward-shift suspects
    "zigzag",
    "fractal",
    "ichimoku",  # senkou spans plotted forward; safer to exclude by default
}


# Heuristic keywords to flag look-ahead from name or doc
LOOKAHEAD_KEYWORDS = [
    r"\bforward\b",
    r"\blead\b",
    r"\bcenter(ed|)\b",
    r"\brepaint(ing|s|)\b",
    r"\bfuture\b",
    r"\bshift\b",
]


def load_catalog(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_callable_indicator(meta: Dict[str, Any]) -> bool:
    # basic sanity: must have a signature text and params list
    if "text" not in meta or "params" not in meta:
        return False
    # exclude empty signatures or classes accidentally captured
    sig = meta.get("text", "")
    return sig != "()"


def _looks_lookahead(name: str, meta: Dict[str, Any]) -> bool:
    lname = name.lower()
    if lname in BLACKLIST_NAMES:
        return True
    text = meta.get("text", "").lower()
    doc = (meta.get("doc") or "").lower()
    hay = " ".join([lname, text, doc])
    return any(re.search(pat, hay) for pat in LOOKAHEAD_KEYWORDS)


def filter_catalog(catalog: Dict[str, Any]) -> Dict[str, Any]:
    cats: Dict[str, Any] = catalog.get("categories", {})
    kept: Dict[str, Any] = {}
    excluded_categories: List[str] = []
    flagged_lookahead: List[Tuple[str, str]] = []
    total_kept = 0

    for cat_name, cat_data in cats.items():
        if cat_name not in KEEP_CATEGORIES:
            excluded_categories.append(cat_name)
            continue
        indicators = {}
        for ind_name, meta in cat_data.get("indicators", {}).items():
            if not _is_callable_indicator(meta):
                continue
            lookahead = _looks_lookahead(ind_name, meta)
            # annotate but keep only if not lookahead
            if lookahead:
                flagged_lookahead.append((cat_name, ind_name))
                continue
            # keep minimal fields
            indicators[ind_name] = {
                "signature": meta.get("text", ""),
                "params": meta.get("params", []),
            }
        kept[cat_name] = {"count": len(indicators), "indicators": indicators}
        total_kept += len(indicators)

    out = {
        "pandas_ta_version": catalog.get("pandas_ta_version", "unknown"),
        "total_count": total_kept,
        "kept_categories": sorted(list(KEEP_CATEGORIES & set(cats.keys()))),
        "excluded_categories": sorted(excluded_categories),
        "flagged_lookahead": [
            {"category": c, "indicator": i} for c, i in sorted(flagged_lookahead)
        ],
        "categories": kept,
    }
    return out


def save_catalog(catalog: Dict[str, Any], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


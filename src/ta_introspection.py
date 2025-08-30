# -*- coding: utf-8 -*-
"""
Utilities to introspect pandas_ta: list categories, indicators, and signatures.

This module provides a safe, import-only inspection of the pandas_ta package
so you can catalog which indicators are available in your environment, grouped
by category, together with their call signatures.

Typical usage (in a notebook):

    from src.ta_introspection import (
        get_available_categories,
        get_indicators_for_category,
        get_indicator_signature,
        build_full_catalog,
        save_catalog_json,
    )

    cats = get_available_categories()
    inds = get_indicators_for_category('momentum')
    sig = get_indicator_signature('momentum', 'rsi')
    catalog = build_full_catalog(include_doc=False)
    save_catalog_json(catalog, 'data/pandas_ta_catalog.json')

All functions handle missing pandas_ta gracefully by raising a clear ImportError.
"""
from __future__ import annotations

import inspect
import json
import types
from typing import Any, Dict, List, Optional


def _import_pandas_ta():
    """Import pandas_ta and return the module. Raises informative ImportError if missing."""
    try:
        import pandas_ta as ta  # type: ignore
        return ta
    except Exception as e:
        raise ImportError(
            "pandas_ta is not installed or failed to import. Install with `pip install pandas-ta`.\n"
            f"Original error: {e}"
        )


def get_available_categories() -> List[str]:
    """Return a list of available pandas_ta categories (submodules) present in this install."""
    ta = _import_pandas_ta()
    cats: List[str] = []
    for name in dir(ta):
        if name.startswith("_"):
            continue
        obj = getattr(ta, name)
        if isinstance(obj, types.ModuleType) and getattr(obj, "__package__", "").startswith("pandas_ta"):
            # ensure it has at least one callable indicator
            if any(
                callable(getattr(obj, attr)) and not attr.startswith("_")
                for attr in dir(obj)
            ):
                cats.append(name)
    # Provide a stable ordering
    return sorted(set(cats))


def get_indicators_for_category(category: str) -> List[str]:
    """Return a sorted list of indicator function names within a pandas_ta category."""
    ta = _import_pandas_ta()
    try:
        mod = getattr(ta, category)
    except AttributeError:
        raise ValueError(f"Category '{category}' not found in pandas_ta")
    names: List[str] = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        func = getattr(mod, name)
        if callable(func):
            names.append(name)
    return sorted(set(names))


def _signature_dict(func: Any) -> Dict[str, Any]:
    """Return a JSON-serializable signature description for a callable."""
    sig = inspect.signature(func)
    params: List[Dict[str, Any]] = []
    for p in sig.parameters.values():
        default = p.default
        if default is inspect._empty:  # type: ignore[attr-defined]
            default_repr = "<required>"
        else:
            try:
                default_repr = repr(default)
            except Exception:
                default_repr = str(default)
        params.append(
            {
                "name": p.name,
                "kind": str(p.kind).split(".")[-1],
                "default": default_repr,
                "annotation": str(p.annotation) if p.annotation is not inspect._empty else "",
            }
        )
    return {
        "text": str(sig),
        "params": params,
    }


def get_indicator_signature(category: str, indicator: str) -> Dict[str, Any]:
    """Return the signature metadata for a specific indicator within a category."""
    ta = _import_pandas_ta()
    try:
        mod = getattr(ta, category)
    except AttributeError:
        raise ValueError(f"Category '{category}' not found in pandas_ta")
    try:
        func = getattr(mod, indicator)
    except AttributeError:
        raise ValueError(f"Indicator '{indicator}' not found in pandas_ta.{category}")
    if not callable(func):
        raise ValueError(f"pandas_ta.{category}.{indicator} is not callable")
    out = _signature_dict(func)
    # include first line of doc if available
    doc = inspect.getdoc(func) or ""
    out["doc"] = doc.splitlines()[0] if doc else ""
    return out


def build_full_catalog(include_doc: bool = False) -> Dict[str, Any]:
    """Build a full catalog of pandas_ta indicators with signatures by category."""
    ta = _import_pandas_ta()
    cats = get_available_categories()
    catalog: Dict[str, Any] = {
        "pandas_ta_version": getattr(ta, "__version__", "unknown"),
        "categories": {},
    }
    for cat in cats:
        indicators = {}
        for ind in get_indicators_for_category(cat):
            try:
                meta = get_indicator_signature(cat, ind)
                if not include_doc:
                    meta.pop("doc", None)
                indicators[ind] = meta
            except Exception:
                # Best-effort: skip any problematic indicator
                continue
        catalog["categories"][cat] = {
            "count": len(indicators),
            "indicators": indicators,
        }
    return catalog


def save_catalog_json(catalog: Dict[str, Any], path: str) -> None:
    """Save the catalog dictionary to a JSON file."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


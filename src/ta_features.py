# -*- coding: utf-8 -*-
"""
Auto-register pandas_ta indicators as features with default parameters.

Reads the filtered catalog at data/pandas_ta_catalog_filtered.json and
registers one feature per indicator. Each feature computes the indicator with
default parameters and returns a single Series. If the indicator returns
multiple columns, the first column is used as the primary output.

Feature naming: "ta_{category}_{indicator}".

Usage:
    import src.ta_features  # triggers registration on import

Notes:
    - Requires pandas_ta installed and DataFrame with columns: open, high,
      low, close, (optionally) volume.
    - For candle functions using 'open_' param name, the 'open' column is used.
    - Basic derived inputs like hl2/hlc3/ohlc4/oc2 are provided if requested.
"""
from __future__ import annotations

from typing import Any, Dict, List, Callable
import json
import os
import pandas as pd
import numpy as np

from .registry import registry


def _import_ta():
    try:
        import warnings
        # Silence pkg_resources deprecation warning emitted by pandas_ta import
        warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
        import pandas_ta as ta  # type: ignore
        return ta
    except Exception as e:
        raise ImportError(
            "pandas_ta is required for ta_features. Install with `pip install pandas-ta`.\n"
            f"Original error: {e}"
        )


def _derive_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if set(["high", "low"]).issubset(df.columns):
        out["hl2"] = (df["high"] + df["low"]) / 2.0
    if set(["high", "low", "close"]).issubset(df.columns):
        out["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
    if set(["open", "high", "low", "close"]).issubset(df.columns):
        out["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    if set(["open", "close"]).issubset(df.columns):
        out["oc2"] = (df["open"] + df["close"]) / 2.0
    return out


def _build_kwargs(df: pd.DataFrame, param_names: List[str]) -> Dict[str, Any]:
    """Map function parameter names to DataFrame columns when available."""
    # base mapping aliases
    colmap = {
        "open": "open",
        "open_": "open",
        "o": "open",
        "high": "high",
        "h": "high",
        "low": "low",
        "l": "low",
        "close": "close",
        # NOTE: Do NOT naively map parameter name 'c' to 'close'.
        # Some indicators (e.g., pandas_ta.momentum.cci) use a scalar constant
        # parameter named 'c'. Passing a Series there breaks their validation
        # (e.g., `float(c)` / truthiness checks). We handle 'c' specially below.
        "volume": "volume",
        "vol": "volume",
        "v": "volume",
    }
    derived = _derive_series(df)
    kwargs: Dict[str, Any] = {}
    # Precompute lowercase parameter set for contextual decisions (e.g., 'c')
    param_set = {p.lower() for p in param_names}
    for p in param_names:
        pl = p.lower()
        if pl in ("kwargs", "*args"):
            continue
        # direct OHLCV mapping
        # Special-case: if the parameter is named 'c' and there is also an explicit
        # 'close' parameter, skip mapping. In libraries like pandas_ta, 'c' often
        # denotes a scalar constant (e.g., CCI's constant) and should not receive a Series.
        if pl == "c" and "close" in param_set:
            # Leave to the indicator's default handling
            continue
        if pl in colmap and colmap[pl] in df.columns:
            kwargs[p] = df[colmap[pl]]
            continue
        # derived common composites
        if pl in derived:
            kwargs[p] = derived[pl]
            continue
        # some indicators accept entire DataFrame via 'df' or 'data'
        if pl in ("df", "data"):
            kwargs[p] = df
            continue
        # leave other params to defaults
    return kwargs


def _primary_series(result: Any) -> pd.Series:
    """Coerce indicator output to a single Series.

    - If it's a Series, return as-is.
    - If it's a DataFrame, return the first column.
    - If it's array-like, wrap into a Series without name.
    """
    if isinstance(result, pd.Series):
        return result
    if hasattr(result, "iloc") and hasattr(result, "columns"):
        # DataFrame-like
        s = result.iloc[:, 0]
        return s
    # Fallback: try to convert
    try:
        return pd.Series(result)
    except Exception:
        # ultimate fallback: empty series of length df
        return pd.Series(dtype=float)


_ALLOWED_REQUIRED = {
    # direct OHLCV and common aliases
    "open", "open_", "o", "high", "h", "low", "l", "close", "c", "volume", "vol", "v",
    # derived composites and DF passthroughs
    "hl2", "hlc3", "ohlc4", "oc2", "df", "data", "kwargs",
}


def _register_indicator(cat: str, name: str, params_meta: List[Dict[str, Any]]) -> bool:
    ta = _import_ta()
    try:
        mod = getattr(ta, cat)
        func = getattr(mod, name)
    except Exception:
        return False  # skip if not found in this install

    feat_name = f"ta_{cat}_{name}"

    # Skip indicators whose required params are not satisfiable from OHLCV/derived
    required = [p.get("name", "").lower() for p in params_meta if str(p.get("default")) == "<required>"]
    if any((r not in _ALLOWED_REQUIRED) for r in required):
        return False

    param_names = [p.get("name", "") for p in params_meta]

    def _sanitize_ohlc_in_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure OHLC passed to candle functions are finite and numeric
        for k in ("open", "open_", "high", "low", "close"):
            if k in kw and isinstance(kw[k], pd.Series):
                s = pd.to_numeric(kw[k], errors="coerce")
                s = s.replace([np.inf, -np.inf], np.nan)
                # Fill interior NaNs with forward/backward fill to avoid int casting errors downstream
                s = s.ffill().bfill()
                # As a last resort, fill remaining NaNs with 0.0
                if s.isna().any():
                    s = s.fillna(0.0)
                kw[k] = s.astype(float)
        return kw

    @registry.register_feature(feat_name)
    def _feature(df: pd.DataFrame) -> pd.Series:
        kwargs = _build_kwargs(df, param_names)
        # Special handling for candle indicators
        if cat == "candles":
            kwargs = _sanitize_ohlc_in_kwargs(kwargs)
            # Many candle functions accept 'asbool' to avoid int-coded patterns that require int casting
            if "asbool" in [p.lower() for p in param_names] or name == "cdl_pattern":
                # For cdl_pattern, **kwargs is forwarded to pattern functions
                kwargs.setdefault("asbool", True)
        try:
            res = func(**kwargs)
        except TypeError:
            # Some functions are positional (e.g., high, low) — try positional order
            pos = [kwargs[k] for k in param_names if k in kwargs]
            res = func(*pos)
        s = _primary_series(res)
        s.name = feat_name
        return s

    return True


def _load_filtered_catalog(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: build on the fly
    from .ta_introspection import build_full_catalog
    cat = build_full_catalog(include_doc=False)
    return cat


def register_all_from_catalog(path: str = "data/pandas_ta_catalog_filtered.json") -> int:
    """Register all indicators from the filtered catalog. Returns count registered."""
    cat = _load_filtered_catalog(path)
    categories = cat.get("categories", {})
    count = 0
    for cname, cdata in categories.items():
        inds = cdata.get("indicators", {})
        for iname, meta in inds.items():
            params = meta.get("params", [])
            ok = _register_indicator(cname, iname, params)
            if ok:
                count += 1
    return count


# Trigger registration on import
try:
    register_all_from_catalog()
except Exception:
    # Do not fail import for environments without pandas_ta; user can call manually later
    pass

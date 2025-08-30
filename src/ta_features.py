# -*- coding: utf-8 -*-
"""
Auto-register pandas_ta indicators as features with default parameters.

Reads the filtered catalog at data/pandas_ta_catalog_filtered.json and
registers feature functions for every indicator. Indicators that return a
single Series become one feature named ``"ta_{category}_{indicator}"``. If an
indicator returns a ``DataFrame`` with multiple columns, a separate feature is
registered for each column named ``"ta_{category}_{indicator}_{column}"``.
Each feature recomputes the indicator on demand and returns the selected
output column. For example, ``pandas_ta.macd`` yields features
``ta_momentum_macd_macd``, ``ta_momentum_macd_signal``, and
``ta_momentum_macd_histogram``.

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

    # Discover indicator columns by running once on a dummy OHLCV DataFrame
    idx = pd.RangeIndex(10)
    dummy = pd.DataFrame(
        {
            "open": pd.Series(np.arange(1, 11, dtype=float), index=idx),
            "high": pd.Series(np.arange(1, 11, dtype=float) + 1, index=idx),
            "low": pd.Series(np.arange(1, 11, dtype=float) - 1, index=idx),
            "close": pd.Series(np.arange(1, 11, dtype=float), index=idx),
            "volume": pd.Series(np.arange(1, 11, dtype=float), index=idx),
        }
    )
    dkw = _build_kwargs(dummy, param_names)
    if cat == "candles":
        dkw = _sanitize_ohlc_in_kwargs(dkw)
        if "asbool" in [p.lower() for p in param_names] or name == "cdl_pattern":
            dkw.setdefault("asbool", True)
    try:
        dres = func(**dkw)
    except TypeError:
        pos = [dkw[k] for k in param_names if k in dkw]
        dres = func(*pos)
    except Exception:
        return False

    columns: List[str] = list(getattr(dres, "columns", []))

    if columns:
        for col in columns:
            feat_name = f"ta_{cat}_{name}_{col}"

            @registry.register_feature(feat_name)
            def _feature(df: pd.DataFrame, col=col, feat_name=feat_name) -> pd.Series:
                kwargs = _build_kwargs(df, param_names)
                if cat == "candles":
                    kwargs = _sanitize_ohlc_in_kwargs(kwargs)
                    if "asbool" in [p.lower() for p in param_names] or name == "cdl_pattern":
                        kwargs.setdefault("asbool", True)
                try:
                    res = func(**kwargs)
                except TypeError:
                    pos = [kwargs[k] for k in param_names if k in kwargs]
                    res = func(*pos)
                if hasattr(res, "__getitem__"):
                    try:
                        s = res[col]
                    except Exception:
                        s = res
                else:
                    s = res
                if not isinstance(s, pd.Series):
                    try:
                        s = pd.Series(s)
                    except Exception:
                        s = pd.Series(dtype=float)
                s.name = feat_name
                return s

        return True

    # Fallback: single output
    feat_name = f"ta_{cat}_{name}"

    @registry.register_feature(feat_name)
    def _feature(df: pd.DataFrame, feat_name=feat_name) -> pd.Series:
        kwargs = _build_kwargs(df, param_names)
        if cat == "candles":
            kwargs = _sanitize_ohlc_in_kwargs(kwargs)
            if "asbool" in [p.lower() for p in param_names] or name == "cdl_pattern":
                kwargs.setdefault("asbool", True)
        try:
            res = func(**kwargs)
        except TypeError:
            pos = [kwargs[k] for k in param_names if k in kwargs]
            res = func(*pos)
        s = res if isinstance(res, pd.Series) else pd.Series(res)
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

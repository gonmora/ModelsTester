# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple
import math
import pandas as pd
import numpy as np
from datetime import datetime

from .registry import registry


def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        return x.iloc[:, 0]
    return x


def _compute_feature(df: pd.DataFrame, fid: str) -> pd.Series:
    s = registry.features[fid](df)
    s = _as_series(s)
    s = pd.to_numeric(s, errors='coerce')
    return s


def _prefix_stability(df: pd.DataFrame, fid: str, checkpoints: List[float], guard: int, tol: float = 1e-9) -> Tuple[float, Dict[str, float]]:
    f_full = _compute_feature(df, fid)
    n = len(df)
    if n == 0 or f_full is None:
        return float('nan'), {}
    per: Dict[str, float] = {}
    tot_changes = 0
    tot_compared = 0
    for frac in checkpoints:
        K = int(math.floor(max(0.0, min(1.0, float(frac))) * n))
        if K <= max(5, guard):
            per[f"{frac:.2f}"] = float('nan')
            continue
        f_pref = _compute_feature(df.iloc[:K].copy(), fid)
        a = f_full.iloc[: K - guard]
        b = f_pref.iloc[: K - guard]
        m = pd.concat({'a': a, 'b': b}, axis=1).dropna()
        if len(m) == 0:
            per[f"{frac:.2f}"] = float('nan')
            continue
        diff = (m['a'] - m['b']).abs()
        changes = int((diff > tol).sum())
        compared = int(len(diff))
        tot_changes += changes
        tot_compared += compared
        per[f"{frac:.2f}"] = (changes / compared) if compared > 0 else float('nan')
    overall = (tot_changes / tot_compared) if tot_compared > 0 else float('nan')
    return overall, per


def _leakage_proxy(df: pd.DataFrame, x: pd.Series) -> float:
    try:
        close = pd.to_numeric(df['close'], errors='coerce').astype(float)
    except Exception:
        return float('nan')
    ret_fut = close.pct_change(1).shift(-1)
    ret_pst = close.pct_change(1).shift(1)
    def ic(y, z) -> float:
        m = pd.concat([y, z], axis=1).dropna()
        if len(m) < 50 or m.iloc[:, 0].std(ddof=0) == 0 or m.iloc[:, 1].std(ddof=0) == 0:
            return float('nan')
        from scipy.stats import spearmanr
        return float(spearmanr(m.iloc[:, 0], m.iloc[:, 1], nan_policy='omit').statistic)
    ic_f = ic(x, ret_fut)
    ic_p = ic(x, ret_pst)
    if math.isnan(ic_f) or math.isnan(ic_p):
        return float('nan')
    return float(ic_f - ic_p)


def audit_feature(df: pd.DataFrame, fid: str, checkpoints: List[float] | None = None, guard: int = 50) -> Dict[str, object]:
    """Run quick look-ahead checks for a feature and return a compact report.

    Returns keys:
      - nan_rate, prefix_mismatch_rate, leak_score, flag_prefix, flag_leak, checked_at
    """
    if checkpoints is None:
        checkpoints = [0.6, 0.8, 0.9]
    x = _compute_feature(df, fid)
    nan_rate = float(x.isna().mean()) if len(x) else float('nan')
    pmr, _ = _prefix_stability(df, fid, checkpoints, guard=guard)
    leak = _leakage_proxy(df, x)
    res = {
        'nan_rate': nan_rate,
        'prefix_mismatch_rate': pmr,
        'leak_score': leak,
        'flag_prefix': (float(pmr) > 0.0) if not math.isnan(pmr) else False,
        'flag_leak': (float(leak) > 0.2) if not math.isnan(leak) else False,
        'checked_at': datetime.utcnow().isoformat() + 'Z',
    }
    return res


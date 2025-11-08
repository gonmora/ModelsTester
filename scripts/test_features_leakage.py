#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test features for potential look-ahead/leakage issues.

Checks per feature:
  1) Prefix stability (causality proxy):
     - Recompute the feature on multiple prefixes of the DF.
     - Values well before the prefix end should NOT change when extending the data.
     - Reports a mismatch rate (fraction of values that changed beyond tolerance).

  2) Future-vs-Past IC (leakage proxy):
     - Spearman rho(x_t, ret_{t+1}) - rho(x_t, ret_{t-1}).
     - Large positive deltas suggest look-ahead.

Usage:
  python scripts/test_features_leakage.py --df-name BTCUSDT_5m_20230831_20250830 \
                                          --checkpoints 0.5 0.7 0.9 --guard 50 \
                                          --top 0  # 0 = all

Outputs a table and optionally saves CSV.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple
import math

import numpy as np
import pandas as pd

import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.registry import registry
from src import storage


def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # collapse to first column deterministically
        return x.iloc[:, 0]
    return x


def compute_feature(df: pd.DataFrame, fid: str) -> pd.Series:
    try:
        s = registry.features[fid](df)
        s = _as_series(s)
        s = pd.to_numeric(s, errors='coerce')
        return s
    except Exception as e:
        raise RuntimeError(f"Feature '{fid}' failed: {e}")


def prefix_stability(df: pd.DataFrame, fid: str, checkpoints: List[float], guard: int, tol: float = 1e-9) -> Tuple[float, Dict[str, float]]:
    """Return overall mismatch rate and per-checkpoint rates for a feature.

    For each prefix length K = floor(frac * N):
      - compute f_full on full DF, f_pref on DF[:K]
      - compare values up to K-guard (to avoid boundary/rollover effects)
      - mismatch if |full - pref| > tol (ignoring NaNs)
    """
    f_full = compute_feature(df, fid)
    n = len(df)
    if n == 0 or f_full is None:
        return float('nan'), {}
    per: Dict[str, float] = {}
    tot_changes = 0
    tot_compared = 0
    for frac in checkpoints:
        K = int(math.floor(max(0.0, min(1.0, frac)) * n))
        if K <= max(5, guard):
            per[f"{frac:.2f}"] = float('nan')
            continue
        f_pref = compute_feature(df.iloc[:K].copy(), fid)
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


def leakage_proxy(df: pd.DataFrame, x: pd.Series) -> float:
    """Spearman IC with future return minus IC with past return (1 step).

    Positive values suggest x_t aligns more with the future than with the past.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description='Test features for look-ahead/leakage')
    ap.add_argument('--df-name', type=str, default=os.environ.get('DF_NAME', 'BTCUSDT_5m_20230831_20250830'))
    ap.add_argument('--checkpoints', type=float, nargs='*', default=[0.5, 0.7, 0.9])
    ap.add_argument('--guard', type=int, default=50, help='Bars near prefix end to ignore in comparisons')
    ap.add_argument('--top', type=int, default=0, help='Limit number of features (0 = all)')
    ap.add_argument('--save-csv', type=str, default=None)
    args = ap.parse_args()

    # Load DF and ensure registry (TA) is available
    df = storage.load_dataframe(args.df_name)
    try:
        import src.ta_features  # noqa: F401
    except Exception:
        pass

    feats = list(registry.features.keys())
    if args.top and args.top > 0:
        feats = feats[: int(args.top)]

    rows = []
    for fid in feats:
        try:
            x = compute_feature(df, fid)
            nan_rate = float(x.isna().mean()) if len(x) else float('nan')
            overall, per = prefix_stability(df, fid, args.checkpoints, guard=args.guard)
            leak = leakage_proxy(df, x)
            rows.append({
                'feature': fid,
                'nan_rate': nan_rate,
                'prefix_mismatch_rate': overall,
                'leak_score': leak,
                **{f'mismatch_{k}': v for k, v in per.items()},
            })
        except Exception as e:
            rows.append({'feature': fid, 'error': str(e)})

    cols = ['feature','nan_rate','prefix_mismatch_rate','leak_score'] + [f'mismatch_{f:.2f}' for f in args.checkpoints]
    df_out = pd.DataFrame(rows)
    # Order columns if present
    present_cols = [c for c in cols if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in present_cols]
    df_out = df_out[present_cols + other_cols]

    # Heuristic flags
    def _flag_prefix(x: float) -> bool:
        try:
            return (float(x) > 0.0)
        except Exception:
            return False
    def _flag_leak(x: float) -> bool:
        try:
            return (float(x) > 0.2)
        except Exception:
            return False
    if 'prefix_mismatch_rate' in df_out.columns:
        df_out['flag_prefix'] = df_out['prefix_mismatch_rate'].apply(_flag_prefix)
    if 'leak_score' in df_out.columns:
        df_out['flag_leak'] = df_out['leak_score'].apply(_flag_leak)

    with pd.option_context('display.max_rows', None, 'display.width', 140):
        print(df_out)
    if args.save_csv:
        try:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            df_out.to_csv(args.save_csv, index=False)
            print(f"Saved: {args.save_csv}")
        except Exception as e:
            print(f"Save CSV error: {e}")


if __name__ == '__main__':
    main()


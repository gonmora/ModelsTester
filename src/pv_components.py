# -*- coding: utf-8 -*-
"""
Peaks & Valleys derived features and targets.

This module wraps `data_process.peaks_and_valleys.peaks_and_valleys` and
registers a handful of causal features (retrospective-only) and several
targets (prospective-only) in the project registry.

Conventions:
- Features return fractions (not percent). E.g., 0.012 = 1.2%.
- Some features include clipped variants to improve robustness to outliers.
- Targets are binary indicators derived from the prospective signals, or
  time-to-event targets converted to binary within a fixed horizon.

Note: The external dependency is imported lazily inside each function
to avoid import errors in environments where the library is not present.
"""
from __future__ import annotations

from typing import Callable
import os
import numpy as np
import pandas as pd

from .registry import registry
from .storage import DATA_DIR
import os


_PV_MEMO: dict[str, pd.DataFrame] = {}


def _pv(df: pd.DataFrame) -> pd.DataFrame:
    """Safe wrapper to compute peaks_and_valleys on a copy of df.

    Returns a dataframe with at least the columns:
    up, down, last_up, last_down, to_next_peak, to_next_valley,
    since_prev_peak, since_prev_valley, peak, valley.
    """
    # 1) Disk cache by df_name if available
    df_name = None
    try:
        df_name = getattr(df, 'attrs', {}).get('__df_name__')
    except Exception:
        df_name = None
    cache_cols = [
        'up','down','last_up','last_down',
        'to_next_peak','to_next_valley','since_prev_peak','since_prev_valley',
        'peak','valley','line'
    ]
    if df_name:
        key = str(df_name)
        if key in _PV_MEMO:
            return _PV_MEMO[key]
        path = os.path.join(DATA_DIR, f"{df_name}__artifact__pv_base.parquet")
        if os.path.exists(path):
            try:
                out = pd.read_parquet(path)
                _PV_MEMO[key] = out
                return out
            except Exception:
                pass
    # 2) Compute if cache miss
    try:
        from data_process.peaks_and_valleys import peaks_and_valleys
    except Exception as e:
        raise ImportError(
            "data_process.peaks_and_valleys is required for pv_* components."
        ) from e
    out_full = peaks_and_valleys(df.copy())
    if not isinstance(out_full, pd.DataFrame):
        raise RuntimeError("peaks_and_valleys() did not return a DataFrame")
    # Keep only needed columns; fill missing with NaN/False
    out = pd.DataFrame(index=out_full.index)
    for c in cache_cols:
        if c in out_full.columns:
            out[c] = out_full[c]
        else:
            out[c] = np.nan if c not in ('peak','valley') else False
    # Save to disk and memoize
    if df_name:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, f"{df_name}__artifact__pv_base.parquet")
            out.to_parquet(path)
        except Exception:
            pass
        _PV_MEMO[str(df_name)] = out
    return out


def _to_fraction(x: pd.Series) -> pd.Series:
    try:
        y = x.astype(float) / 100.0
    except Exception:
        y = pd.to_numeric(x, errors='coerce') / 100.0
    return y


def _clip_frac(x: pd.Series, lo: float = -0.20, hi: float = 0.20) -> pd.Series:
    try:
        return x.clip(lower=lo, upper=hi)
    except Exception:
        arr = pd.to_numeric(x, errors='coerce')
    return arr.clip(lower=lo, upper=hi)


"""
Important: Peak/valley detection typically needs future data to confirm
turning points. Using those confirmed extremes as inputs can introduce
indirect look-ahead when used as features.

Therefore, by default we DO NOT register pv_* features. You can enable them
explicitly by setting environment variable PV_ENABLE_FEATURES=1 before
importing the engine/registry. Targets remain registered as they are by
definition prospective labels.
"""

_PV_ENABLE_FEATURES = os.environ.get("PV_ENABLE_FEATURES", "0").lower() in ("1", "true", "yes")

if _PV_ENABLE_FEATURES:
    # ---------------------------
    # Features (retrospective, but rely on confirmed extremes)
    # ---------------------------
    @registry.register_feature('pv_last_up')
    def pv_last_up(df: pd.DataFrame) -> pd.Series:
        out = _pv(df)
        s = _to_fraction(out['last_up'])
        s.name = 'pv_last_up'
        return s

    @registry.register_feature('pv_last_up_clip20')
    def pv_last_up_clip20(df: pd.DataFrame) -> pd.Series:
        s = pv_last_up(df)
        s = _clip_frac(s, -0.20, 0.20)
        s.name = 'pv_last_up_clip20'
        return s

    @registry.register_feature('pv_last_down')
    def pv_last_down(df: pd.DataFrame) -> pd.Series:
        out = _pv(df)
        s = _to_fraction(out['last_down'])
        s.name = 'pv_last_down'
        return s

    @registry.register_feature('pv_last_down_clip20')
    def pv_last_down_clip20(df: pd.DataFrame) -> pd.Series:
        s = pv_last_down(df)
        s = _clip_frac(s, -0.20, 0.20)
        s.name = 'pv_last_down_clip20'
        return s

    @registry.register_feature('pv_since_prev_peak')
    def pv_since_prev_peak(df: pd.DataFrame) -> pd.Series:
        out = _pv(df)
        s = out['since_prev_peak'].astype(float)
        s.name = 'pv_since_prev_peak'
        return s

    @registry.register_feature('pv_since_prev_valley')
    def pv_since_prev_valley(df: pd.DataFrame) -> pd.Series:
        out = _pv(df)
        s = out['since_prev_valley'].astype(float)
        s.name = 'pv_since_prev_valley'
        return s

    @registry.register_feature('pv_last_trend')
    def pv_last_trend(df: pd.DataFrame) -> pd.Series:
        """Difference between last_up and last_down (fractions)."""
        out = _pv(df)
        s = _to_fraction(out['last_up']) - _to_fraction(out['last_down'])
        s.name = 'pv_last_trend'
        return s


# ---------------------------
# Targets (prospective)
# ---------------------------

def _register_up_down_threshold(name: str, column: str, thr_frac: float) -> Callable[[pd.DataFrame], pd.Series]:
    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df)
        base = _to_fraction(out[column])  # convert % to fraction
        y = (base >= float(thr_frac)).astype(int)
        y.name = name
        return y
    return _fn


def _register_next_event_horizon(name: str, column: str, horizon: int) -> Callable[[pd.DataFrame], pd.Series]:
    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df)
        t = pd.to_numeric(out[column], errors='coerce')
        y = (t <= int(horizon)).astype(int)
        y.name = name
        return y
    return _fn


# Binary up/down magnitude thresholds (fractions)
_register_up_down_threshold('pv_up_ge_0p5pct',  'up',   0.005)
_register_up_down_threshold('pv_up_ge_0p75pct', 'up',   0.0075)
_register_up_down_threshold('pv_up_ge_1pct',    'up',   0.010)
_register_up_down_threshold('pv_up_ge_1p5pct',  'up',   0.015)
_register_up_down_threshold('pv_up_ge_2pct',    'up',   0.020)
_register_up_down_threshold('pv_down_ge_0p5pct','down', 0.005)
_register_up_down_threshold('pv_down_ge_0p75pct','down',0.0075)
_register_up_down_threshold('pv_down_ge_1pct',  'down', 0.010)
_register_up_down_threshold('pv_down_ge_1p5pct','down', 0.015)
_register_up_down_threshold('pv_down_ge_2pct',  'down', 0.020)


# Time-to-event horizons
_register_next_event_horizon('pv_next_peak_in_20',   'to_next_peak', 20)
_register_next_event_horizon('pv_next_peak_in_50',   'to_next_peak', 50)
_register_next_event_horizon('pv_next_peak_in_100',  'to_next_peak', 100)
_register_next_event_horizon('pv_next_peak_in_200',  'to_next_peak', 200)
_register_next_event_horizon('pv_next_valley_in_20', 'to_next_valley', 20)
_register_next_event_horizon('pv_next_valley_in_50', 'to_next_valley', 50)
_register_next_event_horizon('pv_next_valley_in_100','to_next_valley', 100)
_register_next_event_horizon('pv_next_valley_in_200','to_next_valley', 200)


@registry.register_target('pv_next_extreme_is_peak')
def pv_next_extreme_is_peak(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    tnp = pd.to_numeric(out['to_next_peak'], errors='coerce').to_numpy()
    tnv = pd.to_numeric(out['to_next_valley'], errors='coerce').to_numpy()
    y = ((~np.isnan(tnp)) & (np.isnan(tnv) | (tnp <= tnv))).astype(int)
    return pd.Series(y, index=out.index, name='pv_next_extreme_is_peak')


# ---------------------------
# Additional targets from retrospective signals
# (allowed as targets; we disabled them as features by default)
# ---------------------------

def _register_last_threshold(name: str, column: str, thr_frac: float) -> Callable[[pd.DataFrame], pd.Series]:
    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df)
        base = _to_fraction(out[column])
        y = (base >= float(thr_frac)).astype(int)
        y.name = name
        return y
    return _fn


def _register_since_threshold(name: str, column: str, min_bars: int) -> Callable[[pd.DataFrame], pd.Series]:
    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df)
        t = pd.to_numeric(out[column], errors='coerce')
        y = (t >= int(min_bars)).astype(int)
        y.name = name
        return y
    return _fn


# last_up / last_down thresholds (fractions)
_register_last_threshold('pv_last_up_ge_0p5pct',   'last_up',   0.005)
_register_last_threshold('pv_last_up_ge_1pct',     'last_up',   0.010)
_register_last_threshold('pv_last_down_ge_0p5pct', 'last_down', 0.005)
_register_last_threshold('pv_last_down_ge_1pct',   'last_down', 0.010)


# Since thresholds (trend age)
_register_since_threshold('pv_since_prev_peak_ge_50',   'since_prev_peak',   50)
_register_since_threshold('pv_since_prev_valley_ge_50', 'since_prev_valley', 50)


@registry.register_target('pv_phase_up')
def pv_phase_up(df: pd.DataFrame) -> pd.Series:
    """1 si last_up >= last_down; 0 en caso contrario."""
    out = _pv(df)
    a = _to_fraction(out['last_up'])
    b = _to_fraction(out['last_down'])
    y = (a >= b).astype(int)
    y.name = 'pv_phase_up'
    return y


# ---------------------------
# Continuous targets (for regression)
# ---------------------------

def _clip_upper_q(s: pd.Series, q: float = 0.99, nonneg: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    try:
        hi = float(s.quantile(min(max(q, 0.5), 0.999)))
    except Exception:
        hi = float('nan')
    lo = 0.0 if nonneg else float('nan')
    if nonneg:
        s = s.clip(lower=0.0)
    try:
        if hi == hi:  # not NaN
            s = s.clip(upper=hi)
    except Exception:
        pass
    return s


def _clip_symmetric_q(s: pd.Series, q: float = 0.99) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    try:
        a = float(s.abs().quantile(min(max(q, 0.5), 0.999)))
        return s.clip(lower=-a, upper=a)
    except Exception:
        return s


@registry.register_target('pv_up_cont')
def pv_up_cont(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    s = _to_fraction(out['up'])
    s = _clip_upper_q(s, q=0.99, nonneg=True)
    s.name = 'pv_up_cont'
    return s


@registry.register_target('pv_down_cont')
def pv_down_cont(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    s = _to_fraction(out['down'])
    s = _clip_upper_q(s, q=0.99, nonneg=True)
    s.name = 'pv_down_cont'
    return s


@registry.register_target('pv_line_cont')
def pv_line_cont(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    up = _to_fraction(out['up'])
    dn = _to_fraction(out['down'])
    s = (up - dn)
    s = _clip_symmetric_q(s, q=0.99)
    s.name = 'pv_line_cont'
    return s


@registry.register_target('pv_tnext_peak_log')
def pv_tnext_peak_log(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    t = pd.to_numeric(out['to_next_peak'], errors='coerce')
    s = np.log1p(t.clip(lower=0))
    try:
        s = s.clip(upper=float(s.quantile(0.99)))
    except Exception:
        pass
    s.name = 'pv_tnext_peak_log'
    return s


@registry.register_target('pv_tnext_valley_log')
def pv_tnext_valley_log(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    t = pd.to_numeric(out['to_next_valley'], errors='coerce')
    s = np.log1p(t.clip(lower=0))
    try:
        s = s.clip(upper=float(s.quantile(0.99)))
    except Exception:
        pass
    s.name = 'pv_tnext_valley_log'
    return s


@registry.register_target('pv_cycle_pos')
def pv_cycle_pos(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    a = pd.to_numeric(out['since_prev_valley'], errors='coerce').clip(lower=0)
    b = pd.to_numeric(out['to_next_peak'], errors='coerce').clip(lower=0)
    s = a / (a + b + 1e-9)
    s.name = 'pv_cycle_pos'
    return s


@registry.register_target('pv_last_up_cont')
def pv_last_up_cont(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    s = _to_fraction(out['last_up'])
    s = _clip_upper_q(s, q=0.99, nonneg=True)
    s.name = 'pv_last_up_cont'
    return s


@registry.register_target('pv_last_down_cont')
def pv_last_down_cont(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    s = _to_fraction(out['last_down'])
    s = _clip_upper_q(s, q=0.99, nonneg=True)
    s.name = 'pv_last_down_cont'
    return s


@registry.register_target('pv_phase_ratio')
def pv_phase_ratio(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    a = _to_fraction(out['last_up']).clip(lower=0)
    b = _to_fraction(out['last_down']).clip(lower=0)
    s = a / (a + b + 1e-9)
    s.name = 'pv_phase_ratio'
    return s

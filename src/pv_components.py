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

from typing import Any, Callable
import os
import numpy as np
import pandas as pd

from .registry import registry
from .storage import DATA_DIR


_PV_MEMO: dict[tuple[str, str], pd.DataFrame] = {}


def _format_threshold_tag(value: Any) -> str:
    """Format numeric threshold into a cache/tag-friendly suffix."""
    try:
        v = float(value)
    except Exception:
        return str(value).replace('.', 'p').replace('-', 'm')
    text = f"{v:g}"
    return text.replace('.', 'p').replace('-', 'm')

def _artifact_is_stale(df: pd.DataFrame, art: pd.DataFrame) -> bool:
    """Heuristics to decide if a cached PV artifact is stale for the given df.

    - Mismatch in length (artifact shorter than df)
    - Mismatch in last index label (if both are DatetimeIndex or comparable)
    """
    try:
        if len(art) < len(df):
            return True
        di = getattr(df, 'index', None)
        ai = getattr(art, 'index', None)
        if di is not None and ai is not None and len(di) and len(ai):
            try:
                if di[-1] != ai[-1]:
                    return True
            except Exception:
                pass
    except Exception:
        return True
    return False


def _pv(
    df: pd.DataFrame,
    *,
    distance: int = 1000,
    threshold: float = 2.0,
    cache_tag: str | None = None,
) -> pd.DataFrame:
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
    if cache_tag is None:
        if int(distance) == 1000 and float(threshold) == 2.0:
            cache_tag = 'pv_base'
        else:
            cache_tag = f"pv_d{int(distance)}_t{_format_threshold_tag(threshold)}"

    if df_name:
        key = (str(df_name), cache_tag)
        if key in _PV_MEMO:
            memo = _PV_MEMO[key]
            if not _artifact_is_stale(df, memo):
                return memo
            # stale -> drop and recompute
            try:
                del _PV_MEMO[key]
            except Exception:
                pass
        path = os.path.join(DATA_DIR, f"{df_name}__artifact__{cache_tag}.parquet")
        if os.path.exists(path):
            try:
                out = pd.read_parquet(path)
                if not _artifact_is_stale(df, out):
                    _PV_MEMO[key] = out
                    return out
                # else: fall through to recompute
            except Exception:
                pass
    # 2) Compute if cache miss
    try:
        from data_process.peaks_and_valleys import peaks_and_valleys
    except Exception as e:
        raise ImportError(
            "data_process.peaks_and_valleys is required for pv_* components."
        ) from e
    out_full = peaks_and_valleys(df.copy(), distance=distance, threshold=threshold)
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
            path = os.path.join(DATA_DIR, f"{df_name}__artifact__{cache_tag}.parquet")
            out.to_parquet(path)
        except Exception:
            pass
        _PV_MEMO[(str(df_name), cache_tag)] = out
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

def _register_up_down_threshold(
    name: str,
    column: str,
    thr_frac: float,
    *,
    pv_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    params = dict(pv_kwargs or {})

    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df, **params)
        base = _to_fraction(out[column])  # convert % to fraction
        y = (base >= float(thr_frac)).astype(int)
        y.name = name
        return y
    return _fn


def _register_next_event_horizon(
    name: str,
    column: str,
    horizon: int,
    *,
    pv_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    params = dict(pv_kwargs or {})

    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df, **params)
        t = pd.to_numeric(out[column], errors='coerce')
        y = (t <= int(horizon)).astype(int)
        y.name = name
        return y
    return _fn


# Binary up/down magnitude thresholds (fractions)
_UP_DOWN_THRESHOLD_SPECS = [
    ('up_ge_0p5pct', 'up', 0.005),
    ('up_ge_0p75pct', 'up', 0.0075),
    ('up_ge_1pct', 'up', 0.010),
    ('up_ge_1p5pct', 'up', 0.015),
    ('up_ge_2pct', 'up', 0.020),
    ('down_ge_0p5pct', 'down', 0.005),
    ('down_ge_0p75pct', 'down', 0.0075),
    ('down_ge_1pct', 'down', 0.010),
    ('down_ge_1p5pct', 'down', 0.015),
    ('down_ge_2pct', 'down', 0.020),
]

for suffix, column, threshold_value in _UP_DOWN_THRESHOLD_SPECS:
    _register_up_down_threshold(f'pv_{suffix}', column, threshold_value)


# Time-to-event horizons
_NEXT_EVENT_SPECS = [
    ('next_peak_in_20', 'to_next_peak', 20),
    ('next_peak_in_50', 'to_next_peak', 50),
    ('next_peak_in_100', 'to_next_peak', 100),
    ('next_peak_in_200', 'to_next_peak', 200),
    ('next_valley_in_20', 'to_next_valley', 20),
    ('next_valley_in_50', 'to_next_valley', 50),
    ('next_valley_in_100', 'to_next_valley', 100),
    ('next_valley_in_200', 'to_next_valley', 200),
]

for suffix, column, horizon_value in _NEXT_EVENT_SPECS:
    _register_next_event_horizon(f'pv_{suffix}', column, horizon_value)


@registry.register_target('pv_next_extreme_is_peak')
def pv_next_extreme_is_peak(df: pd.DataFrame) -> pd.Series:
    out = _pv(df)
    tnp = pd.to_numeric(out['to_next_peak'], errors='coerce').to_numpy()
    tnv = pd.to_numeric(out['to_next_valley'], errors='coerce').to_numpy()
    y = ((~np.isnan(tnp)) & (np.isnan(tnv) | (tnp <= tnv))).astype(int)
    return pd.Series(y, index=out.index, name='pv_next_extreme_is_peak')


@registry.register_target('pv_next_extreme_is_valley')
def pv_next_extreme_is_valley(df: pd.DataFrame) -> pd.Series:
    """1 si el próximo extremo confirmado es un valley; 0 en caso contrario.

    Es el complemento de pv_next_extreme_is_peak cuando alguno de los dos está definido.
    """
    out = _pv(df)
    tnp = pd.to_numeric(out['to_next_peak'], errors='coerce').to_numpy()
    tnv = pd.to_numeric(out['to_next_valley'], errors='coerce').to_numpy()
    y = ((~np.isnan(tnv)) & (np.isnan(tnp) | (tnv < tnp))).astype(int)
    return pd.Series(y, index=out.index, name='pv_next_extreme_is_valley')


# ---------------------------
# Additional targets from retrospective signals
# (allowed as targets; we disabled them as features by default)
# ---------------------------

def _register_last_threshold(
    name: str,
    column: str,
    thr_frac: float,
    *,
    pv_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    params = dict(pv_kwargs or {})

    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df, **params)
        base = _to_fraction(out[column])
        y = (base >= float(thr_frac)).astype(int)
        y.name = name
        return y
    return _fn


def _register_since_threshold(
    name: str,
    column: str,
    min_bars: int,
    *,
    pv_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    params = dict(pv_kwargs or {})

    @registry.register_target(name)
    def _fn(df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
        out = _pv(df, **params)
        t = pd.to_numeric(out[column], errors='coerce')
        y = (t >= int(min_bars)).astype(int)
        y.name = name
        return y
    return _fn


# last_up / last_down thresholds (fractions)
_LAST_THRESHOLD_SPECS = [
    ('last_up_ge_0p5pct', 'last_up', 0.005),
    ('last_up_ge_1pct', 'last_up', 0.010),
    ('last_down_ge_0p5pct', 'last_down', 0.005),
    ('last_down_ge_1pct', 'last_down', 0.010),
]

for suffix, column, threshold_value in _LAST_THRESHOLD_SPECS:
    _register_last_threshold(f'pv_{suffix}', column, threshold_value)


# Since thresholds (trend age)
_SINCE_THRESHOLD_SPECS = [
    ('since_prev_peak_ge_50', 'since_prev_peak', 50),
    ('since_prev_valley_ge_50', 'since_prev_valley', 50),
]

for suffix, column, min_bars_value in _SINCE_THRESHOLD_SPECS:
    _register_since_threshold(f'pv_{suffix}', column, min_bars_value)


@registry.register_target('pv_phase_up')
def pv_phase_up(df: pd.DataFrame) -> pd.Series:
    """1 si last_up >= last_down; 0 en caso contrario."""
    out = _pv(df)
    a = _to_fraction(out['last_up'])
    b = _to_fraction(out['last_down'])
    y = (a >= b).astype(int)
    y.name = 'pv_phase_up'
    return y


@registry.register_target('pv_phase_down')
def pv_phase_down(df: pd.DataFrame) -> pd.Series:
    """1 si last_down >= last_up; 0 en caso contrario.

    Complemento binario de pv_phase_up.
    """
    out = _pv(df)
    a = _to_fraction(out['last_up'])
    b = _to_fraction(out['last_down'])
    y = (b >= a).astype(int)
    y.name = 'pv_phase_down'
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


def _register_configured_targets(base_name: str, pv_kwargs: dict[str, Any]) -> None:
    """Register a full family of pv_* targets for an alternate PV config."""
    params = dict(pv_kwargs)
    prefix = f"{base_name}_"

    for suffix, column, threshold_value in _UP_DOWN_THRESHOLD_SPECS:
        _register_up_down_threshold(
            f"{prefix}{suffix}", column, threshold_value, pv_kwargs=params
        )

    for suffix, column, horizon_value in _NEXT_EVENT_SPECS:
        _register_next_event_horizon(
            f"{prefix}{suffix}", column, horizon_value, pv_kwargs=params
        )

    for suffix, column, threshold_value in _LAST_THRESHOLD_SPECS:
        _register_last_threshold(
            f"{prefix}{suffix}", column, threshold_value, pv_kwargs=params
        )

    for suffix, column, min_bars_value in _SINCE_THRESHOLD_SPECS:
        _register_since_threshold(
            f"{prefix}{suffix}", column, min_bars_value, pv_kwargs=params
        )

    name_peak = f"{prefix}next_extreme_is_peak"

    @registry.register_target(name_peak)
    def _next_extreme_is_peak(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_peak,
    ) -> pd.Series:
        out = _pv(df, **_params)
        tnp = pd.to_numeric(out['to_next_peak'], errors='coerce').to_numpy()
        tnv = pd.to_numeric(out['to_next_valley'], errors='coerce').to_numpy()
        y = ((~np.isnan(tnp)) & (np.isnan(tnv) | (tnp <= tnv))).astype(int)
        return pd.Series(y, index=out.index, name=_name)

    name_valley = f"{prefix}next_extreme_is_valley"

    @registry.register_target(name_valley)
    def _next_extreme_is_valley(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_valley,
    ) -> pd.Series:
        out = _pv(df, **_params)
        tnp = pd.to_numeric(out['to_next_peak'], errors='coerce').to_numpy()
        tnv = pd.to_numeric(out['to_next_valley'], errors='coerce').to_numpy()
        y = ((~np.isnan(tnv)) & (np.isnan(tnp) | (tnv < tnp))).astype(int)
        return pd.Series(y, index=out.index, name=_name)

    name_phase_up = f"{prefix}phase_up"

    @registry.register_target(name_phase_up)
    def _phase_up(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_phase_up,
    ) -> pd.Series:
        out = _pv(df, **_params)
        a = _to_fraction(out['last_up'])
        b = _to_fraction(out['last_down'])
        y = (a >= b).astype(int)
        y.name = _name
        return y

    name_phase_down = f"{prefix}phase_down"

    @registry.register_target(name_phase_down)
    def _phase_down(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_phase_down,
    ) -> pd.Series:
        out = _pv(df, **_params)
        a = _to_fraction(out['last_up'])
        b = _to_fraction(out['last_down'])
        y = (b >= a).astype(int)
        y.name = _name
        return y

    name_up_cont = f"{prefix}up_cont"

    @registry.register_target(name_up_cont)
    def _up_cont(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_up_cont,
    ) -> pd.Series:
        out = _pv(df, **_params)
        s = _to_fraction(out['up'])
        s = _clip_upper_q(s, q=0.99, nonneg=True)
        s.name = _name
        return s

    name_down_cont = f"{prefix}down_cont"

    @registry.register_target(name_down_cont)
    def _down_cont(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_down_cont,
    ) -> pd.Series:
        out = _pv(df, **_params)
        s = _to_fraction(out['down'])
        s = _clip_upper_q(s, q=0.99, nonneg=True)
        s.name = _name
        return s

    name_line_cont = f"{prefix}line_cont"

    @registry.register_target(name_line_cont)
    def _line_cont(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_line_cont,
    ) -> pd.Series:
        out = _pv(df, **_params)
        up = _to_fraction(out['up'])
        dn = _to_fraction(out['down'])
        s = _clip_symmetric_q(up - dn, q=0.99)
        s.name = _name
        return s

    name_tnext_peak_log = f"{prefix}tnext_peak_log"

    @registry.register_target(name_tnext_peak_log)
    def _tnext_peak_log(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_tnext_peak_log,
    ) -> pd.Series:
        out = _pv(df, **_params)
        t = pd.to_numeric(out['to_next_peak'], errors='coerce')
        s = np.log1p(t.clip(lower=0))
        try:
            s = s.clip(upper=float(s.quantile(0.99)))
        except Exception:
            pass
        s.name = _name
        return s

    name_tnext_valley_log = f"{prefix}tnext_valley_log"

    @registry.register_target(name_tnext_valley_log)
    def _tnext_valley_log(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_tnext_valley_log,
    ) -> pd.Series:
        out = _pv(df, **_params)
        t = pd.to_numeric(out['to_next_valley'], errors='coerce')
        s = np.log1p(t.clip(lower=0))
        try:
            s = s.clip(upper=float(s.quantile(0.99)))
        except Exception:
            pass
        s.name = _name
        return s

    name_cycle_pos = f"{prefix}cycle_pos"

    @registry.register_target(name_cycle_pos)
    def _cycle_pos(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_cycle_pos,
    ) -> pd.Series:
        out = _pv(df, **_params)
        a = pd.to_numeric(out['since_prev_valley'], errors='coerce').clip(lower=0)
        b = pd.to_numeric(out['to_next_peak'], errors='coerce').clip(lower=0)
        s = a / (a + b + 1e-9)
        s.name = _name
        return s

    name_last_up_cont = f"{prefix}last_up_cont"

    @registry.register_target(name_last_up_cont)
    def _last_up_cont(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_last_up_cont,
    ) -> pd.Series:
        out = _pv(df, **_params)
        s = _to_fraction(out['last_up'])
        s = _clip_upper_q(s, q=0.99, nonneg=True)
        s.name = _name
        return s

    name_last_down_cont = f"{prefix}last_down_cont"

    @registry.register_target(name_last_down_cont)
    def _last_down_cont(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_last_down_cont,
    ) -> pd.Series:
        out = _pv(df, **_params)
        s = _to_fraction(out['last_down'])
        s = _clip_upper_q(s, q=0.99, nonneg=True)
        s.name = _name
        return s

    name_phase_ratio = f"{prefix}phase_ratio"

    @registry.register_target(name_phase_ratio)
    def _phase_ratio(
        df: pd.DataFrame,
        _params: dict[str, Any] = params,
        _name: str = name_phase_ratio,
    ) -> pd.Series:
        out = _pv(df, **_params)
        a = _to_fraction(out['last_up']).clip(lower=0)
        b = _to_fraction(out['last_down']).clip(lower=0)
        s = a / (a + b + 1e-9)
        s.name = _name
        return s


_ADDITIONAL_PV_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ('pv_d200_t2', {'distance': 200, 'threshold': 2, 'cache_tag': 'pv_d200_t2'}),
    ('pv_d200_t3', {'distance': 200, 'threshold': 3, 'cache_tag': 'pv_d200_t3'}),
]

for config_name, config_kwargs in _ADDITIONAL_PV_CONFIGS:
    _register_configured_targets(config_name, config_kwargs)

# -*- coding: utf-8 -*-
"""
Basic components (targets, features, models) for the ModelsTester project.

This module defines simple baseline target and feature functions and a basic
classification model. These are registered with the global registry on import.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .registry import registry

# --- Utilities for additional bases and horizons ---
_WINDOW_LABELS_TO_BARS = {
    # Assuming 5-minute bars
    "4h": 4 * 60 // 5,    # 48
    "6h": 6 * 60 // 5,    # 72
    "12h": 12 * 60 // 5,  # 144
    "2d": 2 * 24 * 60 // 5,   # 576
    "2w": 2 * 7 * 24 * 60 // 5,  # 4032
}


def _hlc3(df: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(df.get("high"), errors="coerce")
    l = pd.to_numeric(df.get("low"), errors="coerce")
    c = pd.to_numeric(df.get("close"), errors="coerce")
    return (h + l + c) / 3.0


def _vwap_series(df: pd.DataFrame) -> pd.Series:
    """Attempt to use registered pandas_ta VWAP; fallback to cumulative VWAP."""
    try:
        # Prefer pre-registered pandas_ta VWAP feature if available
        if "ta_volume_vwap" in registry.features:
            s = registry.features["ta_volume_vwap"](df)
            if isinstance(s, pd.Series) and s.notna().any():
                return pd.to_numeric(s, errors="coerce")
    except Exception:
        pass

    # Fallback: cumulative VWAP using typical price (HLC3)
    if "volume" not in df.columns:
        # Without volume, return HLC3 as a harmless fallback
        return _hlc3(df)
    tp = _hlc3(df)
    vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    num = (tp * vol).cumsum()
    den = vol.cumsum().replace(0.0, pd.NA)
    return (num / den).fillna(method="ffill").fillna(method="bfill")

# --- Custom distance-to-rolling-maximum features ---
def _dist_to_rolling_max(close: pd.Series, window: int, as_pct: bool = True) -> pd.Series:
    """Distance from current close to the rolling max over the last `window` bars.

    - as_pct=True: (roll_max - close) / roll_max, in [0, +inf) with 0 when at max.
      Safer across price regimes; returns 0 when roll_max is 0.
    - as_pct=False: (roll_max - close), absolute units.
    """
    roll_max = close.rolling(window=window, min_periods=1).max()
    if as_pct:
        num = (roll_max - close)
        den = roll_max.replace(0, pd.NA)
        out = (num / den).fillna(0.0)
    else:
        out = (roll_max - close)
    return out


# Assuming 5-minute bars: 1d ~= 288 bars, 1w ~= 2016 bars
@registry.register_feature('dist_to_max_1d_pct')
def dist_to_max_1d_pct(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_max(df['close'], window=288, as_pct=True)
    x.name = 'dist_to_max_1d_pct'
    return x


@registry.register_feature('dist_to_max_1w_pct')
def dist_to_max_1w_pct(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_max(df['close'], window=2016, as_pct=True)
    x.name = 'dist_to_max_1w_pct'
    return x


@registry.register_feature('dist_to_max_1d_abs')
def dist_to_max_1d_abs(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_max(df['close'], window=288, as_pct=False)
    x.name = 'dist_to_max_1d_abs'
    return x


@registry.register_feature('dist_to_max_1w_abs')
def dist_to_max_1w_abs(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_max(df['close'], window=2016, as_pct=False)
    x.name = 'dist_to_max_1w_abs'
    return x


# Symmetric distance-to-minimum features
def _dist_to_rolling_min(close: pd.Series, window: int, as_pct: bool = True) -> pd.Series:
    roll_min = close.rolling(window=window, min_periods=1).min()
    if as_pct:
        num = (close - roll_min)
        den = roll_min.replace(0, pd.NA)
        out = (num / den).fillna(0.0)
    else:
        out = (close - roll_min)
    return out


@registry.register_feature('dist_to_min_1d_pct')
def dist_to_min_1d_pct(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_min(df['close'], window=288, as_pct=True)
    x.name = 'dist_to_min_1d_pct'
    return x


@registry.register_feature('dist_to_min_1w_pct')
def dist_to_min_1w_pct(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_min(df['close'], window=2016, as_pct=True)
    x.name = 'dist_to_min_1w_pct'
    return x


@registry.register_feature('dist_to_min_1d_abs')
def dist_to_min_1d_abs(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_min(df['close'], window=288, as_pct=False)
    x.name = 'dist_to_min_1d_abs'
    return x


@registry.register_feature('dist_to_min_1w_abs')
def dist_to_min_1w_abs(df: pd.DataFrame) -> pd.Series:
    x = _dist_to_rolling_min(df['close'], window=2016, as_pct=False)
    x.name = 'dist_to_min_1w_abs'
    return x


# Bars-since-last-extreme features
def _bars_since_rolling_extreme(close: pd.Series, window: int, which: str = 'max') -> pd.Series:
    import numpy as np
    s = pd.to_numeric(close, errors='coerce')
    func = np.nanargmax if which == 'max' else np.nanargmin

    def _since(a: np.ndarray) -> float:
        finite = np.isfinite(a)
        if not finite.any():
            return float('nan')
        idx = int(func(a))
        return float(len(a) - 1 - idx)

    return s.rolling(window=window, min_periods=1).apply(_since, raw=True)


@registry.register_feature('bars_since_max_1d')
def bars_since_max_1d(df: pd.DataFrame) -> pd.Series:
    x = _bars_since_rolling_extreme(df['close'], window=288, which='max')
    x.name = 'bars_since_max_1d'
    return x


@registry.register_feature('bars_since_max_1w')
def bars_since_max_1w(df: pd.DataFrame) -> pd.Series:
    x = _bars_since_rolling_extreme(df['close'], window=2016, which='max')
    x.name = 'bars_since_max_1w'
    return x


@registry.register_feature('bars_since_min_1d')
def bars_since_min_1d(df: pd.DataFrame) -> pd.Series:
    x = _bars_since_rolling_extreme(df['close'], window=288, which='min')
    x.name = 'bars_since_min_1d'
    return x


@registry.register_feature('bars_since_min_1w')
def bars_since_min_1w(df: pd.DataFrame) -> pd.Series:
    x = _bars_since_rolling_extreme(df['close'], window=2016, which='min')
    x.name = 'bars_since_min_1w'
    return x


# --- Auto-register additional distance-to-extreme features ---
def _register_extreme_distance_variants() -> int:
    added = 0

    # 1) Close-based: new horizons (avoid duplicates if already exist)
    for lbl, w in _WINDOW_LABELS_TO_BARS.items():
        for kind in ("max", "min"):
            for mode in ("pct", "abs"):
                name = f"dist_to_{kind}_{lbl}_{mode}"
                if name in registry.features:
                    continue

                @registry.register_feature(name)
                def _f(df: pd.DataFrame, W=w, K=kind, MODE=mode, NAME=name) -> pd.Series:
                    s = pd.to_numeric(df["close"], errors="coerce")
                    if K == "max":
                        out = _dist_to_rolling_max(s, window=int(W), as_pct=(MODE == "pct"))
                    else:
                        out = _dist_to_rolling_min(s, window=int(W), as_pct=(MODE == "pct"))
                    out.name = NAME
                    return out
                added += 1

    # 2) HLC3-based variants (prefixed with base)
    for lbl, w in _WINDOW_LABELS_TO_BARS.items():
        for kind in ("max", "min"):
            for mode in ("pct", "abs"):
                name = f"hlc3_dist_to_{kind}_{lbl}_{mode}"
                if name in registry.features:
                    continue

                @registry.register_feature(name)
                def _f(df: pd.DataFrame, W=w, K=kind, MODE=mode, NAME=name) -> pd.Series:
                    s = _hlc3(df)
                    if K == "max":
                        out = _dist_to_rolling_max(s, window=int(W), as_pct=(MODE == "pct"))
                    else:
                        out = _dist_to_rolling_min(s, window=int(W), as_pct=(MODE == "pct"))
                    out.name = NAME
                    return out
                added += 1

    # 3) VWAP-based variants (prefixed with base)
    for lbl, w in _WINDOW_LABELS_TO_BARS.items():
        for kind in ("max", "min"):
            for mode in ("pct", "abs"):
                name = f"vwap_dist_to_{kind}_{lbl}_{mode}"
                if name in registry.features:
                    continue

                @registry.register_feature(name)
                def _f(df: pd.DataFrame, W=w, K=kind, MODE=mode, NAME=name) -> pd.Series:
                    s = _vwap_series(df)
                    if K == "max":
                        out = _dist_to_rolling_max(s, window=int(W), as_pct=(MODE == "pct"))
                    else:
                        out = _dist_to_rolling_min(s, window=int(W), as_pct=(MODE == "pct"))
                    out.name = NAME
                    return out
                added += 1

    # 4) Bars-since-extreme for new horizons (close-based)
    for lbl, w in _WINDOW_LABELS_TO_BARS.items():
        for kind in ("max", "min"):
            name = f"bars_since_{kind}_{lbl}"
            if name in registry.features:
                continue

            @registry.register_feature(name)
            def _f(df: pd.DataFrame, W=w, K=kind, NAME=name) -> pd.Series:
                s = pd.to_numeric(df["close"], errors="coerce")
                out = _bars_since_rolling_extreme(s, window=int(W), which=str(K))
                out.name = NAME
                return out
            added += 1

    return added


# Try to register on import without failing
try:
    _register_extreme_distance_variants()
except Exception:
    pass


# --- Daily volume sum features (calendar-based) ---
def _ensure_datetime_index(df: pd.DataFrame) -> pd.Index:
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        idx = df.index
    return idx


def _daily_volume_sum_series(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_cal')
    idx = _ensure_datetime_index(df)
    day = idx.floor('D')
    # Sum per day, repeated across bars of that day
    vol_sum_day = df['volume'].groupby(day).transform('sum')
    vol_sum_day.name = 'vol_sum_1d_cal'
    return vol_sum_day


@registry.register_feature('vol_sum_1d_cal')
def vol_sum_1d_cal(df: pd.DataFrame) -> pd.Series:
    return _daily_volume_sum_series(df)


@registry.register_feature('vol_sum_1d_change_abs')
def vol_sum_1d_change_abs(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_change_abs')
    idx = _ensure_datetime_index(df)
    day = idx.floor('D')
    vol_sum_unique = df.groupby(day)['volume'].sum()
    prev = vol_sum_unique.shift(1)
    # Map previous day sum to each bar of the current day (use Series.map for label mapping)
    day_s = pd.Series(day, index=df.index)
    prev_map = day_s.map(prev)
    cur_rep = _daily_volume_sum_series(df)
    out = (cur_rep - prev_map).fillna(0.0)
    out.name = 'vol_sum_1d_change_abs'
    return out


@registry.register_feature('vol_sum_1d_change_pct')
def vol_sum_1d_change_pct(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_change_pct')
    idx = _ensure_datetime_index(df)
    day = idx.floor('D')
    vol_sum_unique = df.groupby(day)['volume'].sum()
    prev = vol_sum_unique.shift(1)
    day_s = pd.Series(day, index=df.index)
    prev_map = day_s.map(prev)
    cur_rep = _daily_volume_sum_series(df)
    den = prev_map.replace(0, pd.NA)
    out = ((cur_rep - prev_map) / den).fillna(0.0)
    out.name = 'vol_sum_1d_change_pct'
    return out


# Weekly (7-day) max of daily volume sum and distances to it
@registry.register_feature('vol_sum_1w_max_cal')
def vol_sum_1w_max_cal(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1w_max_cal')
    idx = _ensure_datetime_index(df)
    day = idx.floor('D')
    vol_sum_unique = df.groupby(day)['volume'].sum()
    max_1w_unique = vol_sum_unique.rolling(window=7, min_periods=1).max()
    day_s = pd.Series(day, index=df.index)
    out = day_s.map(max_1w_unique)
    out = pd.to_numeric(out, errors='coerce').fillna(0.0)
    out.name = 'vol_sum_1w_max_cal'
    return out


@registry.register_feature('vol_sum_to_1w_max_abs')
def vol_sum_to_1w_max_abs(df: pd.DataFrame) -> pd.Series:
    cur = vol_sum_1d_cal(df)
    wk = vol_sum_1w_max_cal(df)
    out = (wk - cur)
    out.name = 'vol_sum_to_1w_max_abs'
    return out


@registry.register_feature('vol_sum_to_1w_max_pct')
def vol_sum_to_1w_max_pct(df: pd.DataFrame) -> pd.Series:
    cur = vol_sum_1d_cal(df)
    wk = vol_sum_1w_max_cal(df)
    den = wk.replace(0, pd.NA)
    out = ((wk - cur) / den).fillna(0.0)
    out.name = 'vol_sum_to_1w_max_pct'
    return out


# Rolling 1-day (window-based) volume sum features (assuming 5m bars: 288 per day)
@registry.register_feature('vol_sum_1d_roll')
def vol_sum_1d_roll(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_roll')
    x = df['volume'].rolling(window=288, min_periods=1).sum()
    x.name = 'vol_sum_1d_roll'
    return x


@registry.register_feature('vol_sum_1d_roll_change_abs')
def vol_sum_1d_roll_change_abs(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_roll_change_abs')
    roll = df['volume'].rolling(window=288, min_periods=1).sum()
    out = roll.diff().fillna(0.0)
    out.name = 'vol_sum_1d_roll_change_abs'
    return out


@registry.register_feature('vol_sum_1d_roll_change_pct')
def vol_sum_1d_roll_change_pct(df: pd.DataFrame) -> pd.Series:
    if 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index, name='vol_sum_1d_roll_change_pct')
    roll = df['volume'].rolling(window=288, min_periods=1).sum()
    out = roll.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out.name = 'vol_sum_1d_roll_change_pct'
    return out


# --- Time-based (UTC) features for 5-minute bars ---
def _dt_index_utc(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        try:
            idx = idx.tz_convert('UTC')
        except Exception:
            pass
    return idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx


@registry.register_feature('tod_sin')
def time_of_day_sin(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    minutes = idx.hour * 60 + idx.minute
    angle = 2 * np.pi * (minutes / 1440.0)
    x = np.sin(angle)
    return pd.Series(x, index=df.index, name='tod_sin')


@registry.register_feature('tod_cos')
def time_of_day_cos(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    minutes = idx.hour * 60 + idx.minute
    angle = 2 * np.pi * (minutes / 1440.0)
    x = np.cos(angle)
    return pd.Series(x, index=df.index, name='tod_cos')


@registry.register_feature('dow_sin')
def day_of_week_sin(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    dow = idx.weekday  # 0=Mon .. 6=Sun
    angle = 2 * np.pi * (dow / 7.0)
    x = np.sin(angle)
    return pd.Series(x, index=df.index, name='dow_sin')


@registry.register_feature('dow_cos')
def day_of_week_cos(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    dow = idx.weekday
    angle = 2 * np.pi * (dow / 7.0)
    x = np.cos(angle)
    return pd.Series(x, index=df.index, name='dow_cos')


@registry.register_feature('is_weekend')
def is_weekend_feature(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    dow = idx.weekday
    x = ((dow >= 5).astype(int))
    return pd.Series(x, index=df.index, name='is_weekend')


@registry.register_feature('hour_norm')
def hour_normalized(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    x = (idx.hour.astype(float) / 23.0)
    return pd.Series(x, index=df.index, name='hour_norm')


@registry.register_feature('is_month_start')
def is_month_start_feature(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    try:
        x = idx.is_month_start.astype(int)
    except Exception:
        x = pd.Series(0, index=df.index)
    return pd.Series(x, index=df.index, name='is_month_start')


@registry.register_feature('is_month_end')
def is_month_end_feature(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    try:
        x = idx.is_month_end.astype(int)
    except Exception:
        x = pd.Series(0, index=df.index)
    return pd.Series(x, index=df.index, name='is_month_end')


@registry.register_feature('is_quarter_start')
def is_quarter_start_feature(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    try:
        x = idx.is_quarter_start.astype(int)
    except Exception:
        x = pd.Series(0, index=df.index)
    return pd.Series(x, index=df.index, name='is_quarter_start')


@registry.register_feature('is_quarter_end')
def is_quarter_end_feature(df: pd.DataFrame) -> pd.Series:
    idx = _dt_index_utc(df)
    try:
        x = idx.is_quarter_end.astype(int)
    except Exception:
        x = pd.Series(0, index=df.index)
    return pd.Series(x, index=df.index, name='is_quarter_end')


def _ensure_utf8_locale() -> None:
    """Ensure subprocesses (SciPy/numpy checks) see a UTF-8 locale.

    Some SciPy/numpy imports run system commands (e.g., `lscpu`) with text=True and
    expect UTF-8 output. On systems with a different default encoding, this can raise
    UnicodeDecodeError. Setting LC_ALL/LANG to C.UTF-8 mitigates it.
    """
    import os
    for var in ("LC_ALL", "LANG", "LANGUAGE"):
        val = os.environ.get(var, "")
        if not val or ("utf" not in val.lower() and "utf-8" not in val.lower()):
            os.environ[var] = "C.UTF-8"
    os.environ.setdefault("PYTHONUTF8", "1")

def _time_stratified_split(y_series: pd.Series, min_train: int = 100, min_test: int = 100) -> int:
    """Choose a time-aware split index ensuring both sets have at least two classes.

    Scans several split fractions (making test larger if needed) and returns the
    first that yields >=2 unique labels in both train and test and meets size thresholds.
    Falls back to 80/20 if no suitable split is found.
    """
    n = int(len(y_series))
    if n <= max(min_train + min_test, 10):
        return int(n * 0.8)
    # Fractions from larger train to larger test
    for frac in (0.8, 0.75, 0.7, 0.65, 0.6):
        split_idx = int(n * frac)
        y_tr = y_series.iloc[:split_idx]
        y_te = y_series.iloc[split_idx:]
        if len(y_tr) < min_train or len(y_te) < min_test:
            continue
        if y_tr.dropna().nunique() >= 2 and y_te.dropna().nunique() >= 2:
            return split_idx
    return int(n * 0.8)

# Targets: simple binary labels based on future price movement

@registry.register_target('future_up')
def future_up(df: pd.DataFrame) -> pd.Series:
    """1 if close[t+1] > close[t], else 0."""
    y = (df['close'].shift(-1) > df['close']).astype(int)
    y.name = 'future_up'
    return y

@registry.register_target('future_up_5')
def future_up_5(df: pd.DataFrame) -> pd.Series:
    """1 if close[t+5] > close[t], else 0."""
    y = (df['close'].shift(-5) > df['close']).astype(int)
    y.name = 'future_up_5'
    return y

@registry.register_target('future_up_20')
def future_up_20(df: pd.DataFrame) -> pd.Series:
    """1 if close[t+20] > close[t], else 0."""
    y = (df['close'].shift(-20) > df['close']).astype(int)
    y.name = 'future_up_20'
    return y

@registry.register_target('future_up_10bp_10')
def future_up_10bp_10(df: pd.DataFrame) -> pd.Series:
    """1 if return over 10 bars > +10bps (~0.1%), else 0."""
    ret10 = df['close'].shift(-10) / df['close'] - 1.0
    y = (ret10 > 0.001).astype(int)
    y.name = 'future_up_10bp_10'
    return y

# Feature 1: Percentage change of closing price
@registry.register_feature('pct_change')
def pct_change_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].pct_change().fillna(0)
    x.name = 'pct_change'
    return x

# Feature 2: 3-period momentum (difference between current close and close 3 periods ago)
@registry.register_feature('momentum_3')
def momentum_3_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'] - df['close'].shift(3)
    x = x.fillna(0)
    x.name = 'momentum_3'
    return x

# Feature 3: 5-period rolling mean of closing price
@registry.register_feature('rolling_mean_5')
def rolling_mean_5_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].rolling(window=5).mean().bfill()
    x.name = 'rolling_mean_5'
    return x

# Feature 4: 5-period rolling standard deviation of closing price
@registry.register_feature('rolling_std_5')
def rolling_std_5_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].rolling(window=5).std().bfill()
    x.name = 'rolling_std_5'
    return x

# Feature 5: Percentage change in volume (or zeros if volume not available)
@registry.register_feature('volume_change')
def volume_change_feature(df: pd.DataFrame) -> pd.Series:
    if 'volume' in df.columns:
        x = df['volume'].pct_change().fillna(0)
    else:
        x = pd.Series(0.0, index=df.index)
    x.name = 'volume_change'
    return x

# Model: simple logistic regression baseline with sklearn if available, otherwise random baseline
@registry.register_model('logistic_baseline')
def logistic_baseline(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """Train a logistic regression model to predict the binary target from features.

    Args:
        y: Target series (binary 0/1).
        features: Dictionary of feature series.
        df: Original dataframe (unused but kept for API consistency).
        selection: Selection dictionary with component names (unused here).

    Returns:
        Tuple of (model, metrics dict). If sklearn is unavailable, returns (None, metrics).
    """
    try:
        _ensure_utf8_locale()
        import time
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score, accuracy_score

        # Prepare feature matrix
        X = pd.concat(features, axis=1)
        # Keep rows where y is present; allow NaNs in X to be imputed later
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        # Use label-based indexing with a boolean mask
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        # Replace infs and drop columns that are entirely NaN to avoid imputer errors
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        # Normalize dtypes to avoid silent downcasting warnings in future pandas
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')

        # If not enough data or only one class overall, skip training
        if y_clean.nunique() < 2 or len(X_clean) < 10:
            return None, {"auc": float('nan'), "accuracy": float('nan')}

        def _time_stratified_split_binary(y_series: pd.Series, min_test: int = 100) -> int:
            n = len(y_series)
            if n <= min_test + 10:
                return int(n * 0.8)
            for frac in (0.8, 0.75, 0.7, 0.65, 0.6):
                split_idx = int(n * frac)
                y_tr, y_te = y_series.iloc[:split_idx], y_series.iloc[split_idx:]
                if len(y_te) < min_test:
                    continue
                if y_tr.nunique() >= 2 and y_te.nunique() >= 2:
                    return split_idx
            return int(n * 0.8)

        split = _time_stratified_split(y_clean, min_train=100, min_test=100)
        X_train = X_clean.iloc[:split]
        X_test = X_clean.iloc[split:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[split:]

        # Impute missing values, scale, and increase iterations to aid convergence
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)),
        ])
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        # Predict
        t1 = time.perf_counter()
        probs = model.predict_proba(X_test)[:, 1]
        pred_time = time.perf_counter() - t1
        preds = (probs >= 0.5).astype(int)
        # Metrics
        try:
            auc = float(roc_auc_score(y_test, probs))
        except Exception:
            auc = None
        acc = float(accuracy_score(y_test, preds)) if len(y_test) else float('nan')
        # Class balance baselines
        pos_rate_train = float(y_train.mean()) if len(y_train) else float('nan')
        pos_rate_test = float(y_test.mean()) if len(y_test) else float('nan')
        return model, {
            "auc": auc,
            "accuracy": acc,
            "pos_rate_train": pos_rate_train,
            "pos_rate_test": pos_rate_test,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "fit_time_sec": float(fit_time),
            "predict_time_sec": float(pred_time),
        }
    except ImportError:
        # sklearn not available: simple random baseline
        y_nonan = y.dropna()
        if len(y_nonan) == 0:
            return None, {"auc": float('nan'), "accuracy": float('nan')}
        preds = np.random.choice([0, 1], size=len(y_nonan))
        acc = float((preds == y_nonan.values).mean())
        return None, {"auc": float('nan'), "accuracy": acc}


# Model: TensorFlow MLP with GPU support (if available)
@registry.register_model('tf_mlp_baseline')
def tf_mlp_baseline(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """Simple MLP in TensorFlow/Keras. Uses train/holdout split like logistic.

    - Imputation: median; Scaling: StandardScaler
    - EarlyStopping + ReduceLROnPlateau
    - Reports auc, ap, accuracy + fit/predict time
    """
    try:
        _ensure_utf8_locale()
        import os, time
        # Quiet TF logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

        # GPU memory growth for safety
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

        # Prepare X/y
        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        if y_clean.nunique() < 2 or len(X_clean) < 100:
            return None, {"auc": float('nan'), "ap": float('nan')}

        split = _time_stratified_split(y_clean, min_train=100, min_test=100)
        X_train_df = X_clean.iloc[:split]
        X_test_df = X_clean.iloc[split:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[split:]

        # Impute + scale
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train_df))
        X_test = scl.transform(imp.transform(X_test_df))
        y_train_np = y_train.to_numpy(dtype=np.float32)
        y_test_np = y_test.to_numpy(dtype=np.float32)

        # Build model (BN + deeper head)
        n_in = X_train.shape[1]
        inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32)
        x = tf.keras.layers.Dense(128, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(name='auc')])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-5),
        ]

        # Optional class weighting for imbalance
        pos_rate_train = float(y_train_np.mean()) if len(y_train_np) else float('nan')
        class_weight = None
        try:
            if 0.0 < pos_rate_train < 1.0:
                w1 = 0.5 / max(1e-6, pos_rate_train)
                w0 = 0.5 / max(1e-6, 1.0 - pos_rate_train)
                # Clip extremes
                w1 = float(min(4.0, max(0.25, w1)))
                w0 = float(min(4.0, max(0.25, w0)))
                class_weight = {0: w0, 1: w1}
        except Exception:
            class_weight = None

        # Train (hold out 10% of train for validation)
        t0 = time.perf_counter()
        model.fit(
            X_train, y_train_np,
            epochs=50,
            batch_size=1024,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )
        fit_time = time.perf_counter() - t0

        # Predict
        t1 = time.perf_counter()
        probs = model.predict(X_test, batch_size=8192, verbose=0).reshape(-1)
        pred_time = time.perf_counter() - t1
        preds = (probs >= 0.5).astype(int)

        # Metrics
        try:
            auc = float(roc_auc_score(y_test_np, probs))
        except Exception:
            auc = None
        try:
            ap = float(average_precision_score(y_test_np, probs))
        except Exception:
            ap = float('nan')
        acc = float(accuracy_score(y_test_np, preds)) if len(y_test_np) else float('nan')
        pos_rate_test = float(y_test_np.mean()) if len(y_test_np) else float('nan')

        metrics = {
            "auc": auc,
            "ap": ap,
            "accuracy": acc,
            "pos_rate_train": pos_rate_train,
            "pos_rate_test": pos_rate_test,
            "n_train": int(len(y_train_np)),
            "n_test": int(len(y_test_np)),
            "fit_time_sec": float(fit_time),
            "predict_time_sec": float(pred_time),
        }
        return model, metrics
    except ImportError:
        return None, {"auc": float('nan'), "ap": float('nan')}


# Model: TensorFlow MLP (BN + AdamW weight decay)
@registry.register_model('tf_mlp_bn_wd')
def tf_mlp_bn_wd(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """MLP with BatchNorm and AdamW weight decay; GPU-capable.

    Similar prepro a tf_mlp_baseline, pero cambia el optimizador a AdamW con weight decay.
    """
    try:
        _ensure_utf8_locale()
        import os, time
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

        try:
            gpus = tf.config.list_physical_devices('GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        if y_clean.nunique() < 2 or len(X_clean) < 100:
            return None, {"auc": float('nan'), "ap": float('nan')}

        split = _time_stratified_split(y_clean, min_train=100, min_test=100)
        X_train_df = X_clean.iloc[:split]
        X_test_df = X_clean.iloc[split:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[split:]

        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train_df))
        X_test = scl.transform(imp.transform(X_test_df))
        y_train_np = y_train.to_numpy(dtype=np.float32)
        y_test_np = y_test.to_numpy(dtype=np.float32)

        n_in = X_train.shape[1]
        inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32)
        # Optional LayerNorm on inputs
        x = tf.keras.layers.LayerNormalization()(inputs)
        x = tf.keras.layers.Dense(128, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=6, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-5),
        ]

        pos_rate_train = float(y_train_np.mean()) if len(y_train_np) else float('nan')
        class_weight = None
        try:
            if 0.0 < pos_rate_train < 1.0:
                w1 = 0.5 / max(1e-6, pos_rate_train)
                w0 = 0.5 / max(1e-6, 1.0 - pos_rate_train)
                w1 = float(min(4.0, max(0.25, w1)))
                w0 = float(min(4.0, max(0.25, w0)))
                class_weight = {0: w0, 1: w1}
        except Exception:
            class_weight = None

        t0 = time.perf_counter()
        model.fit(
            X_train, y_train_np,
            epochs=60,
            batch_size=1024,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = model.predict(X_test, batch_size=8192, verbose=0).reshape(-1)
        pred_time = time.perf_counter() - t1
        preds = (probs >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_test_np, probs))
        except Exception:
            auc = None
        try:
            ap = float(average_precision_score(y_test_np, probs))
        except Exception:
            ap = float('nan')
        acc = float(accuracy_score(y_test_np, preds)) if len(y_test_np) else float('nan')
        pos_rate_test = float(y_test_np.mean()) if len(y_test_np) else float('nan')

        metrics = {
            "auc": auc,
            "ap": ap,
            "accuracy": acc,
            "pos_rate_train": pos_rate_train,
            "pos_rate_test": pos_rate_test,
            "n_train": int(len(y_train_np)),
            "n_test": int(len(y_test_np)),
            "fit_time_sec": float(fit_time),
            "predict_time_sec": float(pred_time),
        }
        return model, metrics
    except ImportError:
        return None, {"auc": float('nan'), "ap": float('nan')}


# Model: TensorFlow MLP multiclass (handles binary and 3-class)
@registry.register_model('tf_mlp_multiclass')
def tf_mlp_multiclass(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    try:
        _ensure_utf8_locale()
        import os, time
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        from sklearn.preprocessing import label_binarize

        # Prepare X/y
        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        if y_clean.nunique() < 2 or len(X_clean) < 100:
            return None, {"auc": float('nan'), "ap": float('nan')}

        # Map classes
        y_vals = y_clean.dropna().astype(float)
        classes = sorted(pd.unique(y_vals))
        # Map {-1,0,1} -> {0,1,2} if needed
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = y_vals.map(class_to_idx).astype(int)
        n_classes = len(classes)

        split = _time_stratified_split(y_clean, min_train=100, min_test=100)
        X_train_df = X_clean.iloc[:split]
        X_test_df = X_clean.iloc[split:]
        y_train_idx = y_idx.iloc[:split]
        y_test_idx = y_idx.iloc[split:]

        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train_df))
        X_test = scl.transform(imp.transform(X_test_df))

        # Build model
        n_in = X_train.shape[1]
        inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32)
        x = tf.keras.layers.Dense(128, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        if n_classes > 2:
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
        else:
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(1e-3)
        model.compile(optimizer=opt, loss=loss)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        ]

        t0 = time.perf_counter()
        model.fit(
            X_train, y_train_idx.to_numpy(),
            epochs=50,
            batch_size=1024,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        if n_classes > 2:
            probs = model.predict(X_test, batch_size=8192, verbose=0)
            pred_time = time.perf_counter() - t1
            # Metrics (macro AUC/AP)
            y_true = y_test_idx.to_numpy()
            try:
                auc = float(roc_auc_score(y_true, probs, multi_class='ovr', average='macro'))
            except Exception:
                auc = None
            try:
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                ap = float(average_precision_score(y_bin, probs, average='macro'))
            except Exception:
                ap = float('nan')
            acc = float(accuracy_score(y_true, probs.argmax(axis=1))) if len(y_true) else float('nan')
        else:
            probs1 = model.predict(X_test, batch_size=8192, verbose=0).reshape(-1)
            pred_time = time.perf_counter() - t1
            y_true = y_test_idx.to_numpy()
            try:
                auc = float(roc_auc_score(y_true, probs1))
            except Exception:
                auc = None
            try:
                ap = float(average_precision_score(y_true, probs1))
            except Exception:
                ap = float('nan')
            acc = float(accuracy_score(y_true, (probs1 >= 0.5).astype(int))) if len(y_true) else float('nan')

        pos_rate_train = float((y_train_idx == (class_to_idx.get(1, 1))).mean()) if n_classes == 2 else float('nan')
        pos_rate_test = float((y_test_idx == (class_to_idx.get(1, 1))).mean()) if n_classes == 2 else float('nan')

        metrics = {
            "auc": auc,
            "ap": ap,
            "accuracy": acc,
            "pos_rate_train": pos_rate_train,
            "pos_rate_test": pos_rate_test,
            "n_train": int(len(y_train_idx)),
            "n_test": int(len(y_test_idx)),
            "fit_time_sec": float(fit_time),
            "predict_time_sec": float(pred_time),
        }
        return model, metrics
    except ImportError:
        return None, {"auc": float('nan'), "ap": float('nan')}


# Model: RandomForest with TimeSeriesSplit CV and PR-AUC reporting
@registry.register_model('rf_baseline')
def rf_baseline(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """RandomForest baseline evaluated with TimeSeriesSplit.

    Returns overall metrics with AUC/AP medians across folds. The 'auc' key is set
    to the median AUC for compatibility with the engine's scoring.
    """
    try:
        _ensure_utf8_locale()
        import time
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Prepare feature matrix
        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')

        if y_clean.nunique() < 2 or len(X_clean) < 100:
            return None, {"auc": float('nan'), "ap": float('nan')}

        # TimeSeriesSplit CV
        tss = TimeSeriesSplit(n_splits=5)
        aucs, aps = [], []
        pos_rates_train, pos_rates_test = [], []

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=100,
                n_jobs=-1,
                random_state=0,
                class_weight=None,
            )),
        ])

        for train_idx, test_idx in tss.split(X_clean):
            X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 1:
                continue
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_t = time.perf_counter() - t0
            try:
                t1 = time.perf_counter()
                proba = pipe.predict_proba(X_te)[:, 1]
                pred_t = time.perf_counter() - t1
            except Exception:
                # should not happen for RF, but just in case
                t1 = time.perf_counter()
                proba = pipe.named_steps["clf"].predict_proba(X_te)[:, 1]
                pred_t = time.perf_counter() - t1
            try:
                aucs.append(float(roc_auc_score(y_te, proba)))
            except Exception:
                pass
            try:
                aps.append(float(average_precision_score(y_te, proba)))
            except Exception:
                pass
            pos_rates_train.append(float(y_tr.mean()))
            pos_rates_test.append(float(y_te.mean()))
            # Accumulate times
            try:
                total_fit_time += fit_t
                total_pred_time += pred_t
            except NameError:
                total_fit_time = fit_t
                total_pred_time = pred_t

        if len(aucs) == 0:
            return None, {"auc": float('nan'), "ap": float('nan')}

        auc_series = pd.Series(aucs)
        ap_series = pd.Series(aps) if len(aps) > 0 else pd.Series(dtype=float)
        pr_train_med = float(np.median(pos_rates_train)) if pos_rates_train else float('nan')
        pr_test_med = float(np.median(pos_rates_test)) if pos_rates_test else float('nan')

        metrics = {
            "auc": float(auc_series.median()),  # used by engine scoring
            "auc_median": float(auc_series.median()),
            "auc_mean": float(auc_series.mean()),
            "auc_q025": float(auc_series.quantile(0.025)),
            "auc_q975": float(auc_series.quantile(0.975)),
            "ap": float(ap_series.median()) if len(ap_series) else float('nan'),
            "ap_mean": float(ap_series.mean()) if len(ap_series) else float('nan'),
            "folds": int(len(auc_series)),
            "pos_rate_train": pr_train_med,
            "pos_rate_test": pr_test_med,
            "fit_time_sec": float(total_fit_time) if 'total_fit_time' in locals() else float('nan'),
            "predict_time_sec": float(total_pred_time) if 'total_pred_time' in locals() else float('nan'),
        }
        return pipe, metrics
    except ImportError:
        return None, {"auc": float('nan'), "ap": float('nan')}
# Model: Gradient Boosting (Histogram) with TimeSeriesSplit CV and PR-AUC
@registry.register_model('hgb_baseline')
def hgb_baseline(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """Histogram Gradient Boosting baseline evaluated with TimeSeriesSplit.

    Similar reporting to rf_baseline; returns median AUC/AP across folds.
    """
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    except Exception:
        pass
    try:
        _ensure_utf8_locale()
        import time
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, average_precision_score

        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')

        if y_clean.nunique() < 2 or len(X_clean) < 100:
            return None, {"auc": float('nan'), "ap": float('nan')}

        tss = TimeSeriesSplit(n_splits=5)
        aucs, aps = [], []
        pos_rates_train, pos_rates_test = [], []

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=None,
                learning_rate=0.05,
                max_iter=600,
                min_samples_leaf=100,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            )),
        ])

        for train_idx, test_idx in tss.split(X_clean):
            X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 1:
                continue
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_t = time.perf_counter() - t0
            # HistGB uses predict_proba in recent sklearn
            t1 = time.perf_counter()
            proba = pipe.predict_proba(X_te)[:, 1]
            pred_t = time.perf_counter() - t1
            try:
                aucs.append(float(roc_auc_score(y_te, proba)))
            except Exception:
                pass
            try:
                aps.append(float(average_precision_score(y_te, proba)))
            except Exception:
                pass
            pos_rates_train.append(float(y_tr.mean()))
            pos_rates_test.append(float(y_te.mean()))
            try:
                total_fit_time += fit_t
                total_pred_time += pred_t
            except NameError:
                total_fit_time = fit_t
                total_pred_time = pred_t

        if len(aucs) == 0:
            return None, {"auc": float('nan'), "ap": float('nan')}

        auc_series = pd.Series(aucs)
        ap_series = pd.Series(aps) if len(aps) > 0 else pd.Series(dtype=float)
        pr_train_med = float(np.median(pos_rates_train)) if pos_rates_train else float('nan')
        pr_test_med = float(np.median(pos_rates_test)) if pos_rates_test else float('nan')

        metrics = {
            "auc": float(auc_series.median()),
            "auc_median": float(auc_series.median()),
            "auc_mean": float(auc_series.mean()),
            "auc_q025": float(auc_series.quantile(0.025)),
            "auc_q975": float(auc_series.quantile(0.975)),
            "ap": float(ap_series.median()) if len(ap_series) else float('nan'),
            "ap_mean": float(ap_series.mean()) if len(ap_series) else float('nan'),
            "folds": int(len(auc_series)),
            "pos_rate_train": pr_train_med,
            "pos_rate_test": pr_test_med,
            "fit_time_sec": float(total_fit_time) if 'total_fit_time' in locals() else float('nan'),
            "predict_time_sec": float(total_pred_time) if 'total_pred_time' in locals() else float('nan'),
        }
        return pipe, metrics
    except ImportError:
        return None, {"auc": float('nan'), "ap": float('nan')}
# --- Margin-based targets (fixed horizon, volatility-scaled threshold) ---
def _make_up_margin_binary(
    df: pd.DataFrame,
    price_col: str = "close",
    H: int = 12,
    k: float = 0.75,
    vol_method: str = "ewm",
    ewm_halflife: int = 60,
    rolling_window: int = 120,
    scale_by_sqrt_H: bool = True,
    log_returns: bool = True,
) -> pd.Series:
    px = df[price_col].astype(float)
    if log_returns:
        r1 = np.log(px).diff()
    else:
        r1 = px.pct_change()
    if vol_method == "ewm":
        sigma1 = r1.ewm(halflife=ewm_halflife, adjust=False).std().fillna(0.0)
    elif vol_method == "rolling":
        sigma1 = r1.rolling(rolling_window, min_periods=max(5, rolling_window // 5)).std().fillna(0.0)
    else:
        sigma1 = r1.ewm(halflife=ewm_halflife, adjust=False).std().fillna(0.0)
    if log_returns:
        r_fut = np.log(px.shift(-H)) - np.log(px)
    else:
        r_fut = px.shift(-H) / px - 1.0
    tau = k * sigma1
    if scale_by_sqrt_H:
        tau = tau * np.sqrt(H)
    y = (r_fut > tau).astype(float)
    # últimas H filas no tienen futuro
    y[r_fut.isna()] = np.nan
    return y.astype("float32")


@registry.register_target('up_margin_H12_k075')
def up_margin_H12_k075(df: pd.DataFrame) -> pd.Series:
    y = _make_up_margin_binary(df, H=12, k=0.75, vol_method="ewm", ewm_halflife=60,
                               rolling_window=120, scale_by_sqrt_H=True, log_returns=True)
    y.name = 'up_margin_H12_k075'
    return y

def _make_updown_margin_3class(
    df: pd.DataFrame,
    price_col: str = "close",
    H: int = 12,
    k: float = 0.75,
    vol_method: str = "ewm",
    ewm_halflife: int = 60,
    rolling_window: int = 120,
    scale_by_sqrt_H: bool = True,
    log_returns: bool = True,
) -> pd.Series:
    px = df[price_col].astype(float)
    if log_returns:
        r1 = np.log(px).diff()
    else:
        r1 = px.pct_change()
    if vol_method == "ewm":
        sigma1 = r1.ewm(halflife=ewm_halflife, adjust=False).std().fillna(0.0)
    elif vol_method == "rolling":
        sigma1 = r1.rolling(rolling_window, min_periods=max(5, rolling_window // 5)).std().fillna(0.0)
    else:
        sigma1 = r1.ewm(halflife=ewm_halflife, adjust=False).std().fillna(0.0)
    if log_returns:
        r_fut = np.log(px.shift(-H)) - np.log(px)
    else:
        r_fut = px.shift(-H) / px - 1.0
    tau = k * sigma1
    if scale_by_sqrt_H:
        tau = tau * np.sqrt(H)
    y = pd.Series(0, index=df.index, dtype="int8")
    y = y.mask(r_fut > tau, 1)
    y = y.mask(r_fut < -tau, -1)
    y[r_fut.isna()] = np.nan
    return y


@registry.register_target('updown_margin_H12_k075')
def updown_margin_H12_k075(df: pd.DataFrame) -> pd.Series:
    y = _make_updown_margin_3class(df, H=12, k=0.75, vol_method="ewm", ewm_halflife=60,
                                   rolling_window=120, scale_by_sqrt_H=True, log_returns=True)
    y.name = 'updown_margin_H12_k075'
    return y

# --- ATR-relative and regime-filtered targets ---
def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    h = pd.to_numeric(h, errors='coerce')
    l = pd.to_numeric(l, errors='coerce')
    c = pd.to_numeric(c, errors='coerce')
    pc = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr_roll(df: pd.DataFrame, length: int = 14, as_relative: bool = True) -> pd.Series:
    """Rolling ATR (mean true range). If as_relative, return ATR/close to match return units."""
    tr = _true_range(df.get('high'), df.get('low'), df.get('close'))
    atr = tr.rolling(window=int(length), min_periods=max(2, int(length)//2)).mean()
    if as_relative:
        px = pd.to_numeric(df.get('close'), errors='coerce')
        atr = atr / px.replace(0.0, np.nan)
    return atr

def _future_return(df: pd.DataFrame, H: int = 12, log_returns: bool = True) -> pd.Series:
    px = pd.to_numeric(df['close'], errors='coerce')
    if log_returns:
        return (np.log(px.shift(-H)) - np.log(px))
    else:
        return (px.shift(-H) / px - 1.0)

def _binary_from_band(r_fut: pd.Series, tau: pd.Series, neutral_frac: float = 0.25) -> pd.Series:
    tau = pd.to_numeric(tau, errors='coerce')
    lower = -tau
    nb = float(neutral_frac) * tau.abs()
    y = pd.Series(np.nan, index=r_fut.index, dtype='float32')
    y = y.mask(r_fut > (tau), 1.0)
    y = y.mask(r_fut < (lower), 0.0)
    # values with |r_fut| <= nb remain NaN (ignored)
    y[r_fut.isna()] = np.nan
    return y

def _adx_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta  # type: ignore
        out = ta.adx(df.get('high'), df.get('low'), df.get('close'), length=int(length))
        s = out.get(f'ADX_{int(length)}') if hasattr(out, 'get') else None
        if s is None and hasattr(out, '__getitem__'):
            s = out[f'ADX_{int(length)}']
        return pd.to_numeric(s, errors='coerce') if s is not None else pd.Series(np.nan, index=df.index)
    except Exception:
        # Fallback: proxy using normalized TR
        tr = _true_range(df.get('high'), df.get('low'), df.get('close'))
        vol = tr.rolling(window=int(length), min_periods=max(2, int(length)//2)).mean()
        s = 100.0 * (vol / (pd.to_numeric(df.get('close'), errors='coerce').replace(0, np.nan))).fillna(0.0)
        return s.clip(0, 100)

def _fmt_k_tag(k: float) -> str:
    try:
        if float(k).is_integer():
            return str(int(k))
    except Exception:
        pass
    s = f"{k}"
    return s.replace('.', '')

def _register_updown_atr_target(H: int, k: float, atr_len: int = 14, neutral_frac: float = 0.25) -> None:
    K_tag = _fmt_k_tag(k)
    nb_tag = f"{neutral_frac:.2f}".replace('.', '')
    name = f"updown_atr_H{int(H)}_k{K_tag}_nb{nb_tag}"
    if name in registry.targets:
        return

    @registry.register_target(name)
    def _t(df: pd.DataFrame, H=H, k=k, atr_len=atr_len, neutral_frac=neutral_frac) -> pd.Series:
        r_fut = _future_return(df, H=H, log_returns=True)
        # Use relative ATR to compare with returns, and scale by sqrt(H)
        atr_rel = _atr_roll(df, length=atr_len, as_relative=True).replace(0.0, np.nan)
        tau = float(k) * atr_rel * np.sqrt(max(1, int(H)))
        y = _binary_from_band(r_fut, tau, neutral_frac=neutral_frac)
        y.name = name
        return y

def _register_trend_mr_atr_target(H: int, k: float, atr_len: int, adx_len: int, adx_thr: float, mode: str = 'trend', neutral_frac: float = 0.25) -> None:
    K_tag = _fmt_k_tag(k)
    T = int(adx_thr)
    tag = 'trend' if mode == 'trend' else 'mr'
    name = f"{tag}_updown_atr_H{int(H)}_k{K_tag}_adx{T}"
    if name in registry.targets:
        return

    @registry.register_target(name)
    def _t(df: pd.DataFrame, H=H, k=k, atr_len=atr_len, adx_len=adx_len, adx_thr=adx_thr, mode=mode, neutral_frac=neutral_frac) -> pd.Series:
        r_fut = _future_return(df, H=H, log_returns=True)
        atr_rel = _atr_roll(df, length=atr_len, as_relative=True).replace(0.0, np.nan)
        tau = float(k) * atr_rel * np.sqrt(max(1, int(H)))
        y = _binary_from_band(r_fut, tau, neutral_frac=neutral_frac)
        adx = _adx_series(df, length=adx_len)
        if mode == 'trend':
            mask = adx >= float(adx_thr)
        else:
            mask = adx <= float(adx_thr)
        y = y.where(mask, np.nan)
        y.name = name
        return y

# Register a small set of ATR targets by default
try:
    for H in (6, 12, 36):
        for k in (0.5, 1.0):
            _register_updown_atr_target(H=H, k=k, atr_len=14, neutral_frac=0.25)
    _register_trend_mr_atr_target(H=12, k=1.0, atr_len=14, adx_len=14, adx_thr=25.0, mode='trend', neutral_frac=0.25)
    _register_trend_mr_atr_target(H=12, k=0.5, atr_len=14, adx_len=14, adx_thr=20.0, mode='mr', neutral_frac=0.25)
except Exception:
    pass


# ---------------------------
# Regression models (unified score via `skill`)
# ---------------------------

@registry.register_model('hgb_regressor')
def hgb_regressor(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """Histogram Gradient Boosting Regressor with time-aware split.

    Metrics: rmse, mae, r2, spearman, baseline_rmse (mean predictor), skill = 1 - rmse/rmse_baseline.
    """
    try:
        import time
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = pd.to_numeric(combined.loc[mask, combined.columns[0]], errors='coerce')
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')

        if len(X_clean) < 200:
            return None, {"rmse": float('nan'), "r2": float('nan')}

        split = _time_stratified_split(y_clean, min_train=200, min_test=200)
        X_tr, X_te = X_clean.iloc[:split], X_clean.iloc[split:]
        y_tr, y_te = y_clean.iloc[:split], y_clean.iloc[split:]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=600,
                min_samples_leaf=100,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            )),
        ])

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = pipe.predict(X_te)
        pred_t = time.perf_counter() - t1

        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred)) if len(y_te) else float('nan')
        try:
            rho, _ = spearmanr(y_te, pred)
            spearman = float(rho)
        except Exception:
            spearman = float('nan')
        # Baseline: predict train mean
        baseline = float(np.mean(y_tr)) if len(y_tr) else 0.0
        rmse_base = float(np.sqrt(mean_squared_error(y_te, np.full_like(y_te, baseline)))) if len(y_te) else float('nan')
        skill = float(1.0 - (rmse / rmse_base)) if (rmse_base and np.isfinite(rmse_base) and rmse_base > 0) else float('nan')

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": spearman,
            "baseline_rmse": rmse_base,
            "skill": skill,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "fit_time_sec": float(fit_t),
            "predict_time_sec": float(pred_t),
        }
        return pipe, metrics
    except Exception:
        return None, {"rmse": float('nan'), "r2": float('nan')}


@registry.register_model('tf_mlp_regressor')
def tf_mlp_regressor(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """TF MLP for regression with MSE loss and early stopping."""
    try:
        import os, time
        _ensure_utf8_locale()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:, 0].notna()
        y_clean = pd.to_numeric(combined.loc[mask, combined.columns[0]], errors='coerce')
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        if len(X_clean) < 200:
            return None, {"rmse": float('nan'), "r2": float('nan')}

        split = _time_stratified_split(y_clean, min_train=200, min_test=200)
        X_train_df = X_clean.iloc[:split]
        X_test_df = X_clean.iloc[split:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[split:]

        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train_df))
        X_test = scl.transform(imp.transform(X_test_df))
        y_train_np = y_train.to_numpy(dtype=np.float32)
        y_test_np = y_test.to_numpy(dtype=np.float32)

        n_in = X_train.shape[1]
        inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32)
        x = tf.keras.layers.Dense(128, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        ]

        t0 = time.perf_counter()
        model.fit(
            X_train, y_train_np,
            epochs=50,
            batch_size=1024,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = model.predict(X_test, batch_size=8192, verbose=0).reshape(-1)
        pred_time = time.perf_counter() - t1

        rmse = float(np.sqrt(((pred - y_test_np) ** 2).mean()))
        mae = float(np.abs(pred - y_test_np).mean())
        try:
            r2 = float(r2_score(y_test_np, pred))
        except Exception:
            r2 = float('nan')
        try:
            rho, _ = spearmanr(y_test_np, pred)
            spearman = float(rho)
        except Exception:
            spearman = float('nan')
        baseline = float(np.mean(y_train_np)) if len(y_train_np) else 0.0
        rmse_base = float(np.sqrt(((y_test_np - baseline) ** 2).mean())) if len(y_test_np) else float('nan')
        skill = float(1.0 - (rmse / rmse_base)) if (rmse_base and np.isfinite(rmse_base) and rmse_base > 0) else float('nan')

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": spearman,
            "baseline_rmse": rmse_base,
            "skill": skill,
            "n_train": int(len(y_train_np)),
            "n_test": int(len(y_test_np)),
            "fit_time_sec": float(fit_time),
            "predict_time_sec": float(pred_time),
        }
        return model, metrics
    except Exception:
        return None, {"rmse": float('nan'), "r2": float('nan')}

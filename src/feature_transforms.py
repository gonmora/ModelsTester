from __future__ import annotations

from typing import List, Optional
import os
import pandas as pd
import numpy as np

from .registry import registry


BASE_FEATURE_IDS: List[str] = [
    "ta_volume_pvi",
    "ta_volume_nvi",
    "ta_volume_pvr",
    "dist_to_max_1w_pct",
    "dist_to_min_1w_pct",
    "dist_to_max_1d_pct",
    "rolling_std_5",
    "ta_statistics_zscore",
    "ta_statistics_tos_stdevall",
    "vol_sum_1d_cal",
    "vol_sum_1d_roll",
    "ta_trend_decay",
    "ta_overlap_ema",
    "ta_overlap_vidya",
    "ta_momentum_dm",
    "ta_trend_aroon",
]


def _to_num(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _diff1(s: pd.Series) -> pd.Series:
    return _to_num(s).diff().fillna(0.0)


def _pct1(s: pd.Series) -> pd.Series:
    s = _to_num(s)
    out = s.pct_change()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _vel_ema(s: pd.Series, halflife: int = 30) -> pd.Series:
    d = _to_num(s).diff()
    return d.ewm(halflife=halflife, adjust=False).mean().fillna(0.0)


def _register_transform(base_id: str) -> None:
    if base_id not in registry.features:
        return

    def _mk(base: str, suffix: str, func) -> None:
        name = f"{base}_{suffix}"
        if name in registry.features:
            return

        @registry.register_feature(name)
        def _feat(df: pd.DataFrame) -> pd.Series:
            x = registry.features[base](df)
            try:
                s = pd.Series(x) if not isinstance(x, pd.Series) else x
            except Exception:
                s = pd.Series(x)
            out = func(s)
            out.name = name
            return out

    _mk(base_id, "diff1", _diff1)
    _mk(base_id, "pct1", _pct1)
    _mk(base_id, "vel_ema", _vel_ema)


def register_all_transforms(ids: List[str] = BASE_FEATURE_IDS) -> int:
    c = 0
    for fid in ids:
        pre = len(registry.features)
        _register_transform(fid)
        post = len(registry.features)
        if post > pre:
            c += (post - pre)
    return c

def register_transforms_for_top_features(
    top_k: int = 50,
    include_unseen: bool = False,
    weights_path: Optional[str] = None,
    exclude_candles: bool = True,
    suffixes_to_skip: Optional[List[str]] = None,
) -> int:
    """Register diff1/pct1/vel_ema transforms for Top‑K base features.

    - Pulls Top‑K from the current weights store (lazy import of reporting).
    - Skips features that already look like transformed variants (e.g., *_diff1).
    - Optionally skips candle pattern features (binary/boolean style).

    Returns the number of feature functions newly registered.
    """
    # Lazy import to avoid circular import at module load time
    try:
        from .reporting import top_features as _top_features
        from .reporting import WeightStore as _WeightStore
    except Exception:
        return 0

    ws = _WeightStore(weights_path) if weights_path else _WeightStore()
    try:
        df = _top_features(top_k, include_unseen=include_unseen, ws=ws)
    except Exception:
        return 0
    if df is None or "feature" not in df.columns or len(df) == 0:
        return 0

    names: List[str] = [str(x) for x in df["feature"].dropna().tolist()]
    # Defaults: skip our own transform suffixes
    suffixes = suffixes_to_skip or ["_diff1", "_pct1", "_vel_ema"]

    def _is_candidate(name: str) -> bool:
        if exclude_candles and name.startswith("ta_candles_"):
            return False
        for suf in suffixes:
            if name.endswith(suf):
                return False
        # Only register for features we actually have in the registry
        return name in registry.features

    candidates = [n for n in names if _is_candidate(n)]
    added = 0
    for fid in candidates:
        pre = len(registry.features)
        _register_transform(fid)
        post = len(registry.features)
        if post > pre:
            added += (post - pre)
    return added

def _ema(s: pd.Series, n: int) -> pd.Series:
    return _to_num(s).ewm(span=max(2, int(n)), adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return _to_num(s).rolling(window=max(2, int(n)), min_periods=max(2, int(n))).mean()

def _rstd(s: pd.Series, n: int) -> pd.Series:
    return _to_num(s).rolling(window=max(2, int(n)), min_periods=max(2, int(n))).std(ddof=0)

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h = _to_num(df.get("high", pd.Series(dtype=float)))
    l = _to_num(df.get("low", pd.Series(dtype=float)))
    c = _to_num(df.get("close", pd.Series(dtype=float)))
    pc = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=max(2, int(n)), adjust=False).mean()

def register_close_distance_features(
    windows: List[int] = [20, 50, 288],
    bb_k: float = 2.0,
    kc_mult: float = 2.0,
    atr_mult: float = 2.0,
) -> int:
    """Register distance/position features relative to close for several channels.

    For each window n in `windows` registers:
      - close_pos_bb_w{n}_k{K}
      - close_z_bb_w{n}_k{K}
      - close_pos_kc_w{n}_m{M}
      - close_pos_dc_w{n}
      - close_dist_sma_std_w{n}
      - close_dist_ema_atr_w{n}
      - close_pos_atrband_w{n}_m{M}
    Returns number of features registered.
    """
    added = 0
    for n in windows:
        N = int(n)
        K_disp = int(bb_k) if float(bb_k).is_integer() else bb_k
        M_disp = int(kc_mult) if float(kc_mult).is_integer() else kc_mult

        # Bollinger position
        name_pos_bb = f"close_pos_bb_w{N}_k{K_disp}"
        if name_pos_bb not in registry.features:
            @registry.register_feature(name_pos_bb)
            def _pos_bb(df: pd.DataFrame, N=N, K=bb_k) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                sma = _sma(c, N)
                std = _rstd(c, N)
                upper = sma + float(K) * std
                lower = sma - float(K) * std
                denom = (upper - lower).replace(0.0, np.nan)
                out = (c - sma) / denom
                return out.clip(-5, 5).fillna(0.0).rename(name_pos_bb)
            added += 1

        # Bollinger z
        name_z_bb = f"close_z_bb_w{N}_k{K_disp}"
        if name_z_bb not in registry.features:
            @registry.register_feature(name_z_bb)
            def _z_bb(df: pd.DataFrame, N=N, K=bb_k) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                sma = _sma(c, N)
                std = _rstd(c, N).replace(0.0, np.nan)
                out = (c - sma) / (float(K) * std)
                return out.clip(-10, 10).fillna(0.0).rename(name_z_bb)
            added += 1

        # Keltner position
        name_pos_kc = f"close_pos_kc_w{N}_m{M_disp}"
        if name_pos_kc not in registry.features:
            @registry.register_feature(name_pos_kc)
            def _pos_kc(df: pd.DataFrame, N=N, M=kc_mult) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                ema = _ema(c, N)
                atr = _atr(df, N)
                denom = (2.0 * float(M) * atr).replace(0.0, np.nan)
                out = (c - ema) / denom
                return out.clip(-5, 5).fillna(0.0).rename(name_pos_kc)
            added += 1

        # Donchian position
        name_pos_dc = f"close_pos_dc_w{N}"
        if name_pos_dc not in registry.features:
            @registry.register_feature(name_pos_dc)
            def _pos_dc(df: pd.DataFrame, N=N) -> pd.Series:
                h = _to_num(df.get("high", pd.Series(dtype=float)))
                l = _to_num(df.get("low", pd.Series(dtype=float)))
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                upper = h.rolling(window=N, min_periods=N).max()
                lower = l.rolling(window=N, min_periods=N).min()
                mid = (upper + lower) / 2.0
                denom = (upper - lower).replace(0.0, np.nan)
                out = (c - mid) / denom
                return out.clip(-5, 5).fillna(0.0).rename(name_pos_dc)
            added += 1

        # Distance to SMA normalized by std
        name_dsma = f"close_dist_sma_std_w{N}"
        if name_dsma not in registry.features:
            @registry.register_feature(name_dsma)
            def _dist_sma_std(df: pd.DataFrame, N=N) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                sma = _sma(c, N)
                std = _rstd(c, N).replace(0.0, np.nan)
                out = (c - sma) / std
                return out.clip(-10, 10).fillna(0.0).rename(name_dsma)
            added += 1

        # Distance to EMA normalized by ATR
        name_dema = f"close_dist_ema_atr_w{N}"
        if name_dema not in registry.features:
            @registry.register_feature(name_dema)
            def _dist_ema_atr(df: pd.DataFrame, N=N) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                ema = _ema(c, N)
                atr = _atr(df, N).replace(0.0, np.nan)
                out = (c - ema) / atr
                return out.clip(-10, 10).fillna(0.0).rename(name_dema)
            added += 1

        # ATR bands position around SMA
        name_patr = f"close_pos_atrband_w{N}_m{M_disp}"
        if name_patr not in registry.features:
            @registry.register_feature(name_patr)
            def _pos_atrband(df: pd.DataFrame, N=N, M=atr_mult) -> pd.Series:
                c = _to_num(df.get("close", pd.Series(dtype=float)))
                sma = _sma(c, N)
                atr = _atr(df, N)
                denom = (2.0 * float(M) * atr).replace(0.0, np.nan)
                out = (c - sma) / denom
                return out.clip(-5, 5).fillna(0.0).rename(name_patr)
            added += 1

    return added


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def register_rsi_rps_features(
    rsi_len: int = 14,
    adx_len: int = 14,
    adx_low: float = 20.0,
    adx_high: float = 25.0,
    norm_win: int = 576,
    slope_k: int = 3,
    gain: float = 0.75,
    add_signals: bool = True,
) -> int:
    """Register RSI RPS derived features based on close/high/low.

    Adds features (with parameterized names):
      - rsi_z_l{rsi}_w{win}
      - rsi_slope_z_l{rsi}_w{win}_k{slope}
      - rsi_rps_regime_a{adx}
      - rsi_rps_l{rsi}_a{adx}_w{win}_k{slope}
      - (optional) rsi_rps_long_..., rsi_rps_short_...
    Returns number of new features registered.
    """
    try:
        import pandas_ta as ta  # type: ignore
    except Exception:
        return 0

    added = 0
    tag = f"l{int(rsi_len)}_a{int(adx_len)}_w{int(norm_win)}_k{int(slope_k)}"
    name_rsi_z = f"rsi_z_l{int(rsi_len)}_w{int(norm_win)}"
    name_rsi_slope_z = f"rsi_slope_z_l{int(rsi_len)}_w{int(norm_win)}_k{int(slope_k)}"
    name_regime = f"rsi_rps_regime_a{int(adx_len)}"
    name_rps = f"rsi_rps_{tag}"
    name_long = f"rsi_rps_long_{tag}"
    name_short = f"rsi_rps_short_{tag}"

    if name_rsi_z not in registry.features:
        @registry.register_feature(name_rsi_z)
        def _feat_rsi_z(df: pd.DataFrame, rsi_len=rsi_len, norm_win=norm_win) -> pd.Series:
            close = _to_num(df.get("close", pd.Series(dtype=float)))
            rsi = ta.rsi(close, length=int(rsi_len))
            minp = max(int(rsi_len), int(norm_win) // 4)
            mu = rsi.rolling(int(norm_win), min_periods=minp).mean()
            sd = rsi.rolling(int(norm_win), min_periods=minp).std(ddof=0)
            out = _safe_div(rsi - mu, sd).fillna(0.0)
            out.name = name_rsi_z
            return out
        added += 1

    if name_rsi_slope_z not in registry.features:
        @registry.register_feature(name_rsi_slope_z)
        def _feat_rsi_slope_z(df: pd.DataFrame, rsi_len=rsi_len, norm_win=norm_win, slope_k=slope_k) -> pd.Series:
            close = _to_num(df.get("close", pd.Series(dtype=float)))
            rsi = ta.rsi(close, length=int(rsi_len))
            slope = rsi.diff(int(slope_k))
            minp = max(int(rsi_len), int(norm_win) // 4)
            slope_sd = slope.rolling(int(norm_win), min_periods=minp).std(ddof=0)
            out = _safe_div(slope, slope_sd).fillna(0.0)
            out.name = name_rsi_slope_z
            return out
        added += 1

    if name_regime not in registry.features:
        @registry.register_feature(name_regime)
        def _feat_regime(df: pd.DataFrame, adx_len=adx_len, adx_low=adx_low, adx_high=adx_high) -> pd.Series:
            h = _to_num(df.get("high", pd.Series(dtype=float)))
            l = _to_num(df.get("low", pd.Series(dtype=float)))
            c = _to_num(df.get("close", pd.Series(dtype=float)))
            adx_df = ta.adx(h, l, c, length=int(adx_len))
            adx_s = _to_num(adx_df.get(f"ADX_{int(adx_len)}", pd.Series(dtype=float)))
            denom = max(1e-9, float(adx_high) - float(adx_low))
            w_trend = ((adx_s - float(adx_low)) / denom).clip(0.0, 1.0)
            out = w_trend.fillna(0.0)
            out.name = name_regime
            return out
        added += 1

    if name_rps not in registry.features:
        @registry.register_feature(name_rps)
        def _feat_rps(
            df: pd.DataFrame,
            rsi_len=rsi_len,
            adx_len=adx_len,
            adx_low=adx_low,
            adx_high=adx_high,
            norm_win=norm_win,
            slope_k=slope_k,
            gain=gain,
        ) -> pd.Series:
            h = _to_num(df.get("high", pd.Series(dtype=float)))
            l = _to_num(df.get("low", pd.Series(dtype=float)))
            c = _to_num(df.get("close", pd.Series(dtype=float)))
            rsi = ta.rsi(c, length=int(rsi_len))
            adx_df = ta.adx(h, l, c, length=int(adx_len))
            adx_s = _to_num(adx_df.get(f"ADX_{int(adx_len)}", pd.Series(dtype=float)))

            # Regime weights
            denom = max(1e-9, float(adx_high) - float(adx_low))
            w_trend = ((adx_s - float(adx_low)) / denom).clip(0.0, 1.0)
            w_range = 1.0 - w_trend

            # Adaptive range normalization
            minp_q = max(int(rsi_len), int(norm_win) // 2)
            q_low = rsi.rolling(int(norm_win), min_periods=minp_q).quantile(0.20, interpolation="linear")
            q_high = rsi.rolling(int(norm_win), min_periods=minp_q).quantile(0.80, interpolation="linear")
            mid = (q_low + q_high) / 2.0
            half_range = (q_high - q_low) / 2.0
            norm_pos = _safe_div(rsi - mid, half_range)

            # Trend/range components
            minp = max(int(rsi_len), int(norm_win) // 4)
            slope = rsi.diff(int(slope_k))
            slope_sd = slope.rolling(int(norm_win), min_periods=minp).std(ddof=0)
            slope_z = _safe_div(slope, slope_sd).fillna(0.0)
            trend_comp = norm_pos + 0.5 * slope_z
            range_comp = -norm_pos
            score_raw = w_trend * trend_comp + w_range * range_comp

            out = np.tanh(float(gain) * score_raw).fillna(0.0)
            out.name = name_rps
            return out
        added += 1

    if add_signals:
        if name_long not in registry.features:
            @registry.register_feature(name_long)
            def _feat_long(df: pd.DataFrame) -> pd.Series:
                s = registry.features[name_rps](df)
                up = (s > 0.5) & (s.shift(1) <= 0.5)
                out = up.astype(int)
                out.name = name_long
                return out
            added += 1
        if name_short not in registry.features:
            @registry.register_feature(name_short)
            def _feat_short(df: pd.DataFrame) -> pd.Series:
                s = registry.features[name_rps](df)
                dn = (s < -0.5) & (s.shift(1) >= -0.5)
                out = dn.astype(int)
                out.name = name_short
                return out
            added += 1

    return added


try:
    register_all_transforms()
except Exception:
    pass

# Optional: auto-register transforms for Top-K features on import
try:
    flag = str(os.environ.get("AUTO_TOP_TRANSFORMS", "")).strip().lower()
    if flag not in ("", "0", "false", "no", "off"):
        top_k_env = os.environ.get("AUTO_TOP_TRANSFORMS_K", "50")
        try:
            top_k_val = int(top_k_env)
        except Exception:
            top_k_val = 50
        include_unseen_flag = str(os.environ.get("AUTO_TOP_TRANSFORMS_INCLUDE_UNSEEN", "0")).strip().lower()
        include_unseen_val = include_unseen_flag in ("1", "true", "yes", "on")
        weights_path_val = os.environ.get("WEIGHTS_JSON")
        register_transforms_for_top_features(
            top_k=top_k_val,
            include_unseen=include_unseen_val,
            weights_path=weights_path_val,
            exclude_candles=True,
        )
except Exception:
    # Never fail import due to optional auto-registration
    pass

# Register close-distance features on import (lightweight registration only)
try:
    register_close_distance_features()
except Exception:
    pass

# Register RSI RPS features on import (guarded if pandas_ta unavailable)
try:
    register_rsi_rps_features()
except Exception:
    pass

from __future__ import annotations

from typing import List
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

try:
    register_all_transforms()
except Exception:
    pass

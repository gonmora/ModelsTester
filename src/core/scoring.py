# src/core/scoring.py
from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy, skew, kurtosis, spearmanr


def _is_binary(y: pd.Series) -> bool:
    z = y.dropna().unique()
    if len(z) <= 2 and set(z).issubset({0, 1}):
        return True
    return False


def _acf(x: pd.Series, lag: int) -> float:
    x = x.dropna().astype(float)
    if lag <= 0 or len(x) <= lag:
        return np.nan
    x0 = x.iloc[:-lag]
    x1 = x.iloc[lag:]
    if x0.std(ddof=0) == 0 or x1.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def _stability_ks(y: pd.Series) -> float:
    """KS-stat entre dos mitades temporales (proxy de estabilidad)."""
    y = y.dropna().astype(float)
    if len(y) < 200:
        return np.nan
    mid = len(y) // 2
    a = y.iloc[:mid]
    b = y.iloc[mid:]
    return float(ks_2samp(a, b, mode="auto").statistic)


def _outlier_rate_iqr(y: pd.Series, k: float = 5.0) -> float:
    yy = y.dropna().astype(float)
    if len(yy) == 0:
        return np.nan
    q1, q3 = np.quantile(yy, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return float(((yy < lo) | (yy > hi)).mean())


def _binary_runs_lengths(y: pd.Series) -> Tuple[float, float]:
    """Duración media y densidad (1/avg_gap) de eventos (y==1)."""
    yy = y.fillna(0).astype(int).values
    if yy.sum() == 0:
        return (0.0, 0.0)
    runs, gaps = [], []
    cnt = 0
    for v in yy:
        if v == 1:
            cnt += 1
        else:
            if cnt > 0:
                runs.append(cnt)
                cnt = 0
    if cnt > 0:
        runs.append(cnt)
    # gaps entre inicios de eventos
    starts = np.flatnonzero((yy == 1) & (np.r_[0, yy[:-1]] == 0))
    if len(starts) >= 2:
        gaps_arr = np.diff(starts)
        gaps = gaps_arr.tolist()
    avg_run = float(np.mean(runs)) if runs else 0.0
    density = float(1.0 / np.mean(gaps)) if gaps else (1.0 / len(yy))
    return avg_run, density


def _spearman_ic(y: pd.Series, x: pd.Series) -> float:
    m = pd.concat([y, x], axis=1).dropna()
    if len(m) < 50 or m.iloc[:, 1].std(ddof=0) == 0:
        return np.nan
    return float(abs(spearmanr(m.iloc[:, 0], m.iloc[:, 1], nan_policy="omit").statistic))


def score_target(
    y: pd.Series,
    df: pd.DataFrame,
    *,
    horizon: Optional[int] = None,
    price_col: str = "close",
) -> Dict[str, Any]:
    """
    Precalifica un target 'y' (Series indexada por tiempo) y devuelve métricas + health_score.

    - Detecta si es binario (clf) o continuo (reg).
    - Calcula salud (NaNs, constancia, outliers, estabilidad).
    - Métricas específicas (balance/entropía para clf; rango/skew/kurtosis/ACF para reg).
    - Predecibilidad proxy: IC con retornos pasados (1,3,12).
    - Leakage proxy: corr con retornos futuros vs pasados.
    """
    out: Dict[str, Any] = {}
    y = y.copy()
    # Asegurar alineación temporal
    if isinstance(df.index, pd.DatetimeIndex) and not isinstance(y.index, pd.DatetimeIndex):
        y.index = df.index[: len(y)]

    # ---------- Salud básica ----------
    nan_rate = float(y.isna().mean())
    value_counts = y.value_counts(dropna=True)
    const_rate = float((value_counts.max() / value_counts.sum()) if len(value_counts) > 0 else 1.0)
    outlier_rate = _outlier_rate_iqr(y)
    stability_ks = _stability_ks(y)

    out.update(
        nan_rate=nan_rate,
        const_rate=const_rate,
        outlier_rate_IQR=outlier_rate,
        stability_ks=stability_ks,
    )

    # ---------- Tipo de target ----------
    is_bin = _is_binary(y)
    out["task_type"] = "clf" if is_bin else "reg"

    # ---------- Métricas específicas ----------
    if is_bin:
        yy = y.dropna().astype(int)
        pos_rate = float(yy.mean()) if len(yy) else 0.0
        # razón mayoría/minoría
        if pos_rate in (0.0, 1.0):
            class_ratio = np.inf
        else:
            class_ratio = float(max(pos_rate, 1 - pos_rate) / max(1e-12, min(pos_rate, 1 - pos_rate)))
        # entropía (bits)
        probs = [pos_rate, 1 - pos_rate] if len(yy) else [1.0, 0.0]
        ent_bits = float(entropy(probs, base=2))
        avg_event_len, label_density = _binary_runs_lengths(yy)
        out.update(
            pos_rate=pos_rate,
            class_ratio=class_ratio,
            entropy_bits=ent_bits,
            avg_event_len=avg_event_len,
            label_density=label_density,
        )
    else:
        yr = y.dropna().astype(float)
        if len(yr) >= 5:
            range_rel = float((yr.max() - yr.min()) / (np.mean(np.abs(yr)) + 1e-12))
            skw = float(skew(yr, nan_policy="omit"))
            krt = float(kurtosis(yr, fisher=True, nan_policy="omit"))
        else:
            range_rel, skw, krt = (np.nan, np.nan, np.nan)
        acf1 = _acf(yr, 1)
        H = int(horizon) if (horizon is not None and horizon > 0) else 1
        acfH = _acf(yr, H)
        out.update(
            range_rel=range_rel,
            skew=skw,
            kurtosis=krt,
            acf_lag1=acf1,
            acf_lagH=acfH,
        )

    # ---------- Predecibilidad proxy (IC con features triviales) ----------
    # Retornos pasados sobre precio 'close'
    ic_map: Dict[str, float] = {}
    if price_col in df.columns:
        close = df[price_col].astype(float)
        rets = {
            "ret_1": close.pct_change(1).shift(1),
            "ret_3": close.pct_change(3).shift(1),
            "ret_12": close.pct_change(12).shift(1),
        }
        for k, s in rets.items():
            ic_map[k] = _spearman_ic(y, s)
    out["ic_vs_past_returns"] = ic_map
    out["ic_median"] = float(np.nanmedian(list(ic_map.values()))) if ic_map else np.nan

    # ---------- Leakage proxy ----------
    # Si correlaciona mucho más con retornos FUTUROS (sin lag) que con pasados, bandera roja.
    leak_flag = False
    leak_score = np.nan
    if price_col in df.columns:
        close = df[price_col].astype(float)
        future_ret_1 = close.pct_change(1).shift(-1)  # futuro inmediato
        past_ret_1 = close.pct_change(1).shift(1)     # pasado inmediato
        ic_future = _spearman_ic(y, future_ret_1)
        ic_past = _spearman_ic(y, past_ret_1)
        leak_score = float(ic_future - ic_past) if (not np.isnan(ic_future) and not np.isnan(ic_past)) else np.nan
        # umbral heurístico
        leak_flag = bool(leak_score > 0.2)
    out.update(leakage_proxy_score=leak_score, leakage_flag=leak_flag)

    # ---------- Health score agregado (0..1) ----------
    # Penalizaciones suaves; puedes re-pesar a gusto
    score = 1.0
    score -= 0.7 * min(1.0, nan_rate)                 # muchos NaN = mal
    score -= 0.5 * max(0.0, const_rate - 0.95)        # constancia extrema
    score -= 0.3 * (0.0 if math.isnan(outlier_rate) else min(1.0, outlier_rate * 2.0))
    score -= 0.3 * (0.0 if math.isnan(stability_ks) else min(1.0, stability_ks))  # KS alto = inestable
    if is_bin:
        pr = out["pos_rate"]
        imb = abs(pr - 0.5) * 2                        # 0 (balanceado) .. 1 (totalmente desbalanceado)
        score -= 0.4 * imb
        if pr in (0.0, 1.0):
            score -= 0.3                               # sin variación
    else:
        # baja varianza efectiva / poca persistencia = más difícil
        if not math.isnan(out.get("range_rel", np.nan)) and out["range_rel"] < 0.1:
            score -= 0.2
        if not math.isnan(out.get("acf_lag1", np.nan)) and abs(out["acf_lag1"]) < 0.02:
            score -= 0.1

    # señal mínima: premiar ic_median moderado
    ic_med = out.get("ic_median", np.nan)
    if not math.isnan(ic_med):
        score += 0.2 * min(0.5, ic_med)  # bonus pequeño por señal estable

    # leakage fuerte → recorte duro
    if leak_flag:
        score -= 0.4

    # clamp
    score = float(max(0.0, min(1.0, score)))
    out["health_score"] = score

    return out

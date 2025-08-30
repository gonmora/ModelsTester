# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Literal, Tuple


def _safe_align(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two pandas Series on their index, dropping any rows with missing values.

    Parameters
    ----------
    y : pd.Series
        The target series.
    x : pd.Series
        The feature series.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        A tuple of the aligned target and feature series.
    """
    df = pd.concat({"y": y, "x": x}, axis=1).dropna()
    return df["y"], df["x"]


def score_pair(y: pd.Series, x: pd.Series, task: Literal["reg", "clf"] = "reg") -> Dict[str, float]:
    """
    Compute various dependency scores between a target series y and a feature series x.

    Returns a dictionary with Pearson correlation, Spearman correlation,
    mutual information, distance correlation (if available), and effective sample size.

    Parameters
    ----------
    y : pd.Series
        The target variable.
    x : pd.Series
        The feature variable.
    task : Literal["reg", "clf"], default "reg"
        Whether the task is regression (continuous y) or classification (discrete y).

    Returns
    -------
    Dict[str, float]
        A dictionary with keys 'pearson', 'spearman', 'mi', 'dcor', 'n_eff'.
    """
    y_, x_ = _safe_align(y, x)
    # ensure numeric float arrays and finite values only
    y_arr = pd.to_numeric(y_, errors="coerce").to_numpy(dtype=float)
    x_arr = pd.to_numeric(x_, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr = y_arr[finite]
    x_arr = x_arr[finite]
    n = len(y_arr)
    # If not enough data, return NaNs
    if n < 10:
        return {"pearson": np.nan, "spearman": np.nan, "mi": np.nan, "dcor": np.nan, "n_eff": float(n)}

    # Pearson and Spearman correlations
    if np.nanstd(y_arr) == 0 or np.nanstd(x_arr) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(pd.Series(y_arr).corr(pd.Series(x_arr), method="pearson"))
        spearman = float(pd.Series(y_arr).corr(pd.Series(x_arr), method="spearman"))

    # Mutual information
    mi = np.nan
    try:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

        X = x_arr.reshape(-1, 1)
        if task == "clf":
            # For classification, ensure y is integer labels
            y_cls = y_arr.astype(int)
            mi = float(mutual_info_classif(X, y_cls, discrete_features=False, random_state=0)[0])
        else:
            mi = float(mutual_info_regression(X, y_arr, random_state=0)[0])
    except Exception:
        # If sklearn isn't available or fails, leave MI as NaN
        pass

    # Distance correlation (optional)
    dcor_val = np.nan
    try:
        import dcor
        # distance correlation expects float arrays with variability
        if np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
            dcor_val = np.nan
        else:
            dcor_val = float(dcor.distance_correlation(x_arr, y_arr))
    except Exception:
        pass

    return {
        "pearson": pearson,
        "spearman": spearman,
        "mi": mi,
        "dcor": dcor_val,
        "n_eff": float(n),
    }

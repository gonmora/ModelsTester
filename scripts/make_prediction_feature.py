#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an out-of-fold prediction feature from a past run selection.

Usage:
  python -m scripts.make_prediction_feature \
    --run-id <RUN_ID> --df-name <DF_NAME> [--folds 4] [--gap 288] [--out-name pred_<t>__<m>__oof]

This script:
  1) Loads the selection (target, features, model) for the given run_id.
  2) Rebuilds y and X over the full dataframe.
  3) Runs temporal K-fold (walk-forward style) with a GAP between train and test.
  4) Computes out-of-fold predictions aligned to the full index.
  5) Saves the feature to storage (Parquet) under the requested name.

Notes:
  - For TF models, reuses the helper from extended_metrics_for_run; otherwise falls back to logistic baseline.
  - The saved feature can be auto-registered by importing src.prediction_features.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable as 'src'
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from src.registry import registry
from src import storage
from scripts.extended_metrics_for_run import _load_selection, _fit_predict_tf  # type: ignore


def _prepare_full_xy(df: pd.DataFrame, sel: Any) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute full-length target and feature matrix (no split), cleaned and aligned."""
    try:
        target_id = sel.target
        feat_ids = sel.features
    except Exception:
        target_id = sel["target"]
        feat_ids = sel["features"]
    # Compute target
    y = registry.targets[target_id](df)
    # Compute features
    feats: Dict[str, pd.Series] = {}
    for fid in feat_ids:
        try:
            x = registry.features[fid](df)
            if isinstance(x, pd.DataFrame) and getattr(x, 'shape', (0,0))[1] == 1:
                x = x.iloc[:, 0]
            feats[fid] = x
        except Exception:
            # skip failing features
            continue
    if not feats:
        raise RuntimeError("No features computed for this DF to build OOF predictions")
    X = pd.concat(feats, axis=1)
    combined = pd.concat([y, X], axis=1)
    mask = combined.iloc[:, 0].notna()
    y_clean = combined.loc[mask, combined.columns[0]]
    X_clean = combined.loc[mask, combined.columns[1:]].replace([np.inf, -np.inf], np.nan)
    try:
        X_clean = X_clean.infer_objects(copy=False)
    except Exception:
        pass
    X_clean = X_clean.dropna(axis=1, how='all')
    if y_clean.nunique() < 2 or len(X_clean) < 200:
        raise RuntimeError("Insufficient data or single-class target after cleaning")
    return y_clean, X_clean


def _fit_predict_sel(model_id: str, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    if model_id.startswith('tf_mlp'):
        return _fit_predict_tf(model_id, X_tr, y_tr, X_te)
    if model_id == 'hgb_regressor':
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

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
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)
    # Logistic baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def make_oof_feature(run_id: str, df_name: str, folds: int = 4, gap: int = 288, out_name: str | None = None, db_path: str | None = None) -> str:
    """Compute and save an OOF prediction feature for a given run selection.

    Returns the saved feature name.
    """
    # Load selection and df
    if not db_path:
        db_path = os.path.join(ROOT, 'runs_24M.db') if os.path.exists(os.path.join(ROOT, 'runs_24M.db')) else 'runs.db'
    sel = _load_selection(db_path, run_id)
    # Enable TA / PV feature registries if available (imports have side effects).
    try:
        import src.ta_features  # noqa: F401
    except Exception:
        pass
    try:
        import src.pv_components  # noqa: F401
    except Exception:
        pass
    df = storage.load_dataframe(df_name)
    # Compute full y/X
    y_full, X_full = _prepare_full_xy(df, sel)
    n = len(y_full)
    # Folds
    folds = max(2, int(folds))
    gap = max(0, int(gap))
    # Result container aligned to y_full index
    oof = pd.Series(index=y_full.index, dtype=float)
    for i in range(folds):
        te_start = int(n * i / folds)
        te_end = int(n * (i + 1) / folds)
        split_idx = max(0, te_start - gap)
        # Minimum sizes
        if split_idx < 100 or (te_end - te_start) < 50:
            continue
        y_tr = y_full.iloc[:split_idx]
        y_te = y_full.iloc[te_start:te_end]
        if y_tr.dropna().nunique() < 2 or y_te.dropna().nunique() < 1:
            continue
        X_tr = X_full.iloc[:split_idx]
        X_te = X_full.iloc[te_start:te_end]
        try:
            preds = _fit_predict_sel(sel.model, X_tr, y_tr, X_te)
            oof.iloc[te_start:te_end] = preds
        except Exception:
            continue

    # Name and save
    base_name = out_name
    if not base_name:
        # Default: pred_<target>__<model>__oof
        safe_model = str(sel.model).replace('/', '_')
        base_name = f"pred_{sel.target}__{safe_model}__oof"
    storage.save_feature(df_name, base_name, oof)
    return base_name


def main() -> None:
    ap = argparse.ArgumentParser(description='Make OOF prediction feature from a past run')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--df-name', required=True)
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--gap', type=int, default=288)
    ap.add_argument('--out-name', default=None)
    ap.add_argument('--db-path', default=None)
    args = ap.parse_args()

    name = make_oof_feature(args.run_id, args.df_name, folds=args.folds, gap=args.gap, out_name=args.out_name, db_path=args.db_path)
    print(f"Saved feature '{name}' for df='{args.df_name}'")


if __name__ == '__main__':
    main()

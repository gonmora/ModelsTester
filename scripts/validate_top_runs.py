#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate top runs from an exported Top (Unified) Excel by recomputing metrics
with a temporal GAP and running a permutation test.

Usage:
  python scripts/validate_top_runs.py --excel Top_Unified_20250901_073221.xlsx \
      --db runs_24M.db --df BTCUSDT_5m_20230831_20250830 --top 3 --gaps 144 288
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root on sys.path for src.* imports
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.extended_metrics_for_run import (  # type: ignore
    _load_selection,
    _prepare_xy,
    _fit_predict_tf,
)


def _clf_probs(sel_model: str, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    if sel_model.startswith("tf_mlp"):
        return _fit_predict_tf(sel_model, X_tr, y_tr, X_te)
    # Fallback: logistic baseline
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


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    auc = float(roc_auc_score(y_true, probs))
    ap = float(average_precision_score(y_true, probs))
    pos_rate = float(y_true.mean()) if len(y_true) else float('nan')
    ap_lift = (ap / pos_rate) if pos_rate and pos_rate > 0 else float('nan')

    # P@k
    def p_at(frac: float) -> float:
        k = max(1, int(len(probs) * frac))
        idx = np.argsort(-probs)[:k]
        return float(y_true[idx].mean())

    # Brier & KS
    from sklearn.metrics import brier_score_loss
    probs_c = np.clip(probs, 0, 1)
    brier = float(brier_score_loss(y_true, probs_c))
    pos = np.sort(probs_c[y_true == 1])
    neg = np.sort(probs_c[y_true == 0])
    ks = float('nan')
    if len(pos) and len(neg):
        thresh = np.unique(np.concatenate([pos, neg]))
        cdf_pos = np.searchsorted(pos, thresh, side='right') / len(pos)
        cdf_neg = np.searchsorted(neg, thresh, side='right') / len(neg)
        ks = float(np.max(np.abs(cdf_pos - cdf_neg)))

    return {
        'auc': auc,
        'ap': ap,
        'pos_rate': pos_rate,
        'ap_lift': ap_lift,
        'p@1%': p_at(0.01),
        'p@5%': p_at(0.05),
        'p@10%': p_at(0.10),
        'brier': brier,
        'ks': ks,
    }


def validate_runs(excel_path: str, db_path: str, df_name: str, top_n: int, gaps: List[int]) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df.sort_values(['score','n_test'], ascending=[False, False]).head(top_n)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        run_id = str(r['run_id'])
        try:
            sel = _load_selection(db_path, run_id)
        except Exception as e:
            rows.append({'run_id': run_id, 'error': f'load_selection: {e}'})
            continue
        for gap in gaps:
            try:
                Xtr, ytr, Xte, yte = _prepare_xy(df_name, sel, gap=gap)
                # Only classification for this validation
                if not (ytr.dropna().nunique() <= 2 and set(ytr.dropna().unique()).issubset({0,1})):
                    rows.append({'run_id': run_id, 'gap': gap, 'error': 'non-binary target path skipped'})
                    continue
                probs = _clf_probs(sel.model, Xtr, ytr, Xte)
                y_true = yte.to_numpy().astype(int)
                m = _metrics(y_true, probs)
                # Permutation test
                rng = np.random.default_rng(42)
                y_perm = y_true.copy()
                rng.shuffle(y_perm)
                m_perm = _metrics(y_perm, probs)
                rows.append({
                    'run_id': run_id,
                    'target': sel.target,
                    'model': sel.model,
                    'gap': int(gap),
                    'n_test': int(len(y_true)),
                    'auc': m['auc'],
                    'ap': m['ap'],
                    'pos_rate': m['pos_rate'],
                    'ap_lift': m['ap_lift'],
                    'p@1%': m['p@1%'],
                    'p@5%': m['p@5%'],
                    'p@10%': m['p@10%'],
                    'brier': m['brier'],
                    'ks': m['ks'],
                    'auc_perm': m_perm['auc'],
                    'ap_perm': m_perm['ap'],
                })
            except Exception as e:
                rows.append({'run_id': run_id, 'gap': int(gap), 'error': str(e)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--excel', required=True)
    ap.add_argument('--db', default='runs_24M.db')
    ap.add_argument('--df', default='BTCUSDT_5m_20230831_20250830')
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--gaps', type=int, nargs='+', default=[144, 288])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = validate_runs(args.excel, args.db, args.df, args.top, args.gaps)
    # Pretty print
    if len(res):
        cols = ['run_id','target','model','gap','n_test','auc','ap','pos_rate','ap_lift','p@1%','p@5%','p@10%','brier','ks','auc_perm','ap_perm','error']
        cols = [c for c in cols if c in res.columns]
        print(res[cols].to_string(index=False))
    else:
        print('No results.')
    if args.out:
        base_dir = os.path.dirname(args.out) or '.'
        os.makedirs(base_dir, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()


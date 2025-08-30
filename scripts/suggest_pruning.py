#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suggest pruning/disable candidates for features, targets, and models based on
historical runs and current weights/stats.

Heuristics (conservative, read-only):
- Models: flag models with low AUC/AP and/or very slow fit times.
- Targets: flag targets with low AUC/AP lift for down-weighting (not hard disable).
- Features: flag features with low mean affinity and many uses, and/or frequent
  compute failures across runs.

Usage:
  python scripts/suggest_pruning.py \
    --db runs_BIG.db \
    --weights data/weights_BIG.json \
    --min-runs 10 --min-feat-n 80 --mean-cut 0.03 --fail-rate 0.05

Outputs a human-readable report. No changes are made to files.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np


def load_runs(db_path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with closing(sqlite3.connect(db_path)) as con:
        cur = con.execute(
            """
            SELECT run_id, status, selection_json, metrics_json, artifacts_json
            FROM runs WHERE metrics_json IS NOT NULL
            """
        )
        for rid, status, sj, mj, aj in cur.fetchall():
            try:
                sel = json.loads(sj) if sj else None
            except Exception:
                sel = None
            try:
                met = json.loads(mj) if mj else None
            except Exception:
                met = None
            try:
                art = json.loads(aj) if aj else None
            except Exception:
                art = None
            if not sel or not met:
                continue
            rows.append({
                'run_id': rid,
                'status': status,
                'target': sel.get('target'),
                'model': sel.get('model'),
                'k': len(sel.get('features', [])) if sel.get('features') is not None else None,
                'auc': met.get('auc'),
                'ap': met.get('ap'),
                'pos_rate_test': met.get('pos_rate_test'),
                'artifacts': art,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    for c in ['auc', 'ap', 'pos_rate_test']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # AP lift
    if 'ap' in df and 'pos_rate_test' in df:
        base = df['pos_rate_test'].replace(0, np.nan)
        df['ap_lift'] = df['ap'] / base
    return df


def aggregate_models(df: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    g = df[df['status'] == 'SUCCESS'].groupby('model')
    out = g.agg(
        runs=('auc', lambda s: int(s.notna().sum())),
        auc_median=('auc', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
        ap_median=('ap', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
        ap_lift_median=('ap_lift', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
    ).reset_index()
    return out[out['runs'] >= min_runs].sort_values(['auc_median', 'runs'], ascending=[False, False])


def aggregate_targets(df: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    g = df[df['status'] == 'SUCCESS'].groupby('target')
    out = g.agg(
        runs=('auc', lambda s: int(s.notna().sum())),
        auc_median=('auc', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
        ap_median=('ap', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
        ap_lift_median=('ap_lift', lambda s: float(pd.to_numeric(s, errors='coerce').median())),
    ).reset_index()
    return out[out['runs'] >= min_runs].sort_values(['auc_median', 'runs'], ascending=[False, False])


def feature_failure_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Count feature compute failures from artifacts
    counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    for _, row in df.iterrows():
        sel_feats = row.get('k') or 0
        arts = row.get('artifacts') or {}
        failed = arts.get('failed_features') or []
        # increment failure counts
        for item in failed:
            try:
                fid = item.get('id') if isinstance(item, dict) else None
            except Exception:
                fid = None
            if not fid:
                continue
            counts[fid] = counts.get(fid, 0) + 1
        # For approximate denominator, increment totals for all failed+successful IDs
        # If selection JSON isn't available in this context per-feature, we only have failures.
        # We keep totals as failure counts for rate ranking.
    if not counts:
        return pd.DataFrame(columns=['feature', 'fail_count', 'fail_rate'])
    data = [{'feature': k, 'fail_count': v, 'fail_rate': float(v)} for k, v in counts.items()]
    return pd.DataFrame(data).sort_values(['fail_rate', 'fail_count'], ascending=[False, False])


def suggest_models(model_tbl: pd.DataFrame, auc_cut: float = 0.515, ap_lift_cut: float = 1.02) -> List[str]:
    to_disable: List[str] = []
    for _, r in model_tbl.iterrows():
        model = r['model']
        auc_med = r['auc_median']
        ap_lift_med = r['ap_lift_median']
        if (not np.isnan(auc_med) and auc_med < auc_cut) and (np.isnan(ap_lift_med) or ap_lift_med < ap_lift_cut):
            to_disable.append(model)
    return to_disable


def suggest_targets(tgt_tbl: pd.DataFrame, auc_cut: float = 0.515, ap_lift_cut: float = 1.02) -> List[str]:
    to_downweight: List[str] = []
    for _, r in tgt_tbl.iterrows():
        tgt = r['target']
        auc_med = r['auc_median']
        ap_lift_med = r['ap_lift_median']
        if (not np.isnan(auc_med) and auc_med < auc_cut) and (np.isnan(ap_lift_med) or ap_lift_med < ap_lift_cut):
            to_downweight.append(tgt)
    return to_downweight


def suggest_features(weights_path: str, min_feat_n: int = 80, mean_cut: float = 0.03,
                     fail_df: pd.DataFrame | None = None, max_list: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        with open(weights_path, 'r', encoding='utf-8') as f:
            w = json.load(f)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    st: Dict[str, Any] = w.get('features_stats', {}) or {}
    rows: List[Dict[str, Any]] = []
    for fid, stats in st.items():
        try:
            n = int(stats.get('n', 0) or 0)
            mean = float(stats.get('mean', 0.0) or 0.0)
            best = float(stats.get('best', 0.0) or 0.0)
        except Exception:
            n, mean, best = 0, 0.0, 0.0
        rows.append({'feature': fid, 'n': n, 'mean': mean, 'best': best})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, pd.DataFrame()
    low_mean = df[(df['n'] >= min_feat_n) & (df['mean'] < mean_cut)].sort_values(['mean', 'n'])
    low_mean = low_mean.head(max_list)
    fail_rank = fail_df[['feature', 'fail_count', 'fail_rate']].head(max_list) if fail_df is not None and len(fail_df) else pd.DataFrame()
    return low_mean, fail_rank


def main() -> None:
    ap = argparse.ArgumentParser(description='Suggest pruning candidates from runs and weights')
    ap.add_argument('--db', dest='db_path', default='runs_BIG.db')
    ap.add_argument('--weights', dest='weights_path', default='data/weights_BIG.json')
    ap.add_argument('--min-runs', dest='min_runs', type=int, default=10)
    ap.add_argument('--min-feat-n', dest='min_feat_n', type=int, default=80)
    ap.add_argument('--mean-cut', dest='mean_cut', type=float, default=0.03)
    ap.add_argument('--fail-rate', dest='fail_rate', type=float, default=0.05)  # reserved, for future denominator
    args = ap.parse_args()

    df = load_runs(args.db_path)
    if len(df) == 0:
        print(f"No runs found in {args.db_path}")
        return

    print(f"Loaded {len(df)} runs from {args.db_path}")
    mdl_tbl = aggregate_models(df, args.min_runs)
    tgt_tbl = aggregate_targets(df, args.min_runs)
    fail_df = feature_failure_stats(df)

    print("\nModels (aggregated):")
    print(mdl_tbl.to_string(index=False))
    mdls_to_disable = suggest_models(mdl_tbl)
    if mdls_to_disable:
        print("\nSuggest disabling models:", ", ".join(mdls_to_disable))
    else:
        print("\nNo models suggested for disabling (based on thresholds).")

    print("\nTargets (aggregated):")
    print(tgt_tbl.to_string(index=False))
    tgts_to_down = suggest_targets(tgt_tbl)
    if tgts_to_down:
        print("\nSuggest down-weighting targets:", ", ".join(tgts_to_down))
    else:
        print("\nNo targets suggested for down-weighting (based on thresholds).")

    low_mean_df, fail_rank = suggest_features(args.weights_path, args.min_feat_n, args.mean_cut, fail_df)
    print("\nFeatures with low mean affinity (n >= {}, mean < {}):".format(args.min_feat_n, args.mean_cut))
    if len(low_mean_df):
        print(low_mean_df.to_string(index=False))
    else:
        print("(none)")
    if len(fail_rank):
        print("\nMost frequent compute failures (top):")
        print(fail_rank.to_string(index=False))
    else:
        print("\nNo feature compute failures recorded (or none parsed).")


if __name__ == '__main__':
    main()


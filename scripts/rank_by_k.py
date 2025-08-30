#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute K-quality rankings (by number of features used) from the runs history.

Produces aggregated medians/IQRs for AUC and AP (and AP lift vs base rate),
optionally grouped globally, by model, or by target.

Usage examples:
  python scripts/rank_by_k.py --db runs.db --by all
  python scripts/rank_by_k.py --db runs_BIG.db --by global --min-runs 5

Outputs pretty-printed tables to stdout. Optionally save CSVs via --out-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


def load_runs(db_path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with closing(sqlite3.connect(db_path)) as con:
        cur = con.execute(
            """
            SELECT run_id, started_at, status, selection_json, metrics_json
            FROM runs
            WHERE status = 'SUCCESS' AND metrics_json IS NOT NULL
            ORDER BY started_at ASC
            """
        )
        for rid, started_at, status, sj, mj in cur.fetchall():
            try:
                sel = json.loads(sj) if sj else None
                met = json.loads(mj) if mj else None
            except Exception:
                sel, met = None, None
            if not sel or not met:
                continue
            try:
                k = int(len(sel.get("features", [])))
            except Exception:
                k = None
            rows.append(
                {
                    "run_id": rid,
                    "started_at": started_at,
                    "status": status,
                    "k": k,
                    "model": sel.get("model"),
                    "target": sel.get("target"),
                    "auc": met.get("auc"),
                    "ap": met.get("ap"),
                    "fit_time_sec": met.get("fit_time_sec"),
                    "predict_time_sec": met.get("predict_time_sec"),
                    "pos_rate_test": met.get("pos_rate_test"),
                }
            )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Coerce to numeric and drop obvious invalids
    for col in [
        "k",
        "auc",
        "ap",
        "fit_time_sec",
        "predict_time_sec",
        "pos_rate_test",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # AP lift when possible
    if "pos_rate_test" in df.columns:
        base = df["pos_rate_test"].replace(0, np.nan)
        df["ap_lift"] = df["ap"]/base
    else:
        df["ap_lift"] = np.nan
    return df


def _agg_table(df: pd.DataFrame, group_cols: List[str], min_runs: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    g = df.groupby(group_cols)
    def q(series: pd.Series, p: float) -> float:
        try:
            return float(series.quantile(p))
        except Exception:
            return float("nan")
    out = g.agg(
        runs=("auc", lambda s: int(s.notna().sum())),
        auc_median=("auc", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        auc_iqr=("auc", lambda s: q(pd.to_numeric(s, errors="coerce"), 0.75) - q(pd.to_numeric(s, errors="coerce"), 0.25)),
        ap_median=("ap", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        ap_lift_median=("ap_lift", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        fit_time_median_sec=("fit_time_sec", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        predict_time_median_sec=("predict_time_sec", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
    ).reset_index()
    out = out[out["runs"] >= int(min_runs)].copy()
    # Sort by AUC median then runs
    sort_cols = ["auc_median", "runs"]
    out = out.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)
    return out


def print_tables(df: pd.DataFrame, by: str, min_runs: int, out_dir: Optional[str] = None) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    if by in ("global", "all"):
        print("\nK quality (global):")
        t = _agg_table(df, ["k"], min_runs)
        print(t.to_string(index=False))
        if out_dir:
            t.to_csv(os.path.join(out_dir, "k_quality_global.csv"), index=False)
    if by in ("model", "all"):
        print("\nK x Model quality:")
        t = _agg_table(df, ["k", "model"], min_runs)
        print(t.to_string(index=False))
        if out_dir:
            t.to_csv(os.path.join(out_dir, "k_quality_by_model.csv"), index=False)
    if by in ("target", "all"):
        print("\nK x Target quality:")
        t = _agg_table(df, ["k", "target"], min_runs)
        print(t.to_string(index=False))
        if out_dir:
            t.to_csv(os.path.join(out_dir, "k_quality_by_target.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank K-quality from runs history")
    ap.add_argument("--db", dest="db_path", default="runs_BIG.db", help="Path to runs SQLite DB (default: runs_BIG.db)")
    ap.add_argument("--by", dest="by", choices=["global", "model", "target", "all"], default="all")
    ap.add_argument("--min-runs", dest="min_runs", type=int, default=5, help="Minimum runs per group")
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="Optional directory to save CSVs")
    args = ap.parse_args()

    df = load_runs(args.db_path)
    if len(df) == 0:
        print(f"No successful runs with metrics found in '{args.db_path}'.")
        return
    print(f"Loaded {len(df)} runs from {args.db_path}")
    print_tables(df, by=args.by, min_runs=args.min_runs, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh ATR-based targets on disk and run a quick smoke test to verify metrics.

Actions:
1) Recompute and overwrite cached Parquet targets for all ATR/regime targets found
   in the registry for the provided dataframe name (DF_NAME).
2) Optionally clear their stats in the active WEIGHTS_JSON (targets_stats entries)
   to remove inflated historical values.
3) Run a simple logistic baseline on a few built-in features to check that AUC is
   computed (non-NaN) for each refreshed target and report a compact summary.

Usage examples:
    python scripts/refresh_atr_targets_and_smoke_test.py \
        --df-name BTCUSDT_5m_20230831_20250831 \
        --weights-json data/weights_24M.json \
        --reset-stats

Notes:
    - This script uses only built-in fast features (pct_change, momentum_3, rolling_std_5)
      to keep the smoke test lightweight and deterministic.
    - It respects the project storage cache and overwrites targets on disk.
"""
from __future__ import annotations

import argparse
import os
from typing import List
import sys

# Ensure project root is importable as 'src'
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df-name", required=True, help="Dataset logical name (DF_NAME) used for cache naming")
    ap.add_argument("--parquet", default=None, help="Optional path to an existing Parquet dataset to load df from (bypasses db_to_df)")
    ap.add_argument("--weights-json", default=os.environ.get("WEIGHTS_JSON", None), help="Path to weights JSON (WEIGHTS_JSON)")
    ap.add_argument("--reset-stats", action="store_true", help="Clear targets_stats for ATR/regime targets in the weights JSON")
    args = ap.parse_args()

    if args.weights_json:
        os.environ["WEIGHTS_JSON"] = args.weights_json

    # Lazy imports after env setup
    from src import storage
    from src.registry import registry
    from src.runner.engine import WeightStore

    # Load dataframe: prefer cached Parquet; optionally allow direct Parquet path to avoid db_to_df
    if args.parquet:
        if not os.path.exists(args.parquet):
            raise FileNotFoundError(f"Parquet not found: {args.parquet}")
        df = pd.read_parquet(args.parquet)
        print(f"Loaded df from explicit Parquet: {args.parquet}")
    else:
        try:
            from src.storage import dataframe_exists
            if not dataframe_exists(args.df_name):
                raise FileNotFoundError(
                    f"Cached dataset not found for DF_NAME='{args.df_name}'.\n"
                    f"Either pass --parquet <path.parquet> or pre-cache the dataset (e.g., run the engine once or save it to data/{args.df_name}.parquet)."
                )
        except Exception:
            # If storage module import fails for some reason, try best-effort load
            pass
        df = storage.load_dataframe(args.df_name)

    # Discover ATR/regime targets in the registry
    prefixes = ("updown_atr_", "trend_updown_atr_", "mr_updown_atr_")
    tgt_names: List[str] = [t for t in registry.targets.keys() if t.startswith(prefixes)]
    if not tgt_names:
        print("No ATR/regime targets found in registry. Nothing to do.")
        return
    tgt_names = sorted(tgt_names)

    print(f"Refreshing {len(tgt_names)} targets for DF='{args.df_name}'...")
    summary = []
    for t in tgt_names:
        try:
            func = registry.targets[t]
            y = func(df)
            # Basic coverage stats
            util = float(pd.Series(y).notna().mean())
            pos = float(pd.Series(y).fillna(0).mean()) if util > 0 else float("nan")
            storage.save_target(args.df_name, t, y)
            summary.append((t, util, pos))
            print(f" - {t}: utiles={util:.3f} pos={pos:.3f}")
        except Exception as e:
            print(f" ! {t}: ERROR {e}")

    # Optionally reset stats for these targets in weights
    if args.reset_stats:
        try:
            ws = WeightStore(path=os.environ.get("WEIGHTS_JSON") or WeightStore().path)
            for t in tgt_names:
                if t in ws.targets_stats:
                    ws.targets_stats.pop(t, None)
            ws.save()
            print(f"Cleared targets_stats for {len(tgt_names)} targets in '{ws.path}'.")
        except Exception as e:
            print(f"Warning: failed to reset stats in weights JSON: {e}")

    # Smoke test with a simple logistic baseline and 3 built-in features
    print("\nSmoke test (logistic_baseline) on 3 built-in features...")
    feat_ids = ["pct_change", "momentum_3", "rolling_std_5"]
    missing = [f for f in feat_ids if f not in registry.features]
    if missing:
        raise RuntimeError(f"Missing required built-in features in registry: {missing}")
    # Compute features once
    features = {fid: registry.features[fid](df) for fid in feat_ids}
    model_fn = registry.models.get("logistic_baseline")
    if model_fn is None:
        print("No logistic_baseline model registered; skipping smoke test.")
        return

    rows = []
    for t in tgt_names:
        try:
            y = storage.load_target(args.df_name, t)
            # Call model with a minimal selection dict
            _, metrics = model_fn(y, features, df, selection={"target": t, "features": feat_ids, "model": "logistic_baseline"})
            auc = metrics.get("auc")
            n_test = metrics.get("n_test")
            rows.append({"target": t, "auc": auc, "n_test": n_test})
        except Exception as e:
            rows.append({"target": t, "auc": float("nan"), "n_test": 0, "error": str(e)})

    out = pd.DataFrame(rows)
    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print("\nAUC by target (logistic baseline):")
        print(out.sort_values(["auc"], ascending=[False]).to_string(index=False))


if __name__ == "__main__":
    main()

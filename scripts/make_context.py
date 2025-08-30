#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a clean run context (DB + weights) for a dataset window, e.g. 24M or 36M.

This does NOT touch your parquet artifacts. It only prepares or resets:
- a runs DB file (SQLite)
- a weights JSON file

Usage examples:
  # Prepare 24M context (defaults: runs_24M.db, data/weights_24M.json)
  python scripts/make_context.py --name 24M --reset

  # Prepare 36M context
  python scripts/make_context.py --name 36M --reset

  # Custom file paths
  python scripts/make_context.py --db runs_MY.db --weights data/weights_MY.json --reset

After running with --reset, use the printed notebook snippet and monitor command.
"""
from __future__ import annotations

import argparse
import json
import os


def default_paths(name: str) -> tuple[str, str]:
    name = name.strip()
    return (f"runs_{name}.db", f"data/weights_{name}.json")


def reset_files(db_path: str, weights_path: str) -> None:
    # Remove DB
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed DB: {db_path}")
    except Exception as e:
        print(f"Warning: could not remove DB {db_path}: {e}")
    # Remove weights
    try:
        if os.path.exists(weights_path):
            os.remove(weights_path)
            print(f"Removed weights: {weights_path}")
    except Exception as e:
        print(f"Warning: could not remove weights {weights_path}: {e}")


def ensure_weights(weights_path: str) -> None:
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    if not os.path.exists(weights_path):
        # Minimal skeleton; engine will add fields on save
        data = {
            "features": {},
            "targets": {},
            "disabled_features": [],
            "disabled_targets": [],
            "k_weights": {},
            "features_stats": {},
            "targets_stats": {},
            "models": {},
            "disabled_models": [],
            "models_stats": {},
        }
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Created weights skeleton: {weights_path}")


def print_instructions(db_path: str, weights_path: str) -> None:
    print("\nNotebook snippet (paste at top of your notebook):\n")
    print("""
import os
from src.runner.engine import run_loop, WeightStore

# Pick DF_NAME for 24M or 36M (set your actual dates)
# DF_NAME = "BTCUSDT_5m_20230830_20250829"  # 24M example
# DF_NAME = "BTCUSDT_5m_20220830_20250829"  # 36M example

os.environ["WEIGHTS_JSON"] = "{weights}"
DB_PATH = "{db}"
ws = WeightStore(path=os.environ["WEIGHTS_JSON"])

# Example run
# run_ids = run_loop(df_name=DF_NAME, split_id="default", n_runs=200, seed=42, db_path=DB_PATH, weight_store=ws)
""".format(weights=weights_path, db=db_path))

    print("Monitor command:\n")
    print(f"python monitor.py --db-path {db_path} --weights-json {weights_path} --interval 2 --top-k 10")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a clean run context (DB + weights)")
    ap.add_argument("--name", default="24M", help="Context name to derive default file paths (e.g., 24M, 36M)")
    ap.add_argument("--db", dest="db_path", default=None, help="Custom DB path (overrides --name)")
    ap.add_argument("--weights", dest="weights_path", default=None, help="Custom weights path (overrides --name)")
    ap.add_argument("--reset", action="store_true", help="Remove existing DB/weights before creating")
    args = ap.parse_args()

    db, weights = args.db_path, args.weights_path
    if db is None or weights is None:
        d, w = default_paths(args.name)
        db = db or d
        weights = weights or w

    if args.reset:
        reset_files(db, weights)

    ensure_weights(weights)
    print_instructions(db, weights)


if __name__ == "__main__":
    main()


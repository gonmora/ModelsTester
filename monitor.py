#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live monitor for ModelsTester runs.

Displays a periodically refreshed summary of top features/targets and recent
runs using the reporting utilities.

Usage:
    python monitor.py --interval 2 --top-k 10 --db-path runs.db

Stop with Ctrl-C.
"""
from __future__ import annotations

import argparse

import os
import warnings

# Avoid heavy TA auto-registration and noisy warnings in the monitor process
os.environ.setdefault("DISABLE_TA_AUTOREG", "1")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")

live_monitor = None  # will import after args/env prepared in main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live monitor for experiment runs")
    p.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    p.add_argument("--top-k", type=int, default=10, help="Number of top features to display")
    p.add_argument("--db-path", type=str, default="runs.db", help="Path to runs SQLite DB")
    p.add_argument("--no-tables", action="store_true", help="Do not display DataFrame tables (text only)")
    p.add_argument("--weights-json", type=str, default=None, help="Path to weights JSON (overrides WEIGHTS_JSON)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Prepare env before importing reporting/engine
    if args is not None:
        # Optional: route weights.json per-dataset
        # If user passes --weights-json, set env so WeightStore picks it up at import time
        if hasattr(args, 'weights_json') and args.weights_json:
            os.environ["WEIGHTS_JSON"] = args.weights_json
    # Defer import so WEIGHTS_JSON is honored
    from src.reporting import live_monitor as _lm
    _lm(interval=args.interval, top_k=args.top_k, db_path=args.db_path, refresh_tables=not args.no_tables)


if __name__ == "__main__":
    main()

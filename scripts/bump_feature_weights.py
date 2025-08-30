#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporarily prioritize a set of features by bumping their weights in the
weights JSON. Safe, idempotent, and reversible (you can lower later).

Usage examples:
  # Boost a few features in BIG weights to weight=3.0
  python scripts/bump_feature_weights.py \
      --weights data/weights_BIG.json \
      --set-weight 3.0 \
      vol_sum_to_1w_max_pct vol_sum_1d_change_pct vol_sum_1w_max_cal

  # Multiply current weights by 2.0
  python scripts/bump_feature_weights.py --weights data/weights_BIG.json --factor 2.0 <feat1> <feat2>

If no feature list is provided, a default set is used (the recent volume-based
features and the tos_stdevall):
  vol_sum_to_1w_max_pct, vol_sum_1d_change_pct, vol_sum_1w_max_cal, ta_statistics_tos_stdevall
"""
from __future__ import annotations

import argparse
import json
from typing import List


DEFAULT_FEATURES = [
    "vol_sum_to_1w_max_pct",
    "vol_sum_1d_change_pct",
    "vol_sum_1w_max_cal",
    "ta_statistics_tos_stdevall",
]


def bump(weights_path: str, features: List[str], set_weight: float | None, factor: float | None) -> None:
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("features", {}) or {}
    changed = []
    for fid in features:
        old = float(feats.get(fid, 1.0))
        if set_weight is not None:
            new = float(set_weight)
        elif factor is not None:
            new = float(old) * float(factor)
        else:
            new = max(old, 3.0)
        feats[fid] = new
        changed.append((fid, old, new))
    data["features"] = feats
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {len(changed)} feature weights in {weights_path}:")
    for fid, old, new in changed:
        print(f" - {fid}: {old:.3f} -> {new:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bump/prioritize feature weights")
    ap.add_argument("--weights", default="data/weights_BIG.json", help="Path to weights JSON")
    ap.add_argument("--set-weight", type=float, default=None, help="Set weight to this value")
    ap.add_argument("--factor", type=float, default=None, help="Multiply current weight by this factor")
    ap.add_argument("features", nargs="*", help="Feature IDs to bump; defaults to a curated list")
    args = ap.parse_args()

    feats = args.features if args.features else DEFAULT_FEATURES
    bump(args.weights, feats, args.set_weight, args.factor)


if __name__ == "__main__":
    main()


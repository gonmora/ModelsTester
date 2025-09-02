# -*- coding: utf-8 -*-
"""
Search engine to schedule and execute weighted experiments.

Features:
- Weighted random selection of target and 3..10 features (without replacement).
- Pluggable weights store persisted to JSON (data/weights.json).
- Uses run_experiment_with_selection to avoid internal random selection.

This is a minimal, incremental "motor" you can extend with smarter policies.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from ..registry import registry
# Ensure basic components (targets/models) and TA features are loaded on import
try:  # side-effect import
    from .. import basic_components  # noqa: F401
except Exception:
    pass
try:  # side-effect import
    if not os.environ.get("DISABLE_TA_AUTOREG"):
        from .. import ta_features  # noqa: F401
except Exception:
    pass

try:  # side-effect import (Peaks & Valleys components)
    if not os.environ.get("DISABLE_PV_AUTOREG"):
        from .. import pv_components  # noqa: F401
except Exception:
    pass

try:  # side-effect import (feature transforms)
    from .. import feature_transforms  # noqa: F401
except Exception:
    pass
from .experiment import run_experiment_with_selection
from ..core import history
from .. import storage
import pandas as pd


DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "logistic_baseline")
DEFAULT_EVAL = "default"
WEIGHTS_PATH = os.environ.get("WEIGHTS_JSON", os.path.join("data", "weights.json"))


class WeightStore:
    def __init__(self, path: str = WEIGHTS_PATH) -> None:
        self.path = path
        self.features: Dict[str, float] = {}
        self.targets: Dict[str, float] = {}
        self.k_weights: Dict[str, float] = {}
        self.disabled_features: List[str] = []
        self.disabled_targets: List[str] = []
        # Models support
        self.models: Dict[str, float] = {}
        self.disabled_models: List[str] = []
        # Usage/performance stats
        # Usage/performance stats
        # features_stats[name] = {"n": int, "mean": float, "best": float, "last": float}
        # targets_stats[name] = {"n": int, "mean": float, "best": float, "last": float}
        self.features_stats: Dict[str, Dict[str, Any]] = {}
        self.targets_stats: Dict[str, Dict[str, Any]] = {}
        # models_stats[name] = {"n": int, "mean": float, "best": float, "last": float}
        self.models_stats: Dict[str, Dict[str, Any]] = {}
        # model_target_stats[target][model] = stats dict
        self.model_target_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                data = json.load(open(self.path, "r", encoding="utf-8"))
                self.features = data.get("features", {})
                self.targets = data.get("targets", {})
                self.disabled_features = data.get("disabled_features", [])
                self.disabled_targets = data.get("disabled_targets", [])
                self.k_weights = data.get("k_weights", {})
                self.features_stats = data.get("features_stats", {})
                self.targets_stats = data.get("targets_stats", {})
                self.models = data.get("models", {})
                self.disabled_models = data.get("disabled_models", [])
                self.models_stats = data.get("models_stats", {})
                self.model_target_stats = data.get("model_target_stats", {})
            except Exception:
                pass
        # Enforce requested defaults: disable some models by default
        if "rf_baseline" not in self.disabled_models:
            self.disabled_models.append("rf_baseline")
        if "logistic_baseline" not in self.disabled_models:
            self.disabled_models.append("logistic_baseline")
        # Disable new 3-class target by default until multiclass models are fully wired
        if "updown_margin_H12_k075" not in self.disabled_targets:
            self.disabled_targets.append("updown_margin_H12_k075")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(
            {
                "features": self.features,
                "targets": self.targets,
                "disabled_features": self.disabled_features,
                "disabled_targets": self.disabled_targets,
                "k_weights": self.k_weights,
                "features_stats": self.features_stats,
                "targets_stats": self.targets_stats,
                "models": self.models,
                "disabled_models": self.disabled_models,
                "models_stats": self.models_stats,
                "model_target_stats": self.model_target_stats,
            },
            open(self.path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

    def weight_for_feature(self, name: str) -> float:
        if name in self.disabled_features:
            return 0.0
        return float(self.features.get(name, 1.0))

    def weight_for_target(self, name: str) -> float:
        if name in self.disabled_targets:
            return 0.0
        return float(self.targets.get(name, 1.0))

    def weight_for_model(self, name: str) -> float:
        if name in self.disabled_models:
            return 0.0
        return float(self.models.get(name, 1.0))


def _weighted_sample(items: List[str], weights: List[float], k: int, rng: random.Random) -> List[str]:
    # sample without replacement using weights; simple sequential algorithm
    pool = list(zip(items, weights))
    chosen: List[str] = []
    k = min(k, len(items))
    for _ in range(k):
        total = sum(w for _, w in pool)
        if total <= 0:
            # fallback to uniform among remaining
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = 0
            for i, (_name, w) in enumerate(pool):
                acc += w
                if r <= acc:
                    idx = i
                    break
        name, _ = pool.pop(idx)
        chosen.append(name)
    return chosen


def build_selection(
    df: pd.DataFrame,
    rng: Optional[random.Random] = None,
    weight_store: Optional[WeightStore] = None,
    nfeat_min: int = 8,
    nfeat_max: int = 20,
    model_id: Optional[str] = None,
    eval_cfg: str = DEFAULT_EVAL,
) -> Dict[str, Any]:
    rng = rng or random
    ws = weight_store or WeightStore()

    # Targets: exclude disabled and zero-weight (treated as frozen/off)
    all_targets = [
        t for t in registry.targets.keys()
        if t not in ws.disabled_targets and ws.weight_for_target(t) > 0.0
    ]
    if not all_targets:
        raise RuntimeError("No available targets to select from")
    t_weights = [ws.weight_for_target(t) for t in all_targets]
    # fallback to uniform if all zeros
    if sum(t_weights) == 0:
        t_weights = [1.0] * len(all_targets)
    target_id = _weighted_sample(all_targets, t_weights, 1, rng)[0]

    # Constraints from target
    n_eff = 0
    is_bin = False
    task_type = 'clf'
    pos = 0
    try:
        y = registry.targets[target_id](df)
        y_nonan = y.dropna()
        n_eff = int(len(y_nonan))
        # Determine task type (classification vs regression)
        # Binary classification if only {0,1}
        if y_nonan.dtype.kind in 'biu':
            vals = set(map(int, set(y_nonan.unique())))
        else:
            vals = set(y_nonan.unique())
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            is_bin = True
            task_type = 'clf'
            try:
                pos = int(y_nonan.sum())
            except Exception:
                pos = 0
        else:
            # Heuristic: floats or many unique values -> regression
            if y_nonan.dtype.kind in 'f' or y_nonan.nunique(dropna=True) > 10:
                task_type = 'reg'
            else:
                task_type = 'clf'
    except Exception:
        pass

    # Features universe: exclude disabled and zero-weight
    all_features = [
        f for f in registry.features.keys()
        if f not in ws.disabled_features and ws.weight_for_feature(f) > 0.0
    ]
    if not all_features:
        raise RuntimeError("No available features to select from")

    # Dynamic K bounds
    K_max_dyn = nfeat_max
    if n_eff > 0:
        K_max_dyn = min(K_max_dyn, max(1, n_eff // 8))
    if is_bin and pos > 0:
        K_max_dyn = min(K_max_dyn, max(1, pos // 8))
    K_max_dyn = min(K_max_dyn, len(all_features))
    K_min_dyn = min(nfeat_min, K_max_dyn)
    K_candidates = list(range(K_min_dyn, K_max_dyn + 1))
    # K weights
    k_weights = [float(ws.k_weights.get(f"k_{k}", 1.0)) for k in K_candidates]
    if sum(k_weights) == 0:
        k_weights = [1.0] * len(K_candidates)
    k_pick = int(_weighted_sample([str(k) for k in K_candidates], k_weights, 1, rng)[0])

    # Feature selection
    # Base weights
    f_weights = [ws.weight_for_feature(f) for f in all_features]
    # Auto-prune: if a feature has enough usage but low mean affinity, downweight it
    effective_weights = []
    for f, w in zip(all_features, f_weights):
        st = ws.features_stats.get(f) or {}
        n = int(st.get("n", 0) or 0)
        mean = st.get("mean", None)
        if n >= 80 and isinstance(mean, (int, float)) and float(mean) < 0.05:
            w = float(w) * 0.1
        effective_weights.append(float(w))
    if sum(effective_weights) == 0:
        effective_weights = [1.0] * len(all_features)
    feature_ids = _weighted_sample(all_features, effective_weights, k_pick, rng)

    # Model selection
    if model_id is None:
        all_models = [m for m in registry.models.keys() if m not in ws.disabled_models]
        # Filter by task type to avoid mixing regression/clf
        if task_type == 'reg':
            candidates = [m for m in all_models if m.endswith('_regressor') or m in ('hgb_regressor','tf_mlp_regressor')]
        else:
            candidates = [m for m in all_models if not (m.endswith('_regressor') or m in ('hgb_regressor','tf_mlp_regressor'))]
        if candidates:
            all_models = candidates
        if not all_models:
            # Fallback to DEFAULT_MODEL if registry empty or all disabled
            model_id = DEFAULT_MODEL if DEFAULT_MODEL in registry.models else next(iter(registry.models))
        else:
            base_weights = [ws.weight_for_model(m) for m in all_models]
            if sum(base_weights) == 0:
                # Prefer TensorFlow MLP variants, then HGB, then Logistic by default
                pref_map = {
                    "tf_mlp_baseline": 3.0,
                    "tf_mlp_bn_wd": 2.5,
                    "hgb_baseline": 1.5,
                    "logistic_baseline": 1.0,
                }
                base_weights = [float(pref_map.get(m, 1.0)) for m in all_models]
            # Bias by target-specific model stats when available
            pair_stats = ws.model_target_stats.get(target_id, {}) if 'target_id' in locals() else {}
            m_weights: List[float] = []
            for m, bw in zip(all_models, base_weights):
                st = pair_stats.get(m) or {}
                mean = float(st.get("mean", 0.0) or 0.0)
                bias = 0.5 + max(0.0, min(1.0, mean))  # [0.5,1.5]
                m_weights.append(float(bw) * bias)
            model_id = _weighted_sample(all_models, m_weights, 1, rng)[0]

    return {
        "target": target_id,
        "features": feature_ids,
        "model": model_id,
        "eval_cfg": eval_cfg,
    }


def run_loop(
    df_name: str,
    split_id: str,
    n_runs: int = 10,
    seed: int = 0,
    db_path: str = "runs.db",
    weight_store: Optional[WeightStore] = None,
    rng: Optional[random.Random] = None,
) -> List[Optional[str]]:
    rng = rng or random.Random(seed)
    ws = weight_store or WeightStore()
    # Load dataframe once for computing dynamic K constraints
    df = storage.load_dataframe(df_name)
    run_ids: List[Optional[str]] = []
    for i in range(n_runs):
        # Sync disabled lists and weights with on-disk store to honor external edits (e.g., GUI)
        try:
            _ws_disk = WeightStore(ws.path)
            ws.disabled_features = list(_ws_disk.disabled_features or [])
            ws.disabled_targets = list(_ws_disk.disabled_targets or [])
            ws.disabled_models = list(_ws_disk.disabled_models or [])
            # Also refresh weights so weight=0 takes effect immediately
            ws.features = dict(_ws_disk.features or {})
            ws.targets = dict(_ws_disk.targets or {})
            ws.models = dict(_ws_disk.models or {})
            ws.k_weights = dict(_ws_disk.k_weights or {})
        except Exception:
            pass
        sel = build_selection(df=df, rng=rng, weight_store=ws)
        run_id = run_experiment_with_selection(
            df_name=df_name,
            split_id=split_id,
            selection=sel,
            seed=seed + i,
            db_path=db_path,
            random_state=rng,
        )
        run_ids.append(run_id)
        if run_id is not None:
            try:
                _update_weights_from_run(run_id, db_path=db_path, ws=ws)
                # Persist weights/stats after each run to enable live reporting
                try:
                    ws.save()
                except Exception:
                    pass
            except Exception:
                # non-fatal; continue the loop
                pass
    # persist weights after loop
    try:
        ws.save()
    except Exception:
        pass
    return run_ids


def _score_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    """Map metrics to a [0,1] score.

    Conservative policy: only use AUC (or auc_median) from a sufficiently large test set.
    Avoids inflating scores from degenerate accuracy when the test set has a single class.
    """
    if not metrics:
        return None
    # Require a minimally sized test set when available
    try:
        n_test = int(metrics.get("n_test") or 0)
        if n_test and n_test < 30:
            return None
    except Exception:
        pass
    # Prefer AUC or auc_median for classification
    for key in ("auc", "auc_median"):
        if key in metrics and metrics[key] is not None:
            try:
                auc = float(metrics[key])
                if not (auc == auc):  # NaN check
                    continue
                return max(0.0, min(1.0, 2.0 * (auc - 0.5)))
            except Exception:
                continue
    # Regression/unified scoring: prefer skill, else r2, else rank correlation
    for key in ("skill",):
        if key in metrics and metrics[key] is not None:
            try:
                sk = float(metrics[key])
                if not (sk == sk):
                    continue
                return max(0.0, min(1.0, sk))
            except Exception:
                continue
    for key in ("r2", "r2_score"):
        if key in metrics and metrics[key] is not None:
            try:
                r2 = float(metrics[key])
                if not (r2 == r2):
                    continue
                return max(0.0, min(1.0, r2))
            except Exception:
                continue
    for key in ("spearman", "corr_spearman", "pearson", "corr_pearson"):
        if key in metrics and metrics[key] is not None:
            try:
                c = float(metrics[key])
                if not (c == c):
                    continue
                return max(0.0, min(1.0, (c + 1.0) / 2.0))
            except Exception:
                continue
    # No reliable score available
    return None


def _score_from_affinity(aff: Dict[str, Any]) -> float:
    """Compute a feature score in [0,1] from affinity metrics with robust defaults."""
    import math

    def _safe(v):
        try:
            return float(v)
        except Exception:
            return math.nan

    pear = abs(_safe(aff.get("pearson")))
    spear = abs(_safe(aff.get("spearman")))
    mi = _safe(aff.get("mi"))
    dcor = _safe(aff.get("dcor"))
    n_eff = _safe(aff.get("n_eff"))

    # Robust aggregation avoiding NaNs
    corr_candidates = [pear, spear]
    corr_vals = [c for c in corr_candidates if not math.isnan(c)]
    s_corr = max(corr_vals) if corr_vals else 0.0
    # MI is unbounded; map typical small values to [0,1]
    s_mi = 0.0 if math.isnan(mi) else max(0.0, min(1.0, mi / 0.2))
    s_dcor = 0.0 if math.isnan(dcor) else max(0.0, min(1.0, dcor))

    s = 0.5 * s_corr + 0.25 * s_mi + 0.25 * s_dcor
    # penalize low effective sample size
    if math.isnan(n_eff):
        w_n = 1.0
    else:
        w_n = max(0.3, min(1.0, n_eff / 200.0))
    s_final = s * w_n
    if math.isnan(s_final):
        s_final = 0.0
    return max(0.0, min(1.0, s_final))


def _ema(old: float, new: float, alpha: float) -> float:
    return (1 - alpha) * old + alpha * new


def _update_weights_from_run(run_id: str, db_path: str, ws: WeightStore, alpha_feat: float = 0.2, alpha_tgt: float = 0.2) -> None:
    con = history.connect(db_path)
    try:
        row = history.get_run(con, run_id)
        if not row or row.get("status") != "SUCCESS":
            return
        selection = row.get("selection") or {}
        metrics = row.get("metrics") or {}
        artifacts = row.get("artifacts") or {}

        # Update target weight (skip if disabled or zero-weight/frozen)
        tgt = selection.get("target")
        s_t = _score_from_metrics(metrics)
        if tgt and s_t is not None and tgt not in ws.disabled_targets and ws.weight_for_target(tgt) > 0.0:
            old = ws.targets.get(tgt, 1.0)
            ws.targets[tgt] = _ema(old, s_t, alpha_tgt)
            # Update target stats
            st = ws.targets_stats.get(tgt) or {"n": 0, "mean": 0.0, "best": 0.0, "last": 0.0}
            n = int(st.get("n", 0)) + 1
            mean = float(st.get("mean", 0.0)) + (float(s_t) - float(st.get("mean", 0.0))) / n
            best = max(float(st.get("best", 0.0)), float(s_t))
            ws.targets_stats[tgt] = {"n": n, "mean": mean, "best": best, "last": float(s_t)}

    # Update feature weights from affinities
        aff_all = artifacts.get("affinity") or {}
        for fid in selection.get("features", []):
            # Skip disabled or zero-weight/frozen features
            if fid in ws.disabled_features or ws.weight_for_feature(fid) <= 0.0:
                continue
            aff = aff_all.get(fid, {}) if isinstance(aff_all, dict) else {}
            s_f = _score_from_affinity(aff)
            old = ws.features.get(fid, 1.0)
            ws.features[fid] = _ema(old, s_f, alpha_feat)
            # Update feature stats
            sf = ws.features_stats.get(fid) or {"n": 0, "mean": 0.0, "best": 0.0, "last": 0.0}
            n = int(sf.get("n", 0)) + 1
            mean = float(sf.get("mean", 0.0)) + (float(s_f) - float(sf.get("mean", 0.0))) / n
            best = max(float(sf.get("best", 0.0)), float(s_f))
            ws.features_stats[fid] = {"n": n, "mean": mean, "best": best, "last": float(s_f)}
    # Update model weights/stats based on overall run score (skip disabled/zero-weight)
        mdl = selection.get("model")
        if mdl and s_t is not None and mdl not in ws.disabled_models and ws.weight_for_model(mdl) > 0.0:
            oldm = ws.models.get(mdl, 1.0)
            ws.models[mdl] = _ema(oldm, s_t, alpha_tgt)
            sm = ws.models_stats.get(mdl) or {"n": 0, "mean": 0.0, "best": 0.0, "last": 0.0}
            nm = int(sm.get("n", 0)) + 1
            meanm = float(sm.get("mean", 0.0)) + (float(s_t) - float(sm.get("mean", 0.0))) / nm
            bestm = max(float(sm.get("best", 0.0)), float(s_t))
            ws.models_stats[mdl] = {"n": nm, "mean": meanm, "best": bestm, "last": float(s_t)}
    finally:
        try:
            con.close()
        except Exception:
            pass
    # Update K weights based on run length
    k_used = len(selection.get("features", []))
    if k_used > 0:
        key = f"k_{k_used}"
        s = _score_from_metrics(metrics)
        if s is not None:
            oldk = ws.k_weights.get(key, 1.0)
            ws.k_weights[key] = _ema(oldk, s, alpha_tgt)
    # Penalize features that failed during computation (if recorded)
    try:
        failed_list = artifacts.get("failed_features") or []
        for item in failed_list:
            fid = item.get("id") if isinstance(item, dict) else None
            if not fid:
                continue
            oldw = float(ws.features.get(fid, 1.0))
            ws.features[fid] = max(0.0, oldw * 0.5)
    except Exception:
        pass

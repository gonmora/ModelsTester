#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute extended metrics for a past run selection (model × target × features).

Defaults:
  - df_name: 'BTCUSDT_5m_20230831_20250830'
  - db_path:  'runs_24M.db'

Given a run_id, this script loads the selection from the DB, reconstructs
the dataset, re-trains a model compatible with the selection (favoring the
TensorFlow MLP variants used in the project), obtains test probabilities,
and reports extended metrics:
  - ROC AUC (+ approx 95% CI), PR AUC (AP), AP lift, accuracy
  - Precision/Recall/F1/MCC at threshold=0.5 and at the F1-optimal threshold
  - Confusion matrix at both thresholds
  - Precision@k% (k in {1,5,10})
  - Brier score, KS statistic

Notes:
  - This refits a fresh model; results may differ slightly from the stored run.
  - If TensorFlow is unavailable, falls back to a logistic baseline.
"""
from __future__ import annotations

import sys
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlite3

# Ensure project root is importable as 'src'
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project imports
from src.storage import load_dataframe, load_feature, feature_exists, load_target, target_exists
from src.registry import registry
from src.basic_components import _time_stratified_split, _ensure_utf8_locale


DEFAULT_DF_NAME = 'BTCUSDT_5m_20230831_20250830'
DEFAULT_DB_PATH = 'runs_24M.db'


@dataclass
class Selection:
    target: str
    features: List[str]
    model: str
    eval_cfg: str = 'default'


def _load_selection(db_path: str, run_id: str) -> Selection:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT selection_json FROM runs WHERE run_id=? LIMIT 1", (run_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError(f"run_id '{run_id}' no encontrado en {db_path}")
        sel = json.loads(row[0])
        return Selection(
            target=str(sel['target']),
            features=[str(x) for x in sel['features']],
            model=str(sel['model']),
            eval_cfg=str(sel.get('eval_cfg', 'default')),
        )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _prepare_xy(df_name: str, sel: Selection, gap: int = 0) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = load_dataframe(df_name)
    # Target
    if target_exists(df_name, sel.target):
        y = load_target(df_name, sel.target)
    else:
        y = registry.targets[sel.target](df)
    # Features
    feats: Dict[str, pd.Series] = {}
    for fid in sel.features:
        if feature_exists(df_name, fid):
            x = load_feature(df_name, fid)
        else:
            x = registry.features[fid](df)
        if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
            x = x.iloc[:, 0]
        feats[fid] = x
    # Align and clean similar to model implementations
    X = pd.concat(feats, axis=1)
    combined = pd.concat([y, X], axis=1)
    mask = combined.iloc[:, 0].notna()
    y_clean = combined.loc[mask, combined.columns[0]]
    X_clean = combined.loc[mask, combined.columns[1:]]
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    try:
        X_clean = X_clean.infer_objects(copy=False)
    except Exception:
        pass
    X_clean = X_clean.dropna(axis=1, how='all')
    if y_clean.nunique() < 2 or len(X_clean) < 100:
        raise RuntimeError("Datos insuficientes o una sola clase para entrenar/evaluar")
    split = _time_stratified_split(y_clean, min_train=100, min_test=100)
    # Enforce an optional temporal GAP between train and test to avoid any subtle
    # overlap through rolling windows or cached states across the boundary.
    split_gap = min(len(X_clean), max(0, int(split) + int(gap)))
    X_train_df = X_clean.iloc[:split]
    X_test_df = X_clean.iloc[split_gap:]
    y_train = y_clean.iloc[:split]
    y_test = y_clean.iloc[split_gap:]
    return X_train_df, y_train, X_test_df, y_test


def _fit_predict_tf(model_id: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fit a small TF MLP consistent with project models and return test probabilities.

    Falls back to logistic baseline if TF is not available.
    """
    _ensure_utf8_locale()
    try:
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # Impute + scale
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(X_train))
        X_te = scl.transform(imp.transform(X_test))
        y_tr = y_train.to_numpy(dtype=np.float32)

        n_in = X_tr.shape[1]
        inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32)

        # Architectures (simplified, close to repo code)
        if model_id == 'tf_mlp_bn_wd':
            reg = tf.keras.regularizers.l2(1e-4)
            x = tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=reg)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=reg)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(32, use_bias=False, kernel_regularizer=reg)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.1)(x)
        else:  # tf_mlp_baseline / tf_mlp_multiclass (binary target path)
            x = tf.keras.layers.Dense(128, use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(64, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(32, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        ]

        # Optional class weighting
        cw = None
        try:
            pos = float(y_tr.mean())
            if 0.0 < pos < 1.0:
                w1 = 0.5 / max(1e-6, pos)
                w0 = 0.5 / max(1e-6, 1.0 - pos)
                cw = {0: w0, 1: w1}
        except Exception:
            cw = None

        model.fit(
            X_tr, y_tr,
            epochs=50,
            batch_size=1024,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
            class_weight=cw,
        )
        probs = model.predict(X_te, batch_size=8192, verbose=0).reshape(-1)
        return probs
    except Exception:
        # Fallback: logistic baseline (sklearn)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)),
        ])
        pipe.fit(X_train, y_train)
        return pipe.predict_proba(X_test)[:, 1]


def _precision_at_k(y_true: np.ndarray, probs: np.ndarray, frac: float) -> float:
    k = max(1, int(len(probs) * frac))
    idx = np.argsort(-probs)[:k]
    return float(y_true[idx].mean())


def _confusion(y_true: np.ndarray, preds: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else float('nan')


def _ks_stat(y_true: np.ndarray, probs: np.ndarray) -> float:
    pos = np.sort(probs[y_true == 1])
    neg = np.sort(probs[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Empirical CDFs on combined thresholds
    thresh = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, thresh, side='right') / len(pos)
    cdf_neg = np.searchsorted(neg, thresh, side='right') / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _auc_ci_hanley_mcneil(auc: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n_pos <= 0 or n_neg <= 0:
        return float('nan'), float('nan')
    Q1 = auc / (2 - auc)
    Q2 = (2 * auc * auc) / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (Q1 - auc * auc) + (n_neg - 1) * (Q2 - auc * auc)) / (n_pos * n_neg)
    se = math.sqrt(max(0.0, var))
    z = 1.959963984540054  # ~95%
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def main() -> None:
    ap = argparse.ArgumentParser(description='Extended metrics for a past run selection')
    ap.add_argument('--run-id', required=True, help='Run ID to analyze')
    ap.add_argument('--df-name', default=DEFAULT_DF_NAME)
    ap.add_argument('--db-path', default=DEFAULT_DB_PATH)
    ap.add_argument('--out-json', default=None, help='Optional path to save metrics JSON')
    ap.add_argument('--gap', type=int, default=0, help='Bars gap between train and test to avoid boundary leakage')
    args = ap.parse_args()

    sel = _load_selection(args.db_path, args.run_id)
    X_train, y_train, X_test, y_test = _prepare_xy(args.df_name, sel, gap=args.gap)

    # Decide task type by model name or y cardinality
    is_reg = sel.model.endswith('_regressor') or (pd.api.types.is_float_dtype(y_train) and y_train.nunique() > 10)

    if not is_reg:
        # Classification path
        if sel.model.startswith('tf_mlp'):
            probs = _fit_predict_tf(sel.model, X_train, y_train, X_test)
        else:
            # Minimal fallback: logistic baseline
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)),
            ])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]

        y_true = y_test.to_numpy().astype(int)
    # Core metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
        auc = float(roc_auc_score(y_true, probs))
        ap = float(average_precision_score(y_true, probs))
        acc50 = float(accuracy_score(y_true, (probs >= 0.5).astype(int)))
    except Exception as e:
        raise RuntimeError(f"Error computing base metrics: {e}")

    pos_rate = float(y_true.mean()) if len(y_true) else float('nan')
    ap_lift = (ap / pos_rate) if pos_rate and pos_rate > 0 else float('nan')
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    lo, hi = _auc_ci_hanley_mcneil(auc, n_pos, n_neg)

    # Threshold optimization for F1
    thresholds = np.unique(np.clip(probs, 0, 1))
    if thresholds.size > 512:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 512))
    f1_vals = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        try:
            f1_vals.append((float(f1_score(y_true, preds)), float(t)))
        except Exception:
            f1_vals.append((float('nan'), float(t)))
    best_f1, best_t = max(f1_vals, key=lambda x: (x[0], x[1])) if f1_vals else (float('nan'), 0.5)

    # Metrics at 0.5 and at best_t
    def thr_metrics(t: float) -> Dict[str, Any]:
        preds = (probs >= t).astype(int)
        tp, fp, tn, fn = _confusion(y_true, preds)
        try:
            prec = float(precision_score(y_true, preds))
            rec = float(recall_score(y_true, preds))
            f1 = float(f1_score(y_true, preds))
        except Exception:
            prec = rec = f1 = float('nan')
        try:
            acc = float(accuracy_score(y_true, preds))
        except Exception:
            acc = float('nan')
        mcc = _mcc(tp, fp, tn, fn)
        return {
            'threshold': float(t),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc,
            'mcc': mcc,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        }

    thr_05 = thr_metrics(0.5)
    thr_best = thr_metrics(float(best_t))

    # Ranking metrics
    p_at_1 = _precision_at_k(y_true, probs, 0.01)
    p_at_5 = _precision_at_k(y_true, probs, 0.05)
    p_at_10 = _precision_at_k(y_true, probs, 0.10)

    # Calibration / separability
    try:
        from sklearn.metrics import brier_score_loss
        brier = float(brier_score_loss(y_true, np.clip(probs, 0, 1)))
    except Exception:
        brier = float('nan')
    ks = _ks_stat(y_true, probs)

    out = {
        'run_id': args.run_id,
        'df_name': args.df_name,
        'target': sel.target,
        'model': sel.model,
        'n_test': int(len(y_true)),
        'pos_rate_test': pos_rate,
        'auc': auc,
        'auc_ci95': [lo, hi],
        'ap': ap,
        'ap_lift': ap_lift,
        'accuracy_at_0.5': acc50,
        'thresholds': {
            't0.5': thr_05,
            'best_f1': thr_best,
        },
        'ranking': {
            'precision_at_1pct': p_at_1,
            'precision_at_5pct': p_at_5,
            'precision_at_10pct': p_at_10,
        },
        'calibration': {
            'brier': brier,
            'ks': ks,
        }
    }

    # Pretty print
    def _fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return 'nan'
            return f"{x:.4f}"
        return str(x)

    print("\n=== Extended metrics ===")
    print(f"run_id: {out['run_id']}")
    print(f"target: {out['target']} | model: {out['model']} | n_test: {out['n_test']}")
    print(f"AUC: {_fmt(out['auc'])} (95% CI ~ [{_fmt(out['auc_ci95'][0])}, {_fmt(out['auc_ci95'][1])}])")
    print(f"AP: {_fmt(out['ap'])} | baseline: {_fmt(out['pos_rate_test'])} | lift: {_fmt(out['ap_lift'])}x")
    print(f"Accuracy@0.5: {_fmt(out['accuracy_at_0.5'])}")
    print(f"Threshold 0.5 -> P={_fmt(out['thresholds']['t0.5']['precision'])}, R={_fmt(out['thresholds']['t0.5']['recall'])}, F1={_fmt(out['thresholds']['t0.5']['f1'])}, MCC={_fmt(out['thresholds']['t0.5']['mcc'])}, Confusion: TP={out['thresholds']['t0.5']['tp']}, FP={out['thresholds']['t0.5']['fp']}, TN={out['thresholds']['t0.5']['tn']}, FN={out['thresholds']['t0.5']['fn']}")
    print(f"Best F1 @ t={_fmt(out['thresholds']['best_f1']['threshold'])} -> P={_fmt(out['thresholds']['best_f1']['precision'])}, R={_fmt(out['thresholds']['best_f1']['recall'])}, F1={_fmt(out['thresholds']['best_f1']['f1'])}, MCC={_fmt(out['thresholds']['best_f1']['mcc'])}, Confusion: TP={out['thresholds']['best_f1']['tp']}, FP={out['thresholds']['best_f1']['fp']}, TN={out['thresholds']['best_f1']['tn']}, FN={out['thresholds']['best_f1']['fn']}")
    print(f"P@1%={_fmt(out['ranking']['precision_at_1pct'])} | P@5%={_fmt(out['ranking']['precision_at_5pct'])} | P@10%={_fmt(out['ranking']['precision_at_10pct'])}")
    print(f"Brier={_fmt(out['calibration']['brier'])} | KS={_fmt(out['calibration']['ks'])}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON -> {args.out_json}")

    # Regression branch
    if is_reg:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        # Choose model consistent with selection
        model_name = sel.model
        if model_name == 'tf_mlp_regressor':
            # Reuse TF build but simpler via scikit if TF missing
            try:
                import tensorflow as tf  # noqa: F401
                # Reuse TF path by calling a small helper here would duplicate code; use sklearn fallback for simplicity
            except Exception:
                pass
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        Xtr = pipe.fit_transform(X_train)
        Xte = pipe.transform(X_test)
        # Prefer HGBRegressor
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, min_samples_leaf=100, early_stopping=True, validation_fraction=0.1, random_state=0)
        except Exception:
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0)
        reg.fit(Xtr, y_train)
        pred = reg.predict(Xte)

        y_true_reg = y_test.to_numpy(dtype=float)
        rmse = float(np.sqrt(mean_squared_error(y_true_reg, pred)))
        mae = float(mean_absolute_error(y_true_reg, pred))
        try:
            r2 = float(r2_score(y_true_reg, pred))
        except Exception:
            r2 = float('nan')
        try:
            rho, _ = spearmanr(y_true_reg, pred)
            sp = float(rho)
        except Exception:
            sp = float('nan')
        baseline = float(np.mean(y_train)) if len(y_train) else 0.0
        rmse_base = float(np.sqrt(mean_squared_error(y_true_reg, np.full_like(y_true_reg, baseline)))) if len(y_true_reg) else float('nan')
        skill = float(1.0 - (rmse / rmse_base)) if (rmse_base and np.isfinite(rmse_base) and rmse_base > 0) else float('nan')

        print("\n=== Extended regression metrics ===")
        print(f"RMSE: {rmse:.6f} | baseline_rmse: {rmse_base:.6f} | skill: {skill:.4f}")
        print(f"MAE: {mae:.6f} | R2: {r2:.4f} | Spearman: {sp:.4f}")


if __name__ == '__main__':
    main()

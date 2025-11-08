# -*- coding: utf-8 -*-
"""
Lightweight reporting utilities for ranking features and targets based on
stored weights and usage statistics.

Usage (in a notebook):
    from src.reporting import top_features, top_targets, features_table, targets_table
    top_features(20)  # DataFrame with best 20 features by composite rank
    top_targets()     # DataFrame with targets ranked by composite score
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import math
import pandas as pd

from .runner.engine import WeightStore
from .registry import registry
from .core import history
import sqlite3, json, datetime as dt
from contextlib import closing
try:
    from IPython.display import clear_output, display
except Exception:
    clear_output = None
    display = None


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _composite_feature_score(row: pd.Series) -> float:
    """Composite score in [0,1] using stats and current weight.

    - Emphasize mean and best affinity when available.
    - Downweight unseen (n == 0).
    """
    n = int(row.get("n") or 0)
    if n <= 0:
        return 0.0
    mean = _safe_float(row.get("mean"), 0.0)
    best = _safe_float(row.get("best"), 0.0)
    w = _safe_float(row.get("weight"), 1.0)
    # Confidence grows with n, saturating around n ~ 10
    conf = 1.0 - math.exp(-n / 10.0)
    # Blend of mean, best and current weight
    base = 0.6 * mean + 0.3 * best + 0.1 * w
    return max(0.0, min(1.0, conf * base))


def _composite_target_score(row: pd.Series) -> float:
    n = int(row.get("n") or 0)
    if n <= 0:
        return 0.0
    mean = _safe_float(row.get("mean"), 0.0)
    best = _safe_float(row.get("best"), 0.0)
    w = _safe_float(row.get("weight"), 1.0)
    conf = 1.0 - math.exp(-n / 10.0)
    base = 0.7 * mean + 0.2 * best + 0.1 * w
    return max(0.0, min(1.0, conf * base))


def _composite_model_score(row: pd.Series) -> float:
    n = int(row.get("n") or 0)
    if n <= 0:
        return 0.0
    mean = _safe_float(row.get("mean"), 0.0)
    best = _safe_float(row.get("best"), 0.0)
    w = _safe_float(row.get("weight"), 1.0)
    conf = 1.0 - math.exp(-n / 10.0)
    base = 0.7 * mean + 0.2 * best + 0.1 * w
    return max(0.0, min(1.0, conf * base))


def features_table(ws: Optional[WeightStore] = None, filter_to_registry: bool = False) -> pd.DataFrame:
    """Return a DataFrame with feature weights and stats, plus a rank column."""
    ws = ws or WeightStore()
    # Include registry features so unseen ones (n==0) appear with default weight
    from .registry import registry as _reg
    names = set(ws.features.keys()) | set(ws.features_stats.keys()) | set(_reg.features.keys())
    if filter_to_registry:
        reg_names = set(registry.features.keys())
        names &= reg_names
    rows = []
    for name in names:
        st: Dict[str, Any] = ws.features_stats.get(name, {})
        rows.append({
            "feature": name,
            "weight": _safe_float(ws.features.get(name, 1.0), 1.0),
            "n": int(st.get("n", 0) or 0),
            "mean": _safe_float(st.get("mean"), math.nan),
            "best": _safe_float(st.get("best"), math.nan),
            "last": _safe_float(st.get("last"), math.nan),
        })
    cols = ["feature", "weight", "n", "mean", "best", "last", "rank"]
    df = pd.DataFrame(rows, columns=cols[:-1])  # rank computed below
    if len(df) == 0:
        # Return empty frame with expected columns
        return pd.DataFrame(columns=cols)
    df["rank"] = df.apply(_composite_feature_score, axis=1)
    df = df.sort_values(["rank", "n", "best"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def targets_table(ws: Optional[WeightStore] = None, filter_to_registry: bool = False) -> pd.DataFrame:
    """Return a DataFrame with target weights and stats, plus a rank column."""
    ws = ws or WeightStore()
    names = set(ws.targets.keys()) | set(ws.targets_stats.keys())
    if filter_to_registry:
        reg_names = set(registry.targets.keys())
        names &= reg_names
    rows = []
    for name in names:
        st: Dict[str, Any] = ws.targets_stats.get(name, {})
        rows.append({
            "target": name,
            "weight": _safe_float(ws.targets.get(name, 1.0), 1.0),
            "n": int(st.get("n", 0) or 0),
            "mean": _safe_float(st.get("mean"), math.nan),
            "best": _safe_float(st.get("best"), math.nan),
            "last": _safe_float(st.get("last"), math.nan),
        })
    cols = ["target", "weight", "n", "mean", "best", "last", "rank"]
    df = pd.DataFrame(rows, columns=cols[:-1])
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    df["rank"] = df.apply(_composite_target_score, axis=1)
    df = df.sort_values(["rank", "n", "best"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def models_table(ws: Optional[WeightStore] = None) -> pd.DataFrame:
    """Return a DataFrame with model weights and stats, plus a rank column."""
    ws = ws or WeightStore()
    # Include registry models so unseen ones appear with n=0
    from .registry import registry
    names = set(ws.models.keys()) | set(ws.models_stats.keys()) | set(registry.models.keys())
    rows = []
    for name in names:
        st: Dict[str, Any] = ws.models_stats.get(name, {})
        rows.append({
            "model": name,
            "weight": _safe_float(ws.models.get(name, 1.0), 1.0),
            "n": int(st.get("n", 0) or 0),
            "mean": _safe_float(st.get("mean"), math.nan),
            "best": _safe_float(st.get("best"), math.nan),
            "last": _safe_float(st.get("last"), math.nan),
        })
    cols = ["model", "weight", "n", "mean", "best", "last", "rank"]
    df = pd.DataFrame(rows, columns=cols[:-1])
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    df["rank"] = df.apply(_composite_model_score, axis=1)
    df = df.sort_values(["rank", "n", "best"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def top_features(k: int = 20, include_unseen: bool = False, ws: Optional[WeightStore] = None) -> pd.DataFrame:
    """Return top-k features by composite rank.

    - include_unseen=False filters out features with n == 0.
    """
    df = features_table(ws)
    if not include_unseen:
        df = df[df["n"] > 0]
    return df.head(k)


def top_targets(k: Optional[int] = None, include_unseen: bool = False, ws: Optional[WeightStore] = None) -> pd.DataFrame:
    df = targets_table(ws)
    if not include_unseen:
        df = df[df["n"] > 0]
    return df if k is None else df.head(k)


def top_models(k: Optional[int] = None, include_unseen: bool = False, ws: Optional[WeightStore] = None) -> pd.DataFrame:
    df = models_table(ws)
    if not include_unseen:
        df = df[df["n"] > 0]
    return df if k is None else df.head(k)


def _fmt_row_feature(row: pd.Series) -> str:
    return (
        f"{row['feature']}: n={int(row['n'])}, mean={row['mean']:.3f}, "
        f"best={row['best']:.3f}, w={row['weight']:.3f}, rank={row['rank']:.3f}"
    )


def _fmt_row_target(row: pd.Series) -> str:
    mean = row['mean']
    best = row['best']
    mean_s = "nan" if (isinstance(mean, float) and math.isnan(mean)) else f"{mean:.3f}"
    best_s = "nan" if (isinstance(best, float) and math.isnan(best)) else f"{best:.3f}"
    return (
        f"{row['target']}: n={int(row['n'])}, mean={mean_s}, "
        f"best={best_s}, w={row['weight']:.3f}, rank={row['rank']:.3f}"
    )


def _fmt_row_model(row: pd.Series) -> str:
    mean = row['mean']
    best = row['best']
    mean_s = "nan" if (isinstance(mean, float) and math.isnan(mean)) else f"{mean:.3f}"
    best_s = "nan" if (isinstance(best, float) and math.isnan(best)) else f"{best:.3f}"
    return (
        f"{row['model']}: n={int(row['n'])}, mean={mean_s}, best={best_s}, "
        f"w={row['weight']:.3f}, rank={row['rank']:.3f}"
    )


def print_summary(top_k: int = 10, db_path: str = 'runs.db') -> None:
    """Print a concise, human-friendly summary of current rankings."""
    ws = WeightStore()
    tf = top_features(top_k, include_unseen=False, ws=ws)
    tt = top_targets(None, include_unseen=False, ws=ws)
    tm = top_models(None, include_unseen=False, ws=ws)
    lines = []
    now = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    lines.append(f"Snapshot: {now}")
    lines.append("")
    if len(tf) > 0:
        lines.append(f"Top {min(top_k, len(tf))} features:")
        for _, r in tf.iterrows():
            lines.append("  - " + _fmt_row_feature(r))
    else:
        # Fallback: show unseen by weight
        tf_unseen = top_features(top_k, include_unseen=True, ws=ws)
        lines.append("Top features (unseen by weight):")
        for _, r in tf_unseen.iterrows():
            lines.append("  - " + _fmt_row_feature(r))
    lines.append("")
    if len(tt) > 0:
        lines.append("Targets:")
        for _, r in tt.iterrows():
            lines.append("  - " + _fmt_row_target(r))
    else:
        lines.append("Targets: (no usage yet)")
    # Models (list omitted to reduce duplication; see timing/quality below)
    # Historical target quality (use the same DB the monitor is reading)
    th = targets_historical(db_path=db_path, min_runs=3)
    if len(th) > 0:
        lines.append("")
        lines.append("Target quality (history):")
        for _, row in th.iterrows():
            auc_med = row['auc_median']
            auc_iqr = row['auc_iqr']
            pr = row['pos_rate_test_median']
            ap_med = row.get('ap_median')
            auc_s = f"{auc_med:.3f}" if pd.notna(auc_med) else "nan"
            iqr_s = f"{auc_iqr:.3f}" if pd.notna(auc_iqr) else "nan"
            pr_s = f"{pr:.3f}" if pd.notna(pr) else "nan"
            ap_s = f"{ap_med:.3f}" if (ap_med is not None and pd.notna(ap_med)) else "nan"
            lines.append(
                f"  - {row['target']}: runs={int(row['runs'])}, auc_med={auc_s} (IQR {iqr_s}), AP_med={ap_s}, pos_rate_test_med={pr_s}"
            )
    # Historical model timing/quality
    mh = models_historical(db_path=db_path, min_runs=3)
    if len(mh) > 0:
        lines.append("")
        lines.append("Model quality & timing (history):")
        for _, row in mh.iterrows():
            auc_med = row.get('auc_median')
            ap_med = row.get('ap_median')
            ft_med = row.get('fit_time_median_sec')
            pt_med = row.get('predict_time_median_sec')
            auc_s = f"{auc_med:.3f}" if pd.notna(auc_med) else "nan"
            ap_s = f"{ap_med:.3f}" if pd.notna(ap_med) else "nan"
            ft_s = f"{ft_med:.2f}s" if pd.notna(ft_med) else "nan"
            pt_s = f"{pt_med:.2f}s" if pd.notna(pt_med) else "nan"
            lines.append(
                f"  - {row['model']}: runs={int(row['runs'])}, auc_med={auc_s}, AP_med={ap_s}, fit_med={ft_s}, predict_med={pt_s}"
            )

    pm = pairs_historical(db_path=db_path, min_runs=3)
    if len(pm) > 0:
        lines.append("")
        lines.append("Target x Model (history):")
        head = pm.head(6)
        for _, row in head.iterrows():
            auc_s = f"{row['auc_median']:.3f}" if pd.notna(row.get('auc_median')) else "nan"
            ap_s = f"{row['ap_median']:.3f}" if pd.notna(row.get('ap_median')) else "nan"
            lines.append(
                f"  - {row['target']} × {row['model']}: runs={int(row['runs'])}, auc_med={auc_s}, AP_med={ap_s}"
            )
    print("\n".join(lines))


def runs_overview(db_path: str = 'runs.db', last: int = 5) -> Dict[str, Any]:
    """Return a dict with run counts by status and the last N runs brief info."""
    out: Dict[str, Any] = {"counts": {}, "last": []}
    try:
        with closing(sqlite3.connect(db_path)) as con:
            cur = con.execute("SELECT status, COUNT(*) FROM runs GROUP BY status")
            out["counts"] = {k: int(v) for k, v in cur.fetchall()}
            cur = con.execute(
                "SELECT run_id, status, selection_json, metrics_json FROM runs ORDER BY started_at DESC LIMIT ?",
                (int(last),),
            )
            for rid, st, sj, mj in cur.fetchall():
                try:
                    metrics = json.loads(mj) if mj else None
                except Exception:
                    metrics = None
                try:
                    sel = json.loads(sj) if sj else None
                except Exception:
                    sel = None
                out["last"].append({
                    "run_id": rid,
                    "status": st,
                    "target": (sel or {}).get('target'),
                    "model": (sel or {}).get('model'),
                    "metrics": metrics,
                })
    except Exception:
        pass
    return out


def targets_historical(db_path: str = 'runs.db', min_runs: int = 5) -> pd.DataFrame:
    """Aggregate metrics over historical runs per target.

    Returns a DataFrame with per-target aggregates: count, auc_median, auc_q025, auc_q975,
    auc_iqr, acc_median, ap_median, pos_rate_test_median.
    """
    rows = []
    try:
        with closing(sqlite3.connect(db_path)) as con:
            cur = con.execute(
                "SELECT selection_json, metrics_json FROM runs WHERE status='SUCCESS' AND metrics_json IS NOT NULL"
            )
            data = cur.fetchall()
        # Collect per target
        by_tgt: Dict[str, Dict[str, list]] = {}
        for sj, mj in data:
            try:
                sel = json.loads(sj) if sj else None
                met = json.loads(mj) if mj else None
            except Exception:
                continue
            if not sel or not met:
                continue
            tgt = sel.get('target')
            if not tgt:
                continue
            d = by_tgt.setdefault(tgt, {"auc": [], "acc": [], "prt": [], "ap": []})
            auc = met.get('auc')
            acc = met.get('accuracy')
            prt = met.get('pos_rate_test')
            if auc is not None:
                try:
                    v = float(auc)
                    import math as _math
                    if v == v and _math.isfinite(v):  # drop NaN/inf
                        d["auc"].append(v)
                except Exception:
                    pass
            if acc is not None:
                try:
                    d["acc"].append(float(acc))
                except Exception:
                    pass
            if prt is not None:
                try:
                    d["prt"].append(float(prt))
                except Exception:
                    pass
            apv = met.get('ap')
            if apv is not None:
                try:
                    d["ap"].append(float(apv))
                except Exception:
                    pass
        # Summarize
        for tgt, vals in by_tgt.items():
            n = len(vals.get("auc", []))
            if n < min_runs:
                continue
            s_auc = pd.Series(vals["auc"]) if vals.get("auc") else pd.Series(dtype=float)
            s_acc = pd.Series(vals["acc"]) if vals.get("acc") else pd.Series(dtype=float)
            s_prt = pd.Series(vals["prt"]) if vals.get("prt") else pd.Series(dtype=float)
            s_ap = pd.Series(vals["ap"]) if vals.get("ap") else pd.Series(dtype=float)
            def q(s, p):
                try:
                    return float(s.quantile(p))
                except Exception:
                    return float('nan')
            rows.append({
                "target": tgt,
                "runs": int(n),
                "auc_median": q(s_auc, 0.5),
                "auc_q025": q(s_auc, 0.025),
                "auc_q975": q(s_auc, 0.975),
                "auc_iqr": q(s_auc, 0.75) - q(s_auc, 0.25),
                "acc_median": q(s_acc, 0.5),
                "ap_median": q(s_ap, 0.5),
                "pos_rate_test_median": q(s_prt, 0.5),
            })
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "target", "runs", "auc_median", "auc_q025", "auc_q975", "auc_iqr", "acc_median", "ap_median", "pos_rate_test_median"
        ])
    return df.sort_values(["auc_median", "runs"], ascending=[False, False]).reset_index(drop=True)




def pairs_historical(db_path: str = 'runs.db', min_runs: int = 5) -> pd.DataFrame:
    """Aggregate metrics over historical runs per (target, model) pair.

    Returns: target, model, runs, auc_median, ap_median.
    """
    rows = []
    try:
        import sqlite3, json
        from contextlib import closing
        with closing(sqlite3.connect(db_path)) as con:
            cur = con.execute(
                "SELECT selection_json, metrics_json FROM runs WHERE status='SUCCESS' AND metrics_json IS NOT NULL"
            )
            data = cur.fetchall()
        by_pair = {}
        for sj, mj in data:
            try:
                sel = json.loads(sj) if sj else None
                met = json.loads(mj) if mj else None
            except Exception:
                continue
            if not sel or not met:
                continue
            tgt = sel.get('target')
            mdl = sel.get('model')
            if not tgt or not mdl:
                continue
            key = (tgt, mdl)
            d = by_pair.setdefault(key, {"auc": [], "ap": []})
            auc = met.get('auc')
            apv = met.get('ap')
            if auc is not None:
                try:
                    d["auc"].append(float(auc))
                except Exception:
                    pass
            if apv is not None:
                try:
                    d["ap"].append(float(apv))
                except Exception:
                    pass
        for (tgt, mdl), vals in by_pair.items():
            n = len(vals.get("auc", []))
            if n < min_runs:
                continue
            import pandas as pd
            s_auc = pd.Series(vals["auc"]) if vals.get("auc") else pd.Series(dtype=float)
            s_ap = pd.Series(vals["ap"]) if vals.get("ap") else pd.Series(dtype=float)
            rows.append({
                "target": tgt,
                "model": mdl,
                "runs": int(n),
                "auc_median": float(s_auc.median()) if len(s_auc) else float('nan'),
                "ap_median": float(s_ap.median()) if len(s_ap) else float('nan'),
            })
    except Exception:
        pass
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["target", "model", "runs", "auc_median", "ap_median"])
    return df.sort_values(["auc_median", "runs"], ascending=[False, False]).reset_index(drop=True)
def models_historical(db_path: str = 'runs.db', min_runs: int = 5) -> pd.DataFrame:
    """Aggregate metrics over historical runs per model.

    Returns a DataFrame with per-model aggregates: runs, auc_median, ap_median,
    fit_time_median, predict_time_median.
    """
    rows = []
    try:
        with closing(sqlite3.connect(db_path)) as con:
            cur = con.execute(
                "SELECT selection_json, metrics_json FROM runs WHERE status='SUCCESS' AND metrics_json IS NOT NULL"
            )
            data = cur.fetchall()
        by_model: Dict[str, Dict[str, list]] = {}
        for sj, mj in data:
            try:
                sel = json.loads(sj) if sj else None
                met = json.loads(mj) if mj else None
            except Exception:
                continue
            if not sel or not met:
                continue
            mdl = sel.get('model')
            if not mdl:
                continue
            d = by_model.setdefault(mdl, {"auc": [], "ap": [], "fit": [], "pred": []})
            auc = met.get('auc')
            apv = met.get('ap')
            fit = met.get('fit_time_sec')
            pred = met.get('predict_time_sec')
            for key, val in (("auc", auc), ("ap", apv), ("fit", fit), ("pred", pred)):
                if val is None:
                    continue
                try:
                    d[key].append(float(val))
                except Exception:
                    pass
        for mdl, vals in by_model.items():
            n = len(vals.get("auc", []))
            if n < min_runs:
                continue
            s_auc = pd.Series(vals["auc"]) if vals.get("auc") else pd.Series(dtype=float)
            s_ap = pd.Series(vals["ap"]) if vals.get("ap") else pd.Series(dtype=float)
            s_fit = pd.Series(vals["fit"]) if vals.get("fit") else pd.Series(dtype=float)
            s_pred = pd.Series(vals["pred"]) if vals.get("pred") else pd.Series(dtype=float)
            def q(s, p):
                try:
                    return float(s.quantile(p))
                except Exception:
                    return float('nan')
            rows.append({
                "model": mdl,
                "runs": int(n),
                "auc_median": q(s_auc, 0.5),
                "ap_median": q(s_ap, 0.5),
                "fit_time_median_sec": q(s_fit, 0.5),
                "predict_time_median_sec": q(s_pred, 0.5),
            })
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "model", "runs", "auc_median", "ap_median", "fit_time_median_sec", "predict_time_median_sec"
        ])
    return df.sort_values(["auc_median", "runs"], ascending=[False, False]).reset_index(drop=True)


def live_monitor(interval: float = 2.0, top_k: int = 10, db_path: str = 'runs.db', refresh_tables: bool = True):
    """Live refresh rankings and recent runs in a notebook.

    Use Ctrl-C to stop. Requires IPython.display for smooth updates.
    """
    def _in_notebook() -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if not ip:
                return False
            return ip.has_trait('kernel') or ip.__class__.__name__.startswith('ZMQ')
        except Exception:
            return False

    use_notebook_mode = clear_output is not None and _in_notebook()

    if use_notebook_mode:
        import time
        try:
            while True:
                clear_output(wait=True)
                ws = WeightStore()
                # Header with data sources
                print(f"Monitoring db='{db_path}', weights='{ws.path}'")
                print_summary(top_k, db_path=db_path)
                ro = runs_overview(db_path=db_path, last=5)
                print("\nRun counts:", ro.get('counts'))
                for r in ro.get('last', []):
                    print(f" - {r['run_id']} {r['status']} model={r.get('model')} {r['metrics']}")
                if refresh_tables:
                    print("\nTables:")
                    tf_df = top_features(top_k, include_unseen=False, ws=ws)
                    if len(tf_df) == 0:
                        tf_df = top_features(top_k, include_unseen=True, ws=ws)
                    display(tf_df)
                    tt_df = top_targets(ws=ws)
                    if len(tt_df) == 0:
                        tt_df = targets_table(ws)
                    display(tt_df)
                    print("\nModel timing (history):")
                    display(models_historical(db_path))
                time.sleep(interval)
        except KeyboardInterrupt:
            return
    else:
        import sys, time
        try:
            while True:
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                ws = WeightStore()
                print(f"Monitoring db='{db_path}', weights='{ws.path}'")
                print_summary(top_k, db_path=db_path)
                ro = runs_overview(db_path=db_path, last=5)
                print("\nRun counts:", ro.get('counts'))
                for r in ro.get('last', []):
                    print(f" - {r['run_id']} {r['status']} model={r.get('model')} {r['metrics']}")
                if refresh_tables:
                    print("\nTables:")
                    ws = WeightStore()
                    tf_df = top_features(top_k, include_unseen=False, ws=ws)
                    if len(tf_df) == 0:
                        tf_df = top_features(top_k, include_unseen=True, ws=ws)
                    with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 120):
                        print(tf_df.to_string(index=False))
                        tt_df = top_targets(ws=ws)
                        if len(tt_df) == 0:
                            tt_df = targets_table(ws)
                        print(tt_df.to_string(index=False))
                        print("\nModel timing (history):")
                        mh_df = models_historical(db_path)
                        print(mh_df.to_string(index=False))
                time.sleep(interval)
        except KeyboardInterrupt:
            return


def cleanup_weights(remove_unregistered: bool = True, path: str = None) -> None:
    """Utility: remove stray stats/weights not in registry and rewrite weights.json.

    Call this once if you see placeholder entries like 'feat_x'/'tgt_y'.
    """
    ws = WeightStore(path or WeightStore().path)
    if remove_unregistered:
        f_ok = set(registry.features.keys())
        t_ok = set(registry.targets.keys())
        ws.features = {k: v for k, v in ws.features.items() if k in f_ok}
        ws.features_stats = {k: v for k, v in ws.features_stats.items() if k in f_ok}
        ws.targets = {k: v for k, v in ws.targets.items() if k in t_ok}
        ws.targets_stats = {k: v for k, v in ws.targets_stats.items() if k in t_ok}
    ws.save()

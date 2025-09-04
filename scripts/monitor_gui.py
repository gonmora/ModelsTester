#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple GUI monitor (tkinter) for ModelsTester.

Features:
- Start/Stop refresh loop
- Select DB path and WEIGHTS_JSON (environment) from inputs
- Set refresh interval (sec) and Top-K for rankings
- Tabs:
  * Top Features
  * Targets
  * Target Quality (history)
  * Model Quality & Timing (history)
  * Target x Model (history)
  * Recent Runs

Note: Uses only standard library (tkinter/ttk). Tables are ttk.Treeview.
"""
from __future__ import annotations

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import pandas as pd
import numpy as np
import datetime as dt
import re
import subprocess
import threading

# Ensure project root is on sys.path so `import src.*` works when running as a script
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# Preferred directories for pickers on this workstation
DEFAULT_DB_DIR = "/home/usuario/Proyectos/ModelsTester/"
DEFAULT_WEIGHTS_DIR = "/home/usuario/Proyectos/ModelsTester/data/"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Avoid heavy TA auto-registration warnings/import in the monitor process
os.environ.setdefault("DISABLE_TA_AUTOREG", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Ensure storage uses project-level data directory when running from scripts/
os.environ.setdefault("DATA_DIR", os.path.join(ROOT, "data"))

try:
    from src.reporting import (
        WeightStore,
        top_features,
        top_targets,
        targets_table,
        models_historical,
        targets_historical,
        runs_overview,
        features_table,
    )
    # pairs_historical removed from this GUI (Target x Model tab removed)
except Exception as e:
    # Fall back to showing a message and exiting gracefully
    try:
        messagebox.showerror("Import error", f"Failed to import reporting: {e}\n\nTip: Run from project root or set PYTHONPATH to include it.")
    except Exception:
        print(f"Failed to import reporting: {e}\nTip: Run from project root or set PYTHONPATH.")
    raise


class MonitorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ModelsTester Monitor")
        self.geometry("1000x700")

        self.running = False
        self.refresh_ms = 2000
        self.top_k = 10
        self.feature_filter_var = tk.StringVar(value="")
        self.target_filter_var = tk.StringVar(value="")

        # Load last-used preferences before building controls so defaults honor them
        try:
            self._prefs_cache = self._load_prefs()
        except Exception:
            self._prefs_cache = {}

        self._build_controls()
        # Apply last-used preferences (if any)
        try:
            self._apply_prefs(getattr(self, "_prefs_cache", {}) or {})
        except Exception:
            pass
        # Save preferences on close
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass
        self._build_tabs()

    # ---- Preferences (persist last-used inputs) ----
    def _prefs_path(self) -> str:
        try:
            data_dir = os.path.join(ROOT, "data")
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, "monitor_gui_prefs.json")
        except Exception:
            # Fallback to current directory
            return os.path.join(os.getcwd(), "monitor_gui_prefs.json")

    def _load_prefs(self) -> dict:
        path = self._prefs_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return {}
        return {}

    def _collect_prefs(self) -> dict:
        try:
            geo = self.geometry()
        except Exception:
            geo = None
        return {
            "db_path": self.db_var.get() if hasattr(self, "db_var") else None,
            "weights_json": self.weights_var.get() if hasattr(self, "weights_var") else None,
            "refresh_secs": self.refresh_var.get() if hasattr(self, "refresh_var") else None,
            "top_k": self.topk_var.get() if hasattr(self, "topk_var") else None,
            "feature_filter": self.feature_filter_var.get() if hasattr(self, "feature_filter_var") else None,
            "target_filter": self.target_filter_var.get() if hasattr(self, "target_filter_var") else None,
            "normalize": bool(self.normalize_var.get()) if hasattr(self, "normalize_var") else False,
            "geometry": geo,
        }

    def _apply_prefs(self, prefs: dict) -> None:
        if not isinstance(prefs, dict):
            return
        try:
            val = prefs.get("db_path")
            if val:
                self.db_var.set(val)
        except Exception:
            pass
        try:
            val = prefs.get("weights_json")
            if val:
                self.weights_var.set(val)
        except Exception:
            pass
        try:
            val = prefs.get("refresh_secs")
            if val:
                self.refresh_var.set(str(val))
        except Exception:
            pass
        try:
            val = prefs.get("top_k")
            if val:
                self.topk_var.set(str(val))
        except Exception:
            pass
        try:
            val = prefs.get("feature_filter")
            if val is not None:
                self.feature_filter_var.set(str(val))
        except Exception:
            pass
        try:
            val = prefs.get("target_filter")
            if val is not None:
                self.target_filter_var.set(str(val))
        except Exception:
            pass
        try:
            if "normalize" in prefs:
                self.normalize_var.set(bool(prefs.get("normalize")))
        except Exception:
            pass
        try:
            geo = prefs.get("geometry")
            if isinstance(geo, str) and geo:
                self.geometry(geo)
        except Exception:
            pass

    def _save_prefs(self) -> None:
        path = self._prefs_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_prefs(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _on_close(self) -> None:
        try:
            self._save_prefs()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _build_controls(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="DB path:").grid(row=0, column=0, sticky=tk.W, padx=4)
        _db_default = (getattr(self, "_prefs_cache", {}) or {}).get("db_path") or "runs_BIG.db"
        self.db_var = tk.StringVar(value=_db_default)
        ttk.Entry(frm, textvariable=self.db_var, width=40).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frm, text="...", command=self._pick_db).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(frm, text="WEIGHTS_JSON:").grid(row=0, column=3, sticky=tk.W, padx=(12,4))
        _w_default = (getattr(self, "_prefs_cache", {}) or {}).get("weights_json") or os.environ.get("WEIGHTS_JSON", "data/weights_BIG.json")
        self.weights_var = tk.StringVar(value=_w_default)
        ttk.Entry(frm, textvariable=self.weights_var, width=40).grid(row=0, column=4, sticky=tk.W)
        ttk.Button(frm, text="...", command=self._pick_weights).grid(row=0, column=5, sticky=tk.W)

        ttk.Label(frm, text="Refresh (s):").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6,0))
        _r_default = str((getattr(self, "_prefs_cache", {}) or {}).get("refresh_secs") or "2.0")
        self.refresh_var = tk.StringVar(value=_r_default)
        ttk.Entry(frm, textvariable=self.refresh_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=(6,0))

        ttk.Label(frm, text="Top-K:").grid(row=1, column=3, sticky=tk.W, padx=(12,4), pady=(6,0))
        _k_default = str((getattr(self, "_prefs_cache", {}) or {}).get("top_k") or "10")
        self.topk_var = tk.StringVar(value=_k_default)
        ttk.Entry(frm, textvariable=self.topk_var, width=6).grid(row=1, column=4, sticky=tk.W, pady=(6,0))

        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(frm, textvariable=self.status_var).grid(row=1, column=5, sticky=tk.W, padx=(12,0), pady=(6,0))

        # Third row: action buttons and filters
        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=(4,0), pady=(8,0))
        ttk.Button(btns, text="Refresh", command=self.refresh_now).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Export Excel", command=self.export_active_to_excel).pack(side=tk.LEFT, padx=4)

        # Feature filter textbox (applies to Top Features tab)
        ttk.Label(frm, text="Filtro features:").grid(row=2, column=3, sticky=tk.W, padx=(12,4), pady=(8,0))
        _f_default = str((getattr(self, "_prefs_cache", {}) or {}).get("feature_filter") or "")
        self.feature_filter_var.set(_f_default)
        ttk.Entry(frm, textvariable=self.feature_filter_var, width=20).grid(row=2, column=4, sticky=tk.W, pady=(8,0))

        # Target filter textbox (applies to Targets tab)
        ttk.Label(frm, text="Filtro targets:").grid(row=3, column=3, sticky=tk.W, padx=(12,4), pady=(8,0))
        _t_default = str((getattr(self, "_prefs_cache", {}) or {}).get("target_filter") or "")
        self.target_filter_var.set(_t_default)
        ttk.Entry(frm, textvariable=self.target_filter_var, width=20).grid(row=3, column=4, sticky=tk.W, pady=(8,0))

        # Normalize toggle (affects Top (Unified) and Top by Target)
        _norm_default = bool((getattr(self, "_prefs_cache", {}) or {}).get("normalize", False))
        self.normalize_var = tk.BooleanVar(value=_norm_default)
        ttk.Checkbutton(frm, text="Normalize (sAUC·sAP)", variable=self.normalize_var).grid(row=2, column=5, sticky=tk.W, padx=(12,0), pady=(8,0))

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        # Map of frame widget id -> tree to find active grid
        self._tree_by_frame_id: dict[str, ttk.Treeview] = {}

        # New: Top Unified (mixed tasks, unified score)
        self.tab_unified = self._add_table_tab(
            "Top (Unified)",
            cols=[
                "run_id","target","model","score","task","auc","ap","skill","r2","spearman","n_test","features"
            ],
        )

        # New: per-target leaderboard
        self.tab_top_by_target = self._add_table_tab(
            "Top by Target",
            cols=[
                "target",
                "run_id",
                "model",
                "accuracy",
                "ap",
                "auc",
                "ap_lift",
                "pos_rate_test",
                "n_test",
                "n_features",
                "performance",
                "score",
                "features",
            ],
        )

        # Tables (features/targets retain 'weight'; models unchanged)
        self.tab_features = self._add_table_tab("Top Features", cols=["feature","weight","n","mean","best","last","rank"]) 
        self.tab_targets = self._add_table_tab("Targets", cols=["target","weight","n","mean","best","last","rank"]) 
        self.tab_tq = self._add_table_tab("Target Quality", cols=["target","runs","auc_median","auc_q025","auc_q975","auc_iqr","acc_median","ap_median","pos_rate_test_median"]) 
        self.tab_mq = self._add_table_tab("Model Quality & Timing", cols=["model","runs","auc_median","ap_median","fit_time_median_sec","predict_time_median_sec"]) 
        self.tab_runs = self._add_table_tab("Recent Runs", cols=["run_id","status","model","metrics"]) 

        # Bind double-click to toggle weight 0/1 for features and targets only
        self.tab_features.bind("<Double-1>", lambda e: self._on_toggle_weight(e, tree=self.tab_features, kind="feature", name_col="feature"))
        self.tab_targets.bind("<Double-1>", lambda e: self._on_toggle_weight(e, tree=self.tab_targets, kind="target", name_col="target"))
        # Bind double-click on Top (Unified) to open validation window
        self.tab_unified.bind("<Double-1>", lambda e: self._on_open_validation(e, tree=self.tab_unified))

    def _add_table_tab(self, title: str, cols: list[str]) -> ttk.Treeview:
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor=tk.W, width=140)
        # Keep mapping for export by active tab
        try:
            self._tree_by_frame_id[str(frame)] = tree
        except Exception:
            pass
        return tree

    def export_active_to_excel(self) -> None:
        """Export the currently active grid to an Excel file.

        Falls back to CSV if Excel export fails due to engine availability.
        """
        try:
            active = self.nb.select()
            tree = self._tree_by_frame_id.get(str(active))
            if tree is None:
                messagebox.showwarning("Export", "No active table to export.")
                return
            cols = list(tree["columns"]) or []
            rows = []
            for item in tree.get_children():
                vals = tree.item(item, "values") or []
                # Ensure list size matches columns
                row = {cols[i]: vals[i] if i < len(vals) else None for i in range(len(cols))}
                rows.append(row)
            if not rows:
                messagebox.showinfo("Export", "No rows to export in this tab.")
                return
            df = pd.DataFrame(rows, columns=cols)
            # Build automatic filename: {tab}_{YYYYMMDD}_{HHMMSS}.xlsx in DB directory or CWD
            try:
                tab_text = self.nb.tab(active, 'text') or 'export'
            except Exception:
                tab_text = 'export'
            safe_tab = re.sub(r'[^A-Za-z0-9_-]+', '_', str(tab_text)).strip('_') or 'export'
            ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            fname_xlsx = f"{safe_tab}_{ts}.xlsx"
            dbp = self.db_var.get().strip()
            base_dir = os.path.dirname(dbp) if (dbp and os.path.isdir(os.path.dirname(dbp))) else os.getcwd()
            path = os.path.join(base_dir, fname_xlsx)
            # Try Excel first
            try:
                df.to_excel(path, index=False)
                messagebox.showinfo("Export", f"Exported to: {path}")
                self._open_file(path)
                return
            except Exception:
                # Fallback to CSV
                try:
                    base, _ = os.path.splitext(path)
                    path = base + '.csv'
                    df.to_csv(path, index=False)
                    messagebox.showinfo("Export", f"Excel engine unavailable. Exported CSV to: {path}")
                    self._open_file(path)
                    return
                except Exception as e:
                    messagebox.showerror("Export error", f"Failed to export: {e}")
        except Exception as e:
            try:
                messagebox.showerror("Export error", str(e))
            except Exception:
                print(f"Export error: {e}")

    def _open_file(self, path: str) -> None:
        """Try to open a file with the default system handler."""
        try:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', path])
            elif os.name == 'nt':
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception:
            pass

    def _pick_db(self) -> None:
        init = self._initial_dir(self.db_var.get().strip(), DEFAULT_DB_DIR)
        path = filedialog.askopenfilename(
            title="Select runs DB",
            initialdir=init,
            filetypes=[("SQLite DB","*.db"), ("All","*.*")],
        )
        if path:
            self.db_var.set(path)

    def _pick_weights(self) -> None:
        init = self._initial_dir(self.weights_var.get().strip(), DEFAULT_WEIGHTS_DIR)
        path = filedialog.askopenfilename(
            title="Select weights JSON",
            initialdir=init,
            filetypes=[("JSON","*.json"), ("All","*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _initial_dir(self, current_path: str, fallback: str) -> str:
        try:
            if current_path and os.path.isdir(current_path):
                return current_path
            d = os.path.dirname(current_path)
            if d and os.path.isdir(d):
                return d
        except Exception:
            pass
        return fallback if os.path.isdir(fallback) else os.getcwd()

    def refresh_now(self) -> None:
        """Fetch and render all tables once (manual refresh)."""
        # Sync auxiliary params
        try:
            self.top_k = int(self.topk_var.get())
        except Exception:
            self.top_k = 10
        # Apply weights env for reporting
        os.environ["WEIGHTS_JSON"] = self.weights_var.get().strip()
        self.status_var.set("Refreshing…")
        try:
            self._refresh_core()
            self.status_var.set("Idle")
        except Exception:
            self.status_var.set("Idle (error)")

    def _refresh(self) -> None:
        """Backwards-compatible: no-op loop scheduler removed. Use refresh_now()."""
        # Intentionally do not reschedule to avoid online loop
        self._refresh_core()

    def _refresh_core(self) -> None:
        db = self.db_var.get().strip()
        # (Top Predictors tab removed)
        # Top unified (score in [0,1])
        try:
            tu_df = self._fetch_top_unified(db_path=db, limit=None, normalize=self.normalize_var.get())
            # Apply target filter to Top (Unified)
            try:
                tfilt = self.target_filter_var.get().strip()
            except Exception:
                tfilt = ""
            if tfilt and len(tu_df) > 0 and 'target' in tu_df.columns:
                try:
                    mask = tu_df.get('target').astype(str).str.contains(tfilt, case=False, na=False)
                    tu_df = tu_df[mask]
                except Exception:
                    pass
            # Apply Top-K after filtering
            try:
                k = int(self.top_k)
                if k and k > 0:
                    tu_df = tu_df.head(k)
            except Exception:
                pass
            self._fill_tree(self.tab_unified, tu_df)
        except Exception as e:
            self._set_tree_error(self.tab_unified, f"Error: {e}")
        # Top Features & Targets (from weights)
        try:
            # Use the selected weights file explicitly to avoid stale module-level defaults
            ws = WeightStore(path=self.weights_var.get().strip())
            # Fetch ALL features ranked by composite score; do not cap here
            try:
                tf_df = features_table(ws=ws)
                # Default behavior: hide unseen (n == 0); keep full list (no cap)
                tf_df = tf_df[tf_df.get('n', 0) > 0]
                if len(tf_df) == 0:
                    # Fallback: include unseen by weight when there are no stats yet
                    tf_df = features_table(ws=ws)
            except Exception:
                tf_df = pd.DataFrame()
            # Apply feature name filter if provided
            try:
                filt = self.feature_filter_var.get().strip()
            except Exception:
                filt = ""
            if filt:
                try:
                    if len(tf_df) > 0 and 'feature' in tf_df.columns:
                        mask = tf_df.get('feature').astype(str).str.contains(filt, case=False, na=False)
                        tf_df = tf_df[mask]
                except Exception:
                    pass

            self._fill_tree(self.tab_features, tf_df)

            # Top targets with fallback to include_unseen or full table if empty
            try:
                tt_df = top_targets(None, include_unseen=False, ws=ws)
                if len(tt_df) == 0:
                    # show all targets even if unseen
                    from src.reporting import targets_table
                    tt_df = targets_table(ws)
            except Exception:
                tt_df = pd.DataFrame()
            # Apply target name filter if provided
            try:
                tfilt = self.target_filter_var.get().strip()
            except Exception:
                tfilt = ""
            if tfilt:
                try:
                    if len(tt_df) > 0 and 'target' in tt_df.columns:
                        mask = tt_df.get('target').astype(str).str.contains(tfilt, case=False, na=False)
                        tt_df = tt_df[mask]
                except Exception:
                    pass
            self._fill_tree(self.tab_targets, tt_df)
        except Exception as e:
            self._set_tree_error(self.tab_features, f"Error: {e}")
            self._set_tree_error(self.tab_targets, f"Error: {e}")
        # Top by Target (grouped)
        try:
            # Best single model per target, including regression via unified score
            tbt_df = self._fetch_top_by_target(db_path=db, per=1, normalize=self.normalize_var.get())
            self._fill_tree(self.tab_top_by_target, tbt_df)
        except Exception as e:
            self._set_tree_error(self.tab_top_by_target, f"Error: {e}")
        # Historical tables (DB)
        try:
            self._fill_tree(self.tab_tq, targets_historical(db_path=db, min_runs=3))
        except Exception as e:
            self._set_tree_error(self.tab_tq, f"Error: {e}")
        try:
            self._fill_tree(self.tab_mq, models_historical(db_path=db, min_runs=3))
        except Exception as e:
            self._set_tree_error(self.tab_mq, f"Error: {e}")
        # Removed Target x Model tab
        # Recent runs
        try:
            ro = runs_overview(db_path=db, last=10)
            rows = []
            for r in ro.get('last', []):
                rows.append({
                    'run_id': r.get('run_id'),
                    'status': r.get('status'),
                    'model': r.get('model'),
                    'metrics': r.get('metrics'),
                })
            df = pd.DataFrame(rows, columns=['run_id','status','model','metrics'])
            self._fill_tree(self.tab_runs, df)
        except Exception as e:
            self._set_tree_error(self.tab_runs, f"Error: {e}")
        # No reschedule (manual refresh only)

    # Removed: _fetch_top_predictors (Top Predictors tab was removed)

    def _fetch_top_unified(self, db_path: str, limit: int | None = None, normalize: bool = False) -> pd.DataFrame:
        """Top runs by unified score.

        Score mapping:
        - Classification: score = clip(2*(auc-0.5), 0, 1)
        - Regression:     score = clip(skill, 0, 1), fallback to r2, then (spearman+1)/2
        """
        import sqlite3, json, math
        rows = []
        con = sqlite3.connect(db_path)
        try:
            cur = con.execute(
                "SELECT run_id, selection_json, metrics_json FROM runs WHERE status='SUCCESS' AND metrics_json IS NOT NULL"
            )
            for rid, sj, mj in cur.fetchall():
                try:
                    sel = json.loads(sj) if sj else None
                    met = json.loads(mj) if mj else None
                except Exception:
                    continue
                if not sel or not met:
                    continue
                auc = met.get('auc'); ap = met.get('ap')
                prt = met.get('pos_rate_test')
                skill = met.get('skill'); r2 = met.get('r2') or met.get('r2_score'); sp = met.get('spearman') or met.get('corr_spearman')
                n_test = met.get('n_test')
                # Compute unified score
                score = None
                try:
                    if isinstance(auc, (int, float)) and not (isinstance(auc, float) and math.isnan(auc)):
                        # Classification
                        if normalize:
                            # sAUC in [0,1]
                            sAUC = max(0.0, min(1.0, (float(auc) - 0.5) / 0.5))
                            # sAP relative to baseline in [0,1]
                            try:
                                pr = float(prt) if prt is not None else None
                                apv = float(ap) if ap is not None else None
                            except Exception:
                                pr = None; apv = None
                            if pr is not None and apv is not None and pr < 1.0:
                                try:
                                    sAP = (apv - pr) / (1.0 - pr)
                                except Exception:
                                    sAP = float('nan')
                            else:
                                sAP = float('nan')
                            score = float(sAUC) * float(sAP) if (sAUC == sAUC and sAP == sAP) else max(0.0, min(1.0, 2.0 * (float(auc) - 0.5)))
                        else:
                            score = max(0.0, min(1.0, 2.0 * (float(auc) - 0.5)))
                        task = 'clf'
                    elif isinstance(skill, (int, float)) and not (isinstance(skill, float) and math.isnan(skill)):
                        score = max(0.0, min(1.0, float(skill)))
                        task = 'reg'
                    elif isinstance(r2, (int, float)) and not (isinstance(r2, float) and math.isnan(r2)):
                        score = max(0.0, min(1.0, float(r2)))
                        task = 'reg'
                    elif isinstance(sp, (int, float)) and not (isinstance(sp, float) and math.isnan(sp)):
                        score = max(0.0, min(1.0, (float(sp) + 1.0) / 2.0))
                        task = 'reg'
                    else:
                        continue
                except Exception:
                    continue
                try:
                    n_test = int(n_test) if n_test is not None else None
                except Exception:
                    n_test = None
                rows.append({
                    'run_id': rid,
                    'target': sel.get('target'),
                    'model': sel.get('model'),
                    'score': score,
                    'task': task,
                    'auc': auc,
                    'ap': ap,
                    'skill': skill,
                    'r2': r2,
                    'spearman': sp,
                    'n_test': n_test,
                    'features': sel.get('features'),
                })
        finally:
            try:
                con.close()
            except Exception:
                pass
        df = pd.DataFrame(rows, columns=['run_id','target','model','score','task','auc','ap','skill','r2','spearman','n_test','features'])
        if len(df) == 0:
            return df
        df = df.sort_values(['score','n_test'], ascending=[False, False])
        if isinstance(limit, int):
            return df.head(limit)
        return df

    def _fetch_top_by_target(self, db_path: str, per: int = 1, normalize: bool = False) -> pd.DataFrame:
        """Return per-target top runs (single best by default) including regression via unified score.

        Uses _fetch_top_unified() to rank both classification and regression by a unified score.
        Maps columns to the existing Top-by-Target table; fields not applicable remain None/NaN.
        """
        base = self._fetch_top_unified(db_path=db_path, limit=50_000, normalize=normalize)
        if len(base) == 0:
            return base
        # Sort within target by unified score then n_test
        base2 = base.sort_values(['target','score','n_test'], ascending=[True, False, False])
        out = []
        for tgt, g in base2.groupby('target', sort=False):
            out.append(g.head(max(1, per)))
        dfu = pd.concat(out, axis=0, ignore_index=True)
        # Prepare output columns expected by the tab
        rows = []
        for _, r in dfu.iterrows():
            feats = r.get('features')
            rows.append({
                'target': r.get('target'),
                'run_id': r.get('run_id'),
                'model': r.get('model'),
                'accuracy': None,  # not available from unified summary
                'ap': r.get('ap'),
                'auc': r.get('auc'),
                'ap_lift': None,
                'pos_rate_test': None,
                'n_test': r.get('n_test'),
                'n_features': (len(feats) if isinstance(feats, list) else None),
                'performance': r.get('score'),
                'score': r.get('score'),
                'features': feats,
            })
        df = pd.DataFrame(rows, columns=['target','run_id','model','accuracy','ap','auc','ap_lift','pos_rate_test','n_test','n_features','performance','score','features'])
        # Order blocks by performance
        best_by_target = df.groupby('target')['performance'].max().sort_values(ascending=False)
        df['__rank'] = df['target'].map({t:i for i,t in enumerate(best_by_target.index)})
        df = df.sort_values(['__rank','performance'], ascending=[True, False]).drop(columns='__rank')
        return df

    def _set_tree_error(self, tree: ttk.Treeview, msg: str) -> None:
        self._fill_tree(tree, pd.DataFrame([{tree["columns"][0]: msg}]))

    def _fill_tree(self, tree: ttk.Treeview, df: pd.DataFrame) -> None:
        # Clear existing
        for item in tree.get_children():
            tree.delete(item)
        cols = list(tree["columns"])
        if df is None or len(getattr(df, 'columns', [])) == 0:
            return
        # Align columns
        # If df has fewer columns, fill; if more, subset
        data_cols = list(df.columns)
        if data_cols != cols:
            # Attempt to reconfigure headings to df columns
            tree.configure(columns=data_cols)
            for c in data_cols:
                tree.heading(c, text=c)
                tree.column(c, anchor=tk.W, width=140)
            cols = data_cols
        # Insert rows. Respect requested Top-K when available; keep a higher safety cap
        # Show full result set returned by the fetcher (those functions already take a 'limit')
        max_rows = len(df)
        for _, row in df.head(max_rows).iterrows():
            vals = [self._fmt_cell(row.get(c, "")) for c in cols]
            tree.insert("", tk.END, values=vals)

    def _fmt_cell(self, v: object) -> str:
        # Pretty formatting: round floats to 5 decimals, keep ints raw
        try:
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            # Avoid formatting dicts/complex objects
            if isinstance(v, dict):
                return str(v)
            # Try as float
            fv = float(v)
            if np.isfinite(fv):
                return f"{fv:.5f}"
        except Exception:
            pass
        return str(v)

    def _on_toggle_weight(self, event: tk.Event, tree: ttk.Treeview, kind: str, name_col: str) -> None:
        try:
            item_id = tree.identify_row(event.y)
            if not item_id:
                return
            cols = list(tree["columns"])
            values = list(tree.item(item_id, "values") or [])
            row = {c: values[i] if i < len(values) else None for i, c in enumerate(cols)}
            name = row.get(name_col)
            if not name:
                return
            # Toggle weight 0/1 only for features and targets
            if kind not in ("feature", "target"):
                return
            ws = WeightStore(path=self.weights_var.get().strip())
            try:
                cur_w = float(row.get("weight"))
            except Exception:
                cur_w = None
            if cur_w is None:
                cur_w = float((ws.features if kind == "feature" else ws.targets).get(name, 1.0))
            new_w = 0.0 if (cur_w and cur_w > 0.0) else 1.0
            if kind == "feature":
                ws.features[name] = float(new_w)
                # Keep engine from using it: maintain disabled list in sync
                if new_w == 0.0:
                    ws.disabled_features = list(sorted(set((ws.disabled_features or []) + [name])))
                else:
                    ws.disabled_features = [x for x in (ws.disabled_features or []) if x != name]
            else:
                ws.targets[name] = float(new_w)
                if new_w == 0.0:
                    ws.disabled_targets = list(sorted(set((ws.disabled_targets or []) + [name])))
                else:
                    ws.disabled_targets = [x for x in (ws.disabled_targets or []) if x != name]
            try:
                ws.save()
            except Exception:
                pass
            # Update weight cell in-place
            if "weight" in row:
                row["weight"] = new_w
                new_vals = [row.get(c, "") for c in cols]
                tree.item(item_id, values=new_vals)
        except Exception:
            pass

    # ------------------- Validation window -------------------
    def _on_open_validation(self, event: tk.Event, tree: ttk.Treeview) -> None:
        try:
            item_id = tree.identify_row(event.y)
            if not item_id:
                return
            cols = list(tree["columns"]) or []
            vals = list(tree.item(item_id, "values") or [])
            row = {c: vals[i] if i < len(vals) else None for i, c in enumerate(cols)}
            run_id = row.get("run_id")
            if not run_id:
                return
            db_path = self.db_var.get().strip() or "runs.db"
            # Try to guess dataset from cache based on selection
            guessed = self._guess_df_name_for_run(db_path, str(run_id))
            default_df = guessed or os.environ.get("DF_NAME", "BTCUSDT_5m_20230831_20250830")
            ValidationWindow(self, db_path=db_path, run_id=str(run_id), default_df=str(default_df))
        except Exception as e:
            try:
                messagebox.showerror("Validation", str(e))
            except Exception:
                pass

    def _guess_df_name_for_run(self, db_path: str, run_id: str) -> str | None:
        """Best-effort: find a cached DF_NAME that has this run's target and most features.

        Scans DATA_DIR for base parquet files and picks the one with target present
        and highest feature coverage.
        """
        try:
            from src.storage import DATA_DIR
            import sqlite3, json
            con = sqlite3.connect(db_path)
            try:
                cur = con.execute("SELECT selection_json FROM runs WHERE run_id=? LIMIT 1", (run_id,))
                row = cur.fetchone()
            finally:
                con.close()
            if not row or not row[0]:
                return None
            sel = json.loads(row[0])
            target = sel.get('target')
            feats = sel.get('features') or []
            # base datasets are *.parquet without __feature__/__target__
            if not os.path.isdir(DATA_DIR):
                return None
            cands = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet') and '__' not in f]
            best = None
            best_score = -1
            for f in cands:
                base = f[:-8]
                tpath = os.path.join(DATA_DIR, f"{base}__target__{target}.parquet")
                if not os.path.exists(tpath):
                    continue
                cov = 0
                for x in feats:
                    xpath = os.path.join(DATA_DIR, f"{base}__feature__{x}.parquet")
                    if os.path.exists(xpath):
                        cov += 1
                score = cov / max(1, len(feats))
                if score > best_score:
                    best_score = score
                    best = base
            return best
        except Exception:
            return None


class ValidationWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, *, db_path: str, run_id: str, default_df: str) -> None:
        super().__init__(master)
        self.title(f"Validate run: {run_id}")
        self.geometry("900x650")
        self.db_path = db_path
        self.run_id = run_id
        self.default_df = default_df
        # Cache for selection/metrics/artifacts
        self._sel_cache = None
        self._met_cache = None
        self._art_cache = None
        try:
            info = self._load_run_row(self.db_path, self.run_id)
            self._sel_cache = info.get('selection')
            self._met_cache = info.get('metrics')
            self._art_cache = info.get('artifacts')
            tgt = (self._sel_cache or {}).get('target')
            if tgt:
                self.title(f"Validate run: {run_id} — {tgt}")
        except Exception:
            pass

        # Header with inputs
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text=f"run_id:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(frm, text=run_id).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(frm, text="GAP (bars):").grid(row=0, column=2, sticky=tk.W, padx=(12,4))
        self.gap_var = tk.StringVar(value="288")
        ttk.Entry(frm, textvariable=self.gap_var, width=8).grid(row=0, column=3, sticky=tk.W)
        ttk.Button(frm, text="Validate", command=self._run_validate_async).grid(row=0, column=4, sticky=tk.W, padx=(12,0))
        # Busy status and progress
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=0, column=5, sticky=tk.W, padx=(12,4))
        self.prog = ttk.Progressbar(frm, mode='indeterminate', length=120)
        self.prog.grid(row=0, column=6, sticky=tk.W)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        # Tabs
        self.tab_info = ttk.Frame(self.nb)
        self.nb.add(self.tab_info, text="Info")
        self._build_info_tab(self.tab_info)
        # Populate info now (from DB caches)
        try:
            self._fill_info_tab()
        except Exception:
            pass
        self.tab_metrics = ttk.Frame(self.nb)
        self.nb.add(self.tab_metrics, text="Metrics")
        self._build_metrics_tab(self.tab_metrics)

        # Permutation tab
        self.tab_perm = ttk.Frame(self.nb)
        self.nb.add(self.tab_perm, text="Permutation Test")
        self._build_perm_tab(self.tab_perm)

        # Walk-forward tab
        self.tab_wf = ttk.Frame(self.nb)
        self.nb.add(self.tab_wf, text="Walk-Forward")
        self._build_wf_tab(self.tab_wf)

        self.tab_other = ttk.Frame(self.nb)
        self.nb.add(self.tab_other, text="Predict on DF")
        self._build_other_df_tab(self.tab_other)

        # Plot tab (close + target/prediction overlays)
        self.tab_plot = ttk.Frame(self.nb)
        self.nb.add(self.tab_plot, text="Plot")
        self._build_plot_tab(self.tab_plot)

        # Legend
        legend = (
            "Legend (ideal): AUC>0.6, AP-lift>1, P@1/5/10% >> base rate, "
            "Brier low (<0.2), KS>0.2."
        )
        ttk.Label(self, text=legend).pack(side=tk.BOTTOM, anchor=tk.W, padx=8, pady=6)

        # Do not auto-run validation; user triggers via Validate button

    def _build_metrics_tab(self, parent: ttk.Frame) -> None:
        self.txt_metrics = tk.Text(parent, height=24)
        self.txt_metrics.pack(fill=tk.BOTH, expand=True)

    def _build_info_tab(self, parent: ttk.Frame) -> None:
        # Summary labels
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.info_target_var = tk.StringVar(value="target: ?")
        self.info_model_var = tk.StringVar(value="model: ?")
        self.info_metrics_var = tk.StringVar(value="metrics: (from DB)")
        ttk.Label(frm, textvariable=self.info_target_var).grid(row=0, column=0, sticky=tk.W, padx=(0,12))
        ttk.Label(frm, textvariable=self.info_model_var).grid(row=0, column=1, sticky=tk.W, padx=(0,12))
        ttk.Label(frm, textvariable=self.info_metrics_var).grid(row=0, column=2, sticky=tk.W)
        # Features affinity grid
        cols = ["feature","pearson","spearman","mi","dcor","n_eff"]
        self.info_tree = ttk.Treeview(parent, columns=cols, show="headings")
        for c in cols:
            self.info_tree.heading(c, text=c)
            self.info_tree.column(c, anchor=tk.W, width=120)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.info_tree.yview)
        self.info_tree.configure(yscrollcommand=vsb.set)
        self.info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _fill_info_tab(self) -> None:
        # Set summary labels
        sel = self._sel_cache or {}
        met = self._met_cache or {}
        tgt = sel.get('target') or 'N/A'
        mdl = sel.get('model') or 'N/A'
        self.info_target_var.set(f"target: {tgt}")
        self.info_model_var.set(f"model: {mdl}")
        # Short metrics line if available
        if isinstance(met, dict) and met:
            parts = []
            for k in ("auc","ap","accuracy","skill","r2","spearman"):
                v = met.get(k)
                if v is None:
                    continue
                try:
                    parts.append(f"{k}={float(v):.3f}")
                except Exception:
                    parts.append(f"{k}={v}")
            self.info_metrics_var.set("metrics: " + (", ".join(parts) if parts else "(none)"))
        else:
            self.info_metrics_var.set("metrics: (none)")
        # Fill affinities
        for it in list(self.info_tree.get_children()):
            self.info_tree.delete(it)
        aff = ((self._art_cache or {}).get('affinity')) if isinstance(self._art_cache, dict) else None
        feats = sel.get('features') or []
        def _fmt(x):
            try:
                xv = float(x)
                if xv == xv:  # not NaN
                    return f"{xv:.4f}"
                return "nan"
            except Exception:
                return ""
        if isinstance(aff, dict) and feats:
            for fid in feats:
                d = aff.get(fid) or {}
                row = [
                    fid,
                    _fmt(d.get('pearson')),
                    _fmt(d.get('spearman')),
                    _fmt(d.get('mi')),
                    _fmt(d.get('dcor')),
                    _fmt(d.get('n_eff')),
                ]
                self.info_tree.insert("", tk.END, values=row)
        else:
            # still insert feature names with blanks if no affinities
            for fid in feats:
                self.info_tree.insert("", tk.END, values=[fid, "", "", "", "", ""]) 

    def _load_run_row(self, db_path: str, run_id: str) -> dict:
        import sqlite3, json
        try:
            con = sqlite3.connect(db_path)
            try:
                cur = con.execute(
                    "SELECT selection_json, metrics_json, artifacts_json FROM runs WHERE run_id=? LIMIT 1",
                    (run_id,),
                )
                row = cur.fetchone()
            finally:
                con.close()
            if not row:
                return {}
            sj, mj, aj = row
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
            return {"selection": sel, "metrics": met, "artifacts": art}
        except Exception:
            return {}

    def _build_other_df_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="DF_NAME:").grid(row=0, column=0, sticky=tk.W)
        self.df_var = tk.StringVar(value=self.default_df)
        ttk.Entry(frm, textvariable=self.df_var, width=48).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frm, text="Pick .parquet", command=self._pick_parquet).grid(row=0, column=2, sticky=tk.W, padx=(8,0))
        ttk.Label(frm, text="GAP:").grid(row=0, column=3, sticky=tk.W, padx=(12,4))
        self.gap2_var = tk.StringVar(value="288")
        ttk.Entry(frm, textvariable=self.gap2_var, width=8).grid(row=0, column=4, sticky=tk.W)
        ttk.Label(frm, text="Folds:").grid(row=0, column=5, sticky=tk.W, padx=(12,4))
        self.oof_folds_var = tk.StringVar(value="4")
        ttk.Entry(frm, textvariable=self.oof_folds_var, width=6).grid(row=0, column=6, sticky=tk.W)
        ttk.Button(frm, text="Run", command=self._run_other_df_async).grid(row=0, column=7, sticky=tk.W, padx=(12,0))
        ttk.Button(frm, text="Make OOF Feature", command=self._run_make_oof_async).grid(row=0, column=8, sticky=tk.W, padx=(8,0))
        self.txt_other = tk.Text(parent, height=22)
        self.txt_other.pack(fill=tk.BOTH, expand=True)

    def _pick_parquet(self) -> None:
        try:
            path = filedialog.askopenfilename(title="Select parquet dataset", initialdir=os.getcwd(), filetypes=[("Parquet","*.parquet"), ("All","*.*")])
            if path:
                self.df_var.set(path)
        except Exception:
            pass

    def _pick_parquet_into(self, var: tk.StringVar) -> None:
        try:
            path = filedialog.askopenfilename(title="Select parquet dataset", initialdir=os.getcwd(), filetypes=[("Parquet","*.parquet"), ("All","*.*")])
            if path:
                var.set(path)
        except Exception:
            pass

    def _build_perm_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="GAP (bars):").grid(row=0, column=0, sticky=tk.W)
        self.gap_perm_var = tk.StringVar(value="288")
        ttk.Entry(frm, textvariable=self.gap_perm_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frm, text="Run Permutation", command=self._run_perm_async).grid(row=0, column=2, sticky=tk.W, padx=(12,0))
        self.txt_perm = tk.Text(parent, height=22)
        self.txt_perm.pack(fill=tk.BOTH, expand=True)

    def _build_wf_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="Folds:").grid(row=0, column=0, sticky=tk.W)
        self.wf_folds_var = tk.StringVar(value="4")
        ttk.Entry(frm, textvariable=self.wf_folds_var, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(frm, text="GAP (bars):").grid(row=0, column=2, sticky=tk.W, padx=(12,4))
        self.wf_gap_var = tk.StringVar(value="288")
        ttk.Entry(frm, textvariable=self.wf_gap_var, width=8).grid(row=0, column=3, sticky=tk.W)
        ttk.Button(frm, text="Run Walk-Forward", command=self._run_wf_async).grid(row=0, column=4, sticky=tk.W, padx=(12,0))
        self.txt_wf = tk.Text(parent, height=22)
        self.txt_wf.pack(fill=tk.BOTH, expand=True)

    # ---- Async wrappers ----
    def _run_validate_async(self) -> None:
        self._start_busy("Validating...")
        threading.Thread(target=self._run_validate_safe, daemon=True).start()

    def _run_other_df_async(self) -> None:
        self._start_busy("Predicting on DF...")
        threading.Thread(target=self._run_other_df_safe, daemon=True).start()

    def _run_make_oof_async(self) -> None:
        self._start_busy("Making OOF feature...")
        threading.Thread(target=self._run_make_oof_safe, daemon=True).start()

    def _run_perm_async(self) -> None:
        self._start_busy("Running permutation test...")
        threading.Thread(target=self._run_perm_safe, daemon=True).start()

    def _run_wf_async(self) -> None:
        self._start_busy("Running walk-forward...")
        threading.Thread(target=self._run_wf_safe, daemon=True).start()

    # ---- Core compute helpers ----
    def _run_validate_safe(self) -> None:
        try:
            gap = int(float(self.gap_var.get()))
        except Exception:
            gap = 0
        # Use the default_df provided (already guessed in parent). If empty, ask user to pick.
        df_name = self.default_df or ""
        if not df_name:
            out = {"error": "No dataset resolved. Use 'Predict on DF' and Pick .parquet, or set DF_NAME env."}
            self.after(0, lambda: (self._render_metrics(self.txt_metrics, out), self._finish_busy()))
            return
        try:
            out = self._compute_metrics(df_name, gap)
        except Exception as e:
            out = {"error": str(e)}
        # Schedule UI update back on main thread
        self.after(0, lambda: (self._render_metrics(self.txt_metrics, out), self._finish_busy()))

    def _run_other_df_safe(self) -> None:
        df_name = self.df_var.get().strip() or self.default_df
        try:
            gap = int(float(self.gap2_var.get()))
        except Exception:
            gap = 0
        try:
            out = self._compute_metrics(df_name, gap)
        except Exception as e:
            out = {"error": str(e)}
        self.after(0, lambda: (self._render_metrics(self.txt_other, out), self._finish_busy()))

    def _run_make_oof_safe(self) -> None:
        df_name = self.df_var.get().strip() or self.default_df
        try:
            folds = max(2, int(float(self.oof_folds_var.get())))
        except Exception:
            folds = 4
        try:
            gap = max(0, int(float(self.gap2_var.get())))
        except Exception:
            gap = 288
        try:
            from scripts.make_prediction_feature import make_oof_feature  # type: ignore
            name = make_oof_feature(self.run_id, df_name, folds=folds, gap=gap, out_name=None, db_path=self.db_path)
            # Try to (re)register prediction features so engine can see it
            try:
                import importlib
                import src.prediction_features as pf  # type: ignore
                importlib.reload(pf)
            except Exception:
                pass
            msg = f"Saved OOF feature: {name}\nYou can now include it in selections."
            self.after(0, lambda: (messagebox.showinfo("OOF Feature", msg), self._finish_busy()))
        except Exception as e:
            _msg = str(e)
            self.after(0, lambda m=_msg: (messagebox.showerror("OOF Feature", m), self._finish_busy()))

    def _compute_metrics(self, df_name: str, gap: int) -> Dict[str, Any]:
        # Lazy import to reuse existing helpers
        from scripts.extended_metrics_for_run import _load_selection, _prepare_xy, _fit_predict_tf  # type: ignore
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss
        import numpy as np

        sel = _load_selection(self.db_path, self.run_id)
        # Ensure TA features are registered in this process
        self._ensure_ta_features()
        # Try standard path first; on failure (no cache/db), fall back to in-memory pipe
        try:
            Xtr, ytr, Xte, yte = _prepare_xy(df_name, sel, gap=gap)
        except Exception as e:
            # Fallback: compute everything locally, loading df either from path or storage
            df = self._load_df_flexible(df_name)
            if df is None:
                try:
                    from src import storage
                    df = storage.load_dataframe(df_name)
                except Exception as e2:
                    # If it was a db_to_df error, show actionable message
                    msg = f"{e}; {e2}"
                    if 'db.db_to_df' in msg:
                        raise RuntimeError(
                            "Dataset not found in cache and DB loader unavailable. "
                            "Use 'Pick .parquet' to select an existing dataset file."
                        )
                    # otherwise bubble up original error chain
                    raise
            Xtr, ytr, Xte, yte = self._prepare_xy_local(df, sel, gap=gap)
        # Only classification path for now
        if not (ytr.dropna().nunique() <= 2 and set(ytr.dropna().unique()).issubset({0,1})):
            return {"error": "Non-binary target not supported in this view yet."}
        if sel.model.startswith('tf_mlp'):
            probs = _fit_predict_tf(sel.model, Xtr, ytr, Xte)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)),
            ])
            pipe.fit(Xtr, ytr)
            probs = pipe.predict_proba(Xte)[:, 1]
        y_true = yte.to_numpy().astype(int)
        auc = float(roc_auc_score(y_true, probs))
        ap = float(average_precision_score(y_true, probs))
        pos_rate = float(y_true.mean()) if len(y_true) else float('nan')
        ap_lift = (ap / pos_rate) if pos_rate and pos_rate > 0 else float('nan')
        # threshold metrics
        def tmet(t: float) -> Dict[str, Any]:
            preds = (probs >= t).astype(int)
            return {
                'precision': float(precision_score(y_true, preds, zero_division=0)),
                'recall': float(recall_score(y_true, preds, zero_division=0)),
                'f1': float(f1_score(y_true, preds, zero_division=0)),
                'accuracy': float(accuracy_score(y_true, preds)),
            }
        thr05 = tmet(0.5)
        # best F1
        thr_grid = np.quantile(np.unique(np.clip(probs,0,1)), np.linspace(0,1,256))
        f1s = [(float(f1_score(y_true, (probs>=t).astype(int), zero_division=0)), float(t)) for t in thr_grid]
        best_f1, best_t = max(f1s, key=lambda x: (x[0], x[1])) if f1s else (float('nan'), 0.5)
        thr_best = tmet(best_t)
        # ranking
        def p_at(frac: float) -> float:
            k = max(1, int(len(probs) * frac))
            idx = np.argsort(-probs)[:k]
            return float(y_true[idx].mean())
        # calibration
        brier = float(brier_score_loss(y_true, np.clip(probs,0,1)))
        pos = np.sort(probs[y_true==1])
        neg = np.sort(probs[y_true==0])
        ks = float('nan')
        if len(pos) and len(neg):
            thr = np.unique(np.concatenate([pos, neg]))
            cdf_pos = np.searchsorted(pos, thr, side='right')/len(pos)
            cdf_neg = np.searchsorted(neg, thr, side='right')/len(neg)
            ks = float(np.max(np.abs(cdf_pos-cdf_neg)))
        return {
            'db_path': self.db_path,
            'df_name': df_name,
            'run_id': self.run_id,
            'target': sel.target,
            'model': sel.model,
            'n_test': int(len(y_true)),
            'pos_rate': pos_rate,
            'auc': auc,
            'ap': ap,
            'ap_lift': ap_lift,
            'p@1%': p_at(0.01),
            'p@5%': p_at(0.05),
            'p@10%': p_at(0.10),
            'brier': brier,
            'ks': ks,
            'thresholds': {
                't0.5': thr05,
                'best_f1': {'threshold': float(best_t), **thr_best},
            },
            'probs': probs,  # include for permutation convenience
            'y_true': y_true,
            'test_index': list(yte.index),
        }

    # ---- Plot tab ----
    def _build_plot_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        # DF selector for plotting (defaults to provided default_df)
        ttk.Label(frm, text="DF_NAME:").grid(row=0, column=0, sticky=tk.W)
        self.plot_df_var = tk.StringVar(value=self.default_df)
        ttk.Entry(frm, textvariable=self.plot_df_var, width=48).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frm, text="Pick .parquet", command=lambda: self._pick_parquet_into(self.plot_df_var)).grid(row=0, column=2, sticky=tk.W, padx=(8,0))

        ttk.Label(frm, text="Start:").grid(row=1, column=0, sticky=tk.W, pady=(6,0))
        self.plot_start_var = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.plot_start_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=(6,0))
        ttk.Label(frm, text="Len:").grid(row=1, column=2, sticky=tk.W, padx=(12,4), pady=(6,0))
        self.plot_len_var = tk.IntVar(value=1000)
        ttk.Entry(frm, textvariable=self.plot_len_var, width=10).grid(row=1, column=3, sticky=tk.W, pady=(6,0))
        ttk.Button(frm, text="Compute", command=self._run_plot_compute_async).grid(row=1, column=4, sticky=tk.W, padx=(12,0), pady=(6,0))
        ttk.Button(frm, text="Anterior", command=self._plot_prev).grid(row=1, column=5, sticky=tk.W, padx=(6,0), pady=(6,0))
        ttk.Button(frm, text="Próximo", command=self._plot_next).grid(row=1, column=6, sticky=tk.W, padx=(6,0), pady=(6,0))
        ttk.Button(frm, text="Redraw", command=self._plot_redraw).grid(row=1, column=7, sticky=tk.W, padx=(6,0), pady=(6,0))
        ttk.Button(frm, text="Jump to Test", command=self._plot_jump_to_test).grid(row=1, column=8, sticky=tk.W, padx=(6,0), pady=(6,0))

        # Info line
        self.plot_info_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.plot_info_var).pack(side=tk.TOP, anchor=tk.W, padx=8)

        # Figure with two axes
        self.fig_plot = plt.Figure(figsize=(9, 6))
        self.ax_gt = self.fig_plot.add_subplot(211)
        self.ax_pred = self.fig_plot.add_subplot(212, sharex=self.ax_gt)
        cnv = FigureCanvasTkAgg(self.fig_plot, master=parent)
        self.canvas_plot = cnv
        cnv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        try:
            NavigationToolbar2Tk(cnv, parent)
        except Exception:
            pass
        # caches
        self._plot_df_cache = None
        self._plot_out_cache = None

    def _run_plot_compute_async(self) -> None:
        self._start_busy("Computing plot data...")
        threading.Thread(target=self._run_plot_compute_safe, daemon=True).start()

    def _run_plot_compute_safe(self) -> None:
        try:
            # Prefer Plot's DF selection; fall back to Predict tab DF, then default
            try:
                df_name = (self.plot_df_var.get().strip()) or self.df_var.get().strip() or self.default_df
            except Exception:
                df_name = self.df_var.get().strip() or self.default_df
            try:
                gap = int(float(self.gap2_var.get()))
            except Exception:
                gap = 0
            # Load df
            df = self._load_df_flexible(df_name)
            if df is None:
                from src import storage
                df = storage.load_dataframe(df_name)
            # Compute metrics to get y_true/probs + test index
            out = self._compute_metrics(df_name, gap)
            self._plot_df_cache = df
            self._plot_out_cache = out
            # Schedule redraw
            self.after(0, lambda: (self._plot_redraw(), self._finish_busy()))
        except Exception as e:
            _msg = str(e)
            self.after(0, lambda m=_msg: (messagebox.showerror("Plot error", m), self._finish_busy()))

    def _plot_redraw(self) -> None:
        df = self._plot_df_cache
        out = self._plot_out_cache
        if df is None or out is None:
            return
        # Prepare window
        try:
            start = max(0, int(self.plot_start_var.get()))
        except Exception:
            start = 0
        try:
            ln = max(10, int(self.plot_len_var.get()))
        except Exception:
            ln = 1000
        end = min(len(df), start + ln)

        dfx = df.iloc[start:end].copy()
        if 'Date' not in dfx.columns:
            dfx['Date'] = dfx.index
        # choose close-like column
        close_col = 'close' if 'close' in dfx.columns else (dfx.columns[0] if len(dfx.columns) else None)
        if close_col is None:
            return

        # Build series for y_true and probs with test indices, then align to visible window
        try:
            idx = pd.Index(out.get('test_index') or [])
        except Exception:
            idx = pd.Index([])
        yt_vals = out.get('y_true')
        if yt_vals is None:
            yt_vals = []
        y_true_s = pd.Series(yt_vals, index=idx)
        pr_vals = out.get('probs')
        if pr_vals is None:
            pr_vals = []
        probs_s = pd.Series(pr_vals, index=idx)
        # Align to the visible window index directly
        y_vis = y_true_s.reindex(dfx.index)
        p_vis = probs_s.reindex(dfx.index)

        # Clear axes
        for ax in (self.ax_gt, self.ax_pred):
            ax.clear()

        # 1) Ground truth scatter over close
        self.ax_gt.plot(dfx['Date'], dfx[close_col], color='black', linewidth=1.0, label=close_col, zorder=1)
        # Map classes to colors
        try:
            classes = sorted(pd.unique(y_vis.dropna().astype(int)))
        except Exception:
            classes = []
        palette = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(classes)}
        mask_gt = y_vis.notna()
        if mask_gt.any():
            y_classes = y_vis[mask_gt].astype(int)
            colors_gt = [color_map.get(int(v), '#7f7f7f') for v in y_classes]
            self.ax_gt.scatter(
                dfx.loc[mask_gt, 'Date'],
                dfx.loc[mask_gt, close_col],
                c=colors_gt,
                s=36,
                alpha=0.9,
                edgecolors='none',
                label='target',
                zorder=3,
            )
        self.ax_gt.set_title('Close + Ground Truth (target)')
        self.ax_gt.grid(True)
        try:
            self.ax_gt.legend()
        except Exception:
            pass

        # 2) Prediction scatter with gradient
        self.ax_pred.plot(dfx['Date'], dfx[close_col], color='black', linewidth=1.0, label=close_col, zorder=1)
        mask_pr = p_vis.notna()
        if mask_pr.any():
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
            cmap = plt.get_cmap('RdYlGn')
            colors_pred = cmap(norm(p_vis[mask_pr].to_numpy(dtype=float)))
            self.ax_pred.scatter(
                dfx.loc[mask_pr, 'Date'],
                dfx.loc[mask_pr, close_col],
                c=colors_pred,
                s=36,
                alpha=0.9,
                edgecolors='none',
                label='probs',
                zorder=3,
            )
        self.ax_pred.set_title('Close + Prediction (prob gradient)')
        self.ax_pred.grid(True)
        try:
            self.ax_pred.legend()
        except Exception:
            pass

        for ax in (self.ax_gt, self.ax_pred):
            try:
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
        try:
            self.fig_plot.tight_layout()
        except Exception:
            pass
        try:
            self.canvas_plot.draw_idle()
        except Exception:
            pass

        # Update info
        try:
            n_gt = int(mask_gt.sum()) if 'mask_gt' in locals() else 0
            n_pr = int(mask_pr.sum()) if 'mask_pr' in locals() else 0
            t_total = int(len(out.get('y_true') or []))
            self.plot_info_var.set(f"Test points in window: {n_gt} | Pred points: {n_pr} | Total test: {t_total}")
        except Exception:
            pass

    def _plot_jump_to_test(self) -> None:
        # Move Start to first test index position
        df = self._plot_df_cache
        out = self._plot_out_cache
        if df is None or out is None:
            return
        idx_list = out.get('test_index') or []
        if not idx_list:
            return
        first = idx_list[0]
        try:
            pos = int(df.index.get_indexer([first])[0])
            if pos < 0:
                return
            try:
                ln = max(10, int(self.plot_len_var.get()))
            except Exception:
                ln = 1000
            start = max(0, pos - ln // 10)
            self.plot_start_var.set(start)
            self._plot_redraw()
        except Exception:
            pass

    def _plot_prev(self) -> None:
        try:
            ln = max(10, int(self.plot_len_var.get()))
        except Exception:
            ln = 1000
        try:
            cur = int(self.plot_start_var.get())
        except Exception:
            cur = 0
        new_start = max(0, cur - ln)
        self.plot_start_var.set(new_start)
        self._plot_redraw()

    def _plot_next(self) -> None:
        df = self._plot_df_cache
        if df is None:
            return
        try:
            ln = max(10, int(self.plot_len_var.get()))
        except Exception:
            ln = 1000
        try:
            cur = int(self.plot_start_var.get())
        except Exception:
            cur = 0
        new_start = min(max(0, len(df) - ln), cur + ln)
        self.plot_start_var.set(new_start)
        self._plot_redraw()

    def _load_df_flexible(self, df_name: str):
        """Load a dataframe from path if provided; returns None if not a file path."""
        import pandas as pd
        import os
        try:
            if isinstance(df_name, str) and os.path.exists(df_name):
                if df_name.endswith('.parquet'):
                    df = pd.read_parquet(df_name)
                elif df_name.endswith('.csv'):
                    # best-effort CSV loader; expects datetime index or a 'date'/'timestamp' column
                    df = pd.read_csv(df_name)
                    for col in ('date','timestamp','time'):
                        if col in df.columns:
                            try:
                                df[col] = pd.to_datetime(df[col])
                                df = df.set_index(col)
                                break
                            except Exception:
                                pass
                else:
                    # attempt parquet by default
                    df = pd.read_parquet(df_name)
                try:
                    df.attrs['__df_name__'] = os.path.splitext(os.path.basename(df_name))[0]
                except Exception:
                    pass
                return df
        except Exception:
            return None
        return None

    def _prepare_xy_local(self, df: 'pd.DataFrame', sel: Dict[str, Any], gap: int = 0):
        """Build train/test matrices from a raw df computing target and features on the fly.

        Accepts either a plain dict selection {target, features, model} or the
        Selection dataclass used by extended_metrics_for_run._load_selection.
        """
        import pandas as pd
        import numpy as np
        from src.registry import registry
        self._ensure_ta_features()
        from src.basic_components import _time_stratified_split
        from src import storage
        # Normalize selection interface (dict or dataclass with attributes)
        try:
            target_id = sel['target']  # type: ignore[index]
            features_ids = sel.get('features', [])  # type: ignore[assignment]
        except Exception:
            # Dataclass path
            target_id = getattr(sel, 'target')
            features_ids = getattr(sel, 'features', [])
        # Target
        y = registry.targets[target_id](df)
        # Try to cache target if we have a df name
        try:
            df_name = getattr(df, 'attrs', {}).get('__df_name__')
            if isinstance(df_name, str) and df_name:
                storage.save_target(df_name, target_id, y)
        except Exception:
            pass
        # Features (compute one by one; ignore failures)
        feats = {}
        for fid in features_ids:
            try:
                x = registry.features[fid](df)
                if isinstance(x, pd.DataFrame) and getattr(x, 'shape', (0,0))[1] == 1:
                    x = x.iloc[:,0]
                feats[fid] = x
                # Cache feature when possible
                try:
                    if isinstance(df_name, str) and df_name:
                        storage.save_feature(df_name, fid, x)
                except Exception:
                    pass
            except Exception:
                continue
        if not feats:
            raise RuntimeError('No features computed successfully for this DF')
        X = pd.concat(feats, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:,0].notna()
        y_clean = combined.loc[mask, combined.columns[0]]
        X_clean = combined.loc[mask, combined.columns[1:]]
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        if y_clean.nunique() < 2 or len(X_clean) < 100:
            raise RuntimeError('Insufficient data or single-class target after cleaning')
        split = _time_stratified_split(y_clean, min_train=100, min_test=100)
        gap_idx = min(len(X_clean), max(0, int(split) + int(gap)))
        X_train_df = X_clean.iloc[:split]
        X_test_df = X_clean.iloc[gap_idx:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[gap_idx:]
        return X_train_df, y_train, X_test_df, y_test

    def _run_perm_safe(self) -> None:
        try:
            try:
                gap = int(float(self.gap_perm_var.get()))
            except Exception:
                gap = 0
            base = self._compute_metrics(self.default_df, gap)
            if 'error' in base:
                out = base
            else:
                import numpy as np
                y_true = base.get('y_true')
                probs = base.get('probs')
                if y_true is None or probs is None:
                    out = {'error': 'No base metrics to permute.'}
                else:
                    rng = np.random.default_rng(42)
                    y_perm = np.copy(y_true)
                    rng.shuffle(y_perm)
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    auc_p = float(roc_auc_score(y_perm, probs))
                    ap_p = float(average_precision_score(y_perm, probs))
                    out = {
                        'note': 'Permutation test (labels shuffled)'
                    }
                    out.update({
                        'auc_perm': auc_p,
                        'ap_perm': ap_p,
                        'baseline_pos_rate': float(np.mean(y_true)),
                    })
        except Exception as e:
            out = {'error': str(e)}
        self.after(0, lambda: (self._render_perm(self.txt_perm, out), self._finish_busy()))

    def _render_perm(self, txt: tk.Text, out: Dict[str, Any]) -> None:
        txt.configure(state=tk.NORMAL)
        txt.delete('1.0', tk.END)
        if not out:
            txt.insert(tk.END, 'No result')
        elif 'error' in out:
            txt.insert(tk.END, f"Error: {out['error']}\n")
        else:
            # Add a rating: ideal permutation behaves like random (AUC~0.5, AP~base)
            score5, label = self._grade_permutation(out)
            lines = []
            lines.append(f"Rating: {score5:.1f} / 5  ({label})")
            lines.append(out.get('note','Permutation'))
            lines.append(f"AUC_perm: {out.get('auc_perm'):.5f}")
            lines.append(f"AP_perm: {out.get('ap_perm'):.5f} | baseline pos_rate: {out.get('baseline_pos_rate'):.5f}")
            txt.insert(tk.END, "\n".join(lines) + "\n")
        txt.configure(state=tk.DISABLED)

    def _run_wf_safe(self) -> None:
        try:
            try:
                folds = max(2, int(float(self.wf_folds_var.get())))
            except Exception:
                folds = 4
            try:
                gap = int(float(self.wf_gap_var.get()))
            except Exception:
                gap = 0
            res = self._walk_forward(self.default_df, folds, gap)
        except Exception as e:
            res = {'error': str(e)}
        self.after(0, lambda: (self._render_wf(self.txt_wf, res), self._finish_busy()))

    def _walk_forward(self, df_name: str, folds: int, gap: int) -> Dict[str, Any]:
        # Build full cleaned dataset (similar to extended_metrics _prepare_xy, but no split)
        from scripts.extended_metrics_for_run import _load_selection  # type: ignore
        from src.registry import registry
        from src import storage
        self._ensure_ta_features()
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score
        sel = _load_selection(self.db_path, self.run_id)
        df = self._load_df_flexible(df_name)
        if df is None:
            # try storage.load_dataframe which can fetch from DB if available
            df = storage.load_dataframe(df_name)
        # Resolve a df_name usable for cache lookups
        try:
            cache_df_name = getattr(df, 'attrs', {}).get('__df_name__') or df_name
        except Exception:
            cache_df_name = df_name
        # Target
        try:
            if storage.target_exists(cache_df_name, sel.target):
                y = storage.load_target(cache_df_name, sel.target)
            else:
                y = registry.targets[sel.target](df)
                try:
                    storage.save_target(cache_df_name, sel.target, y)
                except Exception:
                    pass
        except Exception:
            # fallback to registry if cache load failed unexpectedly
            y = registry.targets[sel.target](df)
        # Features
        feats = {}
        for fid in sel.features:
            try:
                # Prefer cached feature if available
                if storage.feature_exists(cache_df_name, fid):
                    x = storage.load_feature(cache_df_name, fid)
                else:
                    x = registry.features[fid](df)
                    try:
                        storage.save_feature(cache_df_name, fid, x)
                    except Exception:
                        pass
                if isinstance(x, pd.DataFrame) and getattr(x, 'shape', (0,0))[1] == 1:
                    x = x.iloc[:,0]
                feats[fid] = x
            except Exception:
                continue
        if not feats:
            return {'error': 'No features computed for this DF. If your selection uses TA features, ensure pandas_ta is installed (pip install pandas-ta), or run "Predict on DF" first to cache features for this dataset.'}
        import pandas as pd
        X = pd.concat(feats, axis=1)
        combined = pd.concat([y, X], axis=1)
        mask = combined.iloc[:,0].notna()
        y_clean = combined.loc[mask, combined.columns[0]].astype(float)
        X_clean = combined.loc[mask, combined.columns[1:]].replace([np.inf,-np.inf], np.nan)
        try:
            X_clean = X_clean.infer_objects(copy=False)
        except Exception:
            pass
        X_clean = X_clean.dropna(axis=1, how='all')
        n = len(y_clean)
        if n < 500:
            return {'error': 'Insufficient data for walk-forward'}
        # fold ranges
        results = []
        for i in range(folds):
            te_start = int(n * i / folds)
            te_end = int(n * (i+1) / folds)
            split_idx = max(0, te_start - int(gap))
            if split_idx < 100 or (te_end - te_start) < 100:
                continue
            y_tr = y_clean.iloc[:split_idx]
            y_te = y_clean.iloc[te_start:te_end]
            if y_tr.dropna().nunique() < 2 or y_te.dropna().nunique() < 2:
                continue
            X_tr = X_clean.iloc[:split_idx]
            X_te = X_clean.iloc[te_start:te_end]
            # Fit model consistent with selection
            probs = self._fit_predict_sel(sel.model, X_tr, y_tr, X_te)
            yt = y_te.to_numpy().astype(int)
            auc = float(roc_auc_score(yt, probs))
            ap = float(average_precision_score(yt, probs))
            pr = float(yt.mean())
            results.append({'fold': i+1, 'n_test': len(yt), 'pos_rate': pr, 'auc': auc, 'ap': ap})
        if not results:
            return {'error': 'No valid folds produced'}
        aucs = [r['auc'] for r in results]
        aps = [r['ap'] for r in results]
        def q(v, p):
            return float(np.quantile(np.array(v), p))
        summary = {
            'auc_median': q(aucs, 0.5), 'auc_iqr': (q(aucs,0.25), q(aucs,0.75)),
            'ap_median': q(aps, 0.5), 'ap_iqr': (q(aps,0.25), q(aps,0.75)),
            'folds': len(results)
        }
        return {'folds': results, 'summary': summary}

    def _fit_predict_sel(self, model_id: str, X_tr: 'pd.DataFrame', y_tr: 'pd.Series', X_te: 'pd.DataFrame') -> 'np.ndarray':
        # Reuse TF helper if applicable; else logistic
        from scripts.extended_metrics_for_run import _fit_predict_tf  # type: ignore
        if model_id.startswith('tf_mlp'):
            return _fit_predict_tf(model_id, X_tr, y_tr, X_te)
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

    def _render_wf(self, txt: tk.Text, res: Dict[str, Any]) -> None:
        txt.configure(state=tk.NORMAL)
        txt.delete('1.0', tk.END)
        if not res:
            txt.insert(tk.END, 'No result')
        elif 'error' in res:
            txt.insert(tk.END, f"Error: {res['error']}\n")
        else:
            summary = res.get('summary', {})
            folds = res.get('folds', [])
            lines = []
            if summary:
                lines.append(
                    f"AUC median: {summary.get('auc_median', float('nan')):.5f} | IQR: ({summary.get('auc_iqr',(float('nan'),float('nan')))[0]:.5f}, {summary.get('auc_iqr',(float('nan'),float('nan')))[1]:.5f})"
                )
                lines.append(
                    f"AP  median: {summary.get('ap_median', float('nan')):.5f} | IQR: ({summary.get('ap_iqr',(float('nan'),float('nan')))[0]:.5f}, {summary.get('ap_iqr',(float('nan'),float('nan')))[1]:.5f}) | folds: {summary.get('folds')}"
                )
                lines.append("")
            if folds:
                lines.append("Per fold:")
                for r in folds:
                    lines.append(f"fold {r['fold']}: n_test={r['n_test']} pos_rate={r['pos_rate']:.5f} AUC={r['auc']:.5f} AP={r['ap']:.5f}")
            txt.insert(tk.END, "\n".join(lines) + "\n")
        txt.configure(state=tk.DISABLED)

    def _render_metrics(self, txt: tk.Text, out: Dict[str, Any]) -> None:
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        if not out:
            txt.insert(tk.END, "No result")
            txt.configure(state=tk.DISABLED)
            return
        if 'error' in out:
            txt.insert(tk.END, f"Error: {out['error']}\n")
            txt.configure(state=tk.DISABLED)
            return
        def fmt(x):
            try:
                xf = float(x)
                return f"{xf:.5f}"
            except Exception:
                return str(x)
        # Add overall rating (0..5) with qualitative label
        score_5, label = self._grade_overall(out)
        lines = []
        lines.append(f"Rating: {score_5:.1f} / 5  ({label})")
        lines.append(f"DF: {out['df_name']}  |  n_test: {out['n_test']}  |  pos_rate: {fmt(out['pos_rate'])}")
        lines.append(f"AUC: {fmt(out['auc'])}  |  AP: {fmt(out['ap'])}  |  AP-lift: {fmt(out['ap_lift'])}")
        lines.append(f"P@1%: {fmt(out['p@1%'])}  |  P@5%: {fmt(out['p@5%'])}  |  P@10%: {fmt(out['p@10%'])}")
        lines.append(f"Brier: {fmt(out['brier'])}  |  KS: {fmt(out['ks'])}")
        t05 = out['thresholds']['t0.5']
        tb = out['thresholds']['best_f1']
        lines.append(f"@0.5  -> Prec: {fmt(t05['precision'])}, Rec: {fmt(t05['recall'])}, F1: {fmt(t05['f1'])}, Acc: {fmt(t05['accuracy'])}")
        lines.append(f"@best -> t={fmt(tb['threshold'])} | Prec: {fmt(tb['precision'])}, Rec: {fmt(tb['recall'])}, F1: {fmt(tb['f1'])}, Acc: {fmt(tb['accuracy'])}")
        txt.insert(tk.END, "\n".join(lines) + "\n")
        txt.configure(state=tk.DISABLED)

    def _ensure_ta_features(self) -> None:
        """Attempt to import TA feature registry if not yet loaded."""
        try:
            # If features already registered, this will be a no-op
            import src.ta_features  # noqa: F401
        except Exception:
            # Keep UI responsive even if TA not available
            pass

    # Busy helpers
    def _start_busy(self, msg: str) -> None:
        try:
            self.status_var.set(msg)
            self.prog.start(50)
        except Exception:
            pass

    def _finish_busy(self) -> None:
        try:
            self.prog.stop()
            self.status_var.set("Idle")
        except Exception:
            pass

    # ---- Grading helper (0..5) ----
    def _grade_overall(self, out: Dict[str, Any]) -> tuple[float, str]:
        import math
        def _s(val: float, lo: float, hi: float, invert: bool = False) -> float:
            try:
                x = float(val)
            except Exception:
                return 0.0
            if math.isnan(x) or not math.isfinite(x):
                return 0.0
            if invert:
                # smaller is better
                x = max(lo, min(hi, x))
                return 1.0 - (x - lo) / max(1e-9, (hi - lo))
            x = max(lo, min(hi, x))
            return (x - lo) / max(1e-9, (hi - lo))

        auc = float(out.get('auc', 0.0) or 0.0)
        ks = float(out.get('ks', 0.0) or 0.0)
        brier = float(out.get('brier', 1.0) or 1.0)
        ap_lift = float(out.get('ap_lift', 1.0) or 1.0)
        pr = float(out.get('pos_rate', 0.0) or 0.0)
        p1 = float(out.get('p@1%', pr) or pr)
        p5 = float(out.get('p@5%', pr) or pr)
        p10 = float(out.get('p@10%', pr) or pr)
        # Normalize components
        s_auc = _s(auc, 0.5, 0.8)  # >0.8 considered strong
        s_ks = _s(ks, 0.1, 0.6)    # KS 0.1..0.6
        s_brier = _s(brier, 0.05, 0.25, invert=True)  # lower is better
        s_lift = _s(ap_lift, 1.0, 1.2)  # +20% improvement caps
        def _imp(p):
            return 0.0 if p <= pr else (p - pr) / max(1e-6, 1.0 - pr)
        s_rank = max(0.0, min(1.0, (0.5*_imp(p1) + 0.3*_imp(p5) + 0.2*_imp(p10))))
        # Weighted blend
        score01 = (
            0.30 * s_auc +
            0.20 * s_ks +
            0.20 * s_brier +
            0.15 * s_lift +
            0.15 * s_rank
        )
        score01 = max(0.0, min(1.0, score01))
        score5 = 5.0 * score01
        # Labels
        if score5 < 1.0:
            label = "Muy Mala"
        elif score5 < 2.0:
            label = "Mala"
        elif score5 < 3.0:
            label = "Aceptable"
        elif score5 < 4.0:
            label = "Buena"
        else:
            label = "Excelente"
        return score5, label

    def _grade_permutation(self, out: Dict[str, Any]) -> tuple[float, str]:
        """Higher is better if permutation behaves like random.

        - Ideal: auc_perm ~ 0.5, ap_perm ~ baseline_pos_rate.
        - Penalize deviations beyond small tolerances.
        """
        import math
        try:
            auc_p = float(out.get('auc_perm', float('nan')))
        except Exception:
            auc_p = float('nan')
        try:
            ap_p = float(out.get('ap_perm', float('nan')))
        except Exception:
            ap_p = float('nan')
        try:
            base = float(out.get('baseline_pos_rate', float('nan')))
        except Exception:
            base = float('nan')
        # Scores in [0,1]
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))
        # AUC deviation tolerance: 0.0 at |d|>=0.1, full at 0
        if math.isnan(auc_p):
            s_auc = 0.0
        else:
            d_auc = abs(auc_p - 0.5)
            s_auc = clamp01(1.0 - (d_auc / 0.10))
        # AP deviation tolerance: scale by available headroom; guard for extremes
        if math.isnan(ap_p) or math.isnan(base):
            s_ap = 0.0
        else:
            denom = max(0.05, 1.0 - min(0.99, base))  # avoid tiny denominators cuando base es alta
            d_ap = abs(ap_p - base)
            s_ap = clamp01(1.0 - (d_ap / denom))
        score01 = clamp01(0.6 * s_auc + 0.4 * s_ap)
        score5 = 5.0 * score01
        if score5 < 1.0:
            label = "Muy Mala"
        elif score5 < 2.0:
            label = "Mala"
        elif score5 < 3.0:
            label = "Aceptable"
        elif score5 < 4.0:
            label = "Buena"
        else:
            label = "Excelente"
        return score5, label


def main() -> None:
    app = MonitorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

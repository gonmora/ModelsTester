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
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import numpy as np

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

try:
    from src.reporting import (
        WeightStore,
        top_features,
        top_targets,
        targets_table,
        models_historical,
        targets_historical,
        runs_overview,
    )
    # pairs_historical may not exist on older versions
    try:
        from src.reporting import pairs_historical
    except Exception:
        pairs_historical = None
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

        self._build_controls()
        self._build_tabs()

    def _build_controls(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="DB path:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.db_var = tk.StringVar(value="runs_BIG.db")
        ttk.Entry(frm, textvariable=self.db_var, width=40).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frm, text="...", command=self._pick_db).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(frm, text="WEIGHTS_JSON:").grid(row=0, column=3, sticky=tk.W, padx=(12,4))
        self.weights_var = tk.StringVar(value=os.environ.get("WEIGHTS_JSON", "data/weights_BIG.json"))
        ttk.Entry(frm, textvariable=self.weights_var, width=40).grid(row=0, column=4, sticky=tk.W)
        ttk.Button(frm, text="...", command=self._pick_weights).grid(row=0, column=5, sticky=tk.W)

        ttk.Label(frm, text="Refresh (s):").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6,0))
        self.refresh_var = tk.StringVar(value="2.0")
        ttk.Entry(frm, textvariable=self.refresh_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=(6,0))

        ttk.Label(frm, text="Top-K:").grid(row=1, column=3, sticky=tk.W, padx=(12,4), pady=(6,0))
        self.topk_var = tk.StringVar(value="10")
        ttk.Entry(frm, textvariable=self.topk_var, width=6).grid(row=1, column=4, sticky=tk.W, pady=(6,0))

        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(frm, textvariable=self.status_var).grid(row=1, column=5, sticky=tk.W, padx=(12,0), pady=(6,0))

        btns = ttk.Frame(frm)
        btns.grid(row=0, column=6, rowspan=2, padx=(16,0))
        ttk.Button(btns, text="Start", command=self.start).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_features = self._add_table_tab("Top Features", cols=["feature","weight","n","mean","best","last","rank"]) 
        self.tab_targets = self._add_table_tab("Targets", cols=["target","weight","n","mean","best","last","rank"]) 
        self.tab_tq = self._add_table_tab("Target Quality", cols=["target","runs","auc_median","auc_q025","auc_q975","auc_iqr","acc_median","ap_median","pos_rate_test_median"]) 
        self.tab_mq = self._add_table_tab("Model Quality & Timing", cols=["model","runs","auc_median","ap_median","fit_time_median_sec","predict_time_median_sec"]) 
        self.tab_pairs = self._add_table_tab("Target x Model", cols=["target","model","runs","auc_median","ap_median"]) 
        self.tab_runs = self._add_table_tab("Recent Runs", cols=["run_id","status","model","metrics"]) 

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
        return tree

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

    def start(self) -> None:
        try:
            self.refresh_ms = int(float(self.refresh_var.get()) * 1000)
        except Exception:
            self.refresh_ms = 2000
        try:
            self.top_k = int(self.topk_var.get())
        except Exception:
            self.top_k = 10
        # Apply weights env
        os.environ["WEIGHTS_JSON"] = self.weights_var.get().strip()
        self.running = True
        self.status_var.set("Running")
        self._schedule_refresh()

    def stop(self) -> None:
        self.running = False
        self.status_var.set("Stopped")

    def _schedule_refresh(self) -> None:
        if not self.running:
            return
        self.after(self.refresh_ms, self._refresh)

    def _refresh(self) -> None:
        if not self.running:
            return
        db = self.db_var.get().strip()
        # Top Features & Targets (from weights)
        try:
            # Use the selected weights file explicitly to avoid stale module-level defaults
            ws = WeightStore(path=self.weights_var.get().strip())
            # Top features with fallback to include_unseen when stats are empty
            try:
                tf_df = top_features(self.top_k, include_unseen=False, ws=ws)
                if len(tf_df) == 0:
                    tf_df = top_features(self.top_k, include_unseen=True, ws=ws)
            except Exception:
                tf_df = pd.DataFrame()
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
            self._fill_tree(self.tab_targets, tt_df)
        except Exception as e:
            self._set_tree_error(self.tab_features, f"Error: {e}")
            self._set_tree_error(self.tab_targets, f"Error: {e}")
        # Historical tables (DB)
        try:
            self._fill_tree(self.tab_tq, targets_historical(db_path=db, min_runs=3))
        except Exception as e:
            self._set_tree_error(self.tab_tq, f"Error: {e}")
        try:
            self._fill_tree(self.tab_mq, models_historical(db_path=db, min_runs=3))
        except Exception as e:
            self._set_tree_error(self.tab_mq, f"Error: {e}")
        if pairs_historical is not None:
            try:
                self._fill_tree(self.tab_pairs, pairs_historical(db_path=db, min_runs=3))
            except Exception as e:
                self._set_tree_error(self.tab_pairs, f"Error: {e}")
        else:
            self._set_tree_error(self.tab_pairs, "pairs_historical() not available")
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
        # Reschedule
        self._schedule_refresh()

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
        # Insert rows (limit size on large tables)
        max_rows = 200
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


def main() -> None:
    app = MonitorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

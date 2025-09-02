# -*- coding: utf-8 -*-
"""
Reusable Tkinter time series plotter similar to the notebook TimeSeriesPlotter.

Features:
- Embeds a Matplotlib figure in a Tk container (Frame or Toplevel)
- Navigation: Previous/Next window using a single window length 'n'
- Adjustable window length 'n'
- plot_params list supports:
  - 'column': driving series name (also used for color in dynamic_color)
  - 'overlay_on' (optional): y-values to plot instead of the color source
  - 'normalize_with' (optional): column to match min/max range
  - 'dynamic_color' (bool): scatter with colormap based on 'column' values
  - 'vmin'/'vcenter'/'vmax' (optional): color normalization bounds
  - 'colormap' (optional): Matplotlib colormap name (default 'RdYlGn')
  - 'marker'/'linestyle'/'color'/'markersize': line and marker styling

Example:
    import tkinter as tk
    import pandas as pd
    from src.view.tk_plotter import TkTimeSeriesPlotter

    root = tk.Tk()
    df = pd.DataFrame({"close": [1,2,3,2,1]})
    plotter = TkTimeSeriesPlotter(root, df, n=100,
        plot_params=[{"column": "close", "color": "black"}])
    plotter.pack(fill="both", expand=True)
    root.mainloop()
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


Number = Union[int, float]


class TkTimeSeriesPlotter:
    def __init__(
        self,
        parent: Optional[tk.Misc],
        df: pd.DataFrame,
        n: int = 500,
        plot_params: Optional[List[Dict[str, Any]]] = None,
        trades: Optional[Sequence[Any]] = None,
        custom_function: Optional[callable] = None,
        title: str = "Time Series Plot",
        figsize: tuple[Number, Number] = (10, 5),
    ) -> None:
        self._own_toplevel = False
        if parent is None:
            self.root: Union[tk.Toplevel, ttk.Frame] = tk.Toplevel()
            self.root.title(title)
            self._own_toplevel = True
        else:
            self.root = ttk.Frame(parent)

        self.df = df.copy()
        self.n = int(n)
        self.start_index = 0
        self.plot_params = list(plot_params or [])
        self.trades = list(trades or [])
        self.custom_function = custom_function

        self._build_ui(figsize)
        self._plot_data(self.start_index, self.start_index + self.n)

    # ---------- Public embedding helpers ----------
    def pack(self, *args, **kwargs) -> None:
        self.root.pack(*args, **kwargs)

    def grid(self, *args, **kwargs) -> None:
        self.root.grid(*args, **kwargs)

    def place(self, *args, **kwargs) -> None:
        self.root.place(*args, **kwargs)

    def destroy(self) -> None:
        try:
            if isinstance(self.root, tk.Toplevel):
                self.root.destroy()
            else:
                self.root.destroy()
        except Exception:
            pass

    # ---------- Public API ----------
    def update_df(self, new_df: pd.DataFrame) -> None:
        self.df = new_df.copy()
        self.start_index = 0
        self._refresh_plot()

    def set_plot_params(self, plot_params: List[Dict[str, Any]]) -> None:
        self.plot_params = list(plot_params)
        self._refresh_plot()

    # ---------- UI and plotting ----------
    def _build_ui(self, figsize: tuple[Number, Number]) -> None:
        # Controls frame
        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Label(ctrl, text="Start").grid(row=0, column=0, sticky=tk.W)
        self.var_start = tk.IntVar(value=0)
        ttk.Entry(ctrl, textvariable=self.var_start, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="Len (n)").grid(row=0, column=2, sticky=tk.W, padx=(12, 2))
        self.var_len = tk.IntVar(value=self.n)
        ttk.Entry(ctrl, textvariable=self.var_len, width=10).grid(row=0, column=3, sticky=tk.W)

        ttk.Button(ctrl, text="Anterior", command=self._on_prev).grid(row=0, column=4, padx=(12, 2))
        ttk.Button(ctrl, text="Próximo", command=self._on_next).grid(row=0, column=5, padx=(2, 12))
        ttk.Button(ctrl, text="Refresh", command=self._refresh_plot).grid(row=0, column=6)

        if self.custom_function is not None:
            ttk.Button(ctrl, text="Actualizar DF", command=self._on_custom).grid(row=0, column=7, padx=(12, 0))

        # Figure frame
        figfrm = ttk.Frame(self.root)
        figfrm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig: Figure = Figure(figsize=figsize, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figfrm)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        try:
            NavigationToolbar2Tk(self.canvas, figfrm)
        except Exception:
            pass

        # Resize behavior
        try:
            figfrm.columnconfigure(0, weight=1)
            figfrm.rowconfigure(0, weight=1)
        except Exception:
            pass

    def _safe_get_date(self, df: pd.DataFrame) -> pd.Series:
        if 'Date' in df.columns:
            return df['Date']
        idx = df.index
        try:
            if hasattr(idx, "to_pydatetime"):
                return pd.Series(idx.to_pydatetime(), index=df.index)
        except Exception:
            pass
        return pd.Series(idx, index=df.index)

    def _normalize_like(self, base: pd.Series, ref: pd.Series) -> pd.Series:
        try:
            bmin, bmax = float(base.min()), float(base.max())
            rmin, rmax = float(ref.min()), float(ref.max())
            if bmax == bmin:
                return pd.Series((rmax - rmin) / 2.0, index=base.index)
            scaled = (base - bmin) / max(1e-12, (bmax - bmin))
            return scaled * (rmax - rmin)
        except Exception:
            return base

    def _scatter_or_line(self, x: pd.Series, y: pd.Series, color_data: pd.Series, params: Dict[str, Any]) -> None:
        dynamic_color = bool(params.get('dynamic_color', False))
        label = params.get('label')
        # Build default label from column values
        if label is None:
            last_orig = None
            try:
                if len(color_data) > 0:
                    last_orig = float(color_data.iloc[-1])
            except Exception:
                last_orig = None
            label = str(params.get('column', 'series'))
            if last_orig is not None and np.isfinite(last_orig):
                label = f"{label} (Último: {last_orig:.2f})"

        if dynamic_color:
            from matplotlib.colors import Normalize, TwoSlopeNorm
            vmin = params.get('vmin', float(np.nanmin(color_data)))
            vmax = params.get('vmax', float(np.nanmax(color_data)))
            vcenter = params.get('vcenter', 0.0)
            try:
                vmin = float(vmin)
                vmax = float(vmax)
                vcenter = float(vcenter)
            except Exception:
                pass
            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6
            if vmin < vcenter < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            cmap_name = params.get('colormap', 'RdYlGn')
            cmap = matplotlib.cm.get_cmap(cmap_name)
            try:
                colors = cmap(norm(pd.to_numeric(color_data, errors='coerce')))
            except Exception:
                colors = cmap(norm(pd.Series(np.zeros(len(color_data)), index=color_data.index)))
            self.ax.scatter(
                x,
                y,
                c=colors,
                s=float(params.get('markersize', 20)),
                label=label,
            )
        else:
            self.ax.plot(
                x,
                y,
                marker=str(params.get('marker', '')),
                color=str(params.get('color', 'blue')),
                markersize=float(params.get('markersize', 5)),
                linestyle=str(params.get('linestyle', '-')),
                label=label,
            )

    def _plot_data(self, start: int, end: int) -> None:
        self.ax.clear()
        df_copy = self.df.copy()
        if 'Date' not in df_copy.columns:
            try:
                df_copy['Date'] = df_copy.index
            except Exception:
                df_copy['Date'] = np.arange(len(df_copy))

        start_i = max(0, int(start))
        end_i = min(int(end), len(df_copy))
        if end_i <= start_i:
            end_i = min(len(df_copy), start_i + max(1, int(self.var_len.get() if hasattr(self, 'var_len') else self.n)))

        df_visible = df_copy.iloc[start_i:end_i]
        x_series = self._safe_get_date(df_visible)

        for params in self.plot_params:
            col = params.get('column')
            if col is None or col not in df_visible.columns:
                continue
            color_data = pd.to_numeric(df_visible[col], errors='coerce')
            if 'overlay_on' in params and params['overlay_on'] in df_visible.columns:
                y_values = pd.to_numeric(df_visible[params['overlay_on']], errors='coerce')
            else:
                y_values = color_data

            normalize_with = params.get('normalize_with')
            if normalize_with is not None and normalize_with in df_visible.columns:
                ref = pd.to_numeric(df_visible[normalize_with], errors='coerce')
                color_data = self._normalize_like(color_data, ref)
                y_values = self._normalize_like(y_values, ref)

            self._scatter_or_line(x_series, y_values, color_data, params)

        self.ax.grid(True)
        try:
            self.ax.legend()
        except Exception:
            pass
        try:
            self.ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
        try:
            self.fig.autofmt_xdate()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _refresh_plot(self) -> None:
        try:
            self.n = max(1, int(self.var_len.get()))
        except Exception:
            self.n = max(1, int(self.n))
        try:
            self.start_index = max(0, int(self.var_start.get()))
        except Exception:
            self.start_index = max(0, int(self.start_index))
        self._plot_data(self.start_index, self.start_index + self.n)

    def _on_prev(self) -> None:
        step = max(1, int(self.var_len.get())) if hasattr(self, 'var_len') else max(1, int(self.n))
        self.start_index = max(0, self.start_index - step)
        try:
            self.var_start.set(self.start_index)
        except Exception:
            pass
        self._plot_data(self.start_index, self.start_index + step)

    def _on_next(self) -> None:
        step = max(1, int(self.var_len.get())) if hasattr(self, 'var_len') else max(1, int(self.n))
        self.start_index = min(max(0, len(self.df) - step), self.start_index + step)
        try:
            self.var_start.set(self.start_index)
        except Exception:
            pass
        self._plot_data(self.start_index, self.start_index + step)

    def _on_custom(self) -> None:
        if callable(self.custom_function):
            try:
                self.custom_function()
            except Exception:
                pass
            self._refresh_plot()


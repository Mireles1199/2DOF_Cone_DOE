#!/usr/bin/env python
# coding: utf-8
"""
DOE Selector
============
Single window with a generic case selector (multi-column Treeview) and
embedded plots (FigureCanvasTkAgg).

Left area : Treeview with one column for each var_val variable.

                 Columns configurable via "Columns…".
                 Free text filter. Sorted by column (click header).
                 Multi-selection (Ctrl / Shift / Select All).
Right area    : 2 stacked subplots (Axial_disp on top, Axial_vel on bottom)
                 embedded in the same window.

Usage:
    python doe_selector.py
    python doe_selector.py --doe_name DOE_Influence_dt
"""

import os
import sys
import argparse

# ── Backend BEFORE any import of pyplot / doe_plotter ──────────────
import matplotlib
matplotlib.use('TkAgg')

# ── Reuse logic and config from doe_plotter ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doe_plotter import (
    load_results,
    configurar_estilo_global,
    LABEL_KEY,
    SIGNALS,
    SIGNAL_YLABELS,
    DOE_NAME,
    SCRIPT_DIR,
    DECIMATE,
    color_azul,
    color_orange,
    plot_convergence,
    plot_convergence_error_ref,
    plot_convergence_error_consec,
    plot_convergence_time,
)

# ── Matplotlib ──────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Tkinter ────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox

configurar_estilo_global()

# ── DOE plot options ────────────────────────────────────────────────────────────────────────
_DOE_PLOT_ENTRIES = [
    ("Convergencia RMS — disp",      "plot_convergence",              ("Axial_disp", "rms")),
    ("Convergencia RMS — vel",       "plot_convergence",              ("Axial_vel",  "rms")),
    ("Convergencia Max — disp",      "plot_convergence",              ("Axial_disp", "max")),
    ("Convergencia Max — vel",       "plot_convergence",              ("Axial_vel",  "max")),
    ("Error más fino RMS — disp",    "plot_convergence_error_ref",    ("Axial_disp", "rms")),
    ("Error más fino RMS — vel",     "plot_convergence_error_ref",    ("Axial_vel",  "rms")),
    ("Error más fino Max — disp",    "plot_convergence_error_ref",    ("Axial_disp", "max")),
    ("Error más fino Max — vel",     "plot_convergence_error_ref",    ("Axial_vel",  "max")),
    ("Ganancia RMS — disp",          "plot_convergence_error_consec", ("Axial_disp", "rms")),
    ("Ganancia RMS — vel",           "plot_convergence_error_consec", ("Axial_vel",  "rms")),
    ("Ganancia Max — disp",          "plot_convergence_error_consec", ("Axial_disp", "max")),
    ("Ganancia Max — vel",           "plot_convergence_error_consec", ("Axial_vel",  "max")),
    ("Tiempo de ejecución",          "plot_convergence_time",         ()),
]
_DOE_PLOT_LABELS = [e[0] for e in _DOE_PLOT_ENTRIES]


# ===========================================================================
# Helpers
# ===========================================================================

def _all_var_keys(cases: list) -> list:
    """Ordered list of all var_val keys present in the cases.
    LABEL_KEY appears first; the rest, sorted alphabetically."""
    keys: set = set()
    for c in cases:
        keys.update(c["var_val"].keys())
    ordered = []
    if LABEL_KEY in keys:
        ordered.append(LABEL_KEY)
        keys.discard(LABEL_KEY)
    ordered.extend(sorted(keys))
    return ordered


def _fmt_val(v) -> str:
    """Formats a var_val value for the table."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if fv == int(fv) and abs(fv) < 1e9:
            return str(int(fv))
        return f"{fv:.4g}"
    except (TypeError, ValueError):
        return str(v)


def _col_header(key: str) -> str:
    """Readable column name (without $)."""
    return key.replace("$", "")


# ===========================================================================
# Column selection dialog
# ===========================================================================

class ColumnsDialog(tk.Toplevel):
    """Modal window — checkboxes to choose which columns to display."""

    def __init__(self, parent: tk.Tk, all_keys: list, visible_keys: list) -> None:
        super().__init__(parent)
        self.title("Visible Columns")
        self.resizable(False, False)
        self.grab_set()
        self._result = None
        self._vars: dict[str, tk.BooleanVar] = {}

        ttk.Label(
            self, text="Visible columns in the table:",
            font=("Arial", 10, "bold"),
        ).pack(padx=14, pady=(10, 4), anchor=tk.W)

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, padx=14, pady=4)

        # Special header "case" (always visible, not uncheckable)
        ttk.Label(frm, text="case  (fixed)", foreground="#888888").pack(
            anchor=tk.W, pady=1)

        for key in all_keys:
            var = tk.BooleanVar(value=(key in visible_keys))
            self._vars[key] = var
            display = "dt (µs)" if key == "dt_us" else _col_header(key)
            ttk.Checkbutton(frm, text=display, variable=var).pack(
                anchor=tk.W, pady=1)

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=14, pady=(6, 10))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Apply", command=self._apply).pack(
            side=tk.RIGHT, padx=4)

        # Center over the parent window
        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{px}+{py}")

    def _apply(self) -> None:
        self._result = [k for k, v in self._vars.items() if v.get()]
        self.destroy()

    @property
    def result(self):
        return self._result


# ===========================================================================
# Main application
# ===========================================================================

class DoeSelectorApp:
    """Main window: generic Treeview + 2 embedded subplots."""

    _LEFT_WIDTH = 400

    def __init__(self, root: tk.Tk, cases: list, doe_name: str) -> None:
        self.root      = root
        self.cases     = cases
        self.doe_name  = doe_name
        self.lk        = LABEL_KEY.replace("$", "")
        self._cbar     = None
        self._sort_col: str | None = None
        self._sort_rev: bool       = False

        # Column state
        self._all_keys     = _all_var_keys(cases)     # all var_val keys
        self._has_dt       = LABEL_KEY == "$nb_dt_rev$"
        self._visible_keys = list(self._all_keys)     # visible var_val keys
        self._show_dt      = self._has_dt             # dt_us column visible

        # Mapping iid → case for selection
        self._iid_to_case: dict[str, dict] = {}

        # DOE figure references (for cleanup)
        self._doe_fig_top: object = None
        self._doe_fig_bot: object = None

        root.title(f"DOE Selector — {doe_name}")
        root.minsize(1100, 600)
        root.state("zoomed")

        self._build_layout()
        self._build_left_panel()
        self._build_right_panel()
        self._build_doe_panel()

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        self.paned = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief=tk.RAISED,
        )
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.left_frame  = ttk.Frame(self.paned)
        self.right_frame = ttk.Frame(self.paned)
        self.doe_frame   = ttk.Frame(self.paned)
        self.paned.add(self.left_frame,  minsize=260, width=self._LEFT_WIDTH)
        self.paned.add(self.right_frame, minsize=480, width=900)
        self.paned.add(self.doe_frame,   minsize=150, width=300)

    # ── Left panel ─────────────────────────────────────────────────────────
    def _build_left_panel(self) -> None:
        lf = self.left_frame

        # Header
        ttk.Label(lf, text=self.doe_name,
                  font=("Arial", 11, "bold")).pack(
            anchor=tk.W, padx=8, pady=(8, 0))
        ttk.Label(
            lf,
            text=f"{len(self.cases)} cases   ·   {len(self._all_keys)} variables",
            font=("Arial", 9), foreground="#555555",
        ).pack(anchor=tk.W, padx=8, pady=(0, 4))

        # Search bar + Columns button
        bar = ttk.Frame(lf)
        bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._filter_var = tk.StringVar()
        self._filter_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(bar, textvariable=self._filter_var,
                  font=("Arial", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bar, text=" 🔍", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Columnas…",
                   command=self._show_columns_dialog).pack(side=tk.LEFT, padx=(6, 0))

        # Treeview container (recreated when columns change)
        self._tree_frame = ttk.Frame(lf)
        self._tree_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        self._build_tree()

        # Buttons
        btn = ttk.Frame(lf)
        btn.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn, text="Select All",
                   command=self._select_all).pack(fill=tk.X, pady=1)
        ttk.Button(btn, text="Clear Selection",
                   command=self._clear_sel).pack(fill=tk.X, pady=1)
        ttk.Separator(btn).pack(fill=tk.X, pady=4)
        ttk.Button(btn, text="Plot  ▶",
                   command=self._plot).pack(fill=tk.X, pady=1, ipady=3)
        ttk.Button(btn, text="Clear Plot",
                   command=self._clear_plot).pack(fill=tk.X, pady=1)

    # ── Treeview ───────────────────────────────────────────────────────────
    def _build_tree(self) -> None:
        """Create / recreate the Treeview with the currently visible columns."""
        # Destroy previous widgets if they exist
        for attr in ("tree", "_sb_y", "_sb_x"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass

        # Active columns: "case" + visible var_val keys + dt_us (if applicable)
        cols = ["case"] + self._visible_keys
        if self._show_dt:
            cols.append("dt_us")
        self._cols = cols

        # Grid inside _tree_frame
        tf = self._tree_frame
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                 selectmode="extended")

        for col in cols:
            if col == "case":
                hdr, w = "case", 65
            elif col == "dt_us":
                hdr, w = "dt (µs)", 75
            else:
                hdr, w = _col_header(col), 90
            self.tree.heading(col, text=hdr,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, minwidth=40, anchor=tk.CENTER,
                             stretch=True)

        self._sb_y = ttk.Scrollbar(tf, orient=tk.VERTICAL,
                                   command=self.tree.yview)
        self._sb_x = ttk.Scrollbar(tf, orient=tk.HORIZONTAL,
                                   command=self.tree.xview)
        self.tree.configure(yscrollcommand=self._sb_y.set,
                            xscrollcommand=self._sb_x.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self._sb_y.grid(row=0, column=1, sticky="ns")
        self._sb_x.grid(row=1, column=0, sticky="ew")

        self.tree.bind("<Double-1>", lambda _e: self._plot())

        # Populate with the current cases (respecting the active filter)
        self._populate_tree(self._filtered_cases())

    def _populate_tree(self, cases_to_show: list) -> None:
        """Clear and insert rows in the Treeview."""
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._iid_to_case.clear()

        for c in cases_to_show:
            row = []
            for col in self._cols:
                if col == "case":
                    row.append(c["group"])
                elif col == "dt_us":
                    row.append(f"{c['dt_us']:.2f}" if c["dt_us"] is not None else "—")
                else:
                    row.append(_fmt_val(c["var_val"].get(col)))
            iid = self.tree.insert("", tk.END, values=row)
            self._iid_to_case[iid] = c

    # ── Sorting ───────────────────────────────────────────────────────────
    def _sort_by(self, col: str) -> None:
        reverse = (self._sort_col == col) and not self._sort_rev
        self._sort_col = col
        self._sort_rev = reverse

        rows = [(self.tree.set(iid, col), iid)
                for iid in self.tree.get_children()]

        def _key(item):
            try:
                return (0, float(item[0]))
            except (ValueError, TypeError):
                return (1, str(item[0]))

        rows.sort(key=_key, reverse=reverse)
        for idx, (_, iid) in enumerate(rows):
            self.tree.move(iid, "", idx)

        # Update arrows in the headers
        for c in self._cols:
            hdr = "case" if c == "case" else ("dt (µs)" if c == "dt_us" else _col_header(c))
            arrow = (" ▼" if reverse else " ▲") if c == col else ""
            self.tree.heading(c, text=hdr + arrow,
                              command=lambda cc=c: self._sort_by(cc))

    # ── Filtering ───────────────────────────────────────────────────────────
    def _filtered_cases(self) -> list:
        query = self._filter_var.get().strip().lower() if hasattr(self, "_filter_var") else ""
        if not query:
            return self.cases
        return [c for c in self.cases if self._case_matches(c, query)]

    def _case_matches(self, c: dict, query: str) -> bool:
        tokens = [c["group"].lower()]
        for v in c["var_val"].values():
            tokens.append(_fmt_val(v).lower())
        if c["dt_us"] is not None:
            tokens.append(f"{c['dt_us']:.2f}")
        return any(query in t for t in tokens)

    def _apply_filter(self) -> None:
        self._populate_tree(self._filtered_cases())

    # ── Columns dialog ─────────────────────────────────────────────────────
    def _show_columns_dialog(self) -> None:
        all_keys = self._all_keys + (["dt_us"] if self._has_dt else [])
        visible  = list(self._visible_keys) + (["dt_us"] if self._show_dt else [])
        dlg = ColumnsDialog(self.root, all_keys, visible)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self._show_dt      = "dt_us" in dlg.result
            self._visible_keys = [k for k in dlg.result if k != "dt_us"]
            self._build_tree()

    # ── Selection ──────────────────────────────────────────────────────────
    def _select_all(self) -> None:
        self.tree.selection_set(self.tree.get_children())

    def _clear_sel(self) -> None:
        self.tree.selection_remove(self.tree.get_children())

    # ── Legend label ───────────────────────────────────────────────────────
    def _case_label(self, c: dict) -> str:
        """Constructs the legend label with the values of the currently visible columns
        in the Treeview (same info as the selected row)."""
        parts = []
        for col in self._cols:
            if col == "case":
                parts.append(c["group"])
            elif col == "dt_us":
                parts.append(f"dt={c['dt_us']:.2f}\u00b5s" if c["dt_us"] is not None else "dt=—")
            else:
                v = c["var_val"].get(col)
                parts.append(f"{_col_header(col)}={_fmt_val(v)}")
        return "  ".join(parts)

    # ── Right panel (embedded matplotlib) ─────────────────────────────────
    def _build_right_panel(self) -> None:
        rf = self.right_frame
        self.fig     = Figure(constrained_layout=True)
        self.ax_disp = self.fig.add_subplot(2, 1, 1)
        self.ax_vel  = self.fig.add_subplot(2, 1, 2, sharex=self.ax_disp)
        self._init_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=rf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, rf, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.draw()

    # ── DOE panel (right, 2 independent figure slots) ──────────────────────
    def _build_doe_panel(self) -> None:
        """Right panel: 2 vertically stacked DOE figure slots, each with
        an independent combobox to choose the figure type.
        A vertical PanedWindow guarantees both slots have exactly the same height.
        """
        df = self.doe_frame

        # Vertical sash: both slots get equal height, user can drag to adjust
        self._doe_vpaned = tk.PanedWindow(
            df, orient=tk.VERTICAL, sashwidth=5, sashrelief=tk.RAISED,
        )
        self._doe_vpaned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        _top_pane = ttk.Frame(self._doe_vpaned)
        _bot_pane = ttk.Frame(self._doe_vpaned)
        self._doe_vpaned.add(_top_pane, stretch="always")
        self._doe_vpaned.add(_bot_pane, stretch="always")

        def _make_slot(parent, default_label, slot: str):
            zone = ttk.LabelFrame(parent, text="  DOE  ", padding=4)
            zone.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

            # ── combobox + refresh button ──
            hdr = ttk.Frame(zone)
            hdr.pack(fill=tk.X, pady=(0, 3))
            combo = ttk.Combobox(hdr, values=_DOE_PLOT_LABELS,
                                 state="readonly", font=("Arial", 9))
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            combo.set(default_label)
            ttk.Button(
                hdr, text="▶", width=3,
                command=lambda s=slot: self._refresh_doe_plot(s),
            ).pack(side=tk.LEFT, padx=(4, 0))

            # ── canvas area ──
            canvas_frame = ttk.Frame(zone)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(
                canvas_frame,
                text="Presiona ▶\no primero  Plot ▶  un caso",
                foreground="#999999", font=("Arial", 9),
                anchor=tk.CENTER, justify=tk.CENTER,
            ).pack(expand=True)

            # ── toolbar area ──
            toolbar_frame = ttk.Frame(zone)
            toolbar_frame.pack(fill=tk.X)

            return combo, canvas_frame, toolbar_frame

        (self._doe_combo_top,
         self._doe_canvas_frame_top,
         self._doe_toolbar_frame_top) = _make_slot(
            _top_pane, "Convergencia RMS — disp", "top")

        (self._doe_combo_bot,
         self._doe_canvas_frame_bot,
         self._doe_toolbar_frame_bot) = _make_slot(
            _bot_pane, "Error más fino RMS — disp", "bot")

        # Auto-refresh when combobox selection changes
        self._doe_combo_top.bind("<<ComboboxSelected>>",
                                  lambda _: self._refresh_doe_plot("top"))
        self._doe_combo_bot.bind("<<ComboboxSelected>>",
                                  lambda _: self._refresh_doe_plot("bot"))

    def _refresh_doe_plot(self, slot: str) -> None:
        """Renders the selected DOE figure in the given slot ('top' or 'bot').
        All cases plotted (full curve); selected cases highlighted with
        vertical orange dashed lines.
        """
        combo         = self._doe_combo_top  if slot == "top" else self._doe_combo_bot
        canvas_frame  = self._doe_canvas_frame_top  if slot == "top" else self._doe_canvas_frame_bot
        toolbar_frame = self._doe_toolbar_frame_top if slot == "top" else self._doe_toolbar_frame_bot
        fig_attr      = f"_doe_fig_{slot}"

        choice = combo.get()
        fn_map = {label: (fn_name, args) for label, fn_name, args in _DOE_PLOT_ENTRIES}
        entry  = fn_map.get(choice)
        if not entry:
            return

        fn_name, fn_args = entry
        _fn_registry = {
            "plot_convergence":              plot_convergence,
            "plot_convergence_error_ref":    plot_convergence_error_ref,
            "plot_convergence_error_consec": plot_convergence_error_consec,
            "plot_convergence_time":         plot_convergence_time,
        }
        fig = _fn_registry[fn_name](self.cases, *fn_args)

        # Resize to square-ish (3.4 × 3.4 in) so the embedded canvas
        # doesn't request an oversized widget from the geometry manager
        if fig is not None:
            fig.set_size_inches(3.4, 3.4)

        if fig is None:
            # No timing data
            fig = Figure(constrained_layout=True)
            ax  = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "Sin datos de timing\n(ejecutar con --timed)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#666666")
            ax.axis("off")

        # ── Alpha highlight: dim all, only selected at full alpha ──────────
        if fig.axes:
            ax_main = fig.axes[0]
            # Dim every existing artist to background
            for artist in list(ax_main.lines) + list(ax_main.collections):
                artist.set_alpha(0.20)
            # Determine selected x positions
            sel_iids       = self.tree.selection()
            selected_cases = [self._iid_to_case[iid] for iid in sel_iids
                              if iid in self._iid_to_case]
            use_dt = (LABEL_KEY == "$nb_dt_rev$")
            sel_x  = {c["dt_us"] if use_dt else c["label_val"]
                      for c in selected_cases
                      if (c["dt_us"] if use_dt else c["label_val"]) is not None}
            # Match selected x values to main polyline (ax.lines[0])
            if ax_main.lines and sel_x:
                lx = np.asarray(ax_main.lines[0].get_xdata(), dtype=float)
                ly = np.asarray(ax_main.lines[0].get_ydata(), dtype=float)
                sel_xv, sel_yv = [], []
                for xs in sel_x:
                    idx = int(np.argmin(np.abs(lx - float(xs))))
                    if np.isclose(lx[idx], float(xs), rtol=1e-4):
                        sel_xv.append(lx[idx])
                        sel_yv.append(ly[idx])
                if sel_xv:
                    ax_main.scatter(sel_xv, sel_yv, color=color_orange,
                                    alpha=1.0, zorder=5, s=90,
                                    linewidths=0.6, edgecolors="white")

        # Remove from pyplot manager (prevents duplicate Tk windows)
        plt.close(fig)

        # Replace stored figure reference (close old one first)
        old_fig = getattr(self, fig_attr, None)
        if old_fig is not None:
            try:
                plt.close(old_fig)
            except Exception:
                pass
        setattr(self, fig_attr, fig)

        # ── Destroy previous canvas / toolbar widgets ───────────────────
        for w in list(canvas_frame.winfo_children()):
            w.destroy()
        for w in list(toolbar_frame.winfo_children()):
            w.destroy()

        # ── Embed new figure ───────────────────────────────────────────────
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    def _init_axes(self) -> None:
        self.ax_disp.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp."), fontsize=16)
        self.ax_disp.grid(True, linestyle="--", alpha=0.4)
        self.ax_disp.tick_params(labelbottom=False, labelsize=14)
        self.ax_vel.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel."), fontsize=16)
        self.ax_vel.set_xlabel("Time (s)", fontsize=16)
        self.ax_vel.grid(True, linestyle="--", alpha=0.4)
        self.ax_vel.tick_params(labelbottom=True, labelsize=14)
        self.fig.suptitle("Select cases and press  Plot  ▶")

    # ── Acciones de gráfica ────────────────────────────────────────────────
    def _remove_cbar(self) -> None:
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

    def _clear_plot(self) -> None:
        self._remove_cbar()
        self.ax_disp.cla()
        self.ax_vel.cla()
        self._init_axes()
        self.canvas.draw()

    def _plot(self) -> None:
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning(
                "No selection",
                "Select at least one case in the table.",
                parent=self.root,
            )
            return

        selected = [self._iid_to_case[iid] for iid in sel_iids
                    if iid in self._iid_to_case]
        if not selected:
            return

        # Colormap basado en LABEL_KEY
        vals = [c["label_val"] for c in selected if c["label_val"] is not None]
        use_cmap = len(vals) >= 2
        if use_cmap:
            norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
            cmap = cm.viridis

        self._remove_cbar()
        self.ax_disp.cla()
        self.ax_vel.cla()

        axes_map = {"Axial_disp": self.ax_disp, "Axial_vel": self.ax_vel}

        for c in selected:
            nb  = c["label_val"]
            clr = cmap(norm(nb)) if use_cmap else color_azul
            lbl = c["group"]

            for sig, ax in axes_map.items():
                data = c.get(sig)
                if data is None:
                    continue
                t, y = data
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=clr, linewidth=1.0, alpha=0.85, label=lbl, rasterized=True)

        _sci_fmt = mticker.FuncFormatter(lambda x, _: f"{x:.2e}")
        self.ax_disp.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp."), fontsize=16)
        self.ax_disp.grid(True, linestyle="--", alpha=0.4)
        self.ax_disp.tick_params(labelbottom=False, labelsize=14)
        # self.ax_disp.yaxis.set_major_formatter(_sci_fmt)
        self.ax_disp.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


        self.ax_vel.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel."), fontsize=16)
        self.ax_vel.set_xlabel("Time (s)", fontsize=16)
        self.ax_vel.grid(True, linestyle="--", alpha=0.4)
        # self.ax_vel.yaxis.set_major_formatter(_sci_fmt)
        self.ax_vel.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_vel.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.3g}"))

        n = len(selected)
        if n <= 12:
            self.ax_disp.legend(fontsize=10, framealpha=0.7)
            self.ax_vel.legend(fontsize=10, framealpha=0.7)

        if use_cmap:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self._cbar = self.fig.colorbar(
                sm, ax=[self.ax_disp, self.ax_vel],
                label=LABEL_KEY, shrink=0.9,
                orientation="horizontal", pad=0.05,
            )
            self._cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.2e}")
            self._cbar.update_ticks()

        self.fig.suptitle(f"{self.lk}  —  {n} case(s) selected")
        self.canvas.draw()

        # Refresh DOE panels with updated selection highlight
        self._refresh_doe_plot("top")
        self._refresh_doe_plot("bot")


# ===========================================================================
def parse_args():
    epilog = """
DOE Selector — Interactive visualization of DOE cases
======================================================
Single window: Treeview of cases (left) + 2 embedded plots (right).

TREEVIEW CONTROLS
-----------------
  Single click       → select row
  Ctrl + click       → add/remove row from selection
  Shift + click      → select range of rows
  Double click        → plot directly
  Click on column   → sort ▲/▼ by that column
  Search field 🔍          → filter rows in real-time (free text)

BUTTONS
-------
  Select All        → mark all visible cases
  Clear Selection   → unmark all
  Plot ▶            → draw the selected signals in the right panel
  Clear Plot        → clear the plot panel
  Columns…          → choose which var_val variables to display as columns

EXAMPLES
--------
  python doe_selector.py
      Opens DOE_NAME configured in doe_plotter.py (default: DOE_4)

  python doe_selector.py --doe_name DOE_Influence_dt
      Opens the DOE in the folder DOE_Influence_dt/
"""
    parser = argparse.ArgumentParser(
        description="DOE Selector — Nessy2m",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--doe_name", default=None,
        help="Name of the DOE to visualize (overrides DOE_NAME in config)",
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    doe_name = args.doe_name or DOE_NAME
    h5_path  = os.path.join(SCRIPT_DIR, doe_name, "doe_results.h5")

    print(f"[INFO] Loading: {h5_path}")
    cases = load_results(h5_path)
    print(f"[INFO] {len(cases)} cases available")
    for c in cases:
        vv = "  ".join(f"{_col_header(k)}={_fmt_val(v)}"
                       for k, v in c["var_val"].items())
        print(f"       {c['group']}   {vv}")

    root = tk.Tk()
    DoeSelectorApp(root, cases, doe_name)
    root.mainloop()


if __name__ == "__main__":
    main()

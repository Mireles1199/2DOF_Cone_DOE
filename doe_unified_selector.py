#!/usr/bin/env python
# coding: utf-8
"""
doe_unified_selector.py — Selector interactivo unificado para todos los HDF5 DOE.
===================================================================================
Soporta 5 formatos de archivos .h5:

  1. doe_results.h5               — señales (Axial_disp, Axial_vel) por caso DOE
  2. doe_noise_results.h5         — señales degradadas por nivel de ruido SNR
  3. doe_indicator_results.h5     — indicadores + señales por caso DOE
  4. doe_noise_indicator_results.h5 — indicadores por nivel de ruido
  5. doe_model_snr_results.h5     — SNR_mod_dB por señal y caso DOE

Layout de la ventana:
  [Tabla (izq)] | [Señales / I_t (centro)] | [Summary plots (dcha)]

Uso:
    python doe_unified_selector.py
    python doe_unified_selector.py --h5 PATH/doe_indicator_results.h5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional

# ── Backend BEFORE any pyplot import ──────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")

import h5py
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ── Import from existing plotters ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# doe_plotter convergence functions (return Figure)
from doe_plotter import (
    configurar_estilo_global as _cfg_estilo,
    SIGNAL_YLABELS,
    color_azul,
    color_orange,
    plot_convergence,
    plot_convergence_error_ref,
    plot_convergence_error_consec,
    plot_convergence_time,
)

# doe_indicator_plotter functions (some create plt figures, don't return them)
from doe_indicator_plotter import (
    load_indicator_results,
    _plot_td_single,
    _plot_td_per_run,
    plot_It_overlay,
    _resolve_label_key,
    _x_values as _ind_x_values,
    T_GT as _IND_T_GT,
    _RUN_COLORS,
    DECIMATE as _IND_DECIMATE,
    _sanitize,
)

# doe_model_snr_plotter functions
from doe_model_snr_plotter import (
    load_snr_results,
    plot_snr_vs_param,
    _detect_param_key as _snr_detect_param_key,
)

# doe_noise_plotter functions (for TYPE_NOISE_IND right panel)
from doe_noise_plotter import (
    gather_detection_rows        as _noise_gather_df,
    plot_td_for_indicator        as _noise_plot_td_ind,
    plot_td_both_for_indicator   as _noise_plot_td_both,
    plot_td_lollipop             as _noise_plot_lollipop,
    plot_delay_vs_snr            as _noise_plot_delay,
    plot_far_cost_vs_snr         as _noise_plot_far_cost,
    gather_indicator_curves      as _noise_gather_curves,
    plot_it_overlay              as _noise_plot_it_overlay,
)

_cfg_estilo()

# ── Format type constants ──────────────────────────────────────────────────────
TYPE_DOE_RESULTS   = "doe_results"
TYPE_DOE_NOISE     = "doe_noise"
TYPE_DOE_INDICATOR = "doe_indicator"
TYPE_NOISE_IND     = "doe_noise_ind"
TYPE_MODEL_SNR     = "doe_model_snr"

_TYPE_LABELS = {
    TYPE_DOE_RESULTS  : "DOE Results  (señales)",
    TYPE_DOE_NOISE    : "DOE Noise  (señales + SNR)",
    TYPE_DOE_INDICATOR: "DOE Indicator Results",
    TYPE_NOISE_IND    : "DOE Noise Indicators",
    TYPE_MODEL_SNR    : "DOE Model SNR",
}

DECIMATE = 1   # decimación para plots de señales en panel central


# ==============================================================================
# FORMAT DETECTION
# ==============================================================================

def detect_h5_type(h5_path: str) -> str:
    """Auto-detecta el tipo de formato de un HDF5 del DOE."""
    with h5py.File(h5_path, "r") as f:
        groups = list(f.keys())
        if not groups:
            return TYPE_DOE_RESULTS

        has_case_groups = any(g.startswith("case_") for g in groups)
        has_snr_groups  = any(g.startswith("snr_") or g == "control" for g in groups)

        first_grp   = f[groups[0]]
        first_attrs = dict(first_grp.attrs)

        # doe_model_snr: case_* con atributos SNR_mod_dB_*
        has_snr_attrs = any(k.startswith("SNR_mod_dB_") for k in first_attrs)
        if has_case_groups and has_snr_attrs:
            return TYPE_MODEL_SNR

        # Buscar subgrupos con t + I_t (indicadores) en el primer grupo
        has_run_subgroups = False
        if isinstance(first_grp, h5py.Group):
            for sub_name in list(first_grp.keys())[:8]:
                if sub_name in {"Axial_disp", "Axial_vel"}:
                    continue
                sub = first_grp[sub_name]
                if isinstance(sub, h5py.Group) and "t" in sub and "I_t" in sub:
                    has_run_subgroups = True
                    break

        # doe_indicator: case_* + subgrupos run
        if has_case_groups and has_run_subgroups:
            return TYPE_DOE_INDICATOR

        # doe_noise_ind: snr_*/control + subgrupos run
        if has_snr_groups and has_run_subgroups:
            return TYPE_NOISE_IND

        # doe_noise: snr_*/control sólo con señales
        if has_snr_groups:
            return TYPE_DOE_NOISE

        # Por defecto: doe_results (case_* con señales directas)
        return TYPE_DOE_RESULTS


# ==============================================================================
# NORMALIZED LOADERS
# ==============================================================================
# Estructura normalizada de cada case dict:
#   group     : str        — nombre del grupo HDF5
#   label_key : str        — clave DOE principal (eje X tabla)
#   label_val : float      — valor numérico del eje X
#   var_val   : dict       — todas las variables DOE del caso
#   signals   : dict       — {"Axial_disp": (t, y), "Axial_vel": (t, y)} o {}
#   forces    : dict       — {"res_R_p": (t, y)} o {}
#   runs      : dict       — {run_name: {t, I_t, t_d, t_d_no_FAR, attrs}} o {}
#   snr       : dict       — {"Axial_disp": float, ...} o {}
#   dt_us     : float|None — delta t en µs (solo para $nb_dt_rev$)
#   wall_time_s: float|None
# ==============================================================================

_SIGNAL_NAMES = {"Axial_disp", "Axial_vel"}
_FORCE_NAMES = {"res_R_p"}
_OUT_DEFLEX_GROUP = "Out_Deflex"
_OUT_DEFLEX_NAMES = {"Axial_disp_out_deflex", "Axial_vel_out_deflex"}

_LINE_TARGET_ALL = "Todas (tab activo)"
_LINE_TARGETS = [_LINE_TARGET_ALL, "Señales: disp", "Señales: vel",
                 "Fuerzas: F1", "Fuerzas: F2", "Fuerzas: F3",
                 "Deflex: disp", "Deflex: vel", "I_t"]
_LINE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#46f0f0", "#f032e6", "#9a6324", "#000075", "#808000"]


def _read_signals(grp: h5py.Group) -> Dict[str, Any]:
    """Lee señales de tiempo de un grupo HDF5 → {name: (t, y)}."""
    signals = {}
    for sig in _SIGNAL_NAMES:
        if sig in grp:
            sg = grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                signals[sig] = (sg["time"][()], sg["values"][()])
    return signals


def _read_forces(grp: h5py.Group) -> Dict[str, Any]:
    """Lee fuerzas de tiempo de un grupo HDF5 → {name: (t, y)}."""
    forces = {}
    for sig in _FORCE_NAMES:
        if sig in grp:
            sg = grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                forces[sig] = (sg["time"][()], sg["values"][()])
    return forces


def _read_out_deflex(grp: h5py.Group) -> Dict[str, Any]:
    """Lee señales de Out_Deflex de un grupo HDF5 → {name: (t, y)} o {}."""
    out = {}
    if _OUT_DEFLEX_GROUP not in grp:
        return out
    od_grp = grp[_OUT_DEFLEX_GROUP]
    if not isinstance(od_grp, h5py.Group):
        return out
    for sig in _OUT_DEFLEX_NAMES:
        if sig in od_grp:
            sg = od_grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                out[sig] = (sg["time"][()], sg["values"][()])
    return out


def _read_runs(grp: h5py.Group) -> Dict[str, Any]:
    """Lee subgrupos de run (indicadores) de un grupo HDF5."""
    runs = {}
    for rname in grp.keys():
        if rname in _SIGNAL_NAMES:
            continue
        rgrp = grp[rname]
        if not isinstance(rgrp, h5py.Group):
            continue
        if "t" not in rgrp or "I_t" not in rgrp:
            continue
        runs[rname] = {
            "t":          rgrp["t"][()]          if "t"          in rgrp else np.array([]),
            "I_t":        rgrp["I_t"][()]        if "I_t"        in rgrp else np.array([]),
            "t_d":        rgrp["t_d"][()]        if "t_d"        in rgrp else np.array([]),
            "t_d_no_FAR": rgrp["t_d_no_FAR"][()] if "t_d_no_FAR" in rgrp else np.array([]),
            "attrs":      dict(rgrp.attrs),
        }
    return runs


def _auto_label_key(var_val: dict) -> str:
    """Auto-detecta la clave DOE más informativa (la primera disponible)."""
    if var_val:
        return next(iter(var_val))
    return "case_idx"


def _best_label_key(cases: List[Dict]) -> str:
    """Elige la clave DOE con mayor variación *relativa* entre todos los casos
    (coeficiente de variación = spread / |media|).  Esto evita que variables
    con valores absolutos grandes (spin_rate ~12000) ganen sobre variables que
    cambian mucho en términos relativos (nb_dt_rev 1→8).  Si la media es 0 se
    usa el spread absoluto como fallback para esa clave."""
    all_keys: set = set()
    for c in cases:
        all_keys.update(c.get("var_val", {}).keys())

    best_key, best_score = "case_idx", -1.0
    for k in sorted(all_keys):
        vals = []
        for c in cases:
            try:
                vals.append(float(c["var_val"].get(k, float("nan"))))
            except (TypeError, ValueError):
                pass
        valid = [v for v in vals if np.isfinite(v)]
        if len(valid) < 2:
            continue
        spread = max(valid) - min(valid)
        if spread == 0.0:
            continue
        mean_abs = abs(np.mean(valid))
        # Relative spread (CV); fall back to absolute when mean ≈ 0
        score = spread / mean_abs if mean_abs > 1e-12 else spread
        if score > best_score:
            best_score, best_key = score, k
    return best_key


def load_doe_results(h5_path: str) -> List[Dict]:
    """Loader para doe_results.h5 (case_* + señales + $..$ attrs)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(k for k in f.keys() if k.startswith("case_")):
            grp      = f[grp_name]
            attrs    = dict(grp.attrs)
            var_val  = {k.strip("$"): v for k, v in attrs.items() if k != "wall_time_s"}
            wall_t   = float(attrs["wall_time_s"]) if "wall_time_s" in attrs else None
            lk       = _auto_label_key(var_val)
            try:
                lv = float(var_val.get(lk, float("nan")))
            except (TypeError, ValueError):
                lv = float("nan")
            cases.append({
                "group":       grp_name,
                "label_key":   lk,
                "label_val":   lv,
                "var_val":     var_val,
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "out_deflex":  _read_out_deflex(grp),
                "runs":        {},
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": wall_t,
                # Also copy signals as top-level keys for plot_convergence compat
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            # Make signals accessible at top level (doe_plotter compat)
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)

    # Re-normaliza label_key/label_val con la clave de mayor variación
    if cases:
        best_lk = _best_label_key(cases)
        for c in cases:
            c["label_key"] = best_lk
            try:
                c["label_val"] = float(c["var_val"].get(best_lk, float("nan")))
            except (TypeError, ValueError):
                c["label_val"] = float("nan")

    cases.sort(key=lambda c: c["label_val"] if not np.isnan(c["label_val"]) else float("inf"))
    return cases


def load_doe_noise(h5_path: str) -> List[Dict]:
    """Loader para doe_noise_results.h5 (control + snr_* + señales)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp   = f[grp_name]
            attrs = dict(grp.attrs)
            snr   = float(attrs.get("snr_db", 0.0)) if grp_name != "control" else float("inf")
            cases.append({
                "group":       grp_name,
                "label_key":   "snr_db",
                "label_val":   snr,
                "var_val":     {"snr_db": snr},
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "runs":        {},
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": None,
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)
    cases.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))
    return cases


def load_doe_indicator_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_indicator_results.h5 — reutiliza load_indicator_results
    y añade la clave 'signals' y compatibilidad top-level."""
    raw = load_indicator_results(h5_path)
    for c in raw:
        # load_indicator_results ya lee señales dentro del grupo como subgrupos
        # pero NO las pone en c["signals"]; las añadimos aquí
        c.setdefault("signals", {})
        c.setdefault("snr", {})
        c.setdefault("dt_us", None)
        c.setdefault("wall_time_s", None)
        # top-level compat
        c["Axial_disp"] = c["signals"].get("Axial_disp")
        c["Axial_vel"]  = c["signals"].get("Axial_vel")
    # Re-read signals from HDF5 (load_indicator_results doesn't read them)
    with h5py.File(h5_path, "r") as f:
        for c in raw:
            grp = f.get(c["group"])
            if grp is not None:
                sigs = _read_signals(grp)
                c["signals"] = sigs
                c["Axial_disp"] = sigs.get("Axial_disp")
                c["Axial_vel"]  = sigs.get("Axial_vel")

    # Re-normaliza label_key/label_val con la clave de mayor variación relativa
    # (el atributo guardado en el HDF5 puede apuntar a la variable equivocada
    # si el DOE fue generado con otra variable fija como spin_rate)
    if raw:
        best_lk = _best_label_key(raw)
        for c in raw:
            c["label_key"] = best_lk
            try:
                c["label_val"] = float(c["var_val"].get(best_lk, float("nan")))
            except (TypeError, ValueError):
                c["label_val"] = float("nan")
        raw.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))

    return raw


def load_noise_indicator_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_noise_indicator_results.h5 (control + snr_* + runs)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp   = f[grp_name]
            attrs = dict(grp.attrs)
            snr   = float(attrs.get("snr_db", 0.0)) if grp_name != "control" else float("inf")
            cases.append({
                "group":       grp_name,
                "label_key":   "snr_db",
                "label_val":   snr,
                "var_val":     {"snr_db": snr},
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "runs":        _read_runs(grp),
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": None,
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)
    cases.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))
    return cases


def load_model_snr_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_model_snr_results.h5 — reutiliza load_snr_results."""
    raw = load_snr_results(h5_path)
    for c in raw:
        # Normalizar: añadir claves estándar
        c.setdefault("signals", {})
        c.setdefault("runs", {})
        # label_key / label_val
        pk = _snr_detect_param_key(raw) or "case_idx"
        c["label_key"] = pk
        try:
            c["label_val"] = float(c["var_val"].get(pk, c.get("case_idx", float("nan"))))
        except (TypeError, ValueError):
            c["label_val"] = float("nan")
        c.setdefault("dt_us", None)
        c.setdefault("wall_time_s", None)
        c["snr"]       = c.get("snr_by_signal", {})
        c["Axial_disp"] = None
        c["Axial_vel"]  = None
    return raw


def load_h5_unified(h5_path: str, h5_type: str) -> List[Dict]:
    """Dispatcher: carga cualquier formato HDF5 → lista normalizada de casos."""
    loaders = {
        TYPE_DOE_RESULTS  : load_doe_results,
        TYPE_DOE_NOISE    : load_doe_noise,
        TYPE_DOE_INDICATOR: load_doe_indicator_unified,
        TYPE_NOISE_IND    : load_noise_indicator_unified,
        TYPE_MODEL_SNR    : load_model_snr_unified,
    }
    cases = loaders[h5_type](h5_path)
    _assign_case_colors(cases, qualitative=False)
    return cases


def _assign_case_colors(cases: List[Dict], qualitative: bool = False) -> None:
    """Asigna un color persistente a cada caso en c['_color'] y c['_color_hex'].
    Control → siempre rojo.
    qualitative=True  → tab20 por índice (casos con valores cercanos se distinguen bien).
    qualitative=False → viridis normalizado por valor numérico."""
    non_ctrl = [c for c in cases if c.get("group", "") != "control"]

    if qualitative:
        cmap = matplotlib.colormaps["tab20"]
        for i, c in enumerate(non_ctrl):
            rgba = cmap(i % cmap.N)
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)
        for c in cases:
            if c.get("group", "") == "control":
                c["_color"]     = (0.85, 0.05, 0.05, 1.0)
                c["_color_hex"] = "#d90d0d"
        return
    cmap = matplotlib.colormaps["viridis"]
    vals = [c.get("label_val", float("nan")) for c in non_ctrl]
    finite_vals = [v for v in vals if np.isfinite(v)]
    if len(finite_vals) >= 2:
        vmin, vmax = min(finite_vals), max(finite_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=max(len(non_ctrl) - 1, 1))
        # assign by index when all values are the same
        for i, c in enumerate(non_ctrl):
            rgba = cmap(norm(i))
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)
        for c in cases:
            if c.get("group", "") == "control":
                c["_color"]     = (0.85, 0.05, 0.05, 1.0)
                c["_color_hex"] = "#d90d0d"
        return
    for c in cases:
        if c.get("group", "") == "control":
            c["_color"]     = (0.85, 0.05, 0.05, 1.0)
            c["_color_hex"] = "#d90d0d"
        else:
            lv = c.get("label_val", float("nan"))
            rgba = cmap(norm(lv)) if np.isfinite(lv) else (0.5, 0.5, 0.5, 1.0)
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)


# ==============================================================================
# HELPERS
# ==============================================================================

def _fmt_val(v) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
        if np.isnan(fv):
            return "NaN"
        if np.isinf(fv):
            return "∞"
        return f"{fv:.2e}"
    except (TypeError, ValueError):
        return str(v)


def _col_header(key: str) -> str:
    return key.replace("$", "")


def _all_var_keys(cases: List[Dict]) -> List[str]:
    """Todas las claves de var_val presentes en los casos (label_key primero)."""
    keys: set = set()
    for c in cases:
        keys.update(c.get("var_val", {}).keys())
    lk = cases[0]["label_key"] if cases else ""
    ordered = []
    if lk and lk in keys:
        ordered.append(lk)
        keys.discard(lk)
    ordered.extend(sorted(keys))
    return ordered


def _capture_new_figure(func, *args, **kwargs) -> Optional[Figure]:
    """Llama func, captura la última figura matplotlib creada."""
    before = set(plt.get_fignums())
    result = func(*args, **kwargs)
    after  = set(plt.get_fignums())
    new_nums = sorted(after - before)
    if isinstance(result, Figure):
        return result
    if new_nums:
        return plt.figure(new_nums[-1])
    return None


def _embed_figure(fig: Figure, canvas_frame: tk.Frame,
                  toolbar_frame: tk.Frame, fig_holder: dict, slot: str) -> None:
    """Destruye widgets previos y embebe una figura matplotlib en Tkinter."""
    plt.close(fig)   # detach de pyplot window manager (no destroy el objeto)

    # Limpiar previo
    old = fig_holder.get(slot)
    if old is not None:
        try:
            plt.close(old)
        except Exception:
            pass
    fig_holder[slot] = fig

    for w in list(canvas_frame.winfo_children()):
        try:
            w.destroy()
        except Exception:
            pass
    for w in list(toolbar_frame.winfo_children()):
        try:
            w.destroy()
        except Exception:
            pass

    canvas  = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(fill=tk.X)


# ==============================================================================
# COLUMN SELECTION DIALOG (reused from doe_selector.py)
# ==============================================================================

class ColumnsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, all_keys: list, visible_keys: list) -> None:
        super().__init__(parent)
        self.title("Columnas visibles")
        self.resizable(False, False)
        self.grab_set()
        self._result = None
        self._vars: dict = {}

        ttk.Label(self, text="Columnas visibles en la tabla:",
                  font=("Arial", 10, "bold")).pack(padx=14, pady=(10, 4), anchor=tk.W)

        # Botones rápidos
        quick = ttk.Frame(self)
        quick.pack(fill=tk.X, padx=14, pady=(0, 4))
        ttk.Button(quick, text="☑ Todas",
                   command=lambda: [v.set(True)  for v in self._vars.values()]).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick, text="☐ Ninguna",
                   command=lambda: [v.set(False) for v in self._vars.values()]).pack(side=tk.LEFT, padx=2)

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, padx=14, pady=4)

        ttk.Label(frm, text="case  (fijo)", foreground="#888888").pack(anchor=tk.W, pady=1)

        for key in all_keys:
            var = tk.BooleanVar(value=(key in visible_keys))
            self._vars[key] = var
            ttk.Checkbutton(frm, text=_col_header(key), variable=var).pack(
                anchor=tk.W, pady=1)

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=14, pady=(6, 10))
        ttk.Button(btns, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Aplicar", command=self._apply).pack(side=tk.RIGHT, padx=4)

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


# ==============================================================================
# SUMMARY PLOT ENTRIES PER TYPE
# ==============================================================================

# Each entry: (label, callable_or_str, kwargs_template)
# callable_or_str: either a function reference or a special string key

def _make_summary_entries(h5_type: str, cases: list, h5_path: str):
    """Returns list of (label, func, extra_args_dict) for the right panel combobox."""
    entries = []

    if h5_type == TYPE_DOE_RESULTS:
        for lbl, fn, args in [
            ("Convergencia RMS — Axial_disp",      plot_convergence,              ("Axial_disp", "rms")),
            ("Convergencia RMS — Axial_vel",        plot_convergence,              ("Axial_vel",  "rms")),
            ("Convergencia Max — Axial_disp",       plot_convergence,              ("Axial_disp", "max")),
            ("Convergencia Max — Axial_vel",        plot_convergence,              ("Axial_vel",  "max")),
            ("Error más fino RMS — Axial_disp",     plot_convergence_error_ref,    ("Axial_disp", "rms")),
            ("Error más fino RMS — Axial_vel",      plot_convergence_error_ref,    ("Axial_vel",  "rms")),
            ("Ganancia RMS — Axial_disp",           plot_convergence_error_consec, ("Axial_disp", "rms")),
            ("Ganancia RMS — Axial_vel",            plot_convergence_error_consec, ("Axial_vel",  "rms")),
            ("Tiempo de ejecución",                 plot_convergence_time,         ()),
        ]:
            entries.append((lbl, fn, args))

    elif h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND):
        lk       = _resolve_label_key(cases)
        xs       = _ind_x_values(cases, lk)
        all_runs = sorted({rn for c in cases for rn in c.get("runs", {})})

        if h5_type == TYPE_NOISE_IND:
            # Figuras correctas para DOE de ruido: usan doe_noise_plotter
            # Usar los run names completos como clave (son los indicadores reales del HDF5)
            for ind in all_runs:
                # Figura combinada (t_d + t_d_no_FAR juntos)
                entries.append((
                    f"{ind} — t_d & t_d_no_FAR vs SNR",
                    "_noise_td_both",
                    {"h5_path": h5_path, "indicator": ind},
                ))
                # Figuras individuales
                for td_col, lbl_col in (("t_d", "t_d"), ("t_d_no_FAR", "t_d_no_FAR")):
                    entries.append((
                        f"{ind} — {lbl_col} vs SNR",
                        "_noise_td_ind",
                        {"h5_path": h5_path, "indicator": ind, "td_col": td_col},
                    ))
            entries.append(("Lollipop t_d vs indicador",  "_noise_lollipop", {"h5_path": h5_path}))
            entries.append(("Retraso (t_d - t_gt) vs SNR", "_noise_delay",    {"h5_path": h5_path}))
            entries.append(("Coste FAR vs SNR",            "_noise_far_cost", {"h5_path": h5_path}))
            for ind in all_runs:
                entries.append((f"I_t overlay — {ind}", "_noise_it_overlay",
                                {"h5_path": h5_path, "indicator": ind}))
        else:
            for use_no_far, lbl_suffix in [(False, "t_d"), (True, "t_d_no_FAR")]:
                # Figura global (todos los indicadores juntos)
                entries.append((
                    f"{lbl_suffix} vs {_col_header(lk)}  [todos]",
                    _plot_td_single,
                    {"cases": cases, "xs": xs, "all_runs": all_runs,
                     "label_key": lk, "use_no_far": use_no_far, "out_dir": None},
                ))
                # Una figura por indicador
                for rn in all_runs:
                    entries.append((
                        f"{rn} — {lbl_suffix} vs {_col_header(lk)}",
                        _plot_td_per_run,
                        {"cases": cases, "xs": xs, "run_name": rn,
                         "label_key": lk, "use_no_far": use_no_far, "out_dir": None},
                ))
            for use_no_far, lbl_suffix in [(False, "I_t overlay — t_d"), (True, "I_t overlay — no FAR")]:
                entries.append((
                    lbl_suffix,
                    plot_It_overlay,
                    {"cases": cases, "label_key": lk,
                     "run_name_filter": None, "use_no_far": use_no_far, "out_dir": None},
                ))

    elif h5_type == TYPE_DOE_NOISE:
        entries.append(("Overlay Axial_disp por SNR", "_noise_overlay", {"signal": "Axial_disp"}))
        entries.append(("Overlay Axial_vel por SNR",  "_noise_overlay", {"signal": "Axial_vel"}))

    elif h5_type == TYPE_MODEL_SNR:
        pk = _snr_detect_param_key(cases) or "case_idx"
        entries.append((f"SNR_mod_dB vs {_col_header(pk)}", plot_snr_vs_param,
                        {"cases": cases, "param_key": pk, "out_dir": None}))

    return entries


def _build_noise_overlay_fig(cases: list, signal: str) -> Optional[Figure]:
    """Crea una figura de overlay de señales coloreadas por snr_db (doe_noise)."""
    vals = [c["label_val"] for c in cases if np.isfinite(c["label_val"])]
    cmap = matplotlib.colormaps["viridis"]
    if len(vals) > 1:
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in cases:
        data = c["signals"].get(signal)
        if data is None:
            continue
        t, y = data
        lv = c.get("label_val")
        is_control = (c.get("group", "") == "control")
        if is_control or not np.isfinite(lv):
            # control: draw with distinctive style and include in legend
            color = "red"
            ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=2.2, alpha=1.0,
                    label="control", zorder=6, rasterized=True)
            # Also mark control in the plot (small annotation)
            try:
                mid = int(len(t) // 2)
                ax.scatter([t[mid]], [y[mid]], marker="D", color=color, s=30, zorder=7)
            except Exception:
                pass
        else:
            color = cmap(norm(lv))
            label = f"SNR={lv:.0f} dB"
            ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=1.4, alpha=0.85,
                    label=label, rasterized=True)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("SNR (dB)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(SIGNAL_YLABELS.get(signal, signal))
    ax.set_title(f"{signal} — overlay por SNR")
    # ax.grid(True, linestyle=":", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.grid(False)
    fig.tight_layout()
    return fig


def _run_indicator_prefix(run_name: str) -> str:
    """Devuelve el prefijo de indicador de un run_name (maxent, rms_cv, ssq, green, etc.)."""
    m = re.match(r"^(maxent|rms_cv|ssq[^_]*|green[^_]*|sst_svd[^_]*)", run_name)
    if m:
        return m.group(1)
    parts = run_name.split("_")
    return parts[0] if parts else run_name


def _variable_keys_with_variation(cases: List[Dict]) -> List[str]:
    """Devuelve las claves de var_val que tienen al menos 2 valores distintos entre los casos."""
    all_keys: set = set()
    for c in cases:
        all_keys.update(c.get("var_val", {}).keys())
    varying = []
    for k in sorted(all_keys):
        vals = {c["var_val"].get(k) for c in cases if k in c.get("var_val", {})}
        if len(vals) >= 2:
            varying.append(k)
    return varying


def _it_plot_yscale(runs_to_show: List[str]) -> str:
    """Escala Y para I_t: log solo para green* y sst_svd*; resto lineal."""
    if not runs_to_show:
        return "linear"
    prefixes = {_run_indicator_prefix(rn) for rn in runs_to_show}
    log_prefixes = {"green", "sst_svd"}
    return "log" if prefixes and prefixes.issubset(log_prefixes) else "linear"


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class DoeSelectorUnifiedApp:
    """Ventana principal: Treeview + panel central (señales/I_t) + summary."""

    _LEFT_WIDTH   = 420
    _CENTER_WIDTH = 860

    def __init__(self, root: tk.Tk, h5_path: str) -> None:
        self.root     = root
        self._fig_holder: dict = {}    # slot → Figure para Guardar PNG
        self._cbar       = None
        self._force_cbar = None
        self._deflex_cbar = None
        self._sort_col: Optional[str] = None
        self._sort_rev: bool           = False
        self._iid_to_case: dict = {}
        self._manual_control_group: Optional[str] = None  # grupo elegido como control manual
        self._ref_lines: List[dict] = []  # {"kind": "v"|"h", "value": float, "target": str, "color": str}

        self._load_file(h5_path)
        self._build_ui()

    # ── File loading ──────────────────────────────────────────────────────────
    def _load_file(self, h5_path: str) -> None:
        self.h5_path  = h5_path
        self.h5_type  = detect_h5_type(h5_path)
        self.cases    = load_h5_unified(h5_path, self.h5_type)  # colors already assigned
        self.doe_name = os.path.basename(os.path.dirname(h5_path))
        self._all_keys     = _all_var_keys(self.cases)
        self._visible_keys = list(self._all_keys)
        # Run-names available (for indicator types)
        self._all_runs = sorted({rn for c in self.cases for rn in c.get("runs", {})})

        # Sincronizar doe_plotter.LABEL_KEY con la variable real del DOE cargado.
        # Las funciones plot_convergence / plot_overlay usan ese global directamente.
        import doe_plotter as _dp
        if self.cases:
            _dp.LABEL_KEY = self.cases[0].get("label_key", _dp.LABEL_KEY)

        # Summary entries for right panel
        self._summary_entries = _make_summary_entries(self.h5_type, self.cases, h5_path)
        self._summary_labels  = [e[0] for e in self._summary_entries]
        # Filter variables
        self._filter_var_str: Optional[tk.StringVar] = None

        # Refresh label-key combobox if UI already built
        if hasattr(self, "_label_key_combo"):
            self._refresh_label_key_combo()

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.root.title(
            f"{_TYPE_LABELS.get(self.h5_type, self.h5_type)}  —  "
            f"{os.path.basename(self.h5_path)}"
        )
        self.root.minsize(1100, 580)
        self.root.state("zoomed")

        self._has_deflex = (
            self.h5_type == TYPE_DOE_RESULTS
            and any(bool(c.get("out_deflex")) for c in self.cases)
        )
        self._build_topbar()
        self._build_layout()
        self._build_left_panel()
        self._build_center_panel()

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(4, 2))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="📂  Abrir otro .h5",
                   command=self._open_file).pack(side=tk.LEFT, padx=4)
        self._type_label = ttk.Label(
            bar,
            text=f"Tipo: {_TYPE_LABELS.get(self.h5_type, self.h5_type)}  |  "
                 f"{len(self.cases)} casos  |  {os.path.basename(self.h5_path)}",
            foreground="#444444", font=("Arial", 9),
        )
        self._type_label.pack(side=tk.LEFT, padx=8)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._persistent_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar, text="🎨 Color fijo por caso",
            variable=self._persistent_color_var,
            command=self._replot_active_tab,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="α:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._sig_alpha_var = tk.DoubleVar(value=0.5)
        _sc = ttk.Scale(bar, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                         variable=self._sig_alpha_var, length=80)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())
        ttk.Label(bar, text="lw:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self._sig_lw_var = tk.DoubleVar(value=0.9)
        _sc = ttk.Scale(bar, from_=0.3, to=3.0, orient=tk.HORIZONTAL,
                         variable=self._sig_lw_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="ctrl α:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._ctrl_alpha_var = tk.DoubleVar(value=1.0)
        _sc = ttk.Scale(bar, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                         variable=self._ctrl_alpha_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())
        ttk.Label(bar, text="ctrl z:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self._ctrl_zo_var = tk.IntVar(value=100)
        _sc = ttk.Scale(bar, from_=1, to=200, orient=tk.HORIZONTAL,
                         variable=self._ctrl_zo_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._invert_order_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="⇅ Invertir zorder",
            variable=self._invert_order_var,
            command=self._replot_active_tab,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="Eje X:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._label_key_var = tk.StringVar(value="auto")
        self._label_key_combo = ttk.Combobox(
            bar, textvariable=self._label_key_var,
            state="readonly", width=18,
        )
        self._label_key_combo.pack(side=tk.LEFT, padx=4)
        self._label_key_combo.bind("<<ComboboxSelected>>", self._on_label_key_change)
        self._refresh_label_key_combo()

        bar2 = ttk.Frame(self.root, padding=(4, 2))
        bar2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar2, text="línea x=", font=("Arial", 8)).pack(side=tk.LEFT)
        self._vline_entry = ttk.Entry(bar2, width=8)
        self._vline_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(bar2, text="y=", font=("Arial", 8)).pack(side=tk.LEFT)
        self._hline_entry = ttk.Entry(bar2, width=8)
        self._hline_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(bar2, text="en:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._line_target_var = tk.StringVar(value=_LINE_TARGET_ALL)
        ttk.Combobox(bar2, textvariable=self._line_target_var, values=_LINE_TARGETS,
                     state="readonly", width=14).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bar2, text="+ Línea", command=self._add_reference_lines).pack(side=tk.LEFT, padx=2)
        self._line_remove_var = tk.StringVar()
        self._line_remove_combo = ttk.Combobox(bar2, textvariable=self._line_remove_var,
                                               state="readonly", width=20)
        self._line_remove_combo.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar2, text="Borrar sel.", command=self._remove_selected_line).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar2, text="Borrar todas", command=self._clear_reference_lines).pack(side=tk.LEFT, padx=2)

    def _refresh_label_key_combo(self) -> None:
        """Actualiza las opciones del combobox con las variables que tienen variación."""
        varying = _variable_keys_with_variation(self.cases)
        options = ["auto"] + [_col_header(k) for k in varying]
        self._label_key_combo["values"] = options
        # Preseleccionar la key actualmente activa
        current = _col_header(self.cases[0].get("label_key", "")) if self.cases else "auto"
        if current in options:
            self._label_key_var.set(current)
        else:
            self._label_key_var.set("auto")

    def _on_label_key_change(self, _event=None) -> None:
        """Cambia la variable de etiqueta/color en todos los casos y refresca la UI."""
        chosen = self._label_key_var.get()
        if chosen == "auto":
            new_key = _best_label_key(self.cases)
        else:
            # Buscar la clave original con $ que coincida (strip $)
            varying = _variable_keys_with_variation(self.cases)
            new_key = next((k for k in varying if _col_header(k) == chosen), chosen)

        for c in self.cases:
            c["label_key"] = new_key
            c["label_val"] = float(c.get("var_val", {}).get(new_key, float("nan")))

        _assign_case_colors(self.cases, qualitative=False)

        # Actualizar LABEL_KEY del doe_plotter
        import doe_plotter as _dp
        _dp.LABEL_KEY = new_key

        self._populate_tree(self._filtered_cases())
        self._replot_active_tab()

    def _active_tab_info(self):
        """(plot_fn, axes_dict, canvas) del tab actualmente activo, o (None, {}, None)."""
        if hasattr(self, "_nb"):
            current = self._nb.select()
            if hasattr(self, "_sig_tab") and current == str(self._sig_tab):
                return (self._plot_signals,
                        {"Señales: disp": self.ax_disp, "Señales: vel": self.ax_vel},
                        self.sig_canvas)
            if hasattr(self, "_force_tab") and current == str(self._force_tab):
                return (self._plot_forces,
                        {"Fuerzas: F1": self.ax_force_1, "Fuerzas: F2": self.ax_force_2,
                         "Fuerzas: F3": self.ax_force_3},
                        self.force_canvas)
            if hasattr(self, "_It_tab") and current == str(self._It_tab):
                return self._plot_It, {"I_t": self.ax_It}, self.It_canvas
            if hasattr(self, "_deflex_tab") and current == str(self._deflex_tab):
                return (self._plot_deflex,
                        {"Deflex: disp": self.ax_deflex_d, "Deflex: vel": self.ax_deflex_v},
                        self.deflex_canvas)
        elif hasattr(self, "sig_canvas"):
            return (self._plot_signals,
                    {"Señales: disp": self.ax_disp, "Señales: vel": self.ax_vel},
                    self.sig_canvas)
        return None, {}, None

    def _replot_active_tab(self) -> None:
        """Redibuja el tab con la selección actual (tras cambiar la propiedad de color/leyenda)."""
        if not self.tree.selection():
            return
        fn, _, _ = self._active_tab_info()
        if fn is not None:
            fn()

    def _replot_preserving_zoom(self) -> None:
        """Redibuja el tab activo pero mantiene el zoom/pan que ya tenia (usado al agregar/borrar lineas)."""
        _, axes, _ = self._active_tab_info()
        saved = {key: (ax.get_xlim(), ax.get_ylim()) for key, ax in axes.items()}
        self._replot_active_tab()
        _, axes2, canvas2 = self._active_tab_info()
        for key, ax in axes2.items():
            if key in saved:
                xlim, ylim = saved[key]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if canvas2 is not None:
            canvas2.draw()

    def _add_reference_lines(self) -> None:
        """Agrega los valores de los campos x=/y= como lineas de referencia y redibuja."""
        target = self._line_target_var.get() or _LINE_TARGET_ALL
        new_entries = []
        v_txt = self._vline_entry.get().strip()
        if v_txt:
            try:
                color = _LINE_COLORS[len(self._ref_lines) % len(_LINE_COLORS)]
                entry = {"kind": "v", "value": float(v_txt), "target": target, "color": color}
                self._ref_lines.append(entry)
                new_entries.append(entry)
            except ValueError:
                messagebox.showwarning("Valor inválido", f"'{v_txt}' no es un número.", parent=self.root)
        h_txt = self._hline_entry.get().strip()
        if h_txt:
            try:
                color = _LINE_COLORS[len(self._ref_lines) % len(_LINE_COLORS)]
                entry = {"kind": "h", "value": float(h_txt), "target": target, "color": color}
                self._ref_lines.append(entry)
                new_entries.append(entry)
            except ValueError:
                messagebox.showwarning("Valor inválido", f"'{h_txt}' no es un número.", parent=self.root)
        if new_entries:
            self._vline_entry.delete(0, tk.END)
            self._hline_entry.delete(0, tk.END)
            self._refresh_line_remove_combo()
            _, axes, _ = self._active_tab_info()
            if all(self._line_in_view(e, axes) for e in new_entries):
                self._replot_preserving_zoom()
            else:
                self._replot_active_tab()  # la linea nueva queda fuera del zoom actual -> autoescala

    def _line_in_view(self, entry: dict, axes: dict) -> bool:
        target_axes = list(axes.values()) if entry["target"] == _LINE_TARGET_ALL else (
            [axes[entry["target"]]] if entry["target"] in axes else [])
        for ax in target_axes:
            lo, hi = ax.get_xlim() if entry["kind"] == "v" else ax.get_ylim()
            if not (min(lo, hi) <= entry["value"] <= max(lo, hi)):
                return False
        return True

    def _clear_reference_lines(self) -> None:
        self._ref_lines.clear()
        self._refresh_line_remove_combo()
        self._replot_preserving_zoom()

    def _refresh_line_remove_combo(self) -> None:
        labels = [f"{i}: {'x' if e['kind']=='v' else 'y'}={e['value']:g}  [{e['target']}]"
                  for i, e in enumerate(self._ref_lines)]
        self._line_remove_combo["values"] = labels
        self._line_remove_var.set(labels[-1] if labels else "")

    def _remove_selected_line(self) -> None:
        sel = self._line_remove_var.get()
        if not sel:
            return
        idx = int(sel.split(":", 1)[0])
        del self._ref_lines[idx]
        self._refresh_line_remove_combo()
        self._replot_preserving_zoom()

    def _draw_reference_lines(self, axes: dict) -> None:
        """axes: {nombre_target: Axes} de los ejes del plot que se esta dibujando ahora."""
        for entry in self._ref_lines:
            target = entry["target"]
            targets = list(axes.values()) if target == _LINE_TARGET_ALL else (
                [axes[target]] if target in axes else [])
            for ax in targets:
                if entry["kind"] == "v":
                    ax.axvline(entry["value"], color=entry["color"], lw=1.2, linestyle="--", zorder=10)
                    ax.text(entry["value"], 0.98, f"{entry['value']:g}", transform=ax.get_xaxis_transform(),
                            va="top", ha="right", color=entry["color"], rotation=90,
                            fontsize=14, zorder=11, clip_on=True)
                else:
                    ax.axhline(entry["value"], color=entry["color"], lw=1.2, linestyle="--", zorder=10)
                    ax.text(0.02, entry["value"], f"{entry['value']:g}", transform=ax.get_yaxis_transform(),
                            va="bottom", ha="left", color=entry["color"],
                            fontsize=14, zorder=11, clip_on=True)

    def _build_layout(self) -> None:
        self.paned = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief=tk.RAISED,
        )
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.left_frame   = ttk.Frame(self.paned)
        self.center_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame,   minsize=240, width=self._LEFT_WIDTH)
        self.paned.add(self.center_frame, minsize=400, width=self._CENTER_WIDTH)

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    def _build_left_panel(self) -> None:
        lf = self.left_frame

        # Header
        ttk.Label(lf, text=self.doe_name,
                  font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Label(
            lf,
            text=f"{len(self.cases)} casos  ·  {_TYPE_LABELS.get(self.h5_type, '')}",
            font=("Arial", 9), foreground="#555555",
        ).pack(anchor=tk.W, padx=8, pady=(0, 4))

        # Search bar + Columns button
        bar = ttk.Frame(lf)
        bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._filter_var_str = tk.StringVar()
        self._filter_var_str.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(bar, textvariable=self._filter_var_str,
                  font=("Arial", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bar, text=" 🔍", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Columnas…",
                   command=self._show_columns_dialog).pack(side=tk.LEFT, padx=(6, 0))

        # Run filter (only for indicator types)
        if self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND):
            self._build_run_filter(lf)

        # Treeview container
        self._tree_frame = ttk.Frame(lf)
        self._tree_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        self._build_tree()

        # Action buttons
        btn = ttk.Frame(lf)
        btn.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn, text="Seleccionar todo",
                   command=self._select_all).pack(fill=tk.X, pady=1)
        ttk.Button(btn, text="Limpiar selección",
                   command=self._clear_sel).pack(fill=tk.X, pady=1)
        ttk.Separator(btn).pack(fill=tk.X, pady=4)
        ttk.Button(btn, text="Plot Señales ▶",
                   command=self._plot_signals).pack(fill=tk.X, pady=1, ipady=3)
        if self.h5_type == TYPE_DOE_RESULTS:
            ttk.Button(btn, text="Plot Fuerzas ▶",
                       command=self._plot_forces).pack(fill=tk.X, pady=1, ipady=3)
        if self._has_deflex:
            ttk.Button(btn, text="Plot Deflexión ▶",
                       command=self._plot_deflex).pack(fill=tk.X, pady=1, ipady=3)
        if self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND):
            ttk.Button(btn, text="Plot I_t ▶",
                       command=self._plot_It).pack(fill=tk.X, pady=1, ipady=2)
        ttk.Button(btn, text="Limpiar plot",
                   command=self._clear_signal_plot).pack(fill=tk.X, pady=1)
        ttk.Separator(btn).pack(fill=tk.X, pady=4)
        self._ctrl_label_var = tk.StringVar(value="Control manual: ninguno")
        ttk.Label(btn, textvariable=self._ctrl_label_var,
                  foreground="#cc0000", font=("Arial", 8)).pack(anchor=tk.W)
        ttk.Button(btn, text="⭐ Marcar como control",
                   command=self._set_manual_control).pack(fill=tk.X, pady=1)
        ttk.Button(btn, text="✖ Quitar control manual",
                   command=self._clear_manual_control).pack(fill=tk.X, pady=1)

    def _set_manual_control(self) -> None:
        """Marca el caso seleccionado en el árbol como control manual."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Sin selección", "Selecciona un caso primero.", parent=self.root)
            return
        if len(sel) > 1:
            messagebox.showwarning("Selección múltiple", "Selecciona solo un caso.", parent=self.root)
            return
        c = self._iid_to_case.get(sel[0])
        if c is None:
            return
        self._manual_control_group = c["group"]
        self._ctrl_label_var.set(f"Control manual: {c['group']}")
        self._populate_tree(self._filtered_cases())

    def _clear_manual_control(self) -> None:
        """Elimina el control manual."""
        self._manual_control_group = None
        self._ctrl_label_var.set("Control manual: ninguno")
        self._populate_tree(self._filtered_cases())

    def _is_control(self, c: dict) -> bool:
        """True si el caso es control real O control manual asignado en GUI."""
        if c.get("group", "") == "control":
            return True
        if self._manual_control_group and c.get("group", "") == self._manual_control_group:
            return True
        return False

    def _build_run_filter(self, parent: ttk.Frame) -> None:
        """Panel de selección de indicadores (checkboxes) para tipos indicador."""
        frm = ttk.LabelFrame(parent, text="  Indicadores  ", padding=4)
        frm.pack(fill=tk.X, padx=8, pady=(0, 4))

        indicators = self._extract_indicators()
        palette = matplotlib.colormaps.get_cmap("tab10")
        self._ind_color_map = {
            ind: palette(i % palette.N)
            for i, ind in enumerate(indicators)
        }

        # Checkbox "Todos"
        self._ind_all_var = tk.BooleanVar(value=True)
        self._ind_all_chk = ttk.Checkbutton(
            frm, text="(todos)", variable=self._ind_all_var,
            command=self._on_ind_all_toggle,
        )
        self._ind_all_chk.pack(anchor=tk.W)

        # Un checkbox por indicador
        self._ind_check_vars: Dict[str, tk.BooleanVar] = {}
        for ind in indicators:
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                frm, text=ind, variable=var,
                command=self._on_ind_check_toggle,
            )
            chk.pack(anchor=tk.W, padx=(12, 0))
            self._ind_check_vars[ind] = var

        ttk.Separator(frm).pack(fill=tk.X, pady=3)

        # t_d / t_d_no_FAR toggle
        self._use_no_far_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="usar t_d_no_FAR",
                        variable=self._use_no_far_var).pack(anchor=tk.W)

    def _extract_indicators(self) -> List[str]:
        """Extrae los prefijos de indicador de los run_names."""
        prefixes = set()
        for rn in self._all_runs:
            # Intenta extraer el prefijo (maxent, rms_cv, ssq, green_fixed, etc.)
            # Patrón: indicador termina antes de _revo o _fcycle o _dec
            m = re.match(r"^(maxent|rms_cv|ssq[^_]*|green[^_]*)", rn)
            if m:
                prefixes.add(m.group(1))
            else:
                parts = rn.split("_")
                if parts:
                    prefixes.add(parts[0])
        return sorted(prefixes)

    def _on_ind_all_toggle(self) -> None:
        """Activa/desactiva todos los checkboxes de indicadores."""
        val = self._ind_all_var.get()
        for v in self._ind_check_vars.values():
            v.set(val)

    def _on_ind_check_toggle(self) -> None:
        """Sincroniza el checkbox 'todos' según el estado individual."""
        all_on = all(v.get() for v in self._ind_check_vars.values())
        self._ind_all_var.set(all_on)

    def _selected_run_filter(self) -> Optional[str]:
        """Retorna None (compatibilidad; la lógica real está en _get_runs_to_show)."""
        return None

    def _get_runs_to_show(self) -> List[str]:
        """Retorna los runs activos según los checkboxes de indicadores."""
        if not hasattr(self, "_ind_check_vars") or not self._ind_check_vars:
            return self._all_runs
        selected_inds = [ind for ind, v in self._ind_check_vars.items() if v.get()]
        if not selected_inds:
            return self._all_runs  # ninguno marcado → todos
        return [r for r in self._all_runs if any(r.startswith(ind) for ind in selected_inds)]

    # ── TREEVIEW ──────────────────────────────────────────────────────────────
    def _build_tree(self) -> None:
        for attr in ("tree", "_sb_y", "_sb_x"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass

        # Build columns: "case" + visible var_val keys + t_d / t_d_no_FAR per run (indicators)
        cols = ["case"] + self._visible_keys
        if self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND) and self._all_runs:
            for rn in self._all_runs[:4]:
                cols += [f"td_{rn}", f"tdnf_{rn}"]
        if self.h5_type == TYPE_MODEL_SNR and self.cases:
            snr_keys = sorted({k for c in self.cases for k in c.get("snr", {})})
            cols += [f"snr_{s}" for s in snr_keys]
        self._cols = cols

        tf = self._tree_frame
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                 selectmode="extended")
        for col in cols:
            if col == "case":
                hdr, w = "case", 68
            elif col.startswith("td_"):
                hdr = "t_d:" + col[3:][:10]
                w   = 90
            elif col.startswith("tdnf_"):
                hdr = "t_d(noFAR):" + col[5:][:10]
                w   = 110
            elif col.startswith("snr_"):
                hdr = "SNR:" + col[4:]
                w   = 90
            else:
                hdr = _col_header(col)
                w   = 90
            self.tree.heading(col, text=hdr,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, minwidth=40, anchor=tk.CENTER, stretch=True)

        self._sb_y = ttk.Scrollbar(tf, orient=tk.VERTICAL,   command=self.tree.yview)
        self._sb_x = ttk.Scrollbar(tf, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=self._sb_y.set, xscrollcommand=self._sb_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self._sb_y.grid(row=0, column=1, sticky="ns")
        self._sb_x.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<Double-1>", lambda _: self._plot_signals())

        self._populate_tree(self._filtered_cases())

    def _populate_tree(self, cases_to_show: list) -> None:
        prev_selected = {id(self._iid_to_case[i]) for i in self.tree.selection()
                          if i in self._iid_to_case}
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._iid_to_case.clear()

        to_reselect = []
        for c in cases_to_show:
            row = []
            is_control = (c.get("group", "") == "control")
            for col in self._cols:
                if col == "case":
                    row.append(c["group"])
                elif col.startswith("td_"):
                    rn   = col[3:]
                    td   = c.get("runs", {}).get(rn, {}).get("t_d", np.array([]))
                    row.append(f"{td[0]:.2e} s" if td.size > 0 else "—")
                elif col.startswith("tdnf_"):
                    rn   = col[5:]
                    td   = c.get("runs", {}).get(rn, {}).get("t_d_no_FAR", np.array([]))
                    row.append(f"{td[0]:.2e} s" if td.size > 0 else "—")
                elif col.startswith("snr_"):
                    sig  = col[4:]
                    snr  = c.get("snr", {}).get(sig, float("nan"))
                    row.append(f"{snr:.2e} dB" if not np.isnan(snr) else "—")
                elif col == "snr_db":
                    raw_v = c.get("var_val", {}).get("snr_db", float("nan"))
                    if is_control or not np.isfinite(float(raw_v) if raw_v is not None else float("nan")):
                        row.append("control")
                    else:
                        row.append(f"{float(raw_v):.2e} dB")
                else:
                    raw_v = c.get("var_val", {}).get(col)
                    if col == "snr_db" and is_control:
                        row.append("control")
                    else:
                        row.append(_fmt_val(raw_v))
            iid = self.tree.insert("", tk.END, values=row)
            is_manual_ctrl = (self._manual_control_group and
                              c.get("group", "") == self._manual_control_group)
            if is_control or is_manual_ctrl:
                self.tree.tag_configure("control_row", foreground="#cc0000", font=("Arial", 9, "bold"))
                self.tree.item(iid, tags=("control_row",))
            self._iid_to_case[iid] = c
            if id(c) in prev_selected:
                to_reselect.append(iid)

        if to_reselect:
            self.tree.selection_set(to_reselect)

    def _sort_by(self, col: str) -> None:
        reverse          = (self._sort_col == col) and not self._sort_rev
        self._sort_col   = col
        self._sort_rev   = reverse
        rows = [(self.tree.set(iid, col), iid) for iid in self.tree.get_children()]

        def _key(item):
            try:
                return (0, float(item[0].rstrip("sBd")))
            except (ValueError, TypeError):
                return (1, str(item[0]))

        rows.sort(key=_key, reverse=reverse)
        for idx, (_, iid) in enumerate(rows):
            self.tree.move(iid, "", idx)
        for c in self._cols:
            hdr = self.tree.heading(c)["text"].rstrip(" ▲▼")
            arrow = (" ▼" if reverse else " ▲") if c == col else ""
            self.tree.heading(c, text=hdr + arrow, command=lambda cc=c: self._sort_by(cc))

    def _filtered_cases(self) -> list:
        query = (self._filter_var_str.get().strip().lower()
                 if self._filter_var_str else "")
        if not query:
            return self.cases
        return [c for c in self.cases if self._case_matches(c, query)]

    def _case_matches(self, c: dict, query: str) -> bool:
        tokens = [c["group"].lower()]
        for v in c.get("var_val", {}).values():
            tokens.append(_fmt_val(v).lower())
        return any(query in t for t in tokens)

    def _apply_filter(self) -> None:
        self._populate_tree(self._filtered_cases())

    def _show_columns_dialog(self) -> None:
        dlg = ColumnsDialog(self.root, self._all_keys, list(self._visible_keys))
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self._visible_keys = list(dlg.result)
            self._build_tree()

    def _select_all(self) -> None:
        self.tree.selection_set(self.tree.get_children())

    def _clear_sel(self) -> None:
        self.tree.selection_remove(self.tree.get_children())

    # ── CENTER PANEL ──────────────────────────────────────────────────────────
    def _build_center_panel(self) -> None:
        cf = self.center_frame

        has_signals = self.h5_type in (TYPE_DOE_RESULTS, TYPE_DOE_NOISE,
                                        TYPE_DOE_INDICATOR, TYPE_NOISE_IND)
        has_forces  = self.h5_type == TYPE_DOE_RESULTS
        has_runs    = self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND)
        has_deflex  = self._has_deflex
        has_snr_only = self.h5_type == TYPE_MODEL_SNR

        if has_signals and has_forces and has_runs:
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab   = ttk.Frame(self._nb)
            self._force_tab = ttk.Frame(self._nb)
            self._It_tab    = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab,   text=" Señales ")
            self._nb.add(self._force_tab, text=" Fuerzas ")
            self._nb.add(self._It_tab,    text=" I_t(t) ")
            self._build_signal_canvas(self._sig_tab)
            self._build_force_canvas(self._force_tab)
            self._build_It_canvas(self._It_tab)
            if has_deflex:
                self._deflex_tab = ttk.Frame(self._nb)
                self._nb.add(self._deflex_tab, text=" Out Deflex ")
                self._build_deflex_canvas(self._deflex_tab)

        elif has_signals and has_forces:
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab   = ttk.Frame(self._nb)
            self._force_tab = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab,   text=" Señales ")
            self._nb.add(self._force_tab, text=" Fuerzas ")
            self._build_signal_canvas(self._sig_tab)
            self._build_force_canvas(self._force_tab)
            if has_deflex:
                self._deflex_tab = ttk.Frame(self._nb)
                self._nb.add(self._deflex_tab, text=" Out Deflex ")
                self._build_deflex_canvas(self._deflex_tab)

        elif has_signals and has_runs:
            # Notebook con dos tabs
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab = ttk.Frame(self._nb)
            self._It_tab  = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab, text=" Señales ")
            self._nb.add(self._It_tab,  text=" I_t(t) ")
            self._build_signal_canvas(self._sig_tab)
            self._build_It_canvas(self._It_tab)
        elif has_signals and self.h5_type == TYPE_NOISE_IND:
            # For noise_indicator type, show two vertical I_t slots with run checkboxes
            self._build_noise_indicator_slots(cf)
        elif has_signals:
            self._sig_tab = cf
            self._build_signal_canvas(cf)
        elif has_snr_only:
            ttk.Label(cf, text="Señales no disponibles en doe_model_snr_results.h5.\n"
                                "Usa el panel derecho para las figuras de SNR.",
                      foreground="#777777", font=("Arial", 11),
                      anchor=tk.CENTER, justify=tk.CENTER).pack(expand=True)

    def _build_deflex_canvas(self, parent: tk.Frame) -> None:
        """Crea dos subplots embebidos para Out_Deflex (disp y vel sin deflexion estatica)."""
        self.deflex_fig  = Figure(constrained_layout=True)
        self.ax_deflex_d = self.deflex_fig.add_subplot(2, 1, 1)
        self.ax_deflex_v = self.deflex_fig.add_subplot(2, 1, 2, sharex=self.ax_deflex_d)
        self._init_deflex_axes()

        self.deflex_canvas = FigureCanvasTkAgg(self.deflex_fig, master=parent)
        self.deflex_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.deflex_canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.deflex_canvas.draw()

    def _init_deflex_axes(self) -> None:
        self.ax_deflex_d.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp (out deflex)"), fontsize=14)
        self.ax_deflex_d.grid(False)
        self.ax_deflex_d.tick_params(labelbottom=False, labelsize=12)
        self.ax_deflex_d.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_v.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel (out deflex)"), fontsize=14)
        self.ax_deflex_v.set_xlabel("Time (s)", fontsize=14)
        self.ax_deflex_v.grid(False)
        self.ax_deflex_v.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.deflex_fig.suptitle("Selecciona casos y presiona  Plot Deflexión ▶")

    def _build_force_canvas(self, parent: tk.Frame) -> None:
        """Crea tres subplots embebidos para res_R_p (Fx, Fy, Fz)."""
        self.force_fig = Figure(constrained_layout=True)
        self.ax_force_1 = self.force_fig.add_subplot(3, 1, 1)
        self.ax_force_2 = self.force_fig.add_subplot(3, 1, 2, sharex=self.ax_force_1)
        self.ax_force_3 = self.force_fig.add_subplot(3, 1, 3, sharex=self.ax_force_1)
        self._init_force_axes()

        self.force_canvas = FigureCanvasTkAgg(self.force_fig, master=parent)
        self.force_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.force_canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.force_canvas.draw()

    def _init_force_axes(self) -> None:
        labels = ["Fx", "Fy", "Fz"]
        axes = [self.ax_force_1, self.ax_force_2, self.ax_force_3]
        for i, ax in enumerate(axes):
            ax.set_ylabel(labels[i], fontsize=14)
            ax.grid(False)
            ax.tick_params(labelsize=12)
            if i < 2:
                ax.tick_params(labelbottom=False)
        self.ax_force_3.set_xlabel("Time (s)", fontsize=14)
        self.force_fig.suptitle("Selecciona casos y presiona  Plot Fuerzas ▶")

    def _build_signal_canvas(self, parent: tk.Frame) -> None:
        """Crea los dos subplots (Axial_disp + Axial_vel) embebidos."""
        self.sig_fig     = Figure(constrained_layout=True)
        self.ax_disp     = self.sig_fig.add_subplot(2, 1, 1)
        self.ax_vel      = self.sig_fig.add_subplot(2, 1, 2, sharex=self.ax_disp)
        self._init_signal_axes()

        self.sig_canvas = FigureCanvasTkAgg(self.sig_fig, master=parent)
        self.sig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.sig_canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sig_canvas.draw()

    def _build_It_canvas(self, parent: tk.Frame) -> None:
        """Crea el subplot de I_t(t) embebido."""
        self.It_fig = Figure(constrained_layout=True)
        self.ax_It  = self.It_fig.add_subplot(1, 1, 1)
        self._init_It_axis()

        self.It_canvas = FigureCanvasTkAgg(self.It_fig, master=parent)
        self.It_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.It_canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.It_canvas.draw()

    def _build_noise_indicator_slots(self, parent: tk.Frame) -> None:
        """Crea dos subfiguras verticales, cada una con selector de indicador
        y checkboxes multi-run para hacer overlay de I_t(t).
        """
        top_zone = ttk.LabelFrame(parent, text=" I_t Slot Top ", padding=4)
        bot_zone = ttk.LabelFrame(parent, text=" I_t Slot Bot ", padding=4)
        top_zone.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        bot_zone.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Controls and canvas for each zone
        def make_slot(zone, slot):
            ctrl = ttk.Frame(zone)
            ctrl.pack(fill=tk.X)
            ttk.Label(ctrl, text="Indicador:").pack(side=tk.LEFT)
            inds = self._extract_indicators()
            ind_var = tk.StringVar(value=inds[0] if inds else "")
            ind_combo = ttk.Combobox(ctrl, values=inds, textvariable=ind_var, state="readonly", width=20)
            ind_combo.pack(side=tk.LEFT, padx=4)
            ttk.Button(ctrl, text="Refresh runs", command=lambda: self._populate_run_checks(slot)).pack(side=tk.LEFT, padx=4)
            ttk.Button(ctrl, text="Plot ▶", command=lambda: self._plot_It_slot(slot)).pack(side=tk.LEFT, padx=4)

            # Scrollable frame for run checkboxes
            box_frame = ttk.Frame(zone)
            box_frame.pack(fill=tk.BOTH, expand=False, pady=(4,2))
            canvas = tk.Canvas(box_frame, height=120)
            sb = ttk.Scrollbar(box_frame, orient=tk.VERTICAL, command=canvas.yview)
            inner = ttk.Frame(canvas)
            inner_id = canvas.create_window((0,0), window=inner, anchor='nw')
            canvas.configure(yscrollcommand=sb.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.LEFT, fill=tk.Y)

            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Canvas for figure
            fig_frame = ttk.Frame(zone)
            fig_frame.pack(fill=tk.BOTH, expand=True)
            fig = Figure(constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            canvas_fig = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas_fig, fig_frame, pack_toolbar=False)
            toolbar.update(); toolbar.pack(fill=tk.X)

            # store slot state
            self.__dict__[f"_{slot}_ind_combo"] = ind_combo
            self.__dict__[f"_{slot}_run_frame"] = inner
            self.__dict__[f"_{slot}_fig"] = fig
            self.__dict__[f"_{slot}_ax"] = ax
            self.__dict__[f"_{slot}_canvas"] = canvas_fig

        make_slot(top_zone, "topslot")
        make_slot(bot_zone, "botslot")

        # populate runs initially
        self._populate_run_checks("topslot")
        self._populate_run_checks("botslot")

    def _populate_run_checks(self, slot: str) -> None:
        frame = self.__dict__.get(f"_{slot}_run_frame")
        combo = self.__dict__.get(f"_{slot}_ind_combo")
        if frame is None or combo is None:
            return
        # clear
        for w in list(frame.winfo_children()):
            w.destroy()
        ind = combo.get()
        runs = [r for r in self._all_runs if r.startswith(ind)] if ind else list(self._all_runs)
        self.__dict__[f"_{slot}_run_vars"] = {}
        for r in runs:
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text=r, variable=var)
            chk.pack(anchor=tk.W)
            self.__dict__[f"_{slot}_run_vars"] [r] = var

    def _plot_It_slot(self, slot: str) -> None:
        run_vars = self.__dict__.get(f"_{slot}_run_vars", {})
        selected_runs = [r for r, v in run_vars.items() if v.get()]
        if not selected_runs:
            messagebox.showwarning("Sin selection", "Selecciona al menos un run.", parent=self.root)
            return
        ax = self.__dict__.get(f"_{slot}_ax")
        fig = self.__dict__.get(f"_{slot}_fig")
        canvas_fig = self.__dict__.get(f"_{slot}_canvas")
        if ax is None or fig is None or canvas_fig is None:
            return
        ax.cla()
        # color map
        vals = [c.get("label_val", float("nan")) for c in self.cases]
        vals_f = [v for v in vals if np.isfinite(v)]
        cmap_It = matplotlib.colormaps["viridis"]
        norm_It = mcolors.Normalize(vmin=min(vals_f) if vals_f else 0, vmax=max(vals_f) if vals_f else 1)

        for c in self.cases:
            pv = c.get("label_val", float("nan"))
            is_ctrl = (c.get("group", "") == "control")
            if is_ctrl:
                color = (0.85, 0.05, 0.05)   # rojo siempre para control
            elif np.isfinite(pv):
                color = cmap_It(norm_It(pv))
            else:
                color = (0.5, 0.5, 0.5)
            for rn in selected_runs:
                run_data = c.get("runs", {}).get(rn)
                if run_data is None:
                    continue
                t = run_data.get("t", np.array([]))
                I_t = run_data.get("I_t", np.array([]))
                if t.size == 0 or I_t.size == 0:
                    continue
                lbl = f"control|{rn}" if is_ctrl else f"{c.get('label_val','?'):.1f}dB|{rn}"
                lw  = 2.2 if is_ctrl else 1.6
                ax.plot(t[::_IND_DECIMATE], I_t[::_IND_DECIMATE], color=color, lw=lw,
                        alpha=1.0 if is_ctrl else 0.9, label=lbl, zorder=5 if is_ctrl else 3)
                td = run_data.get("t_d", np.array([]))
                if td.size>0:
                    ax.axvline(td[0], color=color, lw=2.0, linestyle="--", zorder=6 if is_ctrl else 2)

        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$I(t)$")
        ax.set_yscale(_it_plot_yscale(selected_runs))
        # ax.grid(True, linestyle=":", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.grid(False)
        fig.tight_layout()
        canvas_fig.draw()

    def _init_signal_axes(self) -> None:
        self.ax_disp.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp."), fontsize=14)
        # self.ax_disp.grid(True, linestyle=":", color="#bfbfbf", linewidth=0.6, alpha=0.6)
        self.ax_disp.grid(False)
        self.ax_disp.tick_params(labelbottom=False, labelsize=12)
        self.ax_vel.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel."), fontsize=14)
        self.ax_vel.set_xlabel("Time (s)", fontsize=14)
        # self.ax_vel.grid(True, linestyle=":", color="#bfbfbf", linewidth=0.6, alpha=0.6)
        self.ax_vel.grid(False)
        self.ax_vel.tick_params(labelsize=12)
        self.sig_fig.suptitle("Selecciona casos y presiona  Plot Señales ▶")

    def _init_It_axis(self) -> None:
        self.ax_It.set_xlabel(r"$t$ (s)", fontsize=14)
        self.ax_It.set_ylabel(r"$I(t)$", fontsize=14)
        # self.ax_It.grid(True, linestyle=":", color="#bfbfbf", linewidth=0.6, alpha=0.6)
        self.ax_It.grid(False)
        self.ax_It.tick_params(labelsize=12)
        self.It_fig.suptitle("Selecciona casos y presiona  Plot I_t ▶")

    # ── SIGNAL PLOT ───────────────────────────────────────────────────────────
    def _plot_signals(self) -> None:
        if not hasattr(self, "sig_canvas"):
            messagebox.showinfo("Sin panel", "Este formato no tiene panel de señales.",
                                parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("Sin selección",
                                   "Selecciona al menos un caso en la tabla.",
                                   parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.ax_disp.cla()
        self.ax_vel.cla()

        use_fixed = getattr(self, "_persistent_color_var", None)
        use_fixed = use_fixed.get() if use_fixed is not None else True

        # Dynamic colormap — recalculated on selected cases only
        if not use_fixed:
            dyn_vals = [c["label_val"] for c in selected
                        if c.get("group", "") != "control" and np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]
            dyn_idx  = {id(c): i for i, c in enumerate(
                [c for c in selected if c.get("group", "") != "control"])}

        _invert = getattr(self, "_invert_order_var", None)
        _invert = _invert.get() if _invert is not None else False
        _lw_nc  = self._sig_lw_var.get()  if hasattr(self, "_sig_lw_var")  else 0.9
        _alp_nc = self._sig_alpha_var.get() if hasattr(self, "_sig_alpha_var") else 0.5
        # No-control primero (zorder incremental), control siempre al final (zorder 100)
        _non_ctrl = [c for c in selected if not self._is_control(c)]
        _ctrl_lst = [c for c in selected if self._is_control(c)]
        _non_ctrl = list(reversed(_non_ctrl)) if _invert else _non_ctrl
        _plot_order = _non_ctrl + _ctrl_lst
        for ci, c in enumerate(_plot_order):
            is_ctrl  = self._is_control(c)
            if is_ctrl:
                clr = "red"
            elif use_fixed:
                clr = c.get("_color", color_azul)
            else:
                lv_dyn = c.get("label_val", float("nan"))
                clr = dyn_cmap(dyn_norm(lv_dyn)) if np.isfinite(lv_dyn) else color_azul
            lk   = c.get("label_key", "")
            lv   = c.get("label_val", float("nan"))
            if is_ctrl:
                lbl = "control"
            elif np.isfinite(lv):
                lbl = f"{_col_header(lk)}={lv:.3g}"
            else:
                lbl = c.get("group", "?")
            lw   = 2.2   if is_ctrl else _lw_nc
            _ctrl_zo  = int(self._ctrl_zo_var.get())  if hasattr(self, "_ctrl_zo_var")  else 100
            _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
            zo    = _ctrl_zo  if is_ctrl else (3 + ci)
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            for sig, ax in (("Axial_disp", self.ax_disp), ("Axial_vel", self.ax_vel)):
                data = c.get("signals", {}).get(sig) or c.get(sig)
                if data is None:
                    continue
                t, y = data
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=clr, lw=lw,
                        alpha=alpha, label=lbl,
                        zorder=zo, rasterized=True)

        self.ax_disp.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp."), fontsize=14)
        # self.ax_disp.grid(False, linestyle="--", alpha=0.4)
        self.ax_disp.grid(False)
        self.ax_disp.tick_params(labelbottom=False, labelsize=12)
        self.ax_disp.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_vel.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel."), fontsize=14)
        self.ax_vel.set_xlabel("Time (s)", fontsize=14)
        # self.ax_vel.grid(False, linestyle="--", alpha=0.4)
        self.ax_vel.grid(False)
        self.ax_vel.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_vel.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.3g}"))

        n = len(selected)
        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        if n <= 12:
            self.ax_disp.legend(fontsize=9, framealpha=0.7, loc="upper left")
            self.ax_vel.legend(fontsize=9, framealpha=0.7, loc="upper left")

        # Colorbar horizontal — escala global (fijo) o sobre seleccionados (dinámico)
        if use_fixed:
            cbar_cases = self.cases
        else:
            cbar_cases = selected
        cb_vals = [c.get("label_val", float("nan"))
                   for c in cbar_cases if c.get("group", "") != "control"]
        finite_vals = [v for v in cb_vals if np.isfinite(v)]
        if len(finite_vals) >= 2:
            cmap_s = matplotlib.colormaps["viridis"]
            norm_s = mcolors.Normalize(vmin=min(finite_vals), vmax=max(finite_vals))
            sm = cm.ScalarMappable(cmap=cmap_s, norm=norm_s)
            sm.set_array([])
            if self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
            self._cbar = self.sig_fig.colorbar(
                sm, ax=[self.ax_disp, self.ax_vel],
                label=lk_disp, shrink=0.85,
                orientation="horizontal", pad=0.08,
            )
            self._cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.3g}")
            self._cbar.update_ticks()

        self.sig_fig.suptitle(f"{lk_disp}  —  {n} caso(s)")
        self._draw_reference_lines({"Señales: disp": self.ax_disp, "Señales: vel": self.ax_vel})
        self.sig_canvas.draw()

        # Switch to Signals tab if Notebook exists
        if hasattr(self, "_nb"):
            self._nb.select(0)

    def _plot_forces(self) -> None:
        if not hasattr(self, "force_canvas"):
            messagebox.showinfo("Sin panel", "Este formato no tiene panel de fuerzas.", parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("Sin selección", "Selecciona al menos un caso en la tabla.", parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        for ax in (self.ax_force_1, self.ax_force_2, self.ax_force_3):
            ax.cla()

        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True
        _invert = getattr(self, "_invert_order_var", None)
        _invert = _invert.get() if _invert is not None else False
        _alp_nc = self._sig_alpha_var.get() if hasattr(self, "_sig_alpha_var") else 0.5
        _lw_nc = self._sig_lw_var.get() if hasattr(self, "_sig_lw_var") else 0.9
        _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _ctrl_zo = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var") else 100

        non_ctrl = [c for c in selected if not self._is_control(c)]
        ctrl_lst = [c for c in selected if self._is_control(c)]
        non_ctrl = list(reversed(non_ctrl)) if _invert else non_ctrl
        plot_order = non_ctrl + ctrl_lst

        if not use_fixed:
            dyn_vals = [c.get("label_val", float("nan")) for c in non_ctrl if np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]

        for ci, c in enumerate(plot_order):
            data = c.get("forces", {}).get("res_R_p")
            if data is None:
                continue
            t, y = data
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, np.newaxis]
            if y_arr.shape[1] < 3:
                continue
            is_ctrl = self._is_control(c)
            if is_ctrl:
                color = "red"
            elif use_fixed:
                color = c.get("_color", color_azul)
            else:
                lv = c.get("label_val", float("nan"))
                color = dyn_cmap(dyn_norm(lv)) if np.isfinite(lv) else color_azul
            label = "control" if is_ctrl else c.get("group", "case")
            lw = 2.2 if is_ctrl else _lw_nc
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            zorder = _ctrl_zo if is_ctrl else (3 + ci)
            self.ax_force_1.plot(t[::DECIMATE], y_arr[::DECIMATE, 0], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)
            self.ax_force_2.plot(t[::DECIMATE], y_arr[::DECIMATE, 1], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)
            self.ax_force_3.plot(t[::DECIMATE], y_arr[::DECIMATE, 2], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)

        self.ax_force_1.tick_params(labelbottom=False)
        self.ax_force_2.tick_params(labelbottom=False)
        self.ax_force_1.legend(fontsize=8, framealpha=0.7, loc="upper left")
        self.ax_force_2.legend(fontsize=8, framealpha=0.7, loc="upper left")
        self.ax_force_3.legend(fontsize=8, framealpha=0.7, loc="upper left")

        # Colorbar
        lk_f = _col_header(selected[0]["label_key"]) if selected else ""
        if use_fixed:
            cbar_cases_f = self.cases
        else:
            cbar_cases_f = selected
        cb_vals_f = [c.get("label_val", float("nan")) for c in cbar_cases_f if c.get("group", "") != "control"]
        finite_f  = [v for v in cb_vals_f if np.isfinite(v)]
        if self._force_cbar is not None:
            try:
                self._force_cbar.remove()
            except Exception:
                pass
            self._force_cbar = None
        if len(finite_f) >= 2:
            sm_f = cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"],
                                     norm=mcolors.Normalize(vmin=min(finite_f), vmax=max(finite_f)))
            sm_f.set_array([])
            self._force_cbar = self.force_fig.colorbar(
                sm_f, ax=[self.ax_force_1, self.ax_force_2, self.ax_force_3],
                label=lk_f, shrink=0.85, orientation="horizontal", pad=0.08)
            self._force_cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.3g}")
            self._force_cbar.update_ticks()

        self.force_fig.suptitle(f"res_R_p — {len(selected)} caso(s)")
        self._draw_reference_lines({"Fuerzas: F1": self.ax_force_1, "Fuerzas: F2": self.ax_force_2,
                                    "Fuerzas: F3": self.ax_force_3})
        self.force_canvas.draw()

        if hasattr(self, "_nb"):
            try:
                self._nb.select(1)
            except Exception:
                pass

    def _plot_deflex(self) -> None:
        if not hasattr(self, "deflex_canvas"):
            messagebox.showinfo("Sin panel", "Este archivo no contiene datos Out_Deflex.", parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("Sin selección", "Selecciona al menos un caso en la tabla.", parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        self.ax_deflex_d.cla()
        self.ax_deflex_v.cla()

        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True
        _invert   = getattr(self, "_invert_order_var", None)
        _invert   = _invert.get() if _invert is not None else False
        _alp_nc   = self._sig_alpha_var.get()  if hasattr(self, "_sig_alpha_var")  else 0.5
        _lw_nc    = self._sig_lw_var.get()     if hasattr(self, "_sig_lw_var")    else 0.9
        _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _ctrl_zo  = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var") else 100

        non_ctrl   = [c for c in selected if not self._is_control(c)]
        ctrl_lst   = [c for c in selected if self._is_control(c)]
        non_ctrl   = list(reversed(non_ctrl)) if _invert else non_ctrl
        plot_order = non_ctrl + ctrl_lst

        if not use_fixed:
            dyn_vals = [c.get("label_val", float("nan")) for c in non_ctrl
                        if np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]

        _disp_key = "Axial_disp_out_deflex"
        _vel_key  = "Axial_vel_out_deflex"

        for ci, c in enumerate(plot_order):
            od = c.get("out_deflex", {})
            is_ctrl = self._is_control(c)
            color   = "red" if is_ctrl else (
                c.get("_color", color_azul) if use_fixed else
                (dyn_cmap(dyn_norm(c["label_val"])) if np.isfinite(c.get("label_val", float("nan"))) else color_azul)
            )
            lk  = c.get("label_key", "")
            lv  = c.get("label_val", float("nan"))
            lbl = "control" if is_ctrl else (
                f"{_col_header(lk)}={lv:.3g}" if np.isfinite(lv) else c.get("group", "?")
            )
            lw    = 2.2 if is_ctrl else _lw_nc
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            zo    = _ctrl_zo  if is_ctrl else (3 + ci)

            for sig, ax in ((_disp_key, self.ax_deflex_d), (_vel_key, self.ax_deflex_v)):
                data = od.get(sig)
                if data is None:
                    continue
                t, y = data
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=lw,
                        alpha=alpha, label=lbl, zorder=zo, rasterized=True)

        self.ax_deflex_d.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp (out deflex)"), fontsize=14)
        self.ax_deflex_d.tick_params(labelbottom=False, labelsize=12)
        self.ax_deflex_d.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_d.grid(False)
        self.ax_deflex_v.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel (out deflex)"), fontsize=14)
        self.ax_deflex_v.set_xlabel("Time (s)", fontsize=14)
        self.ax_deflex_v.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_v.grid(False)

        n = len(selected)
        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        if n <= 12:
            self.ax_deflex_d.legend(fontsize=9, framealpha=0.7, loc="upper left")
            self.ax_deflex_v.legend(fontsize=9, framealpha=0.7, loc="upper left")

        # Colorbar
        if use_fixed:
            cbar_cases_d = self.cases
        else:
            cbar_cases_d = selected
        cb_vals_d = [c.get("label_val", float("nan")) for c in cbar_cases_d if c.get("group", "") != "control"]
        finite_d  = [v for v in cb_vals_d if np.isfinite(v)]
        if self._deflex_cbar is not None:
            try:
                self._deflex_cbar.remove()
            except Exception:
                pass
            self._deflex_cbar = None
        if len(finite_d) >= 2:
            sm_d = cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"],
                                     norm=mcolors.Normalize(vmin=min(finite_d), vmax=max(finite_d)))
            sm_d.set_array([])
            self._deflex_cbar = self.deflex_fig.colorbar(
                sm_d, ax=[self.ax_deflex_d, self.ax_deflex_v],
                label=lk_disp, shrink=0.85, orientation="horizontal", pad=0.08)
            self._deflex_cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.3g}")
            self._deflex_cbar.update_ticks()

        self.deflex_fig.suptitle(f"Out Deflex  —  {lk_disp}  —  {n} caso(s)")
        self._draw_reference_lines({"Deflex: disp": self.ax_deflex_d, "Deflex: vel": self.ax_deflex_v})
        self.deflex_canvas.draw()

        if hasattr(self, "_nb"):
            try:
                tabs = [self._nb.tab(i, "text") for i in range(self._nb.index("end"))]
                idx  = next(i for i, t in enumerate(tabs) if "Deflex" in t)
                self._nb.select(idx)
            except (StopIteration, Exception):
                pass

    def _clear_signal_plot(self) -> None:
        if not hasattr(self, "sig_canvas"):
            return
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self.ax_disp.cla()
        self.ax_vel.cla()
        self._init_signal_axes()
        self.sig_canvas.draw()

    # ── I_t PLOT ──────────────────────────────────────────────────────────────
    def _plot_It(self) -> None:
        if not hasattr(self, "It_canvas"):
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("Sin selección",
                                   "Selecciona al menos un caso en la tabla.",
                                   parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        run_filter = self._selected_run_filter()
        use_no_far = self._use_no_far_var.get() if hasattr(self, "_use_no_far_var") else False
        td_key     = "t_d_no_FAR" if use_no_far else "t_d"

        # Filtrar runs según indicadores seleccionados
        runs_to_show = self._get_runs_to_show() if hasattr(self, "_get_runs_to_show") else self._all_runs
        if not runs_to_show:
            runs_to_show = self._all_runs

        vals = [c.get("label_val", float("nan")) for c in selected]

        self.ax_It.cla()

        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        plotted = False

        # Decide coloring strategy:
        #   · varios indicadores  → color por indicador (tab10)
        #   · un solo indicador   → color por caso (viridis, fijo o dinámico según toggle)
        n_indicators = len({_run_indicator_prefix(rn) for rn in runs_to_show})
        color_by_case = (n_indicators == 1)
        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True

        # Paleta dinámica (sólo casos no-control) para color_by_case + no fijo
        non_ctrl_cases = [c for c in selected if not self._is_control(c)]
        if not use_fixed and non_ctrl_cases:
            _dyn_cmap = matplotlib.colormaps.get_cmap("viridis")
            _dyn_colors = {id(c): _dyn_cmap(i / max(len(non_ctrl_cases) - 1, 1))
                           for i, c in enumerate(non_ctrl_cases)}
        else:
            _dyn_colors = {}

        _it_alp_nc  = self._sig_alpha_var.get()  if hasattr(self, "_sig_alpha_var")  else 0.5
        _it_lw_nc   = self._sig_lw_var.get()      if hasattr(self, "_sig_lw_var")      else 0.9
        _it_ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _it_ctrl_zo  = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var")  else 100
        _invert_it = getattr(self, "_invert_order_var", None)
        _invert_it = _invert_it.get() if _invert_it is not None else False
        # No-control primero, control al final (siempre encima)
        _nc_pairs   = [(c, v) for c, v in zip(selected, vals) if not self._is_control(c)]
        _ctrl_pairs = [(c, v) for c, v in zip(selected, vals) if self._is_control(c)]
        _nc_pairs   = list(reversed(_nc_pairs)) if _invert_it else _nc_pairs
        _it_order   = _nc_pairs + _ctrl_pairs
        n_sel = max(len(selected), 1)
        for ci, (c, pv) in enumerate(_it_order):
            is_ctrl = self._is_control(c)
            case_alpha = _it_alp_nc
            for rn in runs_to_show:
                run_data = c.get("runs", {}).get(rn)
                if run_data is None:
                    continue
                t   = run_data["t"]
                I_t = run_data["I_t"]
                if t.size == 0 or I_t.size == 0:
                    continue
                if is_ctrl:
                    if color_by_case:
                        color = "red"   # un indicador → control siempre rojo
                    else:
                        ind_prefix = _run_indicator_prefix(rn)
                        ind_color  = self._ind_color_map.get(ind_prefix, None) if hasattr(self, "_ind_color_map") else None
                        color = ind_color if ind_color is not None else "red"
                elif color_by_case:
                    # Respeta toggle fijo/dinámico igual que señales
                    if use_fixed:
                        color = c.get("_color", (0.5, 0.5, 0.5))
                    else:
                        color = _dyn_colors.get(id(c), c.get("_color", (0.5, 0.5, 0.5)))
                else:
                    ind_prefix = _run_indicator_prefix(rn)
                    color = self._ind_color_map.get(ind_prefix, (0.5, 0.5, 0.5)) if hasattr(self, "_ind_color_map") else c.get("_color", (0.5, 0.5, 0.5))
                lv_str = "control" if is_ctrl else (f"{pv:.3g}" if np.isfinite(pv) else "?")
                lw = 2.2 if is_ctrl else _it_lw_nc
                self.ax_It.plot(t[::_IND_DECIMATE], I_t[::_IND_DECIMATE],
                                color=color, lw=lw,
                                alpha=_it_ctrl_alp if is_ctrl else case_alpha,
                                label=f"{lk_disp}={lv_str} | {rn}",
                                zorder=_it_ctrl_zo if is_ctrl else (3 + ci),
                                rasterized=True)
                # t_d / t_d_no_FAR vlines
                for key, style in (("t_d", "--"), ("t_d_no_FAR", ":")):
                    td = run_data.get(key, np.array([]))
                    if td.size > 0:
                        self.ax_It.axvline(td[0], color=color, lw=2.2,
                                           linestyle=style,
                                           alpha=_it_ctrl_alp if is_ctrl else case_alpha,
                                           zorder=6)
                        if t.size > 1:
                            y_td = float(np.interp(td[0], t, I_t))
                            self.ax_It.scatter([td[0]], [y_td], s=24, color=color,
                                               edgecolor="black", linewidths=0.3, zorder=7)
                plotted = True

        # t_GT reference
        self.ax_It.axvline(_IND_T_GT, color="black", lw=2.0,
                           linestyle=":", label=rf"$t_{{GT}}$={_IND_T_GT:.2f}s", zorder=5)

        self.ax_It.set_xlabel(r"$t$ (s)", fontsize=14)
        self.ax_It.set_ylabel(r"$I(t)$", fontsize=14)
        # self.ax_It.grid(False, linestyle="--", alpha=0.3)
        self.ax_It.grid(False)
        self.ax_It.set_yscale(_it_plot_yscale(runs_to_show))

        far_txt = " (no FAR)" if use_no_far else ""
        run_txt = run_filter or "(todos)"
        self.ax_It.set_title(f"I_t(t){far_txt}  —  run: {run_txt}", fontsize=13)

        n = len(selected) * len(runs_to_show)
        if n <= 10 and plotted:
            self.ax_It.legend(fontsize=14, loc="upper left")

        self._draw_reference_lines({"I_t": self.ax_It})
        self.It_fig.tight_layout()
        self.It_canvas.draw()

        # Switch to I_t tab if Notebook exists
        if hasattr(self, "_nb"):
            self._nb.select(1)

    # ── SUMMARY PLOT ──────────────────────────────────────────────────────────
    def _refresh_summary(self) -> None:
        if not self._summary_entries:
            return
        choice = self._sum_combo.get()
        entry  = next((e for e in self._summary_entries if e[0] == choice), None)
        if entry is None:
            return

        label, func, extra = entry
        fig = None

        try:
            if isinstance(extra, tuple):
                # plot_convergence*(cases, *args) — returns a Figure
                fig = _capture_new_figure(func, self.cases, *extra)
            elif isinstance(extra, dict):
                if func == "_noise_overlay":
                    fig = _build_noise_overlay_fig(self.cases, extra["signal"])
                else:
                    kw = dict(extra)
                    kw["out_dir"] = None   # preview only
                    fig = _capture_new_figure(func, **kw)

            if fig is None:
                messagebox.showwarning("Sin figura",
                                       f"No se pudo generar la figura:\n{label}",
                                       parent=self.root)
                return

            fig.set_size_inches(4.5, 4.5)
            _embed_figure(fig, self._sum_canvas_frame,
                          self._sum_toolbar_frame, self._fig_holder, "summary")

        except Exception as exc:
            messagebox.showerror("Error al generar figura",
                                 f"{type(exc).__name__}: {exc}", parent=self.root)

    def _save_summary(self) -> None:
        fig = self._fig_holder.get("summary")
        if fig is None:
            messagebox.showinfo("Sin figura",
                                "Primero presiona ▶ Preview para generar una figura.",
                                parent=self.root)
            return
        out_dir = os.path.join(os.path.dirname(self.h5_path), "figs_indicators")
        os.makedirs(out_dir, exist_ok=True)
        label   = self._sum_combo.get()
        fname   = _sanitize(label) + ".png"
        path    = os.path.join(out_dir, fname)
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Guardado", f"Figura guardada en:\n{path}", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Error al guardar", str(exc), parent=self.root)

    # ── OPEN FILE ─────────────────────────────────────────────────────────────
    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Abrir archivo HDF5 DOE",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.h5_path),
        )
        if not path:
            return
        try:
            self._load_file(path)
        except Exception as exc:
            messagebox.showerror("Error al cargar", str(exc), parent=self.root)
            return

        # Rebuild UI
        for w in self.root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._fig_holder.clear()
        self._cbar    = None
        self._sort_col = None
        self._sort_rev = False
        self._iid_to_case.clear()
        self._build_ui()


# ==============================================================================
# CLI + MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="doe_unified_selector — Selector interactivo unificado para HDF5 DOE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python doe_unified_selector.py
  python doe_unified_selector.py --h5 DOE_xxx/doe_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_indicator_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_noise_indicator_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_model_snr_results.h5
""",
    )
    p.add_argument("--h5", default=None, metavar="PATH",
                   help="Ruta al archivo .h5 a abrir (omitir → FileDialog)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    h5_path = args.h5
    if not h5_path:
        # Sin parent explicito: tkinter crea y maneja su propio root implicito,
        # mas confiable en Windows que un root manual withdraw()-eado.
        h5_path = filedialog.askopenfilename(
            title="Selecciona un archivo HDF5 DOE",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not h5_path:
            print("[INFO] No se seleccionó ningún archivo. Saliendo.")
            return

    if not os.path.isfile(h5_path):
        print(f"[ERROR] Archivo no encontrado: {h5_path}")
        return

    print(f"[INFO] Cargando: {h5_path}")
    h5_type = detect_h5_type(h5_path)
    print(f"[INFO] Formato detectado: {_TYPE_LABELS.get(h5_type, h5_type)}")

    root = tk.Tk()
    root.update()  # pinta la ventana ya, antes de la carga pesada del .h5
    DoeSelectorUnifiedApp(root, h5_path)
    root.mainloop()


if __name__ == "__main__":
    main()

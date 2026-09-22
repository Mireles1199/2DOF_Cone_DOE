"""doe_indicator_plotter.py — Visualiza resultados de doe_indicator_results.h5.

Figuras generadas:
  1. t_d vs parámetro DOE  — tiempo de detección por indicador vs LABEL_KEY
  2. I_t(t) overlay        — curvas I_t(t) superpuestas (una figura por run_name)

Uso desde terminal:
  python doe_indicator_plotter.py --ind_results DOE_xxx\\doe_indicator_results.h5
  python doe_indicator_plotter.py --ind_results ...  --plot-td
  python doe_indicator_plotter.py --ind_results ...  --plot-It
  python doe_indicator_plotter.py --ind_results ...  --plot-td --plot-It --show
  python doe_indicator_plotter.py --ind_results ...  --run_name maxent_revo_dec7_1step
  python doe_indicator_plotter.py --ind_results ...  --out-dir figs/
  python doe_indicator_plotter.py --ind_results ...  --list

Uso desde VS Code (sin argumentos):
  Editar el bloque CONFIG y ejecutar directamente.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import re
import sys
from typing import Any, Dict, List, Optional

import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# SKILL: indicator-plot-style
# ==============================================================================

def _hls(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return (r, g, b)

color_red    = _hls(346, 0.45, 0.99)
color_orange = _hls(36,  0.45, 0.99)
color_purple = _hls(279, 0.36, 0.99)
color_verde  = _hls(98,  0.36, 0.99)
color_azul   = _hls(207, 0.41, 0.56)
color_gray   = (0.55, 0.55, 0.55)

_RUN_COLORS = [color_azul, color_orange, color_purple, color_verde, color_red]


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    width = base_width * ncols * scale
    return (width, width * 0.8)


def configurar_estilo_global() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 25,   "axes.labelsize": 25,
        "xtick.labelsize": 23,  "ytick.labelsize": 23, "legend.fontsize": 23,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.linewidth": 0.8,  "grid.linewidth": 0.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in",  "ytick.direction": "in",
        "xtick.major.size": 4,  "ytick.major.size": 4,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
        "legend.frameon": False, "legend.loc": "best",
        "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "savefig.transparent": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "path.simplify": True, "path.simplify_threshold": 1.0,
    })
configurar_estilo_global()

# ==============================================================================
# CONFIG — editar aquí para lanzar desde VS Code sin argumentos
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = SCRIPT_DIR

DOE_NAME   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
# DOE_NAME   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"
# None → auto-detectar desde attrs del HDF5; str → clave explícita ej. "$dxl_size$"
LABEL_KEY: Optional[str] = None

# Tiempo de onset de chatter (línea de referencia en figura t_d)
T_GT = 5.365770208787228   # [s]

# Filtro de run_names: None → todos los runs; str → solo ese run_name
RUN_NAME_FILTER: Optional[str] = None

# Índice del caso control dentro de `cases` (se destaca con otro marcador)
CONTROL_IDX = 4

# Decimación para curvas I_t(t) (1 = sin decimación; >1 = más rápido)
DECIMATE = 1

# Ventana temporal del RMS móvil [s]. Sube o baja este valor para controlar el suavizado.
f = 150
RMS_WINDOW_SECONDS = 20/f

# Umbral de RMS para la figura móvil.
# "geometric_mean" y "log_median" son las opciones recomendadas para escala log.
RMS_THRESHOLD_MODE = "geometric_mean"

# Figuras a generar cuando se lanza sin args desde VS Code
PLOT_TD = False    # Figura t_d vs parámetro DOE
PLOT_IT = False    # Figura I_t(t) overlay por run_name (con t_d y t_d_no_FAR)
PLOT_SIGNALS = True  # Figuras de superposición de señales (Axial_disp y Axial_vel)
PLOT_SIGNAL_RMS = True  # Figuras RMS móvil de señales (Axial_disp y Axial_vel)
PLOT_RMS_CROSSING = False  # Figuras de tiempo de cruce RMS móvil vs parámetro DOE
RMS_TREND_MODE = "none"  # "none", "linear", "poly", "exp" o "power_law"
RMS_TREND_POLY_DEGREE = 3   # Grado usado cuando RMS_TREND_MODE = "poly"

PLOT_LOG_RMS = False        # Figuras log10(RMS) vs tiempo — crecimiento exponencial como recta
PLOT_LOG_RMS_CROSSING = False  # Figuras tiempo de transicion en log(RMS) vs parametro DOE
# Ventana estable inicial [s] para normalizar el RMS — separada por señal.
LOG_RMS_NORM_WINDOW_DISP = 5   # [s] ventana de referencia para Axial_disp
LOG_RMS_NORM_WINDOW_VEL  = 5   # [s] ventana de referencia para Axial_vel
# Criterio de pendiente sostenida para detectar inicio de exponencial:
# N sigmas sobre la pendiente en el tramo inicial del mismo caso.
LOG_RMS_SLOPE_N_SIGMA = 3.0
# Duracion minima [s] que la pendiente debe sostenerse para contar como inicio.
LOG_RMS_SLOPE_MIN_DUR = 0.5
SHOW    = True    # plt.show() al final
OUT_DIR: Optional[str] = None   # None → solo mostrar; str → guardar figuras
# OUT_DIR = os.path.join(BASE_DIR, DOE_NAME, "figs_indicators")

# ==============================================================================
# HELPERS
# ==============================================================================

def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _pretty_indicator_name(indicator: str) -> str:
    """Devuelve un nombre legible para mostrar en figuras y leyendas."""
    mapping = {
        "green_integral": "Green Integral",
        "green": "Green",
        "sst_svd": "SST-SVD",
        "maxent": "MaxEnt",
        "rms_cv": "RMS-CV",
        "ssq_chatter": "SSQ",
        "emd_hht": "EMD-HHT",
        "lollipop": "Lollipop",
        "td": "Detection Time",
    }
    key = indicator.lower()
    if key in mapping:
        return mapping[key]
    if key.startswith("green_integral") or key.startswith("green"):
        return "Green Integral"
    if key.startswith("maxent"):
        return "MaxEnt"
    if key.startswith("rms_cv"):
        return "RMS-CV"
    if key.startswith("ssq"):
        return "SSQ"
    return indicator.replace("_", " ").replace("-", " ").title()


def _pretty_signal_name(signal_name: str) -> str:
    """Devuelve un nombre legible para señales de desplazamiento y velocidad."""
    mapping = {
        "Axial_disp": "Axial Displacement",
        "Axial_vel": "Axial Velocity",
    }
    if signal_name in mapping:
        return mapping[signal_name]
    key = signal_name.lower()
    if "disp" in key:
        return "Axial displacement"
    if "vel" in key:
        return "Axial velocity"
    return signal_name.replace("_", " ").replace("-", " ").title()


def _pretty_label_key(label_key: str) -> str:
    """Convierte nombres técnicos del DOE en etiquetas más legibles."""
    mapping = {
        "dxl_size": "Dexel size",
        "$dxl_size$": "Dexel size",
    }
    if label_key in mapping:
        return mapping[label_key]
    cleaned = label_key.strip("$")
    if cleaned in mapping:
        return mapping[cleaned]
    return cleaned.replace("_", " ").replace("-", " ").title()


def _ind_h5_default() -> str:
    return os.path.join(BASE_DIR, DOE_NAME, "doe_indicator_results.h5")

# ==============================================================================
# LECTURA DE RESULTADOS
# ==============================================================================

def load_indicator_results(h5_path: str) -> List[Dict[str, Any]]:
    """Lee doe_indicator_results.h5 → lista de dicts por caso.

    Cada dict:
      {group, label_key, label_val, var_val,
    runs: {run_name: {t, I_t, t_d, t_d_no_FAR, attrs}},
    signals: {Axial_disp, Axial_vel: {t, y}}}
    """
    _SIGNAL_NAMES = {"Axial_disp", "Axial_vel"}
    cases = []

    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(k for k in f.keys() if k.startswith("case_")):
            grp        = f[grp_name]
            case_attrs = dict(grp.attrs)

            lk = str(case_attrs.get("label_key", ""))
            try:
                lv = float(case_attrs.get("label_val", float("nan")))
            except (TypeError, ValueError):
                lv = float("nan")

            var_val = {
                k.strip("$"): v for k, v in case_attrs.items()
                if k.startswith("$") and k.endswith("$")
            }

            runs: Dict[str, Any] = {}
            signals: Dict[str, Any] = {}
            for rname in grp.keys():
                signal_key = None
                if rname in _SIGNAL_NAMES:
                    signal_key = rname
                elif rname.lower() in {"axial_disp", "axial_velocity", "axial_vel"}:
                    signal_key = "Axial_disp" if "disp" in rname.lower() else "Axial_vel"
                if signal_key is not None:
                    rgrp = grp[rname]
                    if isinstance(rgrp, h5py.Group):
                        t = rgrp["time"][()] if "time" in rgrp else rgrp["t"][()] if "t" in rgrp else np.array([])
                        if "values" in rgrp:
                            y = rgrp["values"][()]
                        elif "y" in rgrp:
                            y = rgrp["y"][()]
                        elif "signal" in rgrp:
                            y = rgrp["signal"][()]
                        else:
                            y = np.array([])
                        signals[signal_key] = {"t": t, "y": y}
                    continue
                rgrp = grp[rname]
                if not isinstance(rgrp, h5py.Group):
                    continue
                runs[rname] = {
                    "t":          rgrp["t"][()]          if "t"          in rgrp else np.array([]),
                    "I_t":        rgrp["I_t"][()]        if "I_t"        in rgrp else np.array([]),
                    "t_d":        rgrp["t_d"][()]        if "t_d"        in rgrp else np.array([]),
                    "t_d_no_FAR": rgrp["t_d_no_FAR"][()] if "t_d_no_FAR" in rgrp else np.array([]),
                    "attrs":      dict(rgrp.attrs),
                }

            cases.append({
                "group":     grp_name,
                "label_key": lk,
                "label_val": lv,
                "var_val":   var_val,
                "runs":      runs,
                "signals":   signals,
            })

    # fallback: leer label_val de var_val si no estaba en attr
    for c in cases:
        if np.isnan(c["label_val"]) and c["label_key"] and c["label_key"] in c["var_val"]:
            try:
                c["label_val"] = float(c["var_val"][c["label_key"]])
            except (TypeError, ValueError):
                pass

    cases.sort(
        key=lambda x: x["label_val"] if not np.isnan(x["label_val"]) else float("inf")
    )
    return cases


def _resolve_label_key(cases: List[Dict[str, Any]], override: Optional[str] = None) -> str:
    """Devuelve el LABEL_KEY a usar: override > attr guardado > primer var_val."""
    if override:
        return override
    for c in cases:
        if c["label_key"]:
            return c["label_key"]
    for c in cases:
        keys = list(c["var_val"].keys())
        if keys:
            return keys[0]
    return "case_idx"


def _x_values(cases: List[Dict[str, Any]], lk: str) -> List[float]:
    """Extrae valores numéricos del eje X para cada caso."""
    xs = []
    for c in cases:
        if lk == c.get("label_key") and not np.isnan(c["label_val"]):
            xs.append(c["label_val"])
        else:
            raw = c["var_val"].get(lk)
            try:
                xs.append(float(raw) if raw is not None else float("nan"))
            except (TypeError, ValueError):
                xs.append(float("nan"))
    return xs


def _format_sci(value: float, precision: int = 2) -> str:
    """Formatea un número en notación científica compacta."""
    if np.isnan(value):
        return "?"
    return f"{value:.{precision}e}"


def _use_log_scale_for_it(run_name: str) -> bool:
    """Devuelve True solo para familias green, ssq y sst_svd en las figuras I(t)."""
    key = run_name.lower()
    return key.startswith("green") or key.startswith("ssq") or key.startswith("sst_svd")


def _colorbar_ticks_from_data(values: List[float], normalization: mcolors.Normalize) -> List[float]:
    clean = sorted({float(v) for v in values if not np.isnan(v)})
    if not clean:
        return []
    if isinstance(normalization, mcolors.LogNorm):
        positive = [v for v in clean if v > 0]
        if not positive:
            return clean[: min(5, len(clean))]
        if len(positive) == 1:
            return positive
        count = min(5, len(positive))
        ticks = np.geomspace(positive[0], positive[-1], num=count)
        ticks = [float(t) for t in ticks]
        ticks[0] = positive[0]
        ticks[-1] = positive[-1]
        return ticks

    if len(clean) <= 5:
        return clean
    indices = np.linspace(0, len(clean) - 1, num=5)
    return [clean[int(round(i))] for i in indices]


def _colorbar_ticklabels(ticks: List[float]) -> List[str]:
    return [_format_sci(t, precision=2) for t in ticks]


def _decorate_control_colorbar(cbar, cb_ticks: List[float], xs: List[float]) -> None:
    """Ensures the control value appears as a red tick and a red horizontal marker."""
    if not (0 <= CONTROL_IDX < len(xs)):
        return

    control_value = xs[CONTROL_IDX]
    if np.isnan(control_value):
        return

    if not any(np.isclose(tick_value, control_value, rtol=1e-8, atol=1e-12) for tick_value in cb_ticks):
        cb_ticks = sorted(list(cb_ticks) + [control_value])

    cbar.set_ticks(cb_ticks)
    cb_labels = _colorbar_ticklabels(cb_ticks)
    control_tick_idx = None
    for idx, tick_value in enumerate(cb_ticks):
        if np.isclose(tick_value, control_value, rtol=1e-8, atol=1e-12):
            control_tick_idx = idx
            break
    cbar.set_ticklabels(cb_labels)
    cbar.ax.axhline(control_value, color=color_red, lw=2.0, linestyle="-", alpha=0.95)

    if control_tick_idx is not None:
        try:
            cbar.ax.get_yticklabels()[control_tick_idx].set_color(color_red)
            # cbar.ax.get_yticklabels()[control_tick_idx].set_fontweight("bold")
        except IndexError:
            pass


def _plot_signal_overlay(
    cases: List[Dict[str, Any]],
    signal_name: str,
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    """Una figura por señal: y(t) de todos los casos coloreados por label_val.

    La lógica, la barra de color y el estilo siguen plot_It_overlay.
    """

    if not cases:
        print("  No hay casos.")
        return

    xs = _x_values(cases, label_key)
    
    all_signals = sorted({signal_name})
    if run_name_filter:
        all_signals = [signal_name]
    if not all_signals:
        print(f"  No se encontraron señales (filter={run_name_filter}).")
        return

    param_vals = [x for x in xs if not np.isnan(x)]
    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        vmin = min(param_vals) if param_vals else 0.0
        vmax = max(param_vals) if param_vals else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    signal_label = _pretty_signal_name(signal_name)
    readable_title = signal_label
    fig, ax = plt.subplots(figsize=(fig_size(scale=3.0)[0] * 1.0, fig_size(scale=3.0)[1]))

    control_entry = None
    for case_idx, (c, pv) in enumerate(zip(cases, xs)):
        sig_data = c.get("signals", {}).get(signal_name)
        if sig_data is None:
            continue
        t = sig_data.get("t", np.array([]))
        y = sig_data.get("y", np.array([]))
        if t.size == 0 or y.size == 0:
            continue

        color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
        lv_str = _format_sci(pv, precision=2)

        if case_idx == CONTROL_IDX:
            control_entry = (t, y, lv_str)
        else:
            ax.plot(t[::DECIMATE], y[::DECIMATE],
                color=color, lw=1.9, alpha=0.95,
                label=f"{lv_str}",
                rasterized=True)

    if control_entry is not None:
        t, y, lv_str = control_entry
        ax.plot(t[::DECIMATE], y[::DECIMATE],
            color=color_red, lw=2.4, alpha=1.0,
            label=f"Control ={lv_str}",
            rasterized=True,)

    if not ax.lines:
        print(f"  No se encontraron datos para {signal_name}.")
        plt.close(fig)
        return

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
    if cb_ticks:
        _decorate_control_colorbar(cbar, cb_ticks, xs)
    cbar.set_label(_pretty_label_key(label_key), fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$t$ (s)")
    if signal_name == "Axial_disp":
        ax.set_ylabel(rf"{signal_label} [mm]")
    elif signal_name == "Axial_vel":
        ax.set_ylabel(rf"{signal_label} [mm/s]")
    else:
        ax.set_ylabel(signal_label)
    ax.set_title(readable_title)
    ax.legend(fontsize=13, loc="upper left", ncol=1)
    # ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_yscale("linear")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"signal_overlay_{_sanitize(signal_name)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"Signal overlay — {signal_name}")


def plot_signal_disp_overlay(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    _plot_signal_overlay(cases, "Axial_disp", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


def plot_signal_vel_overlay(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    _plot_signal_overlay(cases, "Axial_vel", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


def _estimate_sampling_frequency(t: np.ndarray) -> float:
    """Estimate the sampling frequency from a time vector."""
    if t.size < 2:
        return float("nan")

    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return 1.0 / float(np.median(dt))


def _moving_rms(y: np.ndarray, window_samples: int, hop: int = 200) -> np.ndarray:
    """Compute a centered moving RMS with a fixed window size in samples.

    hop : int
        Step between consecutive windows (default 1 = every sample).
        hop=n computes RMS at every n-th sample and linearly interpolates
        back to the original length.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    window_samples = int(max(1, window_samples))
    window_samples = min(window_samples, y.size)
    hop = int(max(1, hop))

    kernel = np.ones(window_samples, dtype=float) / float(window_samples)

    if hop == 1:
        power = np.convolve(np.square(y), kernel, mode="same")
        return np.sqrt(power)

    # Compute on subsampled grid then interpolate back
    n = y.size
    positions = np.arange(0, n, hop)
    y_sq = np.square(y)
    # Use convolve on the full signal and sample at hop positions (fast path)
    power_full = np.convolve(y_sq, kernel, mode="same")
    rms_at_positions = np.sqrt(power_full[positions])
    out = np.interp(np.arange(n), positions, rms_at_positions)
    return out

def _moving_rms_mad_clip(
    y: np.ndarray,
    window_samples: int,
    n_sigma: float = 3.0,
    hop: int = 200,
) -> np.ndarray:
    """Moving RMS with MAD-clipping inside each window to suppress spike outliers.

    For each window the algorithm:
      1. Computes median(y_window) and MAD = median(|y_window - median|).
      2. Clips samples outside  median ± n_sigma * 1.4826 * MAD  to the boundary.
      3. Computes RMS of the clipped window.

    Unlike SG, clipping does NOT smooth the carrier oscillation — it only
    removes isolated amplitude spikes that deviate far from the local median.

    Parameters
    ----------
    window_samples : int
        Window width in samples (used both for the RMS and the MAD estimate).
    n_sigma : float
        Number of equivalent Gaussian sigma beyond which a sample is clipped
        (default 3.0).  Lower → more aggressive clipping.
    hop : int
        Step between consecutive windows (default 1 = every sample).
        hop=n evaluates every n-th sample and linearly interpolates back to
        the original length for speed.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    window_samples = int(max(1, window_samples))
    window_samples = min(window_samples, y.size)
    hop = int(max(1, hop))

    half = window_samples // 2
    n = y.size
    # Pad symmetrically so border windows stay full-length
    y_pad = np.pad(y, half, mode="reflect")

    _K = 1.4826  # consistency factor: MAD × K ≈ sigma for Gaussian data

    positions = np.arange(0, n, hop)
    rms_at_positions = np.empty(len(positions), dtype=float)

    for k, i in enumerate(positions):
        seg = y_pad[i: i + window_samples]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        limit = n_sigma * _K * mad
        # Clip samples to [med - limit, med + limit]
        clipped = np.clip(seg, med - limit, med + limit)
        fig, ax = plt.subplots()
        ax.plot(seg, label="Original")
        ax.plot(clipped, label="Clipped")
        ax.legend()
        plt.show()
        rms_at_positions[k] = np.sqrt(np.mean(np.square(clipped)))

    if hop == 1:
        return rms_at_positions

    out = np.interp(np.arange(n), positions, rms_at_positions)
    return out


def _first_threshold_crossing(t: np.ndarray, y: np.ndarray, threshold: float) -> Optional[tuple[float, float]]:
    """Return the first interpolated crossing of y with a horizontal threshold."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0 or t.size != y.size or not np.isfinite(threshold):
        return None

    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if t.size < 2:
        return None

    diff = y - threshold
    if np.any(diff == 0):
        idx = int(np.flatnonzero(diff == 0)[0])
        return float(t[idx]), float(y[idx])

    sign = np.sign(diff)
    changes = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    if changes.size == 0:
        return None

    idx = int(changes[0])
    t0, t1 = float(t[idx]), float(t[idx + 1])
    y0, y1 = float(y[idx]), float(y[idx + 1])
    if y1 == y0:
        return t0, threshold

    frac = (threshold - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    tc = t0 + frac * (t1 - t0)
    return float(tc), float(threshold)


def _resolve_rms_threshold(rms_series: List[np.ndarray]) -> float:
    """Resolve the RMS threshold value for the figure."""
    values = np.concatenate([np.asarray(series, dtype=float).ravel() for series in rms_series if np.size(series) > 0])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    positive = values[values > 0]

    if RMS_THRESHOLD_MODE == "midpoint":
        return float((np.min(values) + np.max(values)) / 2.0)
    if RMS_THRESHOLD_MODE == "geometric_mean":
        if positive.size == 0:
            return float("nan")
        return float(np.sqrt(np.min(positive) * np.max(positive)))
    if RMS_THRESHOLD_MODE == "log_median":
        if positive.size == 0:
            return float("nan")
        return float(10.0 ** np.median(np.log10(positive)))
    try:
        return float(RMS_THRESHOLD_MODE)
    except (TypeError, ValueError):
        return float("nan")


def _case_rms_crossing_time(
    case: Dict[str, Any],
    signal_name: str,
    threshold: float,
    window_seconds: float = RMS_WINDOW_SECONDS,
) -> Optional[float]:
    """Return the first RMS threshold crossing time for one case and signal."""
    sig_data = case.get("signals", {}).get(signal_name)
    if sig_data is None:
        return None

    t = np.asarray(sig_data.get("t", np.array([])), dtype=float)
    y = np.asarray(sig_data.get("y", np.array([])), dtype=float)
    if t.size == 0 or y.size == 0:
        return None

    fs = _estimate_sampling_frequency(t)
    if not np.isfinite(fs) or fs <= 0:
        return None

    window_samples = int(max(1, round(window_seconds * fs)))
    if window_samples % 2 == 0:
        window_samples += 1

    rms = _moving_rms(y, window_samples)
    crossing = _first_threshold_crossing(t, rms, threshold)
    if crossing is None:
        return None

    return float(crossing[0])


def _plot_signal_rms_crossing_scatter(
    cases: List[Dict[str, Any]],
    signal_name: str,
    label_key: str,
    window_seconds: float = RMS_WINDOW_SECONDS,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
    trend_mode: str = RMS_TREND_MODE,
    trend_poly_degree: int = RMS_TREND_POLY_DEGREE,
) -> None:
    """Scatter plot of first RMS threshold crossing time versus DOE parameter."""

    if not cases:
        print("  No hay casos.")
        return

    xs = _x_values(cases, label_key)
    param_vals = [x for x in xs if not np.isnan(x)]
    if not param_vals:
        print(f"  No se encontraron valores válidos para {signal_name}.")
        return

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        norm = mcolors.Normalize(vmin=min(param_vals), vmax=max(param_vals))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    signal_label = _pretty_signal_name(signal_name)
    title = rf"RMS Reference Scatter for {signal_label}"
    fig, ax = plt.subplots(figsize=(fig_size(scale=3.0)[0] * 1.0, fig_size(scale=3.0)[1]))

    crossing_times: List[float] = []
    crossing_xs: List[float] = []
    control_entry: Optional[tuple[float, float]] = None

    plotted_rms_series: List[np.ndarray] = []
    for case_idx, (case, pv) in enumerate(zip(cases, xs)):
        sig_data = case.get("signals", {}).get(signal_name)
        if sig_data is None:
            continue

        t = np.asarray(sig_data.get("t", np.array([])), dtype=float)
        y = np.asarray(sig_data.get("y", np.array([])), dtype=float)
        if t.size == 0 or y.size == 0:
            continue

        fs = _estimate_sampling_frequency(t)
        if not np.isfinite(fs) or fs <= 0:
            continue

        window_samples = int(max(1, round(window_seconds * fs)))
        if window_samples % 2 == 0:
            window_samples += 1
        rms = _moving_rms(y, window_samples)
        
        plotted_rms_series.append(rms)

    rms_threshold = _resolve_rms_threshold(plotted_rms_series)
    if not np.isfinite(rms_threshold):
        print(f"  No se pudo resolver el umbral RMS para {signal_name}.")
        plt.close(fig)
        return

    for case_idx, (case, pv) in enumerate(zip(cases, xs)):
        crossing_time = _case_rms_crossing_time(case, signal_name, rms_threshold, window_seconds=window_seconds)
        if crossing_time is None:
            continue

        crossing_xs.append(pv)
        crossing_times.append(crossing_time)

        color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
        lv_str = _format_sci(pv, precision=2)
        if case_idx == CONTROL_IDX:
            control_entry = (pv, crossing_time)
        else:
            ax.scatter(
                [pv], [crossing_time],
                s=90,
                color=color,
                edgecolor="black",
                linewidths=0.6,
                zorder=8,
                label=lv_str,
            )

    if control_entry is not None:
        x_ctrl, t_ctrl = control_entry
        ax.scatter(
            [x_ctrl], [t_ctrl],
            marker="D",
            s=90,
            facecolors=color_red,
            edgecolors="black",
            linewidths=0.9,
            zorder=9,
            label="Control",
        )

    if not crossing_times:
        print(f"  No se encontraron cruces RMS para {signal_name}.")
        plt.close(fig)
        return

    trend_mode = str(trend_mode).strip().lower()
    if trend_mode != "none" and len(crossing_xs) >= 2:
        x_fit = np.asarray(crossing_xs, dtype=float)
        y_fit = np.asarray(crossing_times, dtype=float)
        valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0)
        x_fit = x_fit[valid]
        y_fit = y_fit[valid]
        if x_fit.size >= 2:
            x_line = np.geomspace(float(np.min(x_fit)), float(np.max(x_fit)), num=200)
            log_x_fit = np.log10(x_fit)
            log_x_line = np.log10(x_line)
            if trend_mode == "linear":
                coeffs = np.polyfit(log_x_fit, y_fit, deg=1)
                y_line = np.polyval(coeffs, log_x_line)
                trend_label = "Trend line"
            elif trend_mode == "poly":
                degree = int(max(1, trend_poly_degree))
                degree = min(degree, max(1, x_fit.size - 1))
                coeffs = np.polyfit(log_x_fit, y_fit, deg=degree)
                y_line = np.polyval(coeffs, log_x_line)
                trend_label = f"Poly trend (deg {degree})"
            elif trend_mode in {"exp", "exponential"}:
                positive = y_fit > 0
                x_fit_exp = x_fit[positive]
                y_fit_exp = y_fit[positive]
                if x_fit_exp.size >= 2:
                    coeffs = np.polyfit(np.log10(x_fit_exp), np.log(y_fit_exp), deg=1)
                    y_line = np.exp(np.polyval(coeffs, log_x_line))
                    trend_label = "Exponential trend"
                else:
                    y_line = None
                    trend_label = None
            elif trend_mode in {"power_law", "power-law", "powerlaw"}:
                positive = y_fit > 0
                x_fit_pow = x_fit[positive]
                y_fit_pow = y_fit[positive]
                if x_fit_pow.size >= 2:
                    log_y_fit = np.log(y_fit_pow)
                    coeffs = np.polyfit(np.log10(x_fit_pow), log_y_fit, deg=1)
                    y_line = np.exp(np.polyval(coeffs, log_x_line))
                    trend_label = "Power-law trend"
                else:
                    y_line = None
                    trend_label = None
            else:
                y_line = None
                trend_label = None

            if y_line is not None:
                ax.plot(
                    x_line,
                    y_line,
                    color=color_azul,
                    lw=1.6,
                    linestyle="--",
                    alpha=0.9,
                    zorder=6,
                    label=trend_label,
                )

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
    if cb_ticks:
        _decorate_control_colorbar(cbar, cb_ticks, xs)
    cbar.set_label(_pretty_label_key(label_key), fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(_pretty_label_key(label_key))
    ax.set_ylabel(r"RMS reference time (s)")
    ax.set_title(title)
    ax.set_xscale("log")
    # ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(fontsize=12, loc="lower left")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rms_crossing_{_sanitize(signal_name)}_vs_{_sanitize(label_key)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"RMS crossing time — {signal_name}")


def _plot_signal_rms_overlay(
    cases: List[Dict[str, Any]],
    signal_name: str,
    label_key: str,
    window_seconds: float = RMS_WINDOW_SECONDS,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    """Overlay moving RMS curves for a signal using the same visual style as plot_It."""

    if not cases:
        print("  No hay casos.")
        return

    xs = _x_values(cases, label_key)
    param_vals = [x for x in xs if not np.isnan(x)]
    if not param_vals:
        print(f"  No se encontraron valores válidos para {signal_name}.")
        return

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        norm = mcolors.Normalize(vmin=min(param_vals), vmax=max(param_vals))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    signal_label = _pretty_signal_name(signal_name)
    readable_title = f"Moving RMS of {signal_label}"
    fig, ax = plt.subplots(figsize=(fig_size(scale=3.0)[0] * 1.0, fig_size(scale=3.0)[1]))

    control_entry = None
    crossing_info: List[tuple[str, float, float]] = []
    plotted_rms_series: List[np.ndarray] = []
    # Cache (case_idx, t, rms, color, lv_str) to avoid recomputing RMS later
    rms_cache: dict = {}
    for case_idx, (c, pv) in enumerate(zip(cases, xs)):
        sig_data = c.get("signals", {}).get(signal_name)
        if sig_data is None:
            continue

        t = np.asarray(sig_data.get("t", np.array([])), dtype=float)
        y = np.asarray(sig_data.get("y", np.array([])), dtype=float)
        if t.size == 0 or y.size == 0:
            continue

        fs = _estimate_sampling_frequency(t)
        if not np.isfinite(fs) or fs <= 0:
            continue

        window_samples = int(max(1, round(window_seconds * fs)))
        if window_samples % 2 == 0:
            window_samples += 1
        rms = _moving_rms(y, window_samples)
        # rms_sav = _moving_rms_mad_clip(y, window_samples)

        color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
        lv_str = _format_sci(pv, precision=2)
        rms_cache[case_idx] = (t, rms, color, lv_str)

        if case_idx == CONTROL_IDX:
            control_entry = (t, rms, lv_str)
        else:
            plotted_rms_series.append(rms)
            ax.plot(
                t[::DECIMATE], rms[::DECIMATE],
                color=color, lw=1.9, alpha=0.95,
                label=f"{lv_str}",
                rasterized=True,
            )

    if control_entry is not None:
        t, rms, lv_str = control_entry
        plotted_rms_series.append(rms)
        ax.plot(
            t[::DECIMATE], rms[::DECIMATE],
            color=color_red, lw=2.4, alpha=1.0,
            label=f"Control ={lv_str}",
            rasterized=True,
        )

    if not ax.lines:
        print(f"  No se encontraron datos para {signal_name}.")
        plt.close(fig)
        return

    rms_threshold = _resolve_rms_threshold(plotted_rms_series)
    if np.isfinite(rms_threshold):
        ax.axhline(rms_threshold, color=color_azul, lw=1.0, linestyle="--", alpha=0.9,
                   )
        for case_idx, (c, pv) in enumerate(zip(cases, xs)):
            if case_idx not in rms_cache:
                continue

            t, rms, color, lv_str = rms_cache[case_idx]
            crossing = _first_threshold_crossing(t, rms, rms_threshold)
            if crossing is None:
                continue

            tc, yc = crossing
            crossing_label = "Control" if case_idx == CONTROL_IDX else lv_str
            crossing_info.append((crossing_label, tc, yc))
            if case_idx == CONTROL_IDX:
                ax.vlines(
                    tc,
                    0.0,
                    yc,
                    colors=color_red,
                    linestyles="--",
                    linewidth=1.1,
                    alpha=0.85,
                    zorder=6,
                )
                ax.scatter(
                    [tc], [yc],
                    marker="D",
                    s=90,
                    facecolors=color_red,
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=9,
                )
            else:
                ax.vlines(
                    tc,
                    0.0,
                    yc,
                    colors=color,
                    linestyles="--",
                    linewidth=1.0,
                    alpha=0.75,
                    zorder=5,
                )
                ax.scatter(
                    [tc], [yc],
                    s=90,
                    color=color,
                    edgecolor="black",
                    linewidths=0.6,
                    zorder=8,
                )

        x_left = ax.get_xlim()[0]
        threshold_text_y = rms_threshold * 1.03 if rms_threshold > 0 else rms_threshold
        ax.text(
            x_left,
            threshold_text_y,
            rf"$\mathrm{{RMS\ threshold}} = {rms_threshold:.2e}$",
            color=color_azul,
            fontsize=13,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
            zorder=10,
        )

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
    if cb_ticks:
        _decorate_control_colorbar(cbar, cb_ticks, xs)
    cbar.set_label(_pretty_label_key(label_key), fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$t$ (s)")
    if signal_name == "Axial_disp":
        ax.set_ylabel(r"RMS [mm]")
    elif signal_name == "Axial_vel":
        ax.set_ylabel(r"RMS [mm/s]")
    else:
        ax.set_ylabel(r"RMS")
    ax.set_title(readable_title)
    ax.legend(fontsize=13, loc="upper left", ncol=1)
    # ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_yscale("log")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rms_overlay_{_sanitize(signal_name)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"Moving RMS overlay — {signal_name}")


def plot_signal_disp_rms_overlay(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    # _plot_signal_rms_overlay(cases, "Axial_disp", label_key, run_name_filter=run_name_filter, out_dir=out_dir)
    pass


def plot_signal_vel_rms_overlay(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    _plot_signal_rms_overlay(cases, "Axial_vel", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


# ==============================================================================
# FIGURA log-RMS overlay — log10(RMS normalizado) vs tiempo
# ==============================================================================

def _plot_signal_log_rms_overlay(
    cases,
    signal_name: str,
    label_key: str,
    window_seconds: float = None,
    norm_window: float = None,
    run_name_filter=None,
    out_dir=None,
) -> None:
    """Overlay log10(RMS/RMS_ref) vs tiempo por caso DOE.

    Normaliza cada caso por la mediana de su RMS en los primeros norm_window
    segundos.  El crecimiento exponencial aparece como una recta lineal.
    """
    if window_seconds is None:
        window_seconds = RMS_WINDOW_SECONDS
    if norm_window is None:
        norm_window = LOG_RMS_NORM_WINDOW_DISP if signal_name == "Axial_disp" else LOG_RMS_NORM_WINDOW_VEL

    if not cases:
        print("  No hay casos.")
        return

    xs = _x_values(cases, label_key)
    param_vals = [x for x in xs if not np.isnan(x)]
    if not param_vals:
        print(f"  No hay valores validos para {signal_name}.")
        return

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        norm = mcolors.Normalize(vmin=min(param_vals), vmax=max(param_vals))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    signal_label = _pretty_signal_name(signal_name)
    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    _EPS = 1e-30
    control_entry = None
    # Cache (t, log_rms) per case_idx to avoid recomputing in the onset loop
    log_rms_cache: dict = {}
    for case_idx, (c, pv) in enumerate(zip(cases, xs)):
        sig_data = c.get("signals", {}).get(signal_name)
        if sig_data is None:
            continue
        t = np.asarray(sig_data.get("t", np.array([])), dtype=float)
        y = np.asarray(sig_data.get("y", np.array([])), dtype=float)
        if t.size == 0 or y.size == 0:
            continue
        fs = _estimate_sampling_frequency(t)
        if not np.isfinite(fs) or fs <= 0:
            continue
        window_samples = int(max(1, round(window_seconds * fs)))
        if window_samples % 2 == 0:
            window_samples += 1
        rms = _moving_rms(y, window_samples)

        stable_mask = t <= norm_window
        ref_vals = rms[stable_mask]
        ref_vals = ref_vals[ref_vals > 0]
        if ref_vals.size > 0:
            rms_ref = float(np.median(ref_vals))
        elif np.any(rms > 0):
            rms_ref = float(np.median(rms[rms > 0]))
        else:
            rms_ref = 1.0
        log_rms = np.log10(rms / rms_ref + _EPS)
        log_rms_cache[case_idx] = (t, log_rms)

        color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
        lv_str = _format_sci(pv, precision=2)
        if case_idx == CONTROL_IDX:
            control_entry = (t, log_rms, lv_str)
        else:
            ax.plot(t[::DECIMATE], log_rms[::DECIMATE], color=color, lw=1.9,
                    alpha=0.95, label=lv_str, rasterized=True)

    if control_entry is not None:
        t, log_rms, lv_str = control_entry
        ax.plot(t[::DECIMATE], log_rms[::DECIMATE], color=color_red, lw=2.4,
                alpha=1.0, label=f"Control ={lv_str}", rasterized=True)

    if not ax.lines:
        print(f"  No se encontraron datos para {signal_name}.")
        plt.close(fig)
        return

    ax.axhline(0, color=color_azul, lw=1.0, ls="--", alpha=0.8,
               label=r"$\log_{10}(\mathrm{RMS/RMS_{ref}}) = 0$")
    if T_GT is not None:
        ax.axvline(T_GT, color="black", lw=1.2, ls="--", alpha=0.8,
                   label=rf"$t_{{GT}}={T_GT:.2f}$ s")

    # --- Marcadores de inicio de crecimiento exponencial por caso ---
    # Mismo estilo que RMS overlay: vlines "--" + scatter
    onset_first = True
    y_bot = ax.get_ylim()[0]
    for case_idx, (c, pv) in enumerate(zip(cases, xs)):
        tc = _case_log_rms_transition_time(
            c, signal_name,
            norm_window=norm_window,
            window_seconds=window_seconds,
        )
        if tc is None:
            continue
        color = color_red if case_idx == CONTROL_IDX else (cmap(norm(pv)) if not np.isnan(pv) else color_gray)
        # Interpolate yc from cached log_rms — no recomputation needed
        yc = 0.0
        if case_idx in log_rms_cache:
            _t, _log = log_rms_cache[case_idx]
            yc = float(np.interp(tc, _t, _log))
        label = "Slope onset" if onset_first else None
        onset_first = False
        if case_idx == CONTROL_IDX:
            ax.vlines(tc, y_bot, yc, colors=color_red, linestyles="--",
                      linewidth=1.1, alpha=0.85, zorder=6)
            ax.scatter([tc], [yc], marker="D", s=90,
                       facecolors=color_red, edgecolors="black",
                       linewidths=0.9, zorder=9, label=label)
        else:
            ax.vlines(tc, y_bot, yc, colors=color, linestyles="--",
                      linewidth=1.0, alpha=0.75, zorder=5)
            ax.scatter([tc], [yc], s=90, color=color,
                       edgecolor="black", linewidths=0.6, zorder=8, label=label)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
    if cb_ticks:
        _decorate_control_colorbar(cbar, cb_ticks, xs)
    cbar.set_label(_pretty_label_key(label_key), fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\log_{10}(\mathrm{RMS}/\mathrm{RMS}_{ref})$")
    ax.set_title(f"Log RMS of {signal_label}")
    ax.legend(fontsize=13, loc="upper left", ncol=1)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"log_rms_overlay_{_sanitize(signal_name)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"Log RMS overlay — {signal_name}")


def plot_signal_disp_log_rms_overlay(cases, label_key, run_name_filter=None, out_dir=None):
    _plot_signal_log_rms_overlay(cases, "Axial_disp", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


def plot_signal_vel_log_rms_overlay(cases, label_key, run_name_filter=None, out_dir=None):
    _plot_signal_log_rms_overlay(cases, "Axial_vel", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


# ==============================================================================
# FIGURA log-RMS crossing scatter — tiempo de transicion vs parametro DOE
# ==============================================================================

def _case_log_rms_transition_time(
    case,
    signal_name: str,
    norm_window: float = None,
    window_seconds: float = None,
    slope_n_sigma: float = None,
    slope_min_dur: float = None,
):
    """Primer instante en que la pendiente de log10(RMS/RMS_ref) se vuelve
    claramente positiva y sostenida, usando solo la propia curva del caso.

    Algoritmo:
    1. Calcula log10(RMS/RMS_ref) normalizando por el tramo inicial del caso.
    2. Estima la pendiente local muestra a muestra (diferencias finitas).
    3. Calcula mu y sigma de la pendiente en ese mismo tramo inicial.
    4. El umbral de pendiente = mu + slope_n_sigma * sigma.
    5. Busca el primer instante donde la pendiente supera ese umbral
       durante al menos slope_min_dur segundos consecutivos.
    """
    if norm_window is None:
        norm_window = LOG_RMS_NORM_WINDOW_DISP if signal_name == "Axial_disp" else LOG_RMS_NORM_WINDOW_VEL
    if window_seconds is None:
        window_seconds = RMS_WINDOW_SECONDS
    if slope_n_sigma is None:
        slope_n_sigma = LOG_RMS_SLOPE_N_SIGMA
    if slope_min_dur is None:
        slope_min_dur = LOG_RMS_SLOPE_MIN_DUR

    sig_data = case.get("signals", {}).get(signal_name)
    if sig_data is None:
        return None
    t = np.asarray(sig_data.get("t", np.array([])), dtype=float)
    y = np.asarray(sig_data.get("y", np.array([])), dtype=float)
    if t.size == 0 or y.size == 0:
        return None
    fs = _estimate_sampling_frequency(t)
    if not np.isfinite(fs) or fs <= 0:
        return None
    window_samples = int(max(1, round(window_seconds * fs)))
    if window_samples % 2 == 0:
        window_samples += 1
    rms = _moving_rms(y, window_samples)

    # Normalizar por mediana del tramo inicial del mismo caso
    stable_mask = t <= norm_window
    ref_vals = rms[stable_mask]
    ref_vals = ref_vals[ref_vals > 0]
    if ref_vals.size > 0:
        rms_ref = float(np.median(ref_vals))
    elif np.any(rms > 0):
        rms_ref = float(np.median(rms[rms > 0]))
    else:
        rms_ref = 1.0

    _EPS = 1e-30
    log_rms = np.log10(rms / rms_ref + _EPS)

    # Pendiente sobre bloques promediados (mismo tamaño que ventana RMS).
    # Esto evita que la pendiente muestra-a-muestra oscile demasiado rápido
    # y hace que min_samples sea proporcional a la escala temporal real.
    block = max(1, window_samples)
    n_blocks = len(log_rms) // block
    if n_blocks < 4:
        return None
    log_rms_b = np.array([float(np.mean(log_rms[i * block:(i + 1) * block])) for i in range(n_blocks)])
    t_b = np.array([float(np.mean(t[i * block:(i + 1) * block])) for i in range(n_blocks)])
    dt_b = np.diff(t_b)
    dt_b = np.where(dt_b > 0, dt_b, float("nan"))
    slope_b = np.diff(log_rms_b) / dt_b  # [décadas/s], longitud n_blocks-1
    slope_b = np.append(slope_b, slope_b[-1] if slope_b.size > 0 else 0.0)

    # Suavizar la pendiente con ventana de 3 bloques para reducir ruido
    # (importante en señales de velocidad que tienen mayor variabilidad intrínseca)
    smooth_k = np.ones(3) / 3.0
    slope_b = np.convolve(slope_b, smooth_k, mode="same")

    t_slope = t_b  # misma longitud que slope_b

    # fs efectiva de los bloques
    fs_b = 1.0 / float(block / fs) if fs > 0 else 1.0

    # Referencia de pendiente en el tramo inicial del mismo caso
    stable_mask_b = t_b <= norm_window
    stable_slope = slope_b[stable_mask_b & np.isfinite(slope_b)]
    if stable_slope.size < 2:
        return None
    slope_mu = float(np.mean(stable_slope))
    slope_sigma = float(np.std(stable_slope))
    if slope_sigma <= 0:
        return None

    slope_threshold = slope_mu + slope_n_sigma * slope_sigma
    min_samples = int(max(1, round(slope_min_dur * fs_b)))

    # Buscar primer tramo sostenido donde slope > slope_threshold
    above = (slope_b > slope_threshold) & np.isfinite(slope_b)
    count = 0
    for i, val in enumerate(above):
        if val:
            count += 1
            if count >= min_samples:
                onset_idx = i - min_samples + 1
                return float(t_slope[onset_idx])
        else:
            count = 0
    return None


def _plot_signal_log_rms_crossing_scatter(
    cases,
    signal_name: str,
    label_key: str,
    window_seconds: float = None,
    norm_window: float = None,
    slope_n_sigma: float = None,
    slope_min_dur: float = None,
    run_name_filter=None,
    out_dir=None,
    trend_mode: str = None,
    trend_poly_degree: int = None,
) -> None:
    """Scatter del tiempo de inicio de pendiente sostenida en log-RMS vs parametro DOE."""
    if window_seconds is None:
        window_seconds = RMS_WINDOW_SECONDS
    if norm_window is None:
        norm_window = LOG_RMS_NORM_WINDOW_DISP if signal_name == "Axial_disp" else LOG_RMS_NORM_WINDOW_VEL
    if slope_n_sigma is None:
        slope_n_sigma = LOG_RMS_SLOPE_N_SIGMA
    if slope_min_dur is None:
        slope_min_dur = LOG_RMS_SLOPE_MIN_DUR
    if trend_mode is None:
        trend_mode = RMS_TREND_MODE
    if trend_poly_degree is None:
        trend_poly_degree = RMS_TREND_POLY_DEGREE

    if not cases:
        print("  No hay casos.")
        return

    xs = _x_values(cases, label_key)
    param_vals = [x for x in xs if not np.isnan(x)]
    if not param_vals:
        print(f"  No hay valores validos para {signal_name}.")
        return

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        norm = mcolors.Normalize(vmin=min(param_vals), vmax=max(param_vals))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    signal_label = _pretty_signal_name(signal_name)
    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    crossing_xs = []
    crossing_times = []
    control_entry = None

    for case_idx, (case, pv) in enumerate(zip(cases, xs)):
        tc = _case_log_rms_transition_time(
            case, signal_name,
            norm_window=norm_window,
            window_seconds=window_seconds,
            slope_n_sigma=slope_n_sigma,
            slope_min_dur=slope_min_dur,
        )
        if tc is None:
            continue
        crossing_xs.append(pv)
        crossing_times.append(tc)
        color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
        lv_str = _format_sci(pv, precision=2)
        if case_idx == CONTROL_IDX:
            control_entry = (pv, tc)
        else:
            ax.scatter([pv], [tc], s=90, color=color,
                       edgecolor="black", linewidths=0.6, zorder=8, label=lv_str)

    if control_entry is not None:
        x_ctrl, t_ctrl = control_entry
        ax.scatter([x_ctrl], [t_ctrl], marker="D", s=90,
                   facecolors=color_red, edgecolors="black", linewidths=0.9,
                   zorder=9, label="Control")

    if not crossing_times:
        print(f"  No se encontraron tiempos de transicion para {signal_name}.")
        plt.close(fig)
        return

    if T_GT is not None:
        ax.axhline(T_GT, color=color_red, lw=1.4, ls=":",
                   label=rf"$t_{{GT}}={T_GT:.2f}$ s", zorder=5)

    trend_mode_str = str(trend_mode).strip().lower()
    if trend_mode_str != "none" and len(crossing_xs) >= 2:
        x_fit = np.asarray(crossing_xs, dtype=float)
        y_fit = np.asarray(crossing_times, dtype=float)
        valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0)
        x_fit, y_fit = x_fit[valid], y_fit[valid]
        if x_fit.size >= 2:
            x_line = np.geomspace(float(np.min(x_fit)), float(np.max(x_fit)), num=200)
            log_x_fit = np.log10(x_fit)
            log_x_line = np.log10(x_line)
            y_line = None
            trend_label = None
            if trend_mode_str == "linear":
                coeffs = np.polyfit(log_x_fit, y_fit, deg=1)
                y_line = np.polyval(coeffs, log_x_line)
                trend_label = "Trend line"
            elif trend_mode_str == "poly":
                degree = min(int(max(1, trend_poly_degree)), max(1, x_fit.size - 1))
                coeffs = np.polyfit(log_x_fit, y_fit, deg=degree)
                y_line = np.polyval(coeffs, log_x_line)
                trend_label = f"Poly trend (deg {degree})"
            elif trend_mode_str in {"exp", "exponential"}:
                pos = y_fit > 0
                if pos.sum() >= 2:
                    coeffs = np.polyfit(np.log10(x_fit[pos]), np.log(y_fit[pos]), deg=1)
                    y_line = np.exp(np.polyval(coeffs, log_x_line))
                    trend_label = "Exponential trend"
            elif trend_mode_str in {"power_law", "power-law", "powerlaw"}:
                pos = y_fit > 0
                if pos.sum() >= 2:
                    coeffs = np.polyfit(np.log10(x_fit[pos]), np.log(y_fit[pos]), deg=1)
                    y_line = np.exp(np.polyval(coeffs, log_x_line))
                    trend_label = "Power-law trend"
            if y_line is not None:
                ax.plot(x_line, y_line, color=color_azul, lw=1.6,
                        ls="--", alpha=0.9, zorder=6, label=trend_label)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
    if cb_ticks:
        _decorate_control_colorbar(cbar, cb_ticks, xs)
    cbar.set_label(_pretty_label_key(label_key), fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(_pretty_label_key(label_key))
    ax.set_ylabel("Slope onset time (s)")
    ax.set_title(rf"Log-RMS Slope Onset Time — {signal_label}")
    ax.set_xscale("log")
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir,
            f"log_rms_crossing_{_sanitize(signal_name)}_vs_{_sanitize(label_key)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"Log-RMS transition — {signal_name}")


def plot_signal_disp_log_rms_crossing_scatter(cases, label_key, run_name_filter=None, out_dir=None):
    _plot_signal_log_rms_crossing_scatter(cases, "Axial_disp", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


def plot_signal_vel_log_rms_crossing_scatter(cases, label_key, run_name_filter=None, out_dir=None):
    _plot_signal_log_rms_crossing_scatter(cases, "Axial_vel", label_key, run_name_filter=run_name_filter, out_dir=out_dir)


# ==============================================================================
# FIGURA 1a — t_d vs parámetro DOE
# FIGURA 1b — t_d_no_FAR vs parámetro DOE  (figura separada)
# ==============================================================================

def _plot_td_per_run(
    cases: list,
    xs: list,
    run_name: str,
    label_key: str,
    use_no_far: bool,
    out_dir,
) -> None:
    """Genera una figura de t_d (o t_d_no_FAR) vs label_val para UN solo indicador."""
    key    = "t_d_no_FAR" if use_no_far else "t_d"
    suffix = "_no_FAR" if use_no_far else ""
    ylabel = r"$t_d^{\mathrm{noFAR}}$ (s)" if use_no_far else r"$t_d$ (s)"
    title  = f"{run_name} — {'t_d_no_FAR' if use_no_far else 't_d'} vs {label_key}"

    ys = [
        c["runs"][run_name][key][0]
        if run_name in c.get("runs", {}) and c["runs"][run_name][key].size > 0
        else float("nan")
        for c in cases
    ]
    if not any(not np.isnan(v) for v in ys):
        return

    fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
    col = _RUN_COLORS[0]
    ax.plot(xs, ys, marker="o", color=col, lw=1.8, ms=6, label=run_name, zorder=4)
    ax.axhline(T_GT, color=color_red, lw=1.4, linestyle=":",
               label=rf"$t_{{GT}}$ = {T_GT:.2f} s", zorder=5)
    ax.set_xlabel(_pretty_label_key(label_key))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"td{suffix}_{_sanitize(run_name)}_vs_{_sanitize(label_key)}.png")
        fig.savefig(path)
    else:
        fig.canvas.manager.set_window_title(title)


def _plot_td_single(
    cases: List[Dict[str, Any]],
    xs: List[float],
    all_runs: List[str],
    label_key: str,
    use_no_far: bool,
    out_dir: Optional[str],
) -> None:
    """Genera una figura de t_d (o t_d_no_FAR) vs label_val."""
    key      = "t_d_no_FAR" if use_no_far else "t_d"
    suffix   = "_no_FAR" if use_no_far else ""
    label_key = _pretty_label_key(label_key)
    title    = (
        rf"Detection Time Without FAR vs {label_key}"
        if use_no_far
        else rf"Detection Time vs {label_key}"
    )
    ylabel   = r"$t_d^{\mathrm{noFAR}}$ (s)" if use_no_far else r"$t_d$ (s)"
    win_title = f"t_d{suffix} vs {label_key}"

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))
    plotted = False

    for i, rname in enumerate(all_runs):
        col = _RUN_COLORS[i % len(_RUN_COLORS)]
        ys  = [
            c["runs"][rname][key][0]
            if rname in c["runs"] and c["runs"][rname][key].size > 0
            else float("nan")
            for c in cases
        ]
        if any(not np.isnan(v) for v in ys):
            ax.plot(xs, ys, marker="o", color=col, lw=1.8, ms=6,
                label=_pretty_indicator_name(rname), zorder=4)
            if 0 <= CONTROL_IDX < len(cases):
                y_ctrl = ys[CONTROL_IDX] if CONTROL_IDX < len(ys) else float("nan")
                if not np.isnan(y_ctrl):
                    ax.scatter([xs[CONTROL_IDX]], [y_ctrl], marker="D", s=70,
                               color=col, edgecolor="black",
                               linewidths=0.45, zorder=6,
                               label="Control" if i == 0 else None)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    # Referencia t_GT
    ax.axhline(T_GT, color=color_red, lw=1.4, linestyle=":",
               label=rf"$t_{{GT}}$ = {T_GT:.2f} s", zorder=5)

    ax.set_xlabel(_pretty_label_key(label_key))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=13, loc="best")
    ax.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"td{suffix}_vs_{_sanitize(label_key)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(win_title)


def plot_td_vs_param(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    """Genera dos figuras separadas: t_d y t_d_no_FAR vs label_val."""
    if not cases:
        print("  No hay casos.")
        return

    xs       = _x_values(cases, label_key)
    all_runs = sorted({rn for c in cases for rn in c["runs"]})
    if run_name_filter:
        all_runs = [r for r in all_runs if r == run_name_filter]
    if not all_runs:
        print(f"  No se encontraron runs (filter={run_name_filter}).")
        return

    print("    → Figura t_d ...")
    _plot_td_single(cases, xs, all_runs, label_key, use_no_far=False, out_dir=out_dir)
    print("    → Figura t_d_no_FAR ...")
    _plot_td_single(cases, xs, all_runs, label_key, use_no_far=True,  out_dir=out_dir)

# ==============================================================================
# FIGURA 2a — I_t(t) overlay con t_d por run_name
# FIGURA 2b — I_t(t) overlay con t_d_no_FAR por run_name
# ==============================================================================

def plot_It_overlay(
    cases: List[Dict[str, Any]],
    label_key: str,
    run_name_filter: Optional[str] = None,
    use_no_far: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """Una figura por run_name: I_t(t) de todos los casos coloreados por label_val.

    Si use_no_far=True, las líneas verticales marcan t_d_no_FAR.
    En ambos casos se usa una barra de color continua para el parámetro DOE.
    """

    def _colorbar_ticks_from_data(values: List[float], normalization: mcolors.Normalize) -> List[float]:
        clean = sorted({float(v) for v in values if not np.isnan(v)})
        if not clean:
            return []
        if isinstance(normalization, mcolors.LogNorm):
            positive = [v for v in clean if v > 0]
            if not positive:
                return clean[: min(5, len(clean))]
            if len(positive) == 1:
                return positive
            count = min(5, len(positive))
            ticks = np.geomspace(positive[0], positive[-1], num=count)
            ticks = [float(t) for t in ticks]
            ticks[0] = positive[0]
            ticks[-1] = positive[-1]
            return ticks

        if len(clean) <= 5:
            return clean
        indices = np.linspace(0, len(clean) - 1, num=5)
        return [clean[int(round(i))] for i in indices]

    def _colorbar_ticklabels(ticks: List[float]) -> List[str]:
        return [_format_sci(t, precision=2) for t in ticks]
    
   
    if not cases:
        print("  No hay casos.")
        return

    xs       = _x_values(cases, label_key)
    label_key_legend = _pretty_label_key(label_key)

    all_runs = sorted({rn for c in cases for rn in c["runs"]})
    if run_name_filter:
        all_runs = [r for r in all_runs if r == run_name_filter]
    if not all_runs:
        print(f"  No se encontraron runs (filter={run_name_filter}).")
        return

    # Barra de color continua: color por valor del parámetro DOE
    param_vals = [x for x in xs if not np.isnan(x)]
    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1 and min(param_vals) > 0:
        norm = mcolors.LogNorm(
            vmin=max(min(param_vals), 1e-300),
            vmax=max(max(param_vals), 1e-299),
        )
    else:
        vmin = min(param_vals) if param_vals else 0.0
        vmax = max(param_vals) if param_vals else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])



    for rname in all_runs:
        # Título legible: reemplazar _ por espacio y capitalizar
        readable_title = _pretty_indicator_name(rname)
        if use_no_far:
            readable_title = readable_title + " | No FAR"

        fig, ax = plt.subplots(figsize=(fig_size(scale=3.0)[0] * 1.0, fig_size(scale=3.0)[1]))

        for case_idx, (c, pv) in enumerate(zip(cases, xs)):
            run_data = c["runs"].get(rname)
            if run_data is None:
                continue
            t   = run_data["t"]
            I_t = run_data["I_t"]
            if t.size == 0 or I_t.size == 0:
                continue

            color = cmap(norm(pv)) if not np.isnan(pv) else color_gray
            lv_str = _format_sci(pv, precision=2)


            if case_idx == CONTROL_IDX:
                ax.plot(t[::DECIMATE], I_t[::DECIMATE],
                    color=color_red, lw=1.9, alpha=0.95,
                    label=f"Control ={lv_str}",
                    rasterized=True)
            else:
                ax.plot(t[::DECIMATE], I_t[::DECIMATE],
                    color=color, lw=1.9, alpha=0.95,
                    # label=f"{label_key_legend}={lv_str}",
                    label=f"{lv_str}",
                    rasterized=True)

            # Marca el tiempo de detección elegido con vline más evidente
            td = run_data["t_d_no_FAR"] if use_no_far else run_data["t_d"]
            if td.size > 0:
                if case_idx == CONTROL_IDX:

                    ax.axvline(td[0], color=color_red, lw=2.0, linestyle="-", alpha=0.0,
                           zorder=6)
                else:
                    ax.axvline(td[0], color=color, lw=2.0, linestyle="--", alpha=0.0,
                            zorder=6)
                if t.size > 1:
                    y_td = float(np.interp(td[0], t, I_t))

                    if case_idx == CONTROL_IDX:                
                        ax.scatter(
                            [td[0]], [y_td],
                            marker="D",
                            s=80,
                            facecolors=color_red,
                            edgecolors="black",
                            linewidths=0.8,
                            zorder=8,
                        )
                    else:
                        ax.scatter([td[0]], [y_td], s=80, color=color,
                                edgecolor="black", linewidths=0.5, zorder=7)

        # Referencia t_GT
        ax.axvline(T_GT, color=color_red, lw=2.4, linestyle=":",
                   label=rf"Ground truth $t_{{GT}}$ = {T_GT:.2f} s", zorder=5)

        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cb_ticks = _colorbar_ticks_from_data(param_vals, norm)
        if cb_ticks:
            _decorate_control_colorbar(cbar, cb_ticks, xs)
        cbar.set_label(_pretty_label_key(label_key), fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$I(t)$")

        if _use_log_scale_for_it(rname):
            ax.set_yscale("log")

        ax.set_title(readable_title)
        ax.legend(fontsize=13, loc="upper left", ncol=1)
        ax.grid(True, linestyle=":", alpha=0.25)
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            suffix = "_noFAR" if use_no_far else ""
            path = os.path.join(out_dir, f"It_overlay{suffix}_{_sanitize(rname)}.png")
            fig.savefig(path)
            print(f"  Guardado: {path}")
        else:
            suffix = " (no FAR)" if use_no_far else ""
            fig.canvas.manager.set_window_title(f"I_t overlay{suffix} — {rname}")


# ==============================================================================
# --list
# ==============================================================================

def list_cases(h5_path: str, label_key: Optional[str] = None) -> None:
    cases = load_indicator_results(h5_path)
    if not cases:
        print("No se encontraron casos.")
        return

    lk       = _resolve_label_key(cases, label_key)
    xs       = _x_values(cases, lk)
    all_runs = sorted({rn for c in cases for rn in c["runs"]})

    col_g  = max(len("group"), max(len(c["group"]) for c in cases))
    col_lv = 14

    header = (
        f"{'group':<{col_g}}  {lk:>{col_lv}}"
        + "".join(f"  {'t_d_' + r[:12]:>18}" for r in all_runs)
    )
    print()
    print(header)
    print("-" * len(header))
    for c, pv in zip(cases, xs):
        try:
            pv_str = f"{pv:>{col_lv}g}"
        except (TypeError, ValueError):
            pv_str = f"{'N/A':>{col_lv}}"
        td_cols = ""
        for rname in all_runs:
            td = c["runs"].get(rname, {}).get("t_d", np.array([]))
            val = f"{td[0]:.3f}s" if td.size > 0 else "  no_det"
            td_cols += f"  {val:>18}"
        print(f"{c['group']:<{col_g}}  {pv_str}{td_cols}")

    print()
    print(f"  Total: {len(cases)} casos  |  label_key: {lk}")
    print(f"  Runs disponibles: {all_runs}")
    print()

# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    epilog = """\
Ejemplos:
  python doe_indicator_plotter.py --ind_results DOE_xxx\\doe_indicator_results.h5 --list
  python doe_indicator_plotter.py --ind_results ... --plot-td --show
  python doe_indicator_plotter.py --ind_results ... --plot-It
  python doe_indicator_plotter.py --ind_results ... --plot-td --plot-It --show
  python doe_indicator_plotter.py --ind_results ... --run_name maxent_revo_dec7_1step
  python doe_indicator_plotter.py --ind_results ... --plot-td --out-dir figs/

Configuración (editar en el script):
  DOE_NAME, LABEL_KEY, T_GT, RUN_NAME_FILTER, DECIMATE
"""
    p = argparse.ArgumentParser(
        description="doe_indicator_plotter — Visualiza doe_indicator_results.h5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--ind_results", default=None, metavar="PATH",
                   help="Ruta a doe_indicator_results.h5")
    p.add_argument("--doe_name", default=None, metavar="NAME",
                   help=f"Nombre de la carpeta DOE (default: {DOE_NAME})")
    p.add_argument("--label_key", default=None, metavar="KEY",
                   help="Clave DOE para eje X (default: auto desde HDF5)")
    p.add_argument("--run_name", default=None, metavar="NAME",
                   help="Filtrar figuras a un solo run_name")
    p.add_argument("--t_gt", type=float, default=None, metavar="T",
                   help=f"Tiempo de onset de chatter [s] (default: {T_GT})")
    p.add_argument("--plot-td", action="store_true",
                   help="Genera figura t_d vs parámetro DOE.")
    p.add_argument("--plot-It", action="store_true",
                   help="Genera figura I_t(t) overlay por run_name.")
    p.add_argument("--plot-signal-disp", action="store_true",
                   help="Genera overlay de Axial_disp por caso, con el estilo de I_t.")
    p.add_argument("--plot-signal-vel", action="store_true",
                   help="Genera overlay de Axial_vel por caso, con el estilo de I_t.")
    p.add_argument("--plot-signal-rms-disp", action="store_true",
                   help="Genera RMS móvil de Axial_disp por caso, con el estilo de I_t.")
    p.add_argument("--plot-signal-rms-vel", action="store_true",
                   help="Genera RMS móvil de Axial_vel por caso, con el estilo de I_t.")
    p.add_argument("--plot-rms-crossing-disp", action="store_true",
                   help="Genera scatter del tiempo de cruce RMS de Axial_disp vs parámetro DOE.")
    p.add_argument("--plot-rms-crossing-vel", action="store_true",
                   help="Genera scatter del tiempo de cruce RMS de Axial_vel vs parámetro DOE.")
    p.add_argument("--plot-log-rms-disp", action="store_true",
                   help="Genera log10(RMS) vs tiempo de Axial_disp.")
    p.add_argument("--plot-log-rms-vel", action="store_true",
                   help="Genera log10(RMS) vs tiempo de Axial_vel.")
    p.add_argument("--plot-log-rms-crossing-disp", action="store_true",
                   help="Genera scatter tiempo de transicion log-RMS de Axial_disp vs parametro DOE.")
    p.add_argument("--plot-log-rms-crossing-vel", action="store_true",
                   help="Genera scatter tiempo de transicion log-RMS de Axial_vel vs parametro DOE.")
    p.add_argument("--rms-trend-mode", default=None, metavar="MODE",
                   help="Tendencia de la figura RMS reference scatter: none, linear o poly.")
    p.add_argument("--rms-trend-degree", type=int, default=None, metavar="N",
                   help="Grado del polinomio cuando --rms-trend-mode=poly.")
    p.add_argument("--list", action="store_true",
                   help="Imprime tabla de casos y sale.")
    p.add_argument("--show", action="store_true",
                   help="Llama plt.show() al terminar.")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Carpeta donde guardar las figuras.")
    return p.parse_args()

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    global DOE_NAME, LABEL_KEY, T_GT, RUN_NAME_FILTER, PLOT_TD, PLOT_IT, PLOT_SIGNALS, PLOT_SIGNAL_RMS, SHOW, OUT_DIR  # noqa
    global PLOT_LOG_RMS, PLOT_LOG_RMS_CROSSING  # noqa

    args = parse_args()

    has_cli = any([
        args.ind_results, args.doe_name, args.label_key, args.run_name,
        args.t_gt, args.plot_td, args.plot_It, args.plot_signal_disp, args.plot_signal_vel,
        args.plot_signal_rms_disp, args.plot_signal_rms_vel, args.plot_rms_crossing_disp,
        args.plot_rms_crossing_vel, args.plot_log_rms_disp, args.plot_log_rms_vel,
        args.plot_log_rms_crossing_disp, args.plot_log_rms_crossing_vel,
        args.rms_trend_mode, args.rms_trend_degree, args.list, args.show, args.out_dir,
    ])

    # Aplicar overrides CLI
    if args.doe_name:
        DOE_NAME = args.doe_name
    if args.label_key:
        LABEL_KEY = args.label_key
    if args.t_gt is not None:
        T_GT = args.t_gt
    if args.run_name:
        RUN_NAME_FILTER = args.run_name
    if args.show:
        SHOW = True
    if args.out_dir:
        OUT_DIR = args.out_dir

    # Resolver ruta HDF5
    h5_ind = os.path.normpath(args.ind_results) if args.ind_results else _ind_h5_default()

    if not os.path.isfile(h5_ind):
        print(f"[ERROR] HDF5 no encontrado: {h5_ind}")
        if not has_cli:
            print("  Edita DOE_NAME en el bloque CONFIG del script.")
        print("  Genera primero con: python doe_indicators.py")
        sys.exit(1)

    # --list
    if args.list:
        list_cases(h5_ind, label_key=args.label_key or LABEL_KEY)
        sys.exit(0)

    # Cargar resultados
    cases = load_indicator_results(h5_ind)
    if not cases:
        print("[ERROR] No se encontraron casos en el HDF5.")
        sys.exit(1)

    lk        = _resolve_label_key(cases, args.label_key or LABEL_KEY)
    # lk        = _pretty_label_key(lk)
    rn_filter = RUN_NAME_FILTER
    out_dir   = OUT_DIR

    all_runs = sorted({rn for c in cases for rn in c["runs"]})
    print(f"  HDF5  : {h5_ind}")
    print(f"  Casos : {len(cases)}  |  label_key: {lk}")
    print(f"  Runs  : {all_runs}")

    # Decidir qué figuras generar
    plot_td_flag = args.plot_td if has_cli else PLOT_TD
    plot_It_flag = args.plot_It if has_cli else PLOT_IT
    plot_signal_disp_flag = args.plot_signal_disp if has_cli else PLOT_SIGNALS
    plot_signal_vel_flag = args.plot_signal_vel if has_cli else PLOT_SIGNALS
    plot_signal_rms_disp_flag = args.plot_signal_rms_disp if has_cli else PLOT_SIGNAL_RMS
    plot_signal_rms_vel_flag = args.plot_signal_rms_vel if has_cli else PLOT_SIGNAL_RMS
    plot_rms_crossing_disp_flag = args.plot_rms_crossing_disp if has_cli else PLOT_RMS_CROSSING
    plot_rms_crossing_vel_flag = args.plot_rms_crossing_vel if has_cli else PLOT_RMS_CROSSING
    plot_log_rms_disp_flag = args.plot_log_rms_disp if has_cli else PLOT_LOG_RMS
    plot_log_rms_vel_flag = args.plot_log_rms_vel if has_cli else PLOT_LOG_RMS
    plot_log_rms_crossing_disp_flag = args.plot_log_rms_crossing_disp if has_cli else PLOT_LOG_RMS_CROSSING
    plot_log_rms_crossing_vel_flag = args.plot_log_rms_crossing_vel if has_cli else PLOT_LOG_RMS_CROSSING
    rms_trend_mode = (args.rms_trend_mode if args.rms_trend_mode is not None else RMS_TREND_MODE)
    rms_trend_degree = args.rms_trend_degree if args.rms_trend_degree is not None else RMS_TREND_POLY_DEGREE
    # Sin flags explícitos desde CLI → generar ambas por defecto
    if has_cli and not plot_td_flag and not plot_It_flag and not plot_signal_disp_flag and not plot_signal_vel_flag and not plot_signal_rms_disp_flag and not plot_signal_rms_vel_flag and not plot_rms_crossing_disp_flag and not plot_rms_crossing_vel_flag and not plot_log_rms_disp_flag and not plot_log_rms_vel_flag and not plot_log_rms_crossing_disp_flag and not plot_log_rms_crossing_vel_flag and not args.list:
        plot_td_flag = plot_It_flag = True

    if plot_td_flag:
        print("\n[1/2] t_d vs parámetro DOE ...")
        plot_td_vs_param(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_It_flag:
        print("\n[2/2] I_t(t) overlay con t_d ...")
        plot_It_overlay(cases, lk, run_name_filter=rn_filter, use_no_far=False, out_dir=out_dir)
        print("\n[2/2] I_t(t) overlay con t_d_no_FAR ...")
        plot_It_overlay(cases, lk, run_name_filter=rn_filter, use_no_far=True, out_dir=out_dir)

    if plot_signal_disp_flag:
        print("\n[3/4] Axial_disp overlay ...")
        plot_signal_disp_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_signal_vel_flag:
        print("\n[4/4] Axial_vel overlay ...")
        plot_signal_vel_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_signal_rms_disp_flag:
        print("\n[5/6] Axial_disp RMS móvil ...")
        plot_signal_disp_rms_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_signal_rms_vel_flag:
        print("\n[6/6] Axial_vel RMS móvil ...")
        plot_signal_vel_rms_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_rms_crossing_disp_flag:
        print("\n[7/8] Axial_disp crossing-time scatter ...")
        _plot_signal_rms_crossing_scatter(
            cases,
            "Axial_disp",
            lk,
            run_name_filter=rn_filter,
            out_dir=out_dir,
            trend_mode=rms_trend_mode,
            trend_poly_degree=rms_trend_degree,
        )

    if plot_rms_crossing_vel_flag:
        print("\n[8/8] Axial_vel crossing-time scatter ...")
        _plot_signal_rms_crossing_scatter(
            cases,
            "Axial_vel",
            lk,
            run_name_filter=rn_filter,
            out_dir=out_dir,
            trend_mode=rms_trend_mode,
            trend_poly_degree=rms_trend_degree,
        )

    if plot_log_rms_disp_flag:
        print("\n[9/12] Axial_disp log-RMS overlay ...")
        plot_signal_disp_log_rms_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_log_rms_vel_flag:
        print("\n[10/12] Axial_vel log-RMS overlay ...")
        plot_signal_vel_log_rms_overlay(cases, lk, run_name_filter=rn_filter, out_dir=out_dir)

    if plot_log_rms_crossing_disp_flag:
        print("\n[11/12] Axial_disp log-RMS transition scatter ...")
        _plot_signal_log_rms_crossing_scatter(
            cases, "Axial_disp", lk,
            run_name_filter=rn_filter, out_dir=out_dir,
            trend_mode=rms_trend_mode, trend_poly_degree=rms_trend_degree,
        )

    if plot_log_rms_crossing_vel_flag:
        print("\n[12/12] Axial_vel log-RMS transition scatter ...")
        _plot_signal_log_rms_crossing_scatter(
            cases, "Axial_vel", lk,
            run_name_filter=rn_filter, out_dir=out_dir,
            trend_mode=rms_trend_mode, trend_poly_degree=rms_trend_degree,
        )

    if out_dir is None:
        print("\nGuardado desactivado: usa --out-dir o configura OUT_DIR para guardar figuras.")

    if SHOW or args.show:
        plt.show()
    elif out_dir and (plot_td_flag or plot_It_flag):
        plt.close("all")
        print(f"\nFiguras guardadas en: {out_dir}")


if __name__ == "__main__":
    main()

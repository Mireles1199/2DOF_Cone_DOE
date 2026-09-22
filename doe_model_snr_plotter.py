"""doe_model_snr_plotter.py — Visualiza resultados de doe_model_snr_results.h5.

Figuras generadas:
  1. SNR_mod_dB vs parámetro DOE (scatter/barplot por señal)
  2. Señales superpuestas: control + casos degradados (overlay, una figura por señal)

Uso desde terminal:
  python doe_model_snr_plotter.py --snr_results DOE_xxx\\doe_model_snr_results.h5
  python doe_model_snr_plotter.py --snr_results ...    --doe_name DOE_xxx  [para señales]
  python doe_model_snr_plotter.py --snr_results ...    --plot-snr
  python doe_model_snr_plotter.py --snr_results ...    --plot-signals
  python doe_model_snr_plotter.py --snr_results ...    --plot-snr --plot-signals --show
  python doe_model_snr_plotter.py --snr_results ...    --out-dir figs/
  python doe_model_snr_plotter.py --snr_results ...    --list

Uso desde VS Code (sin argumentos):
  Editar el bloque CONFIG y ejecutar directamente.
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import os
import re
import sys

import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

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


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    width = base_width * ncols * scale
    return (width, width * 0.40)


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
CASE_NAME  = "1DOF_150Hz"
CONTROL_IDX = 3

# Parámetro DOE a usar como eje X en la figura SNR vs param
# (clave de var_val.py — dejar None para auto-detectar)
DOE_PARAM_KEY = "$dxl_size$"

SIGNALS_TO_PLOT = ["Axial_disp", "Axial_vel"]
DECIMATE        = 1    # 1 = sin decimación para señales (4 → más rápido)
RMS_WINDOW_SEC  = 0.10   # Ventana RMS en segundos; se convierte a muestras por señal
HILBERT_TRIM_PCT = 0.001  # Fracción a recortar al inicio y final de la envolvente

SIGNAL_YLABELS = {
    "Axial_disp": "Axial Displacement (m)",
    "Axial_vel":  "Axial Velocity (m/s)",
}

# Por defecto se generan ambas figuras cuando se lanza sin args
PLOT_SNR     = True   # Figura SNR_mod vs parámetro DOE
PLOT_SIGNALS = True   # Figura señales superpuestas
PLOT_RMS     = True   # Figura RMS móvil superpuesta
PLOT_HILBERT = True   # Figura envolvente Hilbert superpuesta
SHOW         = True   # plt.show() al final
OUT_DIR      = None   # None → no guardar (sólo mostrar); str → ruta donde guardar

# ==============================================================================
# HELPERS
# ==============================================================================

def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _pretty_param_label(param_key: str) -> str:
    key = str(param_key).strip()
    mapping = {
        "$dxl_size$": "Dexel size",
        "dxl_size": "Dexel size",
        "$rpm$": "RPM",
        "rpm": "RPM",
        "$ftooth$": r"$f_{tooth}$",
        "ftooth": r"$f_{tooth}$",
        "$ae$": r"$a_e$",
        "ae": r"$a_e$",
        "$ap$": r"$a_p$",
        "ap": r"$a_p$",
        "$feed$": "Feed",
        "feed": "Feed",
    }
    return mapping.get(key, key)


def _format_sci(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}e}"


def _colorbar_ticks_from_data(values: list[float], normalization: mcolors.Normalize) -> list[float]:
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


def _colorbar_ticklabels(ticks: list[float]) -> list[str]:
    return [_format_sci(t, precision=2) for t in ticks]


def _doe_dir() -> str:
    return os.path.normpath(os.path.join(BASE_DIR, DOE_NAME))


def _snr_h5_default() -> str:
    return os.path.join(_doe_dir(), "doe_model_snr_results.h5")


# ==============================================================================
# LECTURA DE DATOS
# ==============================================================================

def load_snr_results(h5_path: str) -> list[dict]:
    """Lee doe_model_snr_results.h5 → lista de dicts por caso.

    Cada dict: {group, case_idx, case_path, var_val, snr_by_signal}
    """
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            if not isinstance(grp, h5py.Group):
                continue
            attrs = dict(grp.attrs)
            snr_by_signal = {
                k.replace("SNR_mod_dB_", ""): float(v)
                for k, v in attrs.items()
                if k.startswith("SNR_mod_dB_")
            }
            var_val = {
                k: v for k, v in attrs.items()
                if k not in ("case_idx", "case_path") and not k.startswith("SNR_mod_dB_")
            }
            cases.append({
                "group":         grp_name,
                "case_idx":      int(attrs.get("case_idx", -1)),
                "case_path":     str(attrs.get("case_path", "")),
                "var_val":       var_val,
                "snr_by_signal": snr_by_signal,
            })
    cases.sort(key=lambda x: x["case_idx"])
    return cases


def _detect_param_key(cases: list[dict]) -> str | None:
    """Auto-detecta el primer parámetro DOE que varía entre casos."""
    if not cases:
        return None
    all_keys = list(cases[0]["var_val"].keys())
    for k in all_keys:
        vals = set(str(c["var_val"].get(k)) for c in cases)
        if len(vals) > 1:
            return k
    return all_keys[0] if all_keys else None


def load_control_signals(doe_dir: str, control_idx: int, case_name: str) -> dict[str, tuple]:
    """Lee las señales del caso control directamente desde sens_out.hdf5."""
    ctrl_path = os.path.join(doe_dir, str(control_idx), case_name, "sens_out.hdf5")
    if not os.path.isfile(ctrl_path):
        print(f"  [AVISO] sens_out.hdf5 del control no encontrado: {ctrl_path}")
        return {}
    signals = {}
    with h5py.File(ctrl_path, "r") as f:
        for key in f:
            item = f[key]
            if not isinstance(item, h5py.Group):
                continue
            if "time" in item and "values" in item:
                signals[key] = (item["time"][:], item["values"][:])
            elif "data" in item:
                arr = item["data"][:]
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    signals[key] = (arr[:, 0], arr[:, 1])
    return signals


def load_degraded_signals(doe_dir: str, case_path: str) -> dict[str, tuple]:
    """Lee señales de un caso degradado desde su sens_out.hdf5."""
    h5_path = os.path.join(doe_dir, case_path, "sens_out.hdf5")
    if not os.path.isfile(h5_path):
        return {}
    signals = {}
    with h5py.File(h5_path, "r") as f:
        for key in f:
            item = f[key]
            if not isinstance(item, h5py.Group):
                continue
            if "time" in item and "values" in item:
                signals[key] = (item["time"][:], item["values"][:])
            elif "data" in item:
                arr = item["data"][:]
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    signals[key] = (arr[:, 0], arr[:, 1])
    return signals


def _sampling_frequency_from_time(time_values: np.ndarray) -> float:
    time_arr = np.asarray(time_values, dtype=float)
    if time_arr.size < 2:
        return float("nan")
    diffs = np.diff(time_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    dt = float(np.mean(diffs))
    if dt <= 0:
        return float("nan")
    return 1.0 / dt


def _window_samples_from_seconds(time_values: np.ndarray, window_sec: float) -> int:
    fs = _sampling_frequency_from_time(time_values)
    if not np.isfinite(fs) or fs <= 0:
        return 1
    return max(1, int(round(fs * window_sec)))


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = int(max(1, window))
    if window == 1:
        return np.sqrt(np.maximum(arr ** 2, 0.0))
    if window > arr.size:
        window = arr.size
    kernel = np.ones(window, dtype=float) / window
    mean_sq = np.convolve(np.square(arr), kernel, mode="same")
    return np.sqrt(np.maximum(mean_sq, 0.0))


def _hilbert_envelope(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    env = np.abs(hilbert(arr))
    trim = int(round(arr.size * HILBERT_TRIM_PCT))
    if trim <= 0 or 2 * trim >= arr.size:
        return env
    return env[trim:-trim]


# ==============================================================================
# FIGURA 1 — SNR_mod_dB vs parámetro DOE
# ==============================================================================

def plot_snr_vs_param(cases: list[dict], param_key: str,
                      out_dir: str = None) -> None:
    """Scatter: eje X = param_key, eje Y = SNR_mod_dB, una curva por señal."""
    if not cases:
        print("  No hay casos para graficar.")
        return

    signals = sorted({s for c in cases for s in c["snr_by_signal"]})
    if not signals:
        print("  No se encontraron SNR en el HDF5.")
        return

    # Valores del parámetro DOE para eje X
    xs = []
    for c in cases:
        try:
            xs.append(float(c["var_val"].get(param_key, c["case_idx"])))
        except (TypeError, ValueError):
            xs.append(float(c["case_idx"]))

    colors_sig = [color_azul, color_orange, color_purple, color_verde, color_red]

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    for i, sig in enumerate(signals):
        ys = [c["snr_by_signal"].get(sig, float("nan")) for c in cases]
        col = colors_sig[i % len(colors_sig)]
        ax.plot(xs, ys, marker="o", color=col, lw=1.8, ms=7, label=sig)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(param_key)
    ax.set_ylabel(r"$\mathrm{SNR}_{\mathrm{mod}}$ (dB)")
    ax.set_title(r"Model SNR vs DOE Parameter")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(False, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"snr_mod_vs_{_sanitize(param_key)}.png")
        fig.savefig(path)
        print(f"  Guardado: {path}")
    else:
        fig.canvas.manager.set_window_title(f"SNR_mod vs {param_key}")


# ==============================================================================
# FIGURA 2 — Señales superpuestas (control + degradados)
# ==============================================================================

def plot_signal_overlay(cases: list[dict], doe_dir: str,
                        param_key: str, out_dir: str = None) -> None:
    """Una figura por señal: control (negro grueso) + degradados (turbo por param)."""
    ctrl_signals = load_control_signals(doe_dir, CONTROL_IDX, CASE_NAME)
    if not ctrl_signals:
        print("  No se pudieron cargar señales del control.")
        return

    # Valores del parámetro para colormap
    param_vals = []
    for c in cases:
        try:
            param_vals.append(float(c["var_val"].get(param_key, c["case_idx"])))
        except (TypeError, ValueError):
            param_vals.append(float(c["case_idx"]))

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1:
        vmin, vmax = min(param_vals), max(param_vals)
    else:
        vmin, vmax = 0.0, 1.0
    if vmin > 0 and vmax > vmin:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-300), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    label_key = _pretty_param_label(param_key)

    for sig in SIGNALS_TO_PLOT:
        if sig not in ctrl_signals:
            continue
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))

        # Control
        t_c, y_c = ctrl_signals[sig]
        ax.plot(t_c[::DECIMATE], y_c[::DECIMATE], color=color_red, lw=2.2,
                label=f"control  ({label_key}={cases[0]['var_val'].get(param_key, '?')})",
                zorder=5, rasterized=True, alpha=0.85)

        # Degradados
        for c, pv in zip(cases, param_vals):
            deg_sigs = load_degraded_signals(doe_dir, c["case_path"])
            if sig not in deg_sigs:
                continue
            t_d, y_d = deg_sigs[sig]
            snr_val = c["snr_by_signal"].get(sig, float("nan"))
            color = cmap(norm(pv))
            ax.plot(t_d[::DECIMATE], y_d[::DECIMATE], color=color, lw=1.4,
                    alpha=0.8, rasterized=True)

        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(_pretty_param_label(param_key), fontsize=18)
        ticks = _colorbar_ticks_from_data(param_vals, norm)
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(_colorbar_ticklabels(ticks))
        cbar.ax.tick_params(labelsize=18)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(SIGNAL_YLABELS.get(sig, sig))
        ax.set_title(f"{sig}  |  {_pretty_param_label(param_key)}")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.legend(fontsize=14, loc="upper left")
        # ax.grid(False, linestyle="--", alpha=0.3)
        
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"signals_overlay_{_sanitize(sig)}.png")
            fig.savefig(path)
            print(f"  Guardado: {path}")
        else:
            fig.canvas.manager.set_window_title(f"Signals overlay — {sig}")


# ==============================================================================
# FIGURA 3 — RMS móvil superpuesta
# ==============================================================================

def plot_rms_overlay(cases: list[dict], doe_dir: str,
                     param_key: str, out_dir: str = None) -> None:
    """Una figura por señal: RMS móvil del control + RMS móvil de casos degradados."""
    ctrl_signals = load_control_signals(doe_dir, CONTROL_IDX, CASE_NAME)
    if not ctrl_signals:
        print("  No se pudieron cargar señales del control para RMS.")
        return

    param_vals = []
    for c in cases:
        try:
            param_vals.append(float(c["var_val"].get(param_key, c["case_idx"])))
        except (TypeError, ValueError):
            param_vals.append(float(c["case_idx"]))

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1:
        vmin, vmax = min(param_vals), max(param_vals)
    else:
        vmin, vmax = 0.0, 1.0
    if vmin > 0 and vmax > vmin:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-300), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    label_key = _pretty_param_label(param_key)

    for sig in SIGNALS_TO_PLOT:
        if sig not in ctrl_signals:
            continue

        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))

        t_c, y_c = ctrl_signals[sig]
        rms_window_c = _window_samples_from_seconds(t_c, RMS_WINDOW_SEC)
        fs_c = _sampling_frequency_from_time(t_c)
        print(f"  [RMS] control {sig}: fs={fs_c:.2f} Hz | window={rms_window_c} samples ({RMS_WINDOW_SEC:.3f} s)")
        rms_c = _rolling_rms(y_c, rms_window_c)
        ax.plot(t_c[::DECIMATE], rms_c[::DECIMATE], color=color_red, lw=2.2,
                label=f"Control  ({label_key}={cases[0]['var_val'].get(param_key, '?')})",
                zorder=5, rasterized=True, alpha=0.85)

        for c, pv in zip(cases, param_vals):
            deg_sigs = load_degraded_signals(doe_dir, c["case_path"])
            if sig not in deg_sigs:
                continue
            t_d, y_d = deg_sigs[sig]
            rms_window_d = _window_samples_from_seconds(t_d, RMS_WINDOW_SEC)
            rms_d = _rolling_rms(y_d, rms_window_d)
            color = cmap(norm(pv))
            ax.plot(t_d[::DECIMATE], rms_d[::DECIMATE], color=color, lw=1.4,
                    alpha=0.8, rasterized=True)

        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(_pretty_param_label(param_key), fontsize=18)
        ticks = _colorbar_ticks_from_data(param_vals, norm)
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(_colorbar_ticklabels(ticks))
        cbar.ax.tick_params(labelsize=18)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"RMS")
        ax.set_title(f"RMS {sig}  |  {_pretty_param_label(param_key)}")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.legend(fontsize=14, loc="upper left")
        # ax.grid(False, linestyle="--", alpha=0.3)
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"rms_overlay_{_sanitize(sig)}.png")
            fig.savefig(path)
            print(f"  Guardado: {path}")
        else:
            fig.canvas.manager.set_window_title(f"RMS overlay — {sig}")


# ==============================================================================
# FIGURA 4 — Envolvente Hilbert superpuesta
# ==============================================================================

def plot_hilbert_overlay(cases: list[dict], doe_dir: str,
                         param_key: str, out_dir: str = None) -> None:
    """Una figura por señal: envolvente de Hilbert del control + casos degradados."""
    ctrl_signals = load_control_signals(doe_dir, CONTROL_IDX, CASE_NAME)
    if not ctrl_signals:
        print("  No se pudieron cargar señales del control para Hilbert.")
        return

    param_vals = []
    for c in cases:
        try:
            param_vals.append(float(c["var_val"].get(param_key, c["case_idx"])))
        except (TypeError, ValueError):
            param_vals.append(float(c["case_idx"]))

    cmap = matplotlib.colormaps["viridis"]
    if len(param_vals) > 1:
        vmin, vmax = min(param_vals), max(param_vals)
    else:
        vmin, vmax = 0.0, 1.0
    if vmin > 0 and vmax > vmin:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-300), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    label_key = _pretty_param_label(param_key)

    for sig in SIGNALS_TO_PLOT:
        if sig not in ctrl_signals:
            continue

        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))

        t_c, y_c = ctrl_signals[sig]
        env_c = _hilbert_envelope(y_c)
        trim = int(round(y_c.size * HILBERT_TRIM_PCT))
        t_env_c = t_c[trim:-trim] if trim > 0 and 2 * trim < y_c.size else t_c
        ax.plot(t_env_c[::DECIMATE], env_c[::DECIMATE], color=color_red, lw=2.2,
                label=f"control  ({label_key}={cases[0]['var_val'].get(param_key, '?')})",
                zorder=5, rasterized=True, alpha=0.85)

        for c, pv in zip(cases, param_vals):
            deg_sigs = load_degraded_signals(doe_dir, c["case_path"])
            if sig not in deg_sigs:
                continue
            t_d, y_d = deg_sigs[sig]
            env_d = _hilbert_envelope(y_d)
            color = cmap(norm(pv))
            trim = int(round(y_d.size * HILBERT_TRIM_PCT))
            t_env_d = t_d[trim:-trim] if trim > 0 and 2 * trim < y_d.size else t_d
            ax.plot(t_env_d[::DECIMATE], env_d[::DECIMATE], color=color, lw=1.4,
                    alpha=0.8, rasterized=True)

        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(_pretty_param_label(param_key), fontsize=18)
        ticks = _colorbar_ticks_from_data(param_vals, norm)
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(_colorbar_ticklabels(ticks))
        cbar.ax.tick_params(labelsize=18)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Envelope")
        ax.set_title(f"Hilbert envelope {sig}  |  {_pretty_param_label(param_key)}")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.legend(fontsize=14, loc="upper left")
        ax.set_yscale("linear")
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"hilbert_overlay_{_sanitize(sig)}.png")
            fig.savefig(path)
            print(f"  Guardado: {path}")
        else:
            fig.canvas.manager.set_window_title(f"Hilbert overlay — {sig}")


# ==============================================================================
# --list
# ==============================================================================

def list_cases(h5_path: str) -> None:
    cases = load_snr_results(h5_path)
    if not cases:
        print("No se encontraron casos.")
        return
    signals = sorted({s for c in cases for s in c["snr_by_signal"]})
    param_key = DOE_PARAM_KEY or _detect_param_key(cases) or "case_idx"

    hdr = f"{'IDX':>5}  {param_key:>14}" + "".join(f"  {'SNR_'+s[:10]:>16}" for s in signals)
    print()
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        try:
            pv = f"{float(c['var_val'].get(param_key, '')):>14g}"
        except (TypeError, ValueError):
            pv = f"{str(c['var_val'].get(param_key, '')):>14}"
        snr_cols = "".join(
            f"  {c['snr_by_signal'].get(s, float('nan')):>15.2f}dB"
            for s in signals
        )
        print(f"{c['case_idx']:>5}  {pv}{snr_cols}")
    print()


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    epilog = """\
Ejemplos:
  python doe_model_snr_plotter.py --snr_results DOE_xxx\\doe_model_snr_results.h5 --list
  python doe_model_snr_plotter.py --snr_results ... --plot-snr --show
  python doe_model_snr_plotter.py --snr_results ... --plot-signals
  python doe_model_snr_plotter.py --snr_results ... --plot-snr --plot-signals --out-dir figs/

Configuración (editar en el script):
  DOE_NAME, CASE_NAME, CONTROL_IDX, DOE_PARAM_KEY, DECIMATE
    RMS_WINDOW
"""
    p = argparse.ArgumentParser(
        description="doe_model_snr_plotter — Visualiza resultados de doe_model_snr.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--snr_results", default=None, metavar="PATH",
                   help="Ruta a doe_model_snr_results.h5")
    p.add_argument("--doe_name", default=None, metavar="NAME",
                   help=f"Nombre de la carpeta DOE (default: {DOE_NAME})")
    p.add_argument("--control_idx", type=int, default=None, metavar="N",
                   help=f"Índice del caso control (default: {CONTROL_IDX})")
    p.add_argument("--param_key", default=None, metavar="KEY",
                   help=f"Parámetro DOE para eje X (default: {DOE_PARAM_KEY})")
    p.add_argument("--plot-snr", action="store_true",
                   help="Genera figura SNR_mod_dB vs parámetro DOE.")
    p.add_argument("--plot-signals", action="store_true",
                   help="Genera figura de señales superpuestas.")
    p.add_argument("--plot-rms", action="store_true",
                   help="Genera figura de RMS móvil superpuesta.")
    p.add_argument("--plot-hilbert", action="store_true",
                   help="Genera figura de envolvente Hilbert superpuesta.")
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
    global DOE_NAME, CONTROL_IDX, DOE_PARAM_KEY, PLOT_SNR, PLOT_SIGNALS, PLOT_RMS, PLOT_HILBERT, SHOW, OUT_DIR  # noqa

    args = parse_args()

    # -- Aplicar overrides de CLI --
    has_cli = any([
        args.snr_results, args.doe_name, args.control_idx,
        args.param_key, args.plot_snr, args.plot_signals, args.plot_rms, args.plot_hilbert,
        args.list, args.show, args.out_dir,
    ])

    if args.doe_name:
        DOE_NAME = args.doe_name
    if args.control_idx is not None:
        CONTROL_IDX = args.control_idx
    if args.param_key:
        DOE_PARAM_KEY = args.param_key
    if args.show:
        SHOW = True
    if args.out_dir:
        OUT_DIR = args.out_dir

    # -- Resolver rutas --
    doe_dir = os.path.normpath(os.path.join(BASE_DIR, DOE_NAME))
    if not os.path.isdir(doe_dir):
        print(f"[ERROR] Carpeta DOE no encontrada: {doe_dir}")
        sys.exit(1)

    snr_h5 = os.path.normpath(args.snr_results) if args.snr_results \
        else _snr_h5_default()

    if not os.path.isfile(snr_h5):
        print(f"[ERROR] HDF5 no encontrado: {snr_h5}")
        print(f"  Genera primero con: python doe_model_snr.py")
        sys.exit(1)

    out_dir = OUT_DIR or os.path.join(doe_dir, "figs_model_snr")

    # -- Cargar resultados --
    cases = load_snr_results(snr_h5)
    if not cases:
        print("[ERROR] No se encontraron casos en el HDF5.")
        sys.exit(1)

    param_key = DOE_PARAM_KEY or _detect_param_key(cases) or "case_idx"
    print(f"  HDF5 : {snr_h5}")
    print(f"  Casos: {len(cases)}  |  param_key: {param_key}")

    # -- --list --
    if args.list or (not has_cli and False):
        list_cases(snr_h5)
        sys.exit(0)

    # -- Decidir qué figuras generar --
    plot_snr_flag     = args.plot_snr     if has_cli else PLOT_SNR
    plot_signals_flag = args.plot_signals if has_cli else PLOT_SIGNALS
    plot_rms_flag     = args.plot_rms     if has_cli else PLOT_RMS
    plot_hilbert_flag = args.plot_hilbert if has_cli else PLOT_HILBERT

    # Si se lanzó sin ningún flag de plot → generar ambas por defecto
    if has_cli and not plot_snr_flag and not plot_signals_flag and not plot_rms_flag and not plot_hilbert_flag and not args.list:
        plot_snr_flag = plot_signals_flag = plot_rms_flag = plot_hilbert_flag = True

    if plot_snr_flag:
        print("\n[1/2] SNR_mod_dB vs parámetro DOE...")
        plot_snr_vs_param(cases, param_key, out_dir=out_dir)

    if plot_signals_flag:
        print("\n[2/2] Señales superpuestas...")
        plot_signal_overlay(cases, doe_dir, param_key, out_dir=out_dir)

    if plot_rms_flag:
        print("\n[3/3] RMS móvil superpuesta...")
        plot_rms_overlay(cases, doe_dir, param_key, out_dir=out_dir)

    if plot_hilbert_flag:
        print("\n[4/4] Envolvente Hilbert superpuesta...")
        plot_hilbert_overlay(cases, doe_dir, param_key, out_dir=out_dir)

    if SHOW or args.show:
        plt.show()
    elif out_dir:
        plt.close("all")
        print(f"\nFiguras guardadas en: {out_dir}")


if __name__ == "__main__":
    main()

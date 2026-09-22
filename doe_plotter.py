#!/usr/bin/env python
# coding: utf-8
"""
DOE Plotter
===========
Visualiza los resultados de extraccion del DOE (doe_results.h5).

Genera 4 figuras interactivas:
  1. Overlay Axial_disp  — todas las curvas coloreadas por nb_dt_rev
  2. Overlay Axial_vel   — idem
  3. Convergencia RMS    — dt (µs) vs RMS por señal
  4. Convergencia Max    — dt (µs) vs amplitud maxima por señal

Uso:
    python doe_plotter.py
    python doe_plotter.py --doe_name DOE_4
    python doe_plotter.py --doe_name DOE_Influence_dt
"""

import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, NullLocator

# ==============================================================================
# ESTILO GLOBAL — CAMP10 indicator-plot-style skill
# ==============================================================================
def configurar_estilo_global() -> None:
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.titlesize': 25,   'axes.labelsize': 25,
        'xtick.labelsize': 16,  'ytick.labelsize': 23, 'legend.fontsize': 16,
        'lines.linewidth': 2.0, 'lines.markersize': 6,
        'axes.linewidth': 0.8,   'grid.linewidth': 0.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'in',  'ytick.direction': 'in',
        'xtick.major.size': 4,  'ytick.major.size': 4,
        'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
        'legend.frameon': False,   'legend.loc': 'best',
        'figure.dpi': 100, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'path.simplify': True, 'path.simplify_threshold': 1.0,
    })


configurар_estilo_global = configurar_estilo_global  # alias
configurар_estilo_global()


def fig_size(scale: float = 1.0, ncols: int = 1, base_width: float = 3.4):
    """(width, height) en pulgadas — aspecto 0.40 para figuras de artículo."""
    width  = base_width * ncols * scale
    height = width * 0.80
    return (width, height)


# Colores canónicos CAMP10
r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99)
color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36/360, 0.45, 0.99)
color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99)
color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98/360, 0.36, 0.99)
color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)
color_azul   = (r, g, b)

# ==============================================================================
# CONFIGURACION  (editar aqui antes de ejecutar)
# ==============================================================================

DOE_NAME  = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180"  # carpeta DOE a usar por defecto
CASE_NAME = "1DOF_150Hz"        # nombre del caso base (no usado en lectura, solo doc)
# LABEL_KEY = "$nb_dt_rev$"       # variable del DOE usada para colorear / eje X
LABEL_KEY = "$dxl_size$"       # variable del DOE usada para colorear / eje X
RPM       = 12000               # rpm del caso (para calcular dt = 60/(RPM*nb_dt_rev))

SIGNALS   = ["Axial_disp", "Axial_vel"]
DECIMATE  = 1     # 1 = sin decimación (rasterized=True acelera el render)
XAXIS_FIXED_TICKS = False   # True → ticks solo en los datos reales (FixedLocator)
                           # False → ticks automáticos de matplotlib

# matplotlib: omite puntos a menos de N píxeles entre sí → render mucho más ligero
import matplotlib as mpl
mpl.rcParams["path.simplify"]           = True
mpl.rcParams["path.simplify_threshold"] = 1.0

SIGNAL_YLABELS = {
    "Axial_disp": "Axial Displacement [m]",
    "Axial_vel":  "Axial Velocity [m/s]",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================


def load_results(h5_path: str) -> list:
    """Carga doe_results.h5 y retorna lista de dicts ordenada por LABEL_KEY."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"doe_results.h5 no encontrado: {h5_path}")

    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            var_val = dict(grp.attrs)

            label_val = var_val.get(LABEL_KEY, None)
            if LABEL_KEY == "$nb_dt_rev$" and label_val is not None:
                dt_us = (60.0 / (RPM * label_val)) * 1e6
            else:
                dt_us = None

            wall_time_s = float(var_val.pop("wall_time_s")) if "wall_time_s" in var_val else None

            entry = {
                "group":       grp_name,
                "var_val":     var_val,
                "label_val":   float(label_val) if label_val is not None else None,
                "dt_us":       dt_us,
                "wall_time_s": wall_time_s,
            }

            for sig in SIGNALS:
                if sig in grp:
                    t = grp[f"{sig}/time"][()]
                    y = grp[f"{sig}/values"][()]
                    entry[sig] = (t, y)
                else:
                    entry[sig] = None

            cases.append(entry)

    # Ordenar por valor de LABEL_KEY ascendente
    cases.sort(key=lambda c: c["label_val"] if c["label_val"] is not None else 0)
    return cases


def _colormap_for(cases):
    """Retorna (norm, cmap) basado en los valores de LABEL_KEY."""
    vals = [c["label_val"] for c in cases if c["label_val"] is not None]
    if len(vals) < 2:
        vals = [vals[0] * 0.5, vals[0] * 1.5] if vals else [0, 1]
    norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
    return norm, cm.turbo


def plot_overlay(cases: list, signal: str) -> plt.Figure:
    """Overlay de todas las curvas de 'signal', coloreadas por nb_dt_rev."""
    norm, cmap = _colormap_for(cases)

    fig, ax = plt.subplots(figsize=fig_size(scale=3), constrained_layout=True)

    for case in cases:
        data = case[signal]
        if data is None:
            continue
        t, y = data
        nb    = case["label_val"]
        dt    = case["dt_us"]
        color = cmap(norm(nb))
        _lk   = LABEL_KEY.replace("$", "")
        lbl   = f"{_lk}={nb:.3g}" + (f"  (dt={dt:.1f} µs)" if dt is not None else "")
        ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, linewidth=1.5, alpha=0.85, label=lbl, rasterized=True)

    # Eje X inferior — tiempo (s)
    ax.set_xlabel("Tiempo (s)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3g}"))

    # Eje X superior — revoluciones  (t × RPM/60)
    revs_per_sec = RPM / 60.0
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda t: t * revs_per_sec, lambda r: r / revs_per_sec),
    )
    secax.set_xlabel(f"Revolución  (RPM={RPM})")
    secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3g}"))

    ax.set_ylabel(SIGNAL_YLABELS.get(signal, signal), fontsize=16)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    def _anchor_yoffset(event, _ax=ax):
        ot = _ax.yaxis.get_offset_text()
        ot.set_x(-0.01)
        ot.set_ha("right")
    fig.canvas.mpl_connect("draw_event", _anchor_yoffset)
    fig.suptitle(f"Overlay — {signal}  |  DOE: {LABEL_KEY.replace('$', '')}")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(LABEL_KEY)
    cbar.ax.tick_params(labelsize=16)

    # if len(cases) <= 12:
    #     ax.legend( loc="best", framealpha=0.7)

    # ax.legend().remove()  # eliminar leyenda si hay demasiados casos (demasiado ruido visual)
    return fig


def _nb_from_dt(d):
    """Convierte dt (µs) → nb_dt_rev. Función biyectiva (su propia inversa)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.asarray(d, dtype=float)
        return np.where(d != 0, 60.0e6 / (RPM * d), np.nan)


def _lk_display() -> str:
    """Etiqueta legible del LABEL_KEY para ejes y títulos."""
    return "Núm. dt por rev." if LABEL_KEY == "$nb_dt_rev$" else LABEL_KEY.replace("$", "")


def _add_hlines_with_labels(ax, x_vals, y_vals, fmt="{:.4g}", color="red"):
    """Línea horizontal punteada en cada y_val + texto al borde derecho del eje."""
    for val in y_vals:
        ax.axhline(y=val, linestyle=":", linewidth=1.0, color=color, alpha=0.45)
    for val in y_vals:
        ax.text(
            1.01, val, fmt.format(val),
            ha="left", va="center",
            # fontsize=12,
            transform=ax.get_yaxis_transform(), clip_on=False,
        )


def _setup_xaxis_conv(ax, x_vals, use_dt):
    """Configura eje X principal y secundario con ticks FIJOS en los datos reales.
    FixedLocator garantiza que no aparezcan ticks extra al hacer zoom/pan.
    """
    _sci = plt.FuncFormatter(lambda x, _: f"{x:.2e}")
    lkd = _lk_display()
    ax.invert_xaxis()          # siempre: mayor valor = más grueso → izquierda
    if use_dt:
        ax.set_xlabel(r"$dt$ (µs)" + "\nfino " + r"$\leftarrow\rightarrow$ grueso")
        ax.set_xscale("log")
        # ax.invert_xaxis()
        if x_vals and XAXIS_FIXED_TICKS:
            ax.xaxis.set_major_locator(FixedLocator(x_vals))
            ax.xaxis.set_minor_locator(NullLocator())
            # ax.tick_params(axis="x", rotation=45, labelsize=12)
            ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_formatter(_sci)

        # secax = ax.secondary_xaxis("top", functions=(_nb_from_dt, _nb_from_dt))
        # secax.set_xlabel(lkd)
        # if x_vals and XAXIS_FIXED_TICKS:
        #     nb_ticks = [float(_nb_from_dt(xv)) for xv in x_vals]
        #     secax.xaxis.set_major_locator(FixedLocator(nb_ticks))
        #     secax.xaxis.set_minor_locator(NullLocator())
        #     # secax.tick_params(axis="x", rotation=45, labelsize=12)
        #     secax.tick_params(axis="x", rotation=45)
        # secax.xaxis.set_major_formatter(_sci)

        if x_vals:
            secax = ax.secondary_xaxis("top")
            secax.set_xlabel(f"{lkd}  (valores DOE)")
            secax.xaxis.set_major_locator(FixedLocator(x_vals))
            secax.xaxis.set_minor_locator(NullLocator())
            secax.xaxis.set_major_formatter(_sci)
            secax.tick_params(axis="x", rotation=45, labelsize=10)


    else:
        ax.set_xlabel(lkd)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(_sci)
        # Eje superior con ticks fijos en los valores reales del DOE
        if x_vals:
            secax = ax.secondary_xaxis("top")
            secax.set_xlabel(f"{lkd}  (valores DOE)")
            secax.xaxis.set_major_locator(FixedLocator(x_vals))
            secax.xaxis.set_minor_locator(NullLocator())
            secax.xaxis.set_major_formatter(_sci)
            secax.tick_params(axis="x", rotation=45, labelsize=10)


def plot_convergence(cases: list, signal: str, metric: str) -> plt.Figure:
    """Eje X = dt(µs) si LABEL_KEY='$nb_dt_rev$', de lo contrario = valor de LABEL_KEY.

    signal : 'Axial_disp' | 'Axial_vel'
    metric : 'rms'        | 'max'
    """
    assert metric in ("rms", "max"), "metric debe ser 'rms' o 'max'"

    use_dt = (LABEL_KEY == "$nb_dt_rev$")
    lk     = LABEL_KEY.replace("$", "")
    lkd    = _lk_display()

    x_vals, metric_vals = [], []
    for case in cases:
        if case[signal] is None:
            continue
        x = case["dt_us"] if use_dt else case["label_val"]
        if x is None:
            continue
        _, y = case[signal]
        val = float(np.sqrt(np.mean(y ** 2))) if metric == "rms" else float(np.max(np.abs(y)))
        x_vals.append(x)
        metric_vals.append(val)

    metric_label = "RMS" if metric == "rms" else "Amplitud máx"
    ylabel = SIGNAL_YLABELS.get(signal, signal)

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0), constrained_layout=True)

    if x_vals:
        ax.plot(x_vals, metric_vals, marker="o", color=color_azul,
                linewidth=1.5, markersize=7)
        _add_hlines_with_labels(ax, x_vals, metric_vals)

    ax.set_ylabel(f"{metric_label}\n[{ylabel}]")
    # ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    def _anchor_yoffset(event, _ax=ax):
        ot = _ax.yaxis.get_offset_text()
        # ot.set_x(-0.005)
        ot.set_ha("right")
    fig.canvas.mpl_connect("draw_event", _anchor_yoffset)

    _setup_xaxis_conv(ax, x_vals, use_dt)
    ax.invert_xaxis()

    if use_dt:
        ax.set_title(f"Convergencia {signal}\n{metric_label} vs $dt$  (RPM={RPM})")
    else:
        ax.set_title(f"Convergencia {signal}\n{metric_label} vs {lkd}")

    return fig


def plot_convergence_error_ref(cases: list, signal: str, metric: str) -> plt.Figure:
    """Error relativo de cada caso respecto al caso más fino (mayor label_val).

    error[i] = |metric[i] - metric_finest| / |metric_finest| × 100 (%)

    signal : 'Axial_disp' | 'Axial_vel'
    metric : 'rms'        | 'max'
    """
    assert metric in ("rms", "max"), "metric debe ser 'rms' o 'max'"

    use_dt = (LABEL_KEY == "$nb_dt_rev$")
    lkd    = _lk_display()

    pts = []
    for case in cases:
        if case[signal] is None:
            continue
        x = case["dt_us"] if use_dt else case["label_val"]
        if x is None:
            continue
        _, y = case[signal]
        val = float(np.sqrt(np.mean(y ** 2))) if metric == "rms" else float(np.max(np.abs(y)))
        pts.append((x, val))

    metric_label = "RMS" if metric == "rms" else "Amplitud máx"
    ylabel = SIGNAL_YLABELS.get(signal, signal)

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0), constrained_layout=True)

    if len(pts) < 2:
        ax.text(0.5, 0.5, "Datos insuficientes (min. 2 casos)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Ordenar: mayor x (más grueso) primero → pts[-1] siempre es el más fino
    pts.sort(key=lambda p: p[0], reverse=True)
    ref_val    = pts[-1][1]
    x_vals     = [p[0] for p in pts[:-1]]
    error_vals = [abs(p[1] - ref_val) / abs(ref_val) * 100 for p in pts[:-1]]

    ax.plot(x_vals, error_vals, marker="o", color=color_red,
            linewidth=1.5, markersize=7)
    _add_hlines_with_labels(ax, x_vals, error_vals, fmt="{:.2e}%")

    ax.set_ylabel(f"Error relativo al caso más fino (%)\n[{metric_label}]")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    def _anchor_yoffset(event, _ax=ax):
        ot = _ax.yaxis.get_offset_text()
        ot.set_x(-0.01)
        ot.set_ha("right")
    fig.canvas.mpl_connect("draw_event", _anchor_yoffset)

    _setup_xaxis_conv(ax, x_vals, use_dt)
    ax.invert_xaxis()

    if use_dt:
        ax.set_title(f"Error relativo al caso más fino\n{signal} — {metric_label}  (RPM={RPM})")
    else:
        ax.set_title(f"Error relativo al caso más fino\n{signal} — {metric_label}")

    # Línea de referencia 1%
    ax.axhline(y=1.0, linestyle="--", linewidth=1.0,
               color=color_orange, alpha=0.75, label="1 % umbral")
    ax.legend(fontsize=9)
    return fig


def plot_convergence_error_consec(cases: list, signal: str, metric: str) -> plt.Figure:
    """Diferencia porcentual entre casos consecutivos (ganancia marginal).

    diff[i] = |metric[i+1] - metric[i]| / |metric[i]| × 100 (%)
    El punto se coloca en la posición del caso más fino del par (i+1).

    signal : 'Axial_disp' | 'Axial_vel'
    metric : 'rms'        | 'max'
    """
    assert metric in ("rms", "max"), "metric debe ser 'rms' o 'max'"

    use_dt = (LABEL_KEY == "$nb_dt_rev$")
    lkd    = _lk_display()

    pts = []
    for case in cases:
        if case[signal] is None:
            continue
        x = case["dt_us"] if use_dt else case["label_val"]
        if x is None:
            continue
        _, y = case[signal]
        val = float(np.sqrt(np.mean(y ** 2))) if metric == "rms" else float(np.max(np.abs(y)))
        pts.append((x, val))

    metric_label = "RMS" if metric == "rms" else "Amplitud máx"
    ylabel = SIGNAL_YLABELS.get(signal, signal)

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0), constrained_layout=True)

    if len(pts) < 2:
        ax.text(0.5, 0.5, "Datos insuficientes (min. 2 casos)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Ordenar: mayor x (más grueso) primero → i→i+1 siempre es grueso→fino
    pts.sort(key=lambda p: p[0], reverse=True)
    # x = posición del caso más fino (i+1); diff = cambio respecto a i
    x_vals    = [pts[i + 1][0] for i in range(len(pts) - 1)]
    diff_vals = [
        abs(pts[i + 1][1] - pts[i][1]) / abs(pts[i][1]) * 100
        for i in range(len(pts) - 1)
    ]

    ax.plot(x_vals, diff_vals, marker="s", color=color_verde,
            linewidth=1.5, markersize=7)
    _add_hlines_with_labels(ax, x_vals, diff_vals, fmt="{:.2e}%")

    ax.set_ylabel(f"Ganancia marginal (%)\n[{metric_label} entre consecutivos]")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    def _anchor_yoffset(event, _ax=ax):
        ot = _ax.yaxis.get_offset_text()
        ot.set_x(-0.01)
        ot.set_ha("right")
    fig.canvas.mpl_connect("draw_event", _anchor_yoffset)

    _setup_xaxis_conv(ax, x_vals, use_dt)
    ax.invert_xaxis()

    if use_dt:
        ax.set_title(f"Ganancia marginal (consecutivos)\n{signal} — {metric_label}  (RPM={RPM})")
    else:
        ax.set_title(f"Ganancia marginal (consecutivos)\n{signal} — {metric_label}")

    # Línea de referencia 1%
    ax.axhline(y=1.0, linestyle="--", linewidth=1.0,
               color=color_orange, alpha=0.75, label="1 % umbral")
    ax.legend(fontsize=9)
    return fig


def plot_convergence_time(cases: list):
    """Tiempo de ejecución (wall_time_s) vs dt (o LABEL_KEY).
    Solo dibuja si hay datos de timing (generados con --timed en doe_runner).
    Retorna Figure o None si no hay datos.
    """
    use_dt = (LABEL_KEY == "$nb_dt_rev$")
    pts = [
        (c["dt_us"] if use_dt else c["label_val"], c["wall_time_s"])
        for c in cases
        if c.get("wall_time_s") is not None
        and (c["dt_us"] if use_dt else c["label_val"]) is not None
    ]
    if not pts:
        return None
    x_vals = [p[0] for p in pts]
    t_vals = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=fig_size(scale=3.0), constrained_layout=True)
    ax.plot(x_vals, t_vals, marker="o", color=color_orange, linewidth=1.5, markersize=7)
    _add_hlines_with_labels(ax, x_vals, t_vals, fmt="{:.1f}s")
    ax.set_ylabel("Tiempo de ejecución (s)")

    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_yscale("log")

    def _anchor_yoffset(event, _ax=ax):
        ot = _ax.yaxis.get_offset_text()
        # ot.set_x(-0.2)
        ot.set_ha("right")
    fig.canvas.mpl_connect("draw_event", _anchor_yoffset)

    _setup_xaxis_conv(ax, x_vals, use_dt)
    ax.invert_xaxis()
    if use_dt:
        ax.set_title(f"Tiempo de cómputo\nvs $dt$  (RPM={RPM})")
    else:
        ax.set_title(f"Tiempo de cómputo\nvs {_lk_display()}")
    return fig


def parse_args():
    epilog = """
DOE Plotter — Genera 6 figuras de resultados DOE
=================================================
Lee doe_results.h5 y produce figuras matplotlib:
  · Overlay Axial_disp   — curvas superpuestas coloreadas por LABEL_KEY
  · Overlay Axial_vel    — ídem
  · Convergencia RMS     — valor RMS vs LABEL_KEY (x2 señales)
  · Convergencia Máx     — amplitud máx vs LABEL_KEY (x2 señales)

EJEMPLOS
--------
  python doe_plotter.py
      Usa DOE_NAME configurado en el script (por defecto: DOE_4)

  python doe_plotter.py --doe_name DOE_Influence_dt
      Visualiza el DOE en la carpeta DOE_Influence_dt/

CONFIGURACIÓN (editar en el script)
------------------------------------
  DOE_NAME  = "DOE_4"           # carpeta DOE por defecto
  LABEL_KEY = "$nb_dt_rev$"     # variable del eje X / colormap
  RPM       = 12000             # RPM del caso
  SIGNALS   = ["Axial_disp", "Axial_vel"]
"""
    parser = argparse.ArgumentParser(
        description="DOE Plotter — Nessy2m",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--doe_name", default=None,
                        help="Nombre del DOE a graficar (sobreescribe DOE_NAME del config)")
    return parser.parse_args()


def main():
    args   = parse_args()
    doe_name = args.doe_name or DOE_NAME
    h5_path  = os.path.join(SCRIPT_DIR, doe_name, "doe_results.h5")

    print(f"[INFO] Cargando: {h5_path}")
    cases = load_results(h5_path)
    print(f"[INFO] Casos cargados: {len(cases)}")
    _lk = LABEL_KEY.replace("$", "")
    for c in cases:
        dt_str = f"  dt={c['dt_us']:.1f} µs" if c["dt_us"] is not None else ""
        print(f"  {c['group']}  {_lk}={c['label_val']}{dt_str}")

    plot_overlay(cases, "Axial_disp")
    plot_overlay(cases, "Axial_vel")
    plot_convergence(cases, "Axial_disp", "rms")
    plot_convergence(cases, "Axial_disp", "max")
    plot_convergence(cases, "Axial_vel",  "rms")
    plot_convergence(cases, "Axial_vel",  "max")
    plot_convergence_error_ref(cases, "Axial_disp", "rms")
    plot_convergence_error_ref(cases, "Axial_vel",  "rms")
    plot_convergence_error_consec(cases, "Axial_disp", "rms")
    plot_convergence_error_consec(cases, "Axial_vel",  "rms")
    plot_convergence_error_ref(cases, "Axial_disp", "max")
    plot_convergence_error_ref(cases, "Axial_vel",  "max")
    plot_convergence_error_consec(cases, "Axial_disp", "max")
    plot_convergence_error_consec(cases, "Axial_vel",  "max")
    plot_convergence_time(cases)  # solo dibuja si hay timing.json en el DOE
    plt.show()


if __name__ == "__main__":
    main()

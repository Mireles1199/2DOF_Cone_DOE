"""doe_noise_plotter.py — Visualiza señales y tiempos de detección de los DOE HDF5.

Modos de uso:
  # Señales crudas (Axial_disp / Axial_vel) con/sin ruido:
  python doe_noise_plotter.py --noise_results doe_noise_results.h5 --plot-signals [--show]

  # Tiempos de detección t_d y t_d_no_FAR vs SNR por indicador:
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-detections [--show] [--out-dir figs]

  # Listar SNRs disponibles:
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --list-snr

  # Ambos a la vez:
  python doe_noise_plotter.py --noise_results doe_noise_results.h5 --indicator_results doe_noise_indicator_results.h5 --plot-signals --plot-detections --show

Configuración de señales (editar bloque CONFIG):
  SIGNALS_TO_PLOT  : señales a graficar
    CONTROL_CASE     : caso de control a usar en ambos bloques
    SNR_CASES_TO_PLOT: casos SNR a usar en señales crudas e I_t
"""

import os
import re
import sys
import argparse
import colorsys

import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==============================================================================
# SKILL: indicator-plot-style — paleta y estilo global
# ==============================================================================

r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99);  color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36/360,  0.45, 0.99);  color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99);  color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98/360,  0.36, 0.99);  color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603); color_azul = (r, g, b)


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    width = base_width * ncols * scale
    return (width, width * 0.7)


def configurar_estilo_global() -> None:
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.titlesize': 25,   'axes.labelsize': 25,
        'xtick.labelsize': 23,  'ytick.labelsize': 23, 'legend.fontsize': 23,
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

# ==============================================================================
# CONFIG — editar aquí
# ==============================================================================

SIGNALS_TO_PLOT = ["Axial_disp", "Axial_vel"]   # señales a graficar
CONTROL_CASE    = "control"         # caso de control compartido
SNR_CASES_TO_PLOT = None      # casos SNR por defecto para ambos bloques
SIGNAL_SNR_CASE = SNR_CASES_TO_PLOT   # alias para señales crudas

# Archivos por defecto para ejecutar desde VS Code sin argumentos.
# Se usan solo cuando no pasas parámetros por terminal.
_DOE_DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200",
)
DEFAULT_NOISE_RESULTS = os.path.join(_DOE_DEFAULT_DIR, "doe_noise_results.h5")
DEFAULT_INDICATOR_RESULTS = os.path.join(_DOE_DEFAULT_DIR, "doe_noise_indicator_results.h5")

# Arranque por defecto cuando ejecutas el script sin argumentos.
# Cambia estas banderas arriba, sin tocar la lógica de main().
DEFAULT_RUN_PLOT_SIGNALS = True
DEFAULT_RUN_PLOT_DETECTIONS = True
DEFAULT_RUN_PLOT_IT = True
DEFAULT_RUN_PLOT_IT_COMPARE = False
DEFAULT_RUN_PLOT_LOLLIPOP = False
DEFAULT_RUN_PLOT_DELAY = False
DEFAULT_RUN_PLOT_FAR_COST = False
DEFAULT_RUN_SHOW = True

# Tiempo de referencia del chatter real usado en I_t
T_GT = 5.365770208787228   # [s]

DECIMATE = 1   # 1 = sin decimación para señales crudas

# Curvas I_t
IT_SNR_TO_PLOT = SNR_CASES_TO_PLOT    # alias para curvas I_t
IT_DECIMATE    = 1      # decimación para curvas I_t (2989 pts → ~600)

SIGNAL_YLABELS = {
    "Axial_disp": "Axial Displacement (m)",
    "Axial_vel":  "Axial Velocity (m/s)",
}

# ==============================================================================
# HELPERS
# ==============================================================================

def _sanitize(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]+', '_', name)


def _parse_snr_from_name(name: str):
    """Extrae el número SNR del nombre del grupo, ej. 'snr_040.00' → 40.0"""
    m = re.search(r'(-?\d+\.?\d*)', name)
    return float(m.group(1)) if m else None


def _first_or_nan(ds) -> float:
    """Devuelve el primer elemento de un dataset HDF5 o NaN si vacío."""
    try:
        arr = np.asarray(ds)
        return float(arr.flat[0]) if arr.size > 0 else np.nan
    except Exception:
        return np.nan


def _format_sci(value: float, precision: int = 2) -> str:
    """Formatea un número en notación científica compacta."""
    if np.isnan(value):
        return "?"
    return f"{value:.{precision}e}"

def _format_float(value: float, precision: int = 2) -> str:
    """Formatea un número flotante con una cantidad fija de decimales."""
    if np.isnan(value):
        return "?"
    return f"{value:.{precision}f}"


def _use_log_scale_for_it(indicator: str) -> bool:
    """Devuelve True solo para familias green, ssq y sst_svd en I_t."""
    key = indicator.lower()
    return key.startswith("green") or key.startswith("ssq") or key.startswith("sst_svd")


# ==============================================================================
# PARTE 1 — SEÑALES CRUDAS (doe_noise_results.h5)
# ==============================================================================

def load_noise_results(h5_path: str) -> dict:
    """Lee doe_noise_results.h5 → dict {group_name: {attrs, signals}}."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = dict(grp.attrs)
            signals = {}
            for sig in SIGNALS_TO_PLOT:
                if sig in grp:
                    signals[sig] = (grp[f"{sig}/time"][()], grp[f"{sig}/values"][()])
            data[grp_name] = {"attrs": attrs, "signals": signals}
    return data


def _snr_groups(data: dict) -> list:
    return [k for k in sorted(data.keys()) if k.startswith("snr_")]


def _filter_snr_groups(snr_groups: list, snr_to_plot) -> list:
    if snr_to_plot is None:
        return snr_groups
    keep = []
    for g in snr_groups:
        try:
            val = float(g.replace("snr_", ""))
            if any(abs(val - s) < 0.01 for s in snr_to_plot):
                keep.append(g)
        except ValueError:
            pass
    return keep


def _select_signal_cases(data: dict, control_case=None, snr_case=None) -> tuple:
    """Devuelve (control_name, snr_names) para las señales crudas."""
    control_name = None
    snr_names = []

    control_candidates = {k: v for k, v in data.items() if "control" in k.lower()}
    if control_case is None:
        control_name = sorted(control_candidates.keys())[0] if control_candidates else None
    elif control_case in data:
        control_name = control_case
    else:
        for k in control_candidates:
            if k.lower() == str(control_case).lower():
                control_name = k
                break

    snr_candidates = {k: v for k, v in data.items() if k.startswith("snr_")}
    if snr_case is None:
        snr_names = []
    else:
        requested = snr_case if isinstance(snr_case, (list, tuple, set)) else [snr_case]
        seen = set()
        for item in requested:
            matched = None
            if item in data:
                matched = item
            else:
                try:
                    target_snr = float(item)
                except (TypeError, ValueError):
                    target_snr = None
                if target_snr is not None:
                    for k, v in snr_candidates.items():
                        snr_db = v["attrs"].get("snr_db", _parse_snr_from_name(k))
                        try:
                            snr_db = float(snr_db)
                        except Exception:
                            continue
                        if abs(snr_db - target_snr) < 0.1:
                            matched = k
                            break
            if matched is not None and matched not in seen:
                snr_names.append(matched)
                seen.add(matched)

    return control_name, snr_names


def plot_signals(data: dict, out_dir: str = None, show: bool = True,
                 control_case=None, snr_case=None) -> None:
    """Una figura por señal; control en negro, SNR elegido en colores (turbo)."""
    snr_groups  = _snr_groups(data)
    control_name, snr_names = _select_signal_cases(
        data,
        control_case=control_case,
        snr_case=snr_case,
    )

    snr_to_show = snr_names if snr_names else snr_groups

    if not snr_to_show:
        print("  No se encontraron grupos SNR para señales crudas.")
        print(f"  Grupos disponibles: {snr_groups}")
        return

    snr_vals = []
    for g in snr_to_show:
        try:
            snr_vals.append(float(data[g]["attrs"].get("snr_db", g.replace("snr_", ""))))
        except (ValueError, KeyError):
            snr_vals.append(float(g.replace("snr_", "")))

    cmap  = matplotlib.colormaps["turbo"]
    vmin, vmax = min(snr_vals), max(snr_vals)
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm    = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for sig in SIGNALS_TO_PLOT:
        fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

        if control_name is not None and sig in data.get(control_name, {}).get("signals", {}):
            t, y = data[control_name]["signals"][sig]
            ax.plot(t[::DECIMATE], y[::DECIMATE], color="black", lw=2.2,
                    label="control (sin ruido)", zorder=5, rasterized=True)

        for grp_name, snr_val in zip(snr_to_show, snr_vals):
            if sig in data[grp_name]["signals"]:
                t, y = data[grp_name]["signals"][sig]
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=cmap(norm(snr_val)),
                        lw=1.4, alpha=0.85, rasterized=True)

        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("SNR (dB)", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(SIGNAL_YLABELS.get(sig, sig))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_title(sig)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(False, alpha=0.3)
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"signal_{_sanitize(sig)}.png")
            fig.savefig(path, dpi=150)
            print(f"  Guardado: {path}")


# ==============================================================================
# PARTE 2 — DETECTION TIMES (doe_noise_indicator_results.h5)
# ==============================================================================

def gather_detection_rows(h5_path: str) -> pd.DataFrame:
    """
    Recorre case → indicator_subgroup y extrae el primer t_d y t_d_no_FAR.
    Devuelve DataFrame con columnas:
        case, indicator, snr_db (NaN para control), t_d, t_d_no_FAR
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        for case_name in sorted(f.keys()):
            case_grp = f[case_name]
            if not isinstance(case_grp, h5py.Group):
                continue

            # SNR numérico: preferir atributo snr_db, fallback al nombre de grupo
            snr_raw = case_grp.attrs.get("snr_db", None)
            if snr_raw is not None:
                try:
                    snr_db = float(snr_raw)
                except Exception:
                    snr_db = _parse_snr_from_name(case_name)
            elif "control" in case_name.lower():
                snr_db = np.nan
            else:
                snr_db = _parse_snr_from_name(case_name)

            for run_name, run_obj in case_grp.items():
                if not isinstance(run_obj, h5py.Group):
                    continue
                # Ignorar grupos de señales crudas
                if run_name in ("Axial_disp", "Axial_vel"):
                    continue

                t_d        = _first_or_nan(run_obj["t_d"])        if "t_d"        in run_obj else np.nan
                t_d_no_FAR = _first_or_nan(run_obj["t_d_no_FAR"]) if "t_d_no_FAR" in run_obj else np.nan

                rows.append({
                    "case":       case_name,
                    "indicator":  run_name,
                    "snr_db":     snr_db,
                    "t_d":        t_d,
                    "t_d_no_FAR": t_d_no_FAR,
                })

    return pd.DataFrame(rows, columns=["case", "indicator", "snr_db", "t_d", "t_d_no_FAR"])


def _control_x(snrs_sorted: list) -> float:
    """Posición X del punto de control: a la DERECHA del mayor SNR.
    Al invertir el eje X aparece visualmente a la IZQUIERDA."""
    if len(snrs_sorted) == 0:
        return 1.0
    step = float(np.median(np.diff(snrs_sorted))) if len(snrs_sorted) > 1 else 1.0
    return snrs_sorted[-1] + step


def _pretty_indicator_name(indicator: str) -> str:
    """Convierte nombres técnicos en etiquetas más legibles en inglés."""
    name = indicator.lower()
    mapping = [
        ("green", "Green Integral"),
        ("sst_svd", "SST-SVD"),
        ("maxent", "MaxEnt"),
        ("rms_cv", "RMS-CV"),
        ("ssq", "SSQ"),
        ("emd_hht", "EMD-HHT"),
        ("emd", "EMD"),
        ("dbscan", "DBSCAN"),
        ("gmm", "GMM"),
        ("k_means", "K-Means"),
        ("kmeans", "K-Means"),
        ("fft", "FFT"),
        ("hmm", "HMM"),
        ("green_integral", "Green Integral"),
    ]
    for key, label in mapping:
        if key in name:
            return label
    return indicator.replace("_", " ").strip().title()


def _plot_td_single(df_ind: pd.DataFrame, indicator: str,
                    td_col: str, out_dir: str, show: bool,
                    t_gt: float = None) -> None:
    """
    Genera UNA figura para el indicador dado usando la columna td_col
    ('t_d' o 't_d_no_FAR').
    - Control: punto cuadrado negro a la izquierda del primer SNR.
    - No-detección (NaN): símbolo 'X' rojo encima del eje Y.
    - t_gt (opcional): línea horizontal de referencia (tiempo de chatter real).
    """
    snr_rows = df_ind[df_ind["snr_db"].notna()].sort_values("snr_db")
    ctrl_rows = df_ind[df_ind["snr_db"].isna()]

    snrs = snr_rows["snr_db"].tolist()
    x_ctrl = _control_x(snrs)

    pretty_ind = _pretty_indicator_name(indicator)

    # Colormap turbo mapeado a SNR (igual que plot_signals)
    cmap_det = matplotlib.colormaps["turbo"]
    if snrs:
        norm_det = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))
        sm_det   = cm.ScalarMappable(cmap=cmap_det, norm=norm_det)
        sm_det.set_array([])
    else:
        norm_det = None

    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    # Line of connection (neutral) + points colored by SNR
    xs_valid, ys_valid, cs_valid, xs_miss = [], [], [], []
    for _, r in snr_rows.iterrows():
        if np.isnan(r[td_col]):
            xs_miss.append(r["snr_db"])
        else:
            xs_valid.append(r["snr_db"])
            ys_valid.append(r[td_col])
            cs_valid.append(cmap_det(norm_det(r["snr_db"])) if norm_det else color_orange)

    if xs_valid:
        # neutral connecting line
        ax.plot(xs_valid, ys_valid, linestyle="-", color="gray", lw=1.8, zorder=2)
        # colored points by SNR
        for x, y, c in zip(xs_valid, ys_valid, cs_valid):
            ax.scatter(x, y, color=c, s=55, zorder=4)

    # control point
    if not ctrl_rows.empty:
        y_ctrl = ctrl_rows[td_col].iloc[0]
        if not np.isnan(y_ctrl):
            ax.plot(x_ctrl, y_ctrl, marker="s", color="black",
                    markersize=8, label="Control", zorder=5)
        else:
            xs_miss.append(x_ctrl)

    # miss markers
    all_ys = [v for v in ys_valid] + (
        [ctrl_rows[td_col].iloc[0]] if not ctrl_rows.empty and not np.isnan(ctrl_rows[td_col].iloc[0]) else []
    )
    y_top = (max(all_ys) * 1.12 + 0.01) if all_ys else 0.1
    if xs_miss:
        ax.scatter(xs_miss, [y_top] * len(xs_miss),
                   marker="x", color=color_red, s=80, linewidths=2,
                   label="No detection", zorder=6)

    # colorbar
    if snrs and norm_det:
        cbar = fig.colorbar(sm_det, ax=ax, pad=0.01)
        cbar.set_label("SNR (dB)", fontsize=18)

    ylabel = r"$t_d$ (s)" if td_col == "t_d" else r"$t_{d,\mathrm{no\,FAR}}$ (s)"
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{pretty_ind}  —  {ylabel} vs SNR")
    ax.invert_xaxis()
    ax.legend(fontsize=16, loc="best")
    ax.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{_sanitize(indicator)}_{td_col}_vs_snr.png"
        path  = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        print(f"  Guardado: {path}")
def plot_td_for_indicator(df_ind: pd.DataFrame, indicator: str,
                          out_dir: str = None, show: bool = True,
                          t_gt: float = None) -> None:
    _plot_td_single(df_ind, indicator, "t_d", out_dir, show, t_gt=t_gt)


def plot_td_no_far_for_indicator(df_ind: pd.DataFrame, indicator: str,
                                 out_dir: str = None, show: bool = True,
                                 t_gt: float = None) -> None:
    _plot_td_single(df_ind, indicator, "t_d_no_FAR", out_dir, show, t_gt=t_gt)


def plot_td_both_for_indicator(df_ind: pd.DataFrame, indicator: str,
                               out_dir: str = None, show: bool = True,
                               t_gt: float = None) -> None:
    """
    Genera UNA figura con dos subplots (t_d y t_d_no_FAR) para el indicador dado.
    """
    snr_rows = df_ind[df_ind["snr_db"].notna()].sort_values("snr_db")
    ctrl_rows = df_ind[df_ind["snr_db"].isna()]
    snrs = snr_rows["snr_db"].tolist()
    x_ctrl = _control_x(snrs)

    cmap_det = matplotlib.colormaps["turbo"]
    norm_det = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs)) if len(snrs) >= 2 else None
    sm_det = cm.ScalarMappable(cmap=cmap_det, norm=norm_det) if norm_det else None
    if sm_det:
        sm_det.set_array([])

    fig, axes = plt.subplots(1, 2, figsize=(fig_size(scale=3.0)[0] * 2, fig_size(scale=3.0)[1]),
                             sharey=False)
    ylabels = [r"$t_d$ (s)", r"$t_{d,\mathrm{no\,FAR}}$ (s)"]
    for ax, td_col, ylabel in zip(axes, ("t_d", "t_d_no_FAR"), ylabels):
        xs_valid, ys_valid, cs_valid, xs_miss = [], [], [], []
        for _, r in snr_rows.iterrows():
            if np.isnan(r[td_col]):
                xs_miss.append(r["snr_db"])
            else:
                xs_valid.append(r["snr_db"])
                ys_valid.append(r[td_col])
                cs_valid.append(cmap_det(norm_det(r["snr_db"])) if norm_det else color_orange)
        if xs_valid:
            ax.plot(xs_valid, ys_valid, linestyle="-", color="gray", lw=1.8, zorder=2)
            for x, y, c in zip(xs_valid, ys_valid, cs_valid):
                ax.scatter(x, y, color=c, s=55, zorder=4)
        if not ctrl_rows.empty:
            y_ctrl = ctrl_rows[td_col].iloc[0]
            if not np.isnan(y_ctrl):
                ax.plot(x_ctrl, y_ctrl, marker="s", color="black",
                        markersize=8, label="control", zorder=5)
            else:
                xs_miss.append(x_ctrl)
        all_ys = ys_valid + ([ctrl_rows[td_col].iloc[0]] if not ctrl_rows.empty
                              and not np.isnan(ctrl_rows[td_col].iloc[0]) else [])
        y_top = (max(all_ys) * 1.12 + 0.01) if all_ys else 0.1
        if xs_miss:
            ax.scatter(xs_miss, [y_top] * len(xs_miss), marker="x", color=color_red,
                       s=80, linewidths=2, label="no detección", zorder=6)
        if t_gt is not None:
            ax.axhline(t_gt, color="black", lw=1.5, linestyle=":",
                       label=rf"$t_{{GT}}$={t_gt:.2f}s", zorder=3)
        if sm_det:
            cbar = fig.colorbar(sm_det, ax=ax, pad=0.01)
            cbar.set_label("SNR (dB)", fontsize=18)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.invert_xaxis()
        ax.legend(fontsize=24)
        ax.grid(False, linestyle="--", alpha=0.35)

    fig.suptitle(_pretty_indicator_name(indicator), fontsize=16)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{_sanitize(indicator)}_td_both_vs_snr.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_detections(df: pd.DataFrame, out_dir: str, show: bool,
                    indicators_filter: list = None, t_gt: float = None) -> None:
    """Genera dos figuras separadas: t_d vs SNR y t_d_no_FAR vs SNR."""
    indicators = sorted(df["indicator"].dropna().unique())
    if indicators_filter:
        indicators = [i for i in indicators if i in indicators_filter]

    if not indicators:
        print("  No se encontraron indicadores en el HDF5.")
        return

    df_p = df[df["indicator"].isin(indicators)].copy()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for td_col in ("t_d", "t_d_no_FAR"):
        fig, ax = plt.subplots(figsize=(fig_size(scale=3.0)[0], fig_size(scale=3.0)[1]))
        snr_rows = df_p[df_p["snr_db"].notna()].copy()
        snrs = sorted(snr_rows["snr_db"].dropna().unique())
        x_ctrl = _control_x(snrs)

        # cmap_det = matplotlib.colormaps["turbo"]
        # norm_det = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs)) if len(snrs) >= 2 else None
        # sm_det = cm.ScalarMappable(cmap=cmap_det, norm=norm_det) if norm_det else None
        # if sm_det is not None:
        #     sm_det.set_array([])

        for i, ind in enumerate(indicators):
            sub = df_p[df_p["indicator"] == ind]
            snr_ind = sub[sub["snr_db"].notna()].sort_values("snr_db")
            ctrl_ind = sub[sub["snr_db"].isna()]
            c = _COLORS_IND[i % len(_COLORS_IND)]
            pretty_ind = _pretty_indicator_name(ind)

            xs_valid, ys_valid, xs_miss = [], [], []
            for _, r in snr_ind.iterrows():
                if np.isnan(r[td_col]):
                    xs_miss.append(r["snr_db"])
                else:
                    xs_valid.append(r["snr_db"])
                    ys_valid.append(r[td_col])

            if xs_valid:
                ax.plot(xs_valid, ys_valid, marker="o", linestyle="-", color=c, lw=1.8,
                        label=pretty_ind, rasterized=True)

            if not ctrl_ind.empty:
                y_ctrl = ctrl_ind[td_col].iloc[0]
                if not np.isnan(y_ctrl):
                    ax.scatter([x_ctrl], [y_ctrl], marker="D", s=78,
                               facecolors=c, edgecolors="black",
                               linewidths=0.8, zorder=7,
                               label="Control" if i == 0 else None,
                               rasterized=True)
                else:
                    xs_miss.append(x_ctrl)

            if xs_miss:
                all_ys = ys_valid + ([ctrl_ind[td_col].iloc[0]] if not ctrl_ind.empty and not np.isnan(ctrl_ind[td_col].iloc[0]) else [])
                y_top = (max(all_ys) * 1.12 + 0.01) if all_ys else 0.1
                ax.scatter(xs_miss, [y_top] * len(xs_miss), marker="x", color=c,
                           s=60, linewidths=2, zorder=6)

        if t_gt is not None:
            ax.axhline(t_gt, color="black", lw=1.5, linestyle=":",
                       label=rf"Ground truth $t_{{GT}}$ = {t_gt:.2f} s", zorder=3)
            
        # if sm_det is not None:
        #     cbar = fig.colorbar(sm_det, ax=ax, pad=0.02)
        #     cbar.set_label("SNR (dB)", fontsize=16)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(r"$t_d$ (s)" if td_col == "t_d" else r"$t_{d,\mathrm{no\,FAR}}$ (s)")
        ax.set_title("Detection time vs SNR" if td_col == "t_d" else "Detection time without FAR vs SNR")
        ax.invert_xaxis()
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.legend(fontsize=11, loc="best")
        fig.tight_layout()

        if out_dir:
            suffix = "td" if td_col == "t_d" else "td_no_far"
            path = os.path.join(out_dir, f"detections_{suffix}_vs_snr.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"  Guardado: {path}")

# ==============================================================================
# PARTE 3 — CURVAS I_t (doe_noise_indicator_results.h5)
# ==============================================================================

def gather_indicator_curves(h5_path: str, indicators_filter: list = None) -> dict:
    """
    Carga t e I_t por cada (indicador, caso).
    Resultado: {ind: {case_name: {"t": ndarray, "I_t": ndarray, "t_d": float, "t_d_no_FAR": float, "snr_db": float|nan}}}
    """
    result = {}
    with h5py.File(h5_path, "r") as f:
        for case_name in sorted(f.keys()):
            case_grp = f[case_name]
            if not isinstance(case_grp, h5py.Group):
                continue
            snr_raw = case_grp.attrs.get("snr_db", None)
            if snr_raw is not None and str(snr_raw).lower() not in ("none", "nan", ""):
                try:
                    snr_db = float(snr_raw)
                except Exception:
                    snr_db = _parse_snr_from_name(case_name)
            elif "control" in case_name.lower():
                snr_db = np.nan
            else:
                snr_db = _parse_snr_from_name(case_name)

            for run_name, run_obj in case_grp.items():
                if not isinstance(run_obj, h5py.Group):
                    continue
                if run_name in ("Axial_disp", "Axial_vel"):
                    continue
                if indicators_filter and run_name not in indicators_filter:
                    continue
                if "t" not in run_obj or "I_t" not in run_obj:
                    continue
                if run_name not in result:
                    result[run_name] = {}
                result[run_name][case_name] = {
                    "t":      np.asarray(run_obj["t"]),
                    "I_t":    np.asarray(run_obj["I_t"]),
                    "t_d":    _first_or_nan(run_obj["t_d"]) if "t_d" in run_obj else np.nan,
                    "t_d_no_FAR": _first_or_nan(run_obj["t_d_no_FAR"]) if "t_d_no_FAR" in run_obj else np.nan,
                    "snr_db": snr_db,
                }
    return result


def _snr_filter_cases(case_dict: dict, snr_filter) -> dict:
    """Filtra casos SNR según lista de valores (None → todos)."""
    if snr_filter is None:
        return {k: v for k, v in case_dict.items() if not pd.isna(v["snr_db"]) }
    return {k: v for k, v in case_dict.items()
            if not pd.isna(v["snr_db"])
            and any(abs(v["snr_db"] - s) < 0.1 for s in snr_filter)}


def _match_case_by_snr(case_data: dict, case_selector) -> str | None:
    """Resuelve un selector de caso por nombre exacto o por valor de SNR."""
    if case_selector is None:
        return None
    if case_selector in case_data:
        return case_selector
    try:
        target_snr = float(case_selector)
    except (TypeError, ValueError):
        return None

    candidates = []
    for case_name, case_info in case_data.items():
        snr_db = case_info.get("snr_db", np.nan)
        if not pd.isna(snr_db) and abs(float(snr_db) - target_snr) < 0.1:
            candidates.append(case_name)
    if not candidates:
        return None
    return sorted(candidates)[0]


def _filter_case_data(case_data: dict, control_case=None, snr_case=None) -> dict:
    """Filtra un diccionario case_name→datos para dejar solo control/SNR elegidos."""
    selected = {}
    control_name = _match_case_by_snr({k: v for k, v in case_data.items() if pd.isna(v.get("snr_db", np.nan))}, control_case)
    snr_name = _match_case_by_snr({k: v for k, v in case_data.items() if not pd.isna(v.get("snr_db", np.nan))}, snr_case)

    if control_name is not None:
        selected[control_name] = case_data[control_name]
    if snr_name is not None:
        selected[snr_name] = case_data[snr_name]
    return selected


def plot_it_overlay(curves: dict, indicator: str,
                    snr_filter=None, out_dir: str = None,
                    t_gt: float = None,
                    control_case=None, snr_case=None) -> None:
    """
    Figura 1: I_t(t) del control y los SNR declarados con estilo tipo doe_indicator_plotter.
    """

    def _colorbar_ticks_from_data(values: list, normalization: mcolors.Normalize) -> list:
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
        indices = np.linspace(0, len(clean) - 1, num = len(clean)//3+1 )
        return [clean[int(round(i))] for i in indices]
    

    if t_gt is None:
        t_gt = T_GT

    if indicator not in curves:
        print(f"  Indicador '{indicator}' no encontrado.")
        return
    case_data  = curves[indicator]
    ctrl_cases = {k: v for k, v in case_data.items() if pd.isna(v["snr_db"])}
    snr_cases  = _snr_filter_cases(case_data, snr_filter if snr_filter is not None else IT_SNR_TO_PLOT)

    if control_case is not None:
        selected = _filter_case_data(case_data, control_case=control_case, snr_case=None)
        ctrl_cases = {k: v for k, v in selected.items() if pd.isna(v["snr_db"])}
    if snr_case is not None:
        selected = _filter_case_data(case_data, control_case=None, snr_case=snr_case)
        snr_cases = {k: v for k, v in selected.items() if not pd.isna(v["snr_db"])}

    if not ctrl_cases and not snr_cases:
        return

    snr_vals = sorted([v["snr_db"] for v in snr_cases.values()])
    cmap     = matplotlib.colormaps["viridis"]
    norm     = mcolors.Normalize(vmin=min(snr_vals), vmax=max(snr_vals)) if snr_vals else None
    sm       = cm.ScalarMappable(cmap=cmap, norm=norm) if norm else None
    if sm is not None:
        sm.set_array([])

    fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
    dec = max(1, IT_DECIMATE)

    label_key = indicator
    readable_title = _pretty_indicator_name(indicator)

    # Control
    for v in ctrl_cases.values():
        ax.plot(v["t"][::dec], v["I_t"][::dec], color=color_red, lw=2.2,
                label="Control", zorder=5, rasterized=True)

    # SNR
    for k, v in sorted(snr_cases.items(), key=lambda x: x[1]["snr_db"], reverse=True):
        color = cmap(norm(v["snr_db"])) if norm is not None else color_orange
        ax.plot(v["t"][::dec], v["I_t"][::dec], color=color, lw=1.5, alpha=0.85,
                rasterized=True)

        td = _first_or_nan(v.get("t_d"))
        if not np.isnan(td) and len(v["t"]) > 1:
            y_td = float(np.interp(td, v["t"], v["I_t"]))
            ax.scatter([td], [y_td], s=80, color=color,
                       edgecolor="black", linewidths=1.0, zorder=7)

    # Marcador de control al estilo doe_indicator_plotter
    if ctrl_cases:
        for v in ctrl_cases.values():
            y_ctrl = v["I_t"][::dec]
            t_ctrl = v["t"][::dec]
            if len(t_ctrl) > 0 and len(y_ctrl) > 0:
                idx_ctrl = len(t_ctrl) // 2
                ax.scatter([
                    t_ctrl[idx_ctrl]
                ], [y_ctrl[idx_ctrl]], marker="D", s=80,
                   facecolors=color_red, edgecolors="black",
                   linewidths=1.0, zorder=8)

    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cb_ticks = _colorbar_ticks_from_data(snr_vals, norm)
        if cb_ticks:
            cbar.set_ticks(cb_ticks)
            cbar.set_ticklabels([_format_float(t, precision=0) for t in cb_ticks])
        cbar.set_label("SNR (dB)", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    if t_gt is not None:
        ax.axvline(t_gt, color=color_red, lw=2.4, linestyle=":",
                   label=rf"Ground truth $t_{{GT}}$ = {t_gt:.2f} s", zorder=5)

        

    if _use_log_scale_for_it(indicator):
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$I(t)$")
    ax.set_title(f"{readable_title}")
    ax.legend(fontsize=13, loc="upper left", ncol=1)

    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_sanitize(indicator)}_It_overlay.png")
        fig.savefig(path, dpi=300)
        print(f"  Guardado: {path}")


def plot_it_compare(curves: dict, indicator: str,
                    out_dir: str = None, t_gt: float = None,
                    control_case=None, snr_case=None) -> None:
    """
    Figura 2: control vs el SNR declarado más representativo.
    """
    if indicator not in curves:
        return
    case_data  = curves[indicator]
    ctrl_cases = {k: v for k, v in case_data.items() if pd.isna(v["snr_db"])}
    snr_cases  = {k: v for k, v in case_data.items() if not pd.isna(v["snr_db"])}

    if control_case is not None:
        selected = _filter_case_data(case_data, control_case=control_case, snr_case=None)
        ctrl_cases = {k: v for k, v in selected.items() if pd.isna(v["snr_db"])}
    if snr_case is not None:
        selected = _filter_case_data(case_data, control_case=None, snr_case=snr_case)
        snr_cases = {k: v for k, v in selected.items() if not pd.isna(v["snr_db"])}

    if not ctrl_cases or not snr_cases:
        print(f"  No hay control o SNR para '{indicator}' en compare.")
        return

    ctrl_v  = list(ctrl_cases.values())[0]
    min_key = min(snr_cases.keys(), key=lambda k: snr_cases[k]["snr_db"])
    min_val = snr_cases[min_key]["snr_db"]
    noisy_v = snr_cases[min_key]

    fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
    dec = max(1, IT_DECIMATE)

    ax.plot(ctrl_v["t"][::dec], ctrl_v["I_t"][::dec],
            color=color_azul, lw=2.2, label="control (sin ruido)", rasterized=True)
    ax.plot(noisy_v["t"][::dec], noisy_v["I_t"][::dec],
            color=color_orange, lw=2.0, alpha=0.9,
            label=f"SNR = {min_val:.0f} dB (más ruidoso)", rasterized=True)

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$I(t)$")
    ax.set_title(f"{indicator}  —  control vs SNR mínimo")
    ax.legend()
    ax.grid(False, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_sanitize(indicator)}_It_compare.png")
        fig.savefig(path, dpi=300)
        print(f"  Guardado: {path}")


# ==============================================================================
# PARTE 4 — RETRASO Y COSTE FAR (doe_noise_indicator_results.h5)
# ==============================================================================

_COLORS_IND = [color_orange, color_azul, color_purple, color_verde, color_red]


def _x_ctrl_from_df(df_snr: pd.DataFrame) -> float:
    snrs = sorted(df_snr["snr_db"].dropna().unique())
    if not snrs:
        return 0.0
    step = float(np.median(np.diff(snrs))) if len(snrs) > 1 else 1.0
    return snrs[-1] + step


def plot_td_lollipop(df: pd.DataFrame, out_dir: str = None,
                     indicators_filter: list = None) -> None:
    """
    Figura 3: lollipop horizontal — t_d por indicador × SNR.
    Y = indicadores, X = t_d. Un color por SNR (turbo). Control = cuadrado negro.
    """
    df_p = df.copy()
    if indicators_filter:
        df_p = df_p[df_p["indicator"].isin(indicators_filter)]
    indicators = sorted(df_p["indicator"].dropna().unique())
    if not indicators:
        return

    snr_vals = sorted(df_p["snr_db"].dropna().unique(), reverse=True)  # desc → menos ruido arriba
    cmap     = matplotlib.colormaps["turbo"]
    norm     = mcolors.Normalize(vmin=min(snr_vals), vmax=max(snr_vals)) if snr_vals else None
    sm_lp    = cm.ScalarMappable(cmap=cmap, norm=norm) if norm else None
    if sm_lp:
        sm_lp.set_array([])

    fig, ax = plt.subplots(figsize=fig_size(scale=2.2, ncols=1))

    for i, ind in enumerate(indicators):
        sub = df_p[df_p["indicator"] == ind]
        # Control
        ctrl = sub[sub["snr_db"].isna()]
        if not ctrl.empty and not np.isnan(ctrl["t_d"].iloc[0]):
            td_c = ctrl["t_d"].iloc[0]
            ax.plot([0, td_c], [i, i], color="gray", lw=1.4, alpha=0.5)
            ax.plot(td_c, i, marker="s", color="black", markersize=10, zorder=5,
                    label="control" if i == 0 else "")
        # SNR
        for snr in snr_vals:
            row = sub[np.isclose(sub["snr_db"].fillna(np.inf), snr, atol=0.1)]
            if row.empty or np.isnan(row["t_d"].iloc[0]):
                continue
            td = row["t_d"].iloc[0]
            color = cmap(norm(snr)) if norm else color_orange
            ax.plot([0, td], [i, i], color=color, lw=1.8, alpha=0.5)
            ax.plot(td, i, marker="o", color=color, markersize=8, zorder=4)

    if sm_lp:
        cbar = fig.colorbar(sm_lp, ax=ax, pad=0.01)
        cbar.set_label("SNR (dB)", fontsize=18)

    ax.set_yticks(range(len(indicators)))
    ax.set_yticklabels(indicators, fontsize=18)
    ax.set_xlabel(r"$t_d$ (s)")
    ax.set_title(r"$t_d$ por indicador")
    ax.legend(fontsize=16, loc="lower right")
    ax.grid(False, linestyle="--", alpha=0.3, axis="x")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "td_lollipop.png")
        fig.savefig(path, dpi=300)
        print(f"  Guardado: {path}")


def plot_delay_vs_snr(df: pd.DataFrame, t_gt: float, out_dir: str = None,
                      indicators_filter: list = None) -> None:
    """
    Figura 4: (t_d - t_gt) vs SNR — retraso de detección relativo al chatter real.
    Figura combinada, una curva por indicador. Requiere --t-gt.
    """
    df_p = df.copy()
    if indicators_filter:
        df_p = df_p[df_p["indicator"].isin(indicators_filter)]
    indicators = sorted(df_p["indicator"].dropna().unique())
    if not indicators:
        return

    x_ctrl = _x_ctrl_from_df(df_p)
    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    for i, ind in enumerate(indicators):
        sub    = df_p[df_p["indicator"] == ind]
        snr_r  = sub[sub["snr_db"].notna()].sort_values("snr_db")
        ctrl_r = sub[sub["snr_db"].isna()]
        c      = _COLORS_IND[i % len(_COLORS_IND)]

        xs, ys, xs_m = [], [], []
        for _, row in snr_r.iterrows():
            if np.isnan(row["t_d"]):
                xs_m.append(row["snr_db"])
            else:
                xs.append(row["snr_db"])
                ys.append(row["t_d"] - t_gt)

        if xs:
            ax.plot(xs, ys, marker="o", linestyle="-", color=c, lw=1.5, label=ind)
        if not ctrl_r.empty and not np.isnan(ctrl_r["t_d"].iloc[0]):
            ax.plot(x_ctrl, ctrl_r["t_d"].iloc[0] - t_gt,
                    marker="s", color=c, markersize=8, zorder=5)
        if xs_m:
            all_ys = ys + ([ctrl_r["t_d"].iloc[0] - t_gt]
                           if not ctrl_r.empty and not np.isnan(ctrl_r["t_d"].iloc[0]) else [])
            y_top = (max(all_ys) * 1.12 + 0.01) if all_ys else 0.1
            ax.scatter(xs_m, [y_top] * len(xs_m), marker="x", color=c, s=60, zorder=6)

    ax.axhline(0, color="gray", lw=1.4, linestyle=":", label=r"$t_d = t_{gt}$")
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$t_d - t_{gt}$ (s)")
    ax.set_title(r"Retraso de detección $t_d - t_{gt}$ vs SNR")
    ax.legend()
    ax.grid(False, linestyle="--", alpha=0.35)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "delay_td_minus_tgt_vs_snr.png")
        fig.savefig(path, dpi=300)
        print(f"  Guardado: {path}")


def plot_far_cost_vs_snr(df: pd.DataFrame, out_dir: str = None,
                         indicators_filter: list = None) -> None:
    """
    Figura 5: (t_d_no_FAR - t_d) vs SNR — coste del filtro FAR.
    Figura combinada, una curva por indicador.
    """
    df_p = df.copy()
    if indicators_filter:
        df_p = df_p[df_p["indicator"].isin(indicators_filter)]
    indicators = sorted(df_p["indicator"].dropna().unique())
    if not indicators:
        return

    x_ctrl = _x_ctrl_from_df(df_p)
    fig, ax = plt.subplots(figsize=fig_size(scale=3.0))

    for i, ind in enumerate(indicators):
        sub    = df_p[df_p["indicator"] == ind]
        snr_r  = sub[sub["snr_db"].notna()].sort_values("snr_db")
        ctrl_r = sub[sub["snr_db"].isna()]
        c      = _COLORS_IND[i % len(_COLORS_IND)]

        xs, ys, xs_m = [], [], []
        for _, row in snr_r.iterrows():
            if np.isnan(row["t_d"]) or np.isnan(row["t_d_no_FAR"]):
                xs_m.append(row["snr_db"])
            else:
                xs.append(row["snr_db"])
                ys.append(row["t_d_no_FAR"] - row["t_d"])

        if xs:
            ax.plot(xs, ys, marker="o", linestyle="-", color=c, lw=1.5, label=ind)
        if not ctrl_r.empty:
            td_c   = ctrl_r["t_d"].iloc[0]
            td_nf  = ctrl_r["t_d_no_FAR"].iloc[0]
            if not np.isnan(td_c) and not np.isnan(td_nf):
                ax.plot(x_ctrl, td_nf - td_c,
                        marker="s", color=c, markersize=8, zorder=5)
        if xs_m:
            y_top = (max(ys) * 1.12 + 0.01) if ys else 0.1
            ax.scatter(xs_m, [y_top] * len(xs_m), marker="x", color=c, s=60, zorder=6)

    ax.axhline(0, color="gray", lw=1.4, linestyle=":", label="sin retraso FAR")
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$t_{d,\mathrm{no\,FAR}} - t_d$ (s)")
    ax.set_title(r"Coste filtro FAR: $t_{d,\mathrm{no\,FAR}} - t_d$ vs SNR")
    ax.legend()
    ax.grid(False, linestyle="--", alpha=0.35)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "far_cost_vs_snr.png")
        fig.savefig(path, dpi=300)
        print(f"  Guardado: {path}")

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualiza señales y tiempos de detección de DOE HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python doe_noise_plotter.py --noise_results doe_noise_results.h5 --plot-signals
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-detections
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-it --plot-it-compare
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-lollipop
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-delay --t-gt 8.1 --plot-far-cost
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --plot-all --t-gt 8.1
  python doe_noise_plotter.py --indicator_results doe_noise_indicator_results.h5 --list-snr

Modo de ejecución:
    - Con argumentos: se respetan los flags de la línea de comandos.
    - Sin argumentos (por ejemplo, desde VS Code): se usa el preset interno
        definido en _apply_default_actions().
        """,
    )
    p.add_argument("--noise_results",     metavar="PATH", default=None,
                   help="doe_noise_results.h5  (para --plot-signals)")
    p.add_argument("--indicator_results", metavar="PATH", default=None,
                   help="doe_noise_indicator_results.h5  (para todos los demás)")

    # ── Qué graficar ──────────────────────────────────────────────────────────
    p.add_argument("--plot-signals",      action="store_true",
                   help="[S] Señales Axial_disp / Axial_vel con colormap SNR")
    p.add_argument("--plot-detections",   action="store_true",
                   help="[D] t_d y t_d_no_FAR vs SNR — 2 figuras por indicador")
    p.add_argument("--plot-it",           action="store_true",
                   help="[I] I_t(t) overlay — control + SNRs en una figura por indicador")
    p.add_argument("--plot-it-compare",   action="store_true",
                   help="[C] I_t(t) compare — control (azul) vs SNR mínimo (naranja)")
    p.add_argument("--plot-lollipop",     action="store_true",
                   help="[L] Lollipop horizontal: t_d por indicador × SNR")
    p.add_argument("--plot-delay",        action="store_true",
                   help="[4] (t_d - t_gt) vs SNR — requiere --t-gt")
    p.add_argument("--plot-far-cost",     action="store_true",
                   help="[5] (t_d_no_FAR - t_d) vs SNR — coste del filtro FAR")
    p.add_argument("--plot-all",          action="store_true",
                   help="Activa todas las figuras anteriores")

    # ── Parámetros opcionales ─────────────────────────────────────────────────
    p.add_argument("--list-snr",          action="store_true",
                   help="Mostrar SNRs encontrados y salir")
    p.add_argument("--show",              action="store_true", default=True,
                   help="plt.show() interactivo — activo por defecto")
    p.add_argument("--no-show",           action="store_false", dest="show",
                   help="No llamar plt.show() (batch / headless)")
    p.add_argument("--out-dir", "-o",     default=None, metavar="DIR",
                   help="Directorio de salida para guardar PNG (default: no guarda)")
    p.add_argument("--indicators",        default=None, metavar="IND1,IND2",
                   help="Filtro de indicadores separados por coma")
    p.add_argument("--t-gt",              type=float, default=None, metavar="T",
                   help="Tiempo de chatter real para --plot-delay y referencia")
    p.add_argument("--it-snr",            default=None, metavar="SNR1,SNR2",
                   help="SNRs para overlay I_t, ej. '40,80' (default: todos)")
    p.add_argument("--it-snr-case",       default=None, metavar="CASE_OR_SNR",
                   help="Caso SNR a usar en las curvas I_t (nombre exacto o valor SNR del metadata)")
    p.add_argument("--signal-snr-case",    default=None, metavar="CASE_OR_SNR",
                   help="Caso SNR a usar en las señales crudas (nombre exacto o valor SNR del metadata)")
    return p.parse_args()


def main():
    args = parse_args()

    invoked_without_args = (len(sys.argv) == 1)
    if invoked_without_args:
        args.plot_signals = DEFAULT_RUN_PLOT_SIGNALS
        args.plot_detections = DEFAULT_RUN_PLOT_DETECTIONS
        args.plot_it = DEFAULT_RUN_PLOT_IT
        args.plot_it_compare = DEFAULT_RUN_PLOT_IT_COMPARE
        args.plot_lollipop = DEFAULT_RUN_PLOT_LOLLIPOP
        args.plot_delay = DEFAULT_RUN_PLOT_DELAY
        args.plot_far_cost = DEFAULT_RUN_PLOT_FAR_COST
        args.plot_all = False
        args.list_snr = False
        args.show = DEFAULT_RUN_SHOW
        if os.path.isfile(DEFAULT_NOISE_RESULTS):
            args.noise_results = DEFAULT_NOISE_RESULTS
        if os.path.isfile(DEFAULT_INDICATOR_RESULTS):
            args.indicator_results = DEFAULT_INDICATOR_RESULTS
        if args.indicator_results and not args.out_dir:
            args.out_dir = os.path.join(os.path.dirname(args.indicator_results), "figs_indicators")
        if getattr(args, "signal_snr_case", None) is None:
            args.signal_snr_case = SNR_CASES_TO_PLOT
        if args.it_snr_case is None:
            args.it_snr_case = None

    indicators_filter = (
        [s.strip() for s in args.indicators.split(",") if s.strip()]
        if args.indicators else None
    )
    it_snr_filter = (
        [float(s.strip()) for s in args.it_snr.split(",") if s.strip()]
        if args.it_snr else SNR_CASES_TO_PLOT
    )

    # --plot-all activa todo
    if args.plot_all:
        args.plot_signals = args.plot_detections = args.plot_it = True
        args.plot_it_compare = args.plot_lollipop = True
        args.plot_delay = args.plot_far_cost = True

    any_action = any([args.plot_signals, args.plot_detections, args.plot_it,
                      args.plot_it_compare, args.plot_lollipop,
                      args.plot_delay, args.plot_far_cost, args.list_snr])
    if not any_action:
        print("Ninguna acción seleccionada. Usa --plot-all o algún flag de plotting.")
        return

    # ── Señales crudas ────────────────────────────────────────────────────────
    if args.plot_signals:
        if not args.noise_results:
            print("ERROR: --noise_results es obligatorio para --plot-signals"); sys.exit(1)
        h5_noise = os.path.normpath(args.noise_results)
        if not os.path.isfile(h5_noise):
            print(f"ERROR: no encontrado: {h5_noise}"); sys.exit(1)
        data = load_noise_results(h5_noise)
        print(f"Señales — grupos: control + {_snr_groups(data)}")
        plot_signals(
            data,
            out_dir=args.out_dir,
            control_case=CONTROL_CASE,
            snr_case=getattr(args, "signal_snr_case", SNR_CASES_TO_PLOT),
        )

    # ── Requieren indicator_results ───────────────────────────────────────────
    needs_ind = (args.plot_detections or args.plot_it or args.plot_it_compare
                 or args.plot_lollipop or args.plot_delay or args.plot_far_cost
                 or args.list_snr)
    if needs_ind:
        if not args.indicator_results:
            print("ERROR: --indicator_results es obligatorio para estas figuras"); sys.exit(1)
        h5_ind = os.path.normpath(args.indicator_results)
        if not os.path.isfile(h5_ind):
            print(f"ERROR: no encontrado: {h5_ind}"); sys.exit(1)

        df = gather_detection_rows(h5_ind)

        if args.list_snr:
            print(f"SNRs encontrados: {sorted(df['snr_db'].dropna().unique())}")
            return

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            csv_path = os.path.join(args.out_dir, "detection_summary.csv")
            df.to_csv(csv_path, index=False)
            print(f"  CSV resumen: {csv_path}")

        # [D] t_d y t_d_no_FAR vs SNR
        if args.plot_detections:
            plot_detections(df, out_dir=args.out_dir, show=args.show,
                            indicators_filter=indicators_filter, t_gt=args.t_gt)

        # [I] y [C] curvas I_t
        if args.plot_it or args.plot_it_compare:
            curves = gather_indicator_curves(h5_ind, indicators_filter=indicators_filter)
            for ind in sorted(curves.keys()):
                if args.plot_it:
                    plot_it_overlay(curves, ind, snr_filter=it_snr_filter,
                                    out_dir=args.out_dir, t_gt=args.t_gt,
                                    control_case=CONTROL_CASE,
                                    snr_case=args.it_snr_case)
                if args.plot_it_compare:
                    plot_it_compare(curves, ind, out_dir=args.out_dir, t_gt=args.t_gt,
                                    control_case=CONTROL_CASE,
                                    snr_case=args.it_snr_case)

        # [L] Lollipop
        if args.plot_lollipop:
            plot_td_lollipop(df, out_dir=args.out_dir,
                             indicators_filter=indicators_filter)

        # [4] Retraso t_d - t_gt
        if args.plot_delay:
            if args.t_gt is None:
                print("  AVISO: --plot-delay requiere --t-gt. Omitiendo.")
            else:
                plot_delay_vs_snr(df, t_gt=args.t_gt, out_dir=args.out_dir,
                                  indicators_filter=indicators_filter)

        # [5] Coste FAR
        if args.plot_far_cost:
            plot_far_cost_vs_snr(df, out_dir=args.out_dir,
                                 indicators_filter=indicators_filter)

    # Todas las figuras de golpe
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

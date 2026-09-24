"""doe_noise_indicators.py — Aplica indicadores de chatter a cada caso de
doe_noise_results.h5 (control + niveles SNR) y guarda los resultados en
doe_noise_indicator_results.h5.

Uso:
    python doe_noise_indicators.py --doe_noise .\\DOE_xxx\\doe_noise_results.h5
    python doe_noise_indicators.py --doe_noise ... --out resultados.h5
    python doe_noise_indicators.py --doe_noise ... --list
    python doe_noise_indicators.py --doe_noise ... --dry_run
    python doe_noise_indicators.py --doe_noise ... --workers 4

Configuración (editar bloques CONFIG e INDICATOR_CONFIGS):
    _T_GT            : tiempo de onset de chatter conocido [s]
    _CUT_START/END   : ventana de análisis [s]
    NB_WORKERS       : trabajadores paralelos (1 = secuencial)
    ENABLED_CASES    : "all" o lista de grupos ej. ["control", "snr_040.00"]
    INDICATOR_CONFIGS: lista de configuraciones de indicadores a aplicar
"""

from __future__ import annotations

import logging
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import h5py
import numpy as np

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ==============================================================================
# IMPORTS CONDICIONALES DE INDICADORES
# Cada indicador se importa en try/except — si no está instalado, se omite con
# un aviso en lugar de romper todo el script.
# ==============================================================================

_AVAILABLE: Dict[str, bool] = {}

try:
    import sys as _sys
    import os as _os
    _CAMP10 = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP10_Chatter_detection_Methodes"

    # -- MaxEnt-SPRT --
    _maxent_src = _os.path.join(_CAMP10, "indicators", "maxent_sprt", "src")
    if _maxent_src not in _sys.path:
        _sys.path.insert(0, _maxent_src)
    from MaxEnt_SPRT import run_maxent_sprt as _run_maxent_sprt
    from MaxEnt_SPRT import SignalData as _SignalData_maxent
    _AVAILABLE["maxent"] = True
    log.debug("maxent_sprt importado OK")

except ImportError as _e:
    _AVAILABLE["maxent"] = False
    log.warning("maxent_sprt no disponible: %s", _e)

try:
    _rms_src = _os.path.join(_CAMP10, "indicators", "rms_cv", "src")
    if _rms_src not in _sys.path:
        _sys.path.insert(0, _rms_src)
    from rms_cv import run_rms_cv as _run_rms_cv
    from rms_cv import SignalData as _SignalData_rms
    _AVAILABLE["rms_cv"] = True
    log.debug("rms_cv importado OK")
except ImportError as _e:
    _AVAILABLE["rms_cv"] = False
    log.warning("rms_cv no disponible: %s", _e)

try:
    _ssq_src = _os.path.join(_CAMP10, "indicators", "ssq_chatter", "src")
    if _ssq_src not in _sys.path:
        _sys.path.insert(0, _ssq_src)
    from ssq_chatter import run_sst_svd as _run_sst_svd
    from ssq_chatter import SignalData as _SignalData_ssq
    _AVAILABLE["ssq"] = True
    log.debug("ssq_chatter importado OK")
except ImportError as _e:
    _AVAILABLE["ssq"] = False
    log.warning("ssq_chatter no disponible: %s", _e)

try:
    _green_src = _os.path.join(_CAMP10, "indicators", "green_integral", "src")
    if _green_src not in _sys.path:
        _sys.path.insert(0, _green_src)
    from green_integral import run_green_std as _run_green_std        # interfaz estándar
    from green_integral import StdSignalData as _StdSignalData_green  # misma API que MaxEnt
    _AVAILABLE["green_default"] = True
    _AVAILABLE["green_fixed"]   = True
    log.debug("green_integral importado OK")
except ImportError as _e:
    _AVAILABLE["green_default"] = False
    _AVAILABLE["green_fixed"]   = False
    log.warning("green_integral no disponible: %s", _e)

# ==============================================================================
# CONFIG GLOBAL — editar aquí
# ==============================================================================

_T_GT       = 5.365770208787228   # [s] tiempo de onset de chatter (ground truth)
# _T_GT       = 1.07   # [s] tiempo de onset de chatter (ground truth)

_CUT_START  = 0.05                # [s] inicio de la ventana de análisis
_CUT_END    = 16.0                # [s] fin de la ventana de análisis
_RPM        = 12_000.0            # RPM de la simulación
_F_MODAL    = 150.0               # Hz — frecuencia modal del chatter
_T_REV      = 60.0 / _RPM        # [s] periodo de una revolución  (≈ 0.005 s)
_F_REV      = 1.0 / _T_REV       # [Hz] frecuencia de revolución (≈ 200 Hz)
_T_MODAL    = 1.0 / _F_MODAL     # [s] periodo modal (≈ 0.00667 s)

NB_WORKERS       = 6    # 1 = secuencial; >1 = paralelo (ProcessPoolExecutor — un proceso por worker)
ENABLED_CASES    = "all" # "all"  o  lista ej. ["control", "snr_080.00", "snr_040.00"]
SAVE_META_ARRAYS = False  # True → guarda arrays 1D de meta como datasets HDF5; False → los omite

# alpha = beta ≈ norm.sf(3) = 0.00135  →  equivalente a umbral z=3σ
_Z3_ALPHA = 0.00135

# ==============================================================================
# INDICATOR_CONFIGS — lista de configuraciones a aplicar
#
# Campos de cada entrada:
#   enabled         : bool — si False, la entrada se ignora completamente
#   name            : str | None — nombre del grupo en HDF5;
#                     si None → se genera automáticamente como
#                     "{indicator}_{mode_short}_{parámetro_clave}"
#   indicator       : "maxent" | "rms_cv" | "ssq" | "green_default" | "green_fixed"
#   mode            : "by_revolution" | "by_modal"
#   signal          : "Axial_vel" | "Axial_disp"
#   common          : dict con parámetros compartidos DEL INDICADOR
#                     (cada indicador puede tener su propio common distinto)
#   params_physical : dict con parámetros físicos del modo elegido
# ==============================================================================

# -- Common base reutilizable (copiar y ajustar por indicador si es necesario) --
_COMMON_MAXENT = {
    "t_stable_total":     _T_GT,
    "training_intervals": [
        # (_CUT_START, 3.3,   "stable_1"),
        # (3.3,        4.46,  "stable_2"),
        # (4.46,       _T_GT, "stable_1"),
        # (_T_GT,      10.0,  "chatter"),
        (_CUT_START, _T_GT, "stable"),
        (_T_GT,      10.0,  "chatter"),
    ],
    "alpha":       _Z3_ALPHA,
    "beta":        _Z3_ALPHA,
    "reset_on_H0": True,
    "cut_start_time": _CUT_START,
    "cut_end_time":   _CUT_END,
    "t_theorical": _T_GT,  # para debug/plots, no usado en detección
}

#RMS NO USA TRANING INTERVALS 
_COMMON_RMS = {
    # "t_stable_total":     _T_GT,
    # "training_intervals": [
    #     (_CUT_START, _T_GT, "stable"),
    #     (_T_GT,      10.0,  "chatter"),
    # ],
    # "alpha":          _Z3_ALPHA,
    # "beta":           _Z3_ALPHA,
    # "cut_start_time": _CUT_START,
    # "cut_end_time":   _CUT_END,
    # "t_theorical":    _T_GT,  # para debug/plots, no usado en detección

        # fixed threshold (ignored when stable_time is set)
    "cv_threshold":         None,
    "rms_threshold":        None,
    "n_min_cv":             2,
    "warmup_ignore_alerts": False,
    "use_unbiased_std":     True,
    "eps":                  1e-12,
    "detrend":              False,
    "pad_mode":             "none",
    # ── adaptive threshold: 3-sigma on CV of stable region ──────────────
    "stable_time":  (0.0, _T_GT),   # seconds: region known to be stable
    "frac_stable":  0.3610633440512648,         # fallback if stable_time yields no frames
    "z":            3.0,
    "alpha":        0.05,
    "fallback_mad": True,
    "t_theorical":   _T_GT,  # for debug/plots, not used in detection
}

_COMMON_SSQ = {
    "t_stable_total":     _T_GT,
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),
        # (_T_GT,      10.0,  "chatter"),
    ],
    "n_fft_power":  3,
    "mode":         "causal_inclusive",
    "sigma":        6.0,
    "frac_stable":   0.3610633440512648,
    "alpha":        0.05,
    "z":            3.0,
    "fallback_mad": False,
    "t_theorical":  _T_GT,
}

_COMMON_GREEN = {
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),
        # (_T_GT,      10.0,  "chatter"),
    ],
    "z_sigma":        3.0,
    "use_area_threshold": True,
    "t_theorical":    _T_GT,  # para debug/plots, no usado en detección
}

INDICATOR_CONFIGS: List[Dict[str, Any]] = [

    # --------------------------------------------------------------------------
    # MaxEnt-SPRT  ·  by_revolution
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,              # → auto: "maxent_revo_5rev_1step"
        "indicator": "maxent",
        "mode":      "by_revolution",
        "signal":    "Axial_vel",
        "common":    _COMMON_MAXENT,
        "params_physical": {
            "T_rev":         _T_REV,
            "N_rev_window":  7,
            "step_rev":      1,
            "segmentation":  "raw",
            "use_sprt":      True,
        },
    },

    # --------------------------------------------------------------------------
    # MaxEnt-SPRT  ·  by_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,              # → auto: "maxent_modal_3modal_1step"
        "indicator": "maxent",
        "mode":      "by_modal",
        "signal":    "Axial_vel",
        "common":    _COMMON_MAXENT,
        "params_physical": {
            "T_rev":           _T_REV,
            "T_modal":         _T_MODAL,
            "N_modal_window":  3.0,
            "step_modal":      1.0,
            "segmentation":    "raw",
            "use_sprt":        True,
        },
    },

    # --------------------------------------------------------------------------
    # RMS-CV  ·  by_revolution
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,
        "indicator": "rms_cv",
        "mode":      "by_revolution",
        "signal":    "Axial_vel",
        "common":    _COMMON_RMS,
        "params_physical": {
            "T_rev":         _T_REV,
            "N_rev_window": 4,
            "step_rev":     1,
            "n_max_mode":   "frames",
            "n_max_rev":    4,
        },
    },

    # --------------------------------------------------------------------------
    # RMS-CV  ·  by_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "rms_cv",
        "mode":      "by_modal",
        "signal":    "Axial_vel",
        "common":    _COMMON_RMS,
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 1,
            "step_modal":     1,
            "n_max_mode":     "frames",
            "n_max_modal":    16,
        },
    },

    # --------------------------------------------------------------------------
    # SSQ-STFT  ·  by_revolution
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,
        "indicator": "ssq",
        "mode":      "by_revolution",
        "signal":    "Axial_vel",
        "common":    _COMMON_SSQ,
        "params_physical": {
            "T_rev":          _T_REV,
            "N_rev_window":   4,
            "step_rev":       1,
            "Ai_length_mode": "frames",
            "Ai_length_rev":  4,
        },
    },

    # --------------------------------------------------------------------------
    # SSQ-STFT  ·  by_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "ssq",
        "mode":      "by_modal",
        "signal":    "Axial_vel",
        "common":    _COMMON_SSQ,
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 4,
            "step_modal":     1,
            "Ai_length_mode": "frames",
            "Ai_length_modal":2,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Default — ceros cruzados + clustering)  ·  f_cycle=f_rev
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_default",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "f_modal":          _F_MODAL,  # Hz — filtro bandpass
            "f_cycle":          _F_REV,    # Hz — ventana por revolución
            "N_cycles_per_seg": 4.0,
            "step_cycles":      1.0,
            "data_filtrated":       True,
            "hilbert":              False,
            "while_loop_extend":    False,
            "cycles_cluster_points": 35,
            "thein_sen":            False,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Default)  ·  f_cycle=f_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_default",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "f_modal":          _F_MODAL,  # Hz — filtro bandpass y ciclo
            "f_cycle":          _F_MODAL,  # Hz — ventana por periodo modal
            "N_cycles_per_seg": 4,
            "step_cycles":      1.0,
            "data_filtrated":       True,
            "hilbert":              False,
            "while_loop_extend":    False,
            "cycles_cluster_points": 35,
            "thein_sen":            False,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Fixed-Window — Lyapunov σ̂)  ·  f_cycle=f_rev
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,
        "indicator": "green_fixed",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "f_modal":          _F_MODAL,  # Hz — filtro bandpass
            "f_cycle":          _F_REV,    # Hz — ventana por revolución
            "N_cycles_per_seg": 7,
            "step_cycles":      1,
            "data_filtrated":       True,
            "lambda_ewma":          None,
            "accumulate":           False,
            "G_memory":             None,
            "sigma_method":         "ratio",
            "sigma_local_n":        5,
            "area_noise_eps":       1e-25,  # DOE: Axial_disp ~20x menor que tool_dyn original
                                            # área_estable ~ (1e-7)*(3e-7) ~ 3e-14 → eps debe ser < eso
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Fixed-Window)  ·  f_cycle=f_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_fixed",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "param_mode": "by_revolution",

        "params_physical": {
            "f_modal":          _F_MODAL,  # Hz — filtro bandpass y ciclo
            "f_cycle":          _F_MODAL,  # Hz — ventana por periodo modal
            "N_cycles_per_seg": 4,
            "step_cycles":      1.0,
            "data_filtrated":       True,
            "lambda_ewma":          None,
            "accumulate":           False,
            "G_memory":             None,
            "sigma_method":         "ratio",
            "sigma_local_n":        5,
            "area_noise_eps":       1e-25,  # DOE: Axial_disp ~20x menor que tool_dyn original
        },
    },
]

# ==============================================================================
# REGISTRY — mapea indicator key → función run_*
# ==============================================================================

def _build_registry() -> Dict[str, Any]:
    reg = {}
    if _AVAILABLE.get("maxent"):
        reg["maxent"] = _run_maxent_sprt
    if _AVAILABLE.get("rms_cv"):
        reg["rms_cv"] = _run_rms_cv
    if _AVAILABLE.get("ssq"):
        reg["ssq"] = _run_sst_svd
    if _AVAILABLE.get("green_default"):
        reg["green_default"] = _run_green_std   # interfaz estándar
    if _AVAILABLE.get("green_fixed"):
        reg["green_fixed"] = _run_green_std    # mismo runner, func diferente en config
    return reg


_REGISTRY = _build_registry()

# ==============================================================================
# AUTO-NOMBRE
# ==============================================================================

_MODE_SHORT = {
    "by_revolution": "revo",
    "by_modal":      "modal",
}

def _auto_name(cfg: Dict[str, Any]) -> str:
    """Genera un nombre legible desde indicator + parámetros clave.

    Convención:
      dec   = ventana de decisión
      aux   = ventana auxiliar     (N_rev_window / N_modal_window)
      n_aux = nº ventanas aux      (n_max_rev / Ai_length_rev / equiv. modal)
      {s}s  = step

    Formato:
      MaxEnt / Green  → {ind}_{mode}_dec{dec}_{s}s
      RMS / SSQ       → {ind}_{mode}_aux{aux}_naux{n_aux}_dec{dec}_{s}s
                         con  dec = aux + (n_aux - 1) * step
    """
    ind  = cfg["indicator"]
    pp   = cfg.get("params_physical", {})
    mode = _MODE_SHORT.get(cfg.get("mode", ""), cfg.get("mode", ""))

    # ------------------------------------------------------------------
    # Green (default / fixed)  — sin ventana auxiliar
    # ------------------------------------------------------------------
    if ind in ("green_default", "green_fixed"):
        dec = int(pp.get("N_cycles_per_seg", 0))
        s   = int(pp.get("step_cycles", 1))
        return f"{ind}_{mode}_dec{dec}_{s}step"

    # ------------------------------------------------------------------
    # MaxEnt  — sin ventana auxiliar
    # ------------------------------------------------------------------
    if ind == "maxent":
        if cfg.get("mode") == "by_revolution":
            dec = int(pp.get("N_rev_window", 0))
            s   = int(pp.get("step_rev", 1))
        else:  # by_modal
            dec = int(pp.get("N_modal_window", 0))
            s   = int(pp.get("step_modal", 1))
        return f"{ind}_{mode}_dec{dec}_{s}step"

    # ------------------------------------------------------------------
    # RMS-CV / SSQ  — con ventana auxiliar
    # ------------------------------------------------------------------
    if cfg.get("mode") == "by_revolution":
        aux   = int(pp.get("N_rev_window", 0))
        n_aux = int(pp.get("n_max_rev") or pp.get("Ai_length_rev") or 0)
        s     = int(pp.get("step_rev", 1))
    else:  # by_modal
        aux   = int(pp.get("N_modal_window", 0))
        n_aux = int(pp.get("n_max_modal") or pp.get("Ai_length_modal") or 0)
        s     = int(pp.get("step_modal", 1))

    dec = aux + (n_aux - 1) * s
    return f"{ind}_{mode}_aux{aux}_n_aux{n_aux}_dec{dec}_{s}step"


def _run_name(cfg: Dict[str, Any]) -> str:
    """Retorna el nombre del grupo HDF5: explícito si existe, auto si no."""
    return cfg["name"] if cfg.get("name") else _auto_name(cfg)

# ==============================================================================
# HELPERS DE SEÑAL
# ==============================================================================

def _cut_signal(
    t: np.ndarray,
    x: np.ndarray,
    start: float,
    end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t >= start) & (t <= end)
    return t[mask], x[mask]


def _load_case(h5_path: str, grp_name: str, signals: List[str]) -> Dict[str, Any]:
    """Carga un grupo de doe_noise_results.h5 → dict con señales y attrs."""
    with h5py.File(h5_path, "r") as f:
        if grp_name not in f:
            raise KeyError(f"Grupo '{grp_name}' no encontrado en {h5_path}")
        grp   = f[grp_name]
        attrs = dict(grp.attrs)
        sigs  = {}
        for sig in signals:
            if sig in grp:
                t = grp[f"{sig}/time"][()]
                y = grp[f"{sig}/values"][()]
                sigs[sig] = (t, y)
            else:
                log.debug("Señal '%s' no en grupo '%s' — omitida", sig, grp_name)
    return {"group": grp_name, "attrs": attrs, "signals": sigs}


def _make_signal_data(
    t: np.ndarray,
    y: np.ndarray,
    path: str,
    indicator: str,
    meta: Optional[Dict] = None,
):
    """Construye el SignalData adecuado según el indicador.

    - maxent / rms_cv / ssq  : SignalData(t_analysis, signal_analysis, path, fs, meta)
    - green_default / green_fixed : StdSignalData(t_analysis, signal_analysis, path, fs, meta)
      con velocidad en meta["velocity"] si está disponible (green la calcula
      internamente con np.gradient si no se pasa).
    """
    fs  = 1.0 / float(t[1] - t[0])
    m   = meta or {}
    kw  = dict(t_analysis=t, signal_analysis=y, path=path, fs=fs, meta=m)

    if indicator in ("green_default", "green_fixed"):
        return _StdSignalData_green(**kw)

    if indicator == "maxent" and _AVAILABLE.get("maxent"):
        return _SignalData_maxent(**kw)
    if indicator == "rms_cv" and _AVAILABLE.get("rms_cv"):
        return _SignalData_rms(**kw)
    if indicator == "ssq" and _AVAILABLE.get("ssq"):
        return _SignalData_ssq(**kw)
    raise RuntimeError(f"No se pudo construir SignalData para indicador '{indicator}'")

# ==============================================================================
# CONSTRUCCIÓN DEL CONFIG PARA CADA INDICADOR
# ==============================================================================

def _build_indicator_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Combina common + params_physical en el dict que espera cada run_*."""
    indicator = cfg["indicator"]
    mode      = cfg.get("mode", "")
    common    = cfg.get("common", {})
    pp        = cfg.get("params_physical", {})

    if indicator == "maxent":
        return {
            "id":           "MaxEnt_SPRT",
            "func":         "Default",
            "param_mode":   mode,
            "params_physical": {**pp, **common},
        }
    elif indicator == "rms_cv":
        return {
            "id":           "RMS_CV",
            "func":         "Default",
            "param_mode":   mode,
            "params_physical": {**pp, **common},
        }
    elif indicator == "ssq":
        return {
            "id":           "SSQ",
            "func":         "Default",
            "param_mode":   mode,
            "params_physical": {**pp, **common},
        }
    elif indicator in ("green_default", "green_fixed"):
        # Green usa run_green_std con interfaz unificada:
        # f_cycle + N_cycles_per_seg + step_cycles → num_T + dt
        func = "Default" if indicator == "green_default" else "FixedWindow"
        return {
            "func":            func,
            "params_physical": {**pp, **common},
        }
    else:
        raise ValueError(f"Indicador desconocido: '{indicator}'")

# ==============================================================================
# RUNNER POR (caso, config)
# ==============================================================================

def _extract_result(indicator: str, result) -> Dict[str, Any]:
    """Extrae t, I_t, t_d y meta del objeto resultado.

    Todos los indicadores (incluido green) devuelven ahora un IndicatorResult
    estándar con .t, .I_t, .t_d y .meta.  El significado de I_t depende de la
    configuración:
      - maxent / rms_cv / ssq  → LLR acumulado / ratio RMS / energía SSQ
      - green_default          → delta_n  (o A_k si use_area_threshold=True)
      - green_fixed            → sigma_ewma [1/s]  (o A_k si use_area_threshold=True)
    """
    # Todos los runners ahora devuelven IndicatorResult (mismo dataclass)
    t_arr = np.asarray(getattr(result, "t",   []), dtype=float)
    I_arr = np.asarray(getattr(result, "I_t", []), dtype=float)
    t_d   = np.asarray(result.t_d) if result.t_d is not None else np.array([])
    t_d_no_FAR = np.asarray(result.t_d_no_FAR) if result.t_d_no_FAR is not None else np.array([])
    meta  = {k: v for k, v in dict(getattr(result, "meta", {})).items()
             if not callable(v) and k not in ("raw_result", "signal")}
    return {"t": t_arr, "I_t": I_arr, "t_d": t_d, "t_d_no_FAR": t_d_no_FAR, "meta": meta}


def _run_one(
    h5_noise_path: str,
    grp_name: str,
    ind_cfg: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Corre un indicador sobre un caso. Retorna dict con resultados."""
    run_name  = _run_name(ind_cfg)
    indicator = ind_cfg["indicator"]
    signal    = ind_cfg["signal"]

    if not _AVAILABLE.get(indicator, False):
        log.warning("[SKIP] %s/%s — indicador '%s' no disponible", grp_name, run_name, indicator)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "meta": {"skipped": True, "reason": "not_available"},
        }

    log.info("  [%s / %s] INICIO", grp_name, run_name)

    if dry_run:
        log.info("  [%s / %s] DRY-RUN — omitido", grp_name, run_name)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "meta": {"dry_run": True},
        }

    # Cargar señal principal; para green cargar también la velocidad
    _signals_to_load = [signal]
    if indicator in ("green_default", "green_fixed"):
        _signals_to_load.append("Axial_vel")

    case_data  = _load_case(h5_noise_path, grp_name, _signals_to_load)
    sig_tuple  = case_data["signals"].get(signal)
    if sig_tuple is None:
        log.warning("  [%s / %s] señal '%s' no encontrada — SKIP", grp_name, run_name, signal)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "meta": {"skipped": True, "reason": f"signal_{signal}_missing"},
        }

    t_raw, y_raw = sig_tuple
    t_cut, y_cut = _cut_signal(t_raw, y_raw, _CUT_START, _CUT_END)

    # Velocidad para green (si está disponible en el HDF5)
    _vel_meta: Optional[np.ndarray] = None
    if indicator in ("green_default", "green_fixed"):
        vel_tuple = case_data["signals"].get("Axial_vel")
        if vel_tuple is not None:
            _, y_vel_raw = vel_tuple
            _, _vel_meta = _cut_signal(t_raw, y_vel_raw, _CUT_START, _CUT_END)
        else:
            log.debug("  [%s / %s] Axial_vel no encontrada — green usará np.gradient", grp_name, run_name)

    sig_data = _make_signal_data(
        t_cut, y_cut,
        path=h5_noise_path,
        indicator=indicator,
        meta={
            "snr_db":   case_data["attrs"].get("snr_db"),
            "signal":   signal,
            **( {"velocity": _vel_meta} if _vel_meta is not None else {} ),
        },
    )


    ind_config = _build_indicator_config(ind_cfg)

    runner     = _REGISTRY[indicator]

    try:
        result   = runner(sig_data, ind_config)
        extracted = _extract_result(indicator, result)
    except Exception as exc:
        log.error("  [%s / %s] ERROR: %s", grp_name, run_name, exc, exc_info=True)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "t_d_no_FAR": np.array([]), "meta": {"error": str(exc)},
        }

    t_d = extracted["t_d"]
    t_d_no_FAR = extracted.get("t_d_no_FAR", np.array([]))
    if t_d.size > 0:
        log.info("  [%s / %s] t_d = %.4f s", grp_name, run_name, t_d[0])
        log.info("  [%s / %s] t_d_no_FAR = %.4f s", grp_name, run_name, t_d_no_FAR[0])
    else:
        log.info("  [%s / %s] sin detección", grp_name, run_name)

    log.info("  [%s / %s] FIN \n", grp_name, run_name)
    return {
        "case":     grp_name,
        "run_name": run_name,
        "t":        extracted["t"],
        "I_t":      extracted["I_t"],
        "t_d":      t_d,
        "t_d_no_FAR": t_d_no_FAR,
        "meta":     extracted["meta"],
        "ind_cfg":  ind_cfg,
    }

# ==============================================================================
# ESCRITURA HDF5
# ==============================================================================

def _safe_attr(v):
    """Convierte valores de meta a tipos seguros para attrs HDF5.

    HDF5 acepta como atributo: bool, int, float, str, bytes y arrays 1-D de
    tipo numérico escalar.  Todo lo demás se serializa como str.
    """
    if isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, np.ndarray):
        if v.ndim == 1 and v.dtype.kind in ("f", "i", "u", "b"):
            return v
        return str(v.tolist())
    if isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u"):
                return arr
        except Exception:
            pass
        return str(v)
    # dict, object, etc.
    return str(v)


def write_results(out_path: str, all_results: List[Dict[str, Any]],
                  h5_noise_path: Optional[str] = None) -> None:
    """Escribe los resultados en HDF5. Abre en modo 'a' para acumulación incremental.

    Si se proporciona *h5_noise_path*, copia también las señales crudas
    (Axial_disp, Axial_vel, Axial_acc) y los atributos de caso al grupo raíz del caso.
    """
    with h5py.File(out_path, "a") as out_f:
        for res in all_results:
            case_name = res["case"]
            run_name  = res["run_name"]
            path      = f"{case_name}/{run_name}"

            if path in out_f:
                del out_f[path]

            case_grp = out_f.require_group(case_name)

            # ── Copiar attrs + señales crudas del HDF5 fuente (solo 1 vez por caso) ──
            if h5_noise_path and "signals_written" not in case_grp.attrs:
                try:
                    with h5py.File(h5_noise_path, "r") as src_f:
                        if case_name in src_f:
                            # Copiar attrs del caso
                            for k, v in src_f[case_name].attrs.items():
                                case_grp.attrs[k] = v
                            # Copiar señales
                            for sig in ("Axial_disp", "Axial_vel", "Axial_acc"):
                                if sig in src_f[case_name] and sig not in case_grp:
                                    sig_grp = case_grp.require_group(sig)
                                    sig_grp.create_dataset(
                                        "time",
                                        data=src_f[case_name][f"{sig}/time"][()],
                                        compression="gzip",
                                    )
                                    sig_grp.create_dataset(
                                        "values",
                                        data=src_f[case_name][f"{sig}/values"][()],
                                        compression="gzip",
                                    )
                    case_grp.attrs["signals_written"] = True
                except Exception as _e:
                    log.warning("No se pudieron copiar señales del caso '%s': %s", case_name, _e)

            grp = case_grp.require_group(run_name)

            # Datasets principales
            if res["t"].size > 0:
                grp.create_dataset("t",   data=res["t"],   compression="gzip")
            if res["I_t"].size > 0:
                grp.create_dataset("I_t", data=res["I_t"], compression="gzip")
            if res["t_d"].size > 0:
                grp.create_dataset("t_d", data=res["t_d"], compression="gzip")
            if res.get("t_d_no_FAR", np.array([])).size > 0:
                grp.create_dataset("t_d_no_FAR", data=res["t_d_no_FAR"], compression="gzip")
            else:
                grp.create_dataset("t_d_no_FAR", data=np.array([]))

            # Atributos: config usada
            ind_cfg = res.get("ind_cfg", {})
            for field in ("indicator", "mode", "signal"):
                if field in ind_cfg:
                    grp.attrs[field] = str(ind_cfg[field])
            pp = ind_cfg.get("params_physical", {})
            for k, v in pp.items():
                grp.attrs[f"pp_{k}"] = _safe_attr(v)

            # Meta del resultado (solo escalares y strings; arrays según SAVE_META_ARRAYS)
            for k, v in res.get("meta", {}).items():
                if isinstance(v, (list, np.ndarray)):
                    if SAVE_META_ARRAYS:
                        try:
                            arr = np.asarray(v)
                            if arr.dtype.kind in ("f", "i", "u") and arr.ndim == 1:
                                if run_name + f"/{k}" not in out_f:
                                    grp.create_dataset(f"meta_{k}", data=arr, compression="gzip")
                                continue
                        except Exception:
                            pass
                    continue   # omitir si SAVE_META_ARRAYS=False
                grp.attrs[f"meta_{k}"] = _safe_attr(v)

    log.info("Resultados guardados en: %s", out_path)

# ==============================================================================
# EJECUCIÓN PARALELA / SECUENCIAL
# ==============================================================================

def run_all(
    h5_noise_path: str,
    ind_configs: List[Dict[str, Any]],
    nb_workers: int,
    enabled_cases: Any,
    dry_run: bool,
    out_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Corre todos los indicadores sobre todos los casos habilitados.

    Si *out_path* se proporciona, escribe cada resultado al HDF5 de forma
    incremental (en cuanto llega del worker) en lugar de acumular todo en
    memoria.  Esto reduce el pico de RAM de ~(124 × tamaño_resultado) a ~1
    resultado en espera de escritura a la vez.
    """
    # Recopilar grupos disponibles en el HDF5
    with h5py.File(h5_noise_path, "r") as f:
        all_groups = sorted(f.keys())

    if enabled_cases == "all":
        groups = all_groups
    else:
        groups = [g for g in enabled_cases if g in all_groups]
        missing = [g for g in enabled_cases if g not in all_groups]
        if missing:
            log.warning("Grupos no encontrados en HDF5: %s", missing)

    # Solo configs habilitadas y con indicador disponible
    active_cfgs = [
        c for c in ind_configs
        if c.get("enabled", True) and _AVAILABLE.get(c["indicator"], False)
    ]

    tasks = [(grp, cfg) for grp in groups for cfg in active_cfgs]
    total = len(tasks)
    log.info("Total tareas: %d casos × %d configs = %d", len(groups), len(active_cfgs), total)

    results: List[Dict[str, Any]] = []

    def _handle_result(res: Dict[str, Any]) -> None:
        """Escribe incrementalmente si hay out_path; acumula siempre."""
        results.append(res)
        if out_path and not dry_run:
            write_results(out_path, [res], h5_noise_path=h5_noise_path)

    if nb_workers == 1 or dry_run:
        for i, (grp, cfg) in enumerate(tasks, 1):
            log.info("[%d/%d] %s / %s", i, total, grp, _run_name(cfg))
            res = _run_one(h5_noise_path, grp, cfg, dry_run=dry_run)
            _handle_result(res)
    else:
        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            future_map = {
                executor.submit(_run_one, h5_noise_path, grp, cfg, dry_run): (grp, cfg)
                for grp, cfg in tasks
            }
            done = 0
            for future in as_completed(future_map):
                grp, cfg = future_map[future]
                try:
                    res = future.result()
                    _handle_result(res)
                except Exception as exc:
                    log.error("Tarea %s / %s falló: %s", grp, _run_name(cfg), exc)
                done += 1
                log.info("[%d/%d] completado: %s / %s", done, total, grp, _run_name(cfg))

    return results

# ==============================================================================
# --list
# ==============================================================================

def list_cases(h5_path: str) -> None:
    """Muestra tabla de grupos disponibles en el HDF5 de ruido."""
    with h5py.File(h5_path, "r") as f:
        groups = sorted(f.keys())
        if not groups:
            print("  (sin casos)")
            return
        rows = []
        all_keys: List[str] = []
        for grp_name in groups:
            attrs = dict(f[grp_name].attrs)
            rows.append((grp_name, attrs))
            for k in attrs:
                if k not in all_keys:
                    all_keys.append(k)
        col_names = ["idx", "group"] + all_keys
        table = []
        for i, (grp_name, attrs) in enumerate(rows):
            row = [str(i), grp_name] + [str(attrs.get(k, "-")) for k in all_keys]
            table.append(row)
        widths = [
            max(len(col_names[j]), max(len(r[j]) for r in table))
            for j in range(len(col_names))
        ]
        sep    = "  ".join("-" * w for w in widths)
        header = "  ".join(col_names[j].ljust(widths[j]) for j in range(len(col_names)))
        print()
        print(header)
        print(sep)
        for row in table:
            print("  ".join(row[j].ljust(widths[j]) for j in range(len(col_names))))
        print()
        print(f"  Total: {len(rows)} casos")
        print()

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    epilog = """\
Ejemplos:
  python doe_noise_indicators.py --doe_noise .\\DOE_xxx\\doe_noise_results.h5 --list
      Muestra los grupos disponibles y sale.

  python doe_noise_indicators.py --doe_noise .\\DOE_xxx\\doe_noise_results.h5 --dry_run
      Simula la ejecución (no corre indicadores, solo imprime plan).

  python doe_noise_indicators.py --doe_noise .\\DOE_xxx\\doe_noise_results.h5 --workers 4
      Corre 4 indicadores en paralelo.

  python doe_noise_indicators.py --doe_noise ... --out resultados.h5 --workers 1
      Corre secuencial y guarda en archivo personalizado.

Configuración (editar en el script):
  INDICATOR_CONFIGS   : lista de dicts con enabled, name, indicator, mode, signal, common, params_physical
  ENABLED_CASES       : "all" o lista de grupos ej. ["control", "snr_040.00"]
  NB_WORKERS          : workers por defecto (sobreescribible con --workers)
"""
    p = argparse.ArgumentParser(
        description="doe_noise_indicators — Aplica indicadores de chatter a casos con ruido.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--doe_noise", required=True, metavar="PATH",
        help="Ruta a doe_noise_results.h5",
    )
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="Ruta del HDF5 de salida (default: doe_noise_indicator_results.h5 junto a --doe_noise)",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help=f"Número de workers paralelos (default: NB_WORKERS={NB_WORKERS})",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Simula sin correr indicadores — imprime plan de tareas.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Imprime la tabla de casos disponibles y sale.",
    )
    return p.parse_args()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    h5_noise = os.path.normpath(args.doe_noise)
    if not os.path.isfile(h5_noise):
        log.error("Archivo no encontrado: %s", h5_noise)
        sys.exit(1)

    if args.list:
        list_cases(h5_noise)
        sys.exit(0)

    out_path = (
        os.path.normpath(args.out)
        if args.out
        else os.path.join(os.path.dirname(h5_noise), "doe_noise_indicator_results.h5")
    )

    workers  = args.workers if args.workers is not None else NB_WORKERS

    log.info("doe_noise_indicators")
    log.info("  Entrada  : %s", h5_noise)
    log.info("  Salida   : %s", out_path)
    log.info("  Workers  : %d", workers)
    log.info("  Dry-run  : %s", args.dry_run)
    log.info("  Casos    : %s", ENABLED_CASES)
    log.info("  Indicadores disponibles: %s", [k for k, v in _AVAILABLE.items() if v])

    active = [c for c in INDICATOR_CONFIGS if c.get("enabled", True)]
    log.info("  Configs activas: %d", len(active))
    for c in active:
        log.info("    %-20s  mode=%-14s  signal=%s  name=%s",
                 c["indicator"], c.get("mode", "f_cycle"), c["signal"],
                 _run_name(c))

    results = run_all(
        h5_noise_path  = h5_noise,
        ind_configs    = active,
        nb_workers     = workers,
        enabled_cases  = ENABLED_CASES,
        dry_run        = args.dry_run,
        out_path       = out_path if not args.dry_run else None,
    )

    if not args.dry_run:
        log.info("Listo. %d resultados escritos incrementalmente en %s", len(results), out_path)
    else:
        log.info("[DRY-RUN] %d tareas planificadas — nada escrito.", len(results))


if __name__ == "__main__":
    main()

"""doe_indicators.py — Aplica indicadores de chatter a cada caso de
doe_results.h5 (case_000 … case_NNN) y guarda los resultados en
doe_indicator_results.h5.

Uso desde terminal:
    python doe_indicators.py --doe_results .\\DOE_xxx\\doe_results.h5
    python doe_indicators.py --doe_results ...  --out resultados.h5
    python doe_indicators.py --doe_results ...  --label_key "$dxl_size$"
    python doe_indicators.py --doe_results ...  --list
    python doe_indicators.py --doe_results ...  --dry_run
    python doe_indicators.py --doe_results ...  --workers 4

Uso desde VS Code (sin argumentos):
    Editar el bloque CONFIG y ejecutar directamente.

Configuración (editar bloques CONFIG e INDICATOR_CONFIGS):
    DOE_NAME         : nombre de la carpeta DOE dentro de BASE_DIR
    LABEL_KEY        : None → auto-detección desde attrs del HDF5
                       str  → clave explícita ej. "$dxl_size$"
    _T_GT            : tiempo de onset de chatter conocido [s]
    _CUT_START/END   : ventana de análisis [s]
    NB_WORKERS       : trabajadores paralelos (1 = secuencial)
    ENABLED_CASES    : "all" o lista de grupos ej. ["case_000", "case_003"]
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
    from green_integral import run_green_std as _run_green_std
    from green_integral import StdSignalData as _StdSignalData_green
    _AVAILABLE["green_default"] = True
    _AVAILABLE["green_fixed"]   = True
    log.debug("green_integral importado OK")
except ImportError as _e:
    _AVAILABLE["green_default"] = False
    _AVAILABLE["green_fixed"]   = False
    log.warning("green_integral no disponible: %s", _e)

# ==============================================================================
# CONFIG GLOBAL — editar aquí para lanzar desde VS Code sin argumentos
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = SCRIPT_DIR

# DOE_NAME   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
DOE_NAME   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"
CASE_NAME  = "1DOF_150Hz"

# LABEL_KEY: None → auto-detectar desde attrs del HDF5
#            str  → clave explícita ej. "$dxl_size$" o "$nb_dt_rev$"
LABEL_KEY = None

_T_GT       = 5.365770208787228   # [s] tiempo de onset de chatter (ground truth)

_CUT_START  = 0.05               # [s] inicio de la ventana de análisis
_CUT_END    = 16.0                # [s] fin de la ventana de análisis
_RPM        = 12_000.0            # RPM de la simulación
_F_MODAL    = 150.0               # Hz — frecuencia modal del chatter
_T_REV      = 60.0 / _RPM        # [s] periodo de una revolución  (≈ 0.005 s)
_F_REV      = 1.0 / _T_REV       # [Hz] frecuencia de revolución (≈ 200 Hz)
_T_MODAL    = 1.0 / _F_MODAL     # [s] periodo modal (≈ 0.00667 s)

NB_WORKERS       = 6    # 1 = secuencial; >1 = paralelo (ProcessPoolExecutor)
ENABLED_CASES    = "all" # "all"  o  lista ej. ["case_000", "case_003"]
SAVE_META_ARRAYS = False  # True → guarda arrays 1D de meta como datasets HDF5

# alpha = beta ≈ norm.sf(3) = 0.00135  →  equivalente a umbral z=3σ
_Z3_ALPHA = 0.00135

# ==============================================================================
# INDICATOR_CONFIGS — lista de configuraciones a aplicar
#
# Misma estructura que doe_noise_indicators.py.
# Copiar/ajustar según el DOE en uso.
# ==============================================================================

_COMMON_MAXENT = {
    "t_stable_total":     _T_GT,
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),
        (_T_GT,      10.0,  "chatter"),
    ],
    "alpha":       _Z3_ALPHA,
    "beta":        _Z3_ALPHA,
    "reset_on_H0": True,
    "cut_start_time": _CUT_START,
    "cut_end_time":   _CUT_END,
    "t_theorical": _T_GT,
}

_COMMON_RMS = {
    "cv_threshold":         None,
    "rms_threshold":        None,
    "n_min_cv":             2,
    "warmup_ignore_alerts": False,
    "use_unbiased_std":     True,
    "eps":                  1e-12,
    "detrend":              False,
    "pad_mode":             "none",
    "stable_time":  (0.0, _T_GT),
    "frac_stable":  0.3610633440512648,
    "z":            3.0,
    "alpha":        0.05,
    "fallback_mad": True,
    "t_theorical":  _T_GT,
}

_COMMON_SSQ = {
    "t_stable_total":     _T_GT,
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),
    ],
    "n_fft_power":  3,
    "mode":         "causal_inclusive",
    "sigma":        6.0,
    "frac_stable":  0.3610633440512648,
    "alpha":        0.05,
    "z":            3.0,
    "fallback_mad": False,
    "t_theorical":  _T_GT,
}

_COMMON_GREEN = {
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),
    ],
    "z_sigma":            3.0,
    "use_area_threshold": True,
    "t_theorical":        _T_GT,
    "use_zero_crossing_cycles": True, # alpha cycles
    "use_beta_from_cycles": False, # Creat beta from alpha cycles
    "zc_detrend": True,
    "v_cycle_mode": "zero",  # "zero - 1" | "original - dentrend for v=0 , poyecion a Trayectoria Original" | 
                            #  "detrended - detren for v=0 , trayectoria detrend"
    "cycle_area_norm": "none",  # "none" | "mean" | "median"

}

INDICATOR_CONFIGS: List[Dict[str, Any]] = [

    # --------------------------------------------------------------------------
    # MaxEnt-SPRT  ·  by_revolution
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,
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
        "name":      None,
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
            "T_rev":        _T_REV,
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
            "T_modal":         _T_MODAL,
            "N_modal_window":  4,
            "step_modal":      1,
            "Ai_length_mode":  "frames",
            "Ai_length_modal": 2,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Default)  ·  f_cycle=f_rev
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_default",
        "mode":      "by_revolution",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "T_rev":                 _T_REV,
            "N_rev_window":          4.0,
            "step_rev":              1.0,
            "data_filtrated":        True,
            "hilbert":               False,
            "while_loop_extend":     False,
            "cycles_cluster_points": 35,
            "thein_sen":             False,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Default)  ·  f_cycle=f_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_default",
        "mode":      "by_modal",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "T_modal":               _T_MODAL,
            "N_modal_window":        4,
            "step_modal":            1.0,
            "data_filtrated":        True,
            "hilbert":               False,
            "while_loop_extend":     False,
            "cycles_cluster_points": 35,
            "thein_sen":             False,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Fixed-Window)  ·  f_cycle=f_rev
    # --------------------------------------------------------------------------
    {
        "enabled":   True,
        "name":      None,
        "indicator": "green_fixed",
        "mode":      "by_revolution",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "T_rev":            _T_REV,
            "N_rev_window":     7,
            "step_rev":         1,
            "data_filtrated":   True,
            "lambda_ewma":      None,
            "accumulate":       False,
            "G_memory":         None,
            "sigma_method":     "ratio",
            "sigma_local_n":    5,
            "area_noise_eps":   1e-25,
        },
    },

    # --------------------------------------------------------------------------
    # Green Integral (Fixed-Window)  ·  f_cycle=f_modal
    # --------------------------------------------------------------------------
    {
        "enabled":   False,
        "name":      None,
        "indicator": "green_fixed",
        "mode":      "by_modal",
        "signal":    "Axial_disp",
        "common":    _COMMON_GREEN,
        "params_physical": {
            "T_modal":          _T_MODAL,
            "N_modal_window":   4,
            "step_modal":       1.0,
            "data_filtrated":   True,
            "lambda_ewma":      None,
            "accumulate":       False,
            "G_memory":         None,
            "sigma_method":     "ratio",
            "sigma_local_n":    5,
            "area_noise_eps":   1e-25,
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
        reg["green_default"] = _run_green_std
    if _AVAILABLE.get("green_fixed"):
        reg["green_fixed"]   = _run_green_std
    return reg


_REGISTRY = _build_registry()

# ==============================================================================
# AUTO-NOMBRE
# ==============================================================================

_MODE_SHORT = {
    "by_revolution": "revo",
    "by_modal":      "modal",
    "f_cycle":       "fcycle",
    "":              "",
}


def _auto_name(cfg: Dict[str, Any]) -> str:
    ind  = cfg["indicator"]
    pp   = cfg.get("params_physical", {})
    mode = _MODE_SHORT.get(cfg.get("mode", ""), cfg.get("mode", ""))

    if ind in ("green_default", "green_fixed"):
        if cfg.get("mode") == "by_revolution":
            dec = int(pp.get("N_rev_window", 0))
            s   = int(pp.get("step_rev", 1))
        else:
            dec = int(pp.get("N_modal_window", 0))
            s   = int(pp.get("step_modal", 1))
        # green no siempre tiene mode key → omitir si vacío
        mode_str = f"_{mode}" if mode else ""
        return f"{ind}{mode_str}_dec{dec}_{s}step"

    if ind == "maxent":
        if cfg.get("mode") == "by_revolution":
            dec = int(pp.get("N_rev_window", 0))
            s   = int(pp.get("step_rev", 1))
        else:
            dec = int(pp.get("N_modal_window", 0))
            s   = int(pp.get("step_modal", 1))
        return f"{ind}_{mode}_dec{dec}_{s}step"

    if cfg.get("mode") == "by_revolution":
        aux   = int(pp.get("N_rev_window", 0))
        n_aux = int(pp.get("n_max_rev") or pp.get("Ai_length_rev") or 0)
        s     = int(pp.get("step_rev", 1))
    else:
        aux   = int(pp.get("N_modal_window", 0))
        n_aux = int(pp.get("n_max_modal") or pp.get("Ai_length_modal") or 0)
        s     = int(pp.get("step_modal", 1))

    dec = aux + (n_aux - 1) * s
    return f"{ind}_{mode}_aux{aux}_n_aux{n_aux}_dec{dec}_{s}step"


def _run_name(cfg: Dict[str, Any]) -> str:
    return cfg["name"] if cfg.get("name") else _auto_name(cfg)

# ==============================================================================
# AUTO-DETECCIÓN DE LABEL_KEY
# ==============================================================================

def _detect_label_key(h5_path: str) -> str:
    """Auto-detecta el parámetro DOE que varía entre casos en doe_results.h5.

    Reglas:
      - Solo considera atributos con formato "$...$" (parámetros DOE, excluye wall_time_s etc.)
      - Si exactamente 1 varía → retorna ese key.
      - Si >1 varían → lanza ValueError con lista de candidatos y sugerencia --label_key.
      - Si 0 varían → retorna el primer attr "$...$" disponible (caso degenerado).
    """
    with h5py.File(h5_path, "r") as f:
        case_keys = [k for k in f.keys() if k.startswith("case_")]
        if not case_keys:
            raise ValueError(f"No se encontraron grupos 'case_*' en {h5_path}")

        # Recopilar valores por attr (solo claves con patrón $...$)
        all_attr_keys = [
            k for k in f[case_keys[0]].attrs.keys()
            if k.startswith("$") and k.endswith("$")
        ]
        if not all_attr_keys:
            raise ValueError(
                f"No se encontraron atributos DOE (formato '$...$') en {h5_path}. "
                "Usa --label_key para especificar una clave manualmente."
            )

        varying = []
        for ak in all_attr_keys:
            vals = set()
            for ck in case_keys:
                v = f[ck].attrs.get(ak)
                if v is not None:
                    vals.add(str(float(v)))
            if len(vals) > 1:
                varying.append(ak)

    if len(varying) == 1:
        log.info("LABEL_KEY auto-detectado: '%s'", varying[0])
        return varying[0]

    if len(varying) > 1:
        raise ValueError(
            f"Múltiples parámetros DOE varían entre casos: {varying}.\n"
            "  Usa --label_key para especificar uno, p.ej.:\n"
            f"    --label_key \"{varying[0]}\""
        )

    # Ninguno varía (caso degenerado): usar el primero disponible
    log.warning(
        "Ningún parámetro DOE varía entre casos. Usando '%s' como LABEL_KEY.", all_attr_keys[0]
    )
    return all_attr_keys[0]

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
    """Carga un grupo de doe_results.h5 → dict con señales y attrs."""
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
    indicator = cfg["indicator"]
    mode      = cfg.get("mode", "")
    common    = cfg.get("common", {})
    pp        = cfg.get("params_physical", {})

    if indicator == "maxent":
        return {
            "id":              "MaxEnt_SPRT",
            "func":            "Default",
            "param_mode":      mode,
            "params_physical": {**pp, **common},
        }
    elif indicator == "rms_cv":
        return {
            "id":              "RMS_CV",
            "func":            "Default",
            "param_mode":      mode,
            "params_physical": {**pp, **common},
        }
    elif indicator == "ssq":
        return {
            "id":              "SSQ",
            "func":            "Default",
            "param_mode":      mode,
            "params_physical": {**pp, **common},
        }
    elif indicator in ("green_default", "green_fixed"):
        func = "Default" if indicator == "green_default" else "Lyapunov"
        return {
            "func":            func,
            "param_mode":      mode,
            "params_physical": {**pp, **common},
        }
    else:
        raise ValueError(f"Indicador desconocido: '{indicator}'")

# ==============================================================================
# EXTRACCIÓN DEL RESULTADO
# ==============================================================================

def _extract_result(indicator: str, result) -> Dict[str, Any]:
    t_arr = np.asarray(getattr(result, "t",   []), dtype=float)
    I_arr = np.asarray(getattr(result, "I_t", []), dtype=float)
    t_d   = np.asarray(result.t_d) if result.t_d is not None else np.array([])
    t_d_no_FAR = (
        np.asarray(result.t_d_no_FAR)
        if result.t_d_no_FAR is not None
        else np.array([])
    )
    meta = {
        k: v for k, v in dict(getattr(result, "meta", {})).items()
        if not callable(v) and k not in ("raw_result", "signal")
    }
    return {"t": t_arr, "I_t": I_arr, "t_d": t_d, "t_d_no_FAR": t_d_no_FAR, "meta": meta}

# ==============================================================================
# RUNNER POR (caso, config)
# ==============================================================================

def _run_one(
    h5_path: str,
    grp_name: str,
    ind_cfg: Dict[str, Any],
    label_key: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Corre un indicador sobre un caso DOE. Retorna dict con resultados."""
    run_name  = _run_name(ind_cfg)
    indicator = ind_cfg["indicator"]
    signal    = ind_cfg["signal"]

    if not _AVAILABLE.get(indicator, False):
        log.warning("[SKIP] %s/%s — indicador '%s' no disponible", grp_name, run_name, indicator)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "t_d_no_FAR": np.array([]),
            "meta": {"skipped": True, "reason": "not_available"},
            "label_key": label_key, "label_val": float("nan"),
        }

    log.info("  [%s / %s] INICIO", grp_name, run_name)

    if dry_run:
        log.info("  [%s / %s] DRY-RUN — omitido", grp_name, run_name)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "t_d_no_FAR": np.array([]),
            "meta": {"dry_run": True},
            "label_key": label_key, "label_val": float("nan"),
        }

    # Cargar señal principal; para green cargar también la velocidad
    _signals_to_load = [signal]
    if indicator in ("green_default", "green_fixed"):
        _signals_to_load.append("Axial_vel")

    case_data  = _load_case(h5_path, grp_name, _signals_to_load)
    sig_tuple  = case_data["signals"].get(signal)
    if sig_tuple is None:
        log.warning("  [%s / %s] señal '%s' no encontrada — SKIP", grp_name, run_name, signal)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "t_d_no_FAR": np.array([]),
            "meta": {"skipped": True, "reason": f"signal_{signal}_missing"},
            "label_key": label_key, "label_val": float("nan"),
        }

    # Obtener valor del LABEL_KEY para este caso
    raw_label = case_data["attrs"].get(label_key)
    try:
        label_val = float(raw_label) if raw_label is not None else float("nan")
    except (TypeError, ValueError):
        label_val = float("nan")

    t_raw, y_raw = sig_tuple
    t_cut, y_cut = _cut_signal(t_raw, y_raw, _CUT_START, _CUT_END)

    # Velocidad para green (si está disponible)
    _vel_meta: Optional[np.ndarray] = None
    if indicator in ("green_default", "green_fixed"):
        vel_tuple = case_data["signals"].get("Axial_vel")
        if vel_tuple is not None:
            _, y_vel_raw = vel_tuple
            _, _vel_meta = _cut_signal(t_raw, y_vel_raw, _CUT_START, _CUT_END)
        else:
            log.debug(
                "  [%s / %s] Axial_vel no encontrada — green usará np.gradient",
                grp_name, run_name,
            )

    sig_data = _make_signal_data(
        t_cut, y_cut,
        path=h5_path,
        indicator=indicator,
        meta={
            "label_key": label_key,
            "label_val": label_val,
            "signal":    signal,
            **( {"velocity": _vel_meta} if _vel_meta is not None else {} ),
        },
    )

    ind_config = _build_indicator_config(ind_cfg)
    runner     = _REGISTRY[indicator]

    try:
        result    = runner(sig_data, ind_config)
        extracted = _extract_result(indicator, result)
    except Exception as exc:
        log.error("  [%s / %s] ERROR: %s", grp_name, run_name, exc, exc_info=True)
        return {
            "case": grp_name, "run_name": run_name,
            "t": np.array([]), "I_t": np.array([]),
            "t_d": np.array([]), "t_d_no_FAR": np.array([]),
            "meta": {"error": str(exc)},
            "label_key": label_key, "label_val": label_val,
        }

    t_d        = extracted["t_d"]
    t_d_no_FAR = extracted.get("t_d_no_FAR", np.array([]))

    if t_d.size > 0:
        log.info("  [%s / %s] t_d = %.4f s", grp_name, run_name, t_d[0])
        log.info("  [%s / %s] t_d_no_FAR = %.4f s", grp_name, run_name, t_d_no_FAR[0] if t_d_no_FAR.size > 0 else float("nan"))
    else:
        log.info("  [%s / %s] sin detección", grp_name, run_name)

    log.info("  [%s / %s] FIN \n", grp_name, run_name)
    return {
        "case":        grp_name,
        "run_name":    run_name,
        "t":           extracted["t"],
        "I_t":         extracted["I_t"],
        "t_d":         t_d,
        "t_d_no_FAR":  t_d_no_FAR,
        "meta":        extracted["meta"],
        "ind_cfg":     ind_cfg,
        "label_key":   label_key,
        "label_val":   label_val,
    }

# ==============================================================================
# ESCRITURA HDF5
# ==============================================================================

def _safe_attr(v):
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
    return str(v)


def write_results(out_path: str, all_results: List[Dict[str, Any]],
                  h5_doe_path: Optional[str] = None) -> None:
    """Escribe los resultados en HDF5. Abre en modo 'a' para acumulación incremental.

    Si se proporciona *h5_doe_path*, copia también las señales crudas
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
            if h5_doe_path and "signals_written" not in case_grp.attrs:
                try:
                    with h5py.File(h5_doe_path, "r") as src_f:
                        if case_name in src_f:
                            for k, v in src_f[case_name].attrs.items():
                                case_grp.attrs[k] = v
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
                    # Guardar label_key como attr del caso para uso del plotter
                    if res.get("label_key"):
                        case_grp.attrs["label_key"] = res["label_key"]
                    if not np.isnan(res.get("label_val", float("nan"))):
                        case_grp.attrs["label_val"] = res["label_val"]
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
            t_d_no_FAR = res.get("t_d_no_FAR", np.array([]))
            if t_d_no_FAR.size > 0:
                grp.create_dataset("t_d_no_FAR", data=t_d_no_FAR, compression="gzip")
            else:
                grp.create_dataset("t_d_no_FAR", data=np.array([]))

            # Atributos: label_key / label_val
            if res.get("label_key"):
                grp.attrs["label_key"] = res["label_key"]
            label_val = res.get("label_val", float("nan"))
            if not np.isnan(label_val):
                grp.attrs["label_val"] = label_val

            # Atributos: config usada
            ind_cfg = res.get("ind_cfg", {})
            for field in ("indicator", "mode", "signal"):
                if field in ind_cfg:
                    grp.attrs[field] = str(ind_cfg[field])
            for k, v in ind_cfg.get("params_physical", {}).items():
                grp.attrs[f"pp_{k}"] = _safe_attr(v)

            # Meta del resultado
            for k, v in res.get("meta", {}).items():
                if isinstance(v, (list, np.ndarray)):
                    if SAVE_META_ARRAYS:
                        try:
                            arr = np.asarray(v)
                            if arr.dtype.kind in ("f", "i", "u") and arr.ndim == 1:
                                grp.create_dataset(f"meta_{k}", data=arr, compression="gzip")
                                continue
                        except Exception:
                            pass
                    continue
                grp.attrs[f"meta_{k}"] = _safe_attr(v)

    log.info("Resultados guardados en: %s", out_path)

# ==============================================================================
# EJECUCIÓN PARALELA / SECUENCIAL
# ==============================================================================

def run_all(
    h5_doe_path: str,
    ind_configs: List[Dict[str, Any]],
    nb_workers: int,
    enabled_cases: Any,
    dry_run: bool,
    label_key: str,
    out_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Corre todos los indicadores sobre todos los casos habilitados.

    Escribe resultados incrementalmente al HDF5 si se pasa *out_path*.
    """
    with h5py.File(h5_doe_path, "r") as f:
        all_groups = sorted(k for k in f.keys() if k.startswith("case_"))

    if enabled_cases == "all":
        groups = all_groups
    else:
        groups  = [g for g in enabled_cases if g in all_groups]
        missing = [g for g in enabled_cases if g not in all_groups]
        if missing:
            log.warning("Grupos no encontrados en HDF5: %s", missing)

    active_cfgs = [
        c for c in ind_configs
        if c.get("enabled", True) and _AVAILABLE.get(c["indicator"], False)
    ]

    tasks = [(grp, cfg) for grp in groups for cfg in active_cfgs]
    total = len(tasks)
    log.info("Total tareas: %d casos × %d configs = %d", len(groups), len(active_cfgs), total)

    results: List[Dict[str, Any]] = []

    def _handle_result(res: Dict[str, Any]) -> None:
        results.append(res)
        if out_path and not dry_run:
            write_results(out_path, [res], h5_doe_path=h5_doe_path)

    if nb_workers == 1 or dry_run:
        for i, (grp, cfg) in enumerate(tasks, 1):
            log.info("[%d/%d] %s / %s", i, total, grp, _run_name(cfg))
            res = _run_one(h5_doe_path, grp, cfg, label_key=label_key, dry_run=dry_run)
            _handle_result(res)
    else:
        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            future_map = {
                executor.submit(_run_one, h5_doe_path, grp, cfg, label_key, dry_run): (grp, cfg)
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

def list_cases(h5_path: str, label_key: Optional[str] = None) -> None:
    """Muestra tabla de grupos disponibles en doe_results.h5."""
    with h5py.File(h5_path, "r") as f:
        groups = sorted(k for k in f.keys() if k.startswith("case_"))
        if not groups:
            print("  (sin grupos case_*)")
            return

        # Auto-detectar label_key si no se pasa
        if label_key is None:
            try:
                label_key = _detect_label_key(h5_path)
            except ValueError:
                label_key = None

        rows = []
        for grp_name in groups:
            attrs = dict(f[grp_name].attrs)
            signals = [k for k in f[grp_name].keys() if isinstance(f[grp_name][k], h5py.Group)]
            rows.append((grp_name, attrs, signals))

    # Columnas: group | label_key=val | señales
    col_group  = max(len("group"), max(len(r[0]) for r in rows))
    col_label  = 16
    col_sig    = 30

    lk_header = label_key if label_key else "label_key"

    print()
    header = f"{'group':<{col_group}}  {lk_header:>{col_label}}  {'signals':<{col_sig}}"
    print(header)
    print("-" * len(header))
    for grp_name, attrs, signals in rows:
        if label_key:
            raw = attrs.get(label_key)
            try:
                lv = f"{float(raw):>{col_label}g}" if raw is not None else f"{'N/A':>{col_label}}"
            except (TypeError, ValueError):
                lv = f"{str(raw):>{col_label}}"
        else:
            lv = f"{'(auto??)':>{col_label}}"
        sig_str = ", ".join(signals[:4]) + ("..." if len(signals) > 4 else "")
        print(f"{grp_name:<{col_group}}  {lv}  {sig_str:<{col_sig}}")

    print()
    print(f"  Total: {len(rows)} casos  |  label_key: {label_key}")
    print()

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    epilog = """\
Ejemplos:
  python doe_indicators.py --doe_results .\\DOE_xxx\\doe_results.h5 --list
      Muestra los grupos disponibles y sale.

  python doe_indicators.py --doe_results .\\DOE_xxx\\doe_results.h5 --dry_run
      Simula la ejecución (no corre indicadores, solo imprime plan).

  python doe_indicators.py --doe_results .\\DOE_xxx\\doe_results.h5 --workers 4
      Corre 4 workers en paralelo con LABEL_KEY auto-detectado.

  python doe_indicators.py --doe_results ... --label_key "$nb_dt_rev$"
      Sobreescribe la auto-detección de LABEL_KEY.

  python doe_indicators.py --doe_results ... --out resultados.h5
      Guarda en un archivo personalizado.

Configuración (editar en el script):
  DOE_NAME, LABEL_KEY, NB_WORKERS, ENABLED_CASES, INDICATOR_CONFIGS
"""
    p = argparse.ArgumentParser(
        description="doe_indicators — Aplica indicadores de chatter a casos del DOE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--doe_results", default=None, metavar="PATH",
        help="Ruta a doe_results.h5 (si se omite: usa DOE_NAME del CONFIG)",
    )
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="Ruta del HDF5 de salida (default: doe_indicator_results.h5 junto a --doe_results)",
    )
    p.add_argument(
        "--label_key", default=None, metavar="KEY",
        help="Clave DOE para etiquetar cada caso (default: auto-detectar). Ej: \"$dxl_size$\"",
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
    global LABEL_KEY, NB_WORKERS  # noqa: PLW0603

    args = parse_args()

    # -- Detectar si se lanzó desde terminal o desde VS Code --
    has_cli = any([
        args.doe_results, args.out, args.label_key,
        args.workers, args.dry_run, args.list,
    ])

    # -- Resolver ruta al HDF5 de entrada --
    if args.doe_results:
        h5_doe = os.path.normpath(args.doe_results)
    else:
        doe_dir = os.path.normpath(os.path.join(BASE_DIR, DOE_NAME))
        h5_doe  = os.path.join(doe_dir, "doe_results.h5")

    if not os.path.isfile(h5_doe):
        log.error("Archivo no encontrado: %s", h5_doe)
        if not has_cli:
            log.error("  Edita DOE_NAME en el bloque CONFIG del script.")
        sys.exit(1)

    # -- --list --
    if args.list:
        lk = args.label_key or LABEL_KEY
        list_cases(h5_doe, label_key=lk)
        sys.exit(0)

    # -- Resolver LABEL_KEY --
    if args.label_key:
        LABEL_KEY = args.label_key
    if LABEL_KEY is None:
        try:
            LABEL_KEY = _detect_label_key(h5_doe)
        except ValueError as exc:
            log.error("No se pudo auto-detectar LABEL_KEY:\n  %s", exc)
            sys.exit(1)

    # -- Otros overrides de CLI --
    if args.workers is not None:
        NB_WORKERS = args.workers
    workers = NB_WORKERS

    # -- Ruta de salida --
    out_path = (
        os.path.normpath(args.out)
        if args.out
        else os.path.join(os.path.dirname(h5_doe), "doe_indicator_results.h5")
    )

    log.info("doe_indicators")
    log.info("  Entrada    : %s", h5_doe)
    log.info("  Salida     : %s", out_path)
    log.info("  LABEL_KEY  : %s", LABEL_KEY)
    log.info("  Workers    : %d", workers)
    log.info("  Dry-run    : %s", args.dry_run)
    log.info("  Casos      : %s", ENABLED_CASES)
    log.info("  Indicadores disponibles: %s", [k for k, v in _AVAILABLE.items() if v])

    active = [c for c in INDICATOR_CONFIGS if c.get("enabled", True)]
    log.info("  Configs activas: %d", len(active))
    for c in active:
        log.info(
            "    %-20s  mode=%-14s  signal=%s  name=%s",
            c["indicator"], c.get("mode", "f_cycle"), c["signal"], _run_name(c),
        )

    results = run_all(
        h5_doe_path   = h5_doe,
        ind_configs   = active,
        nb_workers    = workers,
        enabled_cases = ENABLED_CASES,
        dry_run       = args.dry_run,
        label_key     = LABEL_KEY,
        out_path      = out_path if not args.dry_run else None,
    )

    if not args.dry_run:
        log.info("Listo. %d resultados escritos incrementalmente en %s", len(results), out_path)
    else:
        log.info("[DRY-RUN] %d tareas planificadas — nada escrito.", len(results))


if __name__ == "__main__":
    main()

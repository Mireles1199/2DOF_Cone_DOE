#!/usr/bin/env python
# coding: utf-8
"""
DOE Runner for Nessy2m
======================
Automatiza la ejecucion de un DOE con el simulador Nessy2m.

Uso:
    python doe_runner.py --case 1DOF_150Hz
    python doe_runner.py --case 1DOF_150Hz --dry-run
    python doe_runner.py --case 1DOF_150Hz --n2m_bat <ruta\n2m.bat>

El script:
  1. Genera val_var segun DOE_MODE (factorial / sweep / manual).
  2. Sobreescribe p/param.py del caso con los valores generados.
  3. Llama a n2m_sch.py (equivalente al comando n2m_sch).
  4. Restaura p/param.py original al terminar.
"""

import os
import sys
import itertools
import subprocess
import argparse
import logging
import re
import shutil
import glob
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import h5py

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 0 - Dexels - Cinematique
# SCRIPT_DIR = os.path.abspath(os.path.join(
#     r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage",
#     "Chatter-Criteria",
#     "CAMP10_Chatter_detection_Methodes",
#     "Convergency_Simulation",
#     "0_Cinematique",
# ))

# Step 1 - Detection Limite Lobes
# SCRIPT_DIR = os.path.abspath(os.path.join(
#     r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage",
#     "Chatter-Criteria",
#     "CAMP10_Chatter_detection_Methodes",
#     "Convergency_Simulation",
#     "1_Detection_Limite_Lobes",
# ))

# Step 2 - Senitivity Dexels
# SCRIPT_DIR = os.path.abspath(os.path.join(
#     r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage",
#     "Chatter-Criteria",
#     "CAMP10_Chatter_detection_Methodes",
#     "Convergency_Simulation",
#     "2_Sensitivity_Dexels",
# ))

# Step 3 - Senitivity dt
# SCRIPT_DIR = os.path.abspath(os.path.join(
#     r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage",
#     "Chatter-Criteria",
#     "CAMP10_Chatter_detection_Methodes",
#     "Convergency_Simulation",
#     "3_Sensitivity_dt",
# ))


# Training - Tube
SCRIPT_DIR = os.path.abspath(os.path.join(
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage",
    "Chatter-Criteria",
    "CAMP10_Chatter_detection_Methodes",
    "Convergency_Simulation",
    "4_DOE_Data_Training_Tube",
))





print("Script directory:", SCRIPT_DIR)

DEFAULT_N2M_BAT = os.path.join(
    r"C:\Users\quiqu\OneDrive-ensam.eu\Desktop\Thesis\03-Code\01-Nessy2m"
    r"\VP2025.1.0\VP2025.1.0\nessy2m",
    "n2m.bat",
)

# ==============================================================================
# CONFIGURACION DEL DOE
# ==============================================================================

# DOE_NAME = "DOE_Dexels_Cinematique_path"   # nombre de la carpeta de salida  (dir_ref2exe)
# DOE_NAME = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"   # nombre de la carpeta de salida  (dir_ref2exe)
# DOE_NAME = "DOE_Influence_dt_RPM_12000_f_005_dexel_005"   # nombre de la carpeta de salida  (dir_ref2exe)
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_200"   # nombre de la carpeta de salida  (dir_ref2exe)
# DOE_NAME = "DOE_Sensitivity_Dexels_factor_4_sup"   # nombre de la carpeta de salida  (dir_ref2exe)

# 2  Campana
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_1_patch_0.96-0.97"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_1_patch_0.985"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_10e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_10e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_10e-5_RUN_1_patch_0.97-0.99"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_10e-5_RUN_1_patch_0.985"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_5e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_5e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_5e-5_RUN_1_patch_0.97-0.99"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_5e-5_RUN_1_patch_0.985"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_2.5e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_2.5e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_2.5e-5_RUN_1"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_2.5e-5_RUN_1_patch_0.985"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_1"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_1_patch_0.985"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_40e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_40e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_40e-5_RUN_1_patch_0.985"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_80e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_80e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_80e-5_RUN_1_patch_0.985"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_160e-5_RUN_10_patch_0.95"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_160e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_160e-5_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_320e-5_RUN_10_patch_1.55"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_320e-5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dxl_320e-5_RUN_1"


# ============ Sensibility DT ===================

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_3200_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_3200_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_3200_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_1600_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_1600_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_1600_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_800_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_800_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_800_RUN_1"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_400_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_400_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_400_RUN_1"


# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_200_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_200_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_200_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_100_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_100_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_100_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_50_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_50_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_50_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_25_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_25_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_25_RUN_1"

# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_12.5_RUN_10"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_12.5_RUN_5"
# DOE_NAME = "DOE_Detection_Limite_Lobes_dt_12.5_RUN_1"



# ============== Training Tube ============
DOE_NAME = "DOE_Training_Tube_dxl_20e-5_RUN_10_0.91-1.09"







NB_PROC  = 4     # numero de procesos paralelos

# ------------------------------------------------------------------------------
# MODO 1 - FACTORIAL COMPLETO (producto cartesiano)
#   N casos = producto de len(valores) de cada variable.
#   Variables con 1 valor quedan fijas.
# ------------------------------------------------------------------------------
DOE_FACTORIAL = {
    "$Ap_start$"  : [15e-3],
    "$Ap_end$"    : [15e-3],
    "$spin_rate$" : [12099.28],

    # ------------ Influence de dt (nb_dt_rev) --------------------------------
    # "$f_tooth$"   : [0.05],
    # "$dxl_size$"  : [0.05e-3,],
    # "$nb_dt_rev$" : [20, 30, 45, 60, 90, 120, 180, 360, 720, 1440],

    #  ------------ Influence de f_tooth --------------------------------
    # "$f_tooth$"   : [0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.08, 0.12, 0.15, 0.2],
    # "$nb_dt_rev$" : [180,],
    # "$dxl_size$"  : [0.05e-3,],

    # # ------------ Influence de dexel_size --------------------------------
    "$f_tooth$"   : [ 0.05,],
    "$nb_dt_rev$" : [200,],
    # RUN 10
    "$dxl_size$"  : [4.302600e-03, 4.732860e-03, 5.163120e-03, 6.023640e-03, 6.884160e-03, 7.744680e-03, 8.605200e-03],

    # # RUN 5
    # "$dxl_size$"  : [],

    # # RUN 1
    # "$dxl_size$"  : [],






}

# ------------------------------------------------------------------------------
# MODO 2 - BARRIDO PAREADO (zip, posicion a posicion)
#   Todas las listas deben tener el mismo numero de elementos.
# ------------------------------------------------------------------------------
Ap_tube = [
        # 4.30E-03, #0.5
        # 5.16E-03, #0.6
        # 6.02E-03, #0.7
        # 6.88E-03, #0.8
        # 7.74E-03, #0.9
        # 8.61E-03, #1.0
        # 9.47E-03, #1.1
        # 1.03E-02, #1.2
        # 1.12E-02, #1.3
        # 1.20E-02, #1.4
        # 1.29E-02, #1.5
        # 1.38E-02, #1.6
        # 1.46E-02, #1.7
        # 1.55E-02, #1.8
        # 1.63E-02, #1.9
        # 1.72E-02  #2.0

        7.83E-03, #0.91
        7.92E-03, #0.92
        8.00E-03, #0.93
        8.09E-03, #0.94
        8.17E-03, #0.95
        8.26E-03, #0.96
        8.35E-03, #0.97
        8.43E-03, #0.98
        8.52E-03, #0.99

        8.69E-03, #1.01
        8.78E-03, #1.02
        8.86E-03, #1.03
        8.95E-03, #1.04
        9.04E-03, #1.05
        9.12E-03, #1.06
        9.21E-03, #1.07
        9.29E-03, #1.08
        9.38E-03 #1.09



]
# Ap_tube = [ 8.48E-03 ]  # 0.985
spin_rate_sweep = 12098.28
f_tooth_sweep = 0.05
dxl_size_sweep = 20e-5
nb_dt_rev_sweep = 200

DOE_SWEEP = {

    # Training Tube 0.5-2
    "$Ap_start$"  : Ap_tube,
    "$Ap_end$"    : Ap_tube,






    "$spin_rate$" : np.linspace(spin_rate_sweep, spin_rate_sweep, len(Ap_tube)).tolist(),

    "$f_tooth$"   : np.linspace(f_tooth_sweep, f_tooth_sweep, len(Ap_tube)).tolist(),

    "$dxl_size$"  : np.linspace(dxl_size_sweep, dxl_size_sweep, len(Ap_tube)).tolist(),

    "$nb_dt_rev$" : np.linspace(nb_dt_rev_sweep, nb_dt_rev_sweep, len(Ap_tube)).tolist(),

}

# ------------------------------------------------------------------------------
# MODO 3 - MANUAL (formato original de n2m_sch)
# ------------------------------------------------------------------------------
DOE_MANUAL_LST = ["$Ap_start$", "$Ap_end$", "$spin_rate$", "$f_tooth$", "$dxl_size$", "$nb_dt_rev$"]
DOE_MANUAL_VAL = [
    [5e-3, 15e-3, 12000.0, 0.05, 0.1e-3, 100],
]

# ► SELECCIONA EL MODO:  "factorial"  |  "sweep"  |  "manual"
DOE_MODE = "sweep"

# ==============================================================================
# LIMPIEZA POST-SIMULACION
#   Directorios a borrar dentro de cada copia de caso tras la simulacion.
#   Dejar [] para no borrar nada.
#   Opciones: 'db', 'out', 'p', 's', 'tmp', 'tool', 'wp'
# ==============================================================================
POST_CLEANUP_DIRS = ['db', 'out', 'p', 's', 'tmp', 'tool', 'wp']

# ==============================================================================
# EXTRACCION DE RESULTADOS
#   Señales a extraer del sens_out.hdf5 de cada caso.
#   Columna 0 = tiempo, columna 1 = valores.
# ==============================================================================
DOE_EXTRACT_SIGNALS = ["Axial_disp", "Axial_vel", "Axial_acc"]
DOE_FORCE_SIGNAL = "res_R_p"


def build_doe_cases(mode: str) -> tuple[list, list]:
    """Devuelve (lst_var, val_var) segun DOE_MODE."""
    if mode == "factorial":
        lst_var = list(DOE_FACTORIAL.keys())
        val_var = [list(c) for c in itertools.product(*DOE_FACTORIAL.values())]
    elif mode == "sweep":
        lst_var = list(DOE_SWEEP.keys())
        val_var = [list(r) for r in zip(*DOE_SWEEP.values())]
    else:  # manual
        lst_var = DOE_MANUAL_LST
        val_var = DOE_MANUAL_VAL
    return lst_var, val_var


def write_post(post_path: str, cleanup_dirs: list) -> None:
    """Agrega al final de post.py un bloque que borra cleanup_dirs tras la sim."""
    block = (
        "\n"
        "# --- Limpieza POST-DOE (doe_runner.py) ---\n"
        "import shutil as _sh\n"
        "import os as _os\n"
        "_cdir = _os.path.abspath(_os.curdir)\n"
        + f"_cdirs = {cleanup_dirs!r}\n"
        + "for _d in _cdirs:\n"
        "    _p = _os.path.join(_cdir, _d)\n"
        "    if _os.path.isdir(_p):\n"
        "        _sh.rmtree(_p)\n"
        "        print(f'[CLEANUP] Borrado: {_p}')\n"
    )
    with open(post_path, "a", encoding="utf-8") as fh:
        fh.write(block)


def write_param(param_path: str, lst_var: list, val_var: list, doe_name: str, nb_proc: int) -> None:
    """Sobreescribe p/param.py con los valores generados."""
    lines = [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "# AUTO-GENERADO por doe_runner.py — no editar manualmente\n",
        "import os\n",
        "param = {\n",
        f"    'nb_proc'     : {nb_proc},\n",
        f"    'dir_ref2exe' : os.path.join('..', '{doe_name}'),\n",
        f"    'lst_var'     : {lst_var!r},\n",
        f"    'val_var'     : {val_var!r},\n",
        "}\n",
    ]
    with open(param_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def write_param_abs(param_path: str, lst_var: list, val_var: list,
                    abs_doe_dir: str, nb_proc: int) -> None:
    """Como write_param pero con dir_ref2exe como ruta absoluta (modo --timed)."""
    lines = [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "# AUTO-GENERADO por doe_runner.py — no editar manualmente\n",
        "import os\n",
        "param = {\n",
        f"    'nb_proc'     : {nb_proc},\n",
        f"    'dir_ref2exe' : {abs_doe_dir!r},\n",
        f"    'lst_var'     : {lst_var!r},\n",
        f"    'val_var'     : {val_var!r},\n",
        "}\n",
    ]
    with open(param_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def load_n2m_env(bat_path: str) -> dict:
    bat_path = os.path.abspath(bat_path)
    if not os.path.isfile(bat_path):
        raise FileNotFoundError(f"No se encontro n2m.bat en: {bat_path}")
    bat_dir = os.path.dirname(bat_path)
    log.info("Cargando entorno Nessy2m desde: %s", bat_path)
    env = dict(os.environ)
    env["cd"] = bat_dir
    env["CD"] = bat_dir
    set_re = re.compile(r"^\s*SET\s+([^=]+)=(.*)$", re.IGNORECASE)

    def expand(val, ctx):
        def _repl(m):
            name = m.group(1)
            for k, v in ctx.items():
                if k.upper() == name.upper():
                    return v
            return m.group(0)
        return re.sub(r"%([^%]+)%", _repl, val)

    with open(bat_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = set_re.match(line)
            if m:
                key = m.group(1).strip()
                val = expand(m.group(2).strip(), env)
                env[key] = val

    if "n2m" not in env:
        log.warning("Variable 'n2m' no encontrada tras parsear el .bat.")
    else:
        log.info("Variable n2m = %s", env["n2m"])
    return env


def get_python_exe(env: dict) -> str:
    for d in env.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(d, "python.exe")
        if os.path.isfile(candidate):
            log.info("Usando Python de Nessy2m: %s", candidate)
            return candidate
    log.warning("Usando Python actual: %s", sys.executable)
    return sys.executable


# ==============================================================================
# FUNCIONES DE EXTRACCION
# ==============================================================================

def _hdf5_find_dataset(group: h5py.Group, name: str):
    """Busqueda recursiva de un Dataset por nombre dentro de un h5py.Group.

    Acepta dos estructuras:
      - Dataset directo:   .../name          (retorna ese dataset)
      - Grupo con data:    .../name/data      (retorna el dataset 'data' dentro)
    """
    for key in group:
        item = group[key]
        if key == name:
            if isinstance(item, h5py.Dataset):
                return item
            if isinstance(item, h5py.Group) and "data" in item:
                return item["data"]
        if isinstance(item, h5py.Group):
            result = _hdf5_find_dataset(item, name)
            if result is not None:
                return result
    return None


def _extract_series_from_dataset(dataset: h5py.Dataset):
    """Extrae tiempo y valores de un dataset 2D donde la primera columna es tiempo."""
    arr = dataset[()]
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Dataset incompatible para serie tiempo/valor: shape={getattr(arr, 'shape', None)}")

    time = arr[:, 0]
    values = arr[:, 1:]
    return time, values


def read_var_val(path: str) -> dict:
    """Lee var_val.py y retorna el dict var_val. Retorna {} si no existe."""
    if not os.path.isfile(path):
        log.warning("var_val.py no encontrado: %s", path)
        return {}
    ns: dict = {"__builtins__": {}}
    with open(path, "r", encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)  # noqa: S102
    return ns.get("var_val", {})


def extract_doe_results(doe_dir: str, case_name: str, signals: list, dry_run: bool = False) -> list:
    """Itera los casos del DOE, extrae señales de sens_out.hdf5 y out.hdf5, y guarda doe_results.h5."""
    pattern = os.path.join(doe_dir, "*", case_name)
    case_dirs = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.basename(os.path.dirname(p)))
    )

    if not case_dirs:
        log.warning("No se encontraron casos con patron: %s", pattern)
        return []

    log.info("Casos encontrados: %d | DOE dir: %s", len(case_dirs), doe_dir)

    if dry_run:
        for cd in case_dirs:
            idx = os.path.basename(os.path.dirname(cd))
            log.info("[DRY-RUN] Caso %s → %s", idx, cd)
        return []

    results = []
    out_path = os.path.join(doe_dir, "doe_results.h5")

    # Eliminar el archivo previo para evitar bloqueo de Windows (errno=0 / GetLastError=33)
    if os.path.isfile(out_path):
        try:
            os.remove(out_path)
        except OSError as e:
            log.error("No se puede sobreescribir %s — ciérralo en otros programas y vuelve a intentar. (%s)", out_path, e)
            return []

    with h5py.File(out_path, "w") as out_f:
        for case_path in case_dirs:
            idx = int(os.path.basename(os.path.dirname(case_path)))
            group_name = f"case_{idx:03d}"

            # -- var_val.py --
            var_val = read_var_val(os.path.join(case_path, "var_val.py"))
            grp = out_f.create_group(group_name)
            for k, v in var_val.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)
            # -- wall_time_s (guardado por --timed dentro de la carpeta del caso) --
            wt_file = os.path.join(os.path.dirname(case_path), "wall_time_s.txt")
            if os.path.isfile(wt_file):
                with open(wt_file, "r", encoding="utf-8") as _f:
                    grp.attrs["wall_time_s"] = float(_f.read().strip())

            # -- sens_out.hdf5 --
            hdf5_path = os.path.join(case_path, "sens_out.hdf5")
            case_result = {"idx": idx, "var_val": var_val}
            if os.path.isfile(hdf5_path):
                with h5py.File(hdf5_path, "r") as src_f:
                    for sig in signals:
                        ds = _hdf5_find_dataset(src_f, sig)
                        if ds is None:
                            log.warning("Senal '%s' no encontrada en %s", sig, hdf5_path)
                            continue
                        arr = ds[()]
                        t = arr[:, 0]
                        y = arr[:, 1]
                        sig_grp = grp.create_group(sig)
                        sig_grp.create_dataset("time",   data=t, compression="gzip")
                        sig_grp.create_dataset("values", data=y, compression="gzip")
                        case_result[sig] = (t, y)
            else:
                log.warning("sens_out.hdf5 no encontrado en: %s", case_path)

            # -- out.hdf5: fuerza res_R_P --
            out_hdf5_path = os.path.join(case_path, "out.hdf5")
            if os.path.isfile(out_hdf5_path):
                with h5py.File(out_hdf5_path, "r") as src_f:
                    ds = _hdf5_find_dataset(src_f, DOE_FORCE_SIGNAL)
                    if ds is None:
                        log.warning("Senal '%s' no encontrada en %s", DOE_FORCE_SIGNAL, out_hdf5_path)
                    else:
                        t_force, f_force = _extract_series_from_dataset(ds)
                        force_grp = grp.create_group(DOE_FORCE_SIGNAL)
                        force_grp.create_dataset("time", data=t_force, compression="gzip")
                        force_grp.create_dataset("values", data=f_force, compression="gzip")
                        case_result[DOE_FORCE_SIGNAL] = (t_force, f_force)
            else:
                log.warning("out.hdf5 no encontrado en: %s", case_path)

            results.append(case_result)
            log.info("Caso %03d extraido | var_val=%s", idx, var_val)

    log.info("doe_results.h5 guardado en: %s", out_path)
    return results


def merge_doe_results(base_doe_dir: str, patch_doe_dir: str, dry_run: bool = False) -> None:
    """Fusiona dos doe_results.h5 — añade los casos de patch al base sin duplicar.

    - Los grupos del patch se renumeran continuando desde el mayor índice del base.
    - Si un grupo con los mismos attrs ya existe en el base, se omite con warning.
    - El base no se toca si dry_run=True.
    """
    base_h5   = os.path.join(base_doe_dir,  "doe_results.h5")
    patch_h5  = os.path.join(patch_doe_dir, "doe_results.h5")

    if not os.path.isfile(base_h5):
        log.error("doe_results.h5 base no encontrado: %s", base_h5)
        return
    if not os.path.isfile(patch_h5):
        log.error("doe_results.h5 patch no encontrado: %s", patch_h5)
        return

    with h5py.File(base_h5, "r" if dry_run else "a") as base_f:
        # Indice máximo actual en el base
        existing_indices = []
        for k in base_f.keys():
            try:
                existing_indices.append(int(k.split("_")[-1]))
            except ValueError:
                pass
        next_idx = max(existing_indices) + 1 if existing_indices else 0

        # Attrs de grupos existentes para detectar duplicados
        existing_attrs = [
            dict(base_f[k].attrs) for k in base_f.keys()
        ]

        with h5py.File(patch_h5, "r") as patch_f:
            added = 0
            skipped = 0
            for grp_name in sorted(patch_f.keys()):
                patch_grp  = patch_f[grp_name]
                patch_attrs = dict(patch_grp.attrs)

                # Comprobar si ya existe un caso con los mismos attrs
                if any(patch_attrs == ea for ea in existing_attrs):
                    log.warning("Omitido (ya existe): %s  attrs=%s", grp_name, patch_attrs)
                    skipped += 1
                    continue

                new_name = f"case_{next_idx:03d}"
                # Carpeta física del caso en el patch: patch_doe_dir/<patch_idx>/
                patch_idx    = int(grp_name.split("_")[-1])
                src_case_dir = os.path.join(patch_doe_dir, str(patch_idx))
                dst_case_dir = os.path.join(base_doe_dir,  str(next_idx))

                if dry_run:
                    log.info("[DRY-RUN] Añadiría: %s -> %s  attrs=%s",
                             grp_name, new_name, patch_attrs)
                    if os.path.isdir(src_case_dir):
                        log.info("[DRY-RUN] Copiaría carpeta: %s -> %s",
                                 src_case_dir, dst_case_dir)
                    else:
                        log.warning("[DRY-RUN] Carpeta física no encontrada: %s", src_case_dir)
                else:
                    patch_f.copy(patch_grp, base_f, name=new_name)
                    log.info("Añadido (HDF5): %s -> %s  attrs=%s",
                             grp_name, new_name, patch_attrs)
                    existing_attrs.append(patch_attrs)
                    if os.path.isdir(src_case_dir):
                        shutil.copytree(src_case_dir, dst_case_dir)
                        log.info("Copiada carpeta: %s -> %s", src_case_dir, dst_case_dir)
                    else:
                        log.warning("Carpeta física no encontrada (solo HDF5): %s", src_case_dir)

                next_idx += 1
                added += 1

    log.info("Merge completado: %d añadidos, %d omitidos (duplicados).", added, skipped)
    if not dry_run:
        log.info("base doe_results.h5 actualizado: %s", base_h5)


def run_n2m_command(command_script, case_dir, env, python_exe, extra_args=None, dry_run=False):
    cmd = [python_exe, command_script] + (extra_args or [])
    log.info("Directorio de trabajo : %s", case_dir)
    log.info("Comando               : %s", " ".join(cmd))
    if dry_run:
        log.info("[DRY-RUN] El comando no fue ejecutado.")
        return 0
    return subprocess.run(cmd, cwd=case_dir, env=env).returncode


def _run_one_case_timed(
    idx: int,
    lst_var: list,
    case_vals: list,
    doe_name: str,
    src_case_dir: str,
    command_script: str,
    env: dict,
    python_exe: str,
    script_dir: str,
    dry_run: bool,
) -> tuple:
    """Corre un único caso directamente en la carpeta final del DOE.
    n2m_sch exige que dir_ref2exe no exista → usamos un subdir staging que se renombra al final.
    Devuelve (idx, wall_time_s, rc).
    """
    final_doe_dir = os.path.join(script_dir, doe_name)
    os.makedirs(final_doe_dir, exist_ok=True)
    case_name    = os.path.basename(src_case_dir)         # ej. "1DOF_150Hz"

    # Directorio de trabajo del caso (copia del caso base) dentro del DOE final
    work_case_root = os.path.join(final_doe_dir, f"_work_{idx:03d}")
    work_case_dir  = os.path.join(work_case_root, case_name)
    # dir_ref2exe: n2m_sch exige que NO exista aún → staging dentro del DOE final
    staging_dir    = os.path.join(final_doe_dir, f"_staging_{idx:03d}")
    try:
        shutil.copytree(src_case_dir, work_case_dir)       # copia con nombre original
        param_path = os.path.join(work_case_dir, "p", "param.py")
        write_param_abs(param_path, lst_var, [case_vals], staging_dir, 1)
        log.info("Caso %03d INICIADO  | vars=%s", idx, case_vals)
        t0 = time.perf_counter()
        rc = 0
        if not dry_run:
            rc = subprocess.run(
                [python_exe, command_script],
                cwd=work_case_dir, env=env,
            ).returncode
        wall_time = time.perf_counter() - t0
        # Renombrar staging/0/ → final_doe_dir/idx/
        src_idx = os.path.join(staging_dir, "0")
        dst_idx = os.path.join(final_doe_dir, str(idx))
        if not dry_run and os.path.isdir(src_idx):
            if os.path.exists(dst_idx):
                shutil.rmtree(dst_idx)
            shutil.move(src_idx, dst_idx)
            # Guardar tiempo dentro de la carpeta del caso
            wt_file = os.path.join(dst_idx, "wall_time_s.txt")
            with open(wt_file, "w", encoding="utf-8") as _f:
                _f.write(str(wall_time))
    finally:
        shutil.rmtree(work_case_root, ignore_errors=True)
        shutil.rmtree(staging_dir,    ignore_errors=True)
    return idx, wall_time, rc


def run_doe_timed(
    lst_var: list,
    val_var: list,
    doe_name: str,
    src_case_dir: str,
    command_script: str,
    env: dict,
    python_exe: str,
    script_dir: str,
    nb_workers: int,
    dry_run: bool,
) -> None:
    """Ejecuta el DOE caso a caso con ThreadPoolExecutor (NB_PROC workers en paralelo).
    Mide el tiempo de cada caso; lo guarda en caso_dir/wall_time_s.txt (leído por extract).
    """
    n = len(val_var)
    log.info("DOE timed: %d casos, %d workers en paralelo", n, nb_workers)
    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        futures = {
            executor.submit(
                _run_one_case_timed,
                idx, lst_var, case_vals, doe_name,
                src_case_dir, command_script, env, python_exe,
                script_dir, dry_run,
            ): idx
            for idx, case_vals in enumerate(val_var)
        }
        for future in as_completed(futures):
            idx, wall_time, rc = future.result()
            status = "DRY-RUN" if dry_run else ("OK" if rc == 0 else f"ERROR rc={rc}")
            log.info("Caso %03d | %.1f s | %s | vars=%s",
                     idx, wall_time, status, val_var[idx])
    log.info("Total casos completados: %d", len(val_var))


def parse_args():
    epilog = """
DOE Runner — Automatización de simulaciones Nessy2m
====================================================
Genera val_var, sobreescribe param.py y ejecuta n2m_sch (o extrae resultados).

COMANDOS  (--command)
---------------------
  n2m_sch    Ejecuta el DOE completo (simulación Nessy2m). [por defecto]
  extract    Lee sens_out.hdf5 de cada caso y guarda doe_results.h5.
             No requiere Nessy2m; corre con el entorno CAMP10.
  nessy2m    Lanza nessy2m directamente.
  pp_creat   Ejecuta pp_creat (post-proceso inicial).
  pp_init    Ejecuta pp_init.

EJEMPLOS
--------
  python doe_runner.py --case 1DOF_150Hz
      Ejecuta el DOE completo para el caso 1DOF_150Hz.

  python doe_runner.py --case 1DOF_150Hz --dry-run
      Simula sin ejecutar (muestra rutas y comandos).

  python doe_runner.py --case 1DOF_150Hz --command extract
      Extrae señales de todos los casos -> guarda doe_results.h5.

  python doe_runner.py --case 1DOF_150Hz --command extract --doe_name DOE_Influence_dt
      Extrae de la carpeta DOE_Influence_dt/ (sobreescribe DOE_NAME del script).

  python doe_runner.py --case 1DOF_150Hz --timed --auto-extract
      Corre el DOE completo (modo timed) y al terminar corre --command extract
      solo, sin tener que lanzarlo aparte despues.

  python doe_runner.py --command merge --doe_name DOE_base --merge_from DOE_patch
      Fusiona DOE_patch/doe_results.h5 en DOE_base/doe_results.h5.
      Los casos nuevos se renumeran continuando desde el último índice del base.
      Los casos con attrs idénticos se omiten automáticamente (no se duplican).

  python doe_runner.py --command merge --doe_name DOE_base --merge_from DOE_patch --merge_out DOE_merged
      Igual que el anterior, pero copia DOE_base -> DOE_merged primero y fusiona
      ahi. DOE_base y DOE_patch quedan intactos.

  python doe_runner.py --case 1DOF_150Hz --timed
      Corre cada caso en directorio aislado, NB_PROC casos en paralelo (runner-side).
      Guarda el tiempo por caso en cada carpeta de caso (wall_time_s.txt).
      Tras extract, doe_plotter muestra figura de tiempo de cómputo vs dt.

  python doe_runner.py --case 1DOF_150Hz --timed --dry-run
      Simula el modo timed sin ejecutar Nessy2m (verifica rutas y configuración).

FLUJO TÍPICO CON TIMING
-----------------------
  1. Editar NB_PROC en CONFIG (ej. NB_PROC = 4 para 4 casos en paralelo)
  2. python doe_runner.py --case <caso> --timed          # simular + medir tiempos
  3. python doe_runner.py --case <caso> --command extract  # extraer (lee wall_time_s.txt)
  4. python doe_plotter.py  --doe_name <DOE>             # ver figura de tiempo

FLUJO PARA AÑADIR CASOS A UN DOE EXISTENTE
--------------------------------------------
  1. Cambiar DOE_NAME = "DOE_base_patch"  (nombre nuevo, para no chocar con n2m_sch)
  2. Poner en DOE_FACTORIAL solo los N casos nuevos
  3. python doe_runner.py --case <caso>                              # simular patch
  4. python doe_runner.py --case <caso> --command extract            # extraer patch
  5. python doe_runner.py --command merge \
         --doe_name DOE_base --merge_from DOE_base_patch             # fusionar
  6. python doe_plotter.py  --doe_name DOE_base                      # visualizar todo

FLUJO TÍPICO
------------
  1. Editar sección CONFIG del script (DOE_NAME, DOE_FACTORIAL, DOE_MODE...)
  2. python doe_runner.py --case <caso>                        # simular
  3. python doe_runner.py --case <caso> --command extract      # extraer
  4. python doe_plotter.py  --doe_name <DOE>                   # visualizar
  5. python doe_selector.py --doe_name <DOE>                   # explorar
"""
    parser = argparse.ArgumentParser(
        description="DOE Runner — Nessy2m",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", default="1DOF_150Hz", help="Nombre del subdirectorio del caso")
    parser.add_argument("--case_dir", default=None, help="Ruta completa al caso (opcional)")
    parser.add_argument("--n2m_bat", default=DEFAULT_N2M_BAT, help="Ruta al n2m.bat")
    parser.add_argument("--command", default="n2m_sch",
                        choices=["n2m_sch", "nessy2m", "pp_creat", "pp_init", "extract", "merge"])
    parser.add_argument("--doe_name", default=None,
                        help="DOE base (extract/merge). Sobreescribe DOE_NAME del config")
    parser.add_argument("--merge_from", default=None,
                        help="(merge) Nombre del DOE patch cuyos casos se añaden al base")
    parser.add_argument("--merge_out", default=None,
                        help="(merge) Si se da, copia --doe_name a esta carpeta nueva y fusiona ahi "
                             "en vez de modificar el DOE base original")
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero de workers para --timed (por defecto usa NB_PROC)")
    parser.add_argument("--timed", action="store_true",
                        help="Corre cada caso individualmente, mide tiempo (wall_time_s.txt por caso -> attr HDF5 tras extract)")
    parser.add_argument("--auto-extract", action="store_true",
                        help="Corre --command extract automaticamente al terminar la corrida")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolver directorio del caso
    case_dir = os.path.abspath(args.case_dir) if args.case_dir else os.path.join(SCRIPT_DIR, args.case)
    case_dir = os.path.normpath(case_dir)   # elimina trailing slash para que dirname() funcione bien
    if not os.path.isdir(case_dir):
        log.error("Directorio del caso no encontrado: %s", case_dir)
        sys.exit(1)
    log.info("Caso DOE: %s", case_dir)

    # --- Rama extract: no necesita param.py ni n2m.bat ---
    if args.command == "extract":
        doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
        doe_dir = os.path.join(os.path.dirname(case_dir), doe_name_eff)
        log.info("Extrayendo DOE: %s", doe_dir)
        extract_doe_results(doe_dir, os.path.basename(case_dir),
                            DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)
        return

    # --- Rama merge: fusiona dos doe_results.h5 ---
    if args.command == "merge":
        doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
        if not args.merge_from:
            log.error("--merge_from es obligatorio con --command merge")
            sys.exit(1)
        base_dir  = os.path.join(os.path.dirname(case_dir), doe_name_eff)
        patch_dir = os.path.join(os.path.dirname(case_dir), args.merge_from)

        merge_target = base_dir
        if args.merge_out:
            out_dir = os.path.join(os.path.dirname(case_dir), args.merge_out)
            if args.dry_run:
                log.info("[DRY-RUN] Copiaria %s -> %s (el resto del preview se muestra sobre el original)",
                         base_dir, out_dir)
            else:
                if os.path.isdir(out_dir):
                    log.error("La carpeta destino ya existe: %s", out_dir)
                    sys.exit(1)
                shutil.copytree(base_dir, out_dir)
                log.info("Copiado %s -> %s", base_dir, out_dir)
                merge_target = out_dir

        log.info("Merge: %s  <--  %s", merge_target, patch_dir)
        merge_doe_results(merge_target, patch_dir, dry_run=args.dry_run)
        return

    # 2. Generar casos DOE
    lst_var, val_var = build_doe_cases(DOE_MODE)
    log.info("DOE modo=%s | casos=%d | salida=%s", DOE_MODE, len(val_var), DOE_NAME)

    # Confirmar si el DOE ya existe: preguntar si desea reemplazar
    doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
    doe_dir = os.path.join(os.path.dirname(case_dir), doe_name_eff)
    if os.path.exists(doe_dir):
        if args.dry_run:
            log.info("DOE existente detectado (dry-run): %s — se ignorará reemplazo.", doe_dir)
        else:
            resp = input(
                f"DOE directory '{doe_dir}' already exists.\n"
                f"Case directory: '{case_dir}'\n"
                "Replace the DOE directory (this will overwrite its contents)? [y/N]: "
            ).strip().lower()
            if resp not in ("y", "yes"):
                log.info("Usuario canceló la operación. No se reemplazará: %s", doe_dir)
                return

    # 3. Cargar entorno Nessy2m (necesario para ambos modos)
    env = load_n2m_env(args.n2m_bat)
    python_exe = get_python_exe(env)
    n2m_bin = env.get("n2m", "")
    command_script = os.path.join(n2m_bin, f"{args.command}.py")
    if not os.path.isfile(command_script):
        log.error("Script no encontrado: %s", command_script)
        sys.exit(1)

    # 4a. Modo --timed: paralelismo en runner, timing por caso
    if args.timed:
        nb_workers = args.workers if args.workers is not None else NB_PROC
        log.info("Workers configurados: %d", nb_workers)
        run_doe_timed(lst_var, val_var, DOE_NAME, case_dir,
                      command_script, env, python_exe,
                      SCRIPT_DIR, nb_workers, args.dry_run)
        if args.auto_extract:
            extract_doe_results(doe_dir, os.path.basename(case_dir),
                                DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)
        return

    # 4b. Modo normal: sobreescribir param.py + un solo n2m_sch
    param_path = os.path.join(case_dir, "p", "param.py")
    param_backup = param_path + ".bak"
    shutil.copy2(param_path, param_backup)
    log.info("Backup de param.py guardado en: %s", param_backup)

    try:
        write_param(param_path, lst_var, val_var, DOE_NAME, NB_PROC)
        log.info("param.py actualizado con %d casos.", len(val_var))

        # 5. Ejecutar n2m_sch
        rc = run_n2m_command(command_script, case_dir, env, python_exe,
                             extra_args=args.args, dry_run=args.dry_run)
    finally:
        # 6. Restaurar param.py original siempre (aunque haya error)
        shutil.copy2(param_backup, param_path)
        os.remove(param_backup)
        log.info("param.py original restaurado.")

    if rc != 0:
        log.error("Comando termino con error: %d", rc)
        sys.exit(rc)
    log.info("Completado exitosamente.")

    if args.auto_extract:
        extract_doe_results(doe_dir, os.path.basename(case_dir),
                            DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

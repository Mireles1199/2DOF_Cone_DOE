"""doe_model_snr.py — Calcula el SNR de degradación de modelo entre el caso
control de un DOE y todos los demás casos degradados.

Uso:
    python doe_model_snr.py
    python doe_model_snr.py --doe_name DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180
    python doe_model_snr.py --control_idx 1
    python doe_model_snr.py --list
    python doe_model_snr.py --dry_run
    python doe_model_snr.py --out resultados_snr.h5

Fórmula SNR de modelo (señales normalizadas por RMS):
    x̂_ctrl = I_control  / RMS(I_control)     (normalizar amplitud)
    x̂_deg  = I_degradado / RMS(I_degradado)  (normalizar amplitud)
    e       = x̂_ctrl - x̂_deg                (error de forma/patrón)
    SNR_mod_dB = 10 * log10(1 / mean(e²))   (≡ 10*log10(P_ref / P_e) con P_ref=1)

    SNR alto (ej. +20 dB) → el degradado tiene la misma forma que el control
    SNR bajo (ej.  +5 dB) → el degradado difiere en forma/patrón
    Siempre positivo si las señales no son idénticas en forma
    Comparable directamente con el SNR del DOE de ruido (doe_noise)

Configuración (editar el bloque CONFIG al inicio del archivo):
    DOE_NAME     : nombre de la carpeta DOE (sobreescribible con --doe_name)
    CASE_NAME    : subcarpeta de simulación dentro de cada índice numérico
    CONTROL_IDX  : índice numérico del caso control (sobreescribible con --control_idx)
    BASE_DIR     : directorio raíz que contiene la carpeta DOE
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

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
# CONFIG GLOBAL — editar aquí
# ==============================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

BASE_DIR    = SCRIPT_DIR   # directorio raíz donde vive la carpeta DOE

DOE_NAME    = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
CASE_NAME   = "1DOF_150Hz"
CONTROL_IDX = 3             # índice numérico del caso que es el control

# ==============================================================================
# UTILIDADES (copiadas de doe_runner.py)
# ==============================================================================

def _hdf5_find_dataset(group: h5py.Group, name: str):
    """Búsqueda recursiva de un Dataset por nombre dentro de un h5py.Group.

    Acepta dos estructuras:
      - Dataset directo:   .../name
      - Grupo con data:    .../name/data
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


def read_var_val(path: str) -> dict:
    """Lee var_val.py y retorna el dict var_val. Retorna {} si no existe."""
    if not os.path.isfile(path):
        log.warning("var_val.py no encontrado: %s", path)
        return {}
    ns: dict = {"__builtins__": {}}
    with open(path, "r", encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)  # noqa: S102
    return ns.get("var_val", {})


# ==============================================================================
# DESCUBRIMIENTO DE CASOS
# ==============================================================================

def discover_cases(doe_dir: str) -> list[tuple[int, str]]:
    """Retorna lista de (idx, case_dir) ordenada numéricamente."""
    pattern = os.path.join(doe_dir, "*", CASE_NAME)
    dirs = glob.glob(pattern)
    result = []
    for d in dirs:
        idx_str = os.path.basename(os.path.dirname(d))
        try:
            result.append((int(idx_str), d))
        except ValueError:
            log.warning("Carpeta no numérica ignorada: %s", d)
    result.sort(key=lambda x: x[0])
    return result


# ==============================================================================
# LECTURA DE SEÑALES
# ==============================================================================

def read_all_signals(sens_path: str) -> dict[str, np.ndarray]:
    """Lee todas las señales de sens_out.hdf5.

    Auto-descubre grupos que contengan time + values (o dataset directo con 2 cols).
    Retorna dict {signal_name: values_array}.
    """
    signals: dict[str, np.ndarray] = {}
    with h5py.File(sens_path, "r") as f:
        for key in f:
            item = f[key]
            if isinstance(item, h5py.Group):
                if "time" in item and "values" in item:
                    signals[key] = item["values"][:]
                elif "data" in item:
                    arr = item["data"][:]
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        signals[key] = arr[:, 1]
            elif isinstance(item, h5py.Dataset):
                arr = item[:]
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    signals[key] = arr[:, 1]
    return signals


# ==============================================================================
# CÁLCULO SNR
# ==============================================================================

def compute_snr_mod(i_control: np.ndarray, i_degraded: np.ndarray) -> float:
    """Calcula SNR_mod_dB normalizado entre señal de control y señal degradada.

    Ambas señales se normalizan por su RMS antes de comparar → el SNR mide
    diferencias de forma/patrón, no de amplitud. El resultado es siempre
    positivo y directamente comparable con el SNR del DOE de ruido.

    Args:
        i_control:  señal de referencia (I_mod)
        i_degraded: señal degradada     (I_ind)

    Returns:
        SNR_mod_dB  (float; +inf si las formas son idénticas)
    """
    n = min(len(i_control), len(i_degraded))
    ic  = i_control[:n].astype(float)
    id_ = i_degraded[:n].astype(float)

    rms_c = np.sqrt(np.mean(ic  ** 2))
    rms_d = np.sqrt(np.mean(id_ ** 2))

    # Evitar división por cero si alguna señal es nula
    if rms_c < 1e-300 or rms_d < 1e-300:
        return float("nan")

    ic_n  = ic  / rms_c   # control normalizado  → RMS = 1
    id_n  = id_ / rms_d   # degradado normalizado → RMS = 1

    e   = ic_n - id_n
    p_e = np.mean(e ** 2)  # potencia del error de forma

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(1.0 / p_e) if p_e > 0 else np.inf

    return float(snr)


# ==============================================================================
# ACCIONES PRINCIPALES
# ==============================================================================

def list_cases(doe_dir: str) -> None:
    """Imprime tabla de casos disponibles y sale."""
    cases = discover_cases(doe_dir)
    if not cases:
        log.warning("No se encontraron casos en: %s", doe_dir)
        return

    # -- Recoger todos los var_val para saber las columnas disponibles --
    rows = []
    all_keys: list[str] = []
    for idx, case_dir in cases:
        vv = read_var_val(os.path.join(case_dir, "var_val.py"))
        rows.append((idx, case_dir, vv))
        for k in vv:
            if k not in all_keys:
                all_keys.append(k)

    # -- Anchos de columna --
    col_idx  = max(3, len("IDX"))
    col_ctrl = 3   # "✓" o ""
    col_vv   = {k: max(len(k), 10) for k in all_keys}
    for _, _, vv in rows:
        for k in all_keys:
            v = vv.get(k, "")
            col_vv[k] = max(col_vv[k], len(str(v)))

    # -- Cabecera --
    sep = "  "
    hdr_parts  = [f"{'IDX':>{col_idx}}", f"{'CTR':>{col_ctrl}}"]
    hdr_parts += [f"{k:>{col_vv[k]}}" for k in all_keys]
    header = sep.join(hdr_parts)

    print()
    print(f"  DOE : {doe_dir}")
    print(f"  CASE: {CASE_NAME}  |  control_idx={CONTROL_IDX}")
    print()
    print(header)
    print("-" * len(header))

    for idx, case_dir, vv in rows:
        is_ctrl = "✓" if idx == CONTROL_IDX else ""
        row_parts  = [f"{idx:>{col_idx}}", f"{is_ctrl:>{col_ctrl}}"]
        row_parts += [f"{str(vv.get(k, '')):>{col_vv[k]}}" for k in all_keys]
        print(sep.join(row_parts))

    print()
    print(f"  Total casos: {len(cases)}  |  degradados: {len(cases) - 1}")


def run_snr(doe_dir: str, out_path: str, dry_run: bool = False) -> None:
    """Calcula SNR de degradación para todos los casos degradados."""

    cases = discover_cases(doe_dir)
    if not cases:
        log.error("No se encontraron casos en: %s", doe_dir)
        sys.exit(1)

    # -- Separar control de degradados --
    control_case = next(((idx, d) for idx, d in cases if idx == CONTROL_IDX), None)
    if control_case is None:
        log.error("No se encontró el caso control (CONTROL_IDX=%d) en %s", CONTROL_IDX, doe_dir)
        sys.exit(1)

    ctrl_idx, ctrl_dir = control_case
    ctrl_hdf5 = os.path.join(ctrl_dir, "sens_out.hdf5")
    if not os.path.isfile(ctrl_hdf5):
        log.error("sens_out.hdf5 no encontrado en control: %s", ctrl_hdf5)
        sys.exit(1)

    degraded_cases = [(idx, d) for idx, d in cases if idx != CONTROL_IDX]

    log.info("Control: idx=%d | %s", ctrl_idx, ctrl_dir)
    log.info("Casos degradados: %d", len(degraded_cases))
    log.info("Output: %s", out_path)

    if dry_run:
        print("\n[DRY-RUN] Casos a procesar:")
        for idx, case_dir in degraded_cases:
            hdf5_ok = os.path.isfile(os.path.join(case_dir, "sens_out.hdf5"))
            vv = read_var_val(os.path.join(case_dir, "var_val.py"))
            vv_str = "  ".join(f"{k}={v}" for k, v in vv.items())
            status = "OK" if hdf5_ok else "FALTA sens_out.hdf5"
            print(f"  caso_{idx:03d}  [{status}]  {vv_str}")
        return

    # -- Leer señales de control --
    log.info("Leyendo señales del control...")
    ctrl_signals = read_all_signals(ctrl_hdf5)
    if not ctrl_signals:
        log.error("No se encontraron señales en el control: %s", ctrl_hdf5)
        sys.exit(1)
    log.info("Señales encontradas: %s", list(ctrl_signals.keys()))

    # -- Procesar cada caso degradado --
    summary_rows: list[dict] = []

    with h5py.File(out_path, "w") as out_f:
        out_f.attrs["doe_name"]     = DOE_NAME
        out_f.attrs["case_name"]    = CASE_NAME
        out_f.attrs["control_idx"]  = CONTROL_IDX

        for idx, case_dir in degraded_cases:
            hdf5_path = os.path.join(case_dir, "sens_out.hdf5")
            if not os.path.isfile(hdf5_path):
                log.warning("Saltando caso %d — sens_out.hdf5 no encontrado: %s", idx, hdf5_path)
                continue

            vv = read_var_val(os.path.join(case_dir, "var_val.py"))
            deg_signals = read_all_signals(hdf5_path)

            group_name = f"case_{idx:03d}"
            grp = out_f.create_group(group_name)

            # -- Atributos de var_val --
            for k, v in vv.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)

            grp.attrs["case_idx"]  = idx
            grp.attrs["case_path"] = os.path.relpath(case_dir, doe_dir)

            # -- SNR por señal --
            snr_values: dict[str, float] = {}
            common_signals = set(ctrl_signals.keys()) & set(deg_signals.keys())

            for sig in sorted(common_signals):
                snr = compute_snr_mod(ctrl_signals[sig], deg_signals[sig])
                snr_values[sig] = snr
                attr_key = f"SNR_mod_dB_{sig}"
                grp.attrs[attr_key] = snr if np.isfinite(snr) else float("nan")

            log.info(
                "caso_%03d  %s",
                idx,
                "  ".join(f"SNR[{s}]={v:.1f}dB" for s, v in snr_values.items()),
            )
            summary_rows.append({"idx": idx, "snr": snr_values, "vv": vv})

    # -- Tabla resumen --
    if summary_rows:
        signals_list = sorted(summary_rows[0]["snr"].keys())
        header_parts = ["IDX"] + [f"SNR_{s[:12]}" for s in signals_list]
        print("\n" + "  ".join(f"{h:>18}" for h in header_parts))
        print("  ".join(["-" * 18] * len(header_parts)))
        for row in summary_rows:
            vals = [f"{row['idx']:>18}"] + [
                f"{row['snr'].get(s, float('nan')):>17.2f}dB" for s in signals_list
            ]
            print("  ".join(vals))
        print(f"\nResultados guardados en: {out_path}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    epilog = """\
Ejemplos:
  python doe_model_snr.py --list
  python doe_model_snr.py --doe_name DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180
  python doe_model_snr.py --control_idx 2 --dry_run
  python doe_model_snr.py --out resultados_snr.h5

Configuración (editar en el script):
  DOE_NAME     : carpeta DOE a usar por defecto (sobreescribible con --doe_name)
  CASE_NAME    : subcarpeta de simulación dentro de cada índice numérico
  CONTROL_IDX  : índice del caso control (sobreescribible con --control_idx)
  BASE_DIR     : directorio raíz que contiene la carpeta DOE
"""
    p = argparse.ArgumentParser(
        description="doe_model_snr — SNR de degradación entre casos DOE y caso control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--doe_name", default=None, metavar="NAME",
        help=f"Nombre de la carpeta DOE (default: DOE_NAME={DOE_NAME})",
    )
    p.add_argument(
        "--control_idx", type=int, default=None, metavar="N",
        help=f"Índice numérico del caso control (default: CONTROL_IDX={CONTROL_IDX})",
    )
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="Ruta del HDF5 de salida (default: doe_model_snr_results.h5 en la carpeta DOE)",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Imprime tabla de casos disponibles y sale.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Simula sin calcular — imprime plan de tareas.",
    )
    return p.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    global DOE_NAME, CONTROL_IDX  # noqa: PLW0603

    args = parse_args()

    if args.doe_name is not None:
        DOE_NAME = args.doe_name
    if args.control_idx is not None:
        CONTROL_IDX = args.control_idx

    doe_dir = os.path.normpath(os.path.join(BASE_DIR, DOE_NAME))
    if not os.path.isdir(doe_dir):
        log.error("Carpeta DOE no encontrada: %s", doe_dir)
        sys.exit(1)

    if args.list:
        list_cases(doe_dir)
        sys.exit(0)

    out_path = (
        os.path.normpath(args.out)
        if args.out
        else os.path.join(doe_dir, "doe_model_snr_results.h5")
    )

    run_snr(doe_dir, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

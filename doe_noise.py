"""doe_noise.py — Aplica ruido gaussiano (varios niveles SNR) a un caso de control
extraído de doe_results.h5 y guarda los resultados en doe_noise_results.h5.

Uso:
    python doe_noise.py --doe_results .\DOE_xxx\doe_results.h5
    python doe_noise.py --doe_results .\DOE_xxx\doe_results.h5 --out ruido.h5

Configuración (editar en bloque CONFIG):
    CONTROL_CASE_IDX : índice del caso de control (case_000 = 0, case_001 = 1, …)
    SNR_LIST         : lista fija de niveles SNR en dB  (None → ignorar)
    SNR_RANGE        : (min_dB, max_dB, step_dB) para generar rango (None → ignorar)
    SEED             : semilla para reproducibilidad del ruido
    SIGNALS          : señales a procesar
"""

import os
import sys
import argparse
import logging
import numpy as np
import h5py

# ==============================================================================
# CONFIG — editar aquí
# ==============================================================================

CONTROL_CASE_IDX = 3                   # índice del caso de control en doe_results.h5
# SNR_LIST         = [10,20,30,40,50, 60,80,90,100,110,120,130,140]      # dB — lista fija (None si no se usa)
SNR_LIST         = None      # dB — lista fija (None si no se usa)
SNR_RANGE        = [5, 200, 5]                 # ej. (5, 50, 5) → [5,10,15,...,50] dB
SEED             = 42                   # semilla para np.random (reproducibilidad)
SIGNALS          = ["Axial_disp", "Axial_vel"]

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def build_snr_list(snr_fixed, snr_range):
    """Combina SNR_LIST y SNR_RANGE, deduplica y ordena descendente."""
    values = list(snr_fixed) if snr_fixed else []
    if snr_range is not None:
        lo, hi, step = snr_range
        n = round((hi - lo) / step)
        for i in range(n + 1):
            v = lo + i * step
            if v not in values:
                values.append(v)
    if not values:
        raise ValueError("No hay niveles SNR definidos. Edita SNR_LIST o SNR_RANGE en CONFIG.")
    return sorted(set(values), reverse=True)


def add_gaussian_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Añade ruido gaussiano blanco de potencia tal que SNR = snr_db dB.

    P_signal = mean(y²)
    sigma    = sqrt(P_signal / 10^(snr_db/10))
    y_noisy  = y + N(0, sigma²)
    """
    p_signal = np.mean(y ** 2)
    if p_signal == 0.0:
        log.warning("Señal con potencia cero — el ruido no tendrá efecto práctico.")
        p_signal = 1e-30
    sigma = np.sqrt(p_signal / (10.0 ** (snr_db / 10.0)))
    return y + rng.normal(0.0, sigma, size=y.shape)


# ------------------------------------------------------------------------------
# Listado de casos
# ------------------------------------------------------------------------------

def list_cases(h5_path: str) -> None:
    """Imprime en consola una tabla con todos los casos del HDF5 y sus atributos."""
    with h5py.File(h5_path, "r") as f:
        groups = sorted(f.keys())
        if not groups:
            print("  (sin casos)")
            return

        # Recopilar todos los attrs de todos los grupos
        rows = []
        all_keys = []
        for grp_name in groups:
            attrs = dict(f[grp_name].attrs)
            rows.append((grp_name, attrs))
            for k in attrs:
                if k not in all_keys:
                    all_keys.append(k)

        # Columnas: idx | group | attrs...
        col_names = ["idx", "group"] + all_keys
        table = []
        for i, (grp_name, attrs) in enumerate(rows):
            row = [str(i), grp_name] + [str(attrs.get(k, "-")) for k in all_keys]
            table.append(row)

        # Calcular anchos de columna
        widths = [max(len(col_names[j]), max(len(r[j]) for r in table))
                  for j in range(len(col_names))]

        sep  = "  ".join("-" * w for w in widths)
        header = "  ".join(col_names[j].ljust(widths[j]) for j in range(len(col_names)))
        print()
        print(header)
        print(sep)
        for row in table:
            print("  ".join(row[j].ljust(widths[j]) for j in range(len(col_names))))
        print()
        print(f"  Total: {len(rows)} casos  |  Para usar como control: edita CONTROL_CASE_IDX en CONFIG")
        print()


# ------------------------------------------------------------------------------
# Carga del caso de control
# ------------------------------------------------------------------------------

def load_control_case(h5_path: str, case_idx: int) -> dict:
    """Lee el grupo case_{idx:03d} del HDF5 y retorna sus señales y atributos."""
    grp_name = f"case_{case_idx:03d}"
    with h5py.File(h5_path, "r") as f:
        if grp_name not in f:
            available = sorted(f.keys())
            raise KeyError(
                f"Grupo '{grp_name}' no encontrado en {h5_path}.\n"
                f"  Grupos disponibles: {available}"
            )
        grp = f[grp_name]
        attrs = dict(grp.attrs)
        signals = {}
        for sig in SIGNALS:
            if sig in grp:
                t = grp[f"{sig}/time"][()]
                y = grp[f"{sig}/values"][()]
                signals[sig] = (t, y)
            else:
                log.warning("Señal '%s' no encontrada en grupo '%s' — se omite.", sig, grp_name)
    return {"group": grp_name, "attrs": attrs, "signals": signals}


# ------------------------------------------------------------------------------
# Escritura HDF5 de salida
# ------------------------------------------------------------------------------

def write_noise_results(out_path: str, control: dict, snr_list: list, seed: int) -> None:
    """Escribe doe_noise_results.h5 con un grupo por nivel SNR + grupo 'control'."""
    rng = np.random.default_rng(seed)

    with h5py.File(out_path, "w") as out_f:
        # --- grupo control (señal original) ---
        ctrl_grp = out_f.create_group("control")
        for k, v in control["attrs"].items():
            ctrl_grp.attrs[k] = v
        ctrl_grp.attrs["snr_db"]       = "None"
        ctrl_grp.attrs["case_source"]  = control["group"]

        for sig, (t, y) in control["signals"].items():
            sg = ctrl_grp.create_group(sig)
            sg.create_dataset("time",   data=t, compression="gzip")
            sg.create_dataset("values", data=y, compression="gzip")

        log.info("Grupo 'control' escrito  (señal original, sin ruido)")

        # --- un grupo por nivel SNR ---
        for snr_db in snr_list:
            grp_name = f"snr_{snr_db:06.2f}"
            snr_grp  = out_f.create_group(grp_name)

            for k, v in control["attrs"].items():
                snr_grp.attrs[k] = v
            snr_grp.attrs["snr_db"]      = float(snr_db)
            snr_grp.attrs["case_source"] = control["group"]
            snr_grp.attrs["seed"]        = seed

            for sig, (t, y) in control["signals"].items():
                y_noisy = add_gaussian_noise(y, snr_db, rng)
                sg = snr_grp.create_group(sig)
                sg.create_dataset("time",   data=t,       compression="gzip")
                sg.create_dataset("values", data=y_noisy, compression="gzip")

            log.info("Grupo '%s' escrito  (SNR = %.1f dB)", grp_name, snr_db)

    log.info("Archivo de salida: %s", out_path)
    log.info("Total grupos escritos: control + %d niveles SNR", len(snr_list))


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Aplica ruido gaussiano (multi-SNR) al caso de control de doe_results.h5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Configuración (editar bloque CONFIG en el script):
  CONTROL_CASE_IDX  índice del caso de control (0 = case_000, 1 = case_001, …)
  SNR_LIST          lista fija de niveles SNR en dB  ej. [5, 10, 20, 40]
  SNR_RANGE         rango (min, max, step) en dB     ej. (5, 50, 5)
  SEED              semilla aleatoria para reproducibilidad (default 42)
  SIGNALS           señales a procesar               ej. ["Axial_disp", "Axial_vel"]

Estructura del HDF5 generado:
  doe_noise_results.h5
  ├── control/          señal original sin ruido
  ├── snr_040.00/       señal con SNR = 40 dB (poco ruido)
  ├── snr_020.00/
  ├── snr_010.00/
  └── snr_005.00/       señal con SNR =  5 dB (mucho ruido)

Ejemplos:
  python doe_noise.py --doe_results .\\DOE_xxx\\doe_results.h5
  python doe_noise.py --doe_results .\\DOE_xxx\\doe_results.h5 --out ruido.h5
        """,
    )
    p.add_argument(
        "--doe_results",
        required=True,
        metavar="PATH",
        help="Ruta al archivo doe_results.h5 generado por doe_runner --command extract",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Imprime la tabla de casos disponibles en el HDF5 y sale.",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Ruta del HDF5 de salida (default: doe_noise_results.h5 junto a --doe_results)",
    )
    return p.parse_args()


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    doe_results = os.path.normpath(args.doe_results)
    if not os.path.isfile(doe_results):
        log.error("Archivo no encontrado: %s", doe_results)
        sys.exit(1)

    if args.list:
        list_cases(doe_results)
        sys.exit(0)

    if args.out is None:
        out_path = os.path.join(os.path.dirname(doe_results), "doe_noise_results.h5")
    else:
        out_path = os.path.normpath(args.out)

    # Construir lista SNR
    snr_list = build_snr_list(SNR_LIST, SNR_RANGE)
    log.info("Niveles SNR (dB): %s", snr_list)

    # Cargar caso de control
    log.info("Cargando caso de control: case_%03d  de  %s", CONTROL_CASE_IDX, doe_results)
    control = load_control_case(doe_results, CONTROL_CASE_IDX)
    log.info("Atributos del caso de control: %s", control["attrs"])
    log.info("Señales encontradas: %s", list(control["signals"].keys()))

    # Escribir resultados
    write_noise_results(out_path, control, snr_list, SEED)


if __name__ == "__main__":
    main()

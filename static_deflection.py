#!/usr/bin/env python
# coding: utf-8
"""static_deflection.py — Resta la deflexión estática a Axial_disp y la guarda en doe_results.h5.

Script standalone: NO importa Etapa_1.py. Copia a propósito las 3 funciones
puras que necesita (_read_case_ap, compute_case_force_from_ap,
compute_static_deflection) en vez de importarlas, para no acoplarse a ese
script ni a que su firma cambie — es solo matemática, sin dependencias.

Escribe en el MISMO esquema que ya usa Etapa_1.py: case_XXX/Out_Deflex/
Axial_disp_out_deflex/{time,values} y .../Axial_vel_out_deflex/{time,values}
(velocidad sin cambios, solo copiada al lado). Así doe_unified_selector.py
(que ya lee Out_Deflex) y reference_dataset.py lo ven igual sin importar
cuál de los dos scripts lo generó.

Uso:
    python static_deflection.py                    # usa DEFAULT_H5_PATH
    python static_deflection.py ruta\\doe_results.h5
    python static_deflection.py --dry-run           # solo imprime, no escribe
    python static_deflection.py --selftest
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np

log = logging.getLogger(__name__)

# ==============================================================================
# CONFIG — editar acá antes de ejecutar
# ==============================================================================
DEFAULT_H5_PATH = None   # ej. r"D:\...\doe_results.h5"; None -> hay que pasarlo por CLI

F_TOOTH_MM = 0.05        # avance por diente [mm/diente]
K_CUT      = 1_000.0     # coeficiente de fuerza específica de corte [N/mm^2]
K_SYS      = 2.13e8      # rigidez equivalente del sistema [N/m]
ALPHA_DEG  = 135.0
THETA_DEG  = 90.0


# ==============================================================================
# Fórmulas — copiadas de Etapa_1.py a propósito (ver docstring del módulo)
# ==============================================================================

def _read_case_ap(grp: h5py.Group) -> float:
    """Lee el a_p del caso desde Ap_start o Ap_end."""
    if "$Ap_start$" in grp.attrs:
        return float(grp.attrs["$Ap_start$"])
    if "$Ap_end$" in grp.attrs:
        return float(grp.attrs["$Ap_end$"])
    raise KeyError(f"[{grp.name}] No se encontró Ap_start ni Ap_end en atributos")


def compute_case_force_from_ap(case_ap: float, f_tooth_mm: float, k_cut: float) -> float:
    """Fuerza teórica del caso a partir de su a_p y constantes de corte."""
    return k_cut * (case_ap * 1e3) * f_tooth_mm


def compute_static_deflection(
    force_ref: float, stiffness: float, alpha_deg: float = 135.0, theta_deg: float = 90.0
) -> float:
    """Deflexión estática proyectando la fuerza en la dirección modal.

    q_s = F_c * cos(alpha - theta) / K_sys  (metros, si K_sys está en N/m).
    """
    modal_force = force_ref * np.cos(np.deg2rad(alpha_deg - theta_deg))
    deflex_static_modal = modal_force / stiffness
    return np.cos(np.deg2rad(alpha_deg - theta_deg)) * deflex_static_modal


def _read_time_values(grp: h5py.Group, signal_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if signal_name not in grp or not isinstance(grp[signal_name], h5py.Group):
        return None
    sig = grp[signal_name]
    if "time" not in sig or "values" not in sig:
        return None
    return sig["time"][()], sig["values"][()]


# ==============================================================================
# Cálculo + escritura
# ==============================================================================

def apply_static_deflection(
    h5_path: str,
    f_tooth_mm: float = F_TOOTH_MM, k_cut: float = K_CUT, k_sys: float = K_SYS,
    alpha_deg: float = ALPHA_DEG, theta_deg: float = THETA_DEG,
    dry_run: bool = False,
) -> None:
    """Por cada caso de `h5_path`: calcula la deflexión estática y la resta de Axial_disp.

    Escribe (o solo informa, si `dry_run`) en case_XXX/Out_Deflex/
    Axial_disp_out_deflex/{time,values} y .../Axial_vel_out_deflex/{time,values}.
    """
    mode = "r" if dry_run else "a"
    with h5py.File(h5_path, mode) as h5f:
        case_names = sorted(name for name in h5f.keys() if name.startswith("case_"))
        for case_name in case_names:
            grp = h5f[case_name]
            case_ap = _read_case_ap(grp)
            force = compute_case_force_from_ap(case_ap, f_tooth_mm, k_cut)
            deflex = compute_static_deflection(force, k_sys, alpha_deg, theta_deg)

            disp_data = _read_time_values(grp, "Axial_disp")
            vel_data = _read_time_values(grp, "Axial_vel")
            if disp_data is None or vel_data is None:
                log.warning("[%s] falta Axial_disp o Axial_vel — omitido", case_name)
                continue
            disp_time, disp_values = disp_data
            vel_time, vel_values = vel_data
            if disp_time.shape != vel_time.shape or not np.allclose(disp_time, vel_time):
                raise ValueError(f"[{case_name}] Axial_disp y Axial_vel no comparten el eje temporal")

            disp_corrected = np.asarray(disp_values, dtype=float) - float(deflex)

            if dry_run:
                print(f"{case_name}: a_p={case_ap:.4g}  fuerza={force:.4g} N  deflex={deflex:.4g} m")
                continue

            out_grp = grp.require_group("Out_Deflex")
            disp_out = out_grp.require_group("Axial_disp_out_deflex")
            vel_out = out_grp.require_group("Axial_vel_out_deflex")
            for target, time_arr, values_arr in (
                (disp_out, disp_time, disp_corrected),
                (vel_out, vel_time, vel_values),
            ):
                if "time" in target:
                    del target["time"]
                target.create_dataset("time", data=time_arr)
                if "values" in target:
                    del target["values"]
                target.create_dataset("values", data=values_arr)

            grp.attrs["deflex_theoric_m"] = deflex
            grp.attrs["force_theoric_N"] = force

    if not dry_run:
        print(f"[OK] Out_Deflex escrito para {len(case_names)} casos en {h5_path}")


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = os.path.join(tmp, "doe_results.h5")
        t = np.linspace(0.0, 1.0, 101)
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("case_000")
            grp.attrs["$Ap_start$"] = 0.01
            for name, vals in (("Axial_disp", t.copy()), ("Axial_vel", np.ones_like(t))):
                sub = grp.create_group(name)
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=vals)

        # dry-run no escribe nada
        apply_static_deflection(h5_path, dry_run=True)
        with h5py.File(h5_path, "r") as f:
            assert "Out_Deflex" not in f["case_000"]

        apply_static_deflection(h5_path)
        expected_force = compute_case_force_from_ap(0.01, F_TOOTH_MM, K_CUT)
        expected_deflex = compute_static_deflection(expected_force, K_SYS, ALPHA_DEG, THETA_DEG)
        with h5py.File(h5_path, "r") as f:
            grp = f["case_000"]
            assert "Out_Deflex" in grp
            disp_corr = grp["Out_Deflex"]["Axial_disp_out_deflex"]["values"][()]
            vel_corr = grp["Out_Deflex"]["Axial_vel_out_deflex"]["values"][()]
            assert np.allclose(disp_corr, t - expected_deflex)
            assert np.allclose(vel_corr, np.ones_like(t))  # velocidad sin cambios
            assert abs(float(grp.attrs["deflex_theoric_m"]) - expected_deflex) < 1e-12
            assert abs(float(grp.attrs["force_theoric_N"]) - expected_force) < 1e-9

        # re-correr no debe fallar (require_group + del antes de reescribir cada dataset)
        apply_static_deflection(h5_path)

    print("self-test OK")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "h5_path", nargs="?", default=DEFAULT_H5_PATH,
        help=f"doe_results.h5 a corregir (default: DEFAULT_H5_PATH = {DEFAULT_H5_PATH!r})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime a_p/fuerza/deflexión por caso, no escribe nada")
    parser.add_argument("--selftest", action="store_true", help="Corre el self-test (assert-based) y sale")
    args = parser.parse_args()

    if args.selftest:
        _self_test()
        return

    if not args.h5_path:
        parser.error("falta h5_path — pasalo como argumento o fijá DEFAULT_H5_PATH arriba del script")

    apply_static_deflection(args.h5_path, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()

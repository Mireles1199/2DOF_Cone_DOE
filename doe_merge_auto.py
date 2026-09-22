#!/usr/bin/env python
# coding: utf-8
"""
Merge automatico de familias de DOE (mismo DOE, distintos lotes de corridas).

Agrupa las subcarpetas de --root cuyo nombre matchea "<familia>_RUN_<N>" (ej.
DOE_Detection_Limite_Lobes_dt_12.5_RUN_10 y ..._RUN_5 -> familia
"DOE_Detection_Limite_Lobes_dt_12.5"). Cada familia con 2+ miembros se copia
completa a una carpeta nueva "<familia>_MERGED" y ahi se van mergeando los
demas miembros con doe_runner.merge_doe_results — las carpetas originales no
se tocan.

Carpetas cuyo nombre no matchea exactamente "<algo>_RUN_<numero>" al final
(ej. "..._RUN_10_patch_0.95") quedan afuera del auto-merge.

Uso:
    python doe_merge_auto.py --root <carpeta_con_varios_DOE>
    python doe_merge_auto.py --root <carpeta> --dry-run
    python doe_merge_auto.py --self-test
"""

import argparse
import logging
import os
import re
import shutil
import tempfile

from doe_runner import merge_doe_results

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_FAMILY_RE = re.compile(r"^(.+)_RUN_(\d+)$")


def find_families(root: str) -> dict:
    """Agrupa subcarpetas de root por familia (<base>_RUN_<N> -> base). Ignora lo que no matchea."""
    families: dict = {}
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        m = _FAMILY_RE.match(name)
        if not m:
            continue
        family, run_n = m.group(1), int(m.group(2))
        families.setdefault(family, []).append((run_n, name))
    return families


def merge_family(root: str, family: str, members: list, dry_run: bool) -> None:
    members = sorted(members, key=lambda t: -t[0])  # RUN mas grande primero -> semilla
    seed_run, seed_name = members[0]
    out_name = f"{family}_MERGED"
    out_dir = os.path.join(root, out_name)

    if os.path.isdir(out_dir):
        log.warning("Ya existe %s, se omite la familia (borrala a mano si queres rehacerla).", out_name)
        return

    log.info("Familia '%s': %d miembros -> %s", family, len(members), out_name)
    if dry_run:
        log.info("[DRY-RUN] Copiaria %s -> %s", seed_name, out_name)
        for _, name in members[1:]:
            log.info("[DRY-RUN] Merge %s -> %s", name, out_name)
        return

    shutil.copytree(os.path.join(root, seed_name), out_dir)
    log.info("Copiado %s -> %s", seed_name, out_name)

    for _, name in members[1:]:
        merge_doe_results(out_dir, os.path.join(root, name), dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge automatico de familias de DOE por nombre de carpeta.")
    parser.add_argument("--root", help="Carpeta que contiene las carpetas de DOE a agrupar")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Corre el self-test y sale")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.root:
        parser.error("--root es obligatorio (o usa --self-test)")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        log.error("Carpeta no encontrada: %s", root)
        return

    families = find_families(root)
    multi   = {k: v for k, v in families.items() if len(v) >= 2}
    single  = {k: v for k, v in families.items() if len(v) == 1}

    if single:
        log.info("Con patron _RUN_<N> pero sin pareja (no se mergean): %s",
                 [name for v in single.values() for _, name in v])

    if not multi:
        log.info("No se encontraron familias con 2+ miembros en %s", root)
        return

    for family, members in sorted(multi.items()):
        merge_family(root, family, members, args.dry_run)


def _self_test() -> None:
    """Self-check del agrupamiento por familia — no toca datos reales."""
    with tempfile.TemporaryDirectory() as tmp:
        names = [
            "DOE_x_dt_12.5_RUN_10", "DOE_x_dt_12.5_RUN_5",
            "DOE_x_dt_25_RUN_10", "DOE_x_dt_25_RUN_5",
            "DOE_x_dt_25_RUN_10_patch_0.95",   # no debe matchear (sufijo extra)
            "DOE_x_dt_50_RUN_10",              # sin pareja
            "not_a_doe_folder",                # no matchea
        ]
        for n in names:
            os.makedirs(os.path.join(tmp, n))

        families = find_families(tmp)
        assert families["DOE_x_dt_12.5"] == [(10, "DOE_x_dt_12.5_RUN_10"), (5, "DOE_x_dt_12.5_RUN_5")]
        assert families["DOE_x_dt_25"] == [(10, "DOE_x_dt_25_RUN_10"), (5, "DOE_x_dt_25_RUN_5")]
        assert families["DOE_x_dt_50"] == [(10, "DOE_x_dt_50_RUN_10")]
        assert "DOE_x_dt_25_RUN_10_patch_0.95" not in [n for v in families.values() for _, n in v]
        assert "not_a_doe_folder" not in families

    print("self-test OK")


if __name__ == "__main__":
    main()

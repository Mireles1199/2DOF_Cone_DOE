#!/usr/bin/env python
# coding: utf-8
"""reference_dataset.py — Dataset externo de señales de referencia (Fase 1).

Hoy los indicadores (MaxEnt, RMS-CV, SST, Green-Area) calculan su umbral de
detección cortando un tramo de la MISMA señal que analizan. Este módulo arma
un dataset de referencia EXTERNO — señales ya etiquetadas (stable/unstable)
por intervalos de tiempo — que en una fase futura los indicadores podrán
consumir en lugar de recortar su propia señal.

Es agnóstico del origen de los datos: `ReferenceDataset`/`ReferenceSignal` no
saben nada de DOE. `from_doe_h5` es el único adaptador que sabe leer
doe_results.h5 / doe_noise_results.h5 (layout de doe_runner.py).

Secuencia de uso:
    1. make_label_template(doe_results.h5, reference_labels.yaml)
    2. (a mano) completar intervalos en reference_labels.yaml
    3. from_doe_h5(doe_results.h5, reference_labels.yaml, channels) -> ReferenceDataset
    4. dataset.to_hdf5("reference_dataset.h5")                      -> portable
    5. (en cualquier lado) ReferenceDataset.from_hdf5(...)

Fuera de alcance de esta fase: etiquetado automático, combinar señales,
features/GMM/GP, y cualquier cambio en doe_indicators.py / doe_noise_indicators.py
o en los indicadores.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)

VALID_LABELS = {"stable", "unstable"}


# ==============================================================================
# PIEZA 2 — Esquema canónico
# ==============================================================================

@dataclass
class ReferenceSignal:
    id: str                                    # ej. "case_007/Axial_vel"
    t: np.ndarray                               # señal completa, sin recortar
    y: np.ndarray
    fs: float
    intervals: List[Tuple[float, float, str]]   # [(t0, t1, label), ...]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceDataset:
    signals: List[ReferenceSignal]

    def segments(self, label: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Tramos recortados (id, t, y) de todas las señales que tengan `label`."""
        out: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for sig in self.signals:
            for t0, t1, lab in sig.intervals:
                if lab != label:
                    continue
                mask = (sig.t >= t0) & (sig.t <= t1)
                out.append((sig.id, sig.t[mask], sig.y[mask]))
        return out

    def to_hdf5(self, path: str) -> None:
        with h5py.File(path, "w") as f:
            for sig in self.signals:
                grp = f.create_group(sig.id.replace("/", "__"))
                grp.create_dataset("t", data=sig.t)
                grp.create_dataset("y", data=sig.y)

                if sig.intervals:
                    bounds = np.array([[t0, t1] for t0, t1, _ in sig.intervals], dtype=float)
                    labels = np.array([lab for _, _, lab in sig.intervals], dtype=object)
                else:
                    bounds = np.zeros((0, 2), dtype=float)
                    labels = np.array([], dtype=object)
                grp.create_dataset("interval_bounds", data=bounds)
                grp.create_dataset("interval_labels", data=labels, dtype=h5py.string_dtype())

                grp.attrs["id"] = sig.id
                grp.attrs["fs"] = sig.fs
                for k, v in sig.attrs.items():
                    try:
                        grp.attrs[k] = v
                    except Exception:
                        grp.attrs[k] = str(v)

    @classmethod
    def from_hdf5(cls, path: str) -> "ReferenceDataset":
        signals = []
        with h5py.File(path, "r") as f:
            for grp_name in f.keys():
                grp = f[grp_name]
                t = grp["t"][()]
                y = grp["y"][()]

                bounds = grp["interval_bounds"][()] if "interval_bounds" in grp else np.zeros((0, 2))
                raw_labels = grp["interval_labels"][()] if "interval_labels" in grp else np.array([])
                intervals = [
                    (float(b[0]), float(b[1]), (lab.decode() if isinstance(lab, bytes) else str(lab)))
                    for b, lab in zip(bounds, raw_labels)
                ]

                attrs = dict(grp.attrs)
                sig_id = attrs.pop("id", grp_name)
                fs = attrs.pop("fs", 1.0 / float(t[1] - t[0]))
                signals.append(ReferenceSignal(id=sig_id, t=t, y=y, fs=fs, intervals=intervals, attrs=attrs))
        return cls(signals=signals)


# ==============================================================================
# PIEZA 1 — Etiquetado (manual)
# ==============================================================================

def make_label_template(h5_path: str, out_yaml: str) -> None:
    """Genera `out_yaml` con todos los grupos de `h5_path` sin etiquetar.

    Se niega a sobrescribir un YAML ya existente, para no perder etiquetas
    hechas a mano.
    """
    if os.path.exists(out_yaml):
        raise FileExistsError(
            f"{out_yaml} ya existe — no se sobrescribe (podrías perder etiquetas hechas a mano)."
        )

    lines = [f"source: {os.path.basename(h5_path)}", "cases:"]
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = dict(grp.attrs)
            kappa_bits = {k: v for k, v in attrs.items() if str(k).startswith("kappa")}

            t_range = None
            for key in grp.keys():
                sub = grp[key]
                if isinstance(sub, h5py.Group) and "time" in sub:
                    t = sub["time"]
                    t_range = (float(t[0]), float(t[-1]))
                    break

            comment_bits = []
            if kappa_bits:
                comment_bits.append(", ".join(f"{k}={v}" for k, v in kappa_bits.items()))
            if t_range is not None:
                comment_bits.append(f"t=[{t_range[0]:.2f}, {t_range[1]:.2f}] s")
            comment = f"  # {'   '.join(comment_bits)}" if comment_bits else ""

            lines.append(f"  {grp_name}: []{comment}")

    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_labels_file(labels_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """Lee y valida la sintaxis de `labels_path` (sin tocar el .h5)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cases = raw.get("cases") or {}

    parsed: Dict[str, List[Tuple[float, float, str]]] = {}
    for grp_name, raw_intervals in cases.items():
        intervals: List[Tuple[float, float, str]] = []
        for iv in raw_intervals or []:
            if len(iv) != 3:
                raise ValueError(
                    f"{labels_path}: intervalo inválido en '{grp_name}': {iv!r} (esperado [t0, t1, label])"
                )
            t0, t1, label = float(iv[0]), float(iv[1]), iv[2]
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{labels_path}: label desconocido '{label}' en '{grp_name}' (válidos: {sorted(VALID_LABELS)})"
                )
            if not (t0 < t1):
                raise ValueError(f"{labels_path}: intervalo inválido en '{grp_name}': t0={t0} >= t1={t1}")
            intervals.append((t0, t1, label))

        intervals.sort(key=lambda iv: iv[0])
        for a, b in zip(intervals, intervals[1:]):
            if a[1] > b[0]:
                raise ValueError(f"{labels_path}: intervalos solapados en '{grp_name}': {a} y {b}")
        parsed[grp_name] = intervals

    return parsed


# ==============================================================================
# PIEZA 3 — Adaptador de origen (el único que conoce doe_runner)
# ==============================================================================

def from_doe_h5(h5_path: str, labels_path: str, channels: List[str]) -> ReferenceDataset:
    """Construye un ReferenceDataset a partir de un doe_results.h5 + su YAML de etiquetas.

    NO importa doe_indicators.py (acoplaría el dataset a los 4 indicadores) —
    la lectura de señal/attrs se replica acá, igual layout que `_load_case`.
    """
    cases = _parse_labels_file(labels_path)
    signals: List[ReferenceSignal] = []

    with h5py.File(h5_path, "r") as f:
        for grp_name, intervals in cases.items():
            if not intervals:
                continue  # sin etiquetar -> se ignora
            if grp_name not in f:
                raise KeyError(f"Grupo '{grp_name}' de {labels_path} no existe en {h5_path}")

            grp = f[grp_name]
            base_attrs = dict(grp.attrs)

            for ch in channels:
                if ch not in grp or "time" not in grp[ch]:
                    log.warning("Señal '%s' no está en grupo '%s' — omitida", ch, grp_name)
                    continue

                t = grp[f"{ch}/time"][()]
                y = grp[f"{ch}/values"][()]
                t0_sig, t1_sig = float(t[0]), float(t[-1])
                for a, b, _ in intervals:
                    if a < t0_sig or b > t1_sig:
                        raise ValueError(
                            f"Intervalo [{a}, {b}] fuera del rango de la señal "
                            f"[{t0_sig}, {t1_sig}] en '{grp_name}/{ch}'"
                        )

                attrs = dict(base_attrs)
                attrs["source"] = "doe_h5"
                attrs["source_file"] = os.path.basename(h5_path)
                attrs["group"] = grp_name
                attrs["channel"] = ch

                signals.append(ReferenceSignal(
                    id=f"{grp_name}/{ch}",
                    t=t, y=y,
                    fs=1.0 / float(t[1] - t[0]),
                    intervals=list(intervals),
                    attrs=attrs,
                ))

    return ReferenceDataset(signals=signals)


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = os.path.join(tmp, "doe_results.h5")
        yaml_path = os.path.join(tmp, "reference_labels.yaml")

        # 1. .h5 sintético con layout DOE (3 grupos, canal Axial_vel, attrs kappa)
        t = np.linspace(0.0, 10.0, 1001)
        with h5py.File(h5_path, "w") as f:
            for i, kappa in enumerate([0.8, 1.4, 1.1]):
                grp = f.create_group(f"case_{i:03d}")
                grp.attrs["kappa"] = kappa
                sub = grp.create_group("Axial_vel")
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=np.sin(2 * np.pi * 5 * t) + i)

        # 2. make_label_template -> YAML con los 3 grupos vacíos; no sobrescribe
        make_label_template(h5_path, yaml_path)
        with open(yaml_path, encoding="utf-8") as f:
            content = f.read()
        assert "case_000" in content and "case_001" in content and "case_002" in content
        assert "kappa=0.8" in content
        try:
            make_label_template(h5_path, yaml_path)
            raise AssertionError("debía fallar por sobrescritura")
        except FileExistsError:
            pass

        # 3. completar el YAML programáticamente
        labels = {
            "source": "doe_results.h5",
            "cases": {
                "case_000": [[0.0, 10.0, "stable"]],
                "case_001": [[0.0, 5.0, "stable"], [5.0, 10.0, "unstable"]],
                "case_002": [],
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(labels, f)

        # 4. from_doe_h5 -> 2 señales (case_002 se ignora), intervalos y attrs conservados
        ds = from_doe_h5(h5_path, yaml_path, channels=["Axial_vel"])
        assert len(ds.signals) == 2, f"esperaba 2 señales, dio {len(ds.signals)}"
        ids = {s.id for s in ds.signals}
        assert ids == {"case_000/Axial_vel", "case_001/Axial_vel"}
        sig1 = next(s for s in ds.signals if s.id == "case_001/Axial_vel")
        assert sig1.intervals == [(0.0, 5.0, "stable"), (5.0, 10.0, "unstable")]
        assert sig1.attrs["source"] == "doe_h5"
        assert sig1.attrs["group"] == "case_001"
        assert sig1.attrs["channel"] == "Axial_vel"
        assert abs(sig1.attrs["kappa"] - 1.4) < 1e-9

        # 5. segments("stable") -> 2 tramos; segments("unstable") -> 1 tramo, rango correcto
        stable = ds.segments("stable")
        unstable = ds.segments("unstable")
        assert len(stable) == 2
        assert len(unstable) == 1
        uid, ut, uy = unstable[0]
        assert uid == "case_001/Axial_vel"
        assert ut.min() >= 5.0 - 1e-9 and ut.max() <= 10.0 + 1e-9

        # 6. YAML inválido -> error explícito
        def _expect_value_error(cases_dict):
            bad_path = os.path.join(tmp, f"bad_{len(cases_dict)}_{list(cases_dict)[0]}.yaml")
            with open(bad_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"source": "x", "cases": cases_dict}, f)
            try:
                from_doe_h5(h5_path, bad_path, channels=["Axial_vel"])
                raise AssertionError(f"debía fallar: {cases_dict}")
            except ValueError:
                pass

        _expect_value_error({"case_000": [[0.0, 1.0, "weird_label"]]})       # label desconocido
        _expect_value_error({"case_000": [[0.0, 5.0, "stable"], [4.0, 8.0, "unstable"]]})  # solape
        _expect_value_error({"case_000": [[0.0, 999.0, "stable"]]})          # fuera de rango

        # 7. to_hdf5 -> from_hdf5, round-trip idéntico
        out_h5 = os.path.join(tmp, "reference_dataset.h5")
        ds.to_hdf5(out_h5)
        ds2 = ReferenceDataset.from_hdf5(out_h5)
        assert len(ds2.signals) == len(ds.signals)
        by_id = {s.id: s for s in ds2.signals}
        for sig in ds.signals:
            sig2 = by_id[sig.id]
            assert np.array_equal(sig.t, sig2.t)
            assert np.array_equal(sig.y, sig2.y)
            assert sig.intervals == sig2.intervals
            assert abs(sig.fs - sig2.fs) < 1e-9
            assert sig2.attrs.get("group") == sig.attrs.get("group")
            assert abs(sig2.attrs.get("kappa") - sig.attrs.get("kappa")) < 1e-9

    print("self-test OK")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="Corre el self-test (assert-based)")

    p_template = sub.add_parser("template", help="Genera plantilla de etiquetas YAML")
    p_template.add_argument("h5_path")
    p_template.add_argument("out_yaml")

    p_build = sub.add_parser("build", help="Construye y guarda un ReferenceDataset")
    p_build.add_argument("h5_path")
    p_build.add_argument("labels_yaml")
    p_build.add_argument("out_h5")
    p_build.add_argument("--channels", nargs="+", required=True)

    args = parser.parse_args()

    if args.cmd == "selftest":
        _self_test()
    elif args.cmd == "template":
        make_label_template(args.h5_path, args.out_yaml)
        print(f"Plantilla escrita en {args.out_yaml}")
    elif args.cmd == "build":
        ds = from_doe_h5(args.h5_path, args.labels_yaml, channels=args.channels)
        ds.to_hdf5(args.out_h5)
        print(f"{len(ds.signals)} señales -> {args.out_h5}")


if __name__ == "__main__":
    _main()

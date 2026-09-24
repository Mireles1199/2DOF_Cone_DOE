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
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)

VALID_LABELS = {"stable", "unstable"}

# ==============================================================================
# CONFIG — editar acá los defaults del CLI; los flags de línea de comandos
# los pisan si se pasan (mismo patrón que _T_GT/_CUT_START en doe_indicators.py)
# ==============================================================================
DEFAULT_H5_PATH = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
    r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
    r"\doe_results.h5"
)

DEFAULT_LABELS_PATH     = None   # None -> "<carpeta de h5_path>/reference_labels.yaml"
DEFAULT_OUT_H5          = None   # None -> "<carpeta de h5_path>/reference_dataset.h5"
DEFAULT_CHANNELS        = None   # None -> autodetecta todos los canales de cada caso
DEFAULT_STRATEGY        = "kappa"
DEFAULT_KAPPA_THRESHOLD = 1.0
DEFAULT_WARMUP          = 0.0


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
# PIEZA 1 — Etiquetado
# ==============================================================================

def _label_manual(grp_name: str, attrs: dict, t_range: Tuple[float, float]) -> List[Tuple[float, float, str]]:
    """Estrategia por defecto: no etiqueta nada, el usuario completa a mano."""
    return []


def _label_by_kappa(
    grp_name: str, attrs: dict, t_range: Tuple[float, float],
    threshold: float = 1.0, warmup: float = 0.0,
) -> List[Tuple[float, float, str]]:
    """Etiqueta la señal entera (menos `warmup` al inicio) por umbral de kappa."""
    if "kappa" not in attrs:
        log.warning("Grupo '%s' sin attr 'kappa' — se deja sin etiquetar", grp_name)
        return []
    label = "stable" if attrs["kappa"] < threshold else "unstable"
    return [(t_range[0] + warmup, t_range[1], label)]


# Punto de extensión: sumar acá una estrategia nueva (ej. "por aplicación", a
# definir más adelante) sin tocar make_label_template.
LABEL_STRATEGIES = {
    "manual": _label_manual,
    "kappa":  _label_by_kappa,
}


def _format_intervals(intervals: List[Tuple[float, float, str]]) -> str:
    if not intervals:
        return "[]"
    parts = ", ".join(f'[{t0}, {t1}, "{label}"]' for t0, t1, label in intervals)
    return f"[{parts}]"


def _discover_channels(grp) -> List[str]:
    """Subgrupos de `grp` que son canales de señal (tienen dataset 'time' y 'values')."""
    return [
        key for key in grp.keys()
        if isinstance(grp[key], h5py.Group) and "time" in grp[key] and "values" in grp[key]
    ]


def make_label_template(h5_path: str, out_yaml: str, strategy: str = "manual", **strategy_kwargs) -> None:
    """Genera `out_yaml` con todos los grupos de `h5_path`.

    `strategy` decide el primer pase de etiquetado ("manual" = todo vacío,
    el default de siempre); el YAML resultante sigue siendo editable a mano
    después, sea cual sea la estrategia usada para generarlo.

    Se niega a sobrescribir un YAML ya existente, para no perder etiquetas
    hechas a mano.
    """
    if os.path.exists(out_yaml):
        raise FileExistsError(
            f"{out_yaml} ya existe — no se sobrescribe (podrías perder etiquetas hechas a mano)."
        )
    label_fn = LABEL_STRATEGIES[strategy]

    lines = [f"source: {os.path.basename(h5_path)}", "cases:"]
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = dict(grp.attrs)
            kappa_bits = {k: v for k, v in attrs.items() if str(k).startswith("kappa")}

            case_channels = _discover_channels(grp)
            t_range = None
            if case_channels:
                t = grp[case_channels[0]]["time"]
                t_range = (float(t[0]), float(t[-1]))

            intervals = label_fn(grp_name, attrs, t_range, **strategy_kwargs) if t_range is not None else []

            comment_bits = []
            if kappa_bits:
                comment_bits.append(", ".join(f"{k}={v}" for k, v in kappa_bits.items()))
            if t_range is not None:
                comment_bits.append(f"t=[{t_range[0]:.2f}, {t_range[1]:.2f}] s")
            comment = f"  # {'   '.join(comment_bits)}" if comment_bits else ""

            lines.append(f"  {grp_name}: {_format_intervals(intervals)}{comment}")

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

def from_doe_h5(h5_path: str, labels_path: str, channels: Optional[List[str]] = None) -> ReferenceDataset:
    """Construye un ReferenceDataset a partir de un doe_results.h5 + su YAML de etiquetas.

    `channels`: lista explícita de canales a usar, o None (default) para
    incluir TODOS los canales disponibles de cada caso (autodetectados) —
    qué canal usar queda para la Fase 3, acá se guardan todos.

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
            case_channels = channels if channels is not None else _discover_channels(grp)

            for ch in case_channels:
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

        # 2b. estrategia "kappa": stable / unstable / sin kappa (-> [] + warning, no error)
        kappa_h5 = os.path.join(tmp, "kappa_doe.h5")
        kappa_yaml = os.path.join(tmp, "kappa_labels.yaml")
        with h5py.File(kappa_h5, "w") as f:
            grp = f.create_group("case_low")
            grp.attrs["kappa"] = 0.5
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

            grp = f.create_group("case_high")
            grp.attrs["kappa"] = 1.5
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

            grp = f.create_group("case_no_kappa")
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

        make_label_template(kappa_h5, kappa_yaml, strategy="kappa", threshold=1.0, warmup=0.5)
        kappa_cases = _parse_labels_file(kappa_yaml)
        assert kappa_cases["case_low"] == [(0.5, 10.0, "stable")], kappa_cases["case_low"]
        assert kappa_cases["case_high"] == [(0.5, 10.0, "unstable")], kappa_cases["case_high"]
        assert kappa_cases["case_no_kappa"] == [], kappa_cases["case_no_kappa"]

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

        # 4b. from_doe_h5 sin channels (default None) -> autodetecta TODOS los canales del caso
        multi_h5 = os.path.join(tmp, "multi_ch.h5")
        multi_yaml = os.path.join(tmp, "multi_labels.yaml")
        with h5py.File(multi_h5, "w") as f:
            grp = f.create_group("case_a")
            for ch in ("Axial_vel", "Axial_disp"):
                sub = grp.create_group(ch)
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=t)
            grp.create_dataset("res_R_p", data=np.array([1.0]))  # no es canal (no es grupo time/values)
        with open(multi_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": "x", "cases": {"case_a": [[0.0, 10.0, "stable"]]}}, f)

        ds_multi = from_doe_h5(multi_h5, multi_yaml)  # channels=None -> autodetecta
        assert {s.id for s in ds_multi.signals} == {"case_a/Axial_vel", "case_a/Axial_disp"}

        ds_restricted = from_doe_h5(multi_h5, multi_yaml, channels=["Axial_vel"])
        assert {s.id for s in ds_restricted.signals} == {"case_a/Axial_vel"}

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
    p_template.add_argument("h5_path", nargs="?", default=DEFAULT_H5_PATH)
    p_template.add_argument("out_yaml", nargs="?", default=DEFAULT_LABELS_PATH)
    p_template.add_argument("--strategy", choices=sorted(LABEL_STRATEGIES), default=DEFAULT_STRATEGY)
    p_template.add_argument("--kappa-threshold", type=float, default=DEFAULT_KAPPA_THRESHOLD)
    p_template.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)

    p_build = sub.add_parser("build", help="Construye y guarda un ReferenceDataset")
    p_build.add_argument("h5_path", nargs="?", default=DEFAULT_H5_PATH)
    p_build.add_argument("labels_yaml", nargs="?", default=DEFAULT_LABELS_PATH)
    p_build.add_argument("out_h5", nargs="?", default=DEFAULT_OUT_H5)
    p_build.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)

    args = parser.parse_args()

    if args.cmd == "selftest":
        _self_test()
        return

    if args.cmd in ("template", "build") and args.h5_path is None:
        parser.error(
            "falta h5_path — pasalo como argumento o fijá DEFAULT_H5_PATH arriba del script"
        )

    # out_yaml/labels_yaml/out_h5 no pasados por CLI -> misma carpeta que h5_path
    h5_dir = os.path.dirname(os.path.abspath(args.h5_path))

    if args.cmd == "template":
        out_yaml = args.out_yaml or os.path.join(h5_dir, "reference_labels.yaml")
        kwargs = {"threshold": args.kappa_threshold, "warmup": args.warmup} if args.strategy == "kappa" else {}
        make_label_template(args.h5_path, out_yaml, strategy=args.strategy, **kwargs)
        print(f"Plantilla escrita en {out_yaml}")
    elif args.cmd == "build":
        labels_yaml = args.labels_yaml or os.path.join(h5_dir, "reference_labels.yaml")
        out_h5 = args.out_h5 or os.path.join(h5_dir, "reference_dataset.h5")
        ds = from_doe_h5(args.h5_path, labels_yaml, channels=args.channels)
        ds.to_hdf5(out_h5)
        print(f"{len(ds.signals)} señales -> {out_h5}")


if __name__ == "__main__":
    _main()

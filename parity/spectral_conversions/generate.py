# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""External-reference parity fixture of the spectral conversion methods.

Every case runs one n4m estimator role and one external, independently written
reference on the same input, then records the reference output in
``parity/fixtures/spectral_conversions_external_v1.json``. The Python binding
test ``bindings/python/tests/test_spectral_conversions.py`` replays the fixture
without the references installed.

References (none of them reuses n4m code):

* photobiology (R) ``T2A`` / ``A2T``: absorbance <-> transmittance.
* mdatools (R) ``prep.ref2km``: Kubelka-Munk function.
* stats::uniroot (R) over ``prep.ref2km``: inverse Kubelka-Munk by bracketing.
* astropy ``units``: percent <-> fraction, spectral() and spectral_density().
* scipy ``interpolate``: linear / nearest ``interp1d`` and not-a-knot ``CubicSpline``.
* specdal (Python) ``operator.jump_correct``: additive splice correction.
* spectrolab (R) ``match_sensors``: multiplicative tapered splice correction.
* prospectr (R) ``readASD`` and asdreader (R) ``get_spectra``: reflectance of
  real ASD files (MIT-licensed test files of pyASDReader and prospectr) from
  their target and white-reference spectra.

Usage (needs the references, an n4m build and the ASD files of nirs4all-formats)::

    N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.<abi> \
    PYTHONPATH=bindings/python/src \
    python parity/spectral_conversions/generate.py --formats-samples ../nirs4all-formats/samples \
        [--rscript Rscript] [--check]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "parity" / "fixtures" / "spectral_conversions_external_v1.json"
REPORT = REPO / "parity" / "results" / "spectral_conversions_external.json"
ASD_FILES = {
    # file: (origin, licence)
    "v6sample00000.asd": ("KaiTastic/pyASDReader tests/sample_data", "MIT"),
    "v7sample00000.asd": ("KaiTastic/pyASDReader tests/sample_data", "MIT"),
    "v8sample00001.asd": ("KaiTastic/pyASDReader tests/sample_data", "MIT"),
}


def _hex(values: np.ndarray) -> list:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [float(v).hex() for v in array]
    return [_hex(row) for row in array]


def _unhex(values: Any) -> np.ndarray:
    special = {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan, "NA": math.nan}

    def one(text: str) -> float:
        return special[text] if text in special else float.fromhex(text)

    if isinstance(values, str):
        return np.array(one(values))
    if values and isinstance(values[0], list):
        return np.array([[one(v) for v in row] for row in values])
    return np.array([one(v) for v in values])


def build_inputs(rng: np.random.Generator) -> dict[str, Any]:
    """Deterministic inputs; grids avoid exact interpolation ties."""

    fraction = rng.uniform(0.005, 1.0, size=(6, 40))
    fraction[0, :4] = [1.0, 0.5, 0.1, 1e-3]
    absorbance = rng.uniform(0.0, 3.0, size=(6, 40))
    absorbance[0, :3] = [0.0, 1.0, 2.0]
    reflectance = rng.uniform(1e-4, 0.9999, size=(6, 40))
    reflectance[0, :3] = [0.5, 0.25, 0.9]
    source_axis = np.sort(1000.0 + 1500.0 * rng.random(120))
    source_axis[0], source_axis[-1] = 1000.0, 2500.0
    spectra = np.cumsum(rng.normal(0.0, 0.01, size=(5, 120)), axis=1) + 0.5
    target_axis = np.sort(rng.uniform(1001.0, 2499.0, 70))
    nm_axis = np.sort(rng.uniform(800.0, 2600.0, 60))
    density = rng.uniform(0.1, 5.0, size=(3, 60))
    # ASD-like detectors 350-1000, 1001-1800, 1801-2500 nm with level jumps.
    asd_axis = np.arange(350.0, 2501.0)
    base = (
        0.3
        + 0.2 * np.sin(asd_axis / 300.0)[None, :]
        + rng.normal(0, 0.002, (4, asd_axis.size))
    )
    jumps = np.where(asd_axis <= 1000, 0.97, np.where(asd_axis <= 1800, 1.0, 1.06))
    detectors = base * jumps + np.where(asd_axis <= 1000, -0.01, 0.0)
    return {
        "asd_axis": asd_axis,
        "detectors": detectors,
        "fraction": fraction,
        "absorbance": absorbance,
        "reflectance": reflectance,
        "source_axis": source_axis,
        "spectra": spectra,
        "target_axis": target_axis,
        "nm_axis": nm_axis,
        "density": density,
    }


def r_references(rscript: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "jobs.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(jobs), encoding="utf-8")
        subprocess.run(
            [
                rscript,
                "--vanilla",
                str(Path(__file__).with_name("references.R")),
                str(src),
                str(dst),
            ],
            check=True,
        )
        return json.loads(dst.read_text(encoding="utf-8"))


def python_versions() -> dict[str, str]:
    import astropy
    import pandas
    import scipy
    import specdal

    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "specdal": getattr(specdal, "__version__", "0.2.1"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
    }


def cases(
    inputs: dict[str, Any], samples: Path, rscript: str
) -> tuple[list[dict], dict]:
    import astropy.units as u
    from astropy.constants import c as speed_of_light
    from scipy.interpolate import CubicSpline, interp1d

    out: list[dict] = []

    def case(name, method, params, X, expected, reference, tol, axis=None):
        out.append(
            {
                "name": name,
                "method_id": method,
                "params": params,
                "X": _hex(X),
                "axis": None if axis is None else _hex(axis),
                "expected": _hex(expected),
                "reference": reference,
                "rtol": tol[0],
                "atol": tol[1],
            }
        )

    asd_paths = [samples / "asd" / name for name in ASD_FILES]
    missing = [str(p) for p in asd_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing ASD files: {missing}")
    f, a, r = inputs["fraction"], inputs["absorbance"], inputs["reflectance"]
    km_inputs = np.array([[(1 - v) ** 2 / (2 * v) for v in row] for row in r])
    jobs = [
        {"kind": "T2A", "X": f.tolist()},
        {"kind": "A2T", "X": a.tolist()},
        {"kind": "ref2km", "X": r.tolist()},
        {"kind": "inverse_km", "X": km_inputs.tolist()},
    ]
    asd_axis, detectors = inputs["asd_axis"], inputs["detectors"]
    jobs.append(
        {
            "kind": "match_sensors",
            "X": detectors.tolist(),
            "axis": asd_axis.tolist(),
            "splices": [1001.0, 1801.0],
            "window": 5.0,
        }
    )
    for path in asd_paths:
        jobs.append({"kind": "prospectr_asd", "path": str(path)})
        jobs.append({"kind": "asdreader_asd", "path": str(path)})
    r_out = r_references(rscript, jobs)
    results = r_out["results"]
    t2a, a2t, ref2km, inv_km, matched = (_unhex(results[i]) for i in range(5))

    sc = "preprocessing.signal_conversion."
    case(
        "to_absorbance/fraction",
        sc + "to_absorbance",
        {"is_percent": False},
        f,
        t2a,
        "photobiology::T2A",
        (1e-13, 1e-15),
    )
    case(
        "to_absorbance/percent",
        sc + "to_absorbance",
        {"is_percent": True},
        100.0 * f,
        t2a,
        "photobiology::T2A on the fraction",
        (1e-13, 1e-15),
    )
    case(
        "from_absorbance/fraction",
        sc + "from_absorbance",
        {"is_percent": False},
        a,
        a2t,
        "photobiology::A2T",
        (1e-13, 1e-16),
    )
    case(
        "from_absorbance/percent",
        sc + "from_absorbance",
        {"is_percent": True},
        a,
        100.0 * a2t,
        "photobiology::A2T times 100",
        (1e-13, 1e-14),
    )
    case(
        "kubelka_munk/fraction",
        sc + "kubelka_munk",
        {"is_percent": False},
        r,
        ref2km,
        "mdatools::prep.ref2km",
        (1e-13, 1e-16),
    )
    case(
        "kubelka_munk/percent",
        sc + "kubelka_munk",
        {"is_percent": True},
        100.0 * r,
        ref2km,
        "mdatools::prep.ref2km on the fraction",
        (1e-13, 1e-16),
    )
    case(
        "from_kubelka_munk/fraction",
        sc + "from_kubelka_munk",
        {"is_percent": False},
        km_inputs,
        inv_km,
        "stats::uniroot over mdatools::prep.ref2km",
        (1e-12, 1e-16),
    )
    case(
        "from_kubelka_munk/percent",
        sc + "from_kubelka_munk",
        {"is_percent": True},
        km_inputs,
        100.0 * inv_km,
        "stats::uniroot over mdatools::prep.ref2km, times 100",
        (1e-12, 1e-14),
    )
    pct = np.asarray((f * 100.0 * u.percent).to_value(u.dimensionless_unscaled))
    case(
        "percent_to_fraction",
        sc + "percent_to_fraction",
        {},
        100.0 * f,
        pct,
        "astropy.units percent -> dimensionless",
        (1e-15, 0.0),
    )
    case(
        "fraction_to_percent",
        sc + "fraction_to_percent",
        {},
        f,
        np.asarray((f * u.dimensionless_unscaled).to_value(u.percent)),
        "astropy.units dimensionless -> percent",
        (1e-15, 0.0),
    )

    for k in range(len(asd_paths)):
        name = asd_paths[k].name
        prospectr, asdreader = results[5 + 2 * k], results[6 + 2 * k]
        radiance, reference = (
            _unhex(prospectr["radiance"]),
            _unhex(prospectr["reference"]),
        )
        expected = _unhex(prospectr["reflectance"])
        case(
            f"reflectance_from_references/prospectr/{name}",
            sc + "reflectance_from_references",
            {"white_reference": _hex(reference)},
            radiance[None, :],
            expected[None, :],
            "prospectr::readASD reflectance",
            (1e-13, 1e-16),
        )
        radiance, reference = (
            _unhex(asdreader["radiance"]),
            _unhex(asdreader["reference"]),
        )
        expected = _unhex(asdreader["reflectance"])
        case(
            f"reflectance_from_references/asdreader/{name}",
            sc + "reflectance_from_references",
            {"white_reference": _hex(reference)},
            radiance[None, :],
            expected[None, :],
            "asdreader::get_spectra reflectance",
            (1e-13, 1e-16),
        )

    import pandas as pd
    from specdal.operator import jump_correct

    for reference in (0, 1, 2):
        specdal = np.array(
            [
                jump_correct(
                    pd.Series(row, index=asd_axis), [1000.0, 1800.0], reference
                ).to_numpy()
                for row in detectors
            ]
        )
        case(
            f"splice_correction/specdal_additive/reference_{reference}",
            sc + "splice_correction",
            {
                "splices": _hex(np.array([1000.5, 1800.5])),
                "reference_segment": reference,
                "method": "additive",
            },
            detectors,
            specdal,
            "specdal.operator.jump_correct (additive)",
            (1e-13, 1e-15),
            axis=asd_axis,
        )
    case(
        "splice_correction/spectrolab_match_sensors",
        sc + "splice_correction",
        {
            "splices": _hex(np.array([1001.0, 1801.0])),
            "reference_segment": 1,
            "method": "multiplicative",
            "window": 5.0,
            "taper": True,
        },
        detectors,
        matched,
        "spectrolab::match_sensors (fixed sensor 2, interpolate_wvl 5)",
        (1e-13, 1e-15),
        axis=asd_axis,
    )

    src, Y, tgt = inputs["source_axis"], inputs["spectra"], inputs["target_axis"]
    for method, ref in [
        ("linear", interp1d(src, Y, kind="linear", axis=1)(tgt)),
        ("nearest", interp1d(src, Y, kind="nearest", axis=1)(tgt)),
        ("cubic", CubicSpline(src, Y, axis=1, bc_type="not-a-knot")(tgt)),
    ]:
        tol = (1e-13, 1e-15) if method != "cubic" else (1e-11, 1e-13)
        case(
            f"resampler/{method}",
            "preprocessing.resampling.resampler",
            {"target_wavelengths": _hex(tgt), "method": method},
            Y,
            ref,
            f"scipy.interpolate {'CubicSpline not-a-knot' if method == 'cubic' else 'interp1d ' + method}",
            tol,
            axis=src,
        )

    nm, density = inputs["nm_axis"], inputs["density"]
    units = {"nm": u.nm, "um": u.um, "cm-1": u.cm**-1}
    axes = {"nm": nm, "um": nm / 1000.0, "cm-1": 1e7 / nm[::-1]}
    for source in units:
        for target in units:
            if source == target:
                continue
            x = axes[source]
            converted = (x * units[source]).to_value(
                units[target], equivalencies=u.spectral()
            )
            order = np.argsort(converted, kind="stable")
            # The converted axis is checked as the first row of the output.
            case(
                f"axis_unit_conversion/{source}->{target}/axis",
                "preprocessing.resampling.axis_unit_conversion",
                {"source_unit": source, "target_unit": target},
                x[None, :],
                x[None, order],
                "astropy.units spectral() (column order)",
                (0.0, 0.0),
                axis=x,
            )
            out[-1]["expected_axis"] = _hex(converted[order])
            out[-1]["axis_rtol"] = 1e-15
            # Densities per source unit become densities per target unit.
            flux = density * u.W / u.m**2 / units[source]
            per_hz = u.W / u.m**2 / u.Hz
            c_cm = speed_of_light.to(u.cm / u.s)
            # astropy has no per-wavenumber density: go through f_nu with
            # d nu / d k = c (nu in Hz, k in cm-1).
            if source == "cm-1":
                f_nu = (flux / c_cm).to(per_hz)
                expected = f_nu.to_value(
                    u.W / u.m**2 / units[target],
                    equivalencies=u.spectral_density(x * units[source]),
                )
            elif target == "cm-1":
                f_nu = flux.to(
                    per_hz, equivalencies=u.spectral_density(x * units[source])
                )
                expected = (f_nu * c_cm).to_value(u.W / u.m**2 * u.cm)
            else:
                expected = flux.to_value(u.W / u.m**2 / units[target])
            expected = np.asarray(expected)
            case(
                f"axis_unit_conversion/{source}->{target}/density",
                "preprocessing.resampling.axis_unit_conversion",
                {
                    "source_unit": source,
                    "target_unit": target,
                    "spectral_density": True,
                },
                density,
                expected[:, order],
                "astropy.units spectral_density()",
                (1e-12, 0.0),
                axis=x,
            )
    versions = {**python_versions(), **r_out["versions"]}
    return out, versions


def run_n4m(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    from n4m import roles

    params = {}
    for key, value in entry["params"].items():
        params[key] = (
            [float.fromhex(v) for v in value] if isinstance(value, list) else value
        )
    cls = next(
        c
        for c in roles.__dict__.values()
        if isinstance(c, type) and getattr(c, "_method_id", None) == entry["method_id"]
    )
    X = _unhex(entry["X"])
    est = cls(**params)
    kwargs = {} if entry["axis"] is None else {"axis": _unhex(entry["axis"])}
    est.fit(X, **kwargs)
    axis = None
    if entry["method_id"].endswith("axis_unit_conversion"):
        axis = np.asarray(est.fit_result_["axis"]).ravel()
    return est.transform(X), axis


def compare(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for entry in entries:
        got, axis = run_n4m(entry)
        want = _unhex(entry["expected"])
        diff = np.abs(got - want)
        scale = np.abs(want)
        rel = float(
            np.max(np.where(scale > 0, diff / np.where(scale > 0, scale, 1), diff))
        )
        ok = bool(np.all(diff <= entry["atol"] + entry["rtol"] * scale))
        row = {
            "name": entry["name"],
            "reference": entry["reference"],
            "max_abs": float(diff.max()),
            "max_rel": rel,
            "ok": ok,
        }
        if axis is not None and "expected_axis" in entry:
            want_axis = _unhex(entry["expected_axis"])
            axis_rel = float(np.max(np.abs(axis - want_axis) / want_axis))
            row["axis_max_rel"] = axis_rel
            row["ok"] = ok and axis_rel <= entry["axis_rtol"]
        report.append(row)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats-samples", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed fixture instead of writing it",
    )
    args = parser.parse_args()
    entries, versions = cases(
        build_inputs(np.random.default_rng(20261126)),
        args.formats_samples,
        args.rscript,
    )
    doc = {
        "format": "n4m_spectral_conversions_external_v1",
        "encoding": "float.hex",
        "versions": versions,
        "asd_sources": ASD_FILES,
        "cases": entries,
    }
    report = compare(entries)
    failed = [row for row in report if not row["ok"]]
    for row in report:
        flag = "ok  " if row["ok"] else "FAIL"
        extra = f" axis_rel={row['axis_max_rel']:.2e}" if "axis_max_rel" in row else ""
        print(
            f"{flag} {row['name']:<58} rel={row['max_rel']:.2e} abs={row['max_abs']:.2e}{extra}"
        )
    if args.check:
        committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
        same = committed["cases"] == entries
        print(
            "fixture matches the references"
            if same
            else "fixture differs from the references"
        )
        return 0 if same and not failed else 1
    FIXTURE.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    REPORT.write_text(
        json.dumps({"versions": versions, "cases": report}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE.relative_to(REPO)} and {REPORT.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

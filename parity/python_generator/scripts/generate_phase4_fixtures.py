#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 4 parity fixtures for the stateless derivative / smoothing
operators.

Five operators, all stateless (`_create / _transform / _destroy`):

  * SavitzkyGolay   — `scipy.signal.savgol_filter` parity.  Five modes
                      (mirror, constant, nearest, wrap, interp) crossed with
                      a few (window_length, polyorder, deriv) tuples.
  * FirstDerivative — `np.gradient(X, delta, axis=1, edge_order=2)` parity.
  * SecondDerivative— two passes of `np.gradient` parity.
  * NorrisWilliams  — nirs4all.norris_williams parity (segment-smooth then
                      gap-difference, applied `derivative_order` times).
  * Gaussian        — `scipy.ndimage.gaussian_filter1d` parity.  Five modes
                      (reflect, constant, nearest, mirror, wrap) crossed
                      with a couple (sigma, order) combinations.

All fixtures share the same input matrix shape (50 x 200) generated with the
SEED used by the Phase 2/3 fixture generators (lets us reuse the synthesizer
helper unchanged).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal


NIRS4ALL_ROOT = Path("/home/delete/nirs4all/nirs4all/nirs4all")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nw_mod = _load(NIRS4ALL_ROOT / "operators/transforms/norris_williams.py",
               "_n4a_norris_williams_phase4")


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra (same generator family as Phase 2/3).
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 50
COLS = 200


def synthesize_spectra(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.2, 0.6)
        b = rng.uniform(-0.3, 0.3)
        c = rng.uniform(-0.4, 0.4)
        baseline = a + b * wl + c * wl * wl
        peak1_mu = rng.uniform(0.2, 0.4)
        peak1_sigma = rng.uniform(0.02, 0.06)
        peak1_amp = rng.uniform(0.3, 1.2)
        peak2_mu = rng.uniform(0.6, 0.85)
        peak2_sigma = rng.uniform(0.03, 0.08)
        peak2_amp = rng.uniform(0.2, 0.9)
        peaks = (
            peak1_amp * np.exp(-((wl - peak1_mu) ** 2) / (2.0 * peak1_sigma ** 2))
            + peak2_amp * np.exp(-((wl - peak2_mu) ** 2) / (2.0 * peak2_sigma ** 2))
        )
        noise = rng.normal(0.0, 0.01, size=COLS)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Reference implementations.
# ---------------------------------------------------------------------------

def ref_savgol(X: np.ndarray, *, window_length: int, polyorder: int,
               deriv: int, delta: float, mode: str,
               cval: float) -> np.ndarray:
    return np.asarray(
        signal.savgol_filter(X, window_length, polyorder,
                              deriv=deriv, delta=delta, axis=1,
                              mode=mode, cval=cval)
    )


def ref_first_derivative(X: np.ndarray, *, delta: float,
                          edge_order: int) -> np.ndarray:
    return np.asarray(np.gradient(X, delta, axis=1, edge_order=edge_order))


def ref_second_derivative(X: np.ndarray, *, delta: float,
                           edge_order: int) -> np.ndarray:
    d1 = np.gradient(X, delta, axis=1, edge_order=edge_order)
    return np.asarray(np.gradient(d1, delta, axis=1, edge_order=edge_order))


def ref_norris_williams(X: np.ndarray, *, segment: int, gap: int,
                         derivative_order: int, delta: float) -> np.ndarray:
    return np.asarray(
        nw_mod.norris_williams(X, gap=gap, segment=segment,
                                deriv=derivative_order, delta=delta)
    )


def ref_gaussian(X: np.ndarray, *, sigma: float, order: int,
                  mode: str, cval: float, truncate: float) -> np.ndarray:
    return np.asarray(
        ndimage.gaussian_filter1d(X, sigma=sigma, order=order,
                                   axis=1, mode=mode, cval=cval,
                                   truncate=truncate)
    )


# ---------------------------------------------------------------------------
# Case lists.
# ---------------------------------------------------------------------------

def savgol_cases(X: np.ndarray):
    """SG sweep: 4 (window, polyorder, deriv) tuples x 5 boundary modes plus
    a couple `delta != 1` checks and `polyorder > deriv` short circuit
    test."""
    base_tuples = [
        (5, 2, 0),        # smoothing only
        (11, 3, 0),       # wider smoothing
        (7, 2, 1),        # first derivative
        (9, 3, 2),        # second derivative
    ]
    modes = ["mirror", "constant", "nearest", "wrap", "interp"]
    out: list[tuple[str, dict, Callable[[], np.ndarray]]] = []
    for w, p, d in base_tuples:
        for m in modes:
            cv = 0.0 if m != "constant" else 0.5
            name = f"w{w}_p{p}_d{d}_{m}"
            params = {
                "window_length": w, "polyorder": p,
                "deriv": d, "delta": 1.0,
                "mode": m, "cval": cv,
            }
            out.append((name, params,
                         (lambda X=X, w=w, p=p, d=d, m=m, cv=cv: ref_savgol(
                             X, window_length=w, polyorder=p, deriv=d,
                             delta=1.0, mode=m, cval=cv))))
    # Non-unit delta
    out.append(("w7_p2_d1_delta0p5_mirror",
                {"window_length": 7, "polyorder": 2, "deriv": 1,
                 "delta": 0.5, "mode": "mirror", "cval": 0.0},
                lambda X=X: ref_savgol(X, window_length=7, polyorder=2,
                                        deriv=1, delta=0.5, mode="mirror",
                                        cval=0.0)))
    # deriv > polyorder => zeros (scipy short-circuit)
    out.append(("w5_p2_d3_mirror_zero",
                {"window_length": 5, "polyorder": 2, "deriv": 3,
                 "delta": 1.0, "mode": "mirror", "cval": 0.0},
                lambda X=X: ref_savgol(X, window_length=5, polyorder=2,
                                        deriv=3, delta=1.0, mode="mirror",
                                        cval=0.0)))
    return out


def first_derivative_cases(X: np.ndarray):
    return [
        ("delta1_edge2", {"delta": 1.0, "edge_order": 2},
         lambda X=X: ref_first_derivative(X, delta=1.0, edge_order=2)),
        ("delta1_edge1", {"delta": 1.0, "edge_order": 1},
         lambda X=X: ref_first_derivative(X, delta=1.0, edge_order=1)),
        ("delta2_edge2", {"delta": 2.0, "edge_order": 2},
         lambda X=X: ref_first_derivative(X, delta=2.0, edge_order=2)),
        ("delta0p5_edge2", {"delta": 0.5, "edge_order": 2},
         lambda X=X: ref_first_derivative(X, delta=0.5, edge_order=2)),
    ]


def second_derivative_cases(X: np.ndarray):
    return [
        ("delta1_edge2", {"delta": 1.0, "edge_order": 2},
         lambda X=X: ref_second_derivative(X, delta=1.0, edge_order=2)),
        ("delta1_edge1", {"delta": 1.0, "edge_order": 1},
         lambda X=X: ref_second_derivative(X, delta=1.0, edge_order=1)),
        ("delta2_edge2", {"delta": 2.0, "edge_order": 2},
         lambda X=X: ref_second_derivative(X, delta=2.0, edge_order=2)),
        ("delta0p5_edge2", {"delta": 0.5, "edge_order": 2},
         lambda X=X: ref_second_derivative(X, delta=0.5, edge_order=2)),
    ]


def norris_williams_cases(X: np.ndarray):
    return [
        ("seg5_gap5_d1", {"segment": 5, "gap": 5, "derivative_order": 1,
                          "delta": 1.0},
         lambda X=X: ref_norris_williams(X, segment=5, gap=5,
                                          derivative_order=1, delta=1.0)),
        ("seg7_gap3_d1", {"segment": 7, "gap": 3, "derivative_order": 1,
                          "delta": 1.0},
         lambda X=X: ref_norris_williams(X, segment=7, gap=3,
                                          derivative_order=1, delta=1.0)),
        ("seg5_gap5_d2", {"segment": 5, "gap": 5, "derivative_order": 2,
                          "delta": 1.0},
         lambda X=X: ref_norris_williams(X, segment=5, gap=5,
                                          derivative_order=2, delta=1.0)),
        ("seg1_gap5_d1", {"segment": 1, "gap": 5, "derivative_order": 1,
                          "delta": 1.0},
         lambda X=X: ref_norris_williams(X, segment=1, gap=5,
                                          derivative_order=1, delta=1.0)),
        ("seg9_gap7_d2_delta0p5",
         {"segment": 9, "gap": 7, "derivative_order": 2, "delta": 0.5},
         lambda X=X: ref_norris_williams(X, segment=9, gap=7,
                                          derivative_order=2, delta=0.5)),
    ]


def gaussian_cases(X: np.ndarray):
    tuples = [(1.0, 0), (2.0, 0), (1.5, 1), (1.0, 2), (2.5, 2)]
    modes = ["reflect", "constant", "nearest", "mirror", "wrap"]
    out: list[tuple[str, dict, Callable[[], np.ndarray]]] = []
    for s, o in tuples:
        for m in modes:
            cv = 0.0 if m != "constant" else 0.25
            name = f"sigma{str(s).replace('.', 'p')}_o{o}_{m}"
            params = {"sigma": s, "order": o, "mode": m,
                      "cval": cv, "truncate": 4.0}
            out.append((name, params,
                         (lambda X=X, s=s, o=o, m=m, cv=cv: ref_gaussian(
                             X, sigma=s, order=o, mode=m, cval=cv,
                             truncate=4.0))))
    # Non-default truncate
    out.append(("sigma1p0_o0_reflect_truncate3",
                {"sigma": 1.0, "order": 0, "mode": "reflect",
                 "cval": 0.0, "truncate": 3.0},
                lambda X=X: ref_gaussian(X, sigma=1.0, order=0,
                                          mode="reflect", cval=0.0,
                                          truncate=3.0)))
    return out


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, op_id: str, X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                  out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y = fn()
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
            "output_hex": array_to_hex(Y),
        })
    out = out_dir / f"{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    n4a_init = NIRS4ALL_ROOT / "__init__.py"
    version = "unknown"
    for line in n4a_init.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

    X = synthesize_spectra(SEED)
    print(f"X shape {X.shape}, range [{X.min():.4f}, {X.max():.4f}]")

    write_fixture("savgol", "savgol", X, savgol_cases(X), out_dir, version)
    write_fixture("first_derivative", "first_derivative", X,
                  first_derivative_cases(X), out_dir, version)
    write_fixture("second_derivative", "second_derivative", X,
                  second_derivative_cases(X), out_dir, version)
    write_fixture("norris_williams", "norris_williams", X,
                  norris_williams_cases(X), out_dir, version)
    write_fixture("gaussian", "gaussian", X, gaussian_cases(X), out_dir,
                   version)


if __name__ == "__main__":
    main()

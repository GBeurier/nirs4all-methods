#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 7 parity fixtures for the signal-conversion preprocessing
operators.

Five stateless operators, all pure closed-form arithmetic:

  * ToAbsorbance      — A = -log10(R), epsilon clamp.
  * FromAbsorbance    — R = 10**(-A).
  * PercentToFraction — X / 100.0.
  * FractionToPercent — X * 100.0.
  * KubelkaMunk       — F(R) = (1 - R)^2 / (2 R), symmetric clamp.

Fixture schema (Phase 2 schema, one input matrix per fixture):

```json
{
  "format": "n4m_pp_<op>_v1",
  "numpy_version": "...",
  "nirs4all_version": "0.8.x",
  "encoding": "ieee754_binary64_be_hex",
  "rows": 50, "cols": 200,
  "input_hex": [...],
  "cases": [
    {"name": "...", "params": {...}, "output_hex": [...]},
    ...
  ]
}
```

For percent-input cases (`is_percent: true`) the same fixture file holds the
**fractional-domain** input matrix; the C test scales it by 100 before
feeding to the engine, and the Python reference is invoked on the scaled
matrix as well. The scale-by-100 is bit-exact (100.0 is exactly
representable in binary64), so this is byte-identical to using a separate
percent-domain matrix.

The n4m side decodes the hex back to f64 and compares with a 1e-12 abs /
1e-13 rel tolerance because every operator here is closed-form pure
arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from parity.nirs4all_source import find_nirs4all_root, get_nirs4all_version  # noqa: E402


# ---------------------------------------------------------------------------
# Load nirs4all reference module directly from source.
#
# nirs4all's __init__ pulls in heavy optional dependencies; bypass it by
# loading the operator modules in isolation via importlib so the fixture
# generator only depends on numpy + sklearn.
# ---------------------------------------------------------------------------

NIRS4ALL_ROOT = find_nirs4all_root()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# signal_conversion.py imports nirs4all.data.signal_type — load that first
# so the absolute import resolves without pulling in the full nirs4all
# package.
signal_type_mod = _load(NIRS4ALL_ROOT / "data/signal_type.py",
                         "nirs4all.data.signal_type")
sig_mod = _load(NIRS4ALL_ROOT / "operators/transforms/signal_conversion.py",
                "_n4a_signal_conversion_lite")


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic spectra.
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 50
COLS = 200


def synthesize_reflectance(seed: int) -> np.ndarray:
    """50 × 200 reflectance-like matrix in roughly [0, 1]. A few values land
    just outside [epsilon, 1 - epsilon] so the ToAbsorbance and KubelkaMunk
    clamp branches are exercised."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.15, 0.85)
        b = rng.uniform(-0.2, 0.2)
        c = rng.uniform(-0.25, 0.25)
        baseline = a + b * wl + c * wl * wl
        p1_mu = rng.uniform(0.25, 0.45)
        p1_sigma = rng.uniform(0.03, 0.07)
        p1_amp = rng.uniform(0.05, 0.25)
        p2_mu = rng.uniform(0.6, 0.85)
        p2_sigma = rng.uniform(0.03, 0.08)
        p2_amp = rng.uniform(0.05, 0.2)
        dips = (
            p1_amp * np.exp(-((wl - p1_mu) ** 2) / (2.0 * p1_sigma ** 2))
            + p2_amp * np.exp(-((wl - p2_mu) ** 2) / (2.0 * p2_sigma ** 2))
        )
        noise = rng.normal(0.0, 0.005, size=COLS)
        X[i] = baseline - dips + noise
    return X


def synthesize_absorbance(seed: int) -> np.ndarray:
    """50 × 200 absorbance-like matrix in roughly [0, 3]."""
    rng = np.random.default_rng(seed + 1)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.05, 0.4)
        b = rng.uniform(-0.1, 0.3)
        c = rng.uniform(-0.15, 0.2)
        baseline = a + b * wl + c * wl * wl
        p1_mu = rng.uniform(0.2, 0.4)
        p1_sigma = rng.uniform(0.03, 0.06)
        p1_amp = rng.uniform(0.4, 1.5)
        p2_mu = rng.uniform(0.6, 0.85)
        p2_sigma = rng.uniform(0.03, 0.08)
        p2_amp = rng.uniform(0.3, 1.2)
        peaks = (
            p1_amp * np.exp(-((wl - p1_mu) ** 2) / (2.0 * p1_sigma ** 2))
            + p2_amp * np.exp(-((wl - p2_mu) ** 2) / (2.0 * p2_sigma ** 2))
        )
        noise = rng.normal(0.0, 0.005, size=COLS)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Case lists.
#
# Each case carries:
#   - a name (used in test labels),
#   - a params dict (forwarded to the n4m wrapper as integer / double flags),
#   - a Callable producing the reference output.
#
# Parameter schemas mirror the wrapper signatures exactly:
#   to_absorbance:    {"is_percent": bool, "epsilon": float, "clip_negative": bool}
#   from_absorbance:  {"is_percent": bool}
#   pct_to_frac:      {}  (no params)
#   frac_to_pct:      {}  (no params)
#   kubelka_munk:     {"is_percent": bool, "epsilon": float}
#
# Cases with `"is_percent": true` are computed on `R * 100.0` (fractional
# input scaled by an exactly-representable constant) and the C test
# performs the matching scale before invoking the engine — see
# test_preprocessing_signal_conversion.cpp.
# ---------------------------------------------------------------------------

def to_absorbance_cases(R: np.ndarray):
    R_pct = R * 100.0
    return [
        ("reflectance_default",
         {"is_percent": False, "epsilon": 1e-10, "clip_negative": True},
         lambda: sig_mod.ToAbsorbance(
             source_type="reflectance", epsilon=1e-10, clip_negative=True
         ).fit_transform(R.copy())),
        ("reflectance_no_clip",
         {"is_percent": False, "epsilon": 1e-10, "clip_negative": False},
         lambda: sig_mod.ToAbsorbance(
             source_type="reflectance", epsilon=1e-10, clip_negative=False
         ).fit_transform(R.copy())),
        ("reflectance_eps_1e6",
         {"is_percent": False, "epsilon": 1e-6, "clip_negative": True},
         lambda: sig_mod.ToAbsorbance(
             source_type="reflectance", epsilon=1e-6, clip_negative=True
         ).fit_transform(R.copy())),
        ("reflectance_percent",
         {"is_percent": True, "epsilon": 1e-10, "clip_negative": True},
         lambda: sig_mod.ToAbsorbance(
             source_type="reflectance%", epsilon=1e-10, clip_negative=True
         ).fit_transform(R_pct.copy())),
        ("transmittance_percent",
         {"is_percent": True, "epsilon": 1e-10, "clip_negative": True},
         lambda: sig_mod.ToAbsorbance(
             source_type="transmittance%", epsilon=1e-10, clip_negative=True
         ).fit_transform(R_pct.copy())),
    ]


def from_absorbance_cases(A: np.ndarray):
    return [
        ("reflectance", {"is_percent": False},
         lambda: sig_mod.FromAbsorbance(target_type="reflectance"
                                          ).fit_transform(A.copy())),
        ("reflectance_percent", {"is_percent": True},
         lambda: sig_mod.FromAbsorbance(target_type="reflectance%"
                                          ).fit_transform(A.copy())),
        ("transmittance", {"is_percent": False},
         lambda: sig_mod.FromAbsorbance(target_type="transmittance"
                                          ).fit_transform(A.copy())),
        ("transmittance_percent", {"is_percent": True},
         lambda: sig_mod.FromAbsorbance(target_type="transmittance%"
                                          ).fit_transform(A.copy())),
    ]


def pct_to_frac_cases(R: np.ndarray):
    R_pct = R * 100.0
    return [
        ("default", {},
         lambda: sig_mod.PercentToFraction().fit_transform(R_pct.copy())),
    ]


def frac_to_pct_cases(R: np.ndarray):
    return [
        ("default", {},
         lambda: sig_mod.FractionToPercent().fit_transform(R.copy())),
    ]


def kubelka_munk_cases(R: np.ndarray):
    R_pct = R * 100.0
    return [
        ("reflectance_default",
         {"is_percent": False, "epsilon": 1e-10},
         lambda: sig_mod.KubelkaMunk(
             source_type="reflectance", epsilon=1e-10
         ).fit_transform(R.copy())),
        ("reflectance_eps_1e6",
         {"is_percent": False, "epsilon": 1e-6},
         lambda: sig_mod.KubelkaMunk(
             source_type="reflectance", epsilon=1e-6
         ).fit_transform(R.copy())),
        ("reflectance_percent",
         {"is_percent": True, "epsilon": 1e-10},
         lambda: sig_mod.KubelkaMunk(
             source_type="reflectance%", epsilon=1e-10
         ).fit_transform(R_pct.copy())),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                  out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"n4m_pp_{name}_v1",
        "numpy_version": np.__version__,
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
        if Y.shape != X.shape:
            raise RuntimeError(
                f"{name}/{case_name}: output shape {Y.shape} != input {X.shape}")
        fixture["cases"].append({
            "name": case_name,
            "params": params,
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

    version = get_nirs4all_version(NIRS4ALL_ROOT)

    R = synthesize_reflectance(SEED)
    A = synthesize_absorbance(SEED)
    R_pct = R * 100.0
    print(f"R (reflectance) shape {R.shape}, range [{R.min():.4f}, {R.max():.4f}]")
    print(f"A (absorbance)  shape {A.shape}, range [{A.min():.4f}, {A.max():.4f}]")

    # Each fixture stores the matrix in the natural input domain for that
    # operator. For ToAbsorbance and KubelkaMunk, the fixture input is the
    # fractional reflectance R; percent-mode cases (`is_percent: true`)
    # multiply R by 100.0 in the C test before invoking the engine and the
    # Python reference is invoked on R * 100.0 to match.  100.0 is exactly
    # representable in binary64, so the scale is bit-exact and the resulting
    # comparison stays at the 1e-12 / 1e-13 tolerance.
    write_fixture("to_absorbance",   R,     to_absorbance_cases(R),   out_dir, version)
    write_fixture("from_absorbance", A,     from_absorbance_cases(A), out_dir, version)
    write_fixture("pct_to_frac",     R_pct, pct_to_frac_cases(R),     out_dir, version)
    write_fixture("frac_to_pct",     R,     frac_to_pct_cases(R),     out_dir, version)
    write_fixture("kubelka_munk",    R,     kubelka_munk_cases(R),    out_dir, version)


if __name__ == "__main__":
    main()

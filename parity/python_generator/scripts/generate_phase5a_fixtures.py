#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 5a parity fixtures for the baseline correction operators.

Four operators, all stateless:

  * Detrend  — polynomial baseline subtraction.
  * AsLS     — Eilers & Boelens 2005.
  * AirPLS   — Zhang 2010.
  * ArPLS    — Baek 2015.

The references come from the frozen `c4a_parity_pybaselines_ref` package
under parity/python_generator/src/.  They have been validated once against
pybaselines==1.1.4 and are now the canonical parity floor (so subsequent
pybaselines releases can drift without breaking the c4a parity gates).

Shape: 40 rows x 60 cols of synthetic NIR-like spectra with a curved
baseline and two Gaussian peaks.  Same generator family as Phase 4 but
smaller because the iterative algorithms are O(N x max_iter) and we have
4 operators x 4-8 cases each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "parity" / "python_generator" / "src"))

from c4a_parity_pybaselines_ref import detrend, asls, airpls, arpls  # noqa: E402


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra.
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 40
COLS = 60


def synthesize_spectra(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(-0.4, 0.6)
        c = rng.uniform(-0.5, 0.5)
        baseline = a + b * wl + c * wl * wl
        peak1_mu = rng.uniform(0.2, 0.45)
        peak1_sigma = rng.uniform(0.025, 0.06)
        peak1_amp = rng.uniform(0.4, 1.5)
        peak2_mu = rng.uniform(0.6, 0.85)
        peak2_sigma = rng.uniform(0.03, 0.08)
        peak2_amp = rng.uniform(0.3, 1.0)
        peaks = (
            peak1_amp * np.exp(-((wl - peak1_mu) ** 2) / (2.0 * peak1_sigma ** 2))
            + peak2_amp * np.exp(-((wl - peak2_mu) ** 2) / (2.0 * peak2_sigma ** 2))
        )
        noise = rng.normal(0.0, 0.005, size=COLS)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Case lists — sweep parameters for each operator.
# ---------------------------------------------------------------------------

def detrend_cases(X: np.ndarray):
    return [
        ("polyorder0", {"polyorder": 0},
         lambda X=X: detrend(X, polyorder=0)),
        ("polyorder1", {"polyorder": 1},
         lambda X=X: detrend(X, polyorder=1)),
        ("polyorder2", {"polyorder": 2},
         lambda X=X: detrend(X, polyorder=2)),
        ("polyorder3", {"polyorder": 3},
         lambda X=X: detrend(X, polyorder=3)),
    ]


def asls_cases(X: np.ndarray):
    return [
        ("default",         {"lam": 1e6, "p": 1e-2, "max_iter": 50, "tol": 1e-3},
         lambda X=X: asls(X, lam=1e6, p=1e-2, max_iter=50, tol=1e-3)),
        ("low_lam",         {"lam": 1e4, "p": 1e-2, "max_iter": 50, "tol": 1e-3},
         lambda X=X: asls(X, lam=1e4, p=1e-2, max_iter=50, tol=1e-3)),
        ("high_p",          {"lam": 1e6, "p": 0.1,  "max_iter": 50, "tol": 1e-3},
         lambda X=X: asls(X, lam=1e6, p=0.1,  max_iter=50, tol=1e-3)),
        ("tight_tol_short", {"lam": 1e6, "p": 1e-2, "max_iter": 10, "tol": 1e-4},
         lambda X=X: asls(X, lam=1e6, p=1e-2, max_iter=10, tol=1e-4)),
    ]


def airpls_cases(X: np.ndarray):
    return [
        ("default",         {"lam": 1e6, "max_iter": 50, "tol": 1e-3},
         lambda X=X: airpls(X, lam=1e6, max_iter=50, tol=1e-3)),
        ("low_lam",         {"lam": 1e4, "max_iter": 50, "tol": 1e-3},
         lambda X=X: airpls(X, lam=1e4, max_iter=50, tol=1e-3)),
        ("high_lam",        {"lam": 1e7, "max_iter": 50, "tol": 1e-3},
         lambda X=X: airpls(X, lam=1e7, max_iter=50, tol=1e-3)),
        ("tight_tol_short", {"lam": 1e6, "max_iter": 10, "tol": 1e-4},
         lambda X=X: airpls(X, lam=1e6, max_iter=10, tol=1e-4)),
    ]


def arpls_cases(X: np.ndarray):
    return [
        ("default",         {"lam": 1e5, "max_iter": 50, "tol": 1e-3},
         lambda X=X: arpls(X, lam=1e5, max_iter=50, tol=1e-3)),
        ("low_lam",         {"lam": 1e3, "max_iter": 50, "tol": 1e-3},
         lambda X=X: arpls(X, lam=1e3, max_iter=50, tol=1e-3)),
        ("high_lam",        {"lam": 1e6, "max_iter": 50, "tol": 1e-3},
         lambda X=X: arpls(X, lam=1e6, max_iter=50, tol=1e-3)),
        ("tight_tol_short", {"lam": 1e5, "max_iter": 10, "tol": 1e-4},
         lambda X=X: arpls(X, lam=1e5, max_iter=10, tol=1e-4)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                  out_dir: Path) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{name}_v1",
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "reference": "c4a_parity_pybaselines_ref (frozen against pybaselines==1.1.4)",
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
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = synthesize_spectra(SEED)
    print(f"X shape {X.shape}, range [{X.min():.4f}, {X.max():.4f}]")

    write_fixture("detrend", X, detrend_cases(X), out_dir)
    write_fixture("asls",    X, asls_cases(X),    out_dir)
    write_fixture("airpls",  X, airpls_cases(X),  out_dir)
    write_fixture("arpls",   X, arpls_cases(X),   out_dir)


if __name__ == "__main__":
    main()

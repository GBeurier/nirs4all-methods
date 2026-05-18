#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 5b parity fixtures for the remaining baseline operators.

Six operators, all stateless:

  * ModPoly      — Lieber & Mahadevan-Jansen 2003 modified polynomial.
  * IModPoly     — Gan, Ruan, Mo 2006 σ-stopping modified polynomial.
  * SNIP         — Ryan 1988 / Morháč 1997 LLS-transformed peak-clipping.
  * RollingBall  — Kneen & Annegarn 1996 min-then-max morphology.
  * IAsLS        — He 2014 AsLS with polynomial weight init.
  * BEADS (simp) — Ning & Selesnick 2014 simplified pentadiagonal variant.

References come from the frozen ``c4a_parity_pybaselines_ref`` package.

Shape: 40 rows × 60 cols (same generator as Phase 5a), so the same input
matrix would round-trip through the c4a engine and the NumPy reference.
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

from c4a_parity_pybaselines_ref import (  # noqa: E402
    modpoly,
    imodpoly,
    snip,
    rolling_ball,
    iasls,
    beads,
)


# ---------------------------------------------------------------------------
# Encoding helpers (shared with the Phase 5a generator).
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra — same generator as Phase 5a so reviewers can
# compare baseline outputs across the two phases qualitatively.
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
# Case lists for each operator. Parameters cover the default + a few corner
# variations to exercise convergence regimes.
# ---------------------------------------------------------------------------

def modpoly_cases(X: np.ndarray):
    return [
        ("default",     {"polyorder": 2, "max_iter": 250, "tol": 1e-3},
         lambda X=X: modpoly(X, polyorder=2, max_iter=250, tol=1e-3)),
        ("order1",      {"polyorder": 1, "max_iter": 250, "tol": 1e-3},
         lambda X=X: modpoly(X, polyorder=1, max_iter=250, tol=1e-3)),
        ("order3_tight", {"polyorder": 3, "max_iter": 100, "tol": 1e-5},
         lambda X=X: modpoly(X, polyorder=3, max_iter=100, tol=1e-5)),
        ("few_iter",    {"polyorder": 2, "max_iter": 10,  "tol": 1e-3},
         lambda X=X: modpoly(X, polyorder=2, max_iter=10,  tol=1e-3)),
    ]


def imodpoly_cases(X: np.ndarray):
    return [
        ("default",     {"polyorder": 2, "max_iter": 250, "tol": 1e-3},
         lambda X=X: imodpoly(X, polyorder=2, max_iter=250, tol=1e-3)),
        ("order1",      {"polyorder": 1, "max_iter": 250, "tol": 1e-3},
         lambda X=X: imodpoly(X, polyorder=1, max_iter=250, tol=1e-3)),
        ("order3_tight", {"polyorder": 3, "max_iter": 100, "tol": 1e-5},
         lambda X=X: imodpoly(X, polyorder=3, max_iter=100, tol=1e-5)),
        ("few_iter",    {"polyorder": 2, "max_iter": 10,  "tol": 1e-3},
         lambda X=X: imodpoly(X, polyorder=2, max_iter=10,  tol=1e-3)),
    ]


def snip_cases(X: np.ndarray):
    return [
        ("default",   {"max_half_window": 20},
         lambda X=X: snip(X, max_half_window=20)),
        ("narrow",    {"max_half_window": 5},
         lambda X=X: snip(X, max_half_window=5)),
        ("wide",      {"max_half_window": 28},
         lambda X=X: snip(X, max_half_window=28)),
    ]


def rolling_ball_cases(X: np.ndarray):
    return [
        ("default",        {"half_window": 20, "smooth_half_window": 0},
         lambda X=X: rolling_ball(X, half_window=20, smooth_half_window=0)),
        ("smooth",         {"half_window": 15, "smooth_half_window": 5},
         lambda X=X: rolling_ball(X, half_window=15, smooth_half_window=5)),
        ("narrow",         {"half_window": 5,  "smooth_half_window": 0},
         lambda X=X: rolling_ball(X, half_window=5,  smooth_half_window=0)),
    ]


def iasls_cases(X: np.ndarray):
    return [
        ("default",        {"lam": 1e6, "p": 1e-2, "polyorder": 2,
                            "max_iter": 50, "tol": 1e-3},
         lambda X=X: iasls(X, lam=1e6, p=1e-2, polyorder=2,
                            max_iter=50, tol=1e-3)),
        ("low_lam",        {"lam": 1e4, "p": 1e-2, "polyorder": 2,
                            "max_iter": 50, "tol": 1e-3},
         lambda X=X: iasls(X, lam=1e4, p=1e-2, polyorder=2,
                            max_iter=50, tol=1e-3)),
        ("order1",         {"lam": 1e6, "p": 1e-2, "polyorder": 1,
                            "max_iter": 50, "tol": 1e-3},
         lambda X=X: iasls(X, lam=1e6, p=1e-2, polyorder=1,
                            max_iter=50, tol=1e-3)),
        ("tight_tol_short", {"lam": 1e6, "p": 1e-2, "polyorder": 2,
                             "max_iter": 10, "tol": 1e-4},
         lambda X=X: iasls(X, lam=1e6, p=1e-2, polyorder=2,
                            max_iter=10, tol=1e-4)),
    ]


def beads_cases(X: np.ndarray):
    return [
        ("default",   {"lam_0": 1e2, "lam_1": 0.5, "lam_2": 0.5,
                       "max_iter": 50, "tol": 1e-3},
         lambda X=X: beads(X, lam_0=1e2, lam_1=0.5, lam_2=0.5,
                            max_iter=50, tol=1e-3)),
        ("high_lam0", {"lam_0": 1e3, "lam_1": 0.5, "lam_2": 0.5,
                       "max_iter": 50, "tol": 1e-3},
         lambda X=X: beads(X, lam_0=1e3, lam_1=0.5, lam_2=0.5,
                            max_iter=50, tol=1e-3)),
        ("low_lam0",  {"lam_0": 1e1, "lam_1": 0.5, "lam_2": 0.5,
                       "max_iter": 50, "tol": 1e-3},
         lambda X=X: beads(X, lam_0=1e1, lam_1=0.5, lam_2=0.5,
                            max_iter=50, tol=1e-3)),
        ("few_iter",  {"lam_0": 1e2, "lam_1": 0.5, "lam_2": 0.5,
                       "max_iter": 10, "tol": 1e-4},
         lambda X=X: beads(X, lam_0=1e2, lam_1=0.5, lam_2=0.5,
                            max_iter=10, tol=1e-4)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer (mirrors Phase 5a).
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

    write_fixture("modpoly",      X, modpoly_cases(X),      out_dir)
    write_fixture("imodpoly",     X, imodpoly_cases(X),     out_dir)
    write_fixture("snip",         X, snip_cases(X),         out_dir)
    write_fixture("rolling_ball", X, rolling_ball_cases(X), out_dir)
    write_fixture("iasls",        X, iasls_cases(X),        out_dir)
    write_fixture("beads",        X, beads_cases(X),        out_dir)


if __name__ == "__main__":
    main()

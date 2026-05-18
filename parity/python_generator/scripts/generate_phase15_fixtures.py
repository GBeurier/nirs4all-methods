#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 15 parity fixtures for the seven noise + drift augmenters.

Each operator produces one fixture file under ``parity/fixtures/`` :

  * GaussianAdditiveNoise         -> ``aug_gaussian_noise_v1.json``
  * MultiplicativeNoise           -> ``aug_multiplicative_noise_v1.json``
  * SpikeNoise                    -> ``aug_spike_noise_v1.json``
  * HeteroscedasticNoiseAugmenter -> ``aug_hetero_noise_v1.json``
  * LinearBaselineDrift           -> ``aug_linear_drift_v1.json``
  * PolynomialBaselineDrift       -> ``aug_poly_drift_v1.json``
  * PathLengthAugmenter           -> ``aug_path_length_v1.json``

Fixture schema (one file per operator)::

    {
      "format": "c4a_aug_<NAME>_v1",
      "numpy_version": "...",
      "reference": "c4a_parity_augmenters_ref (frozen against nirs4all 0.8.x)",
      "encoding": "ieee754_binary64_be_hex",
      "rows": <int>, "cols": <int>,
      "X_hex": [ "<16 hex>", ... ],     # row-major flatten of input matrix
      "cases": [
        { "name": "...", "seed": <uint64>, "params": { ... },
          "expected_out_hex": [ "<16 hex>", ... ] },
        ...
      ]
    }

Every case carries a fresh PCG64 seed so the C side calls
``c4a_rng_pcg64_set_seed(rng, S)`` before invoking ``_apply`` -- matching the
"stochastic-with-seeded-PCG64" tolerance class declared in
``roadmap/phase-15-18-augmenters-abi-contract.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "parity" / "python_generator" / "src"))

from c4a_parity_augmenters_ref import (  # noqa: E402
    gaussian_noise_apply,
    hetero_noise_apply,
    linear_drift_apply,
    multiplicative_noise_apply,
    path_length_apply,
    poly_drift_apply,
    spike_noise_apply,
)


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra. The same shape (12 rows x 24 cols) is used
# across all seven operators -- small enough for fast fixture I/O, large
# enough to exercise per-row variations and edge effects.
# ---------------------------------------------------------------------------

ROWS = 12
COLS = 24
INPUT_SEED = 20260518


def synthesize_spectra(seed: int) -> np.ndarray:
    """A modest spectra matrix with realistic NIR-like structure."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.4, 1.2)
        b = rng.uniform(-0.3, 0.5)
        c = rng.uniform(-0.4, 0.4)
        baseline = a + b * wl + c * wl * wl
        peak_mu = rng.uniform(0.3, 0.7)
        peak_sigma = rng.uniform(0.03, 0.08)
        peak_amp = rng.uniform(0.5, 1.2)
        peak = peak_amp * np.exp(-((wl - peak_mu) ** 2) / (2.0 * peak_sigma ** 2))
        small_noise = rng.normal(0.0, 0.002, size=COLS)
        X[i] = baseline + peak + small_noise
    return X


# ---------------------------------------------------------------------------
# Per-operator fixture builders.
# ---------------------------------------------------------------------------

def build_cases(
    op_name: str, X: np.ndarray, cases: list[tuple[str, int, dict, np.ndarray]]
) -> dict[str, Any]:
    return {
        "format": f"c4a_aug_{op_name}_v1",
        "numpy_version": np.__version__,
        "reference": "c4a_parity_augmenters_ref (frozen against nirs4all 0.8.x)",
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "X_hex": array_to_hex(X),
        "cases": [
            {
                "name": name,
                "seed": int(seed),
                "params": params,
                "expected_out_hex": array_to_hex(expected),
            }
            for name, seed, params, expected in cases
        ],
    }


def write_fixture(name: str, fixture: dict[str, Any], out_dir: Path) -> None:
    path = out_dir / f"aug_{name}_v1.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, "
          f"{len(fixture['cases'])} cases)")


# ---------------------------------------------------------------------------
# Operator-specific case lists.
# ---------------------------------------------------------------------------

def gaussian_noise_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, sigma in [
        ("default",     1, 0.01),
        ("strong",      2, 0.10),
        ("very_small", 17, 1e-4),
    ]:
        rng = np.random.default_rng(seed)
        out = gaussian_noise_apply(X, rng=rng, sigma=sigma)
        cases.append((name, seed, {"sigma": sigma}, out))
    write_fixture("gaussian_noise",
                   build_cases("gaussian_noise", X, cases),
                   out_dir)


def multiplicative_noise_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, sigma in [
        ("default",   3, 0.05),
        ("tight",     4, 0.01),
        ("loose",    23, 0.20),
    ]:
        rng = np.random.default_rng(seed)
        out = multiplicative_noise_apply(X, rng=rng, sigma_gain=sigma)
        cases.append((name, seed, {"sigma_gain": sigma}, out))
    write_fixture("multiplicative_noise",
                   build_cases("multiplicative_noise", X, cases),
                   out_dir)


def spike_noise_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, n_min, n_max, amp_lo, amp_hi in [
        ("default",        5, 1, 3, -0.5, 0.5),
        ("single_spike",   6, 1, 1, -0.2, 0.2),
        ("many_spikes",    7, 2, 5,  0.1, 0.4),
        ("positive_only", 31, 1, 3,  0.5, 1.0),
    ]:
        rng = np.random.default_rng(seed)
        out = spike_noise_apply(
            X, rng=rng,
            n_spikes_min=n_min, n_spikes_max=n_max,
            amplitude_min=amp_lo, amplitude_max=amp_hi)
        cases.append((name, seed, {
            "n_spikes_min": n_min,
            "n_spikes_max": n_max,
            "amplitude_min": amp_lo,
            "amplitude_max": amp_hi,
        }, out))
    write_fixture("spike_noise",
                   build_cases("spike_noise", X, cases),
                   out_dir)


def hetero_noise_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, base, dep in [
        ("default",        8, 1e-3, 5e-3),
        ("base_only",      9, 5e-3, 0.0),
        ("dep_only",      10, 0.0,  1e-2),
        ("strong_both",   29, 1e-2, 2e-2),
    ]:
        rng = np.random.default_rng(seed)
        out = hetero_noise_apply(
            X, rng=rng, noise_base=base, noise_signal_dep=dep)
        cases.append((name, seed,
                       {"noise_base": base, "noise_signal_dep": dep},
                       out))
    write_fixture("hetero_noise",
                   build_cases("hetero_noise", X, cases),
                   out_dir)


def linear_drift_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, off_lo, off_hi, sl_lo, sl_hi in [
        ("default",     11, -0.10,  0.10, -0.001, 0.001),
        ("offset_only", 12, -0.05,  0.05,  0.0,   0.0),
        ("slope_only",  13,  0.0,   0.0,  -0.01,  0.01),
        ("asymmetric",  19,  0.05,  0.20, -0.005, 0.0),
    ]:
        rng = np.random.default_rng(seed)
        out = linear_drift_apply(
            X, rng=rng,
            offset_min=off_lo, offset_max=off_hi,
            slope_min=sl_lo,   slope_max=sl_hi)
        cases.append((name, seed, {
            "offset_min": off_lo, "offset_max": off_hi,
            "slope_min":  sl_lo,  "slope_max":  sl_hi,
        }, out))
    write_fixture("linear_drift",
                   build_cases("linear_drift", X, cases),
                   out_dir)


def poly_drift_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    # Default decaying ranges: (-0.1/(k+1), 0.1/(k+1)) for k in [0, degree]
    def decay(degree: int) -> tuple[list[float], list[float]]:
        lo = [-0.1 / (k + 1) for k in range(degree + 1)]
        hi = [ 0.1 / (k + 1) for k in range(degree + 1)]
        return lo, hi
    for name, seed, degree, get_ranges in [
        ("degree3_default", 14, 3, lambda: decay(3)),
        ("degree2_default", 15, 2, lambda: decay(2)),
        ("degree1_default", 16, 1, lambda: decay(1)),
        ("degree4_asym",    27, 4, lambda: (
            [-0.05, -0.02, -0.01, -0.005, -0.001],
            [ 0.10,  0.05,  0.02,  0.01,   0.005])),
    ]:
        lo, hi = get_ranges()
        rng = np.random.default_rng(seed)
        out = poly_drift_apply(X, rng=rng, coeff_min=lo, coeff_max=hi)
        cases.append((name, seed, {
            "degree": degree,
            "coeff_min": lo, "coeff_max": hi,
        }, out))
    write_fixture("poly_drift",
                   build_cases("poly_drift", X, cases),
                   out_dir)


def path_length_fixture(X: np.ndarray, out_dir: Path) -> None:
    cases: list[tuple[str, int, dict, np.ndarray]] = []
    for name, seed, std, min_L in [
        ("default",    20, 0.05, 0.5),
        ("tight",      21, 0.01, 0.5),
        ("wide",       22, 0.20, 0.5),
        ("hard_clip",  30, 0.50, 0.8),
    ]:
        rng = np.random.default_rng(seed)
        out = path_length_apply(
            X, rng=rng,
            path_length_std=std, min_path_length=min_L)
        cases.append((name, seed,
                       {"path_length_std": std, "min_path_length": min_L},
                       out))
    write_fixture("path_length",
                   build_cases("path_length", X, cases),
                   out_dir)


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = synthesize_spectra(INPUT_SEED)
    print(f"input matrix: rows={X.shape[0]}, cols={X.shape[1]}, "
          f"range=[{X.min():.4f}, {X.max():.4f}]")

    gaussian_noise_fixture(X, out_dir)
    multiplicative_noise_fixture(X, out_dir)
    spike_noise_fixture(X, out_dir)
    hetero_noise_fixture(X, out_dir)
    linear_drift_fixture(X, out_dir)
    poly_drift_fixture(X, out_dir)
    path_length_fixture(X, out_dir)


if __name__ == "__main__":
    main()

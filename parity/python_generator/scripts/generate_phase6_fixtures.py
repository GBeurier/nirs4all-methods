#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 6 parity fixtures for the wavelets preprocessing operators.

Phase 6 (6 ops): wavelet, haar, wavelet_denoise, wavelet_features,
wavelet_pca, wavelet_svd.  Each fixture matches the schema used by
phases 5 / 9 (fit + transform matrices, per-case parameters and
``output_rows / output_cols``).

Output: ``parity/fixtures/{wavelet,haar,wavelet_denoise,wavelet_features,
wavelet_pca,wavelet_svd}_v1.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_ROOT))

from c4a_parity_wavelets_ref import (  # noqa: E402
    haar_transform,
    wavelet_denoise_transform,
    wavelet_features_transform,
    wavelet_pca_fit_transform,
    wavelet_svd_fit_transform,
    wavelet_transform,
)


# ---------------------------------------------------------------------------
# Fixture encoding helpers
# ---------------------------------------------------------------------------
def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra (same generator shape as Phase 9)
# ---------------------------------------------------------------------------
SEED = 20260518
ROWS = 32
COLS = 128


def synthesize_spectra(seed: int, rows: int = ROWS,
                        cols: int = COLS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, cols, dtype=np.float64)
    X = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
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
        noise = rng.normal(0.0, 0.01, size=cols)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Per-operator case helpers
# ---------------------------------------------------------------------------
def wavelet_cases(X: np.ndarray):
    return [
        ("haar_period", {"family": "haar", "mode": "periodization"},
         lambda: wavelet_transform(X, family="haar", mode="periodization")),
        ("db4_symmetric", {"family": "db4", "mode": "symmetric"},
         lambda: wavelet_transform(X, family="db4", mode="symmetric")),
        ("sym4_zero", {"family": "sym4", "mode": "zero"},
         lambda: wavelet_transform(X, family="sym4", mode="zero")),
        ("coif1_period", {"family": "coif1", "mode": "periodization"},
         lambda: wavelet_transform(X, family="coif1", mode="periodization")),
    ]


def haar_cases(X: np.ndarray):
    return [
        ("default", {},
         lambda: haar_transform(X)),
    ]


def wavelet_denoise_cases(X: np.ndarray):
    return [
        ("db4_per_l3_soft_median",
         {"family": "db4", "mode": "periodization", "level": 3,
          "threshold_mode": "soft", "noise_estimator": "median"},
         lambda: wavelet_denoise_transform(
            X, family="db4", mode="periodization", level=3,
            threshold_mode="soft", noise_estimator="median")),
        ("db4_per_l4_hard_std",
         {"family": "db4", "mode": "periodization", "level": 4,
          "threshold_mode": "hard", "noise_estimator": "std"},
         lambda: wavelet_denoise_transform(
            X, family="db4", mode="periodization", level=4,
            threshold_mode="hard", noise_estimator="std")),
        ("sym4_symmetric_l2_soft_median",
         {"family": "sym4", "mode": "symmetric", "level": 2,
          "threshold_mode": "soft", "noise_estimator": "median"},
         lambda: wavelet_denoise_transform(
            X, family="sym4", mode="symmetric", level=2,
            threshold_mode="soft", noise_estimator="median")),
    ]


def wavelet_features_cases(X: np.ndarray):
    return [
        ("haar_per_l3",
         {"family": "haar", "mode": "periodization", "max_level": 3},
         lambda: wavelet_features_transform(
            X, family="haar", mode="periodization", max_level=3)),
        ("db4_per_l3",
         {"family": "db4", "mode": "periodization", "max_level": 3},
         lambda: wavelet_features_transform(
            X, family="db4", mode="periodization", max_level=3)),
        ("coif1_symmetric_l2",
         {"family": "coif1", "mode": "symmetric", "max_level": 2},
         lambda: wavelet_features_transform(
            X, family="coif1", mode="symmetric", max_level=2)),
    ]


def wavelet_pca_cases(fit_X: np.ndarray, test_X: np.ndarray):
    return [
        ("haar_per_l2_n5",
         {"family": "haar", "mode": "periodization", "max_level": 2,
          "n_components": 5.0},
         lambda: wavelet_pca_fit_transform(
            fit_X, test_X, family="haar", mode="periodization",
            max_level=2, n_components=5.0)),
        ("db4_per_l3_n3",
         {"family": "db4", "mode": "periodization", "max_level": 3,
          "n_components": 3.0},
         lambda: wavelet_pca_fit_transform(
            fit_X, test_X, family="db4", mode="periodization",
            max_level=3, n_components=3.0)),
        ("db4_per_l3_var0_95",
         {"family": "db4", "mode": "periodization", "max_level": 3,
          "n_components": 0.95},
         lambda: wavelet_pca_fit_transform(
            fit_X, test_X, family="db4", mode="periodization",
            max_level=3, n_components=0.95)),
    ]


def wavelet_svd_cases(fit_X: np.ndarray, test_X: np.ndarray):
    return [
        ("haar_per_l2_n5",
         {"family": "haar", "mode": "periodization", "max_level": 2,
          "n_components": 5.0},
         lambda: wavelet_svd_fit_transform(
            fit_X, test_X, family="haar", mode="periodization",
            max_level=2, n_components=5.0)),
        ("db4_per_l3_n3",
         {"family": "db4", "mode": "periodization", "max_level": 3,
          "n_components": 3.0},
         lambda: wavelet_svd_fit_transform(
            fit_X, test_X, family="db4", mode="periodization",
            max_level=3, n_components=3.0)),
        ("db4_per_l3_var0_99",
         {"family": "db4", "mode": "periodization", "max_level": 3,
          "n_components": 0.99},
         lambda: wavelet_svd_fit_transform(
            fit_X, test_X, family="db4", mode="periodization",
            max_level=3, n_components=0.99)),
    ]


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------
def write_simple_fixture(
    name: str, op_id: str,
    X: np.ndarray,
    cases,
    out_dir: Path,
    pywt_version: str,
) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "pywt_version": pywt_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y = np.ascontiguousarray(fn(), dtype=np.float64)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
            "output_hex": array_to_hex(Y),
        })
    path = out_dir / f"{name}_v1.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, {len(cases)} cases)")


def write_fit_transform_fixture(
    name: str, op_id: str,
    fit_X: np.ndarray, transform_X: np.ndarray,
    cases,
    out_dir: Path,
    pywt_version: str,
) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "pywt_version": pywt_version,
        "encoding": "ieee754_binary64_be_hex",
        "fit_rows": int(fit_X.shape[0]),
        "fit_cols": int(fit_X.shape[1]),
        "fit_input_hex": array_to_hex(fit_X),
        "rows": int(transform_X.shape[0]),
        "cols": int(transform_X.shape[1]),
        "input_hex": array_to_hex(transform_X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y, k = fn()
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
            "k_learned": int(k),
            "output_hex": array_to_hex(Y),
        })
    path = out_dir / f"{name}_v1.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[3] / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pywt_version = "1.6.0"
    try:
        import pywt  # type: ignore
        pywt_version = pywt.__version__
    except ImportError:
        pass

    X = synthesize_spectra(SEED)
    fit_X = X
    transform_X = synthesize_spectra(SEED + 1)
    print(f"X shape {X.shape}, range [{X.min():.4f}, {X.max():.4f}]")

    write_simple_fixture(
        "wavelet", "wavelet", X, wavelet_cases(X), out_dir, pywt_version)
    write_simple_fixture(
        "haar", "haar", X, haar_cases(X), out_dir, pywt_version)
    write_simple_fixture(
        "wavelet_denoise", "wavelet_denoise", X,
        wavelet_denoise_cases(X), out_dir, pywt_version)
    write_simple_fixture(
        "wavelet_features", "wavelet_features", X,
        wavelet_features_cases(X), out_dir, pywt_version)
    write_fit_transform_fixture(
        "wavelet_pca", "wavelet_pca", fit_X, transform_X,
        wavelet_pca_cases(fit_X, transform_X), out_dir, pywt_version)
    write_fit_transform_fixture(
        "wavelet_svd", "wavelet_svd", fit_X, transform_X,
        wavelet_svd_cases(fit_X, transform_X), out_dir, pywt_version)


if __name__ == "__main__":
    main()

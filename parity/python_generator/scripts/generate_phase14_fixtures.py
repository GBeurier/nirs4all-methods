#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 14 parity fixtures for the meta sample filters.

Three filters, all written against the frozen NumPy reference under
``parity/python_generator/src/n4m_parity_filters_ref/``:

  * HighLeverageFilter   — hat-matrix + PCA leverage variants.
  * SpectralQualityFilter — multi-criteria per-row quality checks.
  * CompositeFilter       — ANY / ALL aggregation of two sub-filters.

All filters produce boolean keep-masks. The encoded "output" is a 1-D
length-``rows`` array of 0/1 doubles (using the same IEEE-754 big-endian
hex encoding as the rest of the parity fixture family) so the existing
``fixture_parser.hpp`` can consume it without a special path.

Shapes:
  * leverage : 80 rows × 12 features (rows > cols → hat path) + 30 × 60
               (rows < cols → PCA fallback) cases.
  * quality  : 60 rows × 20 features, with 4 corrupted rows (NaN / Inf /
               zeros / flat).
  * composite: 80 rows × 12 features, sharing the leverage matrix so a
               sub-filter can run on it directly.
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

from n4m_parity_filters_ref import (  # noqa: E402
    composite_mask,
    high_leverage_mask,
    spectral_quality_mask,
)


# ---------------------------------------------------------------------------
# Encoding helpers (shared with previous fixture generators).
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


def mask_to_hex(mask: np.ndarray) -> list[str]:
    """Encode a boolean mask as float64 0.0/1.0 hex sequence so the
    existing fixture parser can consume it unmodified."""
    return array_to_hex(mask.astype(np.float64))


# ---------------------------------------------------------------------------
# Synthetic data — leverage variant (rows > cols and rows < cols).
# ---------------------------------------------------------------------------

LEVERAGE_SEED = 20260518
QUALITY_SEED  = 20260519

def synthesize_leverage_matrix(n_samples: int, n_features: int,
                                 n_outliers: int, seed: int) -> np.ndarray:
    """Build a (n_samples, n_features) NIR-like matrix where the last
    `n_outliers` rows are pushed away from the bulk in feature space (so
    they appear as high-leverage points)."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, n_features, dtype=np.float64)
    X = np.empty((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples - n_outliers):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(-0.4, 0.6)
        c = rng.uniform(-0.5, 0.5)
        baseline = a + b * wl + c * wl * wl
        peak_mu = rng.uniform(0.3, 0.7)
        peak_sigma = rng.uniform(0.025, 0.06)
        peak_amp = rng.uniform(0.4, 1.5)
        peaks = peak_amp * np.exp(-((wl - peak_mu) ** 2)
                                    / (2.0 * peak_sigma ** 2))
        noise = rng.normal(0.0, 0.005, size=n_features)
        X[i] = baseline + peaks + noise
    # Inject high-leverage outliers — large amplitude, shifted spectral
    # shape, far from the bulk in PCA space.
    for k in range(n_outliers):
        i = n_samples - n_outliers + k
        sign = 1.0 if k % 2 == 0 else -1.0
        outlier = 3.0 * sign + 1.5 * wl + 2.0 * np.cos(8.0 * np.pi * wl)
        X[i] = outlier + 0.005 * rng.standard_normal(n_features)
    return X


def synthesize_quality_matrix(n_samples: int, n_features: int,
                                seed: int) -> np.ndarray:
    """Build a (n_samples, n_features) matrix mixing clean and corrupted
    spectra. The first 4 rows are intentionally corrupted to exercise the
    NaN / Inf / zero / flat checks; the rest are reasonable NIR spectra."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, n_features, dtype=np.float64)
    X = np.empty((n_samples, n_features), dtype=np.float64)
    # Row 0 — heavy NaN coverage (~25 %).
    base0 = 0.5 + 0.2 * wl + 0.05 * rng.standard_normal(n_features)
    nan_idx = rng.choice(n_features, size=n_features // 4, replace=False)
    base0[nan_idx] = np.nan
    X[0] = base0
    # Row 1 — single Inf.
    base1 = 0.4 + 0.1 * wl + 0.05 * rng.standard_normal(n_features)
    base1[n_features // 2] = np.inf
    X[1] = base1
    # Row 2 — heavy zero coverage (~70 %).
    base2 = np.zeros(n_features, dtype=np.float64)
    nonzero_idx = rng.choice(n_features, size=n_features // 3, replace=False)
    base2[nonzero_idx] = rng.uniform(0.1, 0.3, size=nonzero_idx.size)
    X[2] = base2
    # Row 3 — constant (zero variance).
    X[3] = np.full(n_features, 0.5, dtype=np.float64)
    # Remaining rows — clean NIR-like spectra.
    for i in range(4, n_samples):
        a = rng.uniform(0.4, 0.6)
        b = rng.uniform(-0.2, 0.4)
        peak_mu = rng.uniform(0.3, 0.7)
        peak_amp = rng.uniform(0.2, 0.8)
        peak_sigma = rng.uniform(0.03, 0.07)
        peaks = peak_amp * np.exp(-((wl - peak_mu) ** 2)
                                    / (2.0 * peak_sigma ** 2))
        noise = rng.normal(0.0, 0.005, size=n_features)
        X[i] = a + b * wl + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Case lists.
# ---------------------------------------------------------------------------

def leverage_cases(X_tall: np.ndarray, X_wide: np.ndarray):
    return [
        # rows > cols → hat path
        ("hat_default",
         {"method": "hat", "threshold_multiplier": 2.0, "absolute_threshold": None,
          "n_components": 0, "center": 1},
         X_tall, lambda X=X_tall: high_leverage_mask(
             X, method="hat", threshold_multiplier=2.0, center=True)),
        ("hat_strict",
         {"method": "hat", "threshold_multiplier": 1.2, "absolute_threshold": None,
          "n_components": 0, "center": 1},
         X_tall, lambda X=X_tall: high_leverage_mask(
             X, method="hat", threshold_multiplier=1.2, center=True)),
        # rows > cols, but force PCA path
        ("pca_default",
         {"method": "pca", "threshold_multiplier": 2.0, "absolute_threshold": None,
          "n_components": 5, "center": 1},
         X_tall, lambda X=X_tall: high_leverage_mask(
             X, method="pca", threshold_multiplier=2.0, n_components=5,
             center=True)),
        # rows < cols → hat path auto-falls back to PCA
        ("auto_fallback",
         {"method": "hat", "threshold_multiplier": 2.0, "absolute_threshold": None,
          "n_components": 0, "center": 1},
         X_wide, lambda X=X_wide: high_leverage_mask(
             X, method="hat", threshold_multiplier=2.0, center=True)),
    ]


def quality_cases(X: np.ndarray):
    return [
        ("default",
         {"max_nan_ratio": 0.1, "max_zero_ratio": 0.5, "min_variance": 1e-8,
          "max_value": None, "min_value": None, "check_inf": True},
         lambda X=X: spectral_quality_mask(X)),
        ("strict_nan",
         {"max_nan_ratio": 0.01, "max_zero_ratio": 0.5, "min_variance": 1e-8,
          "max_value": None, "min_value": None, "check_inf": True},
         lambda X=X: spectral_quality_mask(X, max_nan_ratio=0.01)),
        ("with_bounds",
         {"max_nan_ratio": 0.1, "max_zero_ratio": 0.5, "min_variance": 1e-8,
          "max_value": 1.5, "min_value": -0.5, "check_inf": True},
         lambda X=X: spectral_quality_mask(X, max_value=1.5, min_value=-0.5)),
        ("no_inf_check",
         {"max_nan_ratio": 0.1, "max_zero_ratio": 0.5, "min_variance": 1e-8,
          "max_value": None, "min_value": None, "check_inf": False},
         lambda X=X: spectral_quality_mask(X, check_inf=False)),
    ]


def composite_cases(X: np.ndarray):
    """The composite combines a HighLeverage(default) and a SpectralQuality
    (default) sub-filter. The fixture replays both sub-filter computations
    in Python and aggregates them via ANY / ALL — matching what the C engine
    does internally."""
    leverage = high_leverage_mask(X, method="hat", threshold_multiplier=2.0,
                                    center=True)
    quality = spectral_quality_mask(X)
    return [
        ("any",
         {"mode": "any"},
         lambda lev=leverage, q=quality: composite_mask([lev, q], mode="any")),
        ("all",
         {"mode": "all"},
         lambda lev=leverage, q=quality: composite_mask([lev, q], mode="all")),
    ]


# ---------------------------------------------------------------------------
# Fixture writers.
# ---------------------------------------------------------------------------

def write_leverage_fixture(out_dir: Path,
                            cases: list[tuple]) -> None:
    """Emit two fixtures: ``filter_leverage_v1.json`` for the tall-matrix
    cases (rows > cols → hat or explicit PCA), and
    ``filter_leverage_wide_v1.json`` for the wide-matrix case
    (rows < cols → auto-fallback to PCA). The fixture parser expects a
    single (rows, cols, input_hex) per file, so we split by shape."""
    by_shape: dict[tuple[int, int], list[tuple[str, dict, Callable[[], np.ndarray], np.ndarray]]] = {}
    for case_name, params, X_used, fn in cases:
        shape = (int(X_used.shape[0]), int(X_used.shape[1]))
        by_shape.setdefault(shape, []).append((case_name, params, fn, X_used))

    # Filename mapping: the largest (tall) shape is the canonical primary
    # fixture, the smaller (wide) shape gets the ``_wide`` suffix.
    primary_shape = max(by_shape.keys(), key=lambda s: s[0])
    suffixes = {primary_shape: ""}
    for shape in by_shape:
        if shape == primary_shape:
            continue
        suffixes[shape] = "_wide"

    for shape, sublist in by_shape.items():
        rows, cols = shape
        X = sublist[0][3]  # every case in a bucket shares the same X
        fixture: dict[str, Any] = {
            "format": "n4m_filter_leverage_v1",
            "numpy_version": np.__version__,
            "reference": "n4m_parity_filters_ref (validated against nirs4all 0.8.x)",
            "encoding": "ieee754_binary64_be_hex",
            "rows": rows,
            "cols": cols,
            "input_hex": array_to_hex(X),
            "cases": [],
        }
        for case_name, params, fn, _Xused in sublist:
            mask = fn()
            fixture["cases"].append({
                "name": case_name,
                "params": params,
                "output_hex": mask_to_hex(mask),
            })
        out = out_dir / f"filter_leverage{suffixes[shape]}_v1.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=None)
            f.write("\n")
        print(f"wrote {out} ({out.stat().st_size:,} bytes,"
              f" {len(sublist)} cases, shape {shape})")


def write_mask_fixture(name: str, X: np.ndarray,
                        cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                        out_dir: Path) -> None:
    fixture: dict[str, Any] = {
        "format": f"n4m_filter_{name}_v1",
        "numpy_version": np.__version__,
        "reference": "n4m_parity_filters_ref",
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        mask = fn()
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_hex": mask_to_hex(mask),
        })
    out = out_dir / f"filter_{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tall = synthesize_leverage_matrix(
        n_samples=80, n_features=12, n_outliers=4, seed=LEVERAGE_SEED)
    X_wide = synthesize_leverage_matrix(
        n_samples=30, n_features=60, n_outliers=3, seed=LEVERAGE_SEED + 1)
    print(f"leverage tall: shape {X_tall.shape}, range "
          f"[{X_tall.min():.4f}, {X_tall.max():.4f}]")
    print(f"leverage wide: shape {X_wide.shape}, range "
          f"[{X_wide.min():.4f}, {X_wide.max():.4f}]")

    write_leverage_fixture(out_dir, leverage_cases(X_tall, X_wide))

    X_q = synthesize_quality_matrix(n_samples=60, n_features=20,
                                      seed=QUALITY_SEED)
    print(f"quality:       shape {X_q.shape}")
    write_mask_fixture("quality", X_q, quality_cases(X_q), out_dir)

    # Composite shares the tall matrix.
    write_mask_fixture("composite", X_tall, composite_cases(X_tall), out_dir)


if __name__ == "__main__":
    main()

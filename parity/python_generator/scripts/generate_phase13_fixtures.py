#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 13 parity fixtures for the XOutlierFilter.

One operator, six methods. Each method gets its own JSON fixture file
(matching the per-method file split used by Phase 12 for ``y_outlier``).

Schemas reuse the standard preprocessing/filter shape:

  {
    "format": "c4a_filter_x_outlier_<method>_v1",
    "numpy_version": "...",
    "reference": "c4a_parity_filters_ref (frozen against sklearn)",
    "encoding": "ieee754_binary64_be_hex",
    "rows": <int>, "cols": <int>,
    "input_hex": [ "<16 hex>", ... ],
    "cases": [
      { "name": "...",
        "params": { ... },
        "output_hex": [ "<16 hex>", ... ] (0.0 / 1.0 keep mask as f64) },
      ...
    ]
  }

The C side's deterministic methods (mahalanobis, pca_residual, pca_leverage)
are bit-identical to the sklearn reference within the contracted 1e-10 abs
tolerance. The seeded random paths (robust_mahalanobis, isolation_forest)
use sklearn's NumPy-RandomState seeding which differs from c4a's PCG64,
so the C output is byte-equal to the reference only when c4a uses its
own seed-derived ensemble. The fixtures still capture the sklearn mask
so the parity test can quantify divergence rather than skip the path.
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

from c4a_parity_filters_ref import x_outlier_fit_get_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


def mask_to_hex(mask: np.ndarray) -> list[str]:
    """Encode an integer/boolean mask as float64 0.0/1.0 sequence so the
    existing fixture parser can consume it unmodified."""
    return array_to_hex(mask.astype(np.float64))


# ---------------------------------------------------------------------------
# Synthetic data — NIRS-like spectra with planted outliers.
# ---------------------------------------------------------------------------

SEED = 20260518


def synthesize_xmatrix(n_samples: int, n_features: int,
                        n_outliers: int, seed: int) -> np.ndarray:
    """NIRS-like baseline + Gaussian peak with planted spectral outliers."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, n_features, dtype=np.float64)
    X = np.empty((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples - n_outliers):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(-0.4, 0.6)
        c = rng.uniform(-0.5, 0.5)
        baseline = a + b * wl + c * wl * wl
        peak_mu = rng.uniform(0.3, 0.7)
        peak_sigma = rng.uniform(0.03, 0.07)
        peak_amp = rng.uniform(0.4, 1.5)
        peaks = peak_amp * np.exp(-((wl - peak_mu) ** 2)
                                    / (2.0 * peak_sigma ** 2))
        noise = rng.normal(0.0, 0.005, size=n_features)
        X[i] = baseline + peaks + noise
    for k in range(n_outliers):
        i = n_samples - n_outliers + k
        sign = 1.0 if k % 2 == 0 else -1.0
        outlier = 3.0 * sign + 1.5 * wl + 2.0 * np.cos(8.0 * np.pi * wl)
        X[i] = outlier + 0.005 * rng.standard_normal(n_features)
    return X


# ---------------------------------------------------------------------------
# Case generators per method.
# ---------------------------------------------------------------------------

def cases_for(method: str, X: np.ndarray) -> list[tuple[str, dict,
                                                           Callable[[], dict]]]:
    if method == "mahalanobis":
        return [
            ("default",
             {"method": "mahalanobis", "threshold": None, "n_components": None,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="mahalanobis", random_state=SEED)),
            ("with_n_components",
             {"method": "mahalanobis", "threshold": None, "n_components": 8,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="mahalanobis", n_components=8, random_state=SEED)),
            ("explicit_threshold",
             {"method": "mahalanobis", "threshold": 6.0, "n_components": None,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="mahalanobis", threshold=6.0, random_state=SEED)),
        ]
    if method == "robust_mahalanobis":
        return [
            ("default",
             {"method": "robust_mahalanobis", "threshold": None,
              "n_components": None, "contamination": 0.1,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="robust_mahalanobis", random_state=SEED)),
            ("explicit_threshold",
             {"method": "robust_mahalanobis", "threshold": 8.0,
              "n_components": None, "contamination": 0.1,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="robust_mahalanobis", threshold=8.0,
                 random_state=SEED)),
        ]
    if method == "pca_residual":
        return [
            ("default",
             {"method": "pca_residual", "threshold": None, "n_components": None,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="pca_residual", random_state=SEED)),
            ("k_5",
             {"method": "pca_residual", "threshold": None, "n_components": 5,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="pca_residual", n_components=5, random_state=SEED)),
            ("explicit_threshold",
             {"method": "pca_residual", "threshold": 0.5,
              "n_components": None, "contamination": 0.1,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="pca_residual", threshold=0.5, random_state=SEED)),
        ]
    if method == "pca_leverage":
        return [
            ("default",
             {"method": "pca_leverage", "threshold": None,
              "n_components": None, "contamination": 0.1,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="pca_leverage", random_state=SEED)),
            ("k_5",
             {"method": "pca_leverage", "threshold": None, "n_components": 5,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="pca_leverage", n_components=5, random_state=SEED)),
        ]
    if method == "isolation_forest":
        return [
            ("default",
             {"method": "isolation_forest", "threshold": None,
              "n_components": None, "contamination": 0.1,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="isolation_forest", random_state=SEED)),
            ("higher_contamination",
             {"method": "isolation_forest", "threshold": None,
              "n_components": None, "contamination": 0.2,
              "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="isolation_forest", contamination=0.2,
                 random_state=SEED)),
        ]
    if method == "lof":
        return [
            ("default",
             {"method": "lof", "threshold": None, "n_components": None,
              "contamination": 0.1, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="lof", random_state=SEED)),
            ("higher_contamination",
             {"method": "lof", "threshold": None, "n_components": None,
              "contamination": 0.2, "random_state": SEED},
             lambda X=X: x_outlier_fit_get_mask(
                 X, method="lof", contamination=0.2, random_state=SEED)),
        ]
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(method: str, X: np.ndarray, cases, out_dir: Path) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_filter_x_outlier_{method}_v1",
        "numpy_version": np.__version__,
        "reference": "c4a_parity_filters_ref (frozen against sklearn)",
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        out = fn()
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_hex": mask_to_hex(out["mask"]),
        })
    path = out_dir / f"filter_x_outlier_{method}_v1.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    X = synthesize_xmatrix(n_samples=80, n_features=12, n_outliers=4,
                              seed=SEED)
    print(f"x_outlier matrix: shape {X.shape}, "
          f"range [{X.min():.4f}, {X.max():.4f}]")
    for method in ("mahalanobis", "robust_mahalanobis",
                    "pca_residual", "pca_leverage",
                    "isolation_forest", "lof"):
        write_fixture(method, X, cases_for(method, X), out_dir)


if __name__ == "__main__":
    main()

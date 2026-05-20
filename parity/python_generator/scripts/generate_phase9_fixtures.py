#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 9 parity fixtures for the feature-selection /
dimensionality-reduction operators (FlexiblePCA, FlexibleSVD).

Phase 9 (partial): only the two SVD-based operators are covered here.
CARS and MCUVE require an internal PLS callback and are deferred.

Each operator exposes a count-mode case and a variance-ratio-mode case.
The fit matrix and transform matrix differ (neighbour seeds, same shape)
so the fit/transform split is properly exercised.

Output: ``parity/fixtures/{flexible_pca,flexible_svd}_v1.json``.

Fixture schema (additive over Phase 3 — per-case ``output_cols`` may
differ between cases when variance-ratio mode picks a different k'):

```json
{
  "format": "c4a_pp_<op>_v1",
  "numpy_version": "1.26.4",
  "nirs4all_version": "0.8.x",
  "encoding": "ieee754_binary64_be_hex",
  "fit_rows": 50,  "fit_cols": 200,
  "fit_input_hex": ["..."],
  "rows": 50,      "cols": 200,
  "input_hex": ["..."],
  "cases": [
    {"name": "...", "params": {"n_components": <number>},
     "output_rows": 50, "output_cols": <variable>,
     "output_hex": ["..."]},
    ...
  ]
}
```
"""

from __future__ import annotations

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
# Frozen NumPy reference (no nirs4all dependency).
# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_ROOT))

from c4a_parity_orthog_ref import (  # noqa: E402
    flexible_pca_fit_transform,
    flexible_svd_fit_transform,
)


# ---------------------------------------------------------------------------
# Fixture encoding helpers.
# ---------------------------------------------------------------------------
def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra (same generator family as Phase 3).
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
# Per-operator cases.
# ---------------------------------------------------------------------------
def flexible_pca_cases(fit_X, test_X):
    return [
        ("count_5", {"n_components": 5.0},
         lambda: flexible_pca_fit_transform(fit_X, test_X, n_components=5.0)),
        ("variance_0_95", {"n_components": 0.95},
         lambda: flexible_pca_fit_transform(fit_X, test_X, n_components=0.95)),
        ("count_15", {"n_components": 15.0},
         lambda: flexible_pca_fit_transform(fit_X, test_X, n_components=15.0)),
    ]


def flexible_svd_cases(fit_X, test_X):
    return [
        ("count_5", {"n_components": 5.0},
         lambda: flexible_svd_fit_transform(fit_X, test_X, n_components=5.0)),
        ("variance_0_99", {"n_components": 0.99},
         lambda: flexible_svd_fit_transform(fit_X, test_X, n_components=0.99)),
        ("count_15", {"n_components": 15.0},
         lambda: flexible_svd_fit_transform(fit_X, test_X, n_components=15.0)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------
def write_fixture(name: str, op_id: str,
                  fit_X: np.ndarray, transform_X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], tuple[np.ndarray, int]]]],
                  out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
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
    out = out_dir / f"{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    version = get_nirs4all_version(find_nirs4all_root())

    fit_X       = synthesize_spectra(SEED)
    transform_X = synthesize_spectra(SEED + 1)
    print(f"fit_X shape {fit_X.shape}, range [{fit_X.min():.4f}, {fit_X.max():.4f}]")
    print(f"transform_X shape {transform_X.shape}, range "
          f"[{transform_X.min():.4f}, {transform_X.max():.4f}]")

    write_fixture("flexible_pca", "flex_pca", fit_X, transform_X,
                  flexible_pca_cases(fit_X, transform_X), out_dir, version)
    write_fixture("flexible_svd", "flex_svd", fit_X, transform_X,
                  flexible_svd_cases(fit_X, transform_X), out_dir, version)


if __name__ == "__main__":
    main()

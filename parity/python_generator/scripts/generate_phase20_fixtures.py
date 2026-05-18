#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 20 parity fixtures for the transfer-metrics utility.

The reference comes from the frozen ``c4a_parity_transfer_ref`` package
(SplitMix64 deterministic choice — see the module banner for the rationale).

Fixture shape (one file, multiple cases — each case carries its own source
and target matrices because metric values depend on both):

  {
    "format": "c4a_transfer_metrics_v1",
    "numpy_version": "...",
    "reference": "c4a_parity_transfer_ref",
    "encoding": "ieee754_binary64_be_hex",
    "cases": [
      { "name": "...",
        "params": { "n_components": N, "k_neighbors": K, "seed": S },
        "source_rows": R, "source_cols": C,
        "source_hex": [ ... ],
        "target_rows": R, "target_cols": C,
        "target_hex": [ ... ],
        "expected": {
            "centroid_distance":    "<16 hex>",
            "cka_similarity":       "<16 hex>",
            "grassmann_distance":   "<16 hex>",
            "rv_coefficient":       "<16 hex>",
            "procrustes_disparity": "<16 hex>",
            "trustworthiness":      "<16 hex>",
            "spread_distance":      "<16 hex>",
            "evr_source":           "<16 hex>",
            "evr_target":           "<16 hex>"
        } },
      ...
    ]
  }

Cases sweep ``n_components`` (3 / 5 / 8), ``k_neighbors`` (5 / 10), ``seed``
(0 / 17), and dataset shape (aligned features and a mismatched-features
case to exercise the Grassmann NaN branch).
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

from c4a_parity_transfer_ref import transfer_metrics  # noqa: E402


SEED = 20260518
N_SOURCE = 40
N_TARGET = 35
P_FEATURES = 30


def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


def synthesize_pair(seed: int, n_src: int, n_tgt: int, p_src: int, p_tgt: int,
                    drift: float) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic (X_src, X_tgt) pair with controlled drift.

    Each row is a sum of a shared linear / quadratic baseline and two
    Gaussian peaks at varying positions. The target gets a constant `drift`
    added to its baseline and slightly broader peaks — enough to exercise
    non-trivial centroid / Grassmann / Procrustes signals.
    """
    rng = np.random.default_rng(seed)
    wl_src = np.linspace(0.0, 1.0, p_src, dtype=np.float64)
    wl_tgt = np.linspace(0.0, 1.0, p_tgt, dtype=np.float64)

    def make(wl: np.ndarray, n: int, broaden: float, shift: float) -> np.ndarray:
        rows = []
        for _ in range(n):
            a = rng.uniform(0.5, 1.5)
            b = rng.uniform(-0.4, 0.6)
            c = rng.uniform(-0.5, 0.5)
            baseline = a + b * wl + c * wl * wl + shift
            peak1_mu = rng.uniform(0.2, 0.45)
            peak1_sigma = rng.uniform(0.025, 0.06) * (1.0 + broaden)
            peak1_amp = rng.uniform(0.4, 1.5)
            peak2_mu = rng.uniform(0.6, 0.85)
            peak2_sigma = rng.uniform(0.03, 0.08) * (1.0 + broaden)
            peak2_amp = rng.uniform(0.3, 1.0)
            peaks = (
                peak1_amp * np.exp(-((wl - peak1_mu) ** 2) / (2.0 * peak1_sigma ** 2))
                + peak2_amp * np.exp(-((wl - peak2_mu) ** 2) / (2.0 * peak2_sigma ** 2))
            )
            noise = rng.normal(0.0, 0.005, size=wl.shape[0])
            rows.append(baseline + peaks + noise)
        return np.asarray(rows, dtype=np.float64)

    X_src = make(wl_src, n_src, broaden=0.0, shift=0.0)
    X_tgt = make(wl_tgt, n_tgt, broaden=0.15, shift=drift)
    return X_src, X_tgt


def build_case(name: str, X_src: np.ndarray, X_tgt: np.ndarray,
               n_components: int, k_neighbors: int, seed: int) -> dict[str, Any]:
    metrics = transfer_metrics(
        X_src, X_tgt,
        n_components=n_components,
        k_neighbors=k_neighbors,
        seed=seed,
    )
    return {
        "name": name,
        "params": {
            "n_components": int(n_components),
            "k_neighbors":  int(k_neighbors),
            "seed":         int(seed),
        },
        "source_rows": int(X_src.shape[0]),
        "source_cols": int(X_src.shape[1]),
        "source_hex":  array_to_hex(X_src),
        "target_rows": int(X_tgt.shape[0]),
        "target_cols": int(X_tgt.shape[1]),
        "target_hex":  array_to_hex(X_tgt),
        "expected": {
            "centroid_distance":    double_to_hex(metrics.centroid_distance),
            "cka_similarity":       double_to_hex(metrics.cka_similarity),
            "grassmann_distance":   double_to_hex(metrics.grassmann_distance),
            "rv_coefficient":       double_to_hex(metrics.rv_coefficient),
            "procrustes_disparity": double_to_hex(metrics.procrustes_disparity),
            "trustworthiness":      double_to_hex(metrics.trustworthiness),
            "spread_distance":      double_to_hex(metrics.spread_distance),
            "evr_source":           double_to_hex(metrics.evr_source),
            "evr_target":           double_to_hex(metrics.evr_target),
        },
    }


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []

    # Case 1: aligned features, default-ish settings.
    Xs, Xt = synthesize_pair(SEED, N_SOURCE, N_TARGET, P_FEATURES, P_FEATURES,
                              drift=0.2)
    cases.append(build_case("aligned_default", Xs, Xt,
                             n_components=10, k_neighbors=10, seed=0))

    # Case 2: aligned features, tight PCA truncation + small k.
    Xs, Xt = synthesize_pair(SEED + 1, N_SOURCE, N_TARGET,
                              P_FEATURES, P_FEATURES, drift=0.5)
    cases.append(build_case("aligned_tight", Xs, Xt,
                             n_components=3, k_neighbors=5, seed=17))

    # Case 3: aligned features, generous PCA truncation.
    Xs, Xt = synthesize_pair(SEED + 2, N_SOURCE, N_TARGET,
                              P_FEATURES, P_FEATURES, drift=0.05)
    cases.append(build_case("aligned_wide", Xs, Xt,
                             n_components=8, k_neighbors=8, seed=42))

    # Case 4: mismatched feature dimensions → Grassmann == NaN.
    Xs, Xt = synthesize_pair(SEED + 3, N_SOURCE, N_TARGET,
                              P_FEATURES, P_FEATURES - 5, drift=0.3)
    cases.append(build_case("mismatched_features", Xs, Xt,
                             n_components=5, k_neighbors=6, seed=7))

    # Case 5: small / sparse sample counts.
    Xs, Xt = synthesize_pair(SEED + 4, 12, 10, 15, 15, drift=0.4)
    cases.append(build_case("small_samples", Xs, Xt,
                             n_components=4, k_neighbors=3, seed=2026))

    fixture: dict[str, Any] = {
        "format":         "c4a_transfer_metrics_v1",
        "numpy_version":  np.__version__,
        "reference":      "c4a_parity_transfer_ref",
        "encoding":       "ieee754_binary64_be_hex",
        "cases":          cases,
    }
    out = out_dir / "transfer_metrics_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


if __name__ == "__main__":
    main()

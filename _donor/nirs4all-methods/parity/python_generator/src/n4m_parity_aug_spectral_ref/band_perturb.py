# SPDX-License-Identifier: CECILL-2.1
"""BandPerturbation parity reference (per-sample variation_scope)."""

from __future__ import annotations

import numpy as np


def band_perturb(X: np.ndarray, rng: np.random.Generator, *,
                 n_bands: int,
                 bw_lo: int, bw_hi: int,
                 gain_lo: float, gain_hi: float,
                 offset_lo: float, offset_hi: float) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    centers = rng.integers(0, n_features, size=(n_samples, n_bands))
    widths  = rng.integers(bw_lo, bw_hi, size=(n_samples, n_bands))
    gains   = rng.uniform(gain_lo, gain_hi, size=(n_samples, n_bands))
    offsets = rng.uniform(offset_lo, offset_hi, size=(n_samples, n_bands))
    out = X.copy()
    for i in range(n_samples):
        for b in range(n_bands):
            center = int(centers[i, b])
            width = int(widths[i, b])
            start = max(0, center - width // 2)
            end = min(n_features, center + width // 2)
            if start >= end:
                continue
            out[i, start:end] = out[i, start:end] * gains[i, b] + offsets[i, b]
    return out

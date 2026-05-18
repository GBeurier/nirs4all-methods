# SPDX-License-Identifier: CECILL-2.1
"""BandMasking parity reference."""

from __future__ import annotations

import numpy as np


def band_mask(X: np.ndarray, rng: np.random.Generator, *,
              n_bands_lo: int, n_bands_hi: int,
              bw_lo: int, bw_hi: int,
              mode: str) -> np.ndarray:
    """`mode` ∈ {"zero", "interp"}. The interp branch matches nirs4all's
    locally-linear ramp."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    n_per_sample = rng.integers(n_bands_lo, n_bands_hi + 1, size=n_samples)
    max_bands = n_bands_hi
    centers = rng.integers(0, n_features, size=(n_samples, max_bands))
    widths  = rng.integers(bw_lo, bw_hi, size=(n_samples, max_bands))
    out = X.copy()
    for i in range(n_samples):
        for b in range(int(n_per_sample[i])):
            center = int(centers[i, b])
            width = int(widths[i, b])
            start = max(0, center - width // 2)
            end = min(n_features, center + width // 2)
            if start >= end:
                continue
            if mode == "zero":
                out[i, start:end] = 0.0
            elif mode == "interp":
                val_start = out[i, start - 1] if start > 0 else out[i, start]
                val_end = out[i, end] if end < n_features else out[i, end - 1]
                x_local = np.arange(end - start)
                denom = (end - start + 1)
                slope = (val_end - val_start) / denom if denom > 0 else 0.0
                out[i, start:end] = val_start + slope * (x_local + 1)
            else:
                raise ValueError(f"unknown mode {mode}")
    return out

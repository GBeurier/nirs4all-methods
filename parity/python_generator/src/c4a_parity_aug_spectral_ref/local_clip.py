# SPDX-License-Identifier: CECILL-2.1
"""LocalClipping parity reference."""

from __future__ import annotations

import numpy as np


def local_clip(X: np.ndarray, rng: np.random.Generator, *,
               n_regions: int,
               width_lo: int, width_hi: int) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    centers = rng.integers(0, n_features, size=(n_samples, n_regions))
    widths  = rng.integers(width_lo, width_hi, size=(n_samples, n_regions))
    out = X.copy()
    for i in range(n_samples):
        for r in range(n_regions):
            center = int(centers[i, r])
            width = int(widths[i, r])
            start = max(0, center - width // 2)
            end = min(n_features, center + width // 2)
            if start >= end:
                continue
            limit = float(np.percentile(out[i, start:end], 90))
            out[i, start:end] = np.minimum(out[i, start:end], limit)
    return out

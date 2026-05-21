# SPDX-License-Identifier: CECILL-2.1
"""WaveletFeatures reference: 4-stat per-band features (mean, std, energy, entropy)."""

from __future__ import annotations

import numpy as np

from .dwt import wavedec_1d
from .filters import dwt_max_level


def _band_stats(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros(4, dtype=np.float64)
    mean = float(np.mean(x))
    std_pop = float(np.std(x))  # ddof=0
    energy = float(np.sum(x * x))
    entropy = 0.0
    if energy > 0.0:
        p = (x * x) / energy
        # Shannon entropy in natural log; zeros contribute 0 by convention.
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(p)
        logp = np.where(p > 0.0, logp, 0.0)
        entropy = float(-np.sum(p * logp))
    return np.array([mean, std_pop, energy, entropy], dtype=np.float64)


def wavelet_features_transform(
    X: np.ndarray, *,
    family: str,
    mode: str,
    max_level: int,
) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    rows, cols = X.shape
    eff = min(max_level, dwt_max_level(cols, family))
    out_cols = 4 * (eff + 1)
    out = np.zeros((rows, out_cols), dtype=np.float64)
    for i in range(rows):
        coeffs = wavedec_1d(X[i], family, mode, eff)
        for k in range(eff + 1):
            out[i, 4 * k:4 * k + 4] = _band_stats(coeffs[k])
    return out

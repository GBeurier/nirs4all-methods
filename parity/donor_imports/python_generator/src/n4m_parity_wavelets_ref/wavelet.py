# SPDX-License-Identifier: CECILL-2.1
"""Wavelet operator reference: single-level DWT (cA || cD per row)."""

from __future__ import annotations

import numpy as np

from .dwt import dwt_1d
from .filters import dwt_output_length


def wavelet_transform(X: np.ndarray, *, family: str, mode: str) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    rows, cols = X.shape
    m = dwt_output_length(cols, family, mode)
    out = np.zeros((rows, 2 * m), dtype=np.float64)
    for i in range(rows):
        cA, cD = dwt_1d(X[i], family, mode)
        out[i, :m] = cA
        out[i, m:] = cD
    return out

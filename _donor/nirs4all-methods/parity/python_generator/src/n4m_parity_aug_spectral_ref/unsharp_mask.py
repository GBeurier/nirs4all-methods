# SPDX-License-Identifier: CECILL-2.1
"""UnsharpSpectralMask parity reference."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from .gauss_jitter import _make_kernel


def unsharp_mask(X: np.ndarray, rng: np.random.Generator, *,
                 amount_lo: float, amount_hi: float,
                 sigma: float, kernel_width: int) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, _ = X.shape
    kernel = _make_kernel(sigma, kernel_width)
    smoothed = convolve1d(X, kernel, axis=1, mode="reflect")
    amounts = rng.uniform(amount_lo, amount_hi, size=(n_samples, 1))
    return X + amounts * (X - smoothed)

# SPDX-License-Identifier: CECILL-2.1
"""GaussianSmoothingJitter parity reference."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d


def _make_kernel(sigma: float, width: int) -> np.ndarray:
    """Build a normalized Gaussian kernel of fixed odd `width`."""
    if width % 2 == 0:
        width += 1
    x = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gauss_jitter(X: np.ndarray, rng: np.random.Generator, *,
                 sigma_lo: float, sigma_hi: float,
                 kernel_width: int) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, _ = X.shape
    sigmas = rng.uniform(sigma_lo, sigma_hi, size=n_samples)
    out = np.empty_like(X)
    for i in range(n_samples):
        kernel = _make_kernel(sigmas[i], kernel_width)
        out[i] = convolve1d(X[i], kernel, mode="reflect")
    return out

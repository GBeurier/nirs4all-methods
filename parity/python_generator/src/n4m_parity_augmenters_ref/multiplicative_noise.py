# SPDX-License-Identifier: CECILL-2.1
"""
MultiplicativeNoise — frozen NumPy reference.

Per-sample multiplicative gain:

    eps[i]      = sigma_gain * standard_normal()  # one draw per sample
    out[i, j]   = X[i, j] * (1 + eps[i])

The eps array is drawn from one ``rng.standard_normal(n_samples)`` call so
the RNG stream consumed matches the C side's
``standard_normal_fill(n_samples)``.
"""

from __future__ import annotations

import numpy as np


__all__ = ["multiplicative_noise_apply"]


def multiplicative_noise_apply(
    X: np.ndarray, *, rng: np.random.Generator, sigma_gain: float
) -> np.ndarray:
    """Apply multiplicative Gaussian gain to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    sigma_gain : float
        Standard deviation of the per-sample gain factor.

    Returns
    -------
    out : np.ndarray, same shape and dtype as X
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.ndim}-D")
    n_samples, _ = X.shape
    if n_samples == 0 or X.shape[1] == 0:
        return X.copy()

    eps = rng.standard_normal(size=n_samples) * sigma_gain
    gain = (1.0 + eps).reshape(-1, 1)
    return X * gain

# SPDX-License-Identifier: CECILL-2.1
"""
GaussianAdditiveNoise — frozen NumPy reference.

Per-sample std-scaled Gaussian noise:

    stds[i]      = std(X[i, :])                  # biased estimator, ddof=0
    noise[i, j]  = standard_normal(rows * cols)  # row-major reshape
    out[i, j]    = X[i, j] + noise[i, j] * stds[i] * sigma

The noise matrix is drawn from one
``rng.standard_normal((rows, cols))`` call so the RNG stream consumed
matches the C side's single ``standard_normal_fill(rows * cols)``.
"""

from __future__ import annotations

import numpy as np


__all__ = ["gaussian_noise_apply"]


def gaussian_noise_apply(
    X: np.ndarray, *, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    """Apply additive Gaussian noise to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
        Input spectra (row-major).
    rng : np.random.Generator
        PCG64 generator (seeded by the caller).
    sigma : float
        Relative noise scale; the per-sample std is multiplied by sigma to
        get the actual noise stddev.

    Returns
    -------
    out : np.ndarray, same shape and dtype as X
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.ndim}-D")
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return X.copy()

    # Per-sample std, biased (NumPy default ddof=0).
    stds = np.std(X, axis=1)  # shape (n_samples,)
    # One bulk draw, reshape row-major.
    noise = rng.standard_normal(size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    scale = (stds * sigma).reshape(-1, 1)
    return X + noise * scale

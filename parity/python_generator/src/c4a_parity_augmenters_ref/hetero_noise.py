# SPDX-License-Identifier: CECILL-2.1
"""
HeteroscedasticNoiseAugmenter — frozen NumPy reference.

Signal-dependent Gaussian noise:

    sigma[i, j] = noise_base + noise_signal_dep * |X[i, j]|
    noise[...]  = standard_normal((rows, cols))   # row-major
    out[i, j]   = X[i, j] + noise[i, j] * sigma[i, j]
"""

from __future__ import annotations

import numpy as np


__all__ = ["hetero_noise_apply"]


def hetero_noise_apply(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_base: float,
    noise_signal_dep: float,
) -> np.ndarray:
    """Apply signal-dependent (heteroscedastic) Gaussian noise.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    noise_base : float
        Signal-independent noise stddev.
    noise_signal_dep : float
        Signal-dependent noise coefficient.

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

    sigma = noise_base + noise_signal_dep * np.abs(X)
    noise = rng.standard_normal(size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    return X + noise * sigma

# SPDX-License-Identifier: CECILL-2.1
"""
PathLengthAugmenter — frozen NumPy reference.

Per-sample multiplicative path-length factor:

    L[i]      = 1.0 + path_length_std * standard_normal()
    L[i]      = max(L[i], min_path_length)
    out[i, j] = L[i] * X[i, j]
"""

from __future__ import annotations

import numpy as np


__all__ = ["path_length_apply"]


def path_length_apply(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    path_length_std: float,
    min_path_length: float,
) -> np.ndarray:
    """Apply per-sample path-length scaling to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    path_length_std : float
        Standard deviation of the per-sample factor (centered at 1.0).
    min_path_length : float
        Lower clip applied after the normal draw.

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

    normals = rng.standard_normal(size=n_samples)
    L = 1.0 + path_length_std * normals
    L = np.maximum(L, min_path_length)
    return X * L.reshape(-1, 1)

# SPDX-License-Identifier: CECILL-2.1
"""
PolynomialBaselineDrift — frozen NumPy reference.

Per-sample polynomial drift on a [-1, 1] normalized lambda axis:

    lambdas[j]   = np.linspace(-1, 1, n_features)[j]
    coeffs[i, k] = coeff_min[k] + (coeff_max[k] - coeff_min[k]) * u_ik
    drift[i, j]  = sum_{k=0}^{degree} coeffs[i, k] * lambdas[j]^k
    out[i, j]    = X[i, j] + drift[i, j]

RNG draw order (locked): order-major — for each order k in [0, degree],
draw n_samples uniforms. This matches nirs4all's
``for k: rng.uniform(lo_k, hi_k, size=(n_samples, 1))`` pattern.
"""

from __future__ import annotations

import numpy as np


__all__ = ["poly_drift_apply"]


def poly_drift_apply(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    coeff_min: list[float] | np.ndarray,
    coeff_max: list[float] | np.ndarray,
) -> np.ndarray:
    """Apply per-sample polynomial baseline drift to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    coeff_min, coeff_max : array-like of length (degree + 1)
        Per-order uniform ranges. ``coeff_min[k]`` and ``coeff_max[k]``
        bracket the order-k coefficient for every sample.

    Returns
    -------
    out : np.ndarray, same shape and dtype as X
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.ndim}-D")
    coeff_min_arr = np.ascontiguousarray(coeff_min, dtype=np.float64)
    coeff_max_arr = np.ascontiguousarray(coeff_max, dtype=np.float64)
    if coeff_min_arr.shape != coeff_max_arr.shape or coeff_min_arr.ndim != 1:
        raise ValueError("coeff_min / coeff_max must be 1-D and the same length")
    if np.any(coeff_max_arr < coeff_min_arr):
        raise ValueError("require coeff_min[k] <= coeff_max[k] for every k")

    n_coef = coeff_min_arr.shape[0]
    if n_coef == 0:
        return X.copy()

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return X.copy()

    # lambdas on [-1, 1] (NumPy linspace inclusive of endpoints).
    lambdas = np.linspace(-1.0, 1.0, n_features)

    # Draw coefficients in order-major form: coeffs[k, i].
    coeffs = np.empty((n_coef, n_samples), dtype=np.float64)
    for k in range(n_coef):
        u = rng.random(size=n_samples)
        coeffs[k, :] = coeff_min_arr[k] + (coeff_max_arr[k] - coeff_min_arr[k]) * u

    # Drift accumulation. We use the C-side recurrence (lj_pow *= lj) for
    # parity rather than np.polyval so the multiplication order matches.
    out = X.copy()
    for i in range(n_samples):
        drift = np.zeros(n_features, dtype=np.float64)
        lj_pow = np.ones(n_features, dtype=np.float64)
        for k in range(n_coef):
            drift += coeffs[k, i] * lj_pow
            lj_pow *= lambdas
        out[i, :] = X[i, :] + drift

    return out

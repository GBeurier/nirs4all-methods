# SPDX-License-Identifier: CECILL-2.1
"""
LinearBaselineDrift — frozen NumPy reference.

Per-sample affine baseline drift with uniform-drawn offsets and slopes:

    offsets[i] = offset_min + (offset_max - offset_min) * u_off_i
    slopes[i]  = slope_min  + (slope_max  - slope_min)  * u_slope_i
    out[i, j]  = X[i, j] + offsets[i] + slopes[i] * (j - (n_features - 1)/2)

where ``u_off_i, u_slope_i ~ Uniform[0, 1)`` are drawn from
``rng.random(...)`` (a.k.a. ``next_double`` in the PCG64 contract).

The wavelength axis defaults to ``arange(n_features)`` and is centered;
nirs4all subtracts the mean of ``arange``, which equals ``(n - 1) / 2``.

RNG draw order (locked):
  1. n_samples uniforms for offsets
  2. n_samples uniforms for slopes
"""

from __future__ import annotations

import numpy as np


__all__ = ["linear_drift_apply"]


def linear_drift_apply(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    offset_min: float,
    offset_max: float,
    slope_min: float,
    slope_max: float,
) -> np.ndarray:
    """Apply per-sample linear baseline drift to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    offset_min, offset_max : float
        Uniform range for the per-sample offset term.
    slope_min, slope_max : float
        Uniform range for the per-sample slope term.

    Returns
    -------
    out : np.ndarray, same shape and dtype as X
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.ndim}-D")
    if offset_max < offset_min or slope_max < slope_min:
        raise ValueError("require min <= max for offsets and slopes")

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return X.copy()

    off_span = offset_max - offset_min
    slope_span = slope_max - slope_min

    u_offsets = rng.random(size=n_samples)
    u_slopes = rng.random(size=n_samples)
    offsets = offset_min + off_span * u_offsets
    slopes = slope_min + slope_span * u_slopes

    mean_j = (n_features - 1) * 0.5
    lambdas_centered = np.arange(n_features, dtype=np.float64) - mean_j
    drift = offsets.reshape(-1, 1) + slopes.reshape(-1, 1) * lambdas_centered.reshape(1, -1)
    return X + drift

# SPDX-License-Identifier: CECILL-2.1
"""SmoothMagnitudeWarp parity reference (linear-interp variant)."""

from __future__ import annotations

import numpy as np


def magnitude_warp(X: np.ndarray, rng: np.random.Generator, *,
                   n_control_points: int,
                   gain_lo: float, gain_hi: float,
                   wavelengths: np.ndarray | None = None) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    lambdas = (np.asarray(wavelengths, dtype=np.float64) if wavelengths is not None
               else np.arange(n_features, dtype=np.float64))
    ctrl_x = np.linspace(lambdas[0], lambdas[-1], n_control_points)
    ctrl_y = rng.uniform(gain_lo, gain_hi,
                         size=(n_samples, n_control_points))
    out = np.empty_like(X)
    for i in range(n_samples):
        gains = np.interp(lambdas, ctrl_x, ctrl_y[i])
        out[i] = X[i] * gains
    return out

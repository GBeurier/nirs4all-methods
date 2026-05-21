# SPDX-License-Identifier: CECILL-2.1
"""LocalWavelengthWarp parity reference (linear-interp variant).

Uses `np.interp` for both the control-point->wavelength shift expansion and
the spectrum resampling; this is the deterministic parity floor that the C
engine reproduces bit-exactly under PCG64.
"""

from __future__ import annotations

import numpy as np


def local_warp(X: np.ndarray, rng: np.random.Generator, *,
               n_control_points: int, max_shift: float,
               wavelengths: np.ndarray | None = None) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    lambdas = (np.asarray(wavelengths, dtype=np.float64) if wavelengths is not None
               else np.arange(n_features, dtype=np.float64))
    ctrl_x = np.linspace(lambdas[0], lambdas[-1], n_control_points)
    ctrl_y = rng.uniform(-max_shift, max_shift,
                         size=(n_samples, n_control_points))
    out = np.empty_like(X)
    for i in range(n_samples):
        shift = np.interp(lambdas, ctrl_x, ctrl_y[i])
        query = lambdas - shift
        out[i] = np.interp(query, lambdas, X[i])
    return out

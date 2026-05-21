# SPDX-License-Identifier: CECILL-2.1
"""WavelengthStretch parity reference."""

from __future__ import annotations

import numpy as np


def wavelength_stretch(X: np.ndarray, rng: np.random.Generator, *,
                       stretch_lo: float, stretch_hi: float,
                       wavelengths: np.ndarray | None = None) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    lambdas = (np.asarray(wavelengths, dtype=np.float64) if wavelengths is not None
               else np.arange(n_features, dtype=np.float64))
    center = float(np.mean(lambdas))
    factors = rng.uniform(stretch_lo, stretch_hi, size=n_samples)
    out = np.empty_like(X)
    for i in range(n_samples):
        query = center + (lambdas - center) / factors[i]
        out[i] = np.interp(query, lambdas, X[i])
    return out

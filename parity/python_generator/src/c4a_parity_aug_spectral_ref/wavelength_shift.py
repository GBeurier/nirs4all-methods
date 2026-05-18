# SPDX-License-Identifier: CECILL-2.1
"""WavelengthShift parity reference."""

from __future__ import annotations

import numpy as np


def wavelength_shift(X: np.ndarray, rng: np.random.Generator, *,
                     shift_lo: float, shift_hi: float,
                     wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Shift the wavelength axis by `rng.uniform(shift_lo, shift_hi)` per
    sample, then resample each spectrum back onto the original grid via
    linear `np.interp`."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    lambdas = (np.asarray(wavelengths, dtype=np.float64) if wavelengths is not None
               else np.arange(n_features, dtype=np.float64))
    shifts = rng.uniform(shift_lo, shift_hi, size=n_samples)
    out = np.empty_like(X)
    for i in range(n_samples):
        query = lambdas - shifts[i]
        out[i] = np.interp(query, lambdas, X[i])
    return out

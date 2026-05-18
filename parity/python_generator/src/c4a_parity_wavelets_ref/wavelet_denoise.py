# SPDX-License-Identifier: CECILL-2.1
"""WaveletDenoise reference: multi-level DWT + universal threshold."""

from __future__ import annotations

import numpy as np

from .dwt import wavedec_1d, waverec_1d
from .filters import dwt_max_level


def _apply_threshold(c: np.ndarray, thr: float, mode: str) -> np.ndarray:
    if mode == "soft":
        a = np.abs(c)
        sgn = np.sign(c)
        return sgn * np.maximum(a - thr, 0.0)
    if mode == "hard":
        out = c.copy()
        out[np.abs(c) <= thr] = 0.0
        return out
    raise ValueError(f"unknown threshold mode: {mode}")


def wavelet_denoise_transform(
    X: np.ndarray, *,
    family: str,
    mode: str,
    level: int,
    threshold_mode: str = "soft",
    noise_estimator: str = "median",
) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    rows, cols = X.shape
    max_lvl = dwt_max_level(cols, family)
    eff_level = min(level, max_lvl)
    if eff_level < 1:
        return X.copy()
    out = np.zeros_like(X)
    universal_factor = np.sqrt(2.0 * np.log(cols))
    for i in range(rows):
        coeffs = wavedec_1d(X[i], family, mode, eff_level)
        finest = coeffs[-1]
        if noise_estimator == "median":
            sigma = float(np.median(np.abs(finest)) / 0.6745)
        elif noise_estimator == "std":
            sigma = float(np.std(finest))
        else:
            raise ValueError(f"unknown noise_estimator: {noise_estimator}")
        thr = sigma * universal_factor
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(_apply_threshold(c, thr, threshold_mode))
        out[i] = waverec_1d(new_coeffs, family, mode, cols)
    return out

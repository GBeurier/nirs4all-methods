# SPDX-License-Identifier: CECILL-2.1
"""
RollingBall — Kneen & Annegarn 1996 morphological baseline.

For each row of length N:

  e[i] = min over [max(0, i - W), min(N-1, i + W)] of y
  z[i] = max over [max(0, i - W), min(N-1, i + W)] of e
  if smooth_half_window > 0:
      z = moving-average of z over a (2 * S + 1) clipped centred window
  return y - z

Edges use a centred window clipped to [0, N - 1] (no padding / reflection).
The C engine matches this reference bit-for-bit.

Defaults: half_window = 20, smooth_half_window = 0.
"""

from __future__ import annotations

import numpy as np


def _min_filter(y: np.ndarray, w: int) -> np.ndarray:
    n = y.size
    out = np.empty_like(y)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n - 1, i + w)
        out[i] = y[lo:hi + 1].min()
    return out


def _max_filter(y: np.ndarray, w: int) -> np.ndarray:
    n = y.size
    out = np.empty_like(y)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n - 1, i + w)
        out[i] = y[lo:hi + 1].max()
    return out


def _moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    n = y.size
    out = np.empty_like(y)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n - 1, i + w)
        out[i] = y[lo:hi + 1].mean()
    return out


def rolling_ball(X: np.ndarray, *,
                  half_window: int = 20,
                  smooth_half_window: int = 0) -> np.ndarray:
    """
    Apply rolling-ball baseline correction row-wise to X.
    """
    if half_window < 1:
        raise ValueError("half_window must be >= 1")
    if smooth_half_window < 0:
        raise ValueError("smooth_half_window must be >= 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("rolling_ball requires a 2-D input")
    rows, _cols = X.shape

    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        z = _max_filter(_min_filter(y, half_window), half_window)
        if smooth_half_window > 0:
            z = _moving_avg(z, smooth_half_window)
        out[r] = y - z
    return out

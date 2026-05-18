# SPDX-License-Identifier: CECILL-2.1
"""
Polynomial detrending.

For each row x of X, fit ``np.polyfit(positions, x, polyorder)`` where
``positions = np.arange(n)`` (length-n integer grid), then subtract
``np.polyval(coefs, positions)`` to give the detrended row.

This matches:

  * scipy.signal.detrend(..., type="linear", axis=-1) when polyorder=1
  * the polynomial-detrend pattern used in pybaselines.polynomial.poly
    when called with weights=None and no exclusions.

Output shape == input shape; the operator is stateless (per-row, no fit).
"""

from __future__ import annotations

import numpy as np


def detrend(X: np.ndarray, *, polyorder: int = 1) -> np.ndarray:
    """
    Subtract a polynomial baseline of degree ``polyorder`` from each row of X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Input matrix. Each row is detrended independently.
    polyorder : int
        Polynomial degree (>= 0). Default 1 (linear detrend).

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        The baseline-subtracted matrix.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("detrend requires a 2-D input")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    rows, cols = X.shape
    positions = np.arange(cols, dtype=np.float64)
    out = np.empty_like(X)
    for i in range(rows):
        coefs = np.polyfit(positions, X[i], polyorder)
        baseline = np.polyval(coefs, positions)
        out[i] = X[i] - baseline
    return out

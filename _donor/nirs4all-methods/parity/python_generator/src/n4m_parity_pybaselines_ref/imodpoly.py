# SPDX-License-Identifier: CECILL-2.1
"""
Improved Modified Polynomial (IModPoly) baseline (Gan, Ruan, Mo 2006).

For each row y of length N and a polynomial of degree ``polyorder``:

  positions = arange(N).astype(float64)
  z = polyfit(positions, y, polyorder)
  devr = std(y - z, ddof=0)
  for iter in 0..max_iter:
      threshold = z + devr
      y = where(y > threshold, threshold, y)
      z = polyfit(positions, y, polyorder)
      devr_new = std(y - z, ddof=0)
      conv = |devr_new - devr| / max(devr_new, DBL_MIN)
      if conv < tol: break
      devr = devr_new
  return y_original - z

vs ModPoly: peak-clipping threshold uses (z + devr), not z; convergence is on
the residual stdev rather than on the baseline. The mask_initial_peaks
pre-step is disabled (matches pybaselines `num_std=1` default behaviour for
clean spectra and keeps the parity surface compact).
"""

from __future__ import annotations

import numpy as np

_MIN_FLOAT = np.finfo(np.float64).tiny


def _polyfit_eval(positions: np.ndarray, y: np.ndarray, polyorder: int) -> np.ndarray:
    coefs = np.polyfit(positions, y, polyorder)
    return np.polyval(coefs, positions)


def imodpoly(X: np.ndarray, *,
              polyorder: int = 2,
              max_iter: int = 250,
              tol: float = 1e-3) -> np.ndarray:
    """
    Apply iterative IModPoly baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Spectra. Each row baseline-corrected independently.
    polyorder : int
        Polynomial degree (>= 0). Default 2.
    max_iter : int
        Maximum reweighting iterations. Default 250.
    tol : float
        Convergence tolerance on relative change of residual stdev.

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        Baseline-subtracted matrix.
    """
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if max_iter < 0:
        raise ValueError("max_iter must be >= 0")
    if not tol >= 0:
        raise ValueError("tol must be >= 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("imodpoly requires a 2-D input")
    rows, cols = X.shape
    if cols < polyorder + 1:
        raise ValueError("imodpoly requires at least polyorder + 1 columns")

    positions = np.arange(cols, dtype=np.float64)
    out = np.empty_like(X)
    for i in range(rows):
        y0 = X[i]
        y = y0.copy()
        z = _polyfit_eval(positions, y, polyorder)
        devr = float(np.std(y - z, ddof=0))
        for _ in range(max_iter):
            threshold = z + devr
            y = np.where(y > threshold, threshold, y)
            z = _polyfit_eval(positions, y, polyorder)
            devr_new = float(np.std(y - z, ddof=0))
            denom = max(devr_new, _MIN_FLOAT)
            conv = abs(devr_new - devr) / denom
            devr = devr_new
            if conv < tol:
                break
        out[i] = y0 - z
    return out

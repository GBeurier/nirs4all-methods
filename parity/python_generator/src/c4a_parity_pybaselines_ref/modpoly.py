# SPDX-License-Identifier: CECILL-2.1
"""
Modified Polynomial (ModPoly) baseline (Lieber & Mahadevan-Jansen 2003).

For each row y of length N and a polynomial of degree ``polyorder``:

  positions = arange(N).astype(float64)
  z = polyfit(positions, y, polyorder) -> evaluate at positions
  for iter in 0..max_iter:
      y = minimum(y, z)
      z_new = polyfit(positions, y, polyorder)
      if rel_l2_diff(z, z_new) < tol: break
      z = z_new
  return y_original - z

The fitting basis is ``np.polyfit`` (descending powers, identical to the
``np.vander`` columns used by ``detrend.c``). pybaselines.modpoly uses a
[-1, 1]-mapped Vandermonde with ``np.polynomial.polynomial.polyvander``, but
the recovered baseline (z evaluated at the original positions) is invariant
under the change of basis — the parity check is against this self-consistent
NumPy reference, not against pybaselines directly.

Defaults match pybaselines 1.1.4: polyorder=2, max_iter=250, tol=1e-3.
"""

from __future__ import annotations

import numpy as np

_MIN_FLOAT = np.finfo(np.float64).tiny


def _rel_l2(old: np.ndarray, new: np.ndarray) -> float:
    """Match ``pybaselines.utils.relative_difference``."""
    num = float(np.linalg.norm(new - old))
    den = max(float(np.linalg.norm(old)), _MIN_FLOAT)
    return num / den


def _polyfit_eval(positions: np.ndarray, y: np.ndarray, polyorder: int) -> np.ndarray:
    coefs = np.polyfit(positions, y, polyorder)
    return np.polyval(coefs, positions)


def modpoly(X: np.ndarray, *,
             polyorder: int = 2,
             max_iter: int = 250,
             tol: float = 1e-3) -> np.ndarray:
    """
    Apply iterative ModPoly baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Spectra. Each row baseline-corrected independently.
    polyorder : int
        Polynomial degree (>= 0). Default 2.
    max_iter : int
        Maximum reweighting iterations. Default 250.
    tol : float
        Convergence tolerance on relative baseline change. Default 1e-3.

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
        raise ValueError("modpoly requires a 2-D input")
    rows, cols = X.shape
    if cols < polyorder + 1:
        raise ValueError("modpoly requires at least polyorder + 1 columns")

    positions = np.arange(cols, dtype=np.float64)
    out = np.empty_like(X)
    for i in range(rows):
        y0 = X[i]
        y = y0.copy()
        z = _polyfit_eval(positions, y, polyorder)
        for _ in range(max_iter):
            y = np.minimum(y, z)
            z_new = _polyfit_eval(positions, y, polyorder)
            if _rel_l2(z, z_new) < tol:
                z = z_new
                break
            z = z_new
        out[i] = y0 - z
    return out

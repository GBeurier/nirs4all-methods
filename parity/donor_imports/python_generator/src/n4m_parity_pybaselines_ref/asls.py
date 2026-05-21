# SPDX-License-Identifier: CECILL-2.1
"""
Asymmetric Least Squares (AsLS) baseline.

  Eilers, P. and Boelens, H.F.M. "Baseline correction with asymmetric least
  squares smoothing." Leiden University Medical Centre Report, 2005.

Algorithm:

  Given measured row y of length N, and parameters (lam, p, max_iter, tol):

    w := ones(N)
    for k in 0..max_iter:
        z := solve (diag(w) + lam * D_2^T D_2) z = w * y
        new_w[i] := p   if y[i] > z[i] else 1-p
        if rel_diff(w, new_w) < tol:  break
        w := new_w
    return baseline = z  (so detrended = y - z)

Notes:
  - Pybaselines returns the baseline `z`, not the residual.
  - Convergence uses relative_difference(w, new_w):
      ||new_w - w|| / max(||w||, _MIN_FLOAT)
  - At iteration 0 the initial weights are all 1 — the first solve gives
    a smoothed version of y.

Defaults match pybaselines 1.1.4: lam=1e6, p=1e-2, max_iter=50, tol=1e-3.
"""

from __future__ import annotations

import numpy as np

from ._banded import second_difference_penalty, solve_pls_system

_MIN_FLOAT = np.finfo(np.float64).tiny


def _relative_difference(old: np.ndarray, new: np.ndarray) -> float:
    """Match pybaselines.utils.relative_difference (l2 norm, _MIN_FLOAT denom)."""
    num = np.linalg.norm(new - old)
    den = max(np.linalg.norm(old), _MIN_FLOAT)
    return num / den


def asls(X: np.ndarray, *,
          lam: float = 1e6, p: float = 1e-2,
          max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """
    Apply AsLS baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Measured spectra. Each row baseline-corrected independently.
    lam : float
        Smoothing penalty (default 1e6).
    p : float
        Asymmetric weight (0 < p < 1). Default 1e-2. Points above the
        baseline get weight `p`; below get `1-p`.
    max_iter : int
        Maximum reweighting iterations (default 50).
    tol : float
        Convergence tolerance on the relative weight change (default 1e-3).

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        The baseline-subtracted matrix: ``Y[i] = X[i] - baseline_i``.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if lam <= 0:
        raise ValueError("lam must be > 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("asls requires a 2-D input")
    rows, cols = X.shape
    if cols < 3:
        raise ValueError("asls requires at least 3 wavelengths")

    penalty = second_difference_penalty(cols, lam)
    out = np.empty_like(X)
    for i in range(rows):
        y = X[i]
        w = np.ones(cols, dtype=np.float64)
        baseline = np.zeros_like(y)
        for _ in range(max_iter + 1):
            baseline = solve_pls_system(penalty, w, y)
            new_w = np.where(y > baseline, p, 1.0 - p)
            if _relative_difference(w, new_w) < tol:
                break
            w = new_w
        out[i] = y - baseline
    return out

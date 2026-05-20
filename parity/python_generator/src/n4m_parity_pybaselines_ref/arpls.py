# SPDX-License-Identifier: CECILL-2.1
"""
Asymmetrically Reweighted Penalized Least Squares (arPLS) baseline.

  Baek, S.-J., Park, A., Ahn, Y.-J., Choo, J. "Baseline correction using
  asymmetrically reweighted penalized least squares smoothing." Analyst,
  2015, 140, 250-257.

Algorithm:

  w := ones(N)
  for i = 0 .. max_iter:
      z := solve (diag(w) + lam * D_2^T D_2) z = w * y
      d := y - z
      neg_residual := d[d < 0]
      if neg_residual.size < 2: break   # exit_early
      std := std(neg_residual, ddof=1)  (clamped to _MIN_FLOAT)
      new_w := logistic(-(2/std) * (d - (2*std - mean(neg_residual))))
            = 1 / (1 + exp((2/std) * (d - (2*std - mean(neg_residual)))))
      if rel_diff(w, new_w) < tol: break
      w := new_w
  return baseline = z

Defaults match pybaselines 1.1.4: lam=1e5, max_iter=50, tol=1e-3.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ._banded import second_difference_penalty, solve_pls_system

_MIN_FLOAT = np.finfo(np.float64).tiny


def _relative_difference(old: np.ndarray, new: np.ndarray) -> float:
    num = np.linalg.norm(new - old)
    den = max(np.linalg.norm(old), _MIN_FLOAT)
    return num / den


def _safe_std(neg_residual: np.ndarray) -> float:
    """Match pybaselines._weighting._safe_std (ddof=1, clamped)."""
    if neg_residual.size < 2:
        return _MIN_FLOAT
    std = neg_residual.std(ddof=1)
    if std == 0.0:
        return _MIN_FLOAT
    return float(std)


def _arpls_weights(y: np.ndarray, baseline: np.ndarray):
    """Return (new_weights, exit_early). Mirrors pybaselines._weighting._arpls."""
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        return np.zeros_like(y), True
    std = _safe_std(neg_residual)
    # expit(x) = 1 / (1 + exp(-x))
    weights = expit(-(2.0 / std) * (residual - (2.0 * std - np.mean(neg_residual))))
    return weights, False


def arpls(X: np.ndarray, *,
           lam: float = 1e5,
           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """
    Apply arPLS baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Measured spectra. Each row baseline-corrected independently.
    lam : float
        Smoothing penalty (default 1e5).
    max_iter : int
        Maximum reweighting iterations (default 50).
    tol : float
        Convergence tolerance on the relative weight change (default 1e-3).

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        The baseline-subtracted matrix: ``Y[i] = X[i] - baseline_i``.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("arpls requires a 2-D input")
    rows, cols = X.shape
    if cols < 3:
        raise ValueError("arpls requires at least 3 wavelengths")

    penalty = second_difference_penalty(cols, lam)
    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        w = np.ones(cols, dtype=np.float64)
        baseline = np.zeros_like(y)
        for _ in range(max_iter + 1):
            baseline = solve_pls_system(penalty, w, y)
            new_w, exit_early = _arpls_weights(y, baseline)
            if exit_early:
                break
            if _relative_difference(w, new_w) < tol:
                break
            w = new_w
        out[r] = y - baseline
    return out

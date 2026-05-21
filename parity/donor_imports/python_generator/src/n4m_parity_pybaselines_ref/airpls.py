# SPDX-License-Identifier: CECILL-2.1
"""
Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) baseline.

  Zhang, Z.M., Chen, S., Liang, Y.Z. "Baseline correction using adaptive
  iteratively reweighted penalized least squares." Analyst, 2010, 135(5),
  1138-1146.

Algorithm:

  y_l1_norm := |y|_1
  w := ones(N)
  for i = 1 .. max_iter+1:
      z := solve (diag(w) + lam * D_2^T D_2) z = w * y
      d := y - z
      neg_mask := d < 0
      neg_residual := d[neg_mask]                   # negative residuals
      if neg_residual.size < 2:                     # convergence (exit_early)
          break
      residual_l1_norm := sum(neg_residual)         # signed, negative
      inner := clip( (min(i, 50) / residual_l1_norm) * neg_residual,
                     0, log(max_dtype) - spacing )  # uses signed l1_norm
      new_w := zeros(N); new_w[neg_mask] := exp(inner)
      if |residual_l1_norm| / y_l1_norm < tol:      # convergence
          break
      w := new_w

  Return baseline = z.

Defaults match pybaselines 1.1.4: lam=1e6, max_iter=50, tol=1e-3.
"""

from __future__ import annotations

import numpy as np

from ._banded import second_difference_penalty, solve_pls_system


def _airpls_weights(y: np.ndarray, baseline: np.ndarray, iteration: int):
    """Return (new_weights, residual_l1_norm_abs, exit_early)."""
    residual = y - baseline
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    if neg_residual.size < 2:
        return np.zeros_like(y), 0.0, True
    residual_l1_norm = neg_residual.sum()  # signed (negative)
    # min(iteration, 50): pybaselines caps to avoid overflow at high iters.
    log_max = np.log(np.finfo(y.dtype).max)
    inner = np.clip(
        (min(iteration, 50) / residual_l1_norm) * neg_residual,
        a_min=0.0,
        a_max=log_max - np.spacing(log_max),
    )
    weights = np.zeros_like(y)
    weights[neg_mask] = np.exp(inner)
    return weights, abs(residual_l1_norm), False


def airpls(X: np.ndarray, *,
            lam: float = 1e6,
            max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """
    Apply airPLS baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Measured spectra. Each row baseline-corrected independently.
    lam : float
        Smoothing penalty (default 1e6).
    max_iter : int
        Maximum reweighting iterations (default 50).
    tol : float
        Exit criterion on |residual_l1_norm| / |y|_1 (default 1e-3).

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        The baseline-subtracted matrix: ``Y[i] = X[i] - baseline_i``.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("airpls requires a 2-D input")
    rows, cols = X.shape
    if cols < 3:
        raise ValueError("airpls requires at least 3 wavelengths")

    penalty = second_difference_penalty(cols, lam)
    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        y_l1_norm = np.abs(y).sum()
        w = np.ones(cols, dtype=np.float64)
        baseline = np.zeros_like(y)
        for i in range(1, max_iter + 2):
            baseline = solve_pls_system(penalty, w, y)
            new_w, residual_l1_norm_abs, exit_early = _airpls_weights(y, baseline, i)
            if exit_early:
                break
            if y_l1_norm > 0 and (residual_l1_norm_abs / y_l1_norm) < tol:
                break
            w = new_w
        out[r] = y - baseline
    return out

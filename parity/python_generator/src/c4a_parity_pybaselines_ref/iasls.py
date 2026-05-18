# SPDX-License-Identifier: CECILL-2.1
"""
IAsLS — Improved Asymmetric Least Squares (He et al. 2014).

For each row y of length N:

  positions = arange(N).astype(float64)
  z0 = polyfit(positions, y, polyorder)    # initial polynomial baseline
  w[i] = p if y[i] > z0[i] else (1 - p)    # AsLS-style init from z0
  for iter in 0..max_iter:
      z = solve (diag(w) + lam D_2^T D_2) z = w * y
      w_new[i] = p if y[i] > z[i] else (1 - p)
      if rel_l2_diff(w, w_new) < tol: break
      w = w_new
  return y - z

The only difference from the Phase 5a AsLS reference is the weight
initialisation step (polynomial pre-fit). The penalty / solve / convergence
test reuse the existing ``_banded`` helpers.

Defaults: lam=1e6, p=1e-2, polyorder=2, max_iter=50, tol=1e-3.
"""

from __future__ import annotations

import numpy as np

from ._banded import second_difference_penalty, solve_pls_system

_MIN_FLOAT = np.finfo(np.float64).tiny


def _rel_l2(old: np.ndarray, new: np.ndarray) -> float:
    num = float(np.linalg.norm(new - old))
    den = max(float(np.linalg.norm(old)), _MIN_FLOAT)
    return num / den


def iasls(X: np.ndarray, *,
           lam: float = 1e6, p: float = 1e-2,
           polyorder: int = 2,
           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("iasls requires a 2-D input")
    rows, cols = X.shape
    if cols < 3 or cols < polyorder + 1:
        raise ValueError("iasls requires at least 3 wavelengths and "
                         "polyorder + 1 points")

    penalty = second_difference_penalty(cols, lam)
    positions = np.arange(cols, dtype=np.float64)
    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        coefs = np.polyfit(positions, y, polyorder)
        z0 = np.polyval(coefs, positions)
        w = np.where(y > z0, p, 1.0 - p)
        baseline = np.zeros_like(y)
        for _ in range(max_iter + 1):
            baseline = solve_pls_system(penalty, w, y)
            new_w = np.where(y > baseline, p, 1.0 - p)
            if _rel_l2(w, new_w) < tol:
                break
            w = new_w
        out[r] = y - baseline
    return out

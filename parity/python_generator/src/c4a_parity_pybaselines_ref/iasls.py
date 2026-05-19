# SPDX-License-Identifier: CECILL-2.1
"""
IAsLS — Improved Asymmetric Least Squares (He et al. 2014).

For each row y of length N:

  positions = arange(N).astype(float64)
  z0 = polyfit(positions, y, polyorder)    # initial polynomial baseline
  w[i] = p if y[i] > z0[i] else (1 - p)    # AsLS-style init from z0
  d1_y = lam_1 * (D_1.T @ D_1) @ y
  for iter in 0..max_iter:
      z = solve (diag(w**2) + lam D_2^T D_2 + lam_1 D_1^T D_1) z
                = w**2 * y + d1_y
      w_new[i] = p if y[i] > z[i] else (1 - p)
      if rel_l2_diff(w, w_new) < tol: break
      w = w_new
  return y - z

This matches ``pybaselines.whittaker.Baseline.iasls`` for the supported
``diff_order=2`` contract.  The public c4a surface keeps ``polyorder`` for
compatibility, but pybaselines' default initial fit uses degree 2.

Defaults: lam=1e6, p=1e-2, lam_1=1e-4, polyorder=2, diff_order=2,
max_iter=50, tol=1e-3.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._banded import second_difference_penalty

_MIN_FLOAT = np.finfo(np.float64).tiny


def _rel_l2(old: np.ndarray, new: np.ndarray) -> float:
    num = float(np.linalg.norm(new - old))
    den = max(float(np.linalg.norm(old)), _MIN_FLOAT)
    return num / den


def iasls(X: np.ndarray, *,
           lam: float = 1e6, p: float = 1e-2,
           lam_1: float = 1e-4,
           polyorder: int = 2, diff_order: int = 2,
           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if lam_1 <= 0:
        raise ValueError("lam_1 must be > 0")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if diff_order != 2:
        raise ValueError("only diff_order=2 is supported by this fixture reference")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("iasls requires a 2-D input")
    rows, cols = X.shape
    if cols < 3 or cols < polyorder + 1:
        raise ValueError("iasls requires at least 3 wavelengths and "
                         "polyorder + 1 points")

    penalty = second_difference_penalty(cols, lam)
    d1_main = np.full(cols, 2.0, dtype=np.float64)
    d1_main[0] = d1_main[-1] = 1.0
    d1_off = np.full(cols - 1, -1.0, dtype=np.float64)
    penalty = (penalty + lam_1 * sp.diags(
        [d1_off, d1_main, d1_off], offsets=[-1, 0, 1],
        shape=(cols, cols), format="csc")).tocsc()
    positions = np.arange(cols, dtype=np.float64)
    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        coefs = np.polyfit(positions, y, polyorder)
        z0 = np.polyval(coefs, positions)
        w = np.where(y > z0, p, 1.0 - p)
        d1_y = y.copy()
        d1_y[0] = y[0] - y[1]
        d1_y[-1] = y[-1] - y[-2]
        d1_y[1:-1] = 2 * y[1:-1] - y[:-2] - y[2:]
        d1_y *= lam_1
        baseline = np.zeros_like(y)
        for _ in range(max_iter + 1):
            w2 = w**2
            A = (sp.diags(w2, 0, shape=(cols, cols), format="csc") + penalty).tocsc()
            baseline = spla.spsolve(A, w2 * y + d1_y)
            new_w = np.where(y > baseline, p, 1.0 - p)
            if _rel_l2(w, new_w) < tol:
                break
            w = new_w
        out[r] = y - baseline
    return out

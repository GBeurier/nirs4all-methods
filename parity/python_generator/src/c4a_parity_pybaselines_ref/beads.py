# SPDX-License-Identifier: CECILL-2.1
"""
BEADS (simplified) — Baseline Estimation And Denoising with Sparsity
(Ning & Selesnick 2014).

Phase 5b ships the simplified pentadiagonal variant: a reweighted-L2
sparsity surrogate over the same ``D_2^T D_2`` system used by AsLS, scaled
by ``(lam_1 + lam_2)``. The full BEADS algorithm uses a 7-diagonal system
combining 1st- and 2nd-difference penalties with a Chebyshev approximation
of ``|.|``; see ``docs/algorithms/beads.md`` for the deferred variant.

  For each row y of length N:
    w := ones(N)
    z := zeros(N)
    for iter in 0..max_iter:
        A := diag(lam_0 * w) + (lam_1 + lam_2) * D_2^T D_2
        rhs := lam_0 * w * y
        z_new := A^{-1} rhs
        d := y - z_new
        w[i] := 1 / (|d[i]| + eps)
        w := w * N / sum(w)         # renormalise to mean 1
        if rel_l2_diff(z, z_new) < tol: break
        z := z_new
    return y - z

eps = 1e-3. Defaults: lam_0=1e2, lam_1=0.5, lam_2=0.5, max_iter=50,
tol=1e-3.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._banded import second_difference_penalty

_MIN_FLOAT = np.finfo(np.float64).tiny
_BEADS_EPS = 1.0e-3


def _rel_l2(old: np.ndarray, new: np.ndarray) -> float:
    num = float(np.linalg.norm(new - old))
    den = max(float(np.linalg.norm(old)), _MIN_FLOAT)
    return num / den


def beads(X: np.ndarray, *,
           lam_0: float = 1e2, lam_1: float = 0.5, lam_2: float = 0.5,
           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    if not (lam_0 > 0 and lam_1 > 0 and lam_2 > 0):
        raise ValueError("lam_0 / lam_1 / lam_2 must be > 0")
    if max_iter < 0 or not tol >= 0:
        raise ValueError("max_iter must be >= 0 and tol >= 0")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("beads requires a 2-D input")
    rows, cols = X.shape
    if cols < 3:
        raise ValueError("beads requires at least 3 wavelengths")

    penalty = second_difference_penalty(cols, lam_1 + lam_2)
    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        w = np.ones(cols, dtype=np.float64)
        z = np.zeros(cols, dtype=np.float64)
        for _ in range(max_iter + 1):
            diag_w = sp.diags(lam_0 * w, 0, shape=(cols, cols), format="csc")
            A = (diag_w + penalty).tocsc()
            rhs = lam_0 * w * y
            z_new = spla.spsolve(A, rhs)
            d = y - z_new
            w = 1.0 / (np.abs(d) + _BEADS_EPS)
            w_sum = float(w.sum())
            if w_sum > _MIN_FLOAT:
                w *= float(cols) / w_sum
            rdiff = _rel_l2(z, z_new)
            z = z_new
            if rdiff < tol:
                break
        out[r] = y - z
    return out

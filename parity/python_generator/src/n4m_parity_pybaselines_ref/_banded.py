# SPDX-License-Identifier: CECILL-2.1
"""
Helpers shared by the AsLS / AirPLS / ArPLS reference implementations.

The Whittaker baseline penalty is `lam * D_n^T D_n` where `D_n` is the n-th
order forward difference matrix.  For all four ops in Phase 5a we use
`diff_order = 2`, which produces a pentadiagonal symmetric `D^T D`.

We provide:

  * ``second_difference_penalty(n, lam)``: builds the pentadiagonal
    `lam * D_2^T D_2` matrix as a scipy.sparse.dia_matrix.  Stored
    sparse because pybaselines uses scipy.sparse + splu under the
    hood — keeping the same data path makes parity checks robust to
    fill-in / pivot reordering.

  * ``solve_pls_system(D2TD2, w, y, lam)``: solves
    `(diag(w) + lam D_2^T D_2) z = w * y` and returns z.

All values are float64.  No state is carried between calls.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def second_difference_matrix(n: int) -> sp.csr_matrix:
    """
    Build the (n-2, n) second-order forward difference matrix D_2 such that

        (D_2 y)[i] = y[i+2] - 2*y[i+1] + y[i]    for i = 0 .. n-3.

    Matches scipy ``np.diff(..., n=2)`` for n >= 3.
    """
    if n < 3:
        raise ValueError("second_difference_matrix requires n >= 3")
    main = np.ones(n - 2, dtype=np.float64)
    rows = np.repeat(np.arange(n - 2), 3)
    cols = np.concatenate([
        np.arange(n - 2),
        np.arange(1, n - 1),
        np.arange(2, n),
    ])
    cols = np.stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1).ravel()
    data = np.tile([1.0, -2.0, 1.0], n - 2)
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n - 2, n),
    ) * main[0]  # noqa: redundant *1


def second_difference_penalty(n: int, lam: float) -> sp.csc_matrix:
    """
    Build ``lam * D_2^T D_2`` as a (n, n) pentadiagonal symmetric CSC matrix.
    """
    D2 = second_difference_matrix(n)
    return (lam * (D2.T @ D2)).tocsc()


def solve_pls_system(penalty: sp.csc_matrix, weights: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
    """
    Solve ``(diag(weights) + penalty) z = weights * y``.

    Uses scipy.sparse.linalg.splu under the hood (matches pybaselines).
    """
    n = weights.size
    diag_w = sp.diags(weights, 0, shape=(n, n), format="csc")
    A = (diag_w + penalty).tocsc()
    rhs = weights * y
    return spla.spsolve(A, rhs)

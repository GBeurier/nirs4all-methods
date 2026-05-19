# SPDX-License-Identifier: CECILL-2.1
"""BEADS — Baseline Estimation And Denoising with Sparsity.

The fixture contract follows ``pybaselines.Baseline().beads`` directly and
returns the baseline-corrected signal ``X - baseline`` row-wise. Keeping the
fixture generator tied to the external reference avoids freezing the former
simplified c4a variant into future parity data.
"""

from __future__ import annotations

import numpy as np
import pybaselines


def beads(X: np.ndarray, *,
           lam_0: float = 1e2, lam_1: float = 0.5, lam_2: float = 0.5,
           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("beads requires a 2-D input")

    baseline = pybaselines.Baseline()
    rows = []
    for row in X:
        z, _ = baseline.beads(
            row,
            lam_0=lam_0,
            lam_1=lam_1,
            lam_2=lam_2,
            max_iter=max_iter,
            tol=tol,
        )
        rows.append(row - z)
    return np.asarray(rows, dtype=np.float64)

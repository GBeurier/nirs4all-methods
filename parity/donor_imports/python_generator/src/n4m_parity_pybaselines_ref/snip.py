# SPDX-License-Identifier: CECILL-2.1
"""
SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping baseline.

  Ryan, C., et al. SNIP: A statistics-sensitive background treatment for the
  quantitative analysis of PIXE spectra in geoscience applications.
  Nuclear Instruments and Methods, 1988, B34, 396-402.

  Morháč, M. An algorithm for determination of peak regions and baseline
  elimination in spectroscopic data. Nuclear Instruments and Methods, 2008.

The fixture contract follows ``pybaselines.Baseline().snip`` directly. The
function returns the baseline-corrected signal ``X - baseline`` row-wise, which
is the public n4m operator surface.
"""

from __future__ import annotations

import numpy as np
import pybaselines


def snip(X: np.ndarray, *,
         max_half_window: int = 20) -> np.ndarray:
    """
    Apply SNIP baseline correction row-wise to X.

    Parameters
    ----------
    X : array-like, shape (rows, cols)
        Spectra. Each row baseline-corrected independently.
    max_half_window : int
        Maximum half-window for the iterative clip step. Default 20.

    Returns
    -------
    Y : ndarray, shape (rows, cols)
        Baseline-subtracted matrix.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("snip requires a 2-D input")

    baseline = pybaselines.Baseline()
    rows = []
    for row in X:
        z, _ = baseline.snip(row, max_half_window=max_half_window)
        rows.append(row - z)
    out = np.asarray(rows, dtype=np.float64)
    return out

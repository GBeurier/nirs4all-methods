# SPDX-License-Identifier: CECILL-2.1
"""
SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping baseline.

  Ryan, C., et al. SNIP: A statistics-sensitive background treatment for the
  quantitative analysis of PIXE spectra in geoscience applications.
  Nuclear Instruments and Methods, 1988, B34, 396-402.

  Morháč, M. An algorithm for determination of peak regions and baseline
  elimination in spectroscopic data. Nuclear Instruments and Methods, 2008
  (LLS transform refinement).

For each row of length N:

  v[i] = log(log(sqrt(|y[i]| + 1) + 1) + 1)                # LLS transform
  for w in 1..max_half_window:
      for i in [w, N - w):
          v[i] = min(v[i], (v[i - w] + v[i + w]) / 2)
  # inverse LLS:
  c = exp(v); b = exp(c - 1); a = (b - 1)**2; z = a - 1
  return y - z

Defaults: max_half_window = 20. The C engine matches this reference
bit-for-bit (pure arithmetic, no LDLT).
"""

from __future__ import annotations

import numpy as np


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
    if max_half_window < 1:
        raise ValueError("max_half_window must be >= 1")

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("snip requires a 2-D input")
    rows, cols = X.shape

    out = np.empty_like(X)
    for r in range(rows):
        y = X[r]
        v = np.log(np.log(np.sqrt(np.abs(y) + 1.0) + 1.0) + 1.0)
        for w in range(1, max_half_window + 1):
            if cols <= 2 * w:
                continue
            half = v.copy()
            avg = 0.5 * (half[: cols - 2 * w] + half[2 * w:])
            # Apply in-place — match the C engine which mutates v sequentially.
            for i in range(w, cols - w):
                a = 0.5 * (v[i - w] + v[i + w])
                if a < v[i]:
                    v[i] = a
            del half, avg
        # Inverse LLS.
        c = np.exp(v)
        b = np.exp(c - 1.0)
        a = (b - 1.0) ** 2
        z = a - 1.0
        out[r] = y - z
    return out

# SPDX-License-Identifier: CECILL-2.1
"""
FlexiblePCA — PCA with flexible component specification.

Reference: nirs4all.operators.transforms.feature_selection.FlexiblePCA,
which wraps sklearn's PCA. This frozen reference is a pure NumPy
implementation that matches the c4a engine exactly:

  1. Centre X by subtracting per-column mean.
  2. Compute the compact SVD of the centered matrix.
  3. Apply ``svd_flip(U, Vt, u_based_decision=True)``: for each
     component i, flip the sign of U[:, i] (and Vt[i, :] in lockstep)
     so the element of U[:, i] with the largest absolute value is
     positive. NumPy's ``argmax`` is used for the index — ties are
     broken by smallest index.
  4. Compute explained variance ``S^2 / (n_samples - 1)`` and the
     normalised ratio.
  5. Select the number of components:
       - if ``n_components_param >= 1`` (count mode), keep that many
         (clamped to ``k = min(m, n)``);
       - if ``n_components_param in (0, 1)`` (variance-ratio mode),
         keep the smallest k' s.t. ``cumsum(ratio)[k'-1] >= ratio``.
  6. Transform inputs as ``(X - mean) @ components_.T`` where
     ``components_ = Vt[:k', :]``.

The reference is frozen — it intentionally does NOT depend on sklearn
so upstream sklearn changes can't drift the parity fixture.
"""

from __future__ import annotations

import numpy as np


def _svd_flip_u_based(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign-canonicalise so each U column has its largest-abs entry positive.

    Mirrors ``sklearn.utils.extmath.svd_flip(U, Vt, u_based_decision=True)``.
    Ties (multiple equal-magnitude entries) are broken by smallest index,
    matching ``np.argmax`` on ``|U|``.
    """
    # max_abs_u_cols: index of largest |U[:, j]| per column, length k.
    max_abs_u_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_u_cols, np.arange(U.shape[1])])
    # Convert 0.0 sign (impossible for non-zero columns but defensive) to +1.
    signs = np.where(signs == 0.0, 1.0, signs)
    U = U * signs[np.newaxis, :]
    Vt = Vt * signs[:, np.newaxis]
    return U, Vt


def _select_k(param: float, k_full: int, var_ratio: np.ndarray) -> int:
    """Resolve the flexible n_components to an actual k'."""
    if param >= 1.0:
        # Count mode (sklearn truncates float >= 1 to int).
        k = int(param)
        return max(1, min(k, k_full))
    # Variance-ratio mode.
    cum = np.cumsum(var_ratio)
    # Smallest k' such that cum[k'-1] >= param. searchsorted with
    # side='left' returns first index with cum[i] >= param; +1 -> count.
    idx = int(np.searchsorted(cum, param, side='left'))
    if idx >= k_full:
        return k_full
    return idx + 1


def flexible_pca_fit_transform(
    fit_X: np.ndarray,
    transform_X: np.ndarray,
    *,
    n_components: float,
) -> tuple[np.ndarray, int]:
    """
    Fit FlexiblePCA on ``fit_X`` and transform ``transform_X``.

    Parameters
    ----------
    fit_X : array-like, shape (m, n)
        Training matrix. ``m >= 2``.
    transform_X : array-like, shape (m_t, n)
        Matrix to project onto the fitted components.
    n_components : float
        Flexible specifier:
          - ``>= 1.0`` -> count mode (truncated to int)
          - ``in (0.0, 1.0)`` -> variance-ratio mode

    Returns
    -------
    out : ndarray, shape (m_t, k')
        Projected matrix.
    k_kept : int
        Learned component count.
    """
    fit_X = np.ascontiguousarray(fit_X, dtype=np.float64)
    transform_X = np.ascontiguousarray(transform_X, dtype=np.float64)
    if fit_X.ndim != 2 or transform_X.ndim != 2:
        raise ValueError("FlexiblePCA requires 2-D matrices")
    if transform_X.shape[1] != fit_X.shape[1]:
        raise ValueError("transform_X and fit_X must share n_features")

    m, n = fit_X.shape
    if m < 2 or n < 1:
        raise ValueError("FlexiblePCA: fit_X requires m >= 2 and n >= 1")

    mean = fit_X.mean(axis=0)
    Xc = fit_X - mean

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U, Vt = _svd_flip_u_based(U, Vt)

    k_full = min(m, n)
    evar = (S ** 2) / (m - 1)
    total = float(evar.sum())
    var_ratio = (evar / total) if total > 0.0 else np.zeros_like(evar)

    k_kept = _select_k(float(n_components), k_full, var_ratio)
    components = Vt[:k_kept, :]

    out = (transform_X - mean) @ components.T
    return np.ascontiguousarray(out, dtype=np.float64), k_kept

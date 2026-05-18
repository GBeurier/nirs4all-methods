# SPDX-License-Identifier: CECILL-2.1
"""
FlexibleSVD — Truncated SVD with flexible component specification.

Reference: nirs4all.operators.transforms.feature_selection.FlexibleSVD,
which wraps sklearn's TruncatedSVD. This frozen reference is a pure
NumPy implementation matching the c4a engine:

  1. Compute compact SVD of X directly (NO centering).
  2. Apply ``svd_flip(U, Vt, u_based_decision=True)`` — note the
     deliberate divergence from sklearn TruncatedSVD, which uses
     ``u_based_decision=False``. The c4a engine uses U-based flipping
     uniformly across FlexiblePCA and FlexibleSVD so the same SVD
     primitive can serve both.
  3. Compute ``X_proj = U * S`` and per-component variance
     ``np.var(X_proj, axis=0)``.
  4. ``full_var = np.var(X, axis=0).sum()`` and
     ``var_ratio = explained_variance / full_var``.
  5. Select k' the same way as FlexiblePCA (count or ratio).
  6. Components ``= Vt[:k', :]``; transform ``X @ components_.T``.
"""

from __future__ import annotations

import numpy as np

from .flexible_pca import _select_k, _svd_flip_u_based


def flexible_svd_fit_transform(
    fit_X: np.ndarray,
    transform_X: np.ndarray,
    *,
    n_components: float,
) -> tuple[np.ndarray, int]:
    """
    Fit FlexibleSVD on ``fit_X`` and transform ``transform_X``.

    Parameters
    ----------
    fit_X : array-like, shape (m, n)
        Training matrix. ``m >= 2``.
    transform_X : array-like, shape (m_t, n)
        Matrix to project onto the fitted components.
    n_components : float
        Flexible specifier (count or variance ratio).

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
        raise ValueError("FlexibleSVD requires 2-D matrices")
    if transform_X.shape[1] != fit_X.shape[1]:
        raise ValueError("transform_X and fit_X must share n_features")

    m, n = fit_X.shape
    if m < 2 or n < 1:
        raise ValueError("FlexibleSVD: fit_X requires m >= 2 and n >= 1")

    U, S, Vt = np.linalg.svd(fit_X, full_matrices=False)
    U, Vt = _svd_flip_u_based(U, Vt)

    k_full = min(m, n)
    X_proj = U * S
    explained_variance = X_proj.var(axis=0)
    full_var = float(fit_X.var(axis=0).sum())
    var_ratio = (
        explained_variance / full_var
        if full_var > 0.0
        else np.zeros_like(explained_variance)
    )

    k_kept = _select_k(float(n_components), k_full, var_ratio)
    components = Vt[:k_kept, :]

    out = transform_X @ components.T
    return np.ascontiguousarray(out, dtype=np.float64), k_kept

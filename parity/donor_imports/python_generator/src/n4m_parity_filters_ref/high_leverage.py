# SPDX-License-Identifier: CECILL-2.1
"""
High leverage sample filter — frozen NumPy reference.

For a centred (or raw) feature matrix X of shape (n, p) the leverage of
sample i is the i-th diagonal of the hat matrix:

    H = X (X^T X)^{-1} X^T
    h_ii = X_i (X^T X)^{-1} X_i^T

Two computation methods:

  * ``hat``: direct hat diagonal in feature space. Requires ``n > p`` for a
    non-singular ``X^T X``; otherwise falls back to PCA.
  * ``pca``: project into a PCA-reduced score space then compute
    ``h_ii = ||scores_i||^2`` (since the scores are orthogonal, the precision
    matrix simplifies; the formal computation still uses the precision of
    the score covariance for numeric symmetry with the hat path).

Threshold:
  * ``absolute_threshold`` overrides everything (must be in (0, 1)).
  * Otherwise ``threshold = multiplier * mean(leverages_on_training_X)``.

Output: boolean mask, True = keep (low leverage), False = exclude.

Notes on numerical contract
---------------------------
This reference uses a small Tikhonov regularisation
    reg = 1e-10 * trace(X^T X) / p
to keep the ``X^T X`` inverse well-conditioned (matches the
nirs4all.operators.filters.HighLeverageFilter implementation). The n4m
engine reproduces the same regularisation byte-for-byte through its
Householder QR + back-solve path.
"""

from __future__ import annotations

import numpy as np


def _fit_hat(X_centered: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute (X^T X)^{-1} with a Tikhonov-stabilised inverse."""
    XtX = X_centered.T @ X_centered
    p = XtX.shape[0]
    reg = 1e-10 * np.trace(XtX) / max(p, 1)
    XtX_reg = XtX + reg * np.eye(p)
    try:
        precision = np.linalg.inv(XtX_reg)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(XtX)
    return precision, p


def _fit_pca(X_centered: np.ndarray, n_components: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """SVD-based PCA: returns the precision matrix in score space and the
    PCA components (rows = components). The components are the right-singular
    vectors with positive-sign convention matching sklearn's PCA: each row is
    flipped so that the entry with maximum absolute value is non-negative."""
    n_samples, n_features = X_centered.shape
    if n_components is None:
        k = min(n_samples - 1, n_features, 50)
    else:
        k = min(n_components, n_samples - 1, n_features)
    k = max(k, 1)
    # Use thin SVD: X = U S V^T with V (n_features, k_full).
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Match sklearn sign convention.
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, None]
    components = Vt[:k]  # (k, n_features)
    scores = X_centered @ components.T  # (n_samples, k)
    XtX = scores.T @ scores
    reg = 1e-10 * np.trace(XtX) / max(k, 1)
    XtX_reg = XtX + reg * np.eye(k)
    try:
        precision = np.linalg.inv(XtX_reg)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(XtX)
    return precision, components, k


def _compute_leverages(X_centered: np.ndarray, precision: np.ndarray,
                       components: np.ndarray | None) -> np.ndarray:
    if components is not None:
        X_proj = X_centered @ components.T
    else:
        X_proj = X_centered
    left = X_proj @ precision
    return np.asarray(np.sum(left * X_proj, axis=1), dtype=np.float64)


def high_leverage_leverages(X: np.ndarray,
                             *,
                             method: str = "hat",
                             n_components: int | None = None,
                             center: bool = True) -> np.ndarray:
    """Return per-sample leverages h_ii using the same fit/transform protocol
    as the n4m engine: fit on X, then compute leverages on the same X. This is
    a one-shot call (the n4m handle separates fit and transform but the
    frozen reference fuses them because the parity fixture uses the same
    training and evaluation matrix)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("high_leverage_leverages requires a 2-D input")
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples")

    mean_ = np.mean(X, axis=0) if center else np.zeros(n_features, dtype=np.float64)
    X_centered = X - mean_ if center else X

    if method == "pca" or n_features >= n_samples:
        precision, components, _ = _fit_pca(X_centered, n_components)
    elif method == "hat":
        precision, _ = _fit_hat(X_centered)
        components = None
    else:
        raise ValueError(f"method must be 'hat' or 'pca', got '{method}'")

    return _compute_leverages(X_centered, precision, components)


def high_leverage_mask(X: np.ndarray,
                        *,
                        method: str = "hat",
                        threshold_multiplier: float = 2.0,
                        absolute_threshold: float | None = None,
                        n_components: int | None = None,
                        center: bool = True) -> np.ndarray:
    """
    Compute the boolean keep-mask (True = keep, False = exclude high
    leverage).

    Parameters
    ----------
    X : (n, p) array
    method : "hat" or "pca"
    threshold_multiplier : float
        Multiplier on the average training leverage. Default 2.0.
    absolute_threshold : float or None
        If set, overrides the multiplier-based threshold.
    n_components : int or None
        PCA component count (used by the "pca" path).
    center : bool
        Whether to subtract the per-column mean before computing leverages.

    Returns
    -------
    mask : (n,) bool array.
    """
    leverages = high_leverage_leverages(
        X, method=method, n_components=n_components, center=center
    )
    if absolute_threshold is not None:
        if not (0.0 < absolute_threshold < 1.0):
            raise ValueError("absolute_threshold must be in (0, 1)")
        thr = float(absolute_threshold)
    else:
        if threshold_multiplier <= 0.0:
            raise ValueError("threshold_multiplier must be positive")
        thr = float(threshold_multiplier) * float(np.mean(leverages))
    return leverages <= thr

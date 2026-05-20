# SPDX-License-Identifier: CECILL-2.1
"""
External Parameter Orthogonalization (EPO) reference (Roger 2003).

Univariate external parameter d (length n_samples).  Algorithm:

    fit(X, d):
        if scale:
            X_mean = mean(X, axis=0)
            d_mean = mean(d)
        else:
            X_mean = 0
            d_mean = 0
        Xc = X - X_mean
        dc = d - d_mean
        B[j] = (dc · Xc[:, j]) / (dc · dc)         # for each feature j

    transform(X):                       # no d available at transform time
        Xc = X - X_mean
        out = Xc + X_mean = X            # pass-through

    fit_transform(X, d):
        Xc = X - X_mean
        dc = d - d_mean
        Xf = Xc - outer(dc, B)
        out = Xf + X_mean = X - outer(dc, B)

Note: when scale=False X_mean=0 and d_mean=0, so all branches reduce to
``X - outer(d, B)``.
"""

from __future__ import annotations

import numpy as np

_VAR_FLOOR = 1e-20


def epo_fit_transform(X: np.ndarray, d: np.ndarray, *, scale: bool = True):
    """Fit EPO on (X, d) and return (B, X_mean, d_mean, X_filtered)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64).ravel()
    rows, cols = X.shape
    assert d.shape == (rows,)
    if scale:
        X_mean = np.mean(X, axis=0)
        d_mean = float(np.mean(d))
    else:
        X_mean = np.zeros(cols, dtype=np.float64)
        d_mean = 0.0
    dc = d - d_mean
    dd = float(np.dot(dc, dc))
    if dd < _VAR_FLOOR:
        raise ValueError("External parameter d has zero variance")
    Xc = X - X_mean
    B  = (dc @ Xc) / dd
    X_filtered = X - np.outer(dc, B)
    return {
        "B": B,
        "X_mean": X_mean,
        "d_mean": d_mean,
        "X_filtered": X_filtered,
    }


def epo(X_fit: np.ndarray, d_fit: np.ndarray,
         X_transform: np.ndarray, *, scale: bool = True) -> np.ndarray:
    """Fit EPO on (X_fit, d_fit) and apply on X_transform.  Since d is not
    available at transform time, this returns X_transform unchanged (the
    documented 'd = d_mean' contract from nirs4all 0.8.x)."""
    _ = epo_fit_transform(X_fit, d_fit, scale=scale)
    return np.ascontiguousarray(X_transform, dtype=np.float64).copy()


def epo_fit_apply_with_d(X_fit: np.ndarray, d_fit: np.ndarray,
                          X_transform: np.ndarray, d_transform: np.ndarray, *,
                          scale: bool = True) -> np.ndarray:
    """Fit EPO on (X_fit, d_fit) and apply on (X_transform, d_transform)
    when the caller has d available at transform time."""
    fit_state = epo_fit_transform(X_fit, d_fit, scale=scale)
    B = fit_state["B"]
    d_mean = fit_state["d_mean"]
    Xt = np.ascontiguousarray(X_transform, dtype=np.float64)
    dt = np.ascontiguousarray(d_transform, dtype=np.float64).ravel()
    dc = dt - d_mean
    return Xt - np.outer(dc, B)

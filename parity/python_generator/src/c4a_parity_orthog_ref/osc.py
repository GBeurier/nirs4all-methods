# SPDX-License-Identifier: CECILL-2.1
"""
Orthogonal Signal Correction (OSC / DOSC) reference.

Algorithm — Direct OSC variant (Westerhuis 2001), univariate target:

    fit(X, y):
        X_mean, X_std = center+scale stats (ddof=1)
        y_mean, y_std = center+scale stats (ddof=1)
        Xc = (X - X_mean) / X_std
        yc = (y - y_mean) / y_std
        # SVD-fallback for the PLS weight (univariate y → rank-1 SVD
        # reduces to normalised cross-correlation).
        w_pls = Xc.T @ yc / ||Xc.T @ yc||
        X_res = Xc.copy()
        for i in range(n_components_):
            w        = X_res.T @ yc / ||...||
            t        = X_res @ w
            p        = X_res.T @ t / (t.T @ t)
            w_ortho  = (p - (p . w_pls) * w_pls); normalise
            t_ortho  = X_res @ w_ortho
            p_ortho  = X_res.T @ t_ortho / (t_ortho.T @ t_ortho)
            X_res   -= outer(t_ortho, p_ortho)
            W_ortho[:, i] = w_ortho
            P_ortho[:, i] = p_ortho

    transform(X):
        Xc = (X - X_mean) / X_std
        for i in range(n_components_):
            t_ortho = Xc @ W_ortho[:, i]
            Xc     -= outer(t_ortho, P_ortho[:, i])
        out = Xc * X_std + X_mean

This is the canonical parity floor for the c4a engine — every numerical
choice here (in particular the ddof=1 standard deviation and the early-
break behaviour on degenerate norms) matches what the c4a engine does.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10
_STD_FLOOR = 1e-10


def _center_scale_stats(X: np.ndarray, scale: bool):
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    if scale and n >= 2:
        std = np.std(X, axis=0, ddof=1)
        std = np.where(std < _STD_FLOOR, 1.0, std)
    else:
        std = np.ones(X.shape[1], dtype=np.float64)
    if not scale:
        mean = np.zeros_like(mean)
    return mean, std


def osc_fit_transform(X: np.ndarray, y: np.ndarray, *,
                       n_components: int = 1, scale: bool = True):
    """Fit OSC on (X, y) and return (n_components_actual, W_ortho, P_ortho,
    X_mean, X_std, y_mean, y_std, X_transformed)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    rows, cols = X.shape
    assert y.shape == (rows,)
    max_feasible = min(cols - 1, rows - 2)
    n_eff = min(n_components, max_feasible)
    if n_eff < 1:
        raise ValueError("n_components clipped to < 1")
    X_mean, X_std = _center_scale_stats(X, scale)
    y_mean = float(np.mean(y)) if scale else 0.0
    y_std_raw = float(np.std(y, ddof=1)) if scale and rows >= 2 else 1.0
    y_std = y_std_raw if y_std_raw >= _STD_FLOOR else 1.0
    if not scale:
        y_std = 1.0
        y_mean = 0.0
    Xc = (X - X_mean) / X_std
    yc = (y - y_mean) / y_std
    w_pls = Xc.T @ yc
    w_pls_norm = float(np.linalg.norm(w_pls))
    if w_pls_norm < _EPS:
        raise ValueError("Y has no correlation with X; OSC cannot be applied")
    w_pls = w_pls / w_pls_norm
    W_ortho = np.zeros((cols, n_eff), dtype=np.float64)
    P_ortho = np.zeros((cols, n_eff), dtype=np.float64)
    X_res = Xc.copy()
    n_extracted = 0
    for i in range(n_eff):
        w = X_res.T @ yc
        w_norm = float(np.linalg.norm(w))
        if w_norm < _EPS:
            break
        w = w / w_norm
        t = X_res @ w
        t_norm_sq = float(np.dot(t, t))
        if t_norm_sq < _EPS:
            break
        p = X_res.T @ t / t_norm_sq
        pw = float(np.dot(p, w_pls))
        w_ortho = p - pw * w_pls
        wo_norm = float(np.linalg.norm(w_ortho))
        if wo_norm < _EPS:
            break
        w_ortho = w_ortho / wo_norm
        t_ortho = X_res @ w_ortho
        to_norm_sq = float(np.dot(t_ortho, t_ortho))
        if to_norm_sq < _EPS:
            break
        p_ortho = X_res.T @ t_ortho / to_norm_sq
        X_res = X_res - np.outer(t_ortho, p_ortho)
        W_ortho[:, i] = w_ortho
        P_ortho[:, i] = p_ortho
        n_extracted += 1
    # Apply the learned deflation to produce the fit-time transform.
    Xc2 = (X - X_mean) / X_std
    for i in range(n_extracted):
        t_ortho = Xc2 @ W_ortho[:, i]
        Xc2 = Xc2 - np.outer(t_ortho, P_ortho[:, i])
    X_transformed = Xc2 * X_std + X_mean
    return {
        "n_components_actual": n_extracted,
        "W_ortho": W_ortho[:, :n_extracted].copy(),
        "P_ortho": P_ortho[:, :n_extracted].copy(),
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "X_transformed": X_transformed,
    }


def osc(X_fit: np.ndarray, y: np.ndarray, X_transform: np.ndarray, *,
         n_components: int = 1, scale: bool = True) -> np.ndarray:
    """Fit OSC on (X_fit, y) and apply on X_transform. Returns the
    transformed matrix (same shape as X_transform)."""
    fit_state = osc_fit_transform(X_fit, y, n_components=n_components,
                                   scale=scale)
    W_ortho = fit_state["W_ortho"]
    P_ortho = fit_state["P_ortho"]
    X_mean  = fit_state["X_mean"]
    X_std   = fit_state["X_std"]
    Xt = np.ascontiguousarray(X_transform, dtype=np.float64)
    Xc = (Xt - X_mean) / X_std
    for i in range(fit_state["n_components_actual"]):
        t_ortho = Xc @ W_ortho[:, i]
        Xc = Xc - np.outer(t_ortho, P_ortho[:, i])
    return Xc * X_std + X_mean

"""Per-fixture producers. Each function returns a dict in the v1 schema."""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import betaincinv
from sklearn.cross_decomposition import PLSRegression


def _flatten_rowmajor(arr: np.ndarray) -> list[float]:
    return arr.astype(np.float64, copy=False).reshape(-1, order="C").tolist()


def _expected_from_pls(model: PLSRegression, X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    coef = model.coef_.astype(np.float64, copy=False)  # shape (n_features, n_targets) in 1.4
    if coef.shape == (Y.shape[1], X.shape[1]):
        coef = coef.T
    intercept = np.atleast_1d(model.intercept_).astype(np.float64, copy=False)
    preds = model.predict(X).astype(np.float64, copy=False)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": list(intercept.shape),  "values": intercept.tolist()},
        "x_mean":        {"shape": [X.shape[1]],           "values": model._x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [X.shape[1]],           "values": model._x_std.astype(np.float64).tolist()},
        "y_mean":        {"shape": [Y.shape[1]],           "values": np.atleast_1d(model._y_mean).astype(np.float64).tolist()},
        "y_scale":       {"shape": [Y.shape[1]],           "values": np.atleast_1d(model._y_std).astype(np.float64).tolist()},
        "weights_W":     {"shape": list(model.x_weights_.shape),  "values": _flatten_rowmajor(model.x_weights_),  "sign_invariant": True},
        "loadings_P":    {"shape": list(model.x_loadings_.shape), "values": _flatten_rowmajor(model.x_loadings_), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(model.y_loadings_.shape), "values": _flatten_rowmajor(model.y_loadings_), "sign_invariant": True},
        "rotations_R":   {"shape": list(model.x_rotations_.shape),"values": _flatten_rowmajor(model.x_rotations_),"sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),       "values": _flatten_rowmajor(preds)},
    }


def _expected_from_pls_da(model: PLSRegression, X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    coef = model.coef_.astype(np.float64, copy=False)
    if coef.shape == (Y.shape[1], X.shape[1]):
        coef = coef.T
    coef = coef / model._x_std.astype(np.float64).reshape(-1, 1)
    intercept = np.atleast_1d(model.intercept_).astype(np.float64, copy=False)
    preds = model.predict(X).astype(np.float64, copy=False)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": list(intercept.shape),  "values": intercept.tolist()},
        "x_mean":        {"shape": [X.shape[1]],           "values": model._x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [X.shape[1]],           "values": model._x_std.astype(np.float64).tolist()},
        "y_mean":        {"shape": [Y.shape[1]],           "values": np.atleast_1d(model._y_mean).astype(np.float64).tolist()},
        "y_scale":       {"shape": [Y.shape[1]],           "values": np.atleast_1d(model._y_std).astype(np.float64).tolist()},
        "weights_W":     {"shape": list(model.x_weights_.shape),  "values": _flatten_rowmajor(model.x_weights_),  "sign_invariant": True},
        "loadings_P":    {"shape": list(model.x_loadings_.shape), "values": _flatten_rowmajor(model.x_loadings_), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(model.y_loadings_.shape), "values": _flatten_rowmajor(model.y_loadings_), "sign_invariant": True},
        "rotations_R":   {"shape": list(model.x_rotations_.shape),"values": _flatten_rowmajor(model.x_rotations_),"sign_invariant": True},
        "scores_T":      {"shape": list(model._x_scores.shape),   "values": _flatten_rowmajor(model._x_scores),   "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),       "values": _flatten_rowmajor(preds)},
    }


def _center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    centered = X - mean
    scale = centered.std(axis=0, ddof=1)
    scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
    return centered / scale, mean, scale


def _dominant_direction(S: np.ndarray) -> np.ndarray:
    if S.shape[1] == 1:
        r = S[:, 0].copy()
    else:
        U, _singular_values, _Vt = np.linalg.svd(S, full_matrices=False)
        r = U[:, 0].copy()
    sign_idx = int(np.argmax(np.abs(r)))
    if r[sign_idx] < 0.0:
        r = -r
    return r


def _dominant_svd_pair(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if C.shape[1] == 1:
        left = C[:, 0].astype(np.float64, copy=True)
        norm = np.linalg.norm(left)
        if norm <= np.finfo(np.float64).eps:
            raise RuntimeError("SVD covariance collapsed")
        left = left / norm
        right = np.array([1.0], dtype=np.float64)
    else:
        U, _singular_values, Vt = np.linalg.svd(C, full_matrices=False)
        left = U[:, 0].astype(np.float64, copy=True)
        right = Vt[0, :].astype(np.float64, copy=True)
    sign_idx = int(np.argmax(np.abs(left)))
    if left[sign_idx] < 0.0:
        left = -left
        right = -right
    return left, right


def _dominant_svd_pair_power(C: np.ndarray, max_iter: int = 500, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=np.float64)
    p, q = C.shape
    eps = np.finfo(np.float64).eps
    if q == 1:
        left = C[:, 0].astype(np.float64, copy=True)
        norm = np.linalg.norm(left)
        if norm <= eps:
            raise RuntimeError("power covariance collapsed")
        left = left / norm
        right = np.array([1.0], dtype=np.float64)
    else:
        col_norms = np.sum(C * C, axis=0)
        best = int(np.argmax(col_norms))
        if col_norms[best] <= eps:
            raise RuntimeError("power covariance collapsed")
        right = np.zeros(q, dtype=np.float64)
        right[best] = 1.0
        left = np.zeros(p, dtype=np.float64)
        converged = False
        for _iteration in range(max_iter):
            left = C @ right
            left_norm = np.linalg.norm(left)
            if left_norm <= eps:
                raise RuntimeError("power left singular vector collapsed")
            left = left / left_norm
            next_right = C.T @ left
            right_norm = np.linalg.norm(next_right)
            if right_norm <= eps:
                raise RuntimeError("power right singular vector collapsed")
            next_right = next_right / right_norm
            diff_same = float((next_right - right) @ (next_right - right))
            diff_opposite = float((next_right + right) @ (next_right + right))
            right = next_right
            if min(diff_same, diff_opposite) < tol:
                converged = True
                break
        if not converged:
            raise RuntimeError("power singular vectors failed to converge")
        left = C @ right
        left_norm = np.linalg.norm(left)
        if left_norm <= eps:
            raise RuntimeError("power final left singular vector collapsed")
        left = left / left_norm
        right = C.T @ left
        right_norm = np.linalg.norm(right)
        if right_norm <= eps:
            raise RuntimeError("power final right singular vector collapsed")
        right = right / right_norm

    sign_idx = int(np.argmax(np.abs(left)))
    if left[sign_idx] < 0.0:
        left = -left
        right = -right
    return left, right


_MASK64 = (1 << 64) - 1
_SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15


def _splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + _SPLITMIX_GOLDEN) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return state, z


def _uniform_signed_from_state(state: int) -> tuple[int, float]:
    state, bits = _splitmix64_next(state)
    unit = (bits >> 11) * (1.0 / float(1 << 53))
    return state, 2.0 * unit - 1.0


def _dominant_svd_pair_randomized(
    C: np.ndarray,
    seed: int,
    component: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=np.float64)
    p, q = C.shape
    eps = np.finfo(np.float64).eps
    if q == 1:
        left = C[:, 0].astype(np.float64, copy=True)
        norm = np.linalg.norm(left)
        if norm <= eps:
            raise RuntimeError("randomized SVD covariance collapsed")
        left = left / norm
        right = np.array([1.0], dtype=np.float64)
    else:
        state = (int(seed) + _SPLITMIX_GOLDEN * (int(component) + 1)) & _MASK64
        right = np.zeros(q, dtype=np.float64)
        for target in range(q):
            state, right[target] = _uniform_signed_from_state(state)
        right_norm = np.linalg.norm(right)
        if right_norm <= eps:
            right[:] = 0.0
            right[0] = 1.0
        else:
            right = right / right_norm

        left = np.zeros(p, dtype=np.float64)
        converged = False
        for _iteration in range(max_iter):
            left = C @ right
            left_norm = np.linalg.norm(left)
            if left_norm <= eps:
                raise RuntimeError("randomized SVD left singular vector collapsed")
            left = left / left_norm
            next_right = C.T @ left
            right_norm = np.linalg.norm(next_right)
            if right_norm <= eps:
                raise RuntimeError("randomized SVD right singular vector collapsed")
            next_right = next_right / right_norm
            diff_same = float((next_right - right) @ (next_right - right))
            diff_opposite = float((next_right + right) @ (next_right + right))
            right = next_right
            if min(diff_same, diff_opposite) < tol:
                converged = True
                break
        if not converged:
            raise RuntimeError("randomized SVD singular vectors failed to converge")
        left = C @ right
        left_norm = np.linalg.norm(left)
        if left_norm <= eps:
            raise RuntimeError("randomized SVD final left singular vector collapsed")
        left = left / left_norm
        right = C.T @ left
        right_norm = np.linalg.norm(right)
        if right_norm <= eps:
            raise RuntimeError("randomized SVD final right singular vector collapsed")
        right = right / right_norm

    sign_idx = int(np.argmax(np.abs(left)))
    if left[sign_idx] < 0.0:
        left = -left
        right = -right
    return left, right


def _simpls_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xs, x_mean, x_scale = _center_scale(X)
    Ys, y_mean, y_scale = _center_scale(Y)
    n, p = Xs.shape
    q = Ys.shape[1]
    K = int(n_components)

    Z = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    V = np.zeros((p, K), dtype=np.float64)
    S = Xs.T @ Ys
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        r = _dominant_direction(S)
        t = Xs @ r
        t_norm = np.linalg.norm(t)
        if t_norm <= eps:
            raise RuntimeError(f"SIMPLS score collapsed at component {comp}")
        t = t / t_norm
        r = r / t_norm
        p_load = Xs.T @ t
        q_load = Ys.T @ t
        v = p_load.copy()
        if comp > 0:
            v = v - V[:, :comp] @ (V[:, :comp].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm <= eps:
            raise RuntimeError(f"SIMPLS deflation basis collapsed at component {comp}")
        v = v / v_norm

        Z[:, comp] = r
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        V[:, comp] = v
        S = S - np.outer(v, v.T @ S)

    rotations = Z @ np.linalg.inv(P.T @ Z)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(Z.shape),          "values": _flatten_rowmajor(Z), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _svd_pls_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        x_weights, y_weights = _dominant_svd_pair(Xk.T @ Yk)
        t = Xk @ x_weights
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"SVD score collapsed at component {comp}")
        y_weight_ss = float(y_weights @ y_weights)
        if y_weight_ss <= eps:
            raise RuntimeError(f"SVD Y weights collapsed at component {comp}")
        u = (Yk @ y_weights) / (y_weight_ss + eps)
        p_load = (Xk.T @ t) / t_ss
        q_load = (Yk.T @ t) / t_ss
        Xk = Xk - np.outer(t, p_load)
        Yk = Yk - np.outer(t, q_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        U_scores[:, comp] = u

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _pls_svd_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xs, x_mean, x_scale = _center_scale(X)
    Ys, y_mean, y_scale = _center_scale(Y)
    n, p = Xs.shape
    q = Ys.shape[1]
    K = int(n_components)
    C = Xs.T @ Ys
    U, singular_values, Vt = np.linalg.svd(C, full_matrices=False)
    U = U[:, :K].astype(np.float64, copy=True)
    Vt = Vt[:K, :].astype(np.float64, copy=True)
    for comp in range(K):
        sign_idx = int(np.argmax(np.abs(U[:, comp])))
        if U[sign_idx, comp] < 0.0:
            U[:, comp] = -U[:, comp]
            Vt[comp, :] = -Vt[comp, :]
    V = Vt.T

    T = Xs @ U
    U_scores = Ys @ V
    P = np.zeros((p, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps
    for comp in range(K):
        t = T[:, comp]
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"PLSSVD X score collapsed at component {comp}")
        P[:, comp] = (Xs.T @ t) / t_ss

    coef_std = U @ V.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(U.shape),          "values": _flatten_rowmajor(U), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(V.shape),          "values": _flatten_rowmajor(V), "sign_invariant": True},
        "rotations_R":   {"shape": list(U.shape),          "values": _flatten_rowmajor(U), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _osc_dominant_weight(values: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    col_sumsq = np.sum(values * values, axis=0)
    best_col = int(np.argmax(col_sumsq))
    if float(col_sumsq[best_col]) <= np.finfo(np.float64).eps:
        raise RuntimeError("OSC found no X variation orthogonal to Y")
    weights = np.zeros(values.shape[1], dtype=np.float64)
    weights[best_col] = 1.0
    for _ in range(max_iter):
        scores = values @ weights
        nxt = values.T @ scores
        norm = float(np.linalg.norm(nxt))
        if norm <= np.finfo(np.float64).eps:
            raise RuntimeError("OSC dominant direction vanished")
        nxt = nxt / norm
        if float(nxt @ weights) < 0.0:
            nxt = -nxt
        diff = float((nxt - weights) @ (nxt - weights))
        weights = nxt
        if diff <= tol:
            break
    return weights


def _epo_dominant_direction(covariance: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    row_sumsq = np.sum(covariance * covariance, axis=1)
    best_row = int(np.argmax(row_sumsq))
    if float(row_sumsq[best_row]) <= np.finfo(np.float64).eps:
        raise RuntimeError("EPO external covariance vanished")
    weights = np.zeros(covariance.shape[0], dtype=np.float64)
    weights[best_row] = 1.0
    for _ in range(max_iter):
        right = covariance.T @ weights
        nxt = covariance @ right
        norm = float(np.linalg.norm(nxt))
        if norm <= np.finfo(np.float64).eps:
            raise RuntimeError("EPO dominant direction vanished")
        nxt = nxt / norm
        if float(nxt @ weights) < 0.0:
            nxt = -nxt
        diff = float((nxt - weights) @ (nxt - weights))
        weights = nxt
        if diff <= tol:
            break
    return weights


def _pipeline_apply(X: np.ndarray, operators: list[str], Y: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64).copy()
    y = None if Y is None else np.asarray(Y, dtype=np.float64)
    for op in operators:
        if op == "identity":
            out = out.copy()
        elif op == "center":
            out = out - out.mean(axis=0, keepdims=True)
        elif op == "autoscale":
            mean = out.mean(axis=0, keepdims=True)
            scale = out.std(axis=0, ddof=1, keepdims=True)
            scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
            out = (out - mean) / scale
        elif op == "pareto":
            mean = out.mean(axis=0, keepdims=True)
            scale = np.sqrt(out.std(axis=0, ddof=1, keepdims=True))
            scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
            out = (out - mean) / scale
        elif op == "snv":
            mean = out.mean(axis=1, keepdims=True)
            scale = out.std(axis=1, ddof=1, keepdims=True)
            scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
            out = (out - mean) / scale
        elif op == "msc":
            reference = out.mean(axis=0)
            ref_mean = float(reference.mean())
            ref_ss = float(np.sum((reference - ref_mean) ** 2))
            corrected = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                row_mean = float(row.mean())
                slope = float(np.sum((reference - ref_mean) * (row - row_mean)) / ref_ss)
                intercept = row_mean - slope * ref_mean
                corrected[row_idx, :] = (row - intercept) / slope
            out = corrected
        elif op.startswith("emsc_"):
            degree = int(op.rsplit("_", 1)[1])
            reference = out.mean(axis=0)
            x = np.linspace(-1.0, 1.0, out.shape[1], dtype=np.float64)
            polynomial = np.vander(x, N=degree + 1, increasing=True)
            design = np.column_stack([reference, polynomial])
            corrected = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                coeffs, *_ = np.linalg.lstsq(design, row, rcond=None)
                baseline = design[:, 1:] @ coeffs[1:]
                corrected[row_idx, :] = (row - baseline) / coeffs[0]
            out = corrected
        elif op.startswith("detrend_poly_"):
            degree = int(op.rsplit("_", 1)[1])
            x = np.linspace(-1.0, 1.0, out.shape[1], dtype=np.float64)
            design = np.vander(x, N=degree + 1, increasing=True)
            trend = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                coeffs, *_ = np.linalg.lstsq(design, row, rcond=None)
                trend[row_idx, :] = design @ coeffs
            out = out - trend
        elif op.startswith("savgol_smooth_"):
            _, _, window, poly_degree = op.split("_")
            out = savgol_filter(out,
                                window_length=int(window),
                                polyorder=int(poly_degree),
                                deriv=0,
                                delta=1.0,
                                axis=1,
                                mode="nearest")
        elif op.startswith("savgol_derivative_"):
            _, _, window, poly_degree, derivative_order, delta = op.split("_")
            out = savgol_filter(out,
                                window_length=int(window),
                                polyorder=int(poly_degree),
                                deriv=int(derivative_order),
                                delta=float(delta),
                                axis=1,
                                mode="nearest")
        elif op.startswith("asls_"):
            _, lam, asymmetry, iterations = op.split("_")
            lam_f = float(lam)
            asymmetry_f = float(asymmetry)
            iterations_i = int(iterations)
            n_cols = out.shape[1]
            penalty = np.zeros((n_cols, n_cols), dtype=np.float64)
            for diff in range(n_cols - 2):
                stencil = np.array([1.0, -2.0, 1.0], dtype=np.float64)
                penalty[diff:diff + 3, diff:diff + 3] += np.outer(stencil, stencil)
            penalty *= lam_f
            corrected = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                weights = np.ones(n_cols, dtype=np.float64)
                baseline = np.zeros(n_cols, dtype=np.float64)
                for _ in range(iterations_i):
                    matrix = penalty + np.diag(weights)
                    rhs = weights * row
                    baseline = np.linalg.solve(matrix, rhs)
                    weights = np.where(row > baseline, asymmetry_f, 1.0 - asymmetry_f)
                corrected[row_idx, :] = row - baseline
            out = corrected
        elif op.startswith("norris_williams_"):
            _, _, segment, gap, derivative_order = op.split("_")
            segment_i = int(segment)
            gap_i = int(gap)
            order_i = int(derivative_order)
            coeffs: list[float] = []
            for block in range(order_i + 1):
                sign = 1.0 if ((order_i - block) % 2) == 0 else -1.0
                value = sign * float(math.comb(order_i, block)) / float(segment_i)
                coeffs.extend([value] * segment_i)
                if block < order_i:
                    coeffs.extend([0.0] * gap_i)
            coeff_arr = np.asarray(coeffs, dtype=np.float64)
            half = len(coeffs) // 2
            transformed = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                for col in range(out.shape[1]):
                    acc = 0.0
                    for k, coeff in enumerate(coeff_arr):
                        source = min(max(col + k - half, 0), out.shape[1] - 1)
                        acc += float(coeff) * float(row[source])
                    transformed[row_idx, col] = acc
            out = transformed
        elif op.startswith("wavelet_haar_"):
            _, _, levels, threshold = op.split("_")
            levels_i = int(levels)
            threshold_f = float(threshold)
            n_cols = out.shape[1]
            padded = 1
            while padded < n_cols:
                padded *= 2
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            transformed = np.zeros_like(out)
            for row_idx, row in enumerate(out):
                work = np.empty(padded, dtype=np.float64)
                work[:n_cols] = row
                work[n_cols:] = row[-1]
                active = padded
                for _ in range(levels_i):
                    half = active // 2
                    temp = work.copy()
                    for i in range(half):
                        left = work[2 * i]
                        right = work[2 * i + 1]
                        temp[i] = (left + right) * inv_sqrt2
                        detail = (left - right) * inv_sqrt2
                        temp[half + i] = np.sign(detail) * max(abs(detail) - threshold_f, 0.0)
                    work[:active] = temp[:active]
                    active = half
                active = padded >> levels_i
                for _ in range(levels_i):
                    half = active
                    size = half * 2
                    temp = work.copy()
                    for i in range(half):
                        average = work[i]
                        detail = work[half + i]
                        temp[2 * i] = (average + detail) * inv_sqrt2
                        temp[2 * i + 1] = (average - detail) * inv_sqrt2
                    work[:size] = temp[:size]
                    active = size
                transformed[row_idx, :] = work[:n_cols]
            out = transformed
        elif op == "osc" or op.startswith("osc_"):
            if y is None:
                raise ValueError("OSC pipeline fixture requires Y")
            if op == "osc":
                max_iter_i = 100
                tol_f = 1e-10
            else:
                _prefix, max_iter, tol = op.split("_")
                max_iter_i = int(max_iter)
                tol_f = float(tol)
            x_mean = out.mean(axis=0)
            x_centered = out - x_mean.reshape(1, -1)
            y_centered = y - y.mean(axis=0).reshape(1, -1)
            gram = y_centered.T @ y_centered
            projection = y_centered @ np.linalg.solve(gram, y_centered.T @ x_centered)
            x_orthogonal = x_centered - projection
            weights = _osc_dominant_weight(x_orthogonal, max_iter_i, tol_f)
            scores = x_centered @ weights
            score_ss = float(scores @ scores)
            if score_ss <= np.finfo(np.float64).eps:
                raise RuntimeError("OSC training score vanished")
            loadings = (x_centered.T @ scores) / score_ss
            out = out - np.outer(scores, loadings)
        elif op == "epo" or op.startswith("epo_"):
            if y is None:
                raise ValueError("EPO pipeline fixture requires Y")
            if op == "epo":
                max_iter_i = 100
                tol_f = 1e-10
            else:
                _prefix, max_iter, tol = op.split("_")
                max_iter_i = int(max_iter)
                tol_f = float(tol)
            x_mean = out.mean(axis=0)
            x_centered = out - x_mean.reshape(1, -1)
            y_centered = y - y.mean(axis=0).reshape(1, -1)
            covariance = x_centered.T @ y_centered
            direction = _epo_dominant_direction(covariance, max_iter_i, tol_f)
            scores = x_centered @ direction
            out = out - np.outer(scores, direction)
        else:
            raise ValueError(f"unsupported pipeline operator fixture: {op}")
    return out.astype(np.float64, copy=False)


def _pipeline_expected(X: np.ndarray, operators: list[str], Y: np.ndarray | None = None) -> dict[str, Any]:
    transformed = _pipeline_apply(X, operators, Y)
    return {
        "transform_train": {
            "shape": list(transformed.shape),
            "values": _flatten_rowmajor(transformed),
        },
    }


_AOM_OPERATOR_KIND_IDS = {
    "identity": 0,
    "center": 1,
    "autoscale": 2,
    "pareto": 3,
    "snv": 4,
    "msc": 5,
}


def _aom_preprocessing_expected(
    X: np.ndarray,
    operators: Sequence[str],
    gating_mode: str,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    operator_outputs = [
        _pipeline_apply(X, [str(operator)])
        for operator in operators
    ]
    n_ops = len(operator_outputs)
    if gating_mode == "soft":
        weights = np.full(n_ops, 1.0 / float(n_ops), dtype=np.float64)
        mode_id = 1
    elif gating_mode == "hard":
        weights = np.zeros(n_ops, dtype=np.float64)
        weights[0] = 1.0
        mode_id = 0
    else:
        raise ValueError(f"unsupported AOM gating mode fixture: {gating_mode}")
    transformed = np.zeros_like(X, dtype=np.float64)
    for weight, output in zip(weights, operator_outputs):
        transformed += float(weight) * output
    stacked = np.vstack([output.reshape(1, -1, order="C") for output in operator_outputs])
    return {
        "gating_mode": int(mode_id),
        "operator_kinds": {
            "shape": [1, int(n_ops)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(_AOM_OPERATOR_KIND_IDS[str(operator)]) for operator in operators],
        },
        "weights": {
            "shape": [1, int(n_ops)],
            "layout": "row_major",
            "dtype": "f64",
            "values": weights.astype(np.float64).tolist(),
        },
        "operator_outputs": {
            "shape": [int(n_ops), int(X.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": stacked.reshape(-1, order="C").astype(np.float64).tolist(),
        },
        "transformed": {
            "shape": list(X.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(transformed),
        },
    }


def _percentile_linear(values: np.ndarray, probability: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    position = probability * float(values.size - 1)
    lo = int(math.floor(position))
    hi = min(lo + 1, int(values.size - 1))
    frac = position - float(lo)
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _regression_metrics_expected(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(Y_true, dtype=np.float64)
    y_pred = np.asarray(Y_pred, dtype=np.float64)
    error = y_pred - y_true
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    flat_error = error.reshape(-1)
    n = flat_true.size
    eps = np.finfo(np.float64).eps
    mean_true = float(flat_true.mean())
    mean_pred = float(flat_pred.mean())
    sse = float(np.sum(flat_error * flat_error))
    tss = float(np.sum((flat_true - mean_true) ** 2))
    pred_ss = float(np.sum((flat_pred - mean_pred) ** 2))
    pred_true_cov = float(np.sum((flat_pred - mean_pred) * (flat_true - mean_true)))
    rmse = float(np.sqrt(sse / float(n)))
    r2 = 1.0 if (tss <= eps and sse <= eps) else (0.0 if tss <= eps else 1.0 - sse / tss)
    slope = 0.0 if pred_ss <= eps else pred_true_cov / pred_ss
    intercept = mean_true - slope * mean_pred
    stddev = float(np.sqrt(tss / float(n - 1))) if n > 1 else 0.0
    q1 = _percentile_linear(flat_true, 0.25)
    q3 = _percentile_linear(flat_true, 0.75)
    ratio_denominator = max(rmse, eps)
    values = [
        rmse,
        float(np.mean(np.abs(flat_error))),
        float(np.mean(flat_error)),
        r2,
        r2,
        slope,
        intercept,
        stddev / ratio_denominator,
        (q3 - q1) / ratio_denominator,
    ]
    return {
        "metrics": {
            "shape": [1, len(values)],
            "names": ["rmse", "mae", "bias", "r2", "q2", "slope", "intercept", "rpd", "rpiq"],
            "values": values,
        },
    }


def _auc_average_ranks(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    rank_sum = 0.0
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (float(start + 1) + float(stop)) * 0.5
        rank_sum += average_rank * float(np.sum(sorted_labels[start:stop] == 1))
        start = stop
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    return (rank_sum - positives * (positives + 1.0) * 0.5) / (positives * negatives)


def _binary_classification_metrics_expected(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_pred = y_score >= float(threshold)
    actual = y_true == 1
    tp = int(np.sum(actual & y_pred))
    tn = int(np.sum((~actual) & (~y_pred)))
    fp = int(np.sum((~actual) & y_pred))
    fn = int(np.sum(actual & (~y_pred)))
    positives = int(np.sum(actual))
    negatives = int(y_true.size - positives)
    sensitivity = float(tp / positives)
    specificity = float(tn / negatives)
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    f1 = 0.0 if (precision + sensitivity) == 0.0 else float(2.0 * precision * sensitivity / (precision + sensitivity))
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = 0.0 if denom == 0.0 else float(((tp * tn) - (fp * fn)) / denom)
    values = [
        float(y_true.size),
        float(positives),
        float(negatives),
        float(tp),
        float(tn),
        float(fp),
        float(fn),
        float(threshold),
        sensitivity,
        specificity,
        0.5 * (sensitivity + specificity),
        float((tp + tn) / y_true.size),
        precision,
        f1,
        mcc,
        _auc_average_ranks(y_true, y_score),
    ]
    return {
        "metrics": {
            "shape": [1, len(values)],
            "names": [
                "count", "positives", "negatives", "tp", "tn", "fp", "fn",
                "threshold", "sensitivity", "specificity", "balanced_accuracy",
                "accuracy", "precision", "f1", "mcc", "auc",
            ],
            "values": values,
        },
    }


def _multiclass_classification_metrics_expected(
    labels: np.ndarray,
    scores: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = np.argmax(y_score, axis=1).astype(np.int32)
    confusion = np.zeros((int(n_classes), int(n_classes)), dtype=np.float64)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1.0

    per_class: list[float] = []
    sensitivity_values: list[float] = []
    specificity_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []
    auc_values: list[float] = []
    sum_tp = 0.0
    sum_fp = 0.0
    sum_fn = 0.0
    for cls in range(int(n_classes)):
        tp = float(confusion[cls, cls])
        fn = float(np.sum(confusion[cls, :]) - tp)
        fp = float(np.sum(confusion[:, cls]) - tp)
        tn = float(y_true.size - tp - fn - fp)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sensitivity = 0.0 if (tp + fn) == 0.0 else tp / (tp + fn)
        specificity = 0.0 if (tn + fp) == 0.0 else tn / (tn + fp)
        precision = 0.0 if (tp + fp) == 0.0 else tp / (tp + fp)
        f1 = 0.0 if (precision + sensitivity) == 0.0 else 2.0 * precision * sensitivity / (precision + sensitivity)
        binary = (y_true == cls).astype(np.int32)
        auc_ovr = _auc_average_ranks(binary, y_score[:, cls])
        sensitivity_values.append(float(sensitivity))
        specificity_values.append(float(specificity))
        precision_values.append(float(precision))
        f1_values.append(float(f1))
        auc_values.append(float(auc_ovr))
        per_class.extend([float(sensitivity), float(specificity), float(precision), float(f1), float(auc_ovr)])

    accuracy = float(np.mean(y_true == y_pred))
    micro_precision = 0.0 if (sum_tp + sum_fp) == 0.0 else float(sum_tp / (sum_tp + sum_fp))
    micro_recall = 0.0 if (sum_tp + sum_fn) == 0.0 else float(sum_tp / (sum_tp + sum_fn))
    micro_f1 = 0.0 if (micro_precision + micro_recall) == 0.0 else float(2.0 * micro_precision * micro_recall / (micro_precision + micro_recall))
    summary = [
        float(y_true.size),
        float(n_classes),
        accuracy,
        float(np.mean(sensitivity_values)),
        float(np.mean(specificity_values)),
        float(np.mean(precision_values)),
        float(np.mean(f1_values)),
        float(np.mean(auc_values)),
        micro_precision,
        micro_recall,
        micro_f1,
    ]
    return {
        "summary_metrics": {
            "shape": [1, len(summary)],
            "names": [
                "count", "n_classes", "accuracy", "macro_sensitivity",
                "macro_specificity", "macro_precision", "macro_f1",
                "macro_auc_ovr", "micro_precision", "micro_recall",
                "micro_f1",
            ],
            "values": summary,
        },
        "per_class_metrics": {
            "shape": [int(n_classes), 5],
            "names": ["sensitivity", "specificity", "precision", "f1", "auc_ovr"],
            "values": per_class,
        },
        "confusion_matrix": {
            "shape": [int(n_classes), int(n_classes)],
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(confusion),
        },
    }


def _binary_calibration_curve_expected(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_bins = int(n_bins)
    values: list[float] = []
    for bin_idx in range(n_bins):
        lower = float(bin_idx / n_bins)
        upper = float((bin_idx + 1) / n_bins)
        bins = np.floor(y_score * n_bins).astype(np.int64)
        bins = np.clip(bins, 0, n_bins - 1)
        mask = bins == bin_idx
        count = int(np.sum(mask))
        mean_score = 0.0 if count == 0 else float(np.mean(y_score[mask]))
        positive_rate = 0.0 if count == 0 else float(np.mean(y_true[mask]))
        values.extend([lower, upper, float(count), mean_score, positive_rate])
    return {
        "calibration_curve": {
            "shape": [n_bins, 5],
            "names": ["bin_lower", "bin_upper", "count", "mean_score", "positive_rate"],
            "layout": "row_major",
            "dtype": "f64",
            "values": values,
        },
    }


def _pls_lda_expected(
    X: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    dummy = np.zeros((y.size, int(n_classes)), dtype=np.float64)
    dummy[np.arange(y.size), y] = 1.0
    pls = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
    pls.fit(X, dummy)
    scores = pls._x_scores.astype(np.float64, copy=False)

    means = np.zeros((int(n_classes), int(n_components)), dtype=np.float64)
    counts = np.zeros(int(n_classes), dtype=np.int64)
    for cls in range(int(n_classes)):
        mask = y == cls
        counts[cls] = int(np.sum(mask))
        means[cls, :] = np.mean(scores[mask, :], axis=0)

    covariance = np.zeros((int(n_components), int(n_components)), dtype=np.float64)
    for row, cls in enumerate(y):
        delta = scores[row, :] - means[int(cls), :]
        covariance += np.outer(delta, delta)
    covariance = covariance / float(y.size - int(n_classes))
    covariance_inv = np.linalg.inv(covariance)
    inv_means = means @ covariance_inv.T
    constants = -0.5 * np.sum(means * inv_means, axis=1) + np.log(counts.astype(np.float64) / float(y.size))
    decision_scores = scores @ inv_means.T + constants.reshape(1, -1)
    predictions = np.argmax(decision_scores, axis=1).astype(np.int32)
    return {
        "predictions": {
            "shape": [int(y.size)],
            "layout": "row_major",
            "dtype": "i32",
            "values": predictions.astype(int).tolist(),
        },
        "decision_scores": {
            "shape": list(decision_scores.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(decision_scores),
        },
    }


def _softmax_baseline_logits(design: np.ndarray,
                             beta: np.ndarray,
                             n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    logits_tail = design @ beta.T
    logits = np.column_stack([
        np.zeros(design.shape[0], dtype=np.float64),
        logits_tail,
    ])
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    if probabilities.shape[1] != int(n_classes):
        raise RuntimeError("probability shape mismatch")
    return logits, probabilities


def _baseline_logistic_objective(design: np.ndarray,
                                 labels: np.ndarray,
                                 beta: np.ndarray,
                                 ridge: float,
                                 n_classes: int) -> float:
    logits, _probabilities = _softmax_baseline_logits(design, beta, n_classes)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_den = np.max(logits, axis=1) + np.log(np.sum(np.exp(shifted), axis=1))
    chosen = np.zeros(labels.size, dtype=np.float64)
    non_baseline = labels > 0
    chosen[non_baseline] = logits[np.arange(labels.size)[non_baseline], labels[non_baseline]]
    penalty = 0.5 * float(ridge) * float(np.sum(beta[:, 1:] * beta[:, 1:]))
    return float(np.sum(log_den - chosen) + penalty)


def _fit_baseline_multinomial_logistic(
    scores: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    ridge: float = 1e-4,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    design = np.column_stack([np.ones(scores.shape[0], dtype=np.float64), scores])
    n, d = design.shape
    c = int(n_classes)
    m = c - 1
    beta = np.zeros((m, d), dtype=np.float64)

    for _iteration in range(int(max_iter)):
        _logits, probabilities = _softmax_baseline_logits(design, beta, c)
        gradient = np.zeros((m, d), dtype=np.float64)
        hessian = np.zeros((m * d, m * d), dtype=np.float64)
        for row in range(n):
            z = design[row, :]
            zz = np.outer(z, z)
            for cls_j in range(m):
                pj = probabilities[row, cls_j + 1]
                yj = 1.0 if int(y[row]) == cls_j + 1 else 0.0
                gradient[cls_j, :] += (pj - yj) * z
                for cls_l in range(m):
                    pl = probabilities[row, cls_l + 1]
                    weight = pj * ((1.0 if cls_j == cls_l else 0.0) - pl)
                    start_j = cls_j * d
                    start_l = cls_l * d
                    hessian[start_j:start_j + d, start_l:start_l + d] += weight * zz

        for cls_j in range(m):
            for feature in range(1, d):
                offset = cls_j * d + feature
                gradient[cls_j, feature] += float(ridge) * beta[cls_j, feature]
                hessian[offset, offset] += float(ridge)

        step = np.linalg.solve(hessian, gradient.reshape(-1, order="C")).reshape(m, d)
        max_step = float(np.max(np.abs(step)))
        current = _baseline_logistic_objective(design, y, beta, ridge, c)
        alpha = 1.0
        accepted = False
        candidate = beta.copy()
        for _line_search in range(32):
            candidate = beta - alpha * step
            trial = _baseline_logistic_objective(design, y, candidate, ridge, c)
            if math.isfinite(trial) and (trial <= current or alpha * max_step <= tol):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise RuntimeError("baseline multinomial logistic line search failed")
        beta = candidate
        if alpha * max_step <= tol:
            break

    logits, probabilities = _softmax_baseline_logits(design, beta, c)
    predictions = np.argmax(logits, axis=1).astype(np.int32)
    return beta[:, 0], beta[:, 1:], logits, probabilities, predictions


def _pls_logistic_expected(
    X: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    dummy = np.zeros((y.size, int(n_classes)), dtype=np.float64)
    dummy[np.arange(y.size), y] = 1.0
    pls = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
    pls.fit(X, dummy)
    scores = pls._x_scores.astype(np.float64, copy=False)
    intercepts, coefficients, logits, probabilities, predictions = _fit_baseline_multinomial_logistic(
        scores,
        y,
        int(n_classes),
    )
    return {
        "predictions": {
            "shape": [int(y.size)],
            "layout": "row_major",
            "dtype": "i32",
            "values": predictions.astype(int).tolist(),
        },
        "decision_scores": {
            "shape": list(logits.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(logits),
        },
        "probabilities": {
            "shape": list(probabilities.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(probabilities),
        },
        "intercepts": {
            "shape": [int(n_classes) - 1],
            "layout": "row_major",
            "dtype": "f64",
            "values": intercepts.astype(np.float64).tolist(),
        },
        "coefficients": {
            "shape": [int(n_classes) - 1, int(n_components)],
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(coefficients),
        },
    }


def _mb_pls_transform(
    X: np.ndarray,
    block_sizes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    sizes = [int(size) for size in block_sizes]
    if not sizes or any(size <= 0 for size in sizes) or sum(sizes) != X.shape[1]:
        raise ValueError("block_sizes must be positive and sum to X columns")
    transformed = np.zeros_like(X, dtype=np.float64)
    means = np.zeros(X.shape[1], dtype=np.float64)
    scales = np.ones(X.shape[1], dtype=np.float64)
    feature_weights = np.ones(X.shape[1], dtype=np.float64)
    block_weights = np.zeros(len(sizes), dtype=np.float64)
    start = 0
    for block_idx, size in enumerate(sizes):
        stop = start + size
        block = X[:, start:stop]
        centered = block - np.mean(block, axis=0)
        scale = np.std(centered, axis=0, ddof=1)
        scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
        weight = 1.0 / math.sqrt(float(size))
        transformed[:, start:stop] = centered / scale * weight
        means[start:stop] = np.mean(block, axis=0)
        scales[start:stop] = scale
        feature_weights[start:stop] = weight
        block_weights[block_idx] = weight
        start = stop
    return transformed, means, scales, feature_weights, block_weights


def _mb_pls_expected(
    X: np.ndarray,
    Y: np.ndarray,
    block_sizes: Sequence[int],
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    transformed, means, scales, feature_weights, block_weights = _mb_pls_transform(X, block_sizes)
    model = PLSRegression(n_components=int(n_components), scale=False, max_iter=500, tol=1e-6)
    model.fit(transformed, Y)
    coef = model.coef_.astype(np.float64, copy=False)
    if coef.shape == (Y.shape[1], X.shape[1]):
        coef = coef.T
    coef_original = coef * feature_weights.reshape(-1, 1) / scales.reshape(-1, 1)
    model_x_mean = model._x_mean.astype(np.float64)
    y_mean = np.atleast_1d(model._y_mean).astype(np.float64)
    intercept = y_mean - model_x_mean @ coef - means @ coef_original
    preds = model.predict(transformed).astype(np.float64, copy=False)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return {
        "predictions": {
            "shape": list(preds.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(preds),
        },
        "coefficients": {
            "shape": list(coef_original.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(coef_original),
        },
        "intercept": {
            "shape": [int(Y.shape[1])],
            "layout": "row_major",
            "dtype": "f64",
            "values": intercept.astype(np.float64).tolist(),
        },
        "x_mean": {
            "shape": [int(X.shape[1])],
            "layout": "row_major",
            "dtype": "f64",
            "values": means.astype(np.float64).tolist(),
        },
        "x_scale": {
            "shape": [int(X.shape[1])],
            "layout": "row_major",
            "dtype": "f64",
            "values": scales.astype(np.float64).tolist(),
        },
        "block_weights": {
            "shape": [len(block_weights)],
            "layout": "row_major",
            "dtype": "f64",
            "values": block_weights.astype(np.float64).tolist(),
        },
    }


def _lw_pls_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_neighbors: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Xs, _x_mean, _x_scale = _center_scale(X)
    n = X.shape[0]
    k = int(n_neighbors)
    if k < int(n_components) + 1 or k > n:
        raise ValueError("n_neighbors must be in [n_components + 1, n_samples]")
    predictions = np.zeros((n, Y.shape[1]), dtype=np.float64)
    neighbor_indices = np.zeros((n, k), dtype=np.int64)
    for row in range(n):
        deltas = Xs - Xs[row, :]
        distances = np.sum(deltas * deltas, axis=1)
        order = sorted(range(n), key=lambda idx_value: (float(distances[idx_value]), int(idx_value)))
        selected = np.asarray(order[:k], dtype=np.int64)
        neighbor_indices[row, :] = selected
        model = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
        model.fit(X[selected, :], Y[selected, :])
        pred = model.predict(X[row:row + 1, :]).astype(np.float64, copy=False)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        predictions[row, :] = pred[0, :]
    return {
        "predictions": {
            "shape": list(predictions.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(predictions),
        },
        "neighbor_indices": {
            "shape": list(neighbor_indices.shape),
            "layout": "row_major",
            "dtype": "i64",
            "values": neighbor_indices.reshape(-1, order="C").astype(int).tolist(),
        },
    }


def _variable_importance_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)

    T = model._x_scores.astype(np.float64, copy=False)
    W = model.x_weights_.astype(np.float64, copy=False)
    P = model.x_loadings_.astype(np.float64, copy=False)
    Q = model.y_loadings_.astype(np.float64, copy=False)
    p = X.shape[1]
    component_ssy = np.sum(T * T, axis=0) * np.sum(Q * Q, axis=0)
    vip = np.sqrt(float(p) * ((W * W) @ component_ssy) / float(np.sum(component_ssy)))

    Xs = (X - model._x_mean.astype(np.float64)) / model._x_std.astype(np.float64)
    Xhat = T @ P.T
    explained = np.sum(Xhat * Xhat, axis=0)
    residual = np.sum((Xs - Xhat) * (Xs - Xhat), axis=0)
    selectivity_ratio = explained / np.maximum(residual, np.finfo(np.float64).eps)
    return {
        "vip": {
            "shape": [1, int(vip.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": vip.astype(np.float64).tolist(),
        },
        "selectivity_ratio": {
            "shape": [1, int(selectivity_ratio.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": selectivity_ratio.astype(np.float64).tolist(),
        },
    }


def _rank_descending(values: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    order = sorted(range(scores.size), key=lambda index: (-float(scores[index]), int(index)))
    return np.asarray(order[:int(top_k)], dtype=np.int64)


def _variable_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    base = _variable_importance_expected(X, Y, n_components)
    vip = np.asarray(base["vip"]["values"], dtype=np.float64)
    selectivity_ratio = np.asarray(base["selectivity_ratio"]["values"], dtype=np.float64)

    model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)
    W = model.x_weights_.astype(np.float64, copy=False)
    P = model.x_loadings_.astype(np.float64, copy=False)
    Q = model.y_loadings_.astype(np.float64, copy=False)
    x_scale = model._x_std.astype(np.float64)
    y_scale = np.atleast_1d(model._y_std).astype(np.float64)
    rotations = W @ np.linalg.inv(P.T @ W)
    coefficients = rotations @ Q.T
    coefficients = coefficients * y_scale.reshape(1, -1) / x_scale.reshape(-1, 1)
    coefficient_magnitude = np.max(np.abs(coefficients), axis=1)

    vip_indices = _rank_descending(vip, top_k)
    coefficient_indices = _rank_descending(coefficient_magnitude, top_k)
    selectivity_indices = _rank_descending(selectivity_ratio, top_k)
    return {
        "vip_scores": {
            "shape": [1, int(vip.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": vip.astype(np.float64).tolist(),
        },
        "vip_indices": {
            "shape": [1, int(top_k)],
            "layout": "row_major",
            "dtype": "i64",
            "values": vip_indices.astype(int).tolist(),
        },
        "coefficient_scores": {
            "shape": [1, int(coefficient_magnitude.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_magnitude.astype(np.float64).tolist(),
        },
        "coefficient_indices": {
            "shape": [1, int(top_k)],
            "layout": "row_major",
            "dtype": "i64",
            "values": coefficient_indices.astype(int).tolist(),
        },
        "selectivity_ratio_scores": {
            "shape": [1, int(selectivity_ratio.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": selectivity_ratio.astype(np.float64).tolist(),
        },
        "selectivity_ratio_indices": {
            "shape": [1, int(top_k)],
            "layout": "row_major",
            "dtype": "i64",
            "values": selectivity_indices.astype(int).tolist(),
        },
    }


def _interval_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    step: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    folds = _validation_folds("kfold", X.shape[0], n_splits)

    intervals: list[int] = []
    metrics_rows: list[float] = []
    rmse_values: list[float] = []
    for start in range(0, X.shape[1] - int(interval_width) + 1, int(step)):
        stop = start + int(interval_width)
        predictions = np.zeros_like(Y, dtype=np.float64)
        for train, test in folds:
            train_index = np.asarray(train, dtype=np.int64)
            test_index = np.asarray(test, dtype=np.int64)
            model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
            model.fit(X[train_index, start:stop], Y[train_index])
            fold_predictions = model.predict(X[test_index, start:stop]).astype(np.float64, copy=False)
            if fold_predictions.ndim == 1:
                fold_predictions = fold_predictions.reshape(-1, 1)
            predictions[test_index, :] = fold_predictions
        metrics = _regression_metrics_expected(Y, predictions)["metrics"]["values"]
        intervals.extend([int(start), int(interval_width)])
        metrics_rows.extend(float(value) for value in metrics)
        rmse_values.append(float(metrics[0]))

    best_index = int(np.argmin(np.asarray(rmse_values, dtype=np.float64)))
    return {
        "intervals": {
            "shape": [int(len(rmse_values)), 2],
            "layout": "row_major",
            "dtype": "i64",
            "values": intervals,
        },
        "rmse": {
            "shape": [1, int(len(rmse_values))],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "metrics_by_interval": {
            "shape": [int(len(rmse_values)), 9],
            "layout": "row_major",
            "dtype": "f64",
            "values": metrics_rows,
        },
        "best_index": {
            "shape": [1],
            "layout": "row_major",
            "dtype": "i32",
            "values": [best_index],
        },
        "best_interval": {
            "shape": [1, 2],
            "layout": "row_major",
            "dtype": "i64",
            "values": intervals[2 * best_index: 2 * best_index + 2],
        },
    }


def _interval_partitions(n_features: int, interval_width: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for start in range(0, int(n_features), int(interval_width)):
        length = min(int(interval_width), int(n_features) - start)
        intervals.append((int(start), int(length)))
    return intervals


def _select_interval_columns(X: np.ndarray,
                             intervals: list[tuple[int, int]],
                             active: list[int]) -> np.ndarray:
    columns: list[int] = []
    for interval_index in active:
        start, length = intervals[int(interval_index)]
        columns.extend(range(start, start + length))
    return X[:, np.asarray(columns, dtype=np.int64)]


def _bipls_cv_rmse(X: np.ndarray,
                   Y: np.ndarray,
                   intervals: list[tuple[int, int]],
                   active: list[int],
                   folds: list[tuple[list[int], list[int]]],
                   n_components: int) -> float:
    X_active = _select_interval_columns(X, intervals, active)
    predictions = np.zeros_like(Y, dtype=np.float64)
    for train, test in folds:
        train_index = np.asarray(train, dtype=np.int64)
        test_index = np.asarray(test, dtype=np.int64)
        model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
        model.fit(X_active[train_index], Y[train_index])
        fold_predictions = model.predict(X_active[test_index]).astype(np.float64, copy=False)
        if fold_predictions.ndim == 1:
            fold_predictions = fold_predictions.reshape(-1, 1)
        predictions[test_index, :] = fold_predictions
    metrics = _regression_metrics_expected(Y, predictions)["metrics"]["values"]
    return float(metrics[0])


def _bipls_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    min_intervals: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    intervals = _interval_partitions(X.shape[1], interval_width)
    folds = _validation_folds("kfold", X.shape[0], n_splits)
    active = list(range(len(intervals)))

    rmse_path = [_bipls_cv_rmse(X, Y, intervals, active, folds, n_components)]
    active_counts = [len(active)]
    removed_intervals: list[int] = []
    best_step = 0
    best_rmse = rmse_path[0]
    best_active = active.copy()

    while len(active) > int(min_intervals):
        candidates: list[tuple[float, int, list[int]]] = []
        for interval_index in active:
            trial = [index for index in active if index != interval_index]
            rmse = _bipls_cv_rmse(X, Y, intervals, trial, folds, n_components)
            candidates.append((float(rmse), int(interval_index), trial))
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen_rmse, removed, chosen_active = candidates[0]
        removed_intervals.append(int(removed))
        active = chosen_active
        rmse_path.append(float(chosen_rmse))
        active_counts.append(len(active))
        if chosen_rmse < best_rmse:
            best_rmse = float(chosen_rmse)
            best_step = len(rmse_path) - 1
            best_active = active.copy()

    selected_features: list[int] = []
    for interval_index in best_active:
        start, length = intervals[int(interval_index)]
        selected_features.extend(range(start, start + length))

    return {
        "intervals": {
            "shape": [int(len(intervals)), 2],
            "layout": "row_major",
            "dtype": "i64",
            "values": [value for interval in intervals for value in interval],
        },
        "removed_intervals": {
            "shape": [1, int(len(removed_intervals))],
            "layout": "row_major",
            "dtype": "i64",
            "values": removed_intervals,
        },
        "active_counts": {
            "shape": [1, int(len(active_counts))],
            "layout": "row_major",
            "dtype": "i64",
            "values": active_counts,
        },
        "rmse_path": {
            "shape": [1, int(len(rmse_path))],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_path,
        },
        "best_step": {
            "shape": [1],
            "layout": "row_major",
            "dtype": "i32",
            "values": [int(best_step)],
        },
        "selected_interval_indices": {
            "shape": [1, int(len(best_active))],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(index) for index in best_active],
        },
        "selected_feature_indices": {
            "shape": [1, int(len(selected_features))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_features,
        },
    }


def _sipls_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    combination_size: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    intervals = _interval_partitions(X.shape[1], interval_width)
    folds = _validation_folds("kfold", X.shape[0], n_splits)
    combinations = [
        [int(index) for index in combo]
        for combo in itertools.combinations(range(len(intervals)), int(combination_size))
    ]
    rmse_values = [
        _bipls_cv_rmse(X, Y, intervals, combo, folds, n_components)
        for combo in combinations
    ]
    best_index = min(range(len(combinations)), key=lambda i: (float(rmse_values[i]), combinations[i]))
    best_combo = combinations[best_index]

    selected_features: list[int] = []
    for interval_index in best_combo:
        start, length = intervals[int(interval_index)]
        selected_features.extend(range(start, start + length))

    return {
        "intervals": {
            "shape": [int(len(intervals)), 2],
            "layout": "row_major",
            "dtype": "i64",
            "values": [value for interval in intervals for value in interval],
        },
        "candidate_interval_indices": {
            "shape": [int(len(combinations)), int(combination_size)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(index) for combo in combinations for index in combo],
        },
        "rmse_by_combination": {
            "shape": [1, int(len(rmse_values))],
            "layout": "row_major",
            "dtype": "f64",
            "values": [float(value) for value in rmse_values],
        },
        "best_combination_index": {
            "shape": [1],
            "layout": "row_major",
            "dtype": "i32",
            "values": [int(best_index)],
        },
        "selected_interval_indices": {
            "shape": [1, int(len(best_combo))],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(index) for index in best_combo],
        },
        "selected_feature_indices": {
            "shape": [1, int(len(selected_features))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_features,
        },
    }


def _pls_original_scale_coefficients(model: PLSRegression) -> np.ndarray:
    W = model.x_weights_.astype(np.float64, copy=False)
    P = model.x_loadings_.astype(np.float64, copy=False)
    Q = model.y_loadings_.astype(np.float64, copy=False)
    x_scale = model._x_std.astype(np.float64)
    y_scale = np.atleast_1d(model._y_std).astype(np.float64)
    rotations = W @ np.linalg.inv(P.T @ W)
    coefficients = rotations @ Q.T
    return coefficients * y_scale.reshape(1, -1) / x_scale.reshape(-1, 1)


def _coefficient_stability_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    folds = _advanced_validation_folds("monte_carlo",
                                       X.shape[0],
                                       test_count=test_count,
                                       n_repeats=n_repeats,
                                       seed=seed)
    coefficients: list[np.ndarray] = []
    for train, _test in folds:
        train_index = np.asarray(train, dtype=np.int64)
        model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
        model.fit(X[train_index], Y[train_index])
        coefficients.append(_pls_original_scale_coefficients(model))
    coef_stack = np.stack(coefficients, axis=0)
    mean = np.mean(coef_stack, axis=0)
    std = np.std(coef_stack, axis=0, ddof=1)
    stability = np.abs(mean) / np.maximum(std, np.finfo(np.float64).eps)
    scores = np.max(stability, axis=1)
    selected = _rank_descending(scores, top_k)
    return {
        "stability_scores": {
            "shape": [1, int(scores.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": scores.astype(np.float64).tolist(),
        },
        "selected_indices": {
            "shape": [1, int(top_k)],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected.astype(int).tolist(),
        },
        "mean_coefficients": {
            "shape": list(mean.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(mean),
        },
        "std_coefficients": {
            "shape": list(std.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(std),
        },
    }


def _deterministic_reference_noise(n_samples: int,
                                   n_features: int,
                                   seed: int) -> np.ndarray:
    state = int(seed) & _MASK64
    noise = np.zeros((int(n_samples), int(n_features)), dtype=np.float64)
    for row in range(int(n_samples)):
        for col in range(int(n_features)):
            state, noise[row, col] = _uniform_signed_from_state(state)
    for col in range(int(n_features)):
        values = noise[:, col]
        mean = float(np.mean(values))
        centered = values - mean
        std = float(np.std(centered, ddof=1))
        if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
            std = 1.0
        noise[:, col] = centered / std
    return noise


def _splitmix_permutation(n_samples: int, seed: int) -> np.ndarray:
    state = int(seed) & _MASK64
    permutation = np.arange(int(n_samples), dtype=np.int64)
    for upper in range(int(n_samples) - 1, 0, -1):
        state, bits = _splitmix64_next(state)
        swap_with = int(bits % (upper + 1))
        permutation[upper], permutation[swap_with] = permutation[swap_with], permutation[upper]
    return permutation


def _uve_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    validation_seed: int,
    noise_features: int,
    noise_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    noise = _deterministic_reference_noise(X.shape[0], int(noise_features), int(noise_seed))
    augmented = np.hstack([X, noise])
    stability = _coefficient_stability_expected(augmented,
                                                Y,
                                                n_components,
                                                n_repeats,
                                                test_count,
                                                validation_seed,
                                                top_k=augmented.shape[1])
    scores = np.asarray(stability["stability_scores"]["values"], dtype=np.float64)
    real_scores = scores[:X.shape[1]]
    noise_scores = scores[X.shape[1]:]
    threshold = float(np.max(noise_scores))
    full_order = _rank_descending(real_scores, real_scores.size)
    selected = [int(index) for index in full_order if float(real_scores[int(index)]) > threshold]
    return {
        "real_stability_scores": {
            "shape": [1, int(real_scores.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": real_scores.astype(np.float64).tolist(),
        },
        "noise_stability_scores": {
            "shape": [1, int(noise_scores.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": noise_scores.astype(np.float64).tolist(),
        },
        "noise_threshold": threshold,
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
    }


def _emcuve_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    validation_seed: int,
    noise_features: int,
    noise_seed: int,
    n_ensembles: int,
    vote_threshold: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if n_ensembles < 1:
        raise ValueError("n_ensembles must be positive")
    if not 0.0 < float(vote_threshold) <= 1.0:
        raise ValueError("vote_threshold must be in (0, 1]")

    p = int(X.shape[1])
    vote_counts = np.zeros(p, dtype=np.float64)
    stability_sums = np.zeros(p, dtype=np.float64)
    noise_thresholds = np.zeros(int(n_ensembles), dtype=np.float64)
    for ensemble_index in range(int(n_ensembles)):
        member_seed = (int(noise_seed) + int(ensemble_index) * _SPLITMIX_GOLDEN) & _MASK64
        member = _uve_expected(X,
                               Y,
                               n_components,
                               n_repeats,
                               test_count,
                               validation_seed,
                               noise_features,
                               member_seed)
        real_scores = np.asarray(member["real_stability_scores"]["values"], dtype=np.float64)
        stability_sums += real_scores
        noise_thresholds[ensemble_index] = float(member["noise_threshold"])
        for feature in member["selected_indices"]["values"]:
            vote_counts[int(feature)] += 1.0

    vote_frequencies = vote_counts / float(n_ensembles)
    mean_real_scores = stability_sums / float(n_ensembles)
    selected = [
        int(feature)
        for feature in range(p)
        if float(vote_frequencies[feature]) >= float(vote_threshold)
    ]
    selected.sort(key=lambda feature: (
        -float(vote_frequencies[feature]),
        -float(mean_real_scores[feature]),
        int(feature),
    ))
    return {
        "mean_real_stability_scores": {
            "shape": [1, int(mean_real_scores.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": mean_real_scores.astype(np.float64).tolist(),
        },
        "vote_frequencies": {
            "shape": [1, int(vote_frequencies.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": vote_frequencies.astype(np.float64).tolist(),
        },
        "noise_thresholds": {
            "shape": [1, int(noise_thresholds.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": noise_thresholds.astype(np.float64).tolist(),
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
    }


def _pls_coefficient_scores(X: np.ndarray,
                            Y: np.ndarray,
                            n_components: int) -> np.ndarray:
    model = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)
    coefficients = _pls_original_scale_coefficients(model)
    return np.max(np.abs(coefficients), axis=1).astype(np.float64)


def _randomization_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_permutations: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    observed = _pls_coefficient_scores(X, Y, n_components)
    exceedance_counts = np.zeros(observed.size, dtype=np.int64)
    for permutation_index in range(int(n_permutations)):
        member_seed = (int(seed) + (int(permutation_index) + 1) * _SPLITMIX_GOLDEN) & _MASK64
        permutation = _splitmix_permutation(X.shape[0], member_seed)
        permuted_scores = _pls_coefficient_scores(X, Y[permutation], n_components)
        exceedance_counts += (permuted_scores >= observed).astype(np.int64)

    p_values = (exceedance_counts.astype(np.float64) + 1.0) / float(int(n_permutations) + 1)
    selected = [
        int(feature)
        for feature in range(int(observed.size))
        if float(p_values[feature]) <= float(alpha)
    ]
    selected.sort(key=lambda feature: (
        float(p_values[feature]),
        -float(observed[feature]),
        int(feature),
    ))
    return {
        "observed_scores": {
            "shape": [1, int(observed.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": observed.astype(np.float64).tolist(),
        },
        "p_values": {
            "shape": [1, int(p_values.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": p_values.astype(np.float64).tolist(),
        },
        "exceedance_counts": {
            "shape": [1, int(exceedance_counts.size)],
            "layout": "row_major",
            "dtype": "i64",
            "values": exceedance_counts.astype(int).tolist(),
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
    }


def _standardized_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    centered = X - mean.reshape(1, -1)
    scale = np.std(centered, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > np.finfo(np.float64).eps), scale, 1.0)
    return centered / scale.reshape(1, -1)


def _spa_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)
    coefficients = _pls_original_scale_coefficients(model)
    coefficient_scores = np.max(np.abs(coefficients), axis=1)

    Xs = _standardized_columns(X)
    selected: list[int] = []
    selection_scores: list[float] = []
    start = int(_rank_descending(coefficient_scores, 1)[0])
    selected.append(start)
    selection_scores.append(float(coefficient_scores[start]))

    while len(selected) < int(top_k):
        basis: list[np.ndarray] = []
        for feature in selected:
            vector = Xs[:, int(feature)].astype(np.float64, copy=True)
            for basis_vector in basis:
                vector = vector - float(vector @ basis_vector) * basis_vector
            norm = float(np.linalg.norm(vector))
            if norm > np.finfo(np.float64).eps:
                basis.append(vector / norm)

        best_feature = -1
        best_score = -np.inf
        selected_set = set(selected)
        for feature in range(Xs.shape[1]):
            if feature in selected_set:
                continue
            residual = Xs[:, feature].astype(np.float64, copy=True)
            for basis_vector in basis:
                residual = residual - float(residual @ basis_vector) * basis_vector
            score = float(np.linalg.norm(residual))
            if score > best_score or (score == best_score and feature < best_feature):
                best_feature = feature
                best_score = score
        if best_feature < 0:
            raise RuntimeError("SPA selection failed to find a candidate")
        selected.append(int(best_feature))
        selection_scores.append(float(best_score))

    return {
        "coefficient_scores": {
            "shape": [1, int(coefficient_scores.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_scores.astype(np.float64).tolist(),
        },
        "selection_scores": {
            "shape": [1, int(len(selection_scores))],
            "layout": "row_major",
            "dtype": "f64",
            "values": [float(value) for value in selection_scores],
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
    }


def _copy_columns(X: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)[:, np.asarray([int(c) for c in columns], dtype=np.int64)]


def _pls_coefficient_scores(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> np.ndarray:
    model = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
    coefficients = _pls_original_scale_coefficients(model)
    return np.max(np.abs(coefficients), axis=1)


def _kfold_pls_rmse(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    predictions = np.zeros_like(Y, dtype=np.float64)
    for train, test in _validation_folds("kfold", X.shape[0], n_splits):
        train_index = np.asarray(train, dtype=np.int64)
        test_index = np.asarray(test, dtype=np.int64)
        model = PLSRegression(n_components=int(n_components), scale=True, max_iter=500, tol=1e-6)
        model.fit(X[train_index], Y[train_index])
        fold_predictions = model.predict(X[test_index]).astype(np.float64, copy=False)
        if fold_predictions.ndim == 1:
            fold_predictions = fold_predictions.reshape(-1, 1)
        predictions[test_index, :] = fold_predictions
    metrics = _regression_metrics_expected(Y, predictions)
    return float(metrics["metrics"]["values"][0])


def _cars_retained_count(n_features: int,
                         min_features: int,
                         iteration: int,
                         n_iterations: int) -> int:
    if n_iterations <= 1:
        return int(max(min_features, min(n_features, min_features)))
    ratio = float(min_features) / float(n_features)
    fraction = float(iteration) / float(n_iterations - 1)
    retained = int(math.ceil(float(n_features) * (ratio ** fraction)))
    return int(max(min_features, min(n_features, retained)))


def _cars_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    min_features: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    active = list(range(p))
    coefficient_rows: list[float] = []
    rmse_values: list[float] = []
    retained_counts: list[int] = []
    candidates: list[list[int]] = []
    best_iteration = -1
    best_rmse = math.inf

    for iteration in range(int(n_iterations)):
        active_x = _copy_columns(X, active)
        active_scores = _pls_coefficient_scores(active_x, Y, n_components)
        full_scores = np.zeros(p, dtype=np.float64)
        for local_idx, feature in enumerate(active):
            full_scores[int(feature)] = float(active_scores[local_idx])
        coefficient_rows.extend(full_scores.tolist())

        retain_count = _cars_retained_count(p, int(min_features), iteration, int(n_iterations))
        retain_count = min(retain_count, len(active))
        order = sorted(active, key=lambda feature: (-float(full_scores[int(feature)]), int(feature)))
        candidate = sorted(int(feature) for feature in order[:retain_count])
        candidate_x = _copy_columns(X, candidate)
        rmse = _kfold_pls_rmse(candidate_x, Y, n_components, n_splits)

        retained_counts.append(int(retain_count))
        rmse_values.append(float(rmse))
        candidates.append(candidate)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_iteration = int(iteration)
        active = candidate

    selected = candidates[best_iteration]
    return {
        "coefficient_scores": {
            "shape": [int(n_iterations), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_rows,
        },
        "rmse_by_iteration": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "retained_counts": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "i64",
            "values": retained_counts,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_iteration": int(best_iteration),
        "best_rmse": float(best_rmse),
    }


def _selection_membership(p: int, selected: Sequence[int]) -> list[bool]:
    membership = [False] * int(p)
    for feature in selected:
        membership[int(feature)] = True
    return membership


def _random_frog_add_features(
    selected: list[int],
    p: int,
    global_scores: np.ndarray,
    count: int,
    state: int,
) -> tuple[int, list[int]]:
    candidate = list(selected)
    membership = _selection_membership(p, candidate)
    available = [feature for feature in range(int(p)) if not membership[feature]]
    available.sort(key=lambda feature: (-float(global_scores[feature]), int(feature)))
    for remaining in range(int(count), 0, -1):
        if not available:
            break
        window = max(int(remaining), (len(available) + 1) // 2)
        state, pos = _random_bounded(state, window)
        candidate.append(int(available.pop(pos)))
    return state, sorted(candidate)


def _random_frog_remove_features(
    selected: list[int],
    current_scores: np.ndarray,
    count: int,
    state: int,
) -> tuple[int, list[int]]:
    candidate = list(selected)
    order = sorted(candidate, key=lambda feature: (float(current_scores[int(feature)]), -int(feature)))
    for remaining in range(int(count), 0, -1):
        if not order:
            break
        window = max(int(remaining), (len(order) + 1) // 2)
        state, pos = _random_bounded(state, window)
        feature = int(order.pop(pos))
        candidate.remove(feature)
    return state, sorted(candidate)


def _random_frog_swap_feature(
    selected: list[int],
    p: int,
    global_scores: np.ndarray,
    current_scores: np.ndarray,
    state: int,
) -> tuple[int, list[int]]:
    membership = _selection_membership(p, selected)
    if len(selected) == 0 or all(membership):
        return state, sorted(selected)
    state, reduced = _random_frog_remove_features(selected, current_scores, 1, state)
    membership = _selection_membership(p, selected)
    available = [feature for feature in range(int(p)) if not membership[feature]]
    available.sort(key=lambda feature: (-float(global_scores[feature]), int(feature)))
    if not available:
        return state, sorted(selected)
    window = max(1, (len(available) + 1) // 2)
    state, pos = _random_bounded(state, window)
    reduced.append(int(available[pos]))
    return state, sorted(reduced)


def _random_frog_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    initial_size: int,
    min_size: int,
    max_size: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    global_scores = _pls_coefficient_scores(X, Y, n_components)
    ranked = sorted(range(p), key=lambda feature: (-float(global_scores[feature]), int(feature)))
    current = sorted(ranked[:int(initial_size)])
    current_rmse = _kfold_pls_rmse(_copy_columns(X, current), Y, n_components, n_splits)

    inclusion_counts = np.zeros(p, dtype=np.float64)
    rmse_values: list[float] = []
    subset_sizes: list[int] = []
    best_rmse = float(current_rmse)
    best_indices = list(current)
    state = int(seed) & _MASK64

    for _iteration in range(int(n_iterations)):
        active_x = _copy_columns(X, current)
        active_scores = _pls_coefficient_scores(active_x, Y, n_components)
        current_scores = np.zeros(p, dtype=np.float64)
        for local_idx, feature in enumerate(current):
            current_scores[int(feature)] = float(active_scores[local_idx])

        state, move_raw = _random_bounded(state, 3)
        proposed_size = len(current) + int(move_raw) - 1
        proposed_size = max(int(min_size), min(int(max_size), proposed_size))
        if proposed_size > len(current):
            state, candidate = _random_frog_add_features(current,
                                                         p,
                                                         global_scores,
                                                         proposed_size - len(current),
                                                         state)
        elif proposed_size < len(current):
            state, candidate = _random_frog_remove_features(current,
                                                            current_scores,
                                                            len(current) - proposed_size,
                                                            state)
        else:
            state, candidate = _random_frog_swap_feature(current,
                                                         p,
                                                         global_scores,
                                                         current_scores,
                                                         state)

        if len(candidate) < int(min_size):
            candidate = list(current)
        candidate_rmse = _kfold_pls_rmse(_copy_columns(X, candidate), Y, n_components, n_splits)
        accept = candidate_rmse <= current_rmse
        if not accept:
            temperature = max(abs(float(current_rmse)) * 0.05, 1e-12)
            probability = math.exp((float(current_rmse) - float(candidate_rmse)) / temperature)
            state, unit = _uniform01_from_state(state)
            accept = unit < probability
        if accept:
            current = list(candidate)
            current_rmse = float(candidate_rmse)

        for feature in current:
            inclusion_counts[int(feature)] += 1.0
        rmse_values.append(float(current_rmse))
        subset_sizes.append(int(len(current)))
        if current_rmse < best_rmse:
            best_rmse = float(current_rmse)
            best_indices = list(current)

    inclusion_frequencies = inclusion_counts / float(n_iterations)
    selected = sorted(
        range(p),
        key=lambda feature: (-float(inclusion_frequencies[feature]),
                             -float(global_scores[feature]),
                             int(feature)),
    )[:int(top_k)]
    selected = sorted(int(feature) for feature in selected)
    return {
        "global_scores": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": global_scores.tolist(),
        },
        "inclusion_frequencies": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": inclusion_frequencies.tolist(),
        },
        "rmse_by_iteration": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "subset_sizes": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "i64",
            "values": subset_sizes,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_indices": {
            "shape": [1, int(len(best_indices))],
            "layout": "row_major",
            "dtype": "i64",
            "values": best_indices,
        },
        "best_rmse": float(best_rmse),
    }


def _scars_sample_rows(n_samples: int, sample_count: int, state: int) -> tuple[int, list[int]]:
    keyed: list[tuple[int, int]] = []
    for sample in range(int(n_samples)):
        state, bits = _splitmix64_next(state)
        keyed.append((int(bits), int(sample)))
    keyed.sort()
    return state, sorted(sample for _bits, sample in keyed[:int(sample_count)])


def _scars_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    min_features: int,
    sample_fraction: float,
    seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, p = X.shape
    sample_count = int(math.ceil(float(n) * float(sample_fraction)))
    sample_count = max(int(n_components), max(2, sample_count))
    sample_count = min(int(n), sample_count)

    active = list(range(int(p)))
    coefficient_rows: list[float] = []
    rmse_values: list[float] = []
    retained_counts: list[int] = []
    candidates: list[list[int]] = []
    stability_sums = np.zeros(int(p), dtype=np.float64)
    best_iteration = -1
    best_rmse = math.inf
    state = int(seed) & _MASK64

    for iteration in range(int(n_iterations)):
        state, rows = _scars_sample_rows(n, sample_count, state)
        active_x = X[np.asarray(rows, dtype=np.int64)[:, None], np.asarray(active, dtype=np.int64)]
        active_y = Y[np.asarray(rows, dtype=np.int64), :]
        active_scores = _pls_coefficient_scores(active_x, active_y, n_components)

        full_scores = np.zeros(int(p), dtype=np.float64)
        score_sum = float(np.sum(active_scores))
        for local_idx, feature in enumerate(active):
            raw_score = float(active_scores[local_idx])
            full_scores[int(feature)] = raw_score
            if score_sum > np.finfo(np.float64).eps:
                stability_sums[int(feature)] += raw_score / score_sum
        coefficient_rows.extend(full_scores.tolist())

        running_stability = stability_sums / float(iteration + 1)
        retain_count = _cars_retained_count(p, int(min_features), iteration, int(n_iterations))
        retain_count = min(retain_count, len(active))
        order = sorted(
            active,
            key=lambda feature: (
                -float(running_stability[int(feature)]),
                -float(full_scores[int(feature)]),
                int(feature),
            ),
        )
        candidate = sorted(int(feature) for feature in order[:retain_count])
        candidate_x = _copy_columns(X, candidate)
        rmse = _kfold_pls_rmse(candidate_x, Y, n_components, n_splits)

        retained_counts.append(int(retain_count))
        rmse_values.append(float(rmse))
        candidates.append(candidate)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_iteration = int(iteration)
        active = candidate

    selected = candidates[best_iteration]
    stability_scores = stability_sums / float(n_iterations)
    return {
        "coefficient_scores": {
            "shape": [int(n_iterations), int(p)],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_rows,
        },
        "stability_scores": {
            "shape": [1, int(p)],
            "layout": "row_major",
            "dtype": "f64",
            "values": stability_scores.tolist(),
        },
        "rmse_by_iteration": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "retained_counts": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "i64",
            "values": retained_counts,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "sample_count": int(sample_count),
        "best_iteration": int(best_iteration),
        "best_rmse": float(best_rmse),
    }


def _ga_choose_ranked_features(
    p: int,
    global_scores: np.ndarray,
    target_size: int,
    state: int,
) -> tuple[int, list[int]]:
    available = list(range(int(p)))
    available.sort(key=lambda feature: (-float(global_scores[int(feature)]), int(feature)))
    selected: list[int] = []
    for remaining in range(int(target_size), 0, -1):
        if not available:
            break
        window = max(int(remaining), (len(available) + 1) // 2)
        state, pos = _random_bounded(state, window)
        selected.append(int(available.pop(pos)))
    return state, sorted(selected)


def _ga_remove_feature(
    selected: list[int],
    global_scores: np.ndarray,
    state: int,
) -> tuple[int, list[int]]:
    candidate = list(selected)
    order = sorted(candidate, key=lambda feature: (float(global_scores[int(feature)]), -int(feature)))
    window = max(1, (len(order) + 1) // 2)
    state, pos = _random_bounded(state, window)
    feature = int(order[pos])
    candidate.remove(feature)
    return state, sorted(candidate)


def _ga_add_feature(
    selected: list[int],
    p: int,
    global_scores: np.ndarray,
    state: int,
) -> tuple[int, list[int]]:
    candidate = list(selected)
    membership = _selection_membership(p, candidate)
    available = [feature for feature in range(int(p)) if not membership[feature]]
    available.sort(key=lambda feature: (-float(global_scores[int(feature)]), int(feature)))
    if not available:
        return state, sorted(candidate)
    window = max(1, (len(available) + 1) // 2)
    state, pos = _random_bounded(state, window)
    candidate.append(int(available[pos]))
    return state, sorted(candidate)


def _ga_crossover(
    parent_a: list[int],
    parent_b: list[int],
    p: int,
    global_scores: np.ndarray,
    min_features: int,
    max_features: int,
    state: int,
) -> tuple[int, list[int]]:
    state, move_raw = _random_bounded(state, 3)
    target_size = (len(parent_a) + len(parent_b)) // 2 + int(move_raw) - 1
    target_size = max(int(min_features), min(int(max_features), int(target_size)))
    child = sorted(set(parent_a).intersection(parent_b))
    if len(child) > target_size:
        order = sorted(child, key=lambda feature: (float(global_scores[int(feature)]), -int(feature)))
        child = sorted(feature for feature in child if feature not in set(order[:len(child) - target_size]))
    union = sorted(set(parent_a).union(parent_b),
                   key=lambda feature: (-float(global_scores[int(feature)]), int(feature)))
    while len(child) < target_size:
        membership = _selection_membership(p, child)
        available = [feature for feature in union if not membership[int(feature)]]
        if not available:
            available = [feature for feature in range(int(p)) if not membership[int(feature)]]
            available.sort(key=lambda feature: (-float(global_scores[int(feature)]), int(feature)))
        if not available:
            break
        window = max(1, (len(available) + 1) // 2)
        state, pos = _random_bounded(state, window)
        child.append(int(available[pos]))
        child = sorted(child)
    return state, sorted(child)


def _ga_mutate(
    selected: list[int],
    p: int,
    global_scores: np.ndarray,
    min_features: int,
    max_features: int,
    mutation_rate: float,
    state: int,
) -> tuple[int, list[int]]:
    state, unit = _uniform01_from_state(state)
    if unit >= float(mutation_rate):
        return state, sorted(selected)
    state, move_raw = _random_bounded(state, 3)
    if int(move_raw) == 0 and len(selected) > int(min_features):
        return _ga_remove_feature(selected, global_scores, state)
    if int(move_raw) == 1 and len(selected) < int(max_features):
        return _ga_add_feature(selected, p, global_scores, state)
    if len(selected) > int(min_features) and len(selected) < int(p):
        state, reduced = _ga_remove_feature(selected, global_scores, state)
        return _ga_add_feature(reduced, p, global_scores, state)
    return state, sorted(selected)


def _ga_population_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_generations: int,
    population_size: int,
    min_features: int,
    max_features: int,
    mutation_rate: float,
    seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    global_scores = _pls_coefficient_scores(X, Y, n_components)
    ranked = sorted(range(p), key=lambda feature: (-float(global_scores[int(feature)]), int(feature)))
    span = int(max_features) - int(min_features) + 1
    state = int(seed) & _MASK64
    population: list[list[int]] = [sorted(ranked[:int(max_features)])]
    for individual in range(1, int(population_size)):
        target_size = int(min_features) + (individual % span)
        state, selected = _ga_choose_ranked_features(p, global_scores, target_size, state)
        population.append(selected)

    best_rmse = math.inf
    best_indices: list[int] = []
    best_rmse_by_generation: list[float] = []
    mean_rmse_by_generation: list[float] = []
    best_subset_sizes: list[int] = []
    inclusion_counts = np.zeros(p, dtype=np.float64)

    for generation in range(int(n_generations)):
        evaluated: list[tuple[float, int, list[int]]] = []
        for individual in population:
            rmse = _kfold_pls_rmse(_copy_columns(X, individual), Y, n_components, n_splits)
            evaluated.append((float(rmse), len(individual), list(individual)))
            for feature in individual:
                inclusion_counts[int(feature)] += 1.0
        evaluated.sort(key=lambda item: (item[0], item[1], item[2]))
        generation_best = evaluated[0]
        best_rmse_by_generation.append(float(generation_best[0]))
        mean_rmse_by_generation.append(float(sum(item[0] for item in evaluated) / len(evaluated)))
        best_subset_sizes.append(int(generation_best[1]))
        if generation_best[0] < best_rmse:
            best_rmse = float(generation_best[0])
            best_indices = list(generation_best[2])

        if generation == int(n_generations) - 1:
            break
        elite_count = min(2, int(population_size))
        parent_pool_size = max(elite_count, int(population_size) // 2)
        parent_pool = [item[2] for item in evaluated[:parent_pool_size]]
        next_population = [list(item[2]) for item in evaluated[:elite_count]]
        while len(next_population) < int(population_size):
            state, parent_a_idx = _random_bounded(state, len(parent_pool))
            state, parent_b_idx = _random_bounded(state, len(parent_pool))
            state, child = _ga_crossover(parent_pool[parent_a_idx],
                                         parent_pool[parent_b_idx],
                                         p,
                                         global_scores,
                                         min_features,
                                         max_features,
                                         state)
            state, child = _ga_mutate(child,
                                      p,
                                      global_scores,
                                      min_features,
                                      max_features,
                                      mutation_rate,
                                      state)
            next_population.append(child)
        population = next_population

    inclusion_frequencies = inclusion_counts / float(int(n_generations) * int(population_size))
    return {
        "global_scores": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": global_scores.tolist(),
        },
        "inclusion_frequencies": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": inclusion_frequencies.tolist(),
        },
        "best_rmse_by_generation": {
            "shape": [1, int(n_generations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": best_rmse_by_generation,
        },
        "mean_rmse_by_generation": {
            "shape": [1, int(n_generations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": mean_rmse_by_generation,
        },
        "best_subset_sizes": {
            "shape": [1, int(n_generations)],
            "layout": "row_major",
            "dtype": "i64",
            "values": best_subset_sizes,
        },
        "selected_indices": {
            "shape": [1, int(len(best_indices))],
            "layout": "row_major",
            "dtype": "i64",
            "values": best_indices,
        },
        "best_rmse": float(best_rmse),
    }


def _shaving_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
    shave_fraction: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    active = list(range(p))
    coefficient_rows: list[float] = []
    rmse_values: list[float] = []
    retained_counts: list[int] = []
    candidates: list[list[int]] = []
    best_step = -1
    best_rmse = math.inf

    for step in range(int(n_steps)):
        active_x = _copy_columns(X, active)
        active_scores = _pls_coefficient_scores(active_x, Y, n_components)
        full_scores = np.zeros(p, dtype=np.float64)
        for local_idx, feature in enumerate(active):
            full_scores[int(feature)] = float(active_scores[local_idx])
        coefficient_rows.extend(full_scores.tolist())

        rmse = _kfold_pls_rmse(active_x, Y, n_components, n_splits)
        retained_counts.append(int(len(active)))
        rmse_values.append(float(rmse))
        candidates.append(list(active))
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_step = int(step)

        if step == int(n_steps) - 1 or len(active) <= int(min_features):
            continue
        remove_count = int(math.ceil(float(len(active)) * float(shave_fraction)))
        remove_count = max(1, remove_count)
        remove_count = min(remove_count, len(active) - int(min_features))
        order = sorted(active, key=lambda feature: (float(full_scores[int(feature)]), -int(feature)))
        removed = set(int(feature) for feature in order[:remove_count])
        active = sorted(int(feature) for feature in active if int(feature) not in removed)

    selected = candidates[best_step]
    return {
        "coefficient_scores": {
            "shape": [int(n_steps), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_rows,
        },
        "rmse_by_step": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "retained_counts": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "i64",
            "values": retained_counts,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_step": int(best_step),
        "best_rmse": float(best_rmse),
    }


def _rep_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
    remove_count: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    active = list(range(p))
    coefficient_rows: list[float] = []
    rmse_values: list[float] = []
    retained_counts: list[int] = []
    removed_counts: list[int] = []
    removed_indices: list[int] = []
    candidates: list[list[int]] = []
    best_step = -1
    best_rmse = math.inf

    for step in range(int(n_steps)):
        active_x = _copy_columns(X, active)
        active_scores = _pls_coefficient_scores(active_x, Y, n_components)
        full_scores = np.zeros(p, dtype=np.float64)
        for local_idx, feature in enumerate(active):
            full_scores[int(feature)] = float(active_scores[local_idx])
        coefficient_rows.extend(full_scores.tolist())

        rmse = _kfold_pls_rmse(active_x, Y, n_components, n_splits)
        retained_counts.append(int(len(active)))
        rmse_values.append(float(rmse))
        candidates.append(list(active))
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_step = int(step)

        removed_this_step: list[int] = []
        if step != int(n_steps) - 1 and len(active) > int(min_features):
            actual_remove = min(int(remove_count), len(active) - int(min_features))
            order = sorted(active, key=lambda feature: (float(full_scores[int(feature)]), -int(feature)))
            removed_this_step = sorted(int(feature) for feature in order[:actual_remove])
            removed_set = set(removed_this_step)
            active = sorted(int(feature) for feature in active if int(feature) not in removed_set)
        removed_counts.append(int(len(removed_this_step)))
        removed_indices.extend(removed_this_step)

    selected = candidates[best_step]
    return {
        "coefficient_scores": {
            "shape": [int(n_steps), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": coefficient_rows,
        },
        "rmse_by_step": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "retained_counts": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "i64",
            "values": retained_counts,
        },
        "removed_counts": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "i64",
            "values": removed_counts,
        },
        "removed_indices": {
            "shape": [1, int(len(removed_indices))],
            "layout": "row_major",
            "dtype": "i64",
            "values": removed_indices,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_step": int(best_step),
        "best_rmse": float(best_rmse),
    }


def _ipw_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    top_k: int,
    damping: float,
    weight_floor: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    weights = np.ones(p, dtype=np.float64)
    score_rows: list[float] = []
    weight_rows: list[float] = []
    rmse_values: list[float] = []
    selected_by_iteration: list[int] = []
    candidates: list[list[int]] = []
    best_iteration = -1
    best_rmse = math.inf
    base_scores = _pls_coefficient_scores(X, Y, n_components)

    for iteration in range(int(n_iterations)):
        scores = np.asarray(base_scores, dtype=np.float64) * weights
        max_score = float(np.max(scores))
        if not math.isfinite(max_score) or max_score <= np.finfo(np.float64).eps:
            raise RuntimeError("IPW coefficient scores collapsed")
        scores = scores / max_score
        score_rows.extend(scores.astype(np.float64).tolist())

        weights = float(damping) * weights + (1.0 - float(damping)) * scores
        weights = np.maximum(float(weight_floor), weights)
        max_weight = float(np.max(weights))
        if not math.isfinite(max_weight) or max_weight <= np.finfo(np.float64).eps:
            raise RuntimeError("IPW weights collapsed")
        weights = weights / max_weight
        weight_rows.extend(weights.astype(np.float64).tolist())

        ranking = sorted(range(p), key=lambda feature: (-float(weights[int(feature)]), int(feature)))
        selected = sorted(int(feature) for feature in ranking[:int(top_k)])
        selected_by_iteration.extend(selected)
        rmse = _kfold_pls_rmse(_copy_columns(X, selected), Y, n_components, n_splits)
        rmse_values.append(float(rmse))
        candidates.append(selected)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_iteration = int(iteration)

    final_ranking = sorted(range(p), key=lambda feature: (-float(weights[int(feature)]), int(feature)))
    selected = candidates[best_iteration]
    return {
        "score_path": {
            "shape": [int(n_iterations), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": score_rows,
        },
        "weight_path": {
            "shape": [int(n_iterations), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": weight_rows,
        },
        "rmse_by_iteration": {
            "shape": [1, int(n_iterations)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "selected_by_iteration": {
            "shape": [int(n_iterations), int(top_k)],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_by_iteration,
        },
        "ranking_indices": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(feature) for feature in final_ranking],
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_iteration": int(best_iteration),
        "best_rmse": float(best_rmse),
    }


def _st_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    thresholds: Sequence[float],
    min_selected: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    scores = _pls_coefficient_scores(X, Y, n_components)
    max_score = float(np.max(scores))
    if not math.isfinite(max_score) or max_score <= np.finfo(np.float64).eps:
        raise RuntimeError("ST-PLS coefficient scores collapsed")
    scores = scores / max_score
    ranking = sorted(range(p), key=lambda feature: (-float(scores[int(feature)]), int(feature)))

    rmse_values: list[float] = []
    selected_counts: list[int] = []
    selected_masks: list[int] = []
    candidates: list[list[int]] = []
    best_index = -1
    best_rmse = math.inf

    for threshold_index, threshold in enumerate(thresholds):
        selected = [int(feature) for feature in range(p) if float(scores[int(feature)]) >= float(threshold)]
        if len(selected) < int(min_selected):
            selected = sorted(int(feature) for feature in ranking[:int(min_selected)])
        else:
            selected = sorted(selected)
        selected_set = set(selected)
        selected_masks.extend(1 if feature in selected_set else 0 for feature in range(p))
        rmse = _kfold_pls_rmse(_copy_columns(X, selected), Y, n_components, n_splits)
        rmse_values.append(float(rmse))
        selected_counts.append(int(len(selected)))
        candidates.append(selected)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_index = int(threshold_index)

    selected = candidates[best_index]
    threshold_values = [float(value) for value in thresholds]
    return {
        "coefficient_scores": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": scores.astype(np.float64).tolist(),
        },
        "thresholds": {
            "shape": [1, int(len(threshold_values))],
            "layout": "row_major",
            "dtype": "f64",
            "values": threshold_values,
        },
        "rmse_by_threshold": {
            "shape": [1, int(len(threshold_values))],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "selected_counts": {
            "shape": [1, int(len(threshold_values))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_counts,
        },
        "selected_masks": {
            "shape": [int(len(threshold_values)), p],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_masks,
        },
        "ranking_indices": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(feature) for feature in ranking],
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_threshold_index": int(best_index),
        "best_threshold": float(threshold_values[best_index]),
        "best_rmse": float(best_rmse),
    }


def _bve_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    p = int(X.shape[1])
    active = list(range(p))
    candidate_rmse_rows: list[float] = []
    rmse_values: list[float] = []
    retained_counts: list[int] = []
    removed_indices: list[int] = []
    candidates: list[list[int]] = []
    best_step = -1
    best_rmse = math.inf

    for step in range(int(n_steps)):
        row = np.zeros(p, dtype=np.float64)
        best_removed = -1
        best_candidate: list[int] = []
        best_candidate_rmse = math.inf
        for feature in active:
            candidate = [int(item) for item in active if int(item) != int(feature)]
            if len(candidate) < int(min_features):
                continue
            rmse = _kfold_pls_rmse(_copy_columns(X, candidate), Y, n_components, n_splits)
            row[int(feature)] = float(rmse)
            if (rmse < best_candidate_rmse or
                    (rmse == best_candidate_rmse and (best_removed < 0 or int(feature) < best_removed))):
                best_candidate_rmse = float(rmse)
                best_removed = int(feature)
                best_candidate = candidate
        if best_removed < 0:
            raise RuntimeError("BVE failed to produce a candidate")

        candidate_rmse_rows.extend(row.tolist())
        rmse_values.append(float(best_candidate_rmse))
        retained_counts.append(int(len(best_candidate)))
        removed_indices.append(int(best_removed))
        candidates.append(best_candidate)
        if best_candidate_rmse < best_rmse:
            best_rmse = float(best_candidate_rmse)
            best_step = int(step)
        active = best_candidate

    selected = candidates[best_step]
    return {
        "candidate_rmse": {
            "shape": [int(n_steps), p],
            "layout": "row_major",
            "dtype": "f64",
            "values": candidate_rmse_rows,
        },
        "rmse_by_step": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "retained_counts": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "i64",
            "values": retained_counts,
        },
        "removed_indices": {
            "shape": [1, int(n_steps)],
            "layout": "row_major",
            "dtype": "i64",
            "values": removed_indices,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "best_step": int(best_step),
        "best_rmse": float(best_rmse),
    }


def _round2(values: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64), 2)


def _t2_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    alpha: Sequence[float],
    min_selected: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)
    W = model.x_weights_.astype(np.float64, copy=False)
    p_features, p_components = W.shape
    if p_features <= p_components + 1:
        raise RuntimeError("T2-PLS requires more features than components + 1")

    mean = W.mean(axis=0)
    centered = W - mean.reshape(1, -1)
    covariance = (centered.T @ centered) / float(p_features - 1)
    inv_covariance = np.linalg.inv(covariance)
    raw_t2 = np.einsum("ij,jk,ik->i", centered, inv_covariance, centered)
    t2_scores = _round2(raw_t2)

    alpha_values = [float(item) for item in alpha]
    ucl_values: list[float] = []
    rmse_values: list[float] = []
    selected_counts: list[int] = []
    selected_mask: list[int] = []
    selected_by_alpha: list[list[int]] = []

    beta_a = float(p_components) / 2.0
    beta_b = float(p_features - p_components - 1) / 2.0
    ucl_factor = float((p_features - 1) * (p_features - 1)) / float(p_features)
    for level in alpha_values:
        raw_ucl = ucl_factor * float(betaincinv(beta_a, beta_b, 1.0 - level))
        ucl = float(np.round(raw_ucl, 2))
        ucl_values.append(ucl)
        selected = [int(i) for i, value in enumerate(t2_scores) if float(value) > ucl]
        if len(selected) < int(min_selected):
            order = sorted(range(p_features), key=lambda i: (-float(t2_scores[i]), int(i)))
            selected = [int(i) for i in order[:int(min_selected)]]
        mask = np.zeros(p_features, dtype=np.int64)
        for feature in selected:
            mask[int(feature)] = 1
        selected_mask.extend(int(item) for item in mask.tolist())
        selected_by_alpha.append(selected)
        selected_counts.append(int(len(selected)))
        rmse_values.append(float(_kfold_pls_rmse(_copy_columns(X, selected), Y, n_components, n_splits)))

    best_error_index = min(range(len(alpha_values)), key=lambda i: (rmse_values[i], i))
    min_set_index = min(range(len(alpha_values)), key=lambda i: (selected_counts[i], rmse_values[i], i))
    return {
        "t2_scores": {
            "shape": [1, p_features],
            "layout": "row_major",
            "dtype": "f64",
            "values": t2_scores.astype(np.float64, copy=False).tolist(),
        },
        "ucl_by_alpha": {
            "shape": [1, len(alpha_values)],
            "layout": "row_major",
            "dtype": "f64",
            "values": ucl_values,
        },
        "rmse_by_alpha": {
            "shape": [1, len(alpha_values)],
            "layout": "row_major",
            "dtype": "f64",
            "values": rmse_values,
        },
        "selected_counts": {
            "shape": [1, len(alpha_values)],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_counts,
        },
        "selected_mask": {
            "shape": [len(alpha_values), p_features],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_mask,
        },
        "selected_indices_best_error": {
            "shape": [1, len(selected_by_alpha[best_error_index])],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_by_alpha[best_error_index],
        },
        "selected_indices_min_set": {
            "shape": [1, len(selected_by_alpha[min_set_index])],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected_by_alpha[min_set_index],
        },
        "best_error_index": int(best_error_index),
        "min_set_index": int(min_set_index),
        "best_rmse": float(rmse_values[best_error_index]),
        "min_set_rmse": float(rmse_values[min_set_index]),
    }


def _wvc_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
    normalize: bool,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[1] != 1:
        raise RuntimeError("WVC parity fixture currently targets numeric single-response regression")

    Xwork, _x_mean, _x_scale = _center_scale(X)
    Ywork = Y.astype(np.float64, copy=True)
    p = int(Xwork.shape[1])
    q = int(Ywork.shape[1])
    n_components = int(n_components)
    W = np.zeros((p, n_components), dtype=np.float64)
    P = np.zeros((p, n_components), dtype=np.float64)
    Q = np.zeros((q, n_components), dtype=np.float64)
    WVC = np.zeros((p, n_components), dtype=np.float64)
    ee = np.zeros(n_components, dtype=np.float64)
    ss = np.zeros(n_components, dtype=np.float64)

    eps = np.finfo(np.float64).eps
    for comp in range(n_components):
        w, _right = _dominant_svd_pair(Xwork.T @ Ywork)
        s = Xwork @ w
        denom = float(s @ s)
        if denom <= eps:
            raise RuntimeError("WVC score vector collapsed")
        pvec = (Xwork.T @ s) / denom
        qvec = (Ywork.T @ s) / denom
        ss_value = float((s.reshape(1, -1) @ Ywork @ qvec.reshape(-1, 1))[0, 0])
        ee_value = ss_value / denom
        W[:, comp] = w
        P[:, comp] = pvec
        Q[:, comp] = qvec
        ee[comp] = ee_value
        ss[comp] = ss_value
        denom_wvc = float(ee[:comp + 1] @ ss[:comp + 1])
        if abs(denom_wvc) <= eps:
            raise RuntimeError("WVC denominator collapsed")
        for feature in range(p):
            numerator = float(p) * float((ee[:comp + 1] * (W[feature, :comp + 1] ** 2)) @ ss[:comp + 1])
            WVC[feature, comp] = math.sqrt(max(numerator / denom_wvc, 0.0))
        if normalize:
            max_wvc = float(np.max(WVC[:, comp]))
            if max_wvc > eps:
                WVC[:, comp] /= max_wvc
        Xwork = Xwork - np.outer(s, pvec)
        Ywork = Ywork - np.outer(s, qvec)

    final_scores = WVC[:, n_components - 1]
    selected = sorted(range(p), key=lambda feature: (-float(final_scores[feature]), int(feature)))[:int(top_k)]
    return {
        "weights_W": {
            "shape": list(W.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(W),
        },
        "loadings_P": {
            "shape": list(P.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(P),
        },
        "y_loadings_Q": {
            "shape": list(Q.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(Q),
        },
        "wvc_scores": {
            "shape": list(WVC.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(WVC),
        },
        "final_scores": {
            "shape": [1, p],
            "layout": "row_major",
            "dtype": "f64",
            "values": final_scores.astype(np.float64, copy=False).tolist(),
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(feature) for feature in selected],
        },
    }


def _wvc_threshold_selection_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    normalize: bool,
    score_threshold: float,
    threshold_factor: float,
    min_selected: int,
) -> dict[str, Any]:
    base = _wvc_selection_expected(X,
                                   Y,
                                   n_components,
                                   top_k=np.asarray(X).shape[1],
                                   normalize=normalize)
    final_scores = np.asarray(base["final_scores"]["values"], dtype=np.float64)
    ranked = [int(index) for index in base["selected_indices"]["values"]]
    mean_score = float(np.mean(final_scores))
    effective_threshold = max(float(score_threshold), float(threshold_factor) * mean_score)
    selected = [
        int(index)
        for index in ranked
        if float(final_scores[int(index)]) >= effective_threshold
    ]
    if len(selected) < int(min_selected):
        selected = ranked[:int(min_selected)]

    return {
        "final_scores": base["final_scores"],
        "ranked_indices": {
            "shape": [1, int(len(ranked))],
            "layout": "row_major",
            "dtype": "i64",
            "values": ranked,
        },
        "selected_indices": {
            "shape": [1, int(len(selected))],
            "layout": "row_major",
            "dtype": "i64",
            "values": selected,
        },
        "mean_score": float(mean_score),
        "score_threshold": float(score_threshold),
        "threshold_factor": float(threshold_factor),
        "effective_threshold": float(effective_threshold),
    }


def _component_coefficients_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    model.fit(X, Y)

    W = model.x_weights_.astype(np.float64, copy=False)
    P = model.x_loadings_.astype(np.float64, copy=False)
    Q = model.y_loadings_.astype(np.float64, copy=False)
    x_scale = model._x_std.astype(np.float64)
    y_scale = np.atleast_1d(model._y_std).astype(np.float64)
    blocks: list[np.ndarray] = []
    for prefix in range(1, int(n_components) + 1):
        Wk = W[:, :prefix]
        Pk = P[:, :prefix]
        Qk = Q[:, :prefix]
        rotations = Wk @ np.linalg.inv(Pk.T @ Wk)
        coefficients = rotations @ Qk.T
        coefficients = coefficients * y_scale.reshape(1, -1) / x_scale.reshape(-1, 1)
        blocks.append(coefficients.astype(np.float64, copy=False))
    stacked = np.stack(blocks, axis=0)
    return {
        "coefficients_by_component": {
            "shape": [int(n_components), int(X.shape[1] * Y.shape[1])],
            "layout": "component_major_row_major",
            "dtype": "f64",
            "values": stacked.reshape(-1, order="C").tolist(),
        },
    }


def _validation_folds(kind: str,
                      n_samples: int,
                      n_splits: int = 0,
                      holdout_start: int = 0,
                      holdout_count: int = 0) -> list[tuple[list[int], list[int]]]:
    indices = list(range(int(n_samples)))
    folds: list[tuple[list[int], list[int]]] = []
    if kind == "kfold":
        base = n_samples // n_splits
        remainder = n_samples % n_splits
        start = 0
        for fold in range(n_splits):
            size = base + (1 if fold < remainder else 0)
            stop = start + size
            test = indices[start:stop]
            train = indices[:start] + indices[stop:]
            folds.append((train, test))
            start = stop
    elif kind == "leave_one_out":
        for sample in indices:
            folds.append(([idx for idx in indices if idx != sample], [sample]))
    elif kind == "holdout":
        stop = holdout_start + holdout_count
        folds.append((indices[:holdout_start] + indices[stop:], indices[holdout_start:stop]))
    else:
        raise ValueError(f"unsupported validation fixture kind: {kind}")
    return folds


def _random_bounded(state: int, bound: int) -> tuple[int, int]:
    state, bits = _splitmix64_next(state)
    return state, int(bits % int(bound))


def _uniform01_from_state(state: int) -> tuple[int, float]:
    state, bits = _splitmix64_next(state)
    return state, float(bits >> 11) * (1.0 / float(1 << 53))


def _shuffle_indices(indices: list[int], state: int) -> tuple[int, list[int]]:
    shuffled = list(indices)
    for i in range(len(shuffled), 1, -1):
        state, j = _random_bounded(state, i)
        shuffled[i - 1], shuffled[j] = shuffled[j], shuffled[i - 1]
    return state, shuffled


def _fold_from_test_indices(n_samples: int,
                            test_indices: list[int]) -> tuple[list[int], list[int]]:
    test_set = set(test_indices)
    train = [idx for idx in range(int(n_samples)) if idx not in test_set]
    return train, list(test_indices)


def _fold_from_train_indices(n_samples: int,
                             train_indices: list[int]) -> tuple[list[int], list[int]]:
    train_set = set(train_indices)
    test = [idx for idx in range(int(n_samples)) if idx not in train_set]
    return list(train_indices), test


def _pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n_samples = values.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            delta = values[i] - values[j]
            dist = float(np.dot(delta, delta))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def _representative_indices(distances: np.ndarray,
                            train_count: int) -> list[int]:
    distances = np.asarray(distances, dtype=np.float64)
    n_samples = int(distances.shape[0])
    first = 0
    second = 1
    best_pair = float(distances[0, 1])
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = float(distances[i, j])
            if dist > best_pair:
                best_pair = dist
                first = i
                second = j

    selected = [first, second]
    selected_set = {first, second}
    while len(selected) < int(train_count):
        best_sample = -1
        best_nearest = -1.0
        for sample in range(n_samples):
            if sample in selected_set:
                continue
            nearest = min(float(distances[sample, chosen]) for chosen in selected)
            if nearest > best_nearest:
                best_nearest = nearest
                best_sample = sample
        if best_sample < 0:
            raise RuntimeError("representative sample selection failed")
        selected.append(best_sample)
        selected_set.add(best_sample)
    return selected


def _advanced_validation_folds(kind: str,
                               n_samples: int,
                               n_splits: int = 0,
                               n_repeats: int = 0,
                               test_count: int = 0,
                               train_count: int = 0,
                               seed: int = 0,
                               fold_ids: Sequence[int] | None = None,
                               X: np.ndarray | None = None,
                               Y: np.ndarray | None = None) -> list[tuple[list[int], list[int]]]:
    indices = list(range(int(n_samples)))
    folds: list[tuple[list[int], list[int]]] = []
    if kind == "external":
        if fold_ids is None:
            raise ValueError("external validation fixture requires fold_ids")
        fold_ids = [int(value) for value in fold_ids]
        for fold in range(int(n_splits)):
            test = [idx for idx, fold_id in enumerate(fold_ids) if fold_id == fold]
            train = [idx for idx, fold_id in enumerate(fold_ids) if fold_id != fold]
            folds.append((train, test))
    elif kind == "repeated_kfold":
        state = int(seed) & _MASK64
        for _repeat in range(int(n_repeats)):
            state, shuffled = _shuffle_indices(indices, state)
            base = int(n_samples) // int(n_splits)
            remainder = int(n_samples) % int(n_splits)
            start = 0
            for fold in range(int(n_splits)):
                size = base + (1 if fold < remainder else 0)
                stop = start + size
                folds.append(_fold_from_test_indices(int(n_samples), shuffled[start:stop]))
                start = stop
    elif kind == "monte_carlo":
        state = int(seed) & _MASK64
        for _repeat in range(int(n_repeats)):
            state, shuffled = _shuffle_indices(indices, state)
            folds.append(_fold_from_test_indices(int(n_samples), shuffled[:int(test_count)]))
    elif kind == "kennard_stone":
        if X is None:
            raise ValueError("Kennard-Stone fixture requires X")
        distances = _pairwise_squared_distances(np.asarray(X, dtype=np.float64))
        folds.append(_fold_from_train_indices(int(n_samples),
                                             _representative_indices(distances, int(train_count))))
    elif kind == "spxy":
        if X is None or Y is None:
            raise ValueError("SPXY fixture requires X and Y")
        x_distances = _pairwise_squared_distances(np.asarray(X, dtype=np.float64))
        y_distances = _pairwise_squared_distances(np.asarray(Y, dtype=np.float64))
        max_x = float(np.max(x_distances))
        max_y = float(np.max(y_distances))
        combined = (x_distances / max_x if max_x > 0.0 else np.zeros_like(x_distances))
        combined = combined + (y_distances / max_y if max_y > 0.0 else np.zeros_like(y_distances))
        folds.append(_fold_from_train_indices(int(n_samples),
                                             _representative_indices(combined, int(train_count))))
    else:
        raise ValueError(f"unsupported advanced validation fixture kind: {kind}")
    return folds


def _validation_expected(kind: str,
                         n_samples: int,
                         n_splits: int = 0,
                         holdout_start: int = 0,
                         holdout_count: int = 0) -> dict[str, Any]:
    folds = _validation_folds(kind, n_samples, n_splits, holdout_start, holdout_count)
    train_offsets = [0]
    test_offsets = [0]
    train_indices: list[int] = []
    test_indices: list[int] = []
    for train, test in folds:
        train_indices.extend(train)
        test_indices.extend(test)
        train_offsets.append(len(train_indices))
        test_offsets.append(len(test_indices))
    return {
        "n_folds": len(folds),
        "train_offsets": {"values": train_offsets},
        "train_indices": {"values": train_indices},
        "test_offsets":  {"values": test_offsets},
        "test_indices":  {"values": test_indices},
    }


def _folds_to_validation_expected(folds: list[tuple[list[int], list[int]]]) -> dict[str, Any]:
    train_offsets = [0]
    test_offsets = [0]
    train_indices: list[int] = []
    test_indices: list[int] = []
    for train, test in folds:
        train_indices.extend(train)
        test_indices.extend(test)
        train_offsets.append(len(train_indices))
        test_offsets.append(len(test_indices))
    return {
        "n_folds": len(folds),
        "train_offsets": {"values": train_offsets},
        "train_indices": {"values": train_indices},
        "test_offsets":  {"values": test_offsets},
        "test_indices":  {"values": test_indices},
    }


def _advanced_validation_expected(kind: str,
                                  n_samples: int,
                                  n_splits: int = 0,
                                  n_repeats: int = 0,
                                  test_count: int = 0,
                                  train_count: int = 0,
                                  seed: int = 0,
                                  fold_ids: Sequence[int] | None = None,
                                  X: np.ndarray | None = None,
                                  Y: np.ndarray | None = None) -> dict[str, Any]:
    folds = _advanced_validation_folds(kind,
                                       n_samples,
                                       n_splits,
                                       n_repeats,
                                       test_count,
                                       train_count,
                                       seed,
                                       fold_ids,
                                       X,
                                       Y)
    return _folds_to_validation_expected(folds)


def _cross_validation_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    folds = _validation_folds("kfold", X.shape[0], n_splits)
    predictions = np.zeros_like(Y, dtype=np.float64)
    test_offsets = [0]
    test_indices: list[int] = []
    for train, test in folds:
        train_index = np.asarray(train, dtype=np.int64)
        test_index = np.asarray(test, dtype=np.int64)
        model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
        model.fit(X[train_index], Y[train_index])
        fold_predictions = model.predict(X[test_index]).astype(np.float64, copy=False)
        if fold_predictions.ndim == 1:
            fold_predictions = fold_predictions.reshape(-1, 1)
        predictions[test_index, :] = fold_predictions
        test_indices.extend(test)
        test_offsets.append(len(test_indices))

    return {
        "predictions": {
            "shape": list(predictions.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(predictions),
        },
        "test_offsets": {"values": test_offsets},
        "test_indices": {"values": test_indices},
        **_regression_metrics_expected(Y, predictions),
    }


def _simpls_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    Xs, x_mean, x_scale = _center_scale(X_train)
    Ys, y_mean, y_scale = _center_scale(Y_train)
    p = Xs.shape[1]
    q = Ys.shape[1]
    K = int(n_components)
    Z = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    V = np.zeros((p, K), dtype=np.float64)
    S = Xs.T @ Ys
    eps = np.finfo(np.float64).eps
    for comp in range(K):
        r = _dominant_direction(S)
        t = Xs @ r
        t_norm = np.linalg.norm(t)
        if t_norm <= eps:
            raise RuntimeError(f"SIMPLS score collapsed at component {comp}")
        t = t / t_norm
        r = r / t_norm
        p_load = Xs.T @ t
        q_load = Ys.T @ t
        v = p_load.copy()
        if comp > 0:
            v = v - V[:, :comp] @ (V[:, :comp].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm <= eps:
            raise RuntimeError(f"SIMPLS deflation basis collapsed at component {comp}")
        v = v / v_norm
        Z[:, comp] = r
        P[:, comp] = p_load
        Q[:, comp] = q_load
        V[:, comp] = v
        S = S - np.outer(v, v.T @ S)
    rotations = Z @ np.linalg.inv(P.T @ Z)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    return y_mean.reshape(1, -1) + (X_test - x_mean.reshape(1, -1)) @ coef


def _component_cv_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components_max: int,
    n_splits: int,
    solver: str,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    folds = _validation_folds("kfold", X.shape[0], n_splits)
    rows: list[float] = []
    rmse_values: list[float] = []
    for n_components in range(1, int(n_components_max) + 1):
        predictions = np.zeros_like(Y, dtype=np.float64)
        for train, test in folds:
            train_index = np.asarray(train, dtype=np.int64)
            test_index = np.asarray(test, dtype=np.int64)
            if solver == "simpls":
                fold_predictions = _simpls_predict(X[train_index],
                                                   Y[train_index],
                                                   X[test_index],
                                                   n_components)
            elif solver == "nipals":
                model = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
                model.fit(X[train_index], Y[train_index])
                fold_predictions = model.predict(X[test_index]).astype(np.float64, copy=False)
            else:
                raise ValueError(f"unsupported component CV solver: {solver}")
            if fold_predictions.ndim == 1:
                fold_predictions = fold_predictions.reshape(-1, 1)
            predictions[test_index, :] = fold_predictions
        metrics = _regression_metrics_expected(Y, predictions)["metrics"]["values"]
        rows.extend(metrics)
        rmse_values.append(float(metrics[0]))
    best_n_components = int(np.argmin(np.asarray(rmse_values, dtype=np.float64)) + 1)
    return {
        "metrics_by_component": {
            "shape": [int(n_components_max), 9],
            "layout": "row_major",
            "dtype": "f64",
            "values": rows,
        },
        "best_n_components": {
            "shape": [1],
            "layout": "row_major",
            "dtype": "i32",
            "values": [best_n_components],
        },
    }


def _power_pls_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        x_weights, y_weights = _dominant_svd_pair_power(Xk.T @ Yk)
        t = Xk @ x_weights
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"power PLS score collapsed at component {comp}")
        y_weight_ss = float(y_weights @ y_weights)
        if y_weight_ss <= eps:
            raise RuntimeError(f"power PLS Y weights collapsed at component {comp}")
        u = (Yk @ y_weights) / (y_weight_ss + eps)
        p_load = (Xk.T @ t) / t_ss
        q_load = (Yk.T @ t) / t_ss
        Xk = Xk - np.outer(t, p_load)
        Yk = Yk - np.outer(t, q_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        U_scores[:, comp] = u

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _randomized_svd_pls_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    random_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        x_weights, y_weights = _dominant_svd_pair_randomized(
            Xk.T @ Yk, seed=random_seed, component=comp
        )
        t = Xk @ x_weights
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"randomized SVD PLS score collapsed at component {comp}")
        y_weight_ss = float(y_weights @ y_weights)
        if y_weight_ss <= eps:
            raise RuntimeError(f"randomized SVD PLS Y weights collapsed at component {comp}")
        u = (Yk @ y_weights) / (y_weight_ss + eps)
        p_load = (Xk.T @ t) / t_ss
        q_load = (Yk.T @ t) / t_ss
        Xk = Xk - np.outer(t, p_load)
        Yk = Yk - np.outer(t, q_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        U_scores[:, comp] = u

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _canonical_nipals_pair(
    Xk: np.ndarray,
    Yk: np.ndarray,
    component: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    n, p = Xk.shape
    q = Yk.shape[1]
    eps = np.finfo(np.float64).eps
    y_eps = np.finfo(Yk.dtype).eps

    mask = np.all(np.abs(Yk) < 10.0 * y_eps, axis=0)
    Yk[:, mask] = 0.0

    initial = None
    for target in range(q):
        if np.any(np.abs(Yk[:, target]) > eps):
            initial = target
            break
    if initial is None:
        raise RuntimeError(f"canonical Y residual collapsed at component {component}")

    y_score = Yk[:, initial].copy()
    x_weights_old = np.full(p, 100.0, dtype=np.float64)
    x_weights = np.zeros(p, dtype=np.float64)
    y_weights = np.zeros(q, dtype=np.float64)
    converged = False

    for _iteration in range(max_iter):
        y_score_ss = float(y_score @ y_score)
        if y_score_ss <= eps:
            raise RuntimeError(f"canonical Y score collapsed at component {component}")
        x_weights = (Xk.T @ y_score) / y_score_ss
        x_norm = np.linalg.norm(x_weights)
        if x_norm <= eps:
            raise RuntimeError(f"canonical X weights collapsed at component {component}")
        x_weights = x_weights / (x_norm + eps)

        x_score = Xk @ x_weights
        x_score_ss = float(x_score @ x_score)
        if x_score_ss <= eps:
            raise RuntimeError(f"canonical X score collapsed at component {component}")
        y_weights = (Yk.T @ x_score) / x_score_ss
        y_norm = np.linalg.norm(y_weights)
        if y_norm <= eps:
            raise RuntimeError(f"canonical Y weights collapsed at component {component}")
        y_weights = y_weights / (y_norm + eps)
        y_score = (Yk @ y_weights) / (float(y_weights @ y_weights) + eps)

        diff = float((x_weights - x_weights_old) @ (x_weights - x_weights_old))
        if diff < tol or q == 1:
            converged = True
            break
        x_weights_old = x_weights.copy()

    if not converged:
        raise RuntimeError(f"canonical NIPALS failed to converge at component {component}")

    sign_idx = int(np.argmax(np.abs(x_weights)))
    if x_weights[sign_idx] < 0.0:
        x_weights = -x_weights
        y_weights = -y_weights
    return x_weights, y_weights


def _canonical_pls_expected(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    solver: str,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        if solver == "nipals":
            x_weights, y_weights = _canonical_nipals_pair(Xk, Yk, comp)
        elif solver == "svd":
            x_weights, y_weights = _dominant_svd_pair(Xk.T @ Yk)
        else:
            raise ValueError(f"unsupported canonical solver: {solver}")

        t = Xk @ x_weights
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"canonical X score collapsed at component {comp}")
        u = Yk @ y_weights
        u_ss = float(u @ u)
        if u_ss <= eps:
            raise RuntimeError(f"canonical Y score collapsed at component {comp}")

        p_load = (Xk.T @ t) / t_ss
        Xk = Xk - np.outer(t, p_load)
        q_load = (Yk.T @ u) / u_ss
        Yk = Yk - np.outer(u, q_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        U_scores[:, comp] = u

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _kernel_pls_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    n_targets = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((n_targets, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        initial_target = None
        for target in range(n_targets):
            if np.any(np.abs(Yk[:, target]) > eps):
                initial_target = target
                break
        if initial_target is None:
            raise RuntimeError(f"kernel PLS Y residual collapsed at component {comp}")

        kernel = Xk @ Xk.T
        y_score = Yk[:, initial_target].copy()
        old_t = np.zeros(n, dtype=np.float64)
        converged = False
        x_weights = np.zeros(p, dtype=np.float64)
        t = np.zeros(n, dtype=np.float64)
        q_load = np.zeros(n_targets, dtype=np.float64)

        for iteration in range(500):
            y_ss = float(y_score @ y_score)
            if y_ss <= eps:
                raise RuntimeError(f"kernel PLS Y score collapsed at component {comp}")
            raw_t = (kernel @ y_score) / y_ss
            raw_norm = np.linalg.norm(raw_t)
            if raw_norm <= eps:
                raise RuntimeError(f"kernel PLS X score collapsed at component {comp}")
            t = raw_t / raw_norm
            x_weights = ((Xk.T @ y_score) / y_ss) / raw_norm
            t_ss = float(t @ t)
            q_load = (Yk.T @ t) / t_ss
            q_ss = float(q_load @ q_load)
            if q_ss <= eps:
                raise RuntimeError(f"kernel PLS Y loadings collapsed at component {comp}")
            y_score = (Yk @ q_load) / q_ss

            if iteration > 0 or n_targets == 1:
                diff_same = float((t - old_t) @ (t - old_t))
                diff_opposite = float((t + old_t) @ (t + old_t))
                if min(diff_same, diff_opposite) < 1e-6 or n_targets == 1:
                    converged = True
                    break
            old_t = t.copy()

        if not converged:
            raise RuntimeError(f"kernel PLS iteration failed to converge at component {comp}")

        sign_idx = int(np.argmax(np.abs(x_weights)))
        if x_weights[sign_idx] < 0.0:
            x_weights = -x_weights
            t = -t
            q_load = -q_load

        t_ss = float(t @ t)
        p_load = (Xk.T @ t) / t_ss
        q_ss = float(q_load @ q_load)
        Xk = Xk - np.outer(t, p_load)
        Yk = Yk - np.outer(t, q_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [n_targets],            "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [n_targets],            "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [n_targets],            "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _orthogonal_scores_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Yk, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Yk.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        if q == 1:
            y_score = Yk[:, 0].copy()
            old_t = np.zeros(n, dtype=np.float64)
        else:
            col_ss = np.sum(Yk * Yk, axis=0)
            best = int(np.argmax(col_ss))
            if col_ss[best] <= eps:
                raise RuntimeError(f"orthogonal-scores Y residual collapsed at component {comp}")
            y_score = Yk[:, best].copy()
            old_t = np.zeros(n, dtype=np.float64)

        x_weights = np.zeros(p, dtype=np.float64)
        x_score = np.zeros(n, dtype=np.float64)
        y_load = np.zeros(q, dtype=np.float64)
        converged = False
        for iteration in range(500):
            x_weights = Xk.T @ y_score
            w_norm = np.linalg.norm(x_weights)
            if w_norm <= eps:
                raise RuntimeError(f"orthogonal-scores X weights collapsed at component {comp}")
            x_weights = x_weights / w_norm
            x_score = Xk @ x_weights
            t_ss = float(x_score @ x_score)
            if t_ss <= eps:
                raise RuntimeError(f"orthogonal-scores X score collapsed at component {comp}")
            y_load = Yk.T @ (x_score / t_ss)

            if q == 1:
                converged = True
                break

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (x_score - old_t) / x_score
            ratio = ratio[~np.isnan(ratio)]
            if float(np.sum(np.abs(ratio))) < 1e-6:
                converged = True
                break

            q_ss = float(y_load @ y_load)
            if q_ss <= eps:
                raise RuntimeError(f"orthogonal-scores Y loadings collapsed at component {comp}")
            y_score = (Yk @ y_load) / q_ss
            old_t = x_score.copy()
            if iteration + 1 >= 500:
                raise RuntimeError(f"orthogonal-scores iteration failed at component {comp}")

        if not converged:
            raise RuntimeError(f"orthogonal-scores iteration failed at component {comp}")

        t_ss = float(x_score @ x_score)
        p_load = Xk.T @ (x_score / t_ss)

        sign_idx = int(np.argmax(np.abs(x_weights)))
        if x_weights[sign_idx] < 0.0:
            x_weights = -x_weights
            x_score = -x_score
            p_load = -p_load
            y_load = -y_load
            y_score = -y_score

        Xk = Xk - np.outer(x_score, p_load)
        Yk = Yk - np.outer(x_score, y_load)

        W[:, comp] = x_weights
        P[:, comp] = p_load
        Q[:, comp] = y_load
        T[:, comp] = x_score
        U_scores[:, comp] = y_score

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _opls_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Ys, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Ys.shape[1]
    K = int(n_components)
    eps = np.finfo(np.float64).eps

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    R = np.zeros((p, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    projection = np.eye(p, dtype=np.float64)

    def sign_normalize(
        weights: np.ndarray,
        scores: np.ndarray | None = None,
        loadings: np.ndarray | None = None,
        y_loadings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        sign_idx = int(np.argmax(np.abs(weights)))
        if weights[sign_idx] >= 0.0:
            return weights, scores, loadings, y_loadings
        weights = -weights
        if scores is not None:
            scores = -scores
        if loadings is not None:
            loadings = -loadings
        if y_loadings is not None:
            y_loadings = -y_loadings
        return weights, scores, loadings, y_loadings

    def predictive_component(
        Xwork: np.ndarray,
        component: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if q == 1:
            weights = Xwork.T @ Ys[:, 0]
            w_norm = np.linalg.norm(weights)
            if w_norm <= eps:
                raise RuntimeError(f"OPLS predictive weights collapsed at component {component}")
            weights = weights / w_norm
            y_weights = np.array([1.0], dtype=np.float64)
        else:
            weights, y_weights = _dominant_svd_pair(Xwork.T @ Ys)

        scores = Xwork @ weights
        t_ss = float(scores @ scores)
        if t_ss <= eps:
            raise RuntimeError(f"OPLS predictive score collapsed at component {component}")
        loadings = (Xwork.T @ scores) / t_ss
        y_loadings = (Ys.T @ scores) / t_ss
        y_weight_ss = float(y_weights @ y_weights)
        if y_weight_ss <= eps:
            raise RuntimeError(f"OPLS predictive Y weights collapsed at component {component}")
        y_scores = (Ys @ y_weights) / (y_weight_ss + eps)
        weights, scores, loadings, y_loadings = sign_normalize(
            weights, scores, loadings, y_loadings
        )
        assert scores is not None and loadings is not None and y_loadings is not None
        if q == 1:
            y_scores = Ys[:, 0] * y_loadings[0] / (float(y_loadings @ y_loadings) + eps)
        return weights, scores, loadings, y_loadings, y_scores

    for orth in range(max(K - 1, 0)):
        pred_w, _pred_t, pred_p, _pred_q, _pred_u = predictive_component(Xk, orth + 1)
        alignment = float((pred_w @ pred_p) / (pred_w @ pred_w))
        orth_w = pred_p - pred_w * alignment
        orth_norm = np.linalg.norm(orth_w)
        if orth_norm <= eps:
            raise RuntimeError(f"OPLS orthogonal weights collapsed at component {orth}")
        orth_w = orth_w / orth_norm
        orth_w, _, _, _ = sign_normalize(orth_w)
        orth_t = Xk @ orth_w
        orth_t_ss = float(orth_t @ orth_t)
        if orth_t_ss <= eps:
            raise RuntimeError(f"OPLS orthogonal score collapsed at component {orth}")
        orth_p = (Xk.T @ orth_t) / orth_t_ss
        raw_rotation = projection @ orth_w

        col = orth + 1
        W[:, col] = orth_w
        P[:, col] = orth_p
        R[:, col] = raw_rotation
        T[:, col] = orth_t

        Xk = Xk - np.outer(orth_t, orth_p)
        projection = projection - np.outer(raw_rotation, orth_p)

    pred_w, pred_t, pred_p, pred_q, pred_u = predictive_component(Xk, 0)
    raw_pred = projection @ pred_w
    denom = float(pred_p @ pred_w)
    if abs(denom) <= eps:
        raise RuntimeError("OPLS predictive denominator collapsed")
    coef_std = np.outer(raw_pred, pred_q) / denom
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef

    W[:, 0] = pred_w
    P[:, 0] = pred_p
    Q[:, 0] = pred_q
    R[:, 0] = raw_pred
    T[:, 0] = pred_t
    U_scores[:, 0] = pred_u

    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(R.shape),          "values": _flatten_rowmajor(R), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _pcr_expected(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Xk, x_mean, x_scale = _center_scale(X)
    Ys, y_mean, y_scale = _center_scale(Y)
    n, p = Xk.shape
    q = Ys.shape[1]
    K = int(n_components)

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((q, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    U_scores = np.zeros((n, K), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for comp in range(K):
        eigenvalues, eigenvectors = np.linalg.eigh(Xk.T @ Xk)
        best = int(np.argmax(eigenvalues))
        if eigenvalues[best] <= eps:
            raise RuntimeError(f"PCR X covariance collapsed at component {comp}")
        direction = eigenvectors[:, best].astype(np.float64, copy=True)
        sign_idx = int(np.argmax(np.abs(direction)))
        if direction[sign_idx] < 0.0:
            direction = -direction

        t = Xk @ direction
        t_ss = float(t @ t)
        if t_ss <= eps:
            raise RuntimeError(f"PCR score collapsed at component {comp}")
        p_load = (Xk.T @ t) / t_ss
        q_load = (Ys.T @ t) / t_ss
        q_ss = float(q_load @ q_load)
        if q_ss > eps:
            u = (Ys @ q_load) / q_ss
        else:
            u = np.zeros(n, dtype=np.float64)
        Xk = Xk - np.outer(t, p_load)

        W[:, comp] = direction
        P[:, comp] = p_load
        Q[:, comp] = q_load
        T[:, comp] = t
        U_scores[:, comp] = u

    rotations = W @ np.linalg.inv(P.T @ W)
    coef_std = rotations @ Q.T
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef
    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [q],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [q],                    "values": y_scale.astype(np.float64).tolist()},
        "weights_W":     {"shape": list(W.shape),          "values": _flatten_rowmajor(W), "sign_invariant": True},
        "loadings_P":    {"shape": list(P.shape),          "values": _flatten_rowmajor(P), "sign_invariant": True},
        "y_loadings_Q":  {"shape": list(Q.shape),          "values": _flatten_rowmajor(Q), "sign_invariant": True},
        "rotations_R":   {"shape": list(rotations.shape),  "values": _flatten_rowmajor(rotations), "sign_invariant": True},
        "scores_T":      {"shape": list(T.shape),          "values": _flatten_rowmajor(T), "sign_invariant": True},
        "predict_train": {"shape": list(preds.shape),      "values": _flatten_rowmajor(preds)},
    }


def _fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray, Y: np.ndarray,
    n_components: int,
    tolerance_table_row: str,
) -> dict[str, Any]:
    pls = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6)
    pls.fit(X, Y)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "sklearn.cross_decomposition.PLSRegression",
            "version":          __import__("sklearn").__version__,
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "nipals",
                "max_iter":       500,
                "tol":            1e-6,
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _expected_from_pls(pls, X, Y),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  tolerance_table_row,
        },
    }


def _simpls_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._simpls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "simpls",
                "reference":      "NumPy port of nirs4all AOM_v0 simpls_standard",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _simpls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-simpls",
        },
    }


def _svd_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._svd_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "svd",
                "reference":      "NumPy SVD with regression deflation",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _svd_pls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-svd",
        },
    }


def _pls_svd_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._pls_svd_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "canonical",
                "algorithm":      "pls-svd",
                "reference":      "NumPy mirror of sklearn PLSSVD cross-covariance SVD",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _pls_svd_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "sklearn/PLSSVD",
        },
    }


def _pipeline_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    operators: list[str],
    Y: np.ndarray | None = None,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.zeros((X.shape[0], 1), dtype=np.float64) if Y is None else np.asarray(Y, dtype=np.float64)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._pipeline_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "operators":      operators,
                "reference":      "NumPy/SciPy preprocessing pipeline",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _pipeline_expected(X, operators, Y),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-preprocessing",
        },
    }


def _aom_preprocessing_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    operators: list[str],
    gating_mode: str,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._aom_preprocessing_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "operators":    operators,
                "gating_mode":  gating_mode,
                "reference":    "NumPy AOM preprocessing bank",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
        },
        "expected": _aom_preprocessing_expected(X, operators, gating_mode),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-aom-preprocessing",
        },
    }


_AOM_BENCH_OPERATOR_KIND_IDS = {
    "identity": 0,
    "detrend_d1": 7,
    "sg_smooth_w5_p2": 8,
    "sg_d1_w5_p2": 9,
    "nw_g3_s1_d1": 10,
    "fd_d1": 15,
    "whittaker_l10": 16,
    "fck_a1.00_s1.00_k7": 17,
}


_AOM_BENCH_OPERATOR_PARAMS = {
    "identity": [],
    "detrend_d1": [1.0],
    "sg_smooth_w5_p2": [5.0, 2.0],
    "sg_d1_w5_p2": [5.0, 2.0, 1.0],
    "nw_g3_s1_d1": [1.0, 3.0, 1.0],
    "fd_d1": [1.0],
    "whittaker_l10": [10.0],
    "fck_a1.00_s1.00_k7": [1.0, 1.0, 7.0, 3.0],
}


def _aom_bench_root() -> Path:
    env = os.environ.get("PLS4ALL_AOM_BENCH_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    workspace_root = here.parents[5]
    candidates.extend([
        workspace_root / "nirs4all" / "bench" / "AOM_v0",
        workspace_root / "bench" / "AOM_v0",
        repo_root / ".." / "nirs4all" / "bench" / "AOM_v0",
    ])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "aompls" / "selection.py").exists():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"nirs4all bench AOM_v0 reference not found; searched: {searched}")


def _aom_bench_imports() -> Any:
    root = _aom_bench_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from aompls.operators import (  # type: ignore
        DetrendProjectionOperator,
        FiniteDifferenceOperator,
        FCKOperator,
        IdentityOperator,
        NorrisWilliamsOperator,
        SavitzkyGolayOperator,
        WhittakerOperator,
    )

    return {
        "DetrendProjectionOperator": DetrendProjectionOperator,
        "FiniteDifferenceOperator":  FiniteDifferenceOperator,
        "FCKOperator":               FCKOperator,
        "IdentityOperator":          IdentityOperator,
        "NorrisWilliamsOperator":    NorrisWilliamsOperator,
        "SavitzkyGolayOperator":     SavitzkyGolayOperator,
        "WhittakerOperator":         WhittakerOperator,
    }


def _aom_bench_operator(name: str, p: int) -> Any:
    classes = _aom_bench_imports()
    if name == "identity":
        return classes["IdentityOperator"](p=p)
    if name == "detrend_d1":
        return classes["DetrendProjectionOperator"](degree=1, p=p)
    if name == "sg_smooth_w5_p2":
        return classes["SavitzkyGolayOperator"](window_length=5, polyorder=2, deriv=0, p=p)
    if name == "sg_d1_w5_p2":
        return classes["SavitzkyGolayOperator"](window_length=5, polyorder=2, deriv=1, p=p)
    if name == "nw_g3_s1_d1":
        return classes["NorrisWilliamsOperator"](gap=3, smoothing=1, order=1, p=p)
    if name == "fd_d1":
        return classes["FiniteDifferenceOperator"](order=1, p=p)
    if name == "whittaker_l10":
        return classes["WhittakerOperator"](lam=10.0, p=p)
    if name == "fck_a1.00_s1.00_k7":
        return classes["FCKOperator"](alpha=1.0, scale=1.0, kernel_size=7, sigma=3.0, p=p)
    raise ValueError(f"unsupported bench AOM operator fixture: {name}")


def _aom_bench_operators(names: Sequence[str], p: int) -> list[Any]:
    return [_aom_bench_operator(name, p) for name in names]


def _aom_bench_reference_expected(
    X: np.ndarray,
    Y: np.ndarray,
    operator_names: Sequence[str],
    max_components: int,
    cv: int,
    random_state: int,
) -> dict[str, Any]:
    root = _aom_bench_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from sklearn.model_selection import KFold

    from aompls.estimators import AOMPLSRegressor
    from aompls.scorers import CriterionConfig
    from aompls.selection import _auto_prefix_score

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    y1 = Y.ravel() if Y.shape[1] == 1 else Y
    p = X.shape[1]
    operators = _aom_bench_operators(operator_names, p)
    operator_names = [op.name for op in operators]

    model = AOMPLSRegressor(
        n_components="auto",
        max_components=int(max_components),
        engine="simpls_materialized",
        selection="global",
        criterion="cv",
        operator_bank=operators,
        orthogonalization="transformed",
        center=True,
        scale=False,
        cv=int(cv),
        random_state=int(random_state),
        one_se_rule=False,
    ).fit(X, y1)

    Xc = X - X.mean(axis=0)
    yc = Y - Y.mean(axis=0)
    criterion = CriterionConfig(
        kind="cv",
        cv=int(cv),
        random_state=int(random_state),
        task="regression",
        repeats=1,
        one_se_rule=False,
    )
    curves: list[list[float]] = []
    operator_scores: list[float] = []
    for op_index in range(len(operators)):
        _best_k, score, curve = _auto_prefix_score(
            "simpls_materialized",
            operators,
            [op_index] * int(max_components),
            Xc,
            yc,
            int(max_components),
            criterion,
            "transformed",
        )
        curves.append(np.asarray(curve, dtype=np.float64).tolist())
        operator_scores.append(float(score))

    folds = list(KFold(n_splits=int(cv), shuffle=True, random_state=int(random_state)).split(Xc, yc))
    train_indices: list[int] = []
    test_indices: list[int] = []
    train_offsets = [0]
    test_offsets = [0]
    for train, test in folds:
        train_indices.extend(int(v) for v in train.tolist())
        test_indices.extend(int(v) for v in test.tolist())
        train_offsets.append(len(train_indices))
        test_offsets.append(len(test_indices))

    param_offsets = [0]
    params: list[float] = []
    for name in operator_names:
        params.extend(_AOM_BENCH_OPERATOR_PARAMS[name])
        param_offsets.append(len(params))

    predictions = np.asarray(model.predict(X), dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    selected_operator_index = int(model.selected_operator_indices_[0])
    selected_n_components = int(model.n_components_)
    best_score = float(model.diagnostics_.extras["best_score"])
    return {
        "operator_names": operator_names,
        "operator_kinds": {
            "shape": [1, len(operator_names)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(_AOM_BENCH_OPERATOR_KIND_IDS[name]) for name in operator_names],
        },
        "operator_param_offsets": {
            "shape": [1, len(param_offsets)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in param_offsets],
        },
        "operator_params": {
            "shape": [1, len(params)],
            "layout": "row_major",
            "dtype": "f64",
            "values": [float(v) for v in params],
        },
        "fold_train_offsets": {
            "shape": [1, len(train_offsets)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in train_offsets],
        },
        "fold_train_indices": {
            "shape": [1, len(train_indices)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in train_indices],
        },
        "fold_test_offsets": {
            "shape": [1, len(test_offsets)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in test_offsets],
        },
        "fold_test_indices": {
            "shape": [1, len(test_indices)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in test_indices],
        },
        "operator_scores": {
            "shape": [1, len(operator_scores)],
            "layout": "row_major",
            "dtype": "f64",
            "values": operator_scores,
        },
        "rmse_curves": {
            "shape": [len(operator_names), int(max_components)],
            "layout": "row_major",
            "dtype": "f64",
            "values": np.asarray(curves, dtype=np.float64).reshape(-1, order="C").tolist(),
        },
        "predictions": {
            "shape": list(predictions.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(predictions),
        },
        "selected_operator_index": selected_operator_index,
        "selected_n_components": selected_n_components,
        "best_score": best_score,
    }


def _aom_global_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    operator_names: Sequence[str],
    max_components: int,
    cv: int,
    random_state: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "nirs4all.bench.AOM_v0.aompls",
            "version":          "AOM_v0",
            "git_revision_sha": "unknown",
            "params": {
                "engine":         "simpls_materialized",
                "selection":      "global",
                "criterion":      "cv",
                "operator_names":  [str(name) for name in operator_names],
                "max_components": int(max_components),
                "cv":             int(cv),
                "random_state":   int(random_state),
                "reference":      "nirs4all/bench/AOM_v0/aompls",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed + 1, "values": _flatten_rowmajor(Y)},
        },
        "expected": _aom_bench_reference_expected(X, Y, operator_names, max_components, cv, random_state),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "bench-AOM_v0-aom-selection",
        },
    }


def _aom_bench_operator_expected(
    X: np.ndarray,
    operator_names: Sequence[str],
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    operators = _aom_bench_operators(operator_names, X.shape[1])
    operator_names = [op.name for op in operators]
    outputs = []
    for op in operators:
        op.fit(X)
        outputs.append(np.asarray(op.transform(X), dtype=np.float64).reshape(-1, order="C"))

    param_offsets = [0]
    params: list[float] = []
    for name in operator_names:
        params.extend(_AOM_BENCH_OPERATOR_PARAMS[name])
        param_offsets.append(len(params))

    return {
        "operator_names": operator_names,
        "operator_kinds": {
            "shape": [1, len(operator_names)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(_AOM_BENCH_OPERATOR_KIND_IDS[name]) for name in operator_names],
        },
        "operator_param_offsets": {
            "shape": [1, len(param_offsets)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(v) for v in param_offsets],
        },
        "operator_params": {
            "shape": [1, len(params)],
            "layout": "row_major",
            "dtype": "f64",
            "values": [float(v) for v in params],
        },
        "operator_outputs": {
            "shape": [len(operator_names), int(X.size)],
            "layout": "row_major",
            "dtype": "f64",
            "values": np.asarray(outputs, dtype=np.float64).reshape(-1, order="C").tolist(),
        },
    }


def _aom_operator_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    operator_names: Sequence[str],
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "nirs4all.bench.AOM_v0.aompls.operators",
            "version":          "AOM_v0",
            "git_revision_sha": "unknown",
            "params": {
                "operator_names": [str(name) for name in operator_names],
                "reference":      "nirs4all/bench/AOM_v0/aompls",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
        },
        "expected": _aom_bench_operator_expected(X, operator_names),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "bench-AOM_v0-aom-operators",
        },
    }


def _metrics_fixture(
    fixture_id: str,
    seed: int,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> dict[str, Any]:
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._regression_metrics_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "reference":      "NumPy regression metric kernels",
            },
        },
        "data": {
            "Y_true": {"shape": list(Y_true.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y_true)},
            "Y_pred": {"shape": list(Y_pred.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y_pred)},
        },
        "expected": _regression_metrics_expected(Y_true, Y_pred),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-regression-metrics",
        },
    }


def _classification_metrics_fixture(
    fixture_id: str,
    seed: int,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._binary_classification_metrics_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "threshold":      float(threshold),
                "reference":      "NumPy binary classification metrics with average-rank AUC",
            },
        },
        "data": {
            "labels": {"shape": list(labels.shape), "layout": "row_major", "dtype": "i32", "rng_seed": seed, "values": labels.reshape(-1, order="C").astype(int).tolist()},
            "scores": {"shape": list(scores.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(scores)},
        },
        "expected": _binary_classification_metrics_expected(labels, scores, threshold),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-binary-classification-metrics",
        },
    }


def _classification_extensions_fixture(
    fixture_id: str,
    kind: str,
    seed: int,
    labels: np.ndarray,
    scores: np.ndarray,
    n_classes: int = 0,
    n_bins: int = 0,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if kind == "multiclass":
        expected = _multiclass_classification_metrics_expected(labels, scores, n_classes)
        reference = "NumPy multiclass macro/micro metrics with one-vs-rest AUC"
    elif kind == "calibration":
        expected = _binary_calibration_curve_expected(labels, scores, n_bins)
        reference = "NumPy binary fixed-bin calibration curve"
    else:
        raise ValueError(f"unsupported classification extension fixture kind: {kind}")
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._classification_extensions_fixture",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "kind":       kind,
                "n_classes":  int(n_classes),
                "n_bins":     int(n_bins),
                "reference":  reference,
            },
        },
        "data": {
            "labels": {"shape": list(labels.shape), "layout": "row_major", "dtype": "i32", "rng_seed": seed, "values": labels.reshape(-1, order="C").astype(int).tolist()},
            "scores": {"shape": list(scores.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(scores)},
        },
        "expected": expected,
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-classification-extensions",
        },
    }


def _pls_lda_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._pls_lda_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_lda",
                "solver":       "nipals",
                "n_classes":    int(n_classes),
                "n_components": int(n_components),
                "reference":    "Python sklearn PLSRegression scores with NumPy pooled-covariance LDA",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "labels": {"shape": list(labels.shape), "layout": "row_major", "dtype": "i32", "rng_seed": seed, "values": labels.reshape(-1, order="C").astype(int).tolist()},
        },
        "expected": _pls_lda_expected(X, labels, n_classes, n_components),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-scores-plus-numpy-LDA",
        },
    }


def _pls_logistic_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._pls_logistic_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_logistic",
                "solver":       "nipals",
                "n_classes":    int(n_classes),
                "n_components": int(n_components),
                "ridge":        1e-4,
                "reference":    "Python sklearn PLSRegression scores with NumPy baseline multinomial logistic Newton",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "labels": {"shape": list(labels.shape), "layout": "row_major", "dtype": "i32", "rng_seed": seed, "values": labels.reshape(-1, order="C").astype(int).tolist()},
        },
        "expected": _pls_logistic_expected(X, labels, n_classes, n_components),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-scores-plus-numpy-logistic",
        },
    }


def _mb_pls_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    block_sizes: Sequence[int],
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    sizes = [int(size) for size in block_sizes]
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._mb_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "mb_pls",
                "solver":       "nipals",
                "n_components": int(n_components),
                "block_sizes":  sizes,
                "reference":    "Python block-autoscaled weighted MB-PLS using sklearn PLSRegression(scale=False)",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _mb_pls_expected(X, Y, sizes, n_components),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-block-weighted-MBPLS",
        },
    }


def _lw_pls_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_neighbors: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._lw_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":     "lw_pls",
                "solver":        "nipals",
                "n_components":  int(n_components),
                "n_neighbors":   int(n_neighbors),
                "reference":     "Python global-autoscaled kNN local-window PLS using sklearn PLSRegression",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _lw_pls_expected(X, Y, n_components, n_neighbors),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-local-window-LWPLS",
        },
    }


def _variable_importance_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._variable_importance_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components": int(n_components),
                "reference":    "sklearn.cross_decomposition.PLSRegression VIP and selectivity-ratio formulas",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _variable_importance_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression/variable-importance",
        },
    }


def _variable_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._variable_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components": int(n_components),
                "top_k":        int(top_k),
                "reference":    "sklearn PLSRegression VIP, coefficient magnitude and selectivity-ratio rankers",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _variable_selection_expected(X, Y, n_components, top_k),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-variable-selection-rankers",
        },
    }


def _interval_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    step: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._interval_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":      "pls_regression",
                "solver":         "nipals",
                "n_components":   int(n_components),
                "n_splits":       int(n_splits),
                "interval_width": int(interval_width),
                "step":           int(step),
                "reference":      "sklearn PLSRegression deterministic k-fold interval scan",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _interval_selection_expected(X, Y, n_components, n_splits, interval_width, step),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-interval-selection-cv",
        },
    }


def _bipls_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    min_intervals: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._bipls_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":      "pls_regression",
                "solver":         "nipals",
                "n_components":   int(n_components),
                "n_splits":       int(n_splits),
                "interval_width": int(interval_width),
                "min_intervals":  int(min_intervals),
                "reference":      "sklearn PLSRegression deterministic backward interval PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _bipls_selection_expected(X,
                                              Y,
                                              n_components,
                                              n_splits,
                                              interval_width,
                                              min_intervals),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-biPLS-selection",
        },
    }


def _sipls_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    interval_width: int,
    combination_size: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._sipls_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":        "pls_regression",
                "solver":           "nipals",
                "n_components":     int(n_components),
                "n_splits":         int(n_splits),
                "interval_width":   int(interval_width),
                "combination_size": int(combination_size),
                "reference":        "sklearn PLSRegression deterministic synergy interval PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _sipls_selection_expected(X,
                                              Y,
                                              n_components,
                                              n_splits,
                                              interval_width,
                                              combination_size),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-siPLS-selection",
        },
    }


def _coefficient_stability_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._coefficient_stability_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_repeats":    int(n_repeats),
                "test_count":   int(test_count),
                "top_k":        int(top_k),
                "seed":         int(seed),
                "reference":    "sklearn PLSRegression Monte-Carlo coefficient stability",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _coefficient_stability_expected(X, Y, n_components, n_repeats, test_count, seed, top_k),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-coefficient-stability",
        },
    }


def _uve_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    validation_seed: int,
    noise_features: int,
    noise_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._uve_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":       "pls_regression",
                "solver":          "nipals",
                "n_components":    int(n_components),
                "n_repeats":       int(n_repeats),
                "test_count":      int(test_count),
                "validation_seed": int(validation_seed),
                "noise_features":  int(noise_features),
                "noise_seed":      int(noise_seed),
                "reference":       "sklearn PLSRegression UVE artificial-variable threshold",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _uve_expected(X,
                                  Y,
                                  n_components,
                                  n_repeats,
                                  test_count,
                                  validation_seed,
                                  noise_features,
                                  noise_seed),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-UVE",
        },
    }


def _emcuve_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_repeats: int,
    test_count: int,
    validation_seed: int,
    noise_features: int,
    noise_seed: int,
    n_ensembles: int,
    vote_threshold: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._emcuve_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":       "pls_regression",
                "solver":          "nipals",
                "n_components":    int(n_components),
                "n_repeats":       int(n_repeats),
                "test_count":      int(test_count),
                "validation_seed": int(validation_seed),
                "noise_features":  int(noise_features),
                "noise_seed":      int(noise_seed),
                "n_ensembles":     int(n_ensembles),
                "vote_threshold":  float(vote_threshold),
                "reference":       "sklearn PLSRegression ensemble MC-UVE vote rule",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _emcuve_expected(X,
                                     Y,
                                     n_components,
                                     n_repeats,
                                     test_count,
                                     validation_seed,
                                     noise_features,
                                     noise_seed,
                                     n_ensembles,
                                     vote_threshold),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-EMCUVE",
        },
    }


def _randomization_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_permutations: int,
    randomization_seed: int,
    alpha: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._randomization_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":          "pls_regression",
                "solver":             "nipals",
                "n_components":       int(n_components),
                "n_permutations":     int(n_permutations),
                "randomization_seed": int(randomization_seed),
                "alpha":              float(alpha),
                "reference":          "sklearn PLSRegression deterministic Y-permutation coefficient test",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _randomization_selection_expected(X,
                                                      Y,
                                                      n_components,
                                                      n_permutations,
                                                      randomization_seed,
                                                      alpha),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-randomization-selection",
        },
    }


def _spa_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._spa_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "top_k":        int(top_k),
                "reference":    "sklearn PLSRegression coefficient seed plus NumPy SPA projection",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _spa_selection_expected(X, Y, n_components, top_k),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-SPA-selection",
        },
    }


def _cars_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    min_features: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._cars_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_splits":     int(n_splits),
                "n_iterations": int(n_iterations),
                "min_features": int(min_features),
                "reference":    "sklearn PLSRegression deterministic CARS-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _cars_selection_expected(X,
                                             Y,
                                             n_components,
                                             n_splits,
                                             n_iterations,
                                             min_features),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-CARS-selection",
        },
    }


def _random_frog_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    initial_size: int,
    min_size: int,
    max_size: int,
    top_k: int,
    frog_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._random_frog_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_splits":     int(n_splits),
                "n_iterations": int(n_iterations),
                "initial_size": int(initial_size),
                "min_size":     int(min_size),
                "max_size":     int(max_size),
                "top_k":        int(top_k),
                "seed":         int(frog_seed),
                "reference":    "sklearn PLSRegression deterministic Random Frog PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _random_frog_selection_expected(X,
                                                    Y,
                                                    n_components,
                                                    n_splits,
                                                    n_iterations,
                                                    initial_size,
                                                    min_size,
                                                    max_size,
                                                    top_k,
                                                    frog_seed),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-RandomFrog-selection",
        },
    }


def _scars_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    min_features: int,
    sample_fraction: float,
    scars_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._scars_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":       "pls_regression",
                "solver":          "nipals",
                "n_components":    int(n_components),
                "n_splits":        int(n_splits),
                "n_iterations":    int(n_iterations),
                "min_features":    int(min_features),
                "sample_fraction": float(sample_fraction),
                "seed":            int(scars_seed),
                "reference":       "sklearn PLSRegression deterministic SCARS-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _scars_selection_expected(X,
                                              Y,
                                              n_components,
                                              n_splits,
                                              n_iterations,
                                              min_features,
                                              sample_fraction,
                                              scars_seed),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-SCARS-selection",
        },
    }


def _ga_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_generations: int,
    population_size: int,
    min_features: int,
    max_features: int,
    mutation_rate: float,
    ga_seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._ga_population_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":       "pls_regression",
                "solver":          "nipals",
                "n_components":    int(n_components),
                "n_splits":        int(n_splits),
                "n_generations":   int(n_generations),
                "population_size": int(population_size),
                "min_features":    int(min_features),
                "max_features":    int(max_features),
                "mutation_rate":   float(mutation_rate),
                "seed":            int(ga_seed),
                "reference":       "sklearn PLSRegression deterministic GA-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _ga_population_expected(X,
                                            Y,
                                            n_components,
                                            n_splits,
                                            n_generations,
                                            population_size,
                                            min_features,
                                            max_features,
                                            mutation_rate,
                                            ga_seed),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-GA-selection",
        },
    }


def _shaving_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
    shave_fraction: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._shaving_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":      "pls_regression",
                "solver":         "nipals",
                "n_components":   int(n_components),
                "n_splits":       int(n_splits),
                "n_steps":        int(n_steps),
                "min_features":   int(min_features),
                "shave_fraction": float(shave_fraction),
                "reference":      "sklearn PLSRegression deterministic shaving PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _shaving_selection_expected(X,
                                                Y,
                                                n_components,
                                                n_splits,
                                                n_steps,
                                                min_features,
                                                shave_fraction),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-Shaving-selection",
        },
    }


def _rep_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
    remove_count: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._rep_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_splits":     int(n_splits),
                "n_steps":      int(n_steps),
                "min_features": int(min_features),
                "remove_count": int(remove_count),
                "reference":    "sklearn PLSRegression deterministic REP-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _rep_selection_expected(X,
                                            Y,
                                            n_components,
                                            n_splits,
                                            n_steps,
                                            min_features,
                                            remove_count),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-REP-selection",
        },
    }


def _ipw_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_iterations: int,
    top_k: int,
    damping: float,
    weight_floor: float,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._ipw_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":     "pls_regression",
                "solver":        "nipals",
                "n_components":  int(n_components),
                "n_splits":      int(n_splits),
                "n_iterations":  int(n_iterations),
                "top_k":         int(top_k),
                "damping":       float(damping),
                "weight_floor":  float(weight_floor),
                "reference":     "sklearn PLSRegression deterministic IPW-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _ipw_selection_expected(X,
                                            Y,
                                            n_components,
                                            n_splits,
                                            n_iterations,
                                            top_k,
                                            damping,
                                            weight_floor),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-IPW-selection",
        },
    }


def _st_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    thresholds: Sequence[float],
    min_selected: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    threshold_values = [float(value) for value in thresholds]
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._st_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":     "pls_regression",
                "solver":        "nipals",
                "n_components":  int(n_components),
                "n_splits":      int(n_splits),
                "thresholds":    threshold_values,
                "min_selected":  int(min_selected),
                "reference":     "sklearn PLSRegression deterministic ST-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _st_selection_expected(X,
                                           Y,
                                           n_components,
                                           n_splits,
                                           threshold_values,
                                           min_selected),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-ST-selection",
        },
    }


def _bve_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_steps: int,
    min_features: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._bve_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_splits":     int(n_splits),
                "n_steps":      int(n_steps),
                "min_features": int(min_features),
                "reference":    "sklearn PLSRegression deterministic BVE-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _bve_selection_expected(X,
                                            Y,
                                            n_components,
                                            n_splits,
                                            n_steps,
                                            min_features),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-BVE-selection",
        },
    }


def _t2_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    alpha: Sequence[float],
    min_selected: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    alpha_values = [float(item) for item in alpha]
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._t2_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":     "pls_regression",
                "solver":        "nipals",
                "n_components":  int(n_components),
                "n_splits":      int(n_splits),
                "alpha":         alpha_values,
                "min_selected":  int(min_selected),
                "reference":     "sklearn PLSRegression plus plsVarSel-style Hotelling T2-PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _t2_selection_expected(X,
                                           Y,
                                           n_components,
                                           n_splits,
                                           alpha_values,
                                           min_selected),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression-T2-selection",
        },
    }


def _wvc_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    top_k: int,
    normalize: bool,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._wvc_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "wvc2_svd",
                "n_components": int(n_components),
                "top_k":        int(top_k),
                "normalize":    bool(normalize),
                "reference":    "NumPy mirror of plsVarSel WVC_pls numeric-regression WVC2",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _wvc_selection_expected(X,
                                            Y,
                                            n_components,
                                            top_k,
                                            normalize),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "numpy/WVC-PLS-selection",
        },
    }


def _wvc_threshold_selection_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    normalize: bool,
    score_threshold: float,
    threshold_factor: float,
    min_selected: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._wvc_threshold_selection_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":        "pls_regression",
                "solver":           "wvc2_svd",
                "n_components":     int(n_components),
                "normalize":        bool(normalize),
                "score_threshold":  float(score_threshold),
                "threshold_factor": float(threshold_factor),
                "min_selected":     int(min_selected),
                "reference":        "NumPy WVC-PLS threshold/factor selection rule",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _wvc_threshold_selection_expected(X,
                                                      Y,
                                                      n_components,
                                                      normalize,
                                                      score_threshold,
                                                      threshold_factor,
                                                      min_selected),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "numpy/WVC-PLS-threshold-selection",
        },
    }


def _component_coefficients_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._component_coefficients_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components": int(n_components),
                "reference":    "sklearn.cross_decomposition.PLSRegression prefix coefficient formulas",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _component_coefficients_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression/component-coefficients",
        },
    }


def _validation_fixture(
    fixture_id: str,
    kind: str,
    n_samples: int,
    n_splits: int = 0,
    holdout_start: int = 0,
    holdout_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._validation_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "kind":          kind,
                "n_samples":     int(n_samples),
                "n_splits":      int(n_splits),
                "holdout_start": int(holdout_start),
                "holdout_count": int(holdout_count),
                "reference":     "NumPy/Python deterministic validation splitters",
            },
        },
        "data": {
            "sample_indices": {
                "shape": [int(n_samples)],
                "layout": "row_major",
                "dtype": "i64",
                "values": list(range(int(n_samples))),
            },
        },
        "expected": _validation_expected(kind, n_samples, n_splits, holdout_start, holdout_count),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-python-validation-splits",
        },
    }


def _advanced_validation_fixture(
    fixture_id: str,
    kind: str,
    n_samples: int,
    n_splits: int = 0,
    n_repeats: int = 0,
    test_count: int = 0,
    train_count: int = 0,
    seed: int = 0,
    fold_ids: Sequence[int] | None = None,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sample_indices": {
            "shape": [int(n_samples)],
            "layout": "row_major",
            "dtype": "i64",
            "values": list(range(int(n_samples))),
        },
    }
    if fold_ids is not None:
        data["fold_ids"] = {
            "shape": [int(n_samples)],
            "layout": "row_major",
            "dtype": "i64",
            "values": [int(value) for value in fold_ids],
        }
    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        data["X"] = {
            "shape": list(X.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(X),
        }
    if Y is not None:
        Y = np.asarray(Y, dtype=np.float64)
        data["Y"] = {
            "shape": list(Y.shape),
            "layout": "row_major",
            "dtype": "f64",
            "values": _flatten_rowmajor(Y),
        }
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._advanced_validation_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "kind":        kind,
                "n_samples":   int(n_samples),
                "n_splits":    int(n_splits),
                "n_repeats":   int(n_repeats),
                "test_count":  int(test_count),
                "train_count": int(train_count),
                "seed":        int(seed),
                "reference":   "NumPy/Python deterministic advanced validation splitters",
            },
        },
        "data": data,
        "expected": _advanced_validation_expected(kind,
                                                  n_samples,
                                                  n_splits,
                                                  n_repeats,
                                                  test_count,
                                                  train_count,
                                                  seed,
                                                  fold_ids,
                                                  X,
                                                  Y),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-python-advanced-validation-splits",
        },
    }


def _cross_validation_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._cross_validation_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":    "pls_regression",
                "solver":       "nipals",
                "n_components": int(n_components),
                "n_splits":     int(n_splits),
                "reference":    "sklearn.cross_decomposition.PLSRegression with deterministic k-fold splits",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _cross_validation_expected(X, Y, n_components, n_splits),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "sklearn/PLSRegression/kfold-cv",
        },
    }


def _component_cv_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components_max: int,
    n_splits: int,
    solver: str,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._component_cv_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "algorithm":        "pls_regression",
                "solver":           solver,
                "n_components_max": int(n_components_max),
                "n_splits":         int(n_splits),
                "reference":        "NumPy SIMPLS deterministic k-fold component-count CV",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _component_cv_expected(X, Y, n_components_max, n_splits, solver),
        "comparison_policy": {
            "components_alignment": "exact",
            "sign_resolver":        "none",
            "tolerance_table_row":  "pls4all-numpy-simpls-component-cv",
        },
    }


def _power_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._power_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "power",
                "reference":      "NumPy singular-pair power iteration with regression deflation",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _power_pls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-power",
        },
    }


def _randomized_svd_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    random_seed: int = 123456789,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._randomized_svd_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "randomized-svd",
                "random_seed":    random_seed,
                "reference":      "NumPy mirror of SplitMix64-seeded randomized SVD PLS",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _randomized_svd_pls_expected(X, Y, n_components, random_seed),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-randomized-svd",
        },
    }


def _canonical_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    solver: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._canonical_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "canonical",
                "algorithm":      solver,
                "reference":      f"NumPy mirror of sklearn PLSCanonical({solver})",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _canonical_pls_expected(X, Y, n_components, solver),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  f"sklearn/PLSCanonical/{solver}",
        },
    }


def _pls_da_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    model = PLSRegression(n_components=n_components, scale=True)
    model.fit(X, Y)
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._expected_from_pls_da",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "pls-da-dummy",
                "reference":      "sklearn PLSRegression on dummy-coded class targets",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _expected_from_pls_da(model, X, Y),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "sklearn/PLSDA-dummy/PLSRegression",
        },
    }


def _kernel_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._kernel_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "kernelpls",
                "reference":      "NumPy linear kernel PLS with regression deflation",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _kernel_pls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-kernelpls",
        },
    }


def _wide_kernel_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._kernel_pls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "wide-kernelpls",
                "reference":      "NumPy linear wide-kernel PLS with regression deflation",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _kernel_pls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-wide-kernelpls",
        },
    }


def _orthogonal_scores_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._orthogonal_scores_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "regression",
                "algorithm":      "oscorespls",
                "reference":      "NumPy port of R pls oscorespls.fit recurrence",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _orthogonal_scores_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls-r-oscorespls",
        },
    }


def _opls_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._opls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "orthogonal",
                "algorithm":      "opls-nipals",
                "reference":      "NumPy OPLS1 NIPALS recurrence",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _opls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-opls",
        },
    }


def _opls_da_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    algorithm: str = "opls-da-binary",
    reference: str = "NumPy OPLS1 NIPALS recurrence on one-column dummy Y",
    tolerance_table_row: str = "pls4all-numpy-opls-da-binary",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._opls_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "orthogonal",
                "algorithm":      algorithm,
                "reference":      reference,
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _opls_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  tolerance_table_row,
        },
    }


def _pcr_fixture(
    fixture_id: str,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id":     fixture_id,
        "generator": {
            "name":             "pls4all_parity.suites._pcr_expected",
            "version":          "1",
            "git_revision_sha": "unknown",
            "params": {
                "n_components":   n_components,
                "scale":          True,
                "deflation_mode": "x-only",
                "algorithm":      "pcr",
                "reference":      "NumPy PCA/SVD with score-space least squares",
            },
        },
        "data": {
            "X": {"shape": list(X.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(X)},
            "Y": {"shape": list(Y.shape), "layout": "row_major", "dtype": "f64", "rng_seed": seed, "values": _flatten_rowmajor(Y)},
        },
        "expected": _pcr_expected(X, Y, n_components),
        "comparison_policy": {
            "components_alignment": "first-k-prefix",
            "sign_resolver":        "max_abs_element_positive",
            "tolerance_table_row":  "pls4all-numpy-pcr",
        },
    }


def synthetic_small_pls1_v1() -> dict[str, Any]:
    """50 samples, 20 features, 1 target, n_components=3."""
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(50, 20))
    true_w = rng.standard_normal(size=20) * 0.5
    Y = (X @ true_w).reshape(-1, 1) + rng.standard_normal(size=(50, 1)) * 0.1
    return _fixture("synthetic_small_pls1_v1", seed=0,
                    X=X, Y=Y, n_components=3,
                    tolerance_table_row="sklearn/PLSRegression/self_roundtrip")


def synthetic_small_pls2_v1() -> dict[str, Any]:
    """50 samples, 20 features, 4 targets, n_components=3."""
    rng = np.random.default_rng(seed=1)
    X = rng.standard_normal(size=(50, 20))
    W = rng.standard_normal(size=(20, 4)) * 0.5
    Y = X @ W + rng.standard_normal(size=(50, 4)) * 0.1
    return _fixture("synthetic_small_pls2_v1", seed=1,
                    X=X, Y=Y, n_components=3,
                    tolerance_table_row="sklearn/PLSRegression/self_roundtrip")


def synthetic_tiny_centered_v1() -> dict[str, Any]:
    """10 samples, 3 features, 1 target, n_components=2 — numerical edge case."""
    rng = np.random.default_rng(seed=2)
    X = rng.standard_normal(size=(10, 3))
    X = X - X.mean(axis=0)  # explicitly centred
    Y = (X[:, 0] + 0.3 * X[:, 1] - 0.5 * X[:, 2]).reshape(-1, 1)
    return _fixture("synthetic_tiny_centered_v1", seed=2,
                    X=X, Y=Y, n_components=2,
                    tolerance_table_row="sklearn/PLSRegression/self_roundtrip")


def synthetic_simpls_tiny_pls1_v1() -> dict[str, Any]:
    """8 samples, 4 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=20)
    X = rng.standard_normal(size=(8, 4))
    true_w = np.array([0.7, -0.25, 0.4, -0.1])
    Y = (X @ true_w + rng.standard_normal(size=8) * 0.03).reshape(-1, 1)
    return _simpls_fixture("synthetic_simpls_tiny_pls1_v1", seed=20,
                           X=X, Y=Y, n_components=2)


def synthetic_simpls_small_pls2_v1() -> dict[str, Any]:
    """12 samples, 5 features, 2 targets, n_components=2."""
    rng = np.random.default_rng(seed=21)
    X = rng.standard_normal(size=(12, 5))
    W = np.array([
        [0.50, -0.10],
        [-0.35, 0.45],
        [0.20, 0.15],
        [0.05, -0.40],
        [0.30, 0.25],
    ])
    Y = X @ W + rng.standard_normal(size=(12, 2)) * 0.04
    return _simpls_fixture("synthetic_simpls_small_pls2_v1", seed=21,
                           X=X, Y=Y, n_components=2)


def synthetic_svd_tiny_pls1_v1() -> dict[str, Any]:
    """9 samples, 4 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=30)
    X = rng.standard_normal(size=(9, 4))
    true_w = np.array([0.45, -0.55, 0.20, 0.35])
    Y = (X @ true_w + rng.standard_normal(size=9) * 0.025).reshape(-1, 1)
    return _svd_fixture("synthetic_svd_tiny_pls1_v1", seed=30,
                        X=X, Y=Y, n_components=2)


def synthetic_svd_small_pls2_v1() -> dict[str, Any]:
    """14 samples, 6 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=31)
    X = rng.standard_normal(size=(14, 6))
    W = np.array([
        [0.30, -0.15, 0.20],
        [-0.40, 0.35, -0.10],
        [0.15, 0.25, 0.45],
        [0.05, -0.30, 0.10],
        [0.50, 0.10, -0.35],
        [-0.20, 0.40, 0.25],
    ])
    Y = X @ W + rng.standard_normal(size=(14, 3)) * 0.035
    return _svd_fixture("synthetic_svd_small_pls2_v1", seed=31,
                        X=X, Y=Y, n_components=3)


def synthetic_pls_svd_tiny_v1() -> dict[str, Any]:
    """11 samples, 5 features, 2 targets, n_components=2 for direct PLSSVD."""
    rng = np.random.default_rng(seed=32)
    X = rng.standard_normal(size=(11, 5))
    W = np.array([
        [0.42, -0.20],
        [-0.35, 0.48],
        [0.28, 0.18],
        [0.10, -0.36],
        [0.31, 0.24],
    ])
    Y = X @ W + rng.standard_normal(size=(11, 2)) * 0.03
    return _pls_svd_fixture("synthetic_pls_svd_tiny_v1", seed=32,
                            X=X, Y=Y, n_components=2)


def synthetic_pls_svd_small_v1() -> dict[str, Any]:
    """15 samples, 7 features, 3 targets, n_components=3 for direct PLSSVD."""
    rng = np.random.default_rng(seed=33)
    X = rng.standard_normal(size=(15, 7))
    W = np.array([
        [0.30, -0.15, 0.20],
        [-0.40, 0.35, -0.10],
        [0.15, 0.25, 0.45],
        [0.05, -0.30, 0.10],
        [0.50, 0.10, -0.35],
        [-0.20, 0.40, 0.25],
        [0.18, 0.22, -0.28],
    ])
    Y = X @ W + rng.standard_normal(size=(15, 3)) * 0.035
    return _pls_svd_fixture("synthetic_pls_svd_small_v1", seed=33,
                            X=X, Y=Y, n_components=3)


def synthetic_pipeline_identity_v1() -> dict[str, Any]:
    """4 samples, 3 features for identity preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 8.0],
    ], dtype=np.float64)
    return _pipeline_fixture("synthetic_pipeline_identity_v1", seed=40,
                             X=X, operators=["identity"])


def synthetic_pipeline_center_v1() -> dict[str, Any]:
    """4 samples, 3 features for column centering preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 8.0],
    ], dtype=np.float64)
    return _pipeline_fixture("synthetic_pipeline_center_v1", seed=41,
                             X=X, operators=["center"])


def synthetic_pipeline_autoscale_v1() -> dict[str, Any]:
    """4 samples, 3 features for autoscale preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 8.0],
    ], dtype=np.float64)
    return _pipeline_fixture("synthetic_pipeline_autoscale_v1", seed=42,
                             X=X, operators=["autoscale"])


def synthetic_pipeline_pareto_v1() -> dict[str, Any]:
    """4 samples, 3 features for Pareto scaling preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 8.0],
    ], dtype=np.float64)
    return _pipeline_fixture("synthetic_pipeline_pareto_v1", seed=43,
                             X=X, operators=["pareto"])


def synthetic_pipeline_snv_v1() -> dict[str, Any]:
    """3 samples, 3 features for row-wise SNV preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0],
        [3.0, 5.0, 7.0],
    ], dtype=np.float64)
    return _pipeline_fixture("synthetic_pipeline_snv_v1", seed=44,
                             X=X, operators=["snv"])


def synthetic_aom_soft_preprocessing_v1() -> dict[str, Any]:
    """5 samples, 4 features for equal-weight soft AOM preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0, 5.0],
        [2.0, 3.5, 5.5, 8.0],
        [3.0, 5.0, 7.0, 10.0],
        [4.0, 6.5, 8.5, 11.5],
        [5.0, 8.0, 11.0, 14.0],
    ], dtype=np.float64)
    return _aom_preprocessing_fixture("synthetic_aom_soft_preprocessing_v1",
                                      seed=47,
                                      X=X,
                                      operators=["identity", "autoscale", "snv"],
                                      gating_mode="soft")


def synthetic_aom_hard_preprocessing_v1() -> dict[str, Any]:
    """5 samples, 4 features for hard first-operator AOM preprocessing parity."""
    X = np.array([
        [1.0, 2.0, 3.0, 5.0],
        [2.0, 3.5, 5.5, 8.0],
        [3.0, 5.0, 7.0, 10.0],
        [4.0, 6.5, 8.5, 11.5],
        [5.0, 8.0, 11.0, 14.0],
    ], dtype=np.float64)
    return _aom_preprocessing_fixture("synthetic_aom_hard_preprocessing_v1",
                                      seed=48,
                                      X=X,
                                      operators=["center", "identity", "snv"],
                                      gating_mode="hard")


def synthetic_aom_global_simpls_cv_v1() -> dict[str, Any]:
    """9 samples, 6 features for bench AOM_v0 global SIMPLS CV parity."""
    X = np.array([
        [1.0, 2.0, 3.0, 4.5, 5.5, 7.0],
        [1.2, 2.1, 3.4, 4.8, 5.8, 7.5],
        [0.9, 1.8, 2.8, 4.2, 5.2, 6.6],
        [1.5, 2.5, 3.8, 5.0, 6.1, 7.8],
        [1.8, 2.8, 4.1, 5.4, 6.4, 8.1],
        [2.1, 3.0, 4.5, 5.9, 6.9, 8.6],
        [2.3, 3.4, 4.9, 6.2, 7.3, 9.0],
        [2.6, 3.7, 5.2, 6.6, 7.8, 9.4],
        [2.9, 4.0, 5.6, 7.1, 8.2, 9.9],
    ], dtype=np.float64)
    Y = np.array([1.1, 1.4, 0.9, 1.8, 2.1, 2.5, 2.8, 3.2, 3.6], dtype=np.float64)
    return _aom_global_selection_fixture("synthetic_aom_global_simpls_cv_v1",
                                         seed=49,
                                         X=X,
                                         Y=Y,
                                         operator_names=["identity", "detrend_d1"],
                                         max_components=3,
                                         cv=3,
                                         random_state=7)


def synthetic_aom_strict_operators_v1() -> dict[str, Any]:
    """4 samples, 7 features for bench AOM_v0 strict-linear operator parity."""
    X = np.array([
        [0.20, 0.55, 1.10, 1.75, 2.35, 2.70, 3.10],
        [1.00, 0.85, 0.65, 0.72, 1.05, 1.62, 2.40],
        [2.50, 2.20, 1.80, 1.42, 1.15, 1.02, 0.90],
        [0.10, 0.40, 0.95, 1.35, 1.80, 2.55, 3.45],
    ], dtype=np.float64)
    return _aom_operator_fixture(
        "synthetic_aom_strict_operators_v1",
        seed=50,
        X=X,
        operator_names=[
            "identity",
            "detrend_d1",
            "sg_smooth_w5_p2",
            "sg_d1_w5_p2",
            "fd_d1",
            "nw_g3_s1_d1",
        ],
    )


def synthetic_aom_global_simpls_sg_cv_v1() -> dict[str, Any]:
    """10 samples, 7 features for bench AOM_v0 global SIMPLS CV with SG operators."""
    X = np.array([
        [0.20, 0.55, 1.10, 1.75, 2.35, 2.70, 3.10],
        [0.30, 0.70, 1.20, 1.85, 2.55, 2.95, 3.45],
        [0.18, 0.48, 0.95, 1.55, 2.10, 2.52, 2.95],
        [0.45, 0.88, 1.42, 2.05, 2.72, 3.18, 3.70],
        [0.62, 1.05, 1.65, 2.32, 3.00, 3.45, 4.02],
        [0.78, 1.24, 1.88, 2.58, 3.32, 3.82, 4.38],
        [0.92, 1.45, 2.12, 2.86, 3.62, 4.18, 4.78],
        [1.08, 1.68, 2.38, 3.18, 3.98, 4.58, 5.22],
        [1.22, 1.86, 2.66, 3.52, 4.34, 5.02, 5.70],
        [1.38, 2.08, 2.95, 3.88, 4.76, 5.48, 6.20],
    ], dtype=np.float64)
    Y = np.array([0.45, 0.62, 0.38, 0.78, 0.96, 1.18, 1.42, 1.70, 1.98, 2.30], dtype=np.float64)
    return _aom_global_selection_fixture("synthetic_aom_global_simpls_sg_cv_v1",
                                         seed=51,
                                         X=X,
                                         Y=Y,
                                         operator_names=[
                                             "identity",
                                             "sg_smooth_w5_p2",
                                             "sg_d1_w5_p2",
                                             "fd_d1",
                                             "nw_g3_s1_d1",
                                             "detrend_d1",
                                         ],
                                         max_components=3,
                                         cv=5,
                                         random_state=11)


def synthetic_aom_extended_strict_operators_v1() -> dict[str, Any]:
    """5 samples, 9 features for bench AOM_v0 Whittaker and FCK operator parity."""
    X = np.array([
        [0.10, 0.32, 0.70, 1.18, 1.72, 2.18, 2.52, 2.82, 3.08],
        [1.40, 1.18, 0.96, 0.88, 1.02, 1.36, 1.82, 2.24, 2.60],
        [2.80, 2.42, 2.06, 1.74, 1.48, 1.30, 1.18, 1.06, 0.94],
        [0.18, 0.50, 0.92, 1.38, 1.86, 2.44, 3.10, 3.68, 4.18],
        [1.02, 1.34, 1.82, 2.24, 2.52, 2.72, 2.86, 2.96, 3.02],
    ], dtype=np.float64)
    return _aom_operator_fixture(
        "synthetic_aom_extended_strict_operators_v1",
        seed=52,
        X=X,
        operator_names=[
            "whittaker_l10",
            "fck_a1.00_s1.00_k7",
        ],
    )


def synthetic_aom_global_simpls_fck_whittaker_cv_v1() -> dict[str, Any]:
    """11 samples, 9 features for bench AOM_v0 global SIMPLS CV with Whittaker/FCK."""
    X = np.array([
        [0.10, 0.32, 0.70, 1.18, 1.72, 2.18, 2.52, 2.82, 3.08],
        [0.22, 0.48, 0.86, 1.34, 1.92, 2.42, 2.82, 3.14, 3.45],
        [0.18, 0.38, 0.76, 1.22, 1.78, 2.26, 2.58, 2.88, 3.16],
        [0.38, 0.72, 1.18, 1.72, 2.34, 2.90, 3.32, 3.70, 4.08],
        [0.52, 0.94, 1.48, 2.06, 2.72, 3.28, 3.78, 4.18, 4.62],
        [0.70, 1.18, 1.78, 2.42, 3.10, 3.74, 4.28, 4.76, 5.20],
        [0.88, 1.44, 2.10, 2.82, 3.56, 4.24, 4.84, 5.36, 5.86],
        [1.08, 1.72, 2.44, 3.22, 4.02, 4.78, 5.46, 6.04, 6.60],
        [1.30, 2.02, 2.82, 3.66, 4.54, 5.34, 6.10, 6.76, 7.38],
        [1.54, 2.34, 3.22, 4.14, 5.10, 6.00, 6.82, 7.56, 8.22],
        [1.82, 2.70, 3.66, 4.66, 5.70, 6.68, 7.58, 8.40, 9.16],
    ], dtype=np.float64)
    Y = np.array([0.30, 0.48, 0.36, 0.72, 0.94, 1.22, 1.52, 1.84, 2.20, 2.58, 3.02], dtype=np.float64)
    return _aom_global_selection_fixture("synthetic_aom_global_simpls_fck_whittaker_cv_v1",
                                         seed=53,
                                         X=X,
                                         Y=Y,
                                         operator_names=[
                                             "identity",
                                             "whittaker_l10",
                                             "fck_a1.00_s1.00_k7",
                                             "detrend_d1",
                                         ],
                                         max_components=3,
                                         cv=3,
                                         random_state=13)


def synthetic_pipeline_msc_v1() -> dict[str, Any]:
    """4 samples, 5 wavelengths for multiplicative scatter correction parity."""
    reference = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float64)
    X = np.vstack([
        0.25 + 1.10 * reference,
        -0.35 + 0.82 * reference,
        0.90 + 1.35 * reference,
        0.45 + 0.62 * reference,
    ])
    return _pipeline_fixture("synthetic_pipeline_msc_v1", seed=45,
                             X=X, operators=["msc"])


def synthetic_pipeline_emsc_v1() -> dict[str, Any]:
    """4 samples, 8 wavelengths for extended multiplicative scatter correction."""
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    reference = np.array([1.0, 2.3, 3.8, 5.2, 7.4, 7.9, 10.8, 12.7], dtype=np.float64)
    residual = np.array([0.01, -0.02, 0.00, 0.03, -0.01, 0.02, -0.03, 0.01])
    X = np.vstack([
        0.30 + 1.10 * reference + 0.15 * x - 0.08 * x * x + residual,
        -0.45 + 0.82 * reference - 0.05 * x + 0.12 * x * x - 0.5 * residual,
        0.90 + 1.35 * reference + 0.08 * x - 0.03 * x * x + 0.25 * residual,
        0.15 + 0.68 * reference - 0.10 * x + 0.05 * x * x - residual,
    ])
    return _pipeline_fixture("synthetic_pipeline_emsc_v1", seed=49,
                             X=X, operators=["emsc_2"])


def synthetic_pipeline_detrend_poly_v1() -> dict[str, Any]:
    """3 samples, 6 wavelengths for degree-2 row-wise polynomial detrending."""
    x = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    X = np.vstack([
        1.0 + 0.3 * x + 0.8 * x * x + np.array([0.02, -0.04, 0.03, -0.02, 0.01, 0.00]),
        -0.4 + 0.7 * x - 0.2 * x * x + np.array([-0.01, 0.03, -0.02, 0.04, -0.03, 0.01]),
        0.2 - 0.5 * x + 0.4 * x * x + np.array([0.00, 0.02, -0.03, 0.01, -0.01, 0.02]),
    ])
    return _pipeline_fixture("synthetic_pipeline_detrend_poly_v1", seed=46,
                             X=X, operators=["detrend_poly_2"])


def synthetic_pipeline_savgol_smooth_v1() -> dict[str, Any]:
    """3 samples, 9 wavelengths for Savitzky-Golay smoothing parity."""
    x = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    X = np.vstack([
        0.3 + 0.2 * x + 1.4 * x * x + np.array([0.05, -0.04, 0.02, -0.03, 0.01, 0.02, -0.01, 0.03, -0.02]),
        np.sin(2.0 * x) + np.array([-0.02, 0.03, -0.01, 0.04, -0.02, 0.00, 0.02, -0.03, 0.01]),
        1.1 - 0.5 * x + 0.3 * x * x + np.array([0.00, 0.02, -0.04, 0.01, 0.03, -0.02, 0.01, -0.01, 0.02]),
    ])
    return _pipeline_fixture("synthetic_pipeline_savgol_smooth_v1", seed=47,
                             X=X, operators=["savgol_smooth_5_2"])


def synthetic_pipeline_savgol_derivative_v1() -> dict[str, Any]:
    """3 samples, 9 wavelengths for first-derivative Savitzky-Golay parity."""
    x = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    X = np.vstack([
        0.4 + 1.5 * x - 0.2 * x * x + 0.6 * x * x * x,
        np.cos(2.5 * x) + 0.2 * x,
        -0.7 + 0.4 * x + 0.8 * x * x,
    ])
    return _pipeline_fixture("synthetic_pipeline_savgol_derivative_v1", seed=48,
                             X=X, operators=["savgol_derivative_5_2_1_1.0"])


def synthetic_pipeline_asls_v1() -> dict[str, Any]:
    """3 samples, 10 wavelengths for asymmetric least-squares baseline parity."""
    x = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    baseline = 0.4 + 0.8 * x + 0.25 * x * x
    X = np.vstack([
        baseline + np.array([0.00, 0.04, 0.18, 0.50, 0.22, 0.05, 0.02, 0.00, 0.03, 0.01]),
        0.7 * baseline + np.array([0.02, 0.03, 0.12, 0.24, 0.62, 0.31, 0.07, 0.02, 0.00, 0.01]),
        1.2 * baseline + np.array([0.01, 0.00, 0.05, 0.10, 0.20, 0.45, 0.28, 0.08, 0.03, 0.00]),
    ])
    return _pipeline_fixture("synthetic_pipeline_asls_v1", seed=50,
                             X=X, operators=["asls_100.0_0.01_8"])


def synthetic_pipeline_norris_williams_v1() -> dict[str, Any]:
    """3 samples, 11 wavelengths for Norris-Williams gap-segment parity."""
    x = np.linspace(-1.0, 1.0, 11, dtype=np.float64)
    X = np.vstack([
        0.2 + 1.1 * x + 0.5 * x * x + np.sin(3.0 * x) * 0.1,
        -0.4 + 0.7 * x - 0.2 * x * x + np.cos(2.0 * x) * 0.15,
        1.0 - 0.5 * x + 0.3 * x * x + np.array([0.00, 0.02, -0.01, 0.03, -0.02, 0.00, 0.01, -0.03, 0.02, -0.01, 0.00]),
    ])
    return _pipeline_fixture("synthetic_pipeline_norris_williams_v1", seed=51,
                             X=X, operators=["norris_williams_5_3_1"])


def synthetic_pipeline_wavelet_haar_v1() -> dict[str, Any]:
    """3 samples, 9 wavelengths for Haar wavelet denoising parity."""
    x = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    X = np.vstack([
        0.5 + 0.8 * x + np.array([0.02, -0.03, 0.04, -0.05, 0.03, -0.02, 0.01, -0.01, 0.02]),
        np.sin(2.5 * x) + np.array([-0.01, 0.02, -0.04, 0.03, -0.02, 0.04, -0.03, 0.01, 0.00]),
        1.0 - 0.4 * x + 0.2 * x * x + np.array([0.00, 0.03, -0.02, 0.01, -0.04, 0.02, -0.01, 0.03, -0.02]),
    ])
    return _pipeline_fixture("synthetic_pipeline_wavelet_haar_v1", seed=52,
                             X=X, operators=["wavelet_haar_2_0.03"])


def synthetic_pipeline_osc_v1() -> dict[str, Any]:
    """6 samples, 5 wavelengths for supervised one-component OSC parity."""
    y = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
    orthogonal_score = np.array([0.9, -0.7, 0.4, -0.3, 0.6, -0.9], dtype=np.float64)
    y_loading = np.array([0.40, -0.25, 0.55, 0.10, -0.35], dtype=np.float64)
    orthogonal_loading = np.array([1.00, -0.45, 0.20, 0.70, -0.30], dtype=np.float64)
    residual = np.array([
        [0.02, -0.01, 0.00, 0.01, -0.02],
        [-0.01, 0.03, -0.02, 0.00, 0.01],
        [0.00, -0.02, 0.02, -0.01, 0.03],
        [0.01, 0.00, -0.03, 0.02, -0.01],
        [-0.02, 0.01, 0.01, -0.03, 0.00],
        [0.03, -0.01, 0.00, 0.01, -0.02],
    ], dtype=np.float64)
    X = y @ y_loading.reshape(1, -1) + orthogonal_score.reshape(-1, 1) @ orthogonal_loading.reshape(1, -1) + residual
    return _pipeline_fixture("synthetic_pipeline_osc_v1", seed=53,
                             X=X, Y=y, operators=["osc_100_1e-12"])


def synthetic_pipeline_epo_v1() -> dict[str, Any]:
    """6 samples, 5 wavelengths for one-component EPO parity."""
    external = np.array([-1.0, -0.6, -0.1, 0.2, 0.7, 1.1], dtype=np.float64).reshape(-1, 1)
    analyte = np.linspace(-0.8, 0.9, 6, dtype=np.float64)
    analyte_loading = np.array([0.30, 0.55, -0.20, 0.15, 0.45], dtype=np.float64)
    external_loading = np.array([0.95, -0.35, 0.25, 0.75, -0.40], dtype=np.float64)
    residual = np.array([
        [0.01, -0.02, 0.00, 0.02, -0.01],
        [0.00, 0.01, -0.03, 0.01, 0.02],
        [-0.02, 0.00, 0.02, -0.01, 0.01],
        [0.03, -0.01, 0.01, 0.00, -0.02],
        [-0.01, 0.02, -0.01, 0.03, 0.00],
        [0.02, -0.03, 0.00, -0.02, 0.01],
    ], dtype=np.float64)
    X = (
        analyte.reshape(-1, 1) @ analyte_loading.reshape(1, -1)
        + external @ external_loading.reshape(1, -1)
        + residual
    )
    return _pipeline_fixture("synthetic_pipeline_epo_v1", seed=54,
                             X=X, Y=external, operators=["epo_100_1e-12"])


def synthetic_metrics_regression_v1() -> dict[str, Any]:
    """8 samples, 2 targets for regression metric parity."""
    Y_true = np.array([
        [0.20, -1.10],
        [1.35, -0.45],
        [2.10, 0.05],
        [2.95, 0.62],
        [3.40, 1.20],
        [4.15, 1.95],
        [5.05, 2.35],
        [5.80, 3.10],
    ], dtype=np.float64)
    residual = np.array([
        [0.05, -0.03],
        [-0.10, 0.08],
        [0.12, -0.07],
        [-0.04, 0.10],
        [0.09, -0.11],
        [-0.13, 0.06],
        [0.07, -0.05],
        [-0.08, 0.09],
    ], dtype=np.float64)
    return _metrics_fixture("synthetic_metrics_regression_v1", seed=55,
                            Y_true=Y_true, Y_pred=Y_true + residual)


def synthetic_classification_binary_metrics_v1() -> dict[str, Any]:
    """14 binary labels and continuous scores for classification metric parity."""
    labels = np.array([
        0, 1, 0, 1, 1, 0, 0,
        1, 0, 1, 1, 0, 1, 0,
    ], dtype=np.int32).reshape(-1, 1)
    scores = np.array([
        0.05, 0.81, 0.34, 0.62, 0.49, 0.57, 0.24,
        0.91, 0.45, 0.72, 0.57, 0.12, 0.66, 0.39,
    ], dtype=np.float64).reshape(-1, 1)
    return _classification_metrics_fixture("synthetic_classification_binary_metrics_v1",
                                           seed=58,
                                           labels=labels,
                                           scores=scores,
                                           threshold=0.5)


def synthetic_classification_multiclass_metrics_v1() -> dict[str, Any]:
    """15 labels and 3 score columns for multiclass macro/micro metric parity."""
    labels = np.array([
        0, 1, 2, 1, 0,
        2, 2, 0, 1, 0,
        2, 1, 0, 1, 2,
    ], dtype=np.int32).reshape(-1, 1)
    scores = np.array([
        [0.82, 0.11, 0.07],
        [0.21, 0.62, 0.17],
        [0.16, 0.31, 0.53],
        [0.34, 0.45, 0.21],
        [0.47, 0.42, 0.11],
        [0.12, 0.39, 0.49],
        [0.28, 0.26, 0.46],
        [0.51, 0.18, 0.31],
        [0.22, 0.36, 0.42],
        [0.39, 0.44, 0.17],
        [0.17, 0.23, 0.60],
        [0.27, 0.54, 0.19],
        [0.58, 0.24, 0.18],
        [0.30, 0.48, 0.22],
        [0.24, 0.30, 0.46],
    ], dtype=np.float64)
    return _classification_extensions_fixture("synthetic_classification_multiclass_metrics_v1",
                                              kind="multiclass",
                                              seed=95,
                                              labels=labels,
                                              scores=scores,
                                              n_classes=3)


def synthetic_classification_calibration_curve_v1() -> dict[str, Any]:
    """16 binary labels and calibrated probabilities for fixed-bin calibration parity."""
    labels = np.array([
        0, 0, 1, 0, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 0,
    ], dtype=np.int32).reshape(-1, 1)
    scores = np.array([
        0.03, 0.18, 0.24, 0.31, 0.37, 0.42, 0.46, 0.51,
        0.58, 0.63, 0.68, 0.74, 0.79, 0.83, 0.91, 1.00,
    ], dtype=np.float64).reshape(-1, 1)
    return _classification_extensions_fixture("synthetic_classification_calibration_curve_v1",
                                              kind="calibration",
                                              seed=96,
                                              labels=labels,
                                              scores=scores,
                                              n_bins=5)


def synthetic_pls_lda_multiclass_v1() -> dict[str, Any]:
    """18 samples, 5 features and 3 classes for PLS-score LDA parity."""
    seed = 98
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3, dtype=np.int32), 6)
    class_centers = np.array([
        [1.2, -0.4],
        [-0.7, 1.1],
        [-0.6, -0.9],
    ], dtype=np.float64)
    latent = np.vstack([
        class_centers[int(label)] + rng.standard_normal(size=2) * 0.18
        for label in labels
    ])
    X = np.column_stack([
        0.82 * latent[:, 0] + 0.18 * latent[:, 1],
        -0.31 * latent[:, 0] + 0.67 * latent[:, 1],
        0.44 * latent[:, 0] - 0.29 * latent[:, 1],
        np.sin(np.linspace(0.0, 1.5 * np.pi, labels.size)),
        rng.standard_normal(size=labels.size) * 0.04,
    ])
    return _pls_lda_fixture("synthetic_pls_lda_multiclass_v1",
                            seed=seed,
                            X=X,
                            labels=labels,
                            n_classes=3,
                            n_components=2)


def synthetic_pls_logistic_multiclass_v1() -> dict[str, Any]:
    """24 samples, 6 features and 3 classes for PLS-score logistic parity."""
    seed = 99
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3, dtype=np.int32), 8)
    class_centers = 0.6 * np.array([
        [0.85, -0.35],
        [-0.55, 0.72],
        [-0.32, -0.64],
    ], dtype=np.float64)
    latent = np.vstack([
        class_centers[int(label)] + rng.standard_normal(size=2) * 0.42
        for label in labels
    ])
    trend = np.linspace(-0.45, 0.55, labels.size)
    X = np.column_stack([
        0.74 * latent[:, 0] + 0.22 * latent[:, 1] + 0.06 * trend,
        -0.24 * latent[:, 0] + 0.61 * latent[:, 1] - 0.04 * trend,
        0.48 * latent[:, 0] - 0.21 * latent[:, 1],
        -0.18 * latent[:, 0] - 0.37 * latent[:, 1] + 0.03 * trend,
        np.sin(np.linspace(0.0, 1.25 * np.pi, labels.size)) * 0.28,
        rng.standard_normal(size=labels.size) * 0.09,
    ])
    return _pls_logistic_fixture("synthetic_pls_logistic_multiclass_v1",
                                 seed=seed,
                                 X=X,
                                 labels=labels,
                                 n_classes=3,
                                 n_components=2)


def synthetic_mb_pls_block_weighted_v1() -> dict[str, Any]:
    """20 samples, 3 blocks, 7 features and 2 targets for MB-PLS parity."""
    seed = 100
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(20, 3))
    block_1 = np.column_stack([
        0.82 * latent[:, 0] + 0.15 * latent[:, 1],
        -0.34 * latent[:, 0] + 0.48 * latent[:, 2],
    ]) + rng.standard_normal(size=(20, 2)) * 0.04
    block_2 = np.column_stack([
        0.62 * latent[:, 1] - 0.18 * latent[:, 2],
        0.23 * latent[:, 0] + 0.41 * latent[:, 1],
        -0.27 * latent[:, 0] + 0.54 * latent[:, 2],
    ]) + rng.standard_normal(size=(20, 3)) * 0.05
    grid = np.linspace(0.0, 2.0 * np.pi, 20)
    block_3 = np.column_stack([
        np.sin(grid) + 0.16 * latent[:, 0],
        np.cos(0.5 * grid) + 0.22 * latent[:, 2],
    ]) + rng.standard_normal(size=(20, 2)) * 0.03
    X = np.column_stack([block_1, block_2, block_3])
    Y = np.column_stack([
        1.05 * latent[:, 0] - 0.52 * latent[:, 2] + rng.standard_normal(size=20) * 0.04,
        -0.42 * latent[:, 1] + 0.73 * latent[:, 2] + rng.standard_normal(size=20) * 0.04,
    ])
    return _mb_pls_fixture("synthetic_mb_pls_block_weighted_v1",
                           seed=seed,
                           X=X,
                           Y=Y,
                           block_sizes=[2, 3, 2],
                           n_components=2)


def synthetic_lw_pls_local_window_v1() -> dict[str, Any]:
    """18 samples, 5 features and 2 targets for local-window LW-PLS parity."""
    seed = 101
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(18, 3))
    trend = np.linspace(-0.7, 0.8, 18)
    X = np.column_stack([
        0.91 * latent[:, 0] + 0.14 * trend,
        -0.42 * latent[:, 0] + 0.35 * latent[:, 1],
        0.58 * latent[:, 1] - 0.26 * latent[:, 2],
        0.33 * latent[:, 0] + 0.51 * latent[:, 2],
        np.sin(np.linspace(0.0, 2.3 * np.pi, 18)) + 0.12 * latent[:, 1],
    ]) + rng.standard_normal(size=(18, 5)) * 0.045
    Y = np.column_stack([
        1.02 * latent[:, 0] - 0.48 * latent[:, 2] + 0.08 * trend,
        -0.56 * latent[:, 1] + 0.62 * latent[:, 2] - 0.05 * trend,
    ]) + rng.standard_normal(size=(18, 2)) * 0.04
    return _lw_pls_fixture("synthetic_lw_pls_local_window_v1",
                           seed=seed,
                           X=X,
                           Y=Y,
                           n_components=2,
                           n_neighbors=9)


def synthetic_variable_importance_pls2_v1() -> dict[str, Any]:
    """18 samples, 7 features, 2 targets for VIP and selectivity-ratio parity."""
    seed = 59
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(18, 3))
    X = np.column_stack([
        0.88 * latent[:, 0] + 0.15 * latent[:, 1],
        -0.52 * latent[:, 0] + 0.31 * latent[:, 2],
        0.42 * latent[:, 1] - 0.28 * latent[:, 2],
        0.26 * latent[:, 0] + 0.64 * latent[:, 1],
        -0.18 * latent[:, 1] + 0.59 * latent[:, 2],
        np.cos(np.linspace(0.0, 2.5 * np.pi, 18)),
        rng.standard_normal(size=18) * 0.05,
    ])
    Y = np.column_stack([
        1.12 * latent[:, 0] - 0.44 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
        -0.61 * latent[:, 1] + 0.36 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
    ])
    return _variable_importance_fixture("synthetic_variable_importance_pls2_v1",
                                        seed=seed,
                                        X=X,
                                        Y=Y,
                                        n_components=2)


def synthetic_variable_selection_rankers_v1() -> dict[str, Any]:
    """18 samples, 7 features, 2 targets for deterministic ranker parity."""
    seed = 59
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(18, 3))
    X = np.column_stack([
        0.88 * latent[:, 0] + 0.15 * latent[:, 1],
        -0.52 * latent[:, 0] + 0.31 * latent[:, 2],
        0.42 * latent[:, 1] - 0.28 * latent[:, 2],
        0.26 * latent[:, 0] + 0.64 * latent[:, 1],
        -0.18 * latent[:, 1] + 0.59 * latent[:, 2],
        np.cos(np.linspace(0.0, 2.5 * np.pi, 18)),
        rng.standard_normal(size=18) * 0.05,
    ])
    Y = np.column_stack([
        1.12 * latent[:, 0] - 0.44 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
        -0.61 * latent[:, 1] + 0.36 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
    ])
    return _variable_selection_fixture("synthetic_variable_selection_rankers_v1",
                                       seed=seed,
                                       X=X,
                                       Y=Y,
                                       n_components=2,
                                       top_k=3)


def synthetic_interval_selection_moving_window_v1() -> dict[str, Any]:
    """20 samples, 10 features, 2 targets for moving-window interval CV parity."""
    seed = 60
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(20, 4))
    X = np.column_stack([
        0.15 * latent[:, 0] + 0.07 * rng.standard_normal(size=20),
        0.34 * latent[:, 1] - 0.10 * latent[:, 2] + 0.04 * rng.standard_normal(size=20),
        0.88 * latent[:, 0] + 0.21 * latent[:, 2] + 0.03 * rng.standard_normal(size=20),
        -0.46 * latent[:, 0] + 0.63 * latent[:, 1] + 0.04 * rng.standard_normal(size=20),
        0.51 * latent[:, 1] - 0.38 * latent[:, 2] + 0.05 * rng.standard_normal(size=20),
        -0.29 * latent[:, 2] + 0.44 * latent[:, 3] + 0.06 * rng.standard_normal(size=20),
        0.12 * latent[:, 0] + 0.80 * latent[:, 3] + 0.05 * rng.standard_normal(size=20),
        -0.58 * latent[:, 1] + 0.25 * latent[:, 3] + 0.06 * rng.standard_normal(size=20),
        np.sin(np.linspace(0.0, 2.0 * np.pi, 20)),
        rng.standard_normal(size=20) * 0.08,
    ])
    Y = np.column_stack([
        0.92 * latent[:, 0] - 0.36 * latent[:, 2] + 0.06 * rng.standard_normal(size=20),
        -0.55 * latent[:, 1] + 0.48 * latent[:, 3] + 0.06 * rng.standard_normal(size=20),
    ])
    return _interval_selection_fixture("synthetic_interval_selection_moving_window_v1",
                                       seed=seed,
                                       X=X,
                                       Y=Y,
                                       n_components=2,
                                       n_splits=5,
                                       interval_width=3,
                                       step=1)


def synthetic_bipls_backward_intervals_v1() -> dict[str, Any]:
    """24 samples, 12 features, 2 targets for deterministic backward interval PLS."""
    seed = 87
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(24, 4))
    X = np.column_stack([
        0.92 * latent[:, 0] + 0.10 * latent[:, 1] + 0.04 * rng.standard_normal(size=24),
        -0.42 * latent[:, 0] + 0.50 * latent[:, 2] + 0.04 * rng.standard_normal(size=24),
        0.34 * latent[:, 0] - 0.20 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        0.64 * latent[:, 1] - 0.22 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.20 * latent[:, 0] + 0.76 * latent[:, 3] + 0.04 * rng.standard_normal(size=24),
        -0.53 * latent[:, 2] + 0.28 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.16,
        np.sin(np.linspace(0.0, 3.0 * np.pi, 24)) + 0.05 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.14,
        np.cos(np.linspace(0.0, 2.0 * np.pi, 24)) + 0.05 * rng.standard_normal(size=24),
        np.linspace(-0.5, 0.5, 24) + 0.12 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.18,
    ])
    Y = np.column_stack([
        1.08 * latent[:, 0] - 0.48 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        -0.41 * latent[:, 1] + 0.66 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
    ])
    return _bipls_selection_fixture("synthetic_bipls_backward_intervals_v1",
                                    seed=seed,
                                    X=X,
                                    Y=Y,
                                    n_components=2,
                                    n_splits=4,
                                    interval_width=3,
                                    min_intervals=2)


def synthetic_sipls_interval_combinations_v1() -> dict[str, Any]:
    """26 samples, 15 features, 2 targets for deterministic synergy interval PLS."""
    seed = 88
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(26, 4))
    X = np.column_stack([
        0.22 * latent[:, 0] + 0.16 * latent[:, 1] + 0.14 * rng.standard_normal(size=26),
        -0.18 * latent[:, 2] + 0.10 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.18,
        0.88 * latent[:, 0] - 0.26 * latent[:, 2] + 0.04 * rng.standard_normal(size=26),
        -0.46 * latent[:, 0] + 0.42 * latent[:, 2] + 0.04 * rng.standard_normal(size=26),
        0.31 * latent[:, 0] + 0.35 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        np.sin(np.linspace(0.0, 2.5 * np.pi, 26)) + 0.06 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.13,
        np.linspace(-0.4, 0.4, 26) + 0.08 * rng.standard_normal(size=26),
        0.71 * latent[:, 1] - 0.38 * latent[:, 3] + 0.04 * rng.standard_normal(size=26),
        -0.25 * latent[:, 1] + 0.82 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        0.43 * latent[:, 1] + 0.29 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        np.cos(np.linspace(0.0, 3.0 * np.pi, 26)) + 0.06 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.16,
        np.linspace(0.5, -0.5, 26) + 0.10 * rng.standard_normal(size=26),
    ])
    Y = np.column_stack([
        1.00 * latent[:, 0] - 0.46 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        -0.43 * latent[:, 1] + 0.69 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
    ])
    return _sipls_selection_fixture("synthetic_sipls_interval_combinations_v1",
                                    seed=seed,
                                    X=X,
                                    Y=Y,
                                    n_components=2,
                                    n_splits=4,
                                    interval_width=3,
                                    combination_size=2)


def synthetic_coefficient_stability_mcuve_v1() -> dict[str, Any]:
    """24 samples, 8 features, 2 targets for Monte-Carlo coefficient stability."""
    seed = 61
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(24, 4))
    X = np.column_stack([
        0.82 * latent[:, 0] + 0.12 * latent[:, 1] + 0.05 * rng.standard_normal(size=24),
        -0.35 * latent[:, 0] + 0.61 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        0.49 * latent[:, 1] - 0.22 * latent[:, 3] + 0.04 * rng.standard_normal(size=24),
        0.18 * latent[:, 0] + 0.77 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        -0.57 * latent[:, 2] + 0.38 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.28 * latent[:, 0] - 0.19 * latent[:, 1] + 0.08 * rng.standard_normal(size=24),
        np.cos(np.linspace(0.0, 3.0 * np.pi, 24)) + 0.03 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.10,
    ])
    Y = np.column_stack([
        1.03 * latent[:, 0] - 0.48 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        -0.42 * latent[:, 1] + 0.58 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
    ])
    return _coefficient_stability_fixture("synthetic_coefficient_stability_mcuve_v1",
                                          seed=seed,
                                          X=X,
                                          Y=Y,
                                          n_components=2,
                                          n_repeats=8,
                                          test_count=6,
                                          top_k=3)


def synthetic_uve_artificial_variables_v1() -> dict[str, Any]:
    """26 samples, 9 features, 2 targets for UVE artificial-variable thresholding."""
    seed = 62
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(26, 4))
    X = np.column_stack([
        0.92 * latent[:, 0] + 0.06 * latent[:, 1] + 0.04 * rng.standard_normal(size=26),
        -0.48 * latent[:, 0] + 0.54 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        0.72 * latent[:, 1] - 0.16 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        0.20 * latent[:, 0] + 0.86 * latent[:, 3] + 0.04 * rng.standard_normal(size=26),
        -0.61 * latent[:, 2] + 0.33 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        0.25 * latent[:, 0] - 0.18 * latent[:, 1] + 0.08 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.12,
        np.sin(np.linspace(0.0, 4.0 * np.pi, 26)) + 0.04 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.15,
    ])
    Y = np.column_stack([
        1.18 * latent[:, 0] - 0.53 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        -0.47 * latent[:, 1] + 0.73 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
    ])
    return _uve_fixture("synthetic_uve_artificial_variables_v1",
                        seed=seed,
                        X=X,
                        Y=Y,
                        n_components=2,
                        n_repeats=9,
                        test_count=6,
                        validation_seed=63,
                        noise_features=5,
                        noise_seed=64)


def synthetic_emcuve_pls_ensemble_v1() -> dict[str, Any]:
    """30 samples, 11 features, 2 targets for deterministic EMCUVE voting."""
    seed = 82
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(30, 5))
    trend = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([
        0.96 * latent[:, 0] + 0.04 * latent[:, 1] + 0.04 * rng.standard_normal(size=30),
        -0.50 * latent[:, 0] + 0.58 * latent[:, 2] + 0.05 * rng.standard_normal(size=30),
        0.69 * latent[:, 1] - 0.20 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.18 * latent[:, 0] + 0.87 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        -0.64 * latent[:, 2] + 0.37 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.42 * latent[:, 4] + 0.18 * latent[:, 0] + 0.06 * rng.standard_normal(size=30),
        0.26 * latent[:, 1] - 0.24 * latent[:, 4] + 0.08 * rng.standard_normal(size=30),
        np.sin(np.linspace(0.0, 3.5 * np.pi, 30)) + 0.04 * rng.standard_normal(size=30),
        np.cos(np.linspace(0.0, 2.5 * np.pi, 30)) + 0.05 * rng.standard_normal(size=30),
        0.28 * trend + 0.12 * rng.standard_normal(size=30),
        rng.standard_normal(size=30) * 0.16,
    ])
    Y = np.column_stack([
        1.16 * latent[:, 0] - 0.57 * latent[:, 2] + 0.31 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
        -0.44 * latent[:, 1] + 0.71 * latent[:, 3] - 0.26 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
    ])
    return _emcuve_fixture("synthetic_emcuve_pls_ensemble_v1",
                           seed=seed,
                           X=X,
                           Y=Y,
                           n_components=2,
                           n_repeats=8,
                           test_count=7,
                           validation_seed=83,
                           noise_features=6,
                           noise_seed=84,
                           n_ensembles=5,
                           vote_threshold=0.6)


def synthetic_randomization_pls_permutation_v1() -> dict[str, Any]:
    """28 samples, 10 features, 2 targets for deterministic PLS randomisation tests."""
    seed = 85
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(28, 4))
    trend = np.linspace(-1.0, 1.0, 28)
    X = np.column_stack([
        1.05 * latent[:, 0] + 0.05 * rng.standard_normal(size=28),
        -0.62 * latent[:, 0] + 0.55 * latent[:, 2] + 0.05 * rng.standard_normal(size=28),
        0.74 * latent[:, 1] - 0.18 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.22 * latent[:, 0] + 0.82 * latent[:, 3] + 0.04 * rng.standard_normal(size=28),
        -0.66 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.32 * latent[:, 1] + 0.22 * latent[:, 2] + 0.07 * rng.standard_normal(size=28),
        np.sin(np.linspace(0.0, 3.0 * np.pi, 28)) + 0.05 * rng.standard_normal(size=28),
        np.cos(np.linspace(0.0, 2.0 * np.pi, 28)) + 0.05 * rng.standard_normal(size=28),
        0.30 * trend + 0.12 * rng.standard_normal(size=28),
        rng.standard_normal(size=28) * 0.18,
    ])
    Y = np.column_stack([
        1.24 * latent[:, 0] - 0.55 * latent[:, 2] + 0.05 * rng.standard_normal(size=28),
        -0.48 * latent[:, 1] + 0.78 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
    ])
    return _randomization_selection_fixture("synthetic_randomization_pls_permutation_v1",
                                            seed=seed,
                                            X=X,
                                            Y=Y,
                                            n_components=2,
                                            n_permutations=15,
                                            randomization_seed=86,
                                            alpha=0.25)


def synthetic_spa_pls_projection_v1() -> dict[str, Any]:
    """24 samples, 10 features, 2 targets for deterministic SPA-PLS selection."""
    seed = 65
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(24, 4))
    trend = np.linspace(-1.0, 1.0, 24)
    X = np.column_stack([
        0.84 * latent[:, 0] + 0.08 * latent[:, 1] + 0.04 * rng.standard_normal(size=24),
        -0.44 * latent[:, 0] + 0.52 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        0.64 * latent[:, 1] - 0.26 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.22 * latent[:, 0] + 0.79 * latent[:, 3] + 0.04 * rng.standard_normal(size=24),
        -0.58 * latent[:, 2] + 0.36 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.28 * latent[:, 0] - 0.22 * latent[:, 1] + 0.06 * rng.standard_normal(size=24),
        np.sin(np.linspace(0.0, 3.0 * np.pi, 24)) + 0.04 * rng.standard_normal(size=24),
        np.cos(np.linspace(0.0, 2.0 * np.pi, 24)) + 0.04 * rng.standard_normal(size=24),
        0.35 * trend + 0.12 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.14,
    ])
    Y = np.column_stack([
        1.12 * latent[:, 0] - 0.50 * latent[:, 2] + 0.05 * rng.standard_normal(size=24),
        -0.39 * latent[:, 1] + 0.68 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
    ])
    return _spa_selection_fixture("synthetic_spa_pls_projection_v1",
                                  seed=seed,
                                  X=X,
                                  Y=Y,
                                  n_components=2,
                                  top_k=5)


def synthetic_cars_pls_competitive_v1() -> dict[str, Any]:
    """28 samples, 12 features, 2 targets for deterministic CARS-PLS selection."""
    seed = 66
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(28, 5))
    X = np.column_stack([
        0.95 * latent[:, 0] + 0.05 * latent[:, 1] + 0.04 * rng.standard_normal(size=28),
        -0.52 * latent[:, 0] + 0.55 * latent[:, 2] + 0.05 * rng.standard_normal(size=28),
        0.73 * latent[:, 1] - 0.18 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.20 * latent[:, 0] + 0.83 * latent[:, 3] + 0.04 * rng.standard_normal(size=28),
        -0.62 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.33 * latent[:, 4] + 0.15 * latent[:, 0] + 0.06 * rng.standard_normal(size=28),
        0.24 * latent[:, 1] - 0.21 * latent[:, 4] + 0.07 * rng.standard_normal(size=28),
        np.sin(np.linspace(0.0, 3.5 * np.pi, 28)) + 0.04 * rng.standard_normal(size=28),
        np.cos(np.linspace(0.0, 2.5 * np.pi, 28)) + 0.04 * rng.standard_normal(size=28),
        rng.standard_normal(size=28) * 0.11,
        0.30 * np.linspace(-1.0, 1.0, 28) + 0.10 * rng.standard_normal(size=28),
        rng.standard_normal(size=28) * 0.16,
    ])
    Y = np.column_stack([
        1.16 * latent[:, 0] - 0.48 * latent[:, 2] + 0.34 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
        -0.43 * latent[:, 1] + 0.71 * latent[:, 3] - 0.20 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
    ])
    return _cars_selection_fixture("synthetic_cars_pls_competitive_v1",
                                   seed=seed,
                                   X=X,
                                   Y=Y,
                                   n_components=2,
                                   n_splits=4,
                                   n_iterations=6,
                                   min_features=3)


def synthetic_random_frog_pls_v1() -> dict[str, Any]:
    """30 samples, 11 features, 2 targets for deterministic Random Frog PLS."""
    seed = 67
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(30, 5))
    wave = np.linspace(0.0, 2.0 * np.pi, 30)
    X = np.column_stack([
        0.88 * latent[:, 0] + 0.10 * latent[:, 1] + 0.04 * rng.standard_normal(size=30),
        -0.47 * latent[:, 0] + 0.50 * latent[:, 2] + 0.05 * rng.standard_normal(size=30),
        0.70 * latent[:, 1] - 0.20 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.18 * latent[:, 0] + 0.81 * latent[:, 3] + 0.04 * rng.standard_normal(size=30),
        -0.55 * latent[:, 2] + 0.34 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.39 * latent[:, 4] + 0.14 * latent[:, 0] + 0.06 * rng.standard_normal(size=30),
        0.23 * latent[:, 1] - 0.24 * latent[:, 4] + 0.07 * rng.standard_normal(size=30),
        np.sin(1.5 * wave) + 0.05 * rng.standard_normal(size=30),
        np.cos(1.2 * wave) + 0.05 * rng.standard_normal(size=30),
        0.26 * np.linspace(-1.0, 1.0, 30) + 0.10 * rng.standard_normal(size=30),
        rng.standard_normal(size=30) * 0.14,
    ])
    Y = np.column_stack([
        1.08 * latent[:, 0] - 0.44 * latent[:, 2] + 0.31 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
        -0.40 * latent[:, 1] + 0.69 * latent[:, 3] - 0.22 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
    ])
    return _random_frog_selection_fixture("synthetic_random_frog_pls_v1",
                                          seed=seed,
                                          X=X,
                                          Y=Y,
                                          n_components=2,
                                          n_splits=5,
                                          n_iterations=9,
                                          initial_size=5,
                                          min_size=3,
                                          max_size=8,
                                          top_k=5,
                                          frog_seed=6801)


def synthetic_scars_pls_stability_v1() -> dict[str, Any]:
    """32 samples, 13 features, 2 targets for deterministic SCARS-PLS selection."""
    seed = 69
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(32, 5))
    wave = np.linspace(0.0, 2.4 * np.pi, 32)
    X = np.column_stack([
        0.92 * latent[:, 0] + 0.12 * latent[:, 1] + 0.04 * rng.standard_normal(size=32),
        -0.50 * latent[:, 0] + 0.48 * latent[:, 2] + 0.05 * rng.standard_normal(size=32),
        0.68 * latent[:, 1] - 0.22 * latent[:, 3] + 0.05 * rng.standard_normal(size=32),
        0.20 * latent[:, 0] + 0.84 * latent[:, 3] + 0.04 * rng.standard_normal(size=32),
        -0.52 * latent[:, 2] + 0.36 * latent[:, 3] + 0.05 * rng.standard_normal(size=32),
        0.41 * latent[:, 4] + 0.13 * latent[:, 0] + 0.06 * rng.standard_normal(size=32),
        0.26 * latent[:, 1] - 0.22 * latent[:, 4] + 0.07 * rng.standard_normal(size=32),
        np.sin(1.3 * wave) + 0.05 * rng.standard_normal(size=32),
        np.cos(1.6 * wave) + 0.05 * rng.standard_normal(size=32),
        0.24 * np.linspace(-1.0, 1.0, 32) + 0.10 * rng.standard_normal(size=32),
        rng.standard_normal(size=32) * 0.13,
        0.35 * latent[:, 2] - 0.17 * latent[:, 4] + 0.08 * rng.standard_normal(size=32),
        rng.standard_normal(size=32) * 0.16,
    ])
    Y = np.column_stack([
        1.12 * latent[:, 0] - 0.42 * latent[:, 2] + 0.33 * latent[:, 4] + 0.05 * rng.standard_normal(size=32),
        -0.38 * latent[:, 1] + 0.72 * latent[:, 3] - 0.24 * latent[:, 4] + 0.05 * rng.standard_normal(size=32),
    ])
    return _scars_selection_fixture("synthetic_scars_pls_stability_v1",
                                    seed=seed,
                                    X=X,
                                    Y=Y,
                                    n_components=2,
                                    n_splits=4,
                                    n_iterations=7,
                                    min_features=4,
                                    sample_fraction=0.75,
                                    scars_seed=7201)


def synthetic_ga_pls_wrapper_v1() -> dict[str, Any]:
    """30 samples, 12 features, 2 targets for deterministic GA-PLS selection."""
    seed = 71
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(30, 5))
    wave = np.linspace(0.0, 2.1 * np.pi, 30)
    X = np.column_stack([
        0.86 * latent[:, 0] + 0.16 * latent[:, 1] + 0.04 * rng.standard_normal(size=30),
        -0.49 * latent[:, 0] + 0.52 * latent[:, 2] + 0.05 * rng.standard_normal(size=30),
        0.66 * latent[:, 1] - 0.24 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.22 * latent[:, 0] + 0.82 * latent[:, 3] + 0.04 * rng.standard_normal(size=30),
        -0.51 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.44 * latent[:, 4] + 0.15 * latent[:, 0] + 0.06 * rng.standard_normal(size=30),
        0.28 * latent[:, 1] - 0.20 * latent[:, 4] + 0.07 * rng.standard_normal(size=30),
        np.sin(1.4 * wave) + 0.05 * rng.standard_normal(size=30),
        np.cos(1.1 * wave) + 0.05 * rng.standard_normal(size=30),
        0.23 * np.linspace(-1.0, 1.0, 30) + 0.10 * rng.standard_normal(size=30),
        0.34 * latent[:, 2] - 0.16 * latent[:, 4] + 0.08 * rng.standard_normal(size=30),
        rng.standard_normal(size=30) * 0.15,
    ])
    Y = np.column_stack([
        1.10 * latent[:, 0] - 0.45 * latent[:, 2] + 0.32 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
        -0.41 * latent[:, 1] + 0.70 * latent[:, 3] - 0.21 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
    ])
    return _ga_selection_fixture("synthetic_ga_pls_wrapper_v1",
                                 seed=seed,
                                 X=X,
                                 Y=Y,
                                 n_components=2,
                                 n_splits=5,
                                 n_generations=6,
                                 population_size=8,
                                 min_features=3,
                                 max_features=8,
                                 mutation_rate=0.35,
                                 ga_seed=9101)


def synthetic_shaving_pls_recursive_v1() -> dict[str, Any]:
    """28 samples, 12 features, 2 targets for deterministic shaving PLS."""
    seed = 73
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(28, 5))
    wave = np.linspace(0.0, 2.3 * np.pi, 28)
    X = np.column_stack([
        0.89 * latent[:, 0] + 0.12 * latent[:, 1] + 0.04 * rng.standard_normal(size=28),
        -0.46 * latent[:, 0] + 0.55 * latent[:, 2] + 0.05 * rng.standard_normal(size=28),
        0.64 * latent[:, 1] - 0.25 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.21 * latent[:, 0] + 0.78 * latent[:, 3] + 0.04 * rng.standard_normal(size=28),
        -0.50 * latent[:, 2] + 0.33 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.42 * latent[:, 4] + 0.16 * latent[:, 0] + 0.06 * rng.standard_normal(size=28),
        0.25 * latent[:, 1] - 0.23 * latent[:, 4] + 0.07 * rng.standard_normal(size=28),
        np.sin(1.2 * wave) + 0.05 * rng.standard_normal(size=28),
        np.cos(1.5 * wave) + 0.05 * rng.standard_normal(size=28),
        0.27 * np.linspace(-1.0, 1.0, 28) + 0.10 * rng.standard_normal(size=28),
        0.31 * latent[:, 2] - 0.18 * latent[:, 4] + 0.08 * rng.standard_normal(size=28),
        rng.standard_normal(size=28) * 0.15,
    ])
    Y = np.column_stack([
        1.13 * latent[:, 0] - 0.43 * latent[:, 2] + 0.35 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
        -0.39 * latent[:, 1] + 0.73 * latent[:, 3] - 0.22 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
    ])
    return _shaving_selection_fixture("synthetic_shaving_pls_recursive_v1",
                                      seed=seed,
                                      X=X,
                                      Y=Y,
                                      n_components=2,
                                      n_splits=4,
                                      n_steps=5,
                                      min_features=3,
                                      shave_fraction=0.25)


def synthetic_rep_pls_recursive_v1() -> dict[str, Any]:
    """28 samples, 12 features, 2 targets for deterministic REP-PLS selection."""
    seed = 74
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(28, 5))
    wave = np.linspace(0.0, 2.3 * np.pi, 28)
    X = np.column_stack([
        0.87 * latent[:, 0] + 0.13 * latent[:, 1] + 0.04 * rng.standard_normal(size=28),
        -0.48 * latent[:, 0] + 0.53 * latent[:, 2] + 0.05 * rng.standard_normal(size=28),
        0.62 * latent[:, 1] - 0.27 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.23 * latent[:, 0] + 0.79 * latent[:, 3] + 0.04 * rng.standard_normal(size=28),
        -0.52 * latent[:, 2] + 0.32 * latent[:, 3] + 0.05 * rng.standard_normal(size=28),
        0.43 * latent[:, 4] + 0.14 * latent[:, 0] + 0.06 * rng.standard_normal(size=28),
        0.27 * latent[:, 1] - 0.21 * latent[:, 4] + 0.07 * rng.standard_normal(size=28),
        np.sin(1.25 * wave) + 0.05 * rng.standard_normal(size=28),
        np.cos(1.45 * wave) + 0.05 * rng.standard_normal(size=28),
        0.25 * np.linspace(-1.0, 1.0, 28) + 0.10 * rng.standard_normal(size=28),
        0.33 * latent[:, 2] - 0.17 * latent[:, 4] + 0.08 * rng.standard_normal(size=28),
        rng.standard_normal(size=28) * 0.15,
    ])
    Y = np.column_stack([
        1.11 * latent[:, 0] - 0.45 * latent[:, 2] + 0.33 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
        -0.40 * latent[:, 1] + 0.71 * latent[:, 3] - 0.23 * latent[:, 4] + 0.05 * rng.standard_normal(size=28),
    ])
    return _rep_selection_fixture("synthetic_rep_pls_recursive_v1",
                                  seed=seed,
                                  X=X,
                                  Y=Y,
                                  n_components=2,
                                  n_splits=4,
                                  n_steps=5,
                                  min_features=4,
                                  remove_count=2)


def synthetic_ipw_pls_reweighted_v1() -> dict[str, Any]:
    """30 samples, 11 features, 2 targets for deterministic IPW-PLS selection."""
    seed = 78
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(30, 5))
    wave = np.linspace(0.0, 2.6 * np.pi, 30)
    X = np.column_stack([
        0.94 * latent[:, 0] + 0.11 * latent[:, 1] + 0.04 * rng.standard_normal(size=30),
        -0.46 * latent[:, 0] + 0.58 * latent[:, 2] + 0.05 * rng.standard_normal(size=30),
        0.69 * latent[:, 1] - 0.24 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.21 * latent[:, 0] + 0.82 * latent[:, 3] + 0.04 * rng.standard_normal(size=30),
        -0.50 * latent[:, 2] + 0.34 * latent[:, 3] + 0.05 * rng.standard_normal(size=30),
        0.48 * latent[:, 4] + 0.16 * latent[:, 0] + 0.06 * rng.standard_normal(size=30),
        0.32 * latent[:, 1] - 0.20 * latent[:, 4] + 0.07 * rng.standard_normal(size=30),
        np.sin(1.20 * wave) + 0.05 * rng.standard_normal(size=30),
        np.cos(1.35 * wave) + 0.05 * rng.standard_normal(size=30),
        0.27 * np.linspace(-1.0, 1.0, 30) + 0.09 * rng.standard_normal(size=30),
        rng.standard_normal(size=30) * 0.14,
    ])
    Y = np.column_stack([
        1.08 * latent[:, 0] - 0.42 * latent[:, 2] + 0.31 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
        -0.43 * latent[:, 1] + 0.74 * latent[:, 3] - 0.19 * latent[:, 4] + 0.05 * rng.standard_normal(size=30),
    ])
    return _ipw_selection_fixture("synthetic_ipw_pls_reweighted_v1",
                                  seed=seed,
                                  X=X,
                                  Y=Y,
                                  n_components=2,
                                  n_splits=5,
                                  n_iterations=5,
                                  top_k=5,
                                  damping=0.45,
                                  weight_floor=0.05)


def synthetic_st_pls_threshold_v1() -> dict[str, Any]:
    """29 samples, 12 features, 2 targets for deterministic ST-PLS selection."""
    seed = 82
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(29, 5))
    wave = np.linspace(0.0, 2.4 * np.pi, 29)
    X = np.column_stack([
        0.90 * latent[:, 0] + 0.12 * latent[:, 1] + 0.04 * rng.standard_normal(size=29),
        -0.43 * latent[:, 0] + 0.60 * latent[:, 2] + 0.05 * rng.standard_normal(size=29),
        0.66 * latent[:, 1] - 0.26 * latent[:, 3] + 0.05 * rng.standard_normal(size=29),
        0.24 * latent[:, 0] + 0.78 * latent[:, 3] + 0.04 * rng.standard_normal(size=29),
        -0.53 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=29),
        0.45 * latent[:, 4] + 0.15 * latent[:, 0] + 0.06 * rng.standard_normal(size=29),
        0.29 * latent[:, 1] - 0.18 * latent[:, 4] + 0.07 * rng.standard_normal(size=29),
        0.26 * latent[:, 2] - 0.14 * latent[:, 4] + 0.08 * rng.standard_normal(size=29),
        np.sin(1.15 * wave) + 0.05 * rng.standard_normal(size=29),
        np.cos(1.50 * wave) + 0.05 * rng.standard_normal(size=29),
        0.24 * np.linspace(-1.0, 1.0, 29) + 0.09 * rng.standard_normal(size=29),
        rng.standard_normal(size=29) * 0.14,
    ])
    Y = np.column_stack([
        1.07 * latent[:, 0] - 0.41 * latent[:, 2] + 0.28 * latent[:, 4] + 0.05 * rng.standard_normal(size=29),
        -0.41 * latent[:, 1] + 0.72 * latent[:, 3] - 0.20 * latent[:, 4] + 0.05 * rng.standard_normal(size=29),
    ])
    return _st_selection_fixture("synthetic_st_pls_threshold_v1",
                                 seed=seed,
                                 X=X,
                                 Y=Y,
                                 n_components=2,
                                 n_splits=5,
                                 thresholds=[0.85, 0.70, 0.55, 0.40, 0.25],
                                 min_selected=4)


def synthetic_bve_pls_backward_v1() -> dict[str, Any]:
    """26 samples, 9 features, 2 targets for deterministic BVE-PLS selection."""
    seed = 75
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(26, 4))
    wave = np.linspace(0.0, 2.0 * np.pi, 26)
    X = np.column_stack([
        0.91 * latent[:, 0] + 0.10 * latent[:, 1] + 0.04 * rng.standard_normal(size=26),
        -0.44 * latent[:, 0] + 0.54 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        0.67 * latent[:, 1] - 0.21 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        0.19 * latent[:, 0] + 0.80 * latent[:, 3] + 0.04 * rng.standard_normal(size=26),
        -0.48 * latent[:, 2] + 0.29 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
        np.sin(1.4 * wave) + 0.05 * rng.standard_normal(size=26),
        np.cos(1.1 * wave) + 0.05 * rng.standard_normal(size=26),
        0.24 * np.linspace(-1.0, 1.0, 26) + 0.10 * rng.standard_normal(size=26),
        rng.standard_normal(size=26) * 0.15,
    ])
    Y = np.column_stack([
        1.09 * latent[:, 0] - 0.40 * latent[:, 2] + 0.05 * rng.standard_normal(size=26),
        -0.42 * latent[:, 1] + 0.75 * latent[:, 3] + 0.05 * rng.standard_normal(size=26),
    ])
    return _bve_selection_fixture("synthetic_bve_pls_backward_v1",
                                  seed=seed,
                                  X=X,
                                  Y=Y,
                                  n_components=2,
                                  n_splits=5,
                                  n_steps=5,
                                  min_features=3)


def synthetic_t2_pls_hotelling_v1() -> dict[str, Any]:
    """27 samples, 11 features, 2 targets for deterministic T2-PLS selection."""
    seed = 79
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(27, 4))
    wave = np.linspace(0.0, 2.4 * np.pi, 27)
    X = np.column_stack([
        0.88 * latent[:, 0] + 0.12 * latent[:, 1] + 0.04 * rng.standard_normal(size=27),
        -0.41 * latent[:, 0] + 0.57 * latent[:, 2] + 0.04 * rng.standard_normal(size=27),
        0.63 * latent[:, 1] - 0.28 * latent[:, 3] + 0.05 * rng.standard_normal(size=27),
        0.22 * latent[:, 0] + 0.77 * latent[:, 3] + 0.05 * rng.standard_normal(size=27),
        -0.52 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=27),
        0.43 * latent[:, 0] - 0.18 * latent[:, 1] + 0.06 * rng.standard_normal(size=27),
        np.sin(1.2 * wave) + 0.05 * rng.standard_normal(size=27),
        np.cos(1.6 * wave) + 0.05 * rng.standard_normal(size=27),
        0.30 * np.linspace(-1.0, 1.0, 27) + 0.09 * rng.standard_normal(size=27),
        rng.standard_normal(size=27) * 0.13,
        0.20 * latent[:, 2] - 0.15 * latent[:, 0] + 0.08 * rng.standard_normal(size=27),
    ])
    Y = np.column_stack([
        1.05 * latent[:, 0] - 0.37 * latent[:, 2] + 0.20 * latent[:, 3] + 0.05 * rng.standard_normal(size=27),
        -0.40 * latent[:, 1] + 0.70 * latent[:, 3] - 0.17 * latent[:, 2] + 0.05 * rng.standard_normal(size=27),
    ])
    return _t2_selection_fixture("synthetic_t2_pls_hotelling_v1",
                                 seed=seed,
                                 X=X,
                                 Y=Y,
                                 n_components=3,
                                 n_splits=5,
                                 alpha=[0.8, 0.6, 0.4, 0.2, 0.05],
                                 min_selected=5)


def synthetic_wvc_pls_numeric_v1() -> dict[str, Any]:
    """24 samples, 10 features, 1 target for deterministic WVC-PLS selection."""
    seed = 83
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(24, 4))
    wave = np.linspace(0.0, 2.1 * np.pi, 24)
    X = np.column_stack([
        0.90 * latent[:, 0] + 0.10 * latent[:, 1] + 0.04 * rng.standard_normal(size=24),
        -0.42 * latent[:, 0] + 0.56 * latent[:, 2] + 0.04 * rng.standard_normal(size=24),
        0.65 * latent[:, 1] - 0.26 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.24 * latent[:, 0] + 0.75 * latent[:, 3] + 0.04 * rng.standard_normal(size=24),
        -0.48 * latent[:, 2] + 0.35 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.36 * latent[:, 0] - 0.22 * latent[:, 1] + 0.06 * rng.standard_normal(size=24),
        np.sin(1.3 * wave) + 0.05 * rng.standard_normal(size=24),
        np.cos(1.4 * wave) + 0.05 * rng.standard_normal(size=24),
        0.28 * np.linspace(-1.0, 1.0, 24) + 0.09 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.14,
    ])
    Y = (
        1.12 * latent[:, 0]
        - 0.42 * latent[:, 2]
        + 0.31 * latent[:, 3]
        + 0.05 * rng.standard_normal(size=24)
    ).reshape(-1, 1)
    return _wvc_selection_fixture("synthetic_wvc_pls_numeric_v1",
                                  seed=seed,
                                  X=X,
                                  Y=Y,
                                  n_components=3,
                                  top_k=5,
                                  normalize=True)


def synthetic_wvc_threshold_factor_v1() -> dict[str, Any]:
    """24 samples, 10 features, 1 target for thresholded/factor WVC-PLS selection."""
    seed = 84
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(24, 4))
    wave = np.linspace(0.0, 2.1 * np.pi, 24)
    X = np.column_stack([
        0.86 * latent[:, 0] + 0.13 * latent[:, 1] + 0.04 * rng.standard_normal(size=24),
        -0.38 * latent[:, 0] + 0.61 * latent[:, 2] + 0.04 * rng.standard_normal(size=24),
        0.62 * latent[:, 1] - 0.24 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.22 * latent[:, 0] + 0.78 * latent[:, 3] + 0.04 * rng.standard_normal(size=24),
        -0.51 * latent[:, 2] + 0.31 * latent[:, 3] + 0.05 * rng.standard_normal(size=24),
        0.34 * latent[:, 0] - 0.20 * latent[:, 1] + 0.06 * rng.standard_normal(size=24),
        np.sin(1.2 * wave) + 0.05 * rng.standard_normal(size=24),
        np.cos(1.5 * wave) + 0.05 * rng.standard_normal(size=24),
        0.24 * np.linspace(-1.0, 1.0, 24) + 0.09 * rng.standard_normal(size=24),
        rng.standard_normal(size=24) * 0.14,
    ])
    Y = (
        1.08 * latent[:, 0]
        - 0.40 * latent[:, 2]
        + 0.34 * latent[:, 3]
        + 0.05 * rng.standard_normal(size=24)
    ).reshape(-1, 1)
    return _wvc_threshold_selection_fixture("synthetic_wvc_threshold_factor_v1",
                                            seed=seed,
                                            X=X,
                                            Y=Y,
                                            n_components=3,
                                            normalize=True,
                                            score_threshold=0.62,
                                            threshold_factor=1.05,
                                            min_selected=3)


def synthetic_component_coefficients_pls2_v1() -> dict[str, Any]:
    """16 samples, 6 features, 2 targets for per-component coefficient parity."""
    seed = 60
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(16, 3))
    X = np.column_stack([
        0.71 * latent[:, 0] + 0.33 * latent[:, 1],
        -0.46 * latent[:, 0] + 0.55 * latent[:, 2],
        0.63 * latent[:, 1] - 0.22 * latent[:, 2],
        0.31 * latent[:, 0] + 0.44 * latent[:, 1] - 0.18 * latent[:, 2],
        np.linspace(-0.8, 0.9, 16),
        rng.standard_normal(size=16) * 0.04,
    ])
    Y = np.column_stack([
        0.96 * latent[:, 0] - 0.52 * latent[:, 1] + rng.standard_normal(size=16) * 0.03,
        -0.43 * latent[:, 0] + 0.78 * latent[:, 2] + rng.standard_normal(size=16) * 0.03,
    ])
    return _component_coefficients_fixture("synthetic_component_coefficients_pls2_v1",
                                           seed=seed,
                                           X=X,
                                           Y=Y,
                                           n_components=3)


def synthetic_validation_kfold_balanced_v1() -> dict[str, Any]:
    """10 samples split into 3 contiguous balanced folds."""
    return _validation_fixture("synthetic_validation_kfold_balanced_v1",
                               kind="kfold",
                               n_samples=10,
                               n_splits=3)


def synthetic_validation_leave_one_out_v1() -> dict[str, Any]:
    """5 samples split into deterministic leave-one-out folds."""
    return _validation_fixture("synthetic_validation_leave_one_out_v1",
                               kind="leave_one_out",
                               n_samples=5)


def synthetic_validation_holdout_v1() -> dict[str, Any]:
    """8 samples with a contiguous holdout window."""
    return _validation_fixture("synthetic_validation_holdout_v1",
                               kind="holdout",
                               n_samples=8,
                               holdout_start=2,
                               holdout_count=3)


def synthetic_validation_external_folds_v1() -> dict[str, Any]:
    """9 samples assigned to three externally supplied folds."""
    fold_ids = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    return _advanced_validation_fixture("synthetic_validation_external_folds_v1",
                                        kind="external",
                                        n_samples=9,
                                        n_splits=3,
                                        fold_ids=fold_ids)


def synthetic_validation_repeated_kfold_v1() -> dict[str, Any]:
    """11 samples split into two deterministic shuffled 4-fold repeats."""
    return _advanced_validation_fixture("synthetic_validation_repeated_kfold_v1",
                                        kind="repeated_kfold",
                                        n_samples=11,
                                        n_splits=4,
                                        n_repeats=2,
                                        seed=91)


def synthetic_validation_monte_carlo_v1() -> dict[str, Any]:
    """12 samples with four deterministic Monte-Carlo validation holdouts."""
    return _advanced_validation_fixture("synthetic_validation_monte_carlo_v1",
                                        kind="monte_carlo",
                                        n_samples=12,
                                        test_count=3,
                                        n_repeats=4,
                                        seed=92)


def synthetic_validation_kennard_stone_v1() -> dict[str, Any]:
    """10 samples selected by deterministic Kennard-Stone X-space coverage."""
    seed = 93
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(10, 2))
    X = np.column_stack([
        0.86 * latent[:, 0] + 0.24 * latent[:, 1],
        -0.42 * latent[:, 0] + 0.51 * latent[:, 1],
        np.linspace(-1.0, 1.0, 10),
        rng.standard_normal(size=10) * 0.05,
    ])
    return _advanced_validation_fixture("synthetic_validation_kennard_stone_v1",
                                        kind="kennard_stone",
                                        n_samples=10,
                                        train_count=6,
                                        seed=seed,
                                        X=X)


def synthetic_validation_spxy_v1() -> dict[str, Any]:
    """12 samples selected by deterministic SPXY joint X/Y coverage."""
    seed = 94
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(12, 3))
    X = np.column_stack([
        0.73 * latent[:, 0] + 0.28 * latent[:, 1],
        -0.39 * latent[:, 0] + 0.62 * latent[:, 2],
        0.44 * latent[:, 1] - 0.31 * latent[:, 2],
        np.cos(np.linspace(0.0, np.pi, 12)),
    ])
    Y = np.column_stack([
        0.92 * latent[:, 0] - 0.33 * latent[:, 2],
        -0.57 * latent[:, 1] + 0.48 * latent[:, 2],
    ])
    return _advanced_validation_fixture("synthetic_validation_spxy_v1",
                                        kind="spxy",
                                        n_samples=12,
                                        train_count=7,
                                        seed=seed,
                                        X=X,
                                        Y=Y)


def synthetic_cv_kfold_nipals_pls1_v1() -> dict[str, Any]:
    """12 samples, 5 features, 1 target, 3-fold out-of-sample PLSRegression."""
    seed = 56
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(12, 2))
    X = np.column_stack([
        0.92 * latent[:, 0] + 0.15 * latent[:, 1],
        -0.33 * latent[:, 0] + 0.71 * latent[:, 1],
        0.58 * latent[:, 0] - 0.18 * latent[:, 1],
        np.linspace(-1.0, 1.0, 12),
        rng.standard_normal(size=12) * 0.08,
    ])
    Y = (1.35 * latent[:, 0] - 0.42 * latent[:, 1] +
         rng.standard_normal(size=12) * 0.05).reshape(-1, 1)
    return _cross_validation_fixture("synthetic_cv_kfold_nipals_pls1_v1",
                                     seed=seed, X=X, Y=Y,
                                     n_components=2, n_splits=3)


def synthetic_cv_kfold_nipals_pls2_v1() -> dict[str, Any]:
    """16 samples, 6 features, 2 targets, 4-fold out-of-sample PLSRegression."""
    seed = 57
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(16, 3))
    X = np.column_stack([
        0.76 * latent[:, 0] + 0.22 * latent[:, 1],
        -0.41 * latent[:, 0] + 0.64 * latent[:, 2],
        0.35 * latent[:, 1] - 0.55 * latent[:, 2],
        0.48 * latent[:, 0] + 0.31 * latent[:, 1] - 0.16 * latent[:, 2],
        np.sin(np.linspace(0.0, 2.0 * np.pi, 16)),
        rng.standard_normal(size=16) * 0.06,
    ])
    Y = np.column_stack([
        1.18 * latent[:, 0] - 0.38 * latent[:, 2] + rng.standard_normal(size=16) * 0.04,
        -0.74 * latent[:, 1] + 0.52 * latent[:, 2] + rng.standard_normal(size=16) * 0.04,
    ])
    return _cross_validation_fixture("synthetic_cv_kfold_nipals_pls2_v1",
                                     seed=seed, X=X, Y=Y,
                                     n_components=2, n_splits=4)


def synthetic_component_cv_simpls_pls2_v1() -> dict[str, Any]:
    """18 samples, 6 features, 2 targets, SIMPLS CV for 1..3 components."""
    seed = 97
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(size=(18, 3))
    X = np.column_stack([
        0.81 * latent[:, 0] + 0.22 * latent[:, 1],
        -0.47 * latent[:, 0] + 0.58 * latent[:, 2],
        0.39 * latent[:, 1] - 0.42 * latent[:, 2],
        0.34 * latent[:, 0] + 0.27 * latent[:, 1] - 0.16 * latent[:, 2],
        np.linspace(-0.9, 0.9, 18),
        rng.standard_normal(size=18) * 0.05,
    ])
    Y = np.column_stack([
        1.12 * latent[:, 0] - 0.31 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
        -0.68 * latent[:, 1] + 0.57 * latent[:, 2] + rng.standard_normal(size=18) * 0.035,
    ])
    return _component_cv_fixture("synthetic_component_cv_simpls_pls2_v1",
                                 seed=seed,
                                 X=X,
                                 Y=Y,
                                 n_components_max=3,
                                 n_splits=3,
                                 solver="simpls")


def synthetic_power_tiny_pls1_v1() -> dict[str, Any]:
    """10 samples, 5 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=80)
    X = rng.standard_normal(size=(10, 5))
    true_w = np.array([0.34, -0.48, 0.27, 0.41, -0.18])
    Y = (X @ true_w + rng.standard_normal(size=10) * 0.025).reshape(-1, 1)
    return _power_fixture("synthetic_power_tiny_pls1_v1", seed=80,
                          X=X, Y=Y, n_components=2)


def synthetic_power_small_pls2_v1() -> dict[str, Any]:
    """15 samples, 7 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=81)
    X = rng.standard_normal(size=(15, 7))
    W = np.array([
        [0.36, -0.22, 0.16],
        [-0.28, 0.42, -0.20],
        [0.12, 0.24, 0.46],
        [0.44, -0.18, 0.08],
        [-0.32, 0.30, -0.36],
        [0.18, 0.38, 0.26],
        [0.40, -0.12, 0.32],
    ])
    Y = X @ W + rng.standard_normal(size=(15, 3)) * 0.03
    return _power_fixture("synthetic_power_small_pls2_v1", seed=81,
                          X=X, Y=Y, n_components=3)


def synthetic_randomized_svd_tiny_pls1_v1() -> dict[str, Any]:
    """10 samples, 5 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=90)
    X = rng.standard_normal(size=(10, 5))
    true_w = np.array([0.44, -0.26, 0.31, 0.36, -0.20])
    Y = (X @ true_w + rng.standard_normal(size=10) * 0.025).reshape(-1, 1)
    return _randomized_svd_fixture("synthetic_randomized_svd_tiny_pls1_v1",
                                   seed=90, X=X, Y=Y, n_components=2)


def synthetic_randomized_svd_small_pls2_v1() -> dict[str, Any]:
    """15 samples, 7 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=91)
    X = rng.standard_normal(size=(15, 7))
    W = np.array([
        [0.30, -0.25, 0.18],
        [-0.34, 0.40, -0.16],
        [0.16, 0.22, 0.44],
        [0.42, -0.20, 0.12],
        [-0.26, 0.34, -0.38],
        [0.20, 0.36, 0.24],
        [0.38, -0.10, 0.30],
    ])
    Y = X @ W + rng.standard_normal(size=(15, 3)) * 0.03
    return _randomized_svd_fixture("synthetic_randomized_svd_small_pls2_v1",
                                   seed=91, X=X, Y=Y, n_components=3)


def synthetic_canonical_tiny_pls1_v1() -> dict[str, Any]:
    """9 samples, 4 features, 1 target, n_components=1."""
    rng = np.random.default_rng(seed=100)
    X = rng.standard_normal(size=(9, 4))
    true_w = np.array([0.42, -0.30, 0.25, 0.36])
    Y = (X @ true_w + rng.standard_normal(size=9) * 0.025).reshape(-1, 1)
    return _canonical_fixture("synthetic_canonical_tiny_pls1_v1",
                              seed=100, X=X, Y=Y, n_components=1,
                              solver="nipals")


def synthetic_canonical_small_pls2_v1() -> dict[str, Any]:
    """14 samples, 6 features, 3 targets, n_components=2."""
    rng = np.random.default_rng(seed=101)
    X = rng.standard_normal(size=(14, 6))
    W = np.array([
        [0.36, -0.20, 0.18],
        [-0.28, 0.44, -0.15],
        [0.18, 0.24, 0.38],
        [0.08, -0.34, 0.16],
        [0.42, 0.10, -0.32],
        [-0.20, 0.36, 0.26],
    ])
    Y = X @ W + rng.standard_normal(size=(14, 3)) * 0.035
    return _canonical_fixture("synthetic_canonical_small_pls2_v1",
                              seed=101, X=X, Y=Y, n_components=2,
                              solver="nipals")


def synthetic_canonical_svd_tiny_pls1_v1() -> dict[str, Any]:
    """9 samples, 4 features, 1 target, n_components=1."""
    rng = np.random.default_rng(seed=102)
    X = rng.standard_normal(size=(9, 4))
    true_w = np.array([0.35, -0.45, 0.28, 0.22])
    Y = (X @ true_w + rng.standard_normal(size=9) * 0.025).reshape(-1, 1)
    return _canonical_fixture("synthetic_canonical_svd_tiny_pls1_v1",
                              seed=102, X=X, Y=Y, n_components=1,
                              solver="svd")


def synthetic_canonical_svd_small_pls2_v1() -> dict[str, Any]:
    """15 samples, 7 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=103)
    X = rng.standard_normal(size=(15, 7))
    W = np.array([
        [0.32, -0.18, 0.16],
        [-0.30, 0.42, -0.18],
        [0.14, 0.26, 0.40],
        [0.46, -0.16, 0.10],
        [-0.28, 0.32, -0.34],
        [0.18, 0.34, 0.28],
        [0.36, -0.12, 0.30],
    ])
    Y = X @ W + rng.standard_normal(size=(15, 3)) * 0.03
    return _canonical_fixture("synthetic_canonical_svd_small_pls2_v1",
                              seed=103, X=X, Y=Y, n_components=3,
                              solver="svd")


def synthetic_pls_da_binary_v1() -> dict[str, Any]:
    """18 samples, 6 features, 2 dummy targets, n_components=2."""
    rng = np.random.default_rng(seed=110)
    X = rng.standard_normal(size=(18, 6))
    W = np.array([
        [0.45, -0.20],
        [-0.30, 0.36],
        [0.24, 0.18],
        [0.12, -0.42],
        [0.38, 0.16],
        [-0.22, 0.30],
    ])
    logits = X @ W + rng.standard_normal(size=(18, 2)) * 0.05
    labels = np.argmax(logits, axis=1)
    labels[0] = 0
    labels[1] = 1
    Y = np.eye(2, dtype=np.float64)[labels]
    return _pls_da_fixture("synthetic_pls_da_binary_v1",
                           seed=110, X=X, Y=Y, n_components=2)


def synthetic_pls_da_multiclass_v1() -> dict[str, Any]:
    """21 samples, 7 features, 3 dummy targets, n_components=3."""
    rng = np.random.default_rng(seed=111)
    X = rng.standard_normal(size=(21, 7))
    W = np.array([
        [0.36, -0.16, 0.22],
        [-0.28, 0.42, -0.12],
        [0.18, 0.24, 0.38],
        [0.44, -0.20, 0.08],
        [-0.32, 0.26, -0.34],
        [0.16, 0.36, 0.28],
        [0.30, -0.10, 0.34],
    ])
    logits = X @ W + rng.standard_normal(size=(21, 3)) * 0.05
    labels = np.argmax(logits, axis=1)
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    Y = np.eye(3, dtype=np.float64)[labels]
    return _pls_da_fixture("synthetic_pls_da_multiclass_v1",
                           seed=111, X=X, Y=Y, n_components=3)


def synthetic_opls_tiny_pls1_v1() -> dict[str, Any]:
    """14 samples, 6 features, 1 target, 1 predictive + 1 orthogonal component."""
    rng = np.random.default_rng(seed=120)
    predictive = rng.standard_normal(size=(14, 1))
    orthogonal = rng.standard_normal(size=(14, 2))
    X = np.hstack([
        0.58 * predictive + 0.15 * orthogonal[:, [0]],
        -0.34 * predictive + 0.52 * orthogonal[:, [0]],
        0.26 * predictive - 0.48 * orthogonal[:, [1]],
        0.44 * predictive + 0.20 * orthogonal[:, [1]],
        -0.18 * predictive + 0.42 * orthogonal[:, [0]],
        0.31 * predictive - 0.30 * orthogonal[:, [1]],
    ])
    X += rng.standard_normal(size=X.shape) * 0.025
    Y = (1.25 * predictive[:, 0] + rng.standard_normal(size=14) * 0.03).reshape(-1, 1)
    return _opls_fixture("synthetic_opls_tiny_pls1_v1",
                         seed=120, X=X, Y=Y, n_components=2)


def synthetic_opls_small_pls1_v1() -> dict[str, Any]:
    """18 samples, 8 features, 1 target, 1 predictive + 2 orthogonal components."""
    rng = np.random.default_rng(seed=121)
    predictive = rng.standard_normal(size=(18, 1))
    orthogonal = rng.standard_normal(size=(18, 3))
    X = np.hstack([
        0.62 * predictive + 0.24 * orthogonal[:, [0]],
        -0.41 * predictive + 0.38 * orthogonal[:, [1]],
        0.28 * predictive - 0.45 * orthogonal[:, [2]],
        0.35 * predictive + 0.33 * orthogonal[:, [0]],
        -0.22 * predictive - 0.50 * orthogonal[:, [1]],
        0.18 * predictive + 0.42 * orthogonal[:, [2]],
        0.46 * predictive - 0.21 * orthogonal[:, [0]],
        -0.30 * predictive + 0.28 * orthogonal[:, [1]],
    ])
    X += rng.standard_normal(size=X.shape) * 0.025
    Y = (1.10 * predictive[:, 0] + rng.standard_normal(size=18) * 0.035).reshape(-1, 1)
    return _opls_fixture("synthetic_opls_small_pls1_v1",
                         seed=121, X=X, Y=Y, n_components=3)


def synthetic_opls_small_pls2_v1() -> dict[str, Any]:
    """20 samples, 8 features, 2 targets, one common predictive + 2 orthogonal components."""
    rng = np.random.default_rng(seed=123)
    predictive = rng.standard_normal(size=(20, 1))
    orthogonal = rng.standard_normal(size=(20, 3))
    X = np.hstack([
        0.57 * predictive + 0.28 * orthogonal[:, [0]],
        -0.37 * predictive + 0.41 * orthogonal[:, [1]],
        0.32 * predictive - 0.36 * orthogonal[:, [2]],
        0.48 * predictive + 0.17 * orthogonal[:, [0]],
        -0.25 * predictive - 0.43 * orthogonal[:, [1]],
        0.22 * predictive + 0.39 * orthogonal[:, [2]],
        0.40 * predictive - 0.24 * orthogonal[:, [0]],
        -0.33 * predictive + 0.21 * orthogonal[:, [1]],
    ])
    X += rng.standard_normal(size=X.shape) * 0.025
    Y = np.hstack([
        1.18 * predictive + rng.standard_normal(size=(20, 1)) * 0.035,
        -0.74 * predictive + rng.standard_normal(size=(20, 1)) * 0.035,
    ])
    return _opls_fixture("synthetic_opls_small_pls2_v1",
                         seed=123, X=X, Y=Y, n_components=3)


def synthetic_opls_da_binary_v1() -> dict[str, Any]:
    """20 samples, 7 features, one binary dummy target, 1 predictive + 2 orthogonal components."""
    rng = np.random.default_rng(seed=122)
    predictive = rng.standard_normal(size=(20, 1))
    orthogonal = rng.standard_normal(size=(20, 3))
    X = np.hstack([
        0.52 * predictive + 0.31 * orthogonal[:, [0]],
        -0.46 * predictive + 0.36 * orthogonal[:, [1]],
        0.25 * predictive - 0.41 * orthogonal[:, [2]],
        0.42 * predictive + 0.18 * orthogonal[:, [0]],
        -0.20 * predictive - 0.46 * orthogonal[:, [1]],
        0.33 * predictive + 0.25 * orthogonal[:, [2]],
        0.28 * predictive - 0.22 * orthogonal[:, [0]],
    ])
    X += rng.standard_normal(size=X.shape) * 0.03
    logits = predictive[:, 0] + rng.standard_normal(size=20) * 0.10
    labels = (logits > np.median(logits)).astype(np.float64)
    labels[0] = 0.0
    labels[1] = 1.0
    Y = labels.reshape(-1, 1)
    return _opls_da_fixture("synthetic_opls_da_binary_v1",
                            seed=122, X=X, Y=Y, n_components=3)


def synthetic_opls_da_multiclass_v1() -> dict[str, Any]:
    """24 samples, 8 features, 3 dummy targets, one common predictive + 2 orthogonal components."""
    rng = np.random.default_rng(seed=124)
    predictive = rng.standard_normal(size=(24, 1))
    orthogonal = rng.standard_normal(size=(24, 3))
    X = np.hstack([
        0.55 * predictive + 0.34 * orthogonal[:, [0]],
        -0.43 * predictive + 0.30 * orthogonal[:, [1]],
        0.29 * predictive - 0.40 * orthogonal[:, [2]],
        0.47 * predictive + 0.19 * orthogonal[:, [0]],
        -0.24 * predictive - 0.45 * orthogonal[:, [1]],
        0.35 * predictive + 0.23 * orthogonal[:, [2]],
        0.31 * predictive - 0.26 * orthogonal[:, [0]],
        -0.38 * predictive + 0.20 * orthogonal[:, [1]],
    ])
    X += rng.standard_normal(size=X.shape) * 0.03
    noisy_axis = predictive[:, 0] + rng.standard_normal(size=24) * 0.08
    cuts = np.quantile(noisy_axis, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(noisy_axis, cuts).astype(np.int64)
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    Y = np.eye(3, dtype=np.float64)[labels]
    return _opls_da_fixture("synthetic_opls_da_multiclass_v1",
                            seed=124, X=X, Y=Y, n_components=3,
                            algorithm="opls-da-multiclass",
                            reference="NumPy multivariate OPLS NIPALS recurrence on one-hot Y",
                            tolerance_table_row="pls4all-numpy-opls-da-multiclass")


def synthetic_kernel_tiny_pls1_v1() -> dict[str, Any]:
    """9 samples, 5 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=50)
    X = rng.standard_normal(size=(9, 5))
    true_w = np.array([0.55, -0.20, 0.35, -0.45, 0.15])
    Y = (X @ true_w + rng.standard_normal(size=9) * 0.025).reshape(-1, 1)
    return _kernel_fixture("synthetic_kernel_tiny_pls1_v1", seed=50,
                           X=X, Y=Y, n_components=2)


def synthetic_kernel_wide_pls2_v1() -> dict[str, Any]:
    """8 samples, 12 features, 2 targets, n_components=3."""
    rng = np.random.default_rng(seed=51)
    X = rng.standard_normal(size=(8, 12))
    W = rng.standard_normal(size=(12, 2)) * 0.25
    Y = X @ W + rng.standard_normal(size=(8, 2)) * 0.03
    return _kernel_fixture("synthetic_kernel_wide_pls2_v1", seed=51,
                           X=X, Y=Y, n_components=3)


def synthetic_wide_kernel_pls1_v1() -> dict[str, Any]:
    """7 samples, 18 features, 1 target, n_components=3."""
    rng = np.random.default_rng(seed=60)
    X = rng.standard_normal(size=(7, 18))
    true_w = rng.standard_normal(size=18) * 0.20
    Y = (X @ true_w + rng.standard_normal(size=7) * 0.02).reshape(-1, 1)
    return _wide_kernel_fixture("synthetic_wide_kernel_pls1_v1", seed=60,
                                X=X, Y=Y, n_components=3)


def synthetic_wide_kernel_pls2_v1() -> dict[str, Any]:
    """9 samples, 24 features, 3 targets, n_components=4."""
    rng = np.random.default_rng(seed=61)
    X = rng.standard_normal(size=(9, 24))
    W = rng.standard_normal(size=(24, 3)) * 0.18
    Y = X @ W + rng.standard_normal(size=(9, 3)) * 0.025
    return _wide_kernel_fixture("synthetic_wide_kernel_pls2_v1", seed=61,
                                X=X, Y=Y, n_components=4)


def synthetic_oscores_tiny_pls1_v1() -> dict[str, Any]:
    """11 samples, 5 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=70)
    X = rng.standard_normal(size=(11, 5))
    true_w = np.array([0.42, -0.35, 0.18, 0.50, -0.22])
    Y = (X @ true_w + rng.standard_normal(size=11) * 0.025).reshape(-1, 1)
    return _orthogonal_scores_fixture("synthetic_oscores_tiny_pls1_v1", seed=70,
                                      X=X, Y=Y, n_components=2)


def synthetic_oscores_small_pls2_v1() -> dict[str, Any]:
    """13 samples, 7 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=71)
    X = rng.standard_normal(size=(13, 7))
    W = np.array([
        [0.25, -0.35, 0.18],
        [-0.45, 0.20, -0.15],
        [0.32, 0.28, 0.40],
        [0.10, -0.42, 0.08],
        [0.38, 0.16, -0.30],
        [-0.18, 0.46, 0.22],
        [0.27, -0.05, 0.35],
    ])
    Y = X @ W + rng.standard_normal(size=(13, 3)) * 0.035
    return _orthogonal_scores_fixture("synthetic_oscores_small_pls2_v1", seed=71,
                                      X=X, Y=Y, n_components=3)


def synthetic_pcr_tiny_pls1_v1() -> dict[str, Any]:
    """10 samples, 4 features, 1 target, n_components=2."""
    rng = np.random.default_rng(seed=40)
    X = rng.standard_normal(size=(10, 4))
    true_w = np.array([0.30, -0.60, 0.15, 0.50])
    Y = (X @ true_w + rng.standard_normal(size=10) * 0.03).reshape(-1, 1)
    return _pcr_fixture("synthetic_pcr_tiny_pls1_v1", seed=40,
                        X=X, Y=Y, n_components=2)


def synthetic_pcr_small_pls2_v1() -> dict[str, Any]:
    """16 samples, 6 features, 3 targets, n_components=3."""
    rng = np.random.default_rng(seed=41)
    X = rng.standard_normal(size=(16, 6))
    W = np.array([
        [0.35, -0.20, 0.10],
        [-0.15, 0.50, -0.25],
        [0.40, 0.05, 0.30],
        [-0.45, -0.10, 0.20],
        [0.25, 0.35, -0.40],
        [0.10, -0.30, 0.45],
    ])
    Y = X @ W + rng.standard_normal(size=(16, 3)) * 0.04
    return _pcr_fixture("synthetic_pcr_small_pls2_v1", seed=41,
                        X=X, Y=Y, n_components=3)

"""Per-fixture producers. Each function returns a dict in the v1 schema."""

from __future__ import annotations

from typing import Any

import numpy as np
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
    if Y.shape[1] != 1:
        raise ValueError("OPLS reference currently supports exactly one target")

    Xk, x_mean, x_scale = _center_scale(X)
    Ys, y_mean, y_scale = _center_scale(Y)
    y = Ys[:, 0].copy()
    n, p = Xk.shape
    K = int(n_components)
    eps = np.finfo(np.float64).eps

    W = np.zeros((p, K), dtype=np.float64)
    P = np.zeros((p, K), dtype=np.float64)
    Q = np.zeros((1, K), dtype=np.float64)
    R = np.zeros((p, K), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)
    projection = np.eye(p, dtype=np.float64)

    def sign_normalize(
        weights: np.ndarray,
        scores: np.ndarray | None = None,
        loadings: np.ndarray | None = None,
        y_loading: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
        sign_idx = int(np.argmax(np.abs(weights)))
        if weights[sign_idx] >= 0.0:
            return weights, scores, loadings, y_loading
        weights = -weights
        if scores is not None:
            scores = -scores
        if loadings is not None:
            loadings = -loadings
        if y_loading is not None:
            y_loading = -y_loading
        return weights, scores, loadings, y_loading

    def predictive_component(
        Xwork: np.ndarray,
        component: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        weights = Xwork.T @ y
        w_norm = np.linalg.norm(weights)
        if w_norm <= eps:
            raise RuntimeError(f"OPLS predictive weights collapsed at component {component}")
        weights = weights / w_norm
        scores = Xwork @ weights
        t_ss = float(scores @ scores)
        if t_ss <= eps:
            raise RuntimeError(f"OPLS predictive score collapsed at component {component}")
        loadings = (Xwork.T @ scores) / t_ss
        y_loading = float((y @ scores) / t_ss)
        weights, scores, loadings, y_loading = sign_normalize(
            weights, scores, loadings, y_loading
        )
        assert scores is not None and loadings is not None and y_loading is not None
        return weights, scores, loadings, y_loading

    for orth in range(max(K - 1, 0)):
        pred_w, _pred_t, pred_p, _pred_q = predictive_component(Xk, orth + 1)
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

    pred_w, pred_t, pred_p, pred_q = predictive_component(Xk, 0)
    raw_pred = projection @ pred_w
    denom = float(pred_p @ pred_w)
    if abs(denom) <= eps:
        raise RuntimeError("OPLS predictive denominator collapsed")
    coef_std = (raw_pred.reshape(-1, 1) * pred_q) / denom
    coef = coef_std * (y_scale.reshape(1, -1) / x_scale.reshape(-1, 1))
    preds = y_mean.reshape(1, -1) + (X - x_mean.reshape(1, -1)) @ coef

    W[:, 0] = pred_w
    P[:, 0] = pred_p
    Q[0, 0] = pred_q
    R[:, 0] = raw_pred
    T[:, 0] = pred_t

    return {
        "coefficients":  {"shape": list(coef.shape),       "values": _flatten_rowmajor(coef)},
        "intercept":     {"shape": [1],                    "values": y_mean.astype(np.float64).tolist()},
        "x_mean":        {"shape": [p],                    "values": x_mean.astype(np.float64).tolist()},
        "x_scale":       {"shape": [p],                    "values": x_scale.astype(np.float64).tolist()},
        "y_mean":        {"shape": [1],                    "values": y_mean.astype(np.float64).tolist()},
        "y_scale":       {"shape": [1],                    "values": y_scale.astype(np.float64).tolist()},
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
                "algorithm":      "opls-da-binary",
                "reference":      "NumPy OPLS1 NIPALS recurrence on one-column dummy Y",
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
            "tolerance_table_row":  "pls4all-numpy-opls-da-binary",
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

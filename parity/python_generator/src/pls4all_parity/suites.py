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

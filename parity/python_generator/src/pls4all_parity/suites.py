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

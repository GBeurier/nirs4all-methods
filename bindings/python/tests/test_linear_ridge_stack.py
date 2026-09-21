"""Affine stack compression: independent staged predictions and native Ridge fits."""

import pickle

import numpy as np
import pytest
from n4m.ensemble import LinearRidgeStackRegressor, compress_linear_stack
from sklearn.base import clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def test_linear_ridge_stack_follows_sklearn_regressor_contract():
    assert is_regressor(LinearRidgeStackRegressor())


@pytest.mark.parametrize("targets", [1, 3])
def test_compression_includes_intercepts_and_multitarget(targets):
    rng = np.random.default_rng(332)
    X = rng.normal(size=(17, 23))
    base = rng.normal(size=(23, 7))
    bias = rng.normal(size=7)
    weights = rng.normal(size=(7, targets))
    meta = rng.normal(size=targets)
    expected = (
        np.column_stack([X @ base[:, j] + bias[j] for j in range(7)]) @ weights + meta
    )
    deployed = compress_linear_stack(base, bias, weights, meta)
    actual = deployed.predict(X)
    if targets == 1:
        expected = expected[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(deployed)).predict(X), expected, atol=1e-12
    )
    assert deployed.numeric_bytes == (23 * targets + targets) * 8


@pytest.mark.parametrize("shape", [(54, 15), (30, 60)])
def test_stack_against_independent_sklearn_fixed_alpha(shape):
    rng = np.random.default_rng(441)
    X = rng.normal(size=shape)
    y = X[:, 3] + rng.normal(scale=0.2, size=len(X))
    V = rng.normal(size=(11, shape[1]))
    folds = np.empty(len(X), dtype=np.int32)
    pairs = list(KFold(3, shuffle=True, random_state=16).split(X))
    for i, (_, valid) in enumerate(pairs):
        folds[valid] = i
    model = clone(
        LinearRidgeStackRegressor(
            operators=("identity",), alphas=(0.3,), meta_alphas=(0.7,), fold_ids=folds
        )
    ).fit(X, y)
    oof = np.empty((len(X), 1))
    for train, valid in pairs:
        oof[valid, 0] = (
            Ridge(alpha=0.3, solver="svd").fit(X[train], y[train]).predict(X[valid])
        )
    base = Ridge(alpha=0.3, solver="svd").fit(X, y)
    meta = Ridge(alpha=0.7, solver="svd").fit(oof, y)
    expected = meta.predict(base.predict(V)[:, None])
    np.testing.assert_allclose(model.oof_predictions_, oof, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(model.predict(V), expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        model.predict_uncompressed(V), expected, rtol=1e-8, atol=1e-9
    )
    deployed = model.export_affine()
    model.base_coefficients_[:] = 999
    np.testing.assert_allclose(deployed.predict(V), expected, rtol=1e-8, atol=1e-9)
    assert set(vars(deployed)) == {"coef_", "intercept_"}


def test_nonlinear_view_is_rejected():
    X = np.random.default_rng(2).normal(size=(20, 7))
    with pytest.raises((ValueError, RuntimeError)):
        LinearRidgeStackRegressor(operators=("snv",), alphas=(1,)).fit(X, X[:, 2])


def test_outer_validation_targets_cannot_influence_their_base_oof_predictions():
    rng = np.random.default_rng(721)
    X = rng.normal(size=(60, 63))
    y = X[:, 0] - X[:, 4] + rng.normal(size=60) * 0.2
    folds = np.arange(60, dtype=np.int32) % 3
    changed = y.copy()
    changed[folds == 1] += rng.normal(size=20) * 100
    options = {
        "operators": ("identity", ("detrend_poly", (1,))),
        "alphas": (0.01, 1.0, 100.0),
        "meta_alphas": (0.1, 10.0),
        "fold_ids": folds,
    }
    first = LinearRidgeStackRegressor(**options).fit(X, y)
    second = LinearRidgeStackRegressor(**options).fit(X, changed)
    np.testing.assert_array_equal(
        first.oof_predictions_[folds == 1], second.oof_predictions_[folds == 1]
    )
    for j in range(2):
        assert first.inner_base_alphas_[j][1] == second.inner_base_alphas_[j][1]


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_stack_coefficients_are_rejected(bad):
    base = np.ones((4, 2))
    base[0, 0] = bad
    with pytest.raises((ValueError, RuntimeError)):
        compress_linear_stack(base, [0, 0], [1, 1], [0])


@pytest.mark.parametrize(
    "operator", ["identity", ("detrend_poly", (1,)), ("gaussian", (1, 4, 0))]
)
def test_smaller_gram_route_preserves_pooled_cv_selection(operator):
    from n4m._impl.native import aom_chain_sweep_run
    from n4m.model_selection.aom_calibration import _fit_single_view_ridge

    rng = np.random.default_rng(731)
    X = rng.normal(size=(41, 65))
    y = X[:, 0] + rng.normal(size=41)
    # Unequal folds and differing response noise expose mean-vs-pooled RMSE drift.
    folds = np.repeat([0, 1, 2], [7, 13, 21]).astype(np.int32)
    y[folds == 0] += rng.normal(size=7) * 10
    alphas = np.logspace(-2, 3, 9)
    old = aom_chain_sweep_run(
        X,
        y,
        [[operator]],
        heads=("ridge",),
        ridge_lambdas=alphas,
        cv=3,
        fold_ids=folds,
        center_x=True,
        scale_x=False,
        center_y=True,
        scale_y=False,
    )
    new = _fit_single_view_ridge(X, y, operator, alphas, 3, folds)
    assert old["selected_param"] == new["selected_param"]
    np.testing.assert_allclose(
        old["input_coefficients"], new["input_coefficients"], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(old["intercept"], new["intercept"], rtol=2e-6, atol=2e-7)

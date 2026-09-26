"""Smoke + sklearn-compatibility tests for the Model-based and
MethodResult-based regressor wrappers shipped by ``pls4all.sklearn``.

Each class is exercised through:

* `fit(X, y)` returns `self` and sets `coef_` + `n_features_in_`
* `predict(X)` returns predictions with matching shape
* `pickle.dumps / loads` survives bit-exactly (predictions identical)
* `get_params / set_params` round-trip
* Pipeline composability with `StandardScaler`

Run from the repo root:

    N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so \
        PYTHONPATH=bindings/python/src \
        parity/python_generator/.venv/bin/python -m pytest \
        bindings/python/tests/test_sklearn_regressors.py -q
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pls4all.sklearn import (
    PCR,
    PLSSVD,
    BaggingPLSRegression,
    BoostingPLSRegression,
    ContinuumRegression,
    CPPLSRegression,
    DIPLSRegression,
    ECRegression,
    FusedSparsePLSRegression,
    MBPLSRegression,
    MIRPLSRegression,
    OPLSRegression,
    PLSCanonical,
    PLSRegression,
    RandomSubspacePLSRegression,
    RidgePLSRegression,
    RobustPLSRegression,
    SparsePLSRegression,
    SparseSimplsRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    beta = np.zeros(30)
    beta[[3, 11, 22, 25]] = [3.0, -2.0, 1.5, -1.0]
    y = X @ beta + 0.1 * rng.standard_normal(200)
    Y_multi = np.column_stack(
        [y, X[:, 0] * 2.0 + 0.1 * rng.standard_normal(200)]
    )
    return X.astype(np.float64), y.astype(np.float64), Y_multi.astype(np.float64)


REGRESSORS_SINGLE = [
    (PLSRegression, dict(n_components=5)),
    (OPLSRegression, dict(n_components=3)),
    (PCR, dict(n_components=5)),
    (SparsePLSRegression, dict(n_components=5, sparsity_lambda=0.05)),
    (SparseSimplsRegression, dict(n_components=5, sparsity_lambda=0.05)),
    (CPPLSRegression, dict(n_components=5, gamma=0.5)),
    (RobustPLSRegression, dict(n_components=5, huber_k=1.345)),
    (RidgePLSRegression, dict(n_components=5, ridge_lambda=0.5)),
    (ContinuumRegression, dict(n_components=5, tau=0.5)),
    (ECRegression, dict(n_components=5, alpha=0.5)),
    (MIRPLSRegression, dict(n_components=5)),
    (FusedSparsePLSRegression, dict(n_components=5, l1_lambda=0.05, fusion_lambda=0.05)),
    (BaggingPLSRegression, dict(n_components=5, n_estimators=5, seed=7)),
    (BoostingPLSRegression, dict(n_components=5, n_estimators=5, learning_rate=0.1)),
    (RandomSubspacePLSRegression, dict(n_components=5, n_estimators=5, features_per_subspace=10, seed=7)),
    (MBPLSRegression, dict(n_components=5, block_sizes=[15, 15])),
]

REGRESSORS_MULTI = [
    (PLSCanonical, dict(n_components=2)),
    (PLSSVD, dict(n_components=2)),
]


@pytest.mark.parametrize("cls,kwargs", REGRESSORS_SINGLE)
def test_regressor_single_output_fit_predict(cls, kwargs, regression_data):
    X, y, _ = regression_data
    m = cls(**kwargs).fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (X.shape[0],)
    assert m.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("cls,kwargs", REGRESSORS_MULTI)
def test_regressor_multi_output_fit_predict(cls, kwargs, regression_data):
    X, _, Y = regression_data
    m = cls(**kwargs).fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == (X.shape[0], Y.shape[1])
    assert m.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("cls,kwargs",
                          REGRESSORS_SINGLE + REGRESSORS_MULTI)
def test_regressor_pickle_bitexact(cls, kwargs, regression_data):
    X, y, Y = regression_data
    target = y if (cls, kwargs) in REGRESSORS_SINGLE else Y
    m = cls(**kwargs).fit(X, target)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m.predict(X), m2.predict(X))


def test_dipls_regression_fit_predict(regression_data):
    X, y, _ = regression_data
    rng = np.random.default_rng(7)
    X_target = X + 0.01 * rng.standard_normal(X.shape)
    m = DIPLSRegression(n_components=5, di_lambda=1.0).fit(
        X, y, X_target=X_target)
    preds = m.predict(X)
    assert preds.shape == (X.shape[0],)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(preds, m2.predict(X))


@pytest.mark.parametrize("cls,kwargs", REGRESSORS_SINGLE)
def test_regressor_in_pipeline(cls, kwargs, regression_data):
    X, y, _ = regression_data
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", cls(**kwargs)),
    ])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize("cls,kwargs",
                          REGRESSORS_SINGLE + REGRESSORS_MULTI)
def test_regressor_get_params_roundtrip(cls, kwargs):
    m = cls(**kwargs)
    params = m.get_params()
    m2 = cls(**{k: v for k, v in params.items()})
    assert m2.get_params() == params


# -----------------------------------------------------------------
# Bit-exact wrapper-vs-tier1 parity for MethodResult regressors
# -----------------------------------------------------------------

def _raw_method_result_predict(fn, n_components, X, y, *, X_predict=None,
                               **fn_kwargs):
    """Run the tier-1 *_fit and replay the prediction math the wrapper
    is supposed to use: ``(X - x_mean) @ coef + y_mean``."""
    import pls4all
    from pls4all import Algorithm, Deflation, Solver
    ctx = pls4all.Context()
    cfg = pls4all.Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = int(n_components)
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    cfg.tol = 1e-6
    cfg.max_iter = 500
    try:
        res = fn(ctx, cfg, X, y.reshape(X.shape[0], -1), **fn_kwargs)
    finally:
        cfg.close()
    coef = np.asarray(res.matrix("coefficients"), dtype=np.float64)
    x_mean = np.asarray(res.matrix("x_mean"), dtype=np.float64).ravel()
    y_mean = np.asarray(res.matrix("y_mean"), dtype=np.float64).ravel()
    preds = ((X if X_predict is None else X_predict) - x_mean) @ coef + y_mean
    return preds.ravel() if y.ndim == 1 else preds


@pytest.mark.parametrize("cls,params,fit_name,fit_params", [
    (FusedSparsePLSRegression, {"l1_lambda": 0.08, "fusion_lambda": 0.2},
     "fused_sparse_pls_fit", {"l1_lambda": 0.08, "fusion_lambda": 0.2}),
    (BaggingPLSRegression, {"n_estimators": 7, "seed": 13},
     "bagging_pls_fit", {"n_estimators": 7, "seed": 13}),
    (BoostingPLSRegression, {"n_estimators": 7, "learning_rate": 0.3},
     "boosting_pls_fit", {"n_estimators": 7, "learning_rate": 0.3}),
    (RandomSubspacePLSRegression,
     {"n_estimators": 7, "features_per_subspace": 11, "seed": 13},
     "random_subspace_pls_fit",
     {"n_estimators": 7, "features_per_subspace": 11, "seed": 13}),
])
def test_affine_extra_heldout_matches_native_coefficients(
        cls, params, fit_name, fit_params, regression_data):
    import pls4all
    from pls4all.migration import export_linear_predictor_n4mm

    X, y, _ = regression_data
    X_heldout = X[:9] + 0.031
    estimator = cls(n_components=3, **params).fit(X, y)
    expected = _raw_method_result_predict(
        getattr(pls4all, fit_name), 3, X, y,
        X_predict=X_heldout, **fit_params)
    np.testing.assert_allclose(estimator.predict(X_heldout), expected,
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        estimator.predict(X_heldout),
        pickle.loads(pickle.dumps(estimator)).predict(X_heldout))
    payload = export_linear_predictor_n4mm(
        np.asarray(estimator.coef_, dtype=np.float64).reshape(-1, 1).tolist(),
        [float(estimator.intercept_)],
        source_training_samples=X.shape[0])
    with pls4all.Context() as context, pls4all.Model.from_bytes(context, payload) as model:
        np.testing.assert_allclose(
            model.predict(context, X_heldout).ravel(),
            estimator.predict(X_heldout), rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls,params,fit_name", [
    (FusedSparsePLSRegression, {"l1_lambda": 0.05, "fusion_lambda": 0.05},
     "fused_sparse_pls_fit"),
    (BaggingPLSRegression, {"n_estimators": 5, "seed": 7}, "bagging_pls_fit"),
    (BoostingPLSRegression, {"n_estimators": 5, "learning_rate": 0.1},
     "boosting_pls_fit"),
    (RandomSubspacePLSRegression,
     {"n_estimators": 5, "features_per_subspace": 10, "seed": 7},
     "random_subspace_pls_fit"),
])
def test_affine_extra_multi_target_heldout_and_n4mm(
        cls, params, fit_name, regression_data):
    import pls4all
    from pls4all.migration import export_linear_predictor_n4mm

    X, _, Y = regression_data
    X_heldout = X[:7] - 0.019
    estimator = cls(n_components=3, **params).fit(X, Y)
    native_expected = _raw_method_result_predict(
        getattr(pls4all, fit_name), 3, X, Y,
        X_predict=X_heldout, **params)
    np.testing.assert_allclose(estimator.predict(X_heldout), native_expected,
                               rtol=0, atol=1e-12)
    payload = export_linear_predictor_n4mm(
        np.asarray(estimator.coef_, dtype=np.float64).T.tolist(),
        np.asarray(estimator.intercept_, dtype=np.float64).tolist(),
        source_training_samples=X.shape[0])
    with pls4all.Context() as context, pls4all.Model.from_bytes(context, payload) as model:
        np.testing.assert_allclose(model.predict(context, X_heldout),
                                   estimator.predict(X_heldout), rtol=0, atol=1e-12)


def test_boosting_pls_refuses_rate_above_native_bound(regression_data):
    X, y, _ = regression_data
    with pytest.raises(ValueError, match="learning_rate"):
        BoostingPLSRegression(learning_rate=1.2).fit(X, y)


def test_sparse_simpls_wrapper_bitexact(regression_data):
    """Wrapper.predict(X) must be bit-exact with the C predict math
    applied to tier 1's MethodResult."""
    import pls4all
    X, y, _ = regression_data
    wrapper = SparseSimplsRegression(n_components=5, sparsity_lambda=0.05)
    wrapper.fit(X, y)
    preds_wrapper = wrapper.predict(X)
    preds_raw = _raw_method_result_predict(
        pls4all.sparse_simpls_fit, 5, X, y, sparsity_lambda=0.05)
    assert np.array_equal(preds_wrapper, preds_raw)


def test_ec_regression_wrapper_bitexact(regression_data):
    import pls4all
    X, y, _ = regression_data
    wrapper = ECRegression(n_components=5, alpha=0.5).fit(X, y)
    preds_wrapper = wrapper.predict(X)
    preds_raw = _raw_method_result_predict(
        pls4all.ecr_fit, 5, X, y, alpha=0.5)
    assert np.array_equal(preds_wrapper, preds_raw)


def test_mir_pls_wrapper_bitexact(regression_data):
    import pls4all
    X, y, _ = regression_data
    wrapper = MIRPLSRegression(n_components=5).fit(X, y)
    preds_wrapper = wrapper.predict(X)
    preds_raw = _raw_method_result_predict(
        pls4all.mir_pls_fit, 5, X, y)
    assert np.array_equal(preds_wrapper, preds_raw)


@pytest.mark.parametrize(
    ("cls", "kwargs", "fit_name", "fit_kwargs"),
    [
        (RobustPLSRegression, {"huber_k": 1.4, "max_irls_iter": 8},
         "robust_pls_fit", {"huber_k": 1.4, "max_irls_iter": 8}),
        (RidgePLSRegression, {"ridge_lambda": 0.3},
         "ridge_pls_fit", {"ridge_lambda": 0.3}),
        (ContinuumRegression, {"tau": 0.4},
         "continuum_regression_fit", {"tau": 0.4}),
    ],
)
def test_affine_method_result_predicts_held_out(
    cls, kwargs, fit_name, fit_kwargs, regression_data
):
    import pls4all

    X, y, _ = regression_data
    held_out = X[:17].copy() + 0.025
    model = cls(n_components=3, **kwargs).fit(X, y)
    expected = _raw_method_result_predict(
        getattr(pls4all, fit_name), 3, X, y,
        X_predict=held_out, **fit_kwargs)
    assert np.allclose(model.predict(held_out), expected, rtol=0, atol=1e-10)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict(held_out), model.predict(held_out))


def test_method_result_regressor_failed_refit_preserves_state(regression_data):
    """Codex catch: a failed refit on a fitted MethodResult regressor
    must leave the prior fitted state intact (n_features_in_, coef_,
    predict path)."""
    X, y, _ = regression_data
    m = SparseSimplsRegression(n_components=5, sparsity_lambda=0.05).fit(X, y)
    preds_orig = m.predict(X)
    orig_coef = m.coef_.copy()
    orig_n_features = m.n_features_in_
    # Refit with a NaN — must raise without corrupting state.
    bad_X = X.copy()
    bad_X[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.fit(bad_X, y)
    # State preserved.
    assert m.n_features_in_ == orig_n_features
    assert np.array_equal(m.coef_, orig_coef)
    assert np.array_equal(m.predict(X), preds_orig)


def test_mb_pls_wrapper_predict_matches_in_sample(regression_data):
    """MB-PLS stores its in-sample predictions directly. Wrapper.predict
    on X_train must match res.matrix('predictions')."""
    import pls4all
    from pls4all import Algorithm, Deflation, Solver
    X, y, _ = regression_data
    block_sizes = np.array([15, 15], dtype=np.int64)
    wrapper = MBPLSRegression(n_components=5,
                                 block_sizes=block_sizes.tolist()).fit(X, y)
    preds_wrapper = wrapper.predict(X)
    ctx = pls4all.Context()
    cfg = pls4all.Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.NIPALS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = 5
    cfg.center_x = True
    cfg.center_y = True
    try:
        res = pls4all.mb_pls_fit(ctx, cfg, X, y.reshape(-1, 1), block_sizes)
    finally:
        cfg.close()
    preds_raw = np.asarray(res.matrix("predictions"),
                              dtype=np.float64).ravel()
    # MB-PLS coef is in original space; intercept folds y_mean in. The
    # Python predict must therefore agree bit-exactly with the in-sample
    # C predictions.
    assert np.allclose(preds_wrapper, preds_raw, atol=1e-10, rtol=1e-10)

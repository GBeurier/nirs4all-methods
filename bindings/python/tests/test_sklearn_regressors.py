"""Smoke + sklearn-compatibility tests for the Model-based and
MethodResult-based regressor wrappers shipped by ``pls4all.sklearn``.

Each class is exercised through:

* `fit(X, y)` returns `self` and sets `coef_` + `n_features_in_`
* `predict(X)` returns predictions with matching shape
* `pickle.dumps / loads` survives bit-exactly (predictions identical)
* `get_params / set_params` round-trip
* Pipeline composability with `StandardScaler`

Run from the repo root:

    P4A_LIB_PATH=$PWD/build/dev-release/cpp/src/libp4a.so \
        PYTHONPATH=bindings/python/src \
        parity/python_generator/.venv/bin/python -m pytest \
        bindings/python/tests/test_sklearn_regressors.py -q
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pls4all.sklearn import (
    CPPLSRegression,
    DIPLSRegression,
    ECRegression,
    MBPLSRegression,
    MIRPLSRegression,
    OPLSRegression,
    PCR,
    PLSCanonical,
    PLSRegression,
    PLSSVD,
    SparsePLSRegression,
    SparseSimplsRegression,
)


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
    (ECRegression, dict(n_components=5, alpha=0.5)),
    (MIRPLSRegression, dict(n_components=5)),
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

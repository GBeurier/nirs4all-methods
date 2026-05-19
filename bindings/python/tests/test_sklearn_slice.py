"""Parity, sklearn-compatibility and pickling tests for the sklearn slice.

Run from the repo root:

    P4A_LIB_PATH=$PWD/build/dev-release/cpp/src/libp4a.so \
        PYTHONPATH=bindings/python/src \
        parity/python_generator/.venv/bin/python -m pytest \
        bindings/python/tests/test_sklearn_slice.py -v
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pls4all
from pls4all import Config, Context, Model
from pls4all._types import Algorithm, Deflation, Solver
from pls4all.sklearn import (
    OPLSRegression,
    PLSDAClassifier,
    PLSRegression,
    VIPSelector,
)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    beta = np.zeros(30)
    beta[[2, 7, 12, 20]] = [3.0, -2.0, 1.5, -1.0]
    y = X @ beta + 0.1 * rng.standard_normal(200)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 30))
    beta = np.zeros(30)
    beta[[2, 7, 12, 20]] = [3.0, -2.0, 1.5, -1.0]
    y = (X @ beta > 0).astype(np.int64)
    return X.astype(np.float64), y


# -----------------------------------------------------------------
# Bit-exact parity: wrapper.predict(X) must match raw_fit_predict(X)
# -----------------------------------------------------------------

def _raw_fit_predict(X, y, *, algorithm, solver, deflation, n_components,
                       center_x=True, scale_x=True,
                       center_y=True, scale_y=False):
    """Reference path: pls4all.Model.fit + Model.predict, no sklearn layer."""
    ctx = Context()
    cfg = Config()
    cfg.algorithm = algorithm
    cfg.solver = solver
    cfg.deflation = deflation
    cfg.n_components = n_components
    cfg.center_x = center_x
    cfg.scale_x = scale_x
    cfg.center_y = center_y
    cfg.scale_y = scale_y
    try:
        model = Model.fit(ctx, cfg, X, y)
        preds = model.predict(ctx, X)
    finally:
        cfg.close()
    return preds


def test_pls_regression_bit_exact_against_raw(regression_data):
    X, y = regression_data
    wrapper = PLSRegression(n_components=5).fit(X, y)
    preds_wrapped = wrapper.predict(X)
    preds_raw = _raw_fit_predict(
        X, y[:, None],
        algorithm=Algorithm.PLS_REGRESSION,
        solver=Solver.SIMPLS,
        deflation=Deflation.REGRESSION,
        n_components=5,
    )
    # Wrapper returns 1-D when y was 1-D (sklearn convention).
    assert np.array_equal(preds_wrapped, preds_raw.ravel())


def test_opls_regression_bit_exact_against_raw(regression_data):
    X, y = regression_data
    wrapper = OPLSRegression(n_components=3).fit(X, y)
    preds_wrapped = wrapper.predict(X)
    preds_raw = _raw_fit_predict(
        X, y[:, None],
        algorithm=Algorithm.OPLS,
        solver=Solver.NIPALS,
        deflation=Deflation.ORTHOGONAL,
        n_components=3,
    )
    assert np.array_equal(preds_wrapped, preds_raw.ravel())


def test_plsda_classifier_bit_exact_decision_function(classification_data):
    X, y = classification_data
    wrapper = PLSDAClassifier(n_components=3).fit(X, y)
    scores_wrapped = wrapper.decision_function(X)
    # Reproduce the raw path: one-hot Y then PLS_DA fit.
    Y_dummy = np.zeros((y.shape[0], 2), dtype=np.float64)
    Y_dummy[np.arange(y.shape[0]), y] = 1.0
    scores_raw = _raw_fit_predict(
        X, Y_dummy,
        algorithm=Algorithm.PLS_DA,
        solver=Solver.SIMPLS,
        deflation=Deflation.REGRESSION,
        n_components=3,
    )
    assert np.array_equal(scores_wrapped, scores_raw)


def test_vip_selector_bit_exact_against_raw(regression_data):
    X, y = regression_data
    wrapper = VIPSelector(top_k=5, n_components=4).fit(X, y)
    # Raw: fit Model + variable_select_rank.
    ctx = Context()
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = 4
    cfg.center_x = cfg.scale_x = cfg.center_y = True
    cfg.scale_y = False
    cfg.store_scores = True
    try:
        model = Model.fit(ctx, cfg, X, y[:, None])
        res = pls4all.variable_select_rank(ctx, model, X, method=0, top_k=5)
    finally:
        cfg.close()
    raw_scores = res.matrix("scores").ravel()
    raw_indices = sorted(np.asarray(res.vector_int64("selected_indices"),
                                      dtype=np.int64).tolist())
    assert np.array_equal(wrapper.vip_scores_, raw_scores)
    assert sorted(wrapper.selected_indices_.tolist()) == raw_indices
    model.close()


# -----------------------------------------------------------------
# Sklearn estimator contract: get_params / set_params / clone
# -----------------------------------------------------------------

def test_pls_regression_get_params_roundtrip(regression_data):
    X, y = regression_data
    m = PLSRegression(n_components=4, solver="simpls",
                        scale_x=False, store_scores=True)
    params = m.get_params()
    assert params["n_components"] == 4
    assert params["solver"] == "simpls"
    assert params["scale_x"] is False
    m2 = PLSRegression().set_params(**params)
    assert m2.get_params() == params


def test_plsda_classifier_get_params_roundtrip():
    m = PLSDAClassifier(n_components=4, solver="simpls")
    assert "n_components" in m.get_params()
    PLSDAClassifier(**m.get_params())


def test_vip_selector_get_params_roundtrip():
    m = VIPSelector(top_k=10, n_components=5, solver="simpls")
    p = m.get_params()
    assert p["top_k"] == 10 and p["n_components"] == 5
    VIPSelector(**p)


# -----------------------------------------------------------------
# Pipeline + GridSearchCV smoke tests
# -----------------------------------------------------------------

def test_pls_regression_in_pipeline(regression_data):
    X, y = regression_data
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=3)),
    ])
    pipe.fit(X, y)
    score = pipe.score(X, y)
    assert score > 0.5  # any signal — synthetic dataset is easy.


def test_vip_selector_in_pipeline(regression_data):
    X, y = regression_data
    pipe = Pipeline([
        ("select", VIPSelector(top_k=5, n_components=4)),
        ("pls", PLSRegression(n_components=3)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (X.shape[0],)


def test_pls_regression_gridsearchcv(regression_data):
    X, y = regression_data
    grid = GridSearchCV(
        PLSRegression(),
        param_grid={"n_components": [2, 3, 5, 8]},
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        n_jobs=1,
    )
    grid.fit(X, y)
    assert grid.best_params_["n_components"] in {2, 3, 5, 8}
    assert grid.best_score_ > 0.0


# -----------------------------------------------------------------
# Pickling round-trip: bit-exact predictions before/after pickle
# -----------------------------------------------------------------

def test_pls_regression_pickle_bitexact(regression_data):
    X, y = regression_data
    m = PLSRegression(n_components=5).fit(X, y)
    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)
    assert np.array_equal(m.predict(X), m2.predict(X))
    assert np.array_equal(m.coef_, m2.coef_)


def test_opls_regression_pickle_bitexact(regression_data):
    X, y = regression_data
    m = OPLSRegression(n_components=3).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m.predict(X), m2.predict(X))


def test_plsda_classifier_pickle_bitexact(classification_data):
    X, y = classification_data
    m = PLSDAClassifier(n_components=3).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m.predict(X), m2.predict(X))
    assert np.array_equal(m.decision_function(X), m2.decision_function(X))


def test_vip_selector_pickle_bitexact(regression_data):
    X, y = regression_data
    sel = VIPSelector(top_k=5, n_components=4).fit(X, y)
    sel2 = pickle.loads(pickle.dumps(sel))
    assert np.array_equal(sel.support_, sel2.support_)
    assert np.array_equal(sel.vip_scores_, sel2.vip_scores_)
    assert np.array_equal(sel.transform(X), sel2.transform(X))


def test_pls_regression_deepcopy_bitexact(regression_data):
    X, y = regression_data
    m = PLSRegression(n_components=5).fit(X, y)
    m2 = copy.deepcopy(m)
    assert np.array_equal(m.predict(X), m2.predict(X))


# -----------------------------------------------------------------
# Regression tests for codex-caught edge cases
# -----------------------------------------------------------------

def test_pls_regression_accepts_python_list_y(regression_data):
    """Codex catch: when y comes in as a Python list (no .ndim attr),
    predictions must still come back 1-D — not (n,1)."""
    X, y = regression_data
    m = PLSRegression(n_components=3).fit(X, y.tolist())
    preds = m.predict(X)
    assert preds.ndim == 1, f"expected 1-D predictions, got shape {preds.shape}"
    assert preds.shape == (X.shape[0],)


def test_pls_regression_2d_y_returns_2d_predictions(regression_data):
    """Mirror: when y comes in as (n,1), predictions stay (n,1)."""
    X, y = regression_data
    m = PLSRegression(n_components=3).fit(X, y.reshape(-1, 1))
    preds = m.predict(X)
    assert preds.shape == (X.shape[0], 1)


def test_predict_rejects_wrong_feature_count(regression_data):
    """check_array via _check_X_p4a enforces n_features_in_."""
    X, y = regression_data
    m = PLSRegression(n_components=3).fit(X, y)
    bad_X = X[:, :10]
    with pytest.raises(ValueError, match="features"):
        m.predict(bad_X)


def test_fit_rejects_nan_in_X(regression_data):
    X, y = regression_data
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        PLSRegression(n_components=3).fit(X_bad, y)


def test_plsda_one_class_does_not_corrupt_state(regression_data):
    """Codex catch: a failed fit with only one class must not leave
    `classes_` set, which would let `check_is_fitted` pass falsely."""
    X, _ = regression_data
    y = np.zeros(X.shape[0], dtype=int)  # single class
    m = PLSDAClassifier(n_components=3)
    with pytest.raises(ValueError, match="at least 2 classes"):
        m.fit(X, y)
    # Calling predict should still raise NotFittedError, not crash.
    from sklearn.exceptions import NotFittedError
    with pytest.raises((NotFittedError, AttributeError)):
        m.predict(X)


def test_vip_selector_rejects_invalid_top_k(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="top_k"):
        VIPSelector(top_k=0, n_components=3).fit(X, y)
    with pytest.raises(ValueError, match="top_k"):
        VIPSelector(top_k=X.shape[1] + 1, n_components=3).fit(X, y)


def test_set_params_after_fit_does_not_refit_silently(regression_data):
    """Documented contract: set_params on a fitted estimator does NOT
    invalidate the fit. Same behaviour as every sklearn estimator."""
    X, y = regression_data
    m = PLSRegression(n_components=5).fit(X, y)
    preds_before = m.predict(X)
    m.set_params(n_components=2)  # would change fit, but we haven't refit
    preds_after = m.predict(X)
    assert np.array_equal(preds_before, preds_after)
    assert m.coef_.shape[1] == X.shape[1]  # still 5-component fit's coefs

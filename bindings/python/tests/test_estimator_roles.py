# SPDX-License-Identifier: CECILL-2.1
"""Generic estimator roles (ABI 2.13): Python facade over n4m_estimator_*."""

from __future__ import annotations

import ctypes
import pickle

import numpy as np
import pytest
from n4m import roles
from n4m._errors import N4MError
from n4m._ffi import lib
from n4m._impl import native
from n4m._types import MethodInfoV1
from n4m.roles._base import _REGISTRY
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

N_FEATURES = 12


@pytest.fixture(scope="module")
def data():
    # Two dominant latent factors carry the response, as in spectra.
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(48, 2)) * np.array([3.0, 2.0])
    loadings = rng.normal(size=(2, N_FEATURES))
    X = scores @ loadings + 0.1 * rng.normal(size=(48, N_FEATURES))
    y = scores[:, 0] - 0.5 * scores[:, 1] + 0.05 * rng.normal(size=48)
    X_target = (
        rng.normal(size=(30, 2)) @ loadings
        + 0.3
        + 0.1 * rng.normal(size=(30, N_FEATURES))
    )
    return X[:36], y[:36], X[36:], X_target, y[36:]


def manifest_estimators() -> set[str]:
    count = ctypes.c_int32()
    assert lib.n4m_method_count(ctypes.byref(count)) == 0
    ids = set()
    for i in range(count.value):
        info = MethodInfoV1()
        info.struct_size = ctypes.sizeof(info)
        assert lib.n4m_method_info_v1(i, ctypes.addressof(info)) == 0
        if info.kind == 1:
            ids.add(info.method_id.decode())
    return ids


def build(cls):
    """Instance plus the fit inputs its method requires."""
    params = {"mode_j": 3, "mode_k": 4} if cls is roles.NPLS else {}
    return cls(**params)


def fit_kwargs(cls, X_target):
    return {
        roles.GroupSparsePLS: {"feature_groups": np.arange(N_FEATURES) // 4},
        roles.MBPLS: {"blocks": [4, 4, 4]},
        roles.DIPLS: {"X_target": X_target},
    }.get(cls, {})


ALL = sorted(_REGISTRY.values(), key=lambda c: c._method_id)
REGRESSORS = [c for c in ALL if issubclass(c, roles.NativeRegressor)]
SELECTORS = [c for c in ALL if issubclass(c, roles.NativeSelector)]


def test_generated_classes_match_native_manifest():
    assert set(_REGISTRY) == manifest_estimators()


ROLE_BIT = {
    roles.NativeTransformer: 1 << 0,
    roles.NativeRegressor: 1 << 1,
    roles.NativeSelector: 1 << 3,
}


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_classes_expose_exactly_their_role_interfaces(cls):
    declared = roles.method_info(cls._method_id).roles
    for base, bit in ROLE_BIT.items():
        assert issubclass(cls, base) == bool(declared & bit)
    if not issubclass(cls, (roles.NativeTransformer, roles.NativeSelector)):
        assert not hasattr(cls, "transform")
    if not issubclass(cls, roles.NativeRegressor):
        assert not hasattr(cls, "predict")


@pytest.mark.parametrize("cls", REGRESSORS, ids=lambda c: c.__name__)
def test_fit_predict_roundtrip(cls, data):
    X, y, X_test, X_target, y_test = data
    est = build(cls).fit(X, y, **fit_kwargs(cls, X_target))
    pred = est.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y_test)[0, 1] > 0.9

    restored = roles.NativeEstimator.from_n4me(est.to_n4me())
    assert type(restored) is cls
    assert restored.get_params() == est.get_params()
    np.testing.assert_array_equal(restored.predict(X_test), pred)
    assert restored.to_n4me() == est.to_n4me()

    np.testing.assert_array_equal(pickle.loads(pickle.dumps(est)).predict(X_test), pred)
    assert clone(est).get_params() == est.get_params()
    if isinstance(est, roles.NativeTransformer):
        scores = est.transform(X_test)
        assert scores.shape[0] == X_test.shape[0]
        np.testing.assert_array_equal(restored.transform(X_test), scores)


@pytest.mark.parametrize("cls", REGRESSORS, ids=lambda c: c.__name__)
def test_unused_and_missing_inputs_are_refused(cls, data):
    X, y, _, X_target, _ = data
    with pytest.raises(N4MError, match="not used by this method"):
        build(cls).fit(X, y, groups=np.zeros(X.shape[0]), **fit_kwargs(cls, X_target))
    if fit_kwargs(cls, X_target):
        with pytest.raises(N4MError, match="missing required fit input"):
            build(cls).fit(X, y)


def test_sklearn_cross_validation(data):
    X, y, _, _, _ = data
    scores = cross_val_score(roles.PLSRegression(n_components=3), X, y, cv=3)
    assert np.all(scores > 0.9)


def test_invalid_parameters_are_refused(data):
    X, y, _, _, _ = data
    with pytest.raises(ValueError, match="solver"):
        roles.PLSRegression(solver="bogus").fit(X, y)
    with pytest.raises(ValueError, match="n_components"):
        roles.CPPLS(n_components=0).fit(X, y)
    with pytest.raises(N4MError, match="mode_j"):
        roles.NPLS().fit(X, y)
    with pytest.raises(N4MError):
        roles.PLSRegression().predict(X)


def affine_reference(result: dict, X: np.ndarray) -> np.ndarray:
    coef = np.asarray(result["coefficients"])
    if "intercept" in result:
        return (X @ coef + np.asarray(result["intercept"]).reshape(1, -1)).ravel()
    return (
        (X - np.asarray(result["x_mean"]).reshape(1, -1)) @ coef
        + np.asarray(result["y_mean"]).reshape(1, -1)
    ).ravel()


# The generic estimators reproduce the existing n4m Python entry points with
# the same (manifest) defaults.
REFERENCES = [
    (roles.CPPLS, lambda X, y, t: native.cppls(X, y)),
    (roles.RobustPLS, lambda X, y, t: native.robust_pls(X, y)),
    (roles.RidgePLS, lambda X, y, t: native.ridge_pls(X, y)),
    (roles.ContinuumRegression, lambda X, y, t: native.continuum_regression(X, y)),
    (roles.ECR, lambda X, y, t: native.ecr(X, y)),
    (roles.Ridge, lambda X, y, t: native.ridge(X, y)),
    (roles.DIPLS, lambda X, y, t: native.di_pls(X, y, X_target=t)),
]


@pytest.mark.parametrize(
    "cls,reference", REFERENCES, ids=lambda v: getattr(v, "__name__", "")
)
def test_matches_n4m_reference(cls, reference, data):
    X, y, X_test, X_target, _ = data
    est = cls().fit(X, y, **fit_kwargs(cls, X_target))
    expected = affine_reference(reference(X, y, X_target), X_test)
    np.testing.assert_allclose(est.predict(X_test), expected, rtol=1e-10, atol=1e-10)


def test_pcr_matches_n4m_reference(data):
    X, y, X_test, _, _ = data
    est = roles.PCR(n_components=3).fit(X, y)
    ref = native.pcr(X, y, n_components=3)
    np.testing.assert_allclose(
        est.predict(X_test), affine_reference(ref, X_test), rtol=1e-10, atol=1e-10
    )


def test_weighted_pls_matches_n4m_reference(data):
    X, y, X_test, _, _ = data
    weights = np.linspace(0.5, 2.0, X.shape[0])
    est = roles.WeightedPLS().fit(X, y, sample_weight=weights)
    ref = native.weighted_pls(X, y, sample_weights=weights)
    np.testing.assert_allclose(
        est.predict(X_test), affine_reference(ref, X_test), rtol=1e-10, atol=1e-10
    )
    unweighted = roles.WeightedPLS().fit(X, y)
    np.testing.assert_allclose(
        unweighted.predict(X_test),
        affine_reference(native.weighted_pls(X, y), X_test),
        rtol=1e-10,
        atol=1e-10,
    )


def test_cross_language_fixture_states_replay():
    """The shared N4ME fixture (also replayed by R and JS) still predicts identically."""
    import base64
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[3]
        / "parity"
        / "fixtures"
        / "estimator_roles_n4me.json"
    )
    doc = json.loads(fixture.read_text(encoding="utf-8"))
    X_test = np.asarray(doc["x_test"])
    assert {case["method_id"] for case in doc["cases"]} == set(_REGISTRY)
    for case in doc["cases"]:
        payload = base64.b64decode(case["n4me_base64"])
        est = roles.NativeEstimator.from_n4me(payload)
        if "predict" in case:
            np.testing.assert_allclose(
                est.predict(X_test), case["predict"], rtol=1e-12, atol=1e-12
            )
        if "selected_indices" in case:
            np.testing.assert_array_equal(
                est.selected_indices_, case["selected_indices"]
            )
        if "transform" in case:
            np.testing.assert_allclose(
                est.transform(X_test), case["transform"], rtol=1e-12, atol=1e-12
            )
        assert est.to_n4me() == payload


# Selector role -------------------------------------------------------------

REQUIRED_SELECTOR_PARAMS = {
    "top_k": 4,
    "thresholds": [0.05, 0.1, 0.3],
    "alpha_thresholds": [0.95, 0.99],
}


def selector(cls):
    params = {
        k: v for k, v in REQUIRED_SELECTOR_PARAMS.items() if k in cls._param_types
    }
    if cls is roles.RandomFrog:
        params["initial_size"] = 6
    return cls(**params)


@pytest.mark.parametrize("cls", SELECTORS, ids=lambda c: c.__name__)
def test_selector_roundtrip(cls, data):
    X, y, X_test, _, _ = data
    est = selector(cls).fit(X, y)
    idx = est.selected_indices_
    assert len(set(idx.tolist())) == idx.size > 0
    np.testing.assert_array_equal(est.get_support(indices=True), np.sort(idx))
    np.testing.assert_array_equal(est.transform(X_test), X_test[:, np.sort(idx)])
    restored = roles.NativeEstimator.from_n4me(est.to_n4me())
    assert type(restored) is cls
    np.testing.assert_array_equal(restored.selected_indices_, idx)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(est)).transform(X_test), est.transform(X_test)
    )
    assert not hasattr(est, "predict")


def reference_selector(cls, est):
    """The n4m reference class with the same effective parameters."""
    from n4m.feature_selection import ranking, wrapper

    ref_cls = getattr(wrapper, cls.__name__, None) or getattr(ranking, cls.__name__)
    import inspect

    accepted = inspect.signature(ref_cls.__init__).parameters
    kwargs = {}
    for name, value in est.get_params().items():
        if name == "cv":
            if "n_folds" in accepted:
                kwargs["n_folds"] = value
            continue
        if name == "rank_method":
            value = ("vip", "coefficient", "selectivity_ratio").index(value)
        if name in {
            "min_features",
            "min_size",
            "max_size",
            "max_features",
            "min_selected",
        } and value in (0, -1):
            value = None
        if name in accepted:
            kwargs[name] = value
    return ref_cls(**kwargs)


@pytest.mark.parametrize("cls", SELECTORS, ids=lambda c: c.__name__)
def test_selector_matches_n4m_reference(cls, data):
    import warnings

    X, y, _, _, _ = data
    est = selector(cls).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = reference_selector(cls, est).fit(X, y)
    np.testing.assert_array_equal(
        est.selected_indices_, np.asarray(ref.selected_indices_)
    )

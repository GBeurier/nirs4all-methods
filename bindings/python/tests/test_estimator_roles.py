# SPDX-License-Identifier: CECILL-2.1
"""Generic estimator roles (ABI 2.13): Python facade over n4m_estimator_*."""

from __future__ import annotations

import ctypes
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from n4m import roles
from n4m._errors import N4MError
from n4m._ffi import lib
from n4m._impl import native
from n4m._types import MethodInfoV1
from n4m.roles._base import _REGISTRY

N_FEATURES = 12


@pytest.fixture(scope="module")
def data():
    # Two dominant latent factors carry the response, as in spectra.
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(48, 2)) * np.array([3.0, 2.0])
    loadings = rng.normal(size=(2, N_FEATURES))
    X = scores @ loadings + 0.1 * rng.normal(size=(48, N_FEATURES))
    y = scores[:, 0] - 0.5 * scores[:, 1] + 0.05 * rng.normal(size=48)
    # Paired transfer methods need one target row per training row.
    X_target = (
        rng.normal(size=(36, 2)) @ loadings
        + 0.3
        + 0.1 * rng.normal(size=(36, N_FEATURES))
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


# Fit input name -> n4m_fit_input_t index (n4m/estimator.h).
DATA_INPUTS = {"feature_groups": 4, "blocks": 5, "axis": 6, "X_target": 7}
AXIS = 1000.0 + 2.0 * np.arange(N_FEATURES)


def fit_kwargs(cls, X_target):
    """The data inputs the manifest declares required for ``cls``."""
    values = {
        "feature_groups": np.arange(N_FEATURES) // 4,
        "blocks": [4, 4, 4],
        "axis": AXIS,
        "X_target": X_target,
    }
    info = roles.method_info(cls._method_id)
    return {n: values[n] for n, i in DATA_INPUTS.items() if info.inputs[i] == 2}


ALL = sorted(_REGISTRY.values(), key=lambda c: c._method_id)
REGRESSORS = [c for c in ALL if issubclass(c, roles.NativeRegressor)]
SELECTORS = [c for c in ALL if issubclass(c, roles.NativeSelector)]
CLASSIFIERS = [c for c in ALL if issubclass(c, roles.NativeClassifier)]
PURE_TRANSFORMERS = [
    c
    for c in ALL
    if issubclass(c, roles.NativeTransformer)
    and not issubclass(c, roles.NativeRegressor)
]


def test_generated_classes_match_native_manifest():
    assert set(_REGISTRY) == manifest_estimators()


ROLE_BIT = {
    roles.NativeTransformer: 1 << 0,
    roles.NativeRegressor: 1 << 1,
    roles.NativeClassifier: 1 << 2,
    roles.NativeSelector: 1 << 3,
    roles.NativeSampleFilter: 1 << 4,
}


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_classes_expose_exactly_their_role_interfaces(cls):
    declared = roles.method_info(cls._method_id).roles
    for base, bit in ROLE_BIT.items():
        assert issubclass(cls, base) == bool(declared & bit)
    if not issubclass(cls, (roles.NativeTransformer, roles.NativeSelector)):
        assert not hasattr(cls, "transform")
    if not issubclass(cls, (roles.NativeRegressor, roles.NativeClassifier)):
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
        if case["n4me_base64"] is None:
            continue  # train-only filter without a serializable state
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
        if "mask" in case:
            np.testing.assert_array_equal(
                est.get_mask(X_test, doc["y_test"]), np.asarray(case["mask"], bool)
            )
        if "classes" in case:
            np.testing.assert_array_equal(est.classes_, case["classes"])
            np.testing.assert_array_equal(est.predict(X_test), case["predict_labels"])
            np.testing.assert_allclose(
                est.decision_function(X_test),
                case["decision_function"],
                rtol=1e-12,
                atol=1e-12,
            )
            assert hasattr(est, "predict_proba") == ("predict_proba" in case)
            if "predict_proba" in case:
                np.testing.assert_allclose(
                    est.predict_proba(X_test),
                    case["predict_proba"],
                    rtol=1e-12,
                    atol=1e-12,
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
    from n4m.feature_selection import filter, ranking, wrapper

    ref_cls = next(
        getattr(m, cls.__name__)
        for m in (wrapper, ranking, filter)
        if hasattr(m, cls.__name__)
    )
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
    if hasattr(ref, "selected_indices_"):
        np.testing.assert_array_equal(
            est.selected_indices_, np.asarray(ref.selected_indices_)
        )
    else:  # variance / correlation filters expose the columns in selection order
        np.testing.assert_array_equal(X[:, est.selected_indices_], ref.transform(X))


# Fitted transformers ------------------------------------------------------
# Each case compares the role with its n4m reference class on smooth
# synthetic spectra: train and held-out rows are bitwise equal, and the N4ME
# state round trips.

P_FITTED = 40
GRID = np.linspace(0.0, 1.0, P_FITTED)
FITTED_AXIS = 1000.0 + 2.0 * np.arange(P_FITTED)


@pytest.fixture(scope="module")
def spectra():
    rng = np.random.default_rng(7)

    def draw(n):
        shift = rng.normal(0.0, 0.03, (n, 1))
        a, b = rng.uniform(0.5, 1.5, (2, n, 1))
        return (
            1.5
            + a * np.exp(-((GRID - 0.3 - shift) ** 2) / 0.01)
            + b * np.exp(-((GRID - 0.7 - shift) ** 2) / 0.02)
            + 0.1 * GRID
            + 0.002 * rng.standard_normal((n, P_FITTED))
        )

    X, X_test = draw(40), draw(12)
    y = X[:, 10] * 2.0 - X[:, 30] + 0.01 * rng.standard_normal(40)
    X_target = 1.1 * X + 0.05 + 0.02 * np.sin(6.0 * GRID)
    return X, y, X_test, X_target, draw(1)[0], rng.uniform(0.2, 1.0, P_FITTED)


def _ref(module: str, name: str):
    import importlib

    return getattr(importlib.import_module(module), name)


AL, ST, SC = (
    "n4m.transform.alignment",
    "n4m.domain_adaptation.standardization",
    "n4m.transform.scatter",
)
FE, AUG, PRE, RES = (
    "n4m._impl.feature_extraction",
    "n4m._impl.augmentation",
    "n4m._impl.preprocessing",
    "n4m._impl.resampling",
)

# (role class, reference module, role params, reference params, fit style).
# Fit styles: "X" fits on X, "y" on (X, y), "paired" on (X, X_target), "axis"
# on the source axis. "ref"/"w" in params stand for the reference spectrum and
# the weights of the fixture.
FITTED_CASES = [
    ("CrossCorrelationAlignment", AL, {}, None, "X"),
    ("CrossCorrelationAlignment", AL, {"max_shift": 3, "reference": "ref"}, None, "X"),
    ("IcoshiftAlignment", AL, {}, None, "X"),
    ("IcoshiftAlignment", AL, {"interval_size": 10, "max_shift": 2}, None, "X"),
    ("DynamicTimeWarpingAlignment", AL, {}, None, "X"),
    ("DynamicTimeWarpingAlignment", AL, {"reference": "ref"}, None, "X"),
    ("CorrelationOptimizedWarping", AL, {}, None, "X"),
    (
        "CorrelationOptimizedWarping",
        AL,
        {"interval_size": 8, "max_shift": 2},
        None,
        "X",
    ),
    ("ScoreAugmentedProjectionStandardization", ST, {}, None, "paired"),
    (
        "ScoreAugmentedProjectionStandardization",
        ST,
        {"n_components": 3, "score_weight": 0.5, "fit_intercept": False, "ridge": 1e-3},
        None,
        "paired",
    ),
    ("DirectStandardization", ST, {"ridge": 1e-2}, None, "paired"),
    (
        "DirectStandardization",
        ST,
        {"fit_intercept": False, "ridge": 0.1},
        None,
        "paired",
    ),
    ("RobustDirectStandardization", ST, {"ridge": 1e-2}, None, "paired"),
    (
        "RobustDirectStandardization",
        ST,
        {"ridge": 0.1, "trim_quantile": 0.8, "max_iter": 5},
        None,
        "paired",
    ),
    ("PiecewiseDirectStandardization", ST, {}, None, "paired"),
    (
        "PiecewiseDirectStandardization",
        ST,
        {"window_size": 9, "fit_intercept": False, "ridge": 1e-3},
        None,
        "paired",
    ),
    ("LocalCentering", SC, {}, None, "paired"),
    ("LocalizedMSC", SC, {}, None, "X"),
    (
        "LocalizedMSC",
        SC,
        {"window_size": 7, "reference": "ref", "eps": 1e-9},
        None,
        "X",
    ),
    ("PiecewiseMSC", SC, {}, None, "X"),
    ("PiecewiseMSC", SC, {"window_size": 12, "reference": "ref"}, None, "X"),
    ("PiecewiseSNV", SC, {}, None, "X"),
    ("PiecewiseSNV", SC, {"window_size": 7, "ddof": 1, "eps": 1e-9}, None, "X"),
    ("WeightedSNV", SC, {}, None, "X"),
    ("WeightedSNV", SC, {"weights": "w", "ddof": 1}, None, "X"),
    ("VariableSortingNormalization", SC, {}, None, "X"),
    ("VariableSortingNormalization", SC, {"eps": 1e-6}, None, "X"),
    ("OSC", FE, {"n_components": 1, "scale": True}, None, "y"),
    ("OSC", FE, {"n_components": 3, "scale": False}, None, "y"),
    ("FlexiblePCA", FE, {"n_components": 5.0}, None, "X"),
    ("FlexiblePCA", FE, {"n_components": 0.95}, None, "X"),
    ("FlexibleSVD", FE, {"n_components": 5.0}, None, "X"),
    ("FlexibleSVD", FE, {"n_components": 0.95}, None, "X"),
    (
        "WaveletPCA",
        AUG,
        {
            "family": "haar",
            "mode": "periodization",
            "max_level": 2,
            "n_components": 5.0,
        },
        None,
        "X",
    ),
    (
        "WaveletSVD",
        AUG,
        {"family": "db4", "mode": "symmetric", "max_level": 3, "n_components": 0.9},
        None,
        "X",
    ),
    ("IntegerKBinsDiscretizer", RES, {"n_bins": 5, "strategy": "uniform"}, None, "X"),
    ("IntegerKBinsDiscretizer", RES, {"n_bins": 4, "strategy": "quantile"}, None, "X"),
    ("RangeDiscretizer", RES, {"edges": [1.6, 1.9, 2.2, 2.6]}, None, "X"),
    (
        "FCKStaticTransformer",
        FE,
        {"kernel_size": 7, "alphas": [0.0, 0.5, 1.0], "sigmas": [1.0, 2.0]},
        None,
        "X",
    ),
    ("BaselineCenter", PRE, {}, None, "X"),
    ("LogTransform", PRE, {}, None, "X"),
    (
        "LogTransform",
        PRE,
        {"base": 10.0, "offset": 0.5, "auto_offset": False},
        None,
        "X",
    ),
    ("Derivate", PRE, {"order": 2, "delta": 0.5}, None, "X"),
    (
        "Resampler",
        RES,
        {"method": "linear", "tgt_min": 1003.0, "tgt_step": 4.0, "tgt_n": 18},
        {"method": 0, "tgt_min": 1003.0, "tgt_step": 4.0, "tgt_n": 18},
        "axis",
    ),
    (
        "Resampler",
        RES,
        {
            "method": "cubic",
            "tgt_min": 990.0,
            "tgt_step": 3.0,
            "tgt_n": 30,
            "extrapolate": True,
        },
        {
            "method": 2,
            "tgt_min": 990.0,
            "tgt_step": 3.0,
            "tgt_n": 30,
            "extrapolate": True,
        },
        "axis",
    ),
    (
        "Resampler",
        RES,
        {
            "method": "nearest",
            "tgt_min": 1003.0,
            "tgt_step": 4.0,
            "tgt_n": 18,
            "use_crop": True,
            "crop_min": 1010.0,
            "crop_max": 1060.0,
        },
        {
            "method": 1,
            "tgt_min": 1003.0,
            "tgt_step": 4.0,
            "tgt_n": 18,
            "use_crop": True,
            "crop_min": 1010.0,
            "crop_max": 1060.0,
        },
        "axis",
    ),
]
FITTED_REFERENCE_CLASSES = {case[0] for case in FITTED_CASES} | {
    "Normalize",
    "SimpleScale",
}


@pytest.mark.parametrize(
    "name,module,params,ref_params,style",
    FITTED_CASES,
    ids=[f"{c[0]}-{i}" for i, c in enumerate(FITTED_CASES)],
)
def test_fitted_transformer_matches_n4m_reference(
    name, module, params, ref_params, style, spectra
):
    X, y, X_test, X_target, reference, weights = spectra
    fixture = {"ref": reference, "w": weights}
    resolve = lambda p: {
        k: fixture.get(v, v) if isinstance(v, str) else v for k, v in p.items()
    }
    role = getattr(roles, name)(**resolve(params))
    ref = _ref(module, name)(
        **resolve(ref_params if ref_params is not None else params)
    )
    if style == "paired":
        role.fit(X, X_target=X_target)
        ref.fit(X, X_target)
    elif style == "axis":
        role.fit(X, axis=FITTED_AXIS)
        ref.fit(source_wavelengths=FITTED_AXIS)
    elif style == "y":
        role.fit(X, y)
        ref.fit(X, y)
    else:
        role.fit(X)
        ref.fit(X)
    for rows in (X, X_test):
        np.testing.assert_array_equal(role.transform(rows), ref.transform(rows))
    payload = role.to_n4me()
    back = roles.NativeEstimator.from_n4me(payload)
    assert type(back) is type(role) and back.to_n4me() == payload
    np.testing.assert_array_equal(back.transform(X_test), role.transform(X_test))


def test_refit_relearns_fitted_state(spectra):
    """A second fit on data of another width equals a fresh fit."""
    X, _, X_test, _, _, _ = spectra
    X2 = X_test[:, : P_FITTED - 4]
    for cls in (roles.CrossCorrelationAlignment, roles.PiecewiseMSC, roles.OSC):
        est = cls().fit(X, X[:, 0]).fit(X2, X2[:, 0])
        fresh = cls().fit(X2, X2[:, 0])
        np.testing.assert_array_equal(est.transform(X2), fresh.transform(X2))


def test_scaling_roles_apply_training_statistics(spectra):
    """Normalize and SimpleScale learn column statistics at fit.

    The stateless references recompute them per batch, so they agree on the
    training rows only; held-out rows use the training statistics.
    """
    X, _, X_test, _, _, _ = spectra
    lo, hi = X.min(axis=0), X.max(axis=0)
    for kw in ({}, {"feature_min": 0.0, "feature_max": 2.0}):
        role = roles.Normalize(**kw).fit(X)
        np.testing.assert_array_equal(
            role.transform(X), _ref(PRE, "Normalize")(**kw).fit(X).transform(X)
        )
        if kw:
            expect = kw["feature_min"] + (kw["feature_max"] - kw["feature_min"]) / (
                hi - lo
            ) * (X_test - lo)
        else:
            expect = X_test * (1.0 / np.sqrt((X * X).sum(axis=0)))
        np.testing.assert_allclose(role.transform(X_test), expect, rtol=1e-13, atol=0)
    role = roles.SimpleScale().fit(X)
    np.testing.assert_array_equal(
        role.transform(X), _ref(PRE, "SimpleScale")().fit(X).transform(X)
    )
    np.testing.assert_allclose(
        role.transform(X_test), (X_test - lo) / (hi - lo), rtol=1e-13, atol=0
    )


# Transformer role --------------------------------------------------------

TRANSFORMER_PARAMS = {
    "start": 2,
    "end": 10,
    "num_samples": 8,
    "edges": [0.25, 0.5, 0.75],
    "kernel_size": 5,
    "alphas": [0.0, 1.0],
    "sigmas": [1.0, 2.0],
}


def transformer(cls):
    return cls(**{k: v for k, v in TRANSFORMER_PARAMS.items() if k in cls._param_types})


def positive_spectra(data):
    """Strictly positive, reflectance-like rows (valid for every conversion)."""
    X, y, X_test, X_target, _ = data
    shift = 1.0 - min(X.min(), X_test.min(), X_target.min())
    scale = lambda A: (A + shift) / (2 * shift)
    return scale(X), y, scale(X_test), scale(X_target)


@pytest.mark.parametrize("cls", PURE_TRANSFORMERS, ids=lambda c: c.__name__)
def test_transformer_roundtrip(cls, data):
    X, y, X_test, X_target = positive_spectra(data)
    est = transformer(cls).fit(X, y, **fit_kwargs(cls, X_target))
    out = est.transform(X_test)
    assert out.shape[0] == X_test.shape[0] and np.all(np.isfinite(out))
    restored = roles.NativeEstimator.from_n4me(est.to_n4me())
    assert type(restored) is cls
    np.testing.assert_array_equal(restored.transform(X_test), out)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(est)).transform(X_test), out
    )
    assert not hasattr(est, "predict")


def reference_transformer(cls, est):
    """The n4m reference transformer class with the same parameters."""
    import importlib
    import inspect

    for module in (
        "baseline",
        "smoothing",
        "resampling",
        "scatter",
        "signal_conversion",
        "wavelet",
    ):
        ref_cls = getattr(
            importlib.import_module(f"n4m.transform.{module}"), cls.__name__, None
        )
        if ref_cls is not None:
            break
    accepted = inspect.signature(ref_cls.__init__).parameters
    return ref_cls(**{k: v for k, v in est.get_params().items() if k in accepted})


@pytest.mark.parametrize(
    "cls",
    [c for c in PURE_TRANSFORMERS if c.__name__ not in FITTED_REFERENCE_CLASSES],
    ids=lambda c: c.__name__,
)
def test_transformer_matches_n4m_reference(cls, data):
    X, y, X_test, X_target = positive_spectra(data)
    est = transformer(cls).fit(X, y, **fit_kwargs(cls, X_target))
    ref = reference_transformer(cls, est).fit(X, y)
    np.testing.assert_array_equal(est.transform(X_test), ref.transform(X_test))


# Classifier role ---------------------------------------------------------


@pytest.fixture(scope="module")
def labelled(data):
    """Three classes cut from the latent response, as string labels."""
    X, y, X_test, _, y_test = data
    cuts = np.quantile(y, [1 / 3, 2 / 3])
    names = np.array(["high", "low", "mid"])
    to_names = lambda v: names[[1, 2, 0]][np.digitize(v, cuts)]
    return X, to_names(y), X_test, to_names(y_test)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=lambda c: c.__name__)
def test_classifier_roundtrip(cls, labelled):
    X, labels, X_test, labels_test = labelled
    est = cls().fit(X, labels)
    np.testing.assert_array_equal(est.classes_, ["high", "low", "mid"])
    pred = est.predict(X_test)
    assert pred.dtype.kind == "U" and np.mean(pred == labels_test) > 0.6
    decision = est.decision_function(X_test)
    assert decision.shape == (X_test.shape[0], 3)
    np.testing.assert_array_equal(est.classes_[decision.argmax(axis=1)], pred)
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-12)
        np.testing.assert_array_equal(est.classes_[proba.argmax(axis=1)], pred)
    assert est.score(X, labels) > 0.6

    restored = roles.NativeEstimator.from_n4me(est.to_n4me())
    assert type(restored) is cls
    np.testing.assert_array_equal(restored.decision_function(X_test), decision)
    assert restored.to_n4me() == est.to_n4me()
    unpickled = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(unpickled.predict(X_test), pred)


def test_classifier_integer_labels_and_proba_capability(labelled):
    X, labels, X_test, _ = labelled
    codes = np.unique(labels, return_inverse=True)[1] * 10 + 10
    est = roles.PLSQDA().fit(X, codes)
    np.testing.assert_array_equal(est.classes_, [10, 20, 30])
    assert set(est.predict(X_test)) <= {10, 20, 30}
    assert hasattr(est, "predict_proba")
    assert not hasattr(roles.PLSLDA().fit(X, codes), "predict_proba")
    with pytest.raises(N4MError, match="labels"):
        roles.PLSLDA().fit(X, None)


def pls4all_classifier_config():
    """The PLS projection the classifier roles fit: SIMPLS, centred, unscaled."""
    import pls4all

    cfg = pls4all.Config()
    cfg.n_components = 2
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.scale_x = False
    cfg.scale_y = False
    return pls4all.Context(), cfg


def test_classifiers_match_n4m_kernels(labelled):
    """In-sample scores equal the historical fit-predict kernels."""
    from pls4all import _methods as kernels

    X, labels, _, _ = labelled
    codes = np.unique(labels, return_inverse=True)[1]
    ctx, cfg = pls4all_classifier_config()
    lda = kernels.pls_lda_fit(ctx, cfg, X, codes, 3)
    np.testing.assert_array_equal(
        roles.PLSLDA().fit(X, codes).decision_function(X), lda.matrix("decision_scores")
    )
    logistic = kernels.pls_logistic_fit(ctx, cfg, X, codes, 3)
    est = roles.PLSLogistic().fit(X, codes)
    np.testing.assert_array_equal(
        est.decision_function(X), logistic.matrix("decision_scores")
    )
    np.testing.assert_array_equal(
        est.predict_proba(X), logistic.matrix("probabilities")
    )
    qda = kernels.pls_qda_fit(ctx, cfg, X, codes)
    np.testing.assert_array_equal(
        roles.PLSQDA().fit(X, codes).decision_function(X), qda.matrix("predictions")
    )
    sparse = kernels.sparse_pls_da_fit(ctx, cfg, X, codes)
    decision = roles.SparsePLSDA(sparsity_lambda=0.0).fit(X, codes).decision_function(X)
    reference = (X - sparse.matrix("x_mean")) @ sparse.matrix(
        "coefficients"
    ) + sparse.matrix("y_mean")
    np.testing.assert_allclose(decision, reference, rtol=1e-12, atol=1e-12)


def test_pls_qda_is_quadratic_discriminant_on_scores(labelled):
    """PLSQDA equals scikit-learn QDA fitted on the PLS scores."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    X, labels, X_test, _ = labelled
    codes = np.unique(labels, return_inverse=True)[1]
    est = roles.PLSQDA().fit(X, codes)
    pls = PLSRegression(n_components=2, scale=False).fit(X, np.eye(3)[codes])
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(
        pls.transform(X), codes
    )
    np.testing.assert_array_equal(
        est.predict(X_test), qda.predict(pls.transform(X_test))
    )


# Sample-filter role -------------------------------------------------------

SAMPLE_FILTER_CASES = [
    ("YOutlierFilter", {"method": "iqr", "threshold": 0.5}),
    ("YOutlierFilter", {"method": "zscore", "threshold": 1.0}),
    (
        "YOutlierFilter",
        {"method": "percentile", "lower_percentile": 10.0, "upper_percentile": 90.0},
    ),
    ("YOutlierFilter", {"method": "mad", "threshold": 1.0}),
    ("XOutlierFilter", {"method": "mahalanobis", "n_components": 3}),
    ("XOutlierFilter", {"method": "pca_residual", "n_components": 2}),
    ("XOutlierFilter", {"method": "isolation_forest", "seed": 5, "n_estimators": 20}),
    ("XOutlierFilter", {"method": "lof", "contamination": 0.2}),
    ("HighLeverageFilter", {"method": "hat", "threshold_multiplier": 1.5}),
    ("HighLeverageFilter", {"method": "pca", "n_components": 2}),
    ("HighLeverageFilter", {"absolute_threshold": 0.3}),
    ("SpectralQualityFilter", {"max_value": 3.0}),
    ("SpectralQualityFilter", {"min_value": -1.0, "min_variance": 0.5}),
]


@pytest.mark.parametrize(
    "name,params",
    SAMPLE_FILTER_CASES,
    ids=[f"{c[0]}-{i}" for i, c in enumerate(SAMPLE_FILTER_CASES)],
)
def test_sample_filter_matches_n4m_reference(name, params, data):
    """Keep masks equal the n4m reference filters on training and new rows."""
    from n4m import outlier_detection

    X, y, X_test, _, y_test = data
    role = getattr(roles, name)(**params)
    ref = getattr(outlier_detection, name)(**params)
    reads_y = name == "YOutlierFilter"
    role.fit(X, y)
    if reads_y:
        ref.fit(y)
    elif hasattr(ref, "fit"):
        ref.fit(X)
    for rows, target in ((X, y), (X_test, y_test)):
        expected = ref.apply(target if reads_y else rows)[0].astype(bool)
        np.testing.assert_array_equal(role.get_mask(rows, target), expected)
    if roles.method_info(role._method_id).capabilities & (1 << 7):
        back = roles.NativeEstimator.from_n4me(role.to_n4me())
        assert type(back) is type(role) and back.get_params() == role.get_params()
        np.testing.assert_array_equal(
            back.get_mask(X_test, y_test), role.get_mask(X_test, y_test)
        )
    else:
        with pytest.raises(N4MError):
            role.to_n4me()


def test_sample_filter_roles_and_inputs(data):
    X, y, X_test, _, _ = data
    est = roles.YOutlierFilter().fit(X, y)
    assert not hasattr(est, "transform") and not hasattr(est, "predict")
    with pytest.raises(N4MError):
        est.get_mask(X_test)  # the target filter needs y
    with pytest.raises(N4MError, match="missing required fit input"):
        roles.YOutlierFilter().fit(X)
    mask = roles.HighLeverageFilter().fit(X).get_mask(X_test)
    assert mask.dtype == bool and mask.shape == (X_test.shape[0],)
    est = pickle.loads(pickle.dumps(roles.HighLeverageFilter().fit(X)))
    np.testing.assert_array_equal(est.get_mask(X_test), mask)
    assert roles.HighLeverageFilter().get_params()["absolute_threshold"] is None


def test_public_class_lookup_token_and_tags():
    """Controllers resolve a class by method id; pipelines serialize n4m.roles.<Name>."""
    from sklearn.utils import get_tags

    import n4m.roles as public

    cls = roles.estimator_class("models.pls.cppls")
    assert cls is roles.CPPLS and cls.__module__ == "n4m.roles"
    assert getattr(public, cls.__qualname__) is cls
    with pytest.raises(ValueError, match="no n4m role class"):
        roles.estimator_class("models.pls.missing")
    assert get_tags(roles.CPPLS()).target_tags.required
    assert not get_tags(roles.SNV()).target_tags.required
    assert roles.Resampler.input_requirements()["axis"] == "required"
    assert roles.DIPLS.input_requirements()["X_target"] == "required"
    assert roles.PLSLDA.input_requirements()["labels"] == "required"

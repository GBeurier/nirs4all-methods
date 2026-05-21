"""Smoke + sklearn-compatibility tests for the 28 SelectorMixin
wrappers shipped by `pls4all.sklearn`. Validates that every selector:

* fits without raising on a representative synthetic dataset
* exposes the SelectorMixin contract (`support_`, `transform()`,
  `get_support()`) consistently
* survives ``pickle.dumps`` / ``copy.deepcopy``
* round-trips ``get_params`` / ``set_params``
* plays nicely inside an ``sklearn.pipeline.Pipeline``

All 28 selectors fit a PLS regression internally, then run their
selector-specific scoring/sampling/elimination loop. Bit-exact parity
against tier-1 is exercised in test_sklearn_slice.py for VIPSelector;
the broader bit-exact sweep across all selectors is part of the parity
gate, not these tests.

Run from the repo root:

    N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so \
        PYTHONPATH=bindings/python/src \
        parity/python_generator/.venv/bin/python -m pytest \
        bindings/python/tests/test_sklearn_selectors.py -v
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from pls4all import sklearn as p4sk
from pls4all.sklearn import PLSRegression
from pls4all.sklearn import _selection as selection_mod


@pytest.fixture(scope="module")
def selector_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 40))
    beta = np.zeros(40)
    beta[[3, 11, 22, 35]] = [3.0, -2.0, 1.5, -1.0]
    y = X @ beta + 0.1 * rng.standard_normal(150)
    return X.astype(np.float64), y.astype(np.float64)


def _build_selector(name: str):
    """Return a selector instance with sane defaults for smoke tests."""
    cls = getattr(p4sk, name)
    if name == "STSelector":
        return cls(thresholds=[0.05, 0.1, 0.3], n_components=3)
    if name == "T2Selector":
        return cls(alpha_thresholds=[0.95, 0.99], n_components=3)
    if name in ("VIPSelector", "CoefficientSelector", "SelectivityRatioSelector",
                  "SPASelector", "StabilitySelector", "RandomFrogSelector",
                  "IPWSelector", "WVCSelector", "IRFSelector", "VIPSPASelector"):
        return cls(top_k=5, n_components=3)
    return cls(n_components=3)


SELECTOR_NAMES = sorted(name for name in dir(p4sk) if name.endswith("Selector"))


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_selector_fit_and_contract(name, selector_data):
    X, y = selector_data
    sel = _build_selector(name)
    sel.fit(X, y)
    # SelectorMixin contract.
    assert sel.support_.dtype == bool
    assert sel.support_.shape == (X.shape[1],)
    assert sel.selected_indices_.dtype == np.int64
    assert sel.selected_indices_.ndim == 1
    assert sel.support_.sum() == sel.selected_indices_.size
    # transform must produce X[:, support_]
    X_t = sel.transform(X)
    assert X_t.shape == (X.shape[0], int(sel.support_.sum()))
    # n_features_in_ recorded.
    assert int(sel.n_features_in_) == X.shape[1]


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_selector_pickle_roundtrip(name, selector_data):
    X, y = selector_data
    sel = _build_selector(name)
    sel.fit(X, y)
    blob = pickle.dumps(sel)
    sel2 = pickle.loads(blob)
    # Mask, indices and scores all survive bit-exactly.
    assert np.array_equal(sel.support_, sel2.support_)
    assert np.array_equal(sel.selected_indices_, sel2.selected_indices_)
    assert np.array_equal(sel.scores_, sel2.scores_, equal_nan=True)
    assert np.array_equal(sel.transform(X), sel2.transform(X))


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_selector_deepcopy(name, selector_data):
    X, y = selector_data
    sel = _build_selector(name)
    sel.fit(X, y)
    sel2 = copy.deepcopy(sel)
    assert np.array_equal(sel.support_, sel2.support_)


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_selector_get_params_roundtrip(name):
    sel = _build_selector(name)
    params = sel.get_params()
    cls = type(sel)
    # Rebuild via set_params on a default-ish instance.
    sel2 = cls(**params) if not params else cls(**{k: v for k, v in params.items()})
    assert sel2.get_params() == params


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_selector_in_pipeline(name, selector_data):
    X, y = selector_data
    pipe = Pipeline([
        ("select", _build_selector(name)),
        ("pls", PLSRegression(n_components=1)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (X.shape[0],)


def test_uve_selector_min_features_fallback_is_deterministic(
        monkeypatch, selector_data):
    class EmptyUVEResult:
        def vector_int64(self, name):
            assert name == "selected_indices"
            return np.empty(0, dtype=np.int64)

        def matrix(self, name):
            assert name == "real_stability_scores"
            return np.array([[0.2, 0.5, 0.5, -1.0]], dtype=np.float64)

    def fake_uve_select(*args, **kwargs):
        return EmptyUVEResult()

    monkeypatch.setattr(selection_mod._methods, "uve_select", fake_uve_select)
    X, y = selector_data
    sel = selection_mod.UVESelector(min_features=2)
    with pytest.warns(UserWarning, match="min_features"):
        sel.fit(X[:, :4], y)

    assert np.array_equal(sel.selected_indices_, np.array([1, 2]))
    assert np.array_equal(sel.support_, np.array([False, True, True, False]))


def test_uve_selector_min_features_zero_preserves_empty_support(
        monkeypatch, selector_data):
    class EmptyUVEResult:
        def vector_int64(self, name):
            assert name == "selected_indices"
            return np.empty(0, dtype=np.int64)

        def matrix(self, name):
            assert name == "real_stability_scores"
            return np.array([[0.2, 0.5, 0.5, -1.0]], dtype=np.float64)

    def fake_uve_select(*args, **kwargs):
        return EmptyUVEResult()

    monkeypatch.setattr(selection_mod._methods, "uve_select", fake_uve_select)
    X, y = selector_data
    sel = selection_mod.UVESelector(min_features=0).fit(X[:, :4], y)

    assert sel.selected_indices_.size == 0
    assert not sel.support_.any()

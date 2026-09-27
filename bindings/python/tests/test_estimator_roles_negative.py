# SPDX-License-Identifier: CECILL-2.1
"""Refusals of the generic estimator roles: the shared negative fixture (also
replayed by R, JS/WASM and Rust) and the Python-specific input conversions."""

from __future__ import annotations

import base64
import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from n4m import roles
from n4m._errors import N4MError
from n4m._ffi import lib
from n4m._matrix import numpy_to_view
from n4m.roles._base import _Context, method_class

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "parity"
    / "fixtures"
    / "estimator_roles_negative.json"
)
DOC = json.loads(FIXTURE.read_text(encoding="utf-8"))
X = np.asarray(DOC["x_train"])
X_TEST = np.asarray(DOC["x_test"])


def fit_kwargs(case: dict) -> dict:
    kw = {}
    for name in ("sample_weight", "fold_ids"):
        if name in case:
            kw[name] = np.asarray(case[name])
    return kw


def target(case: dict):
    if "y_matrix" in case:
        return np.asarray(case["y_matrix"])
    if "labels" in case:
        return np.asarray(case["labels"])
    return np.asarray(case["y"])


@pytest.mark.parametrize("case", DOC["cases"], ids=[c["id"] for c in DOC["cases"]])
def test_shared_negative_fixture(case):
    cls = method_class(case["method_id"])
    step = case["step"]
    if step == "fit":
        est = cls(**case.get("params", {}))
        with pytest.raises((ValueError, N4MError), match=case["mentions"]):
            est.fit(X, target(case), **fit_kwargs(case))
    elif step == "import":
        with pytest.raises(N4MError, match=case["mentions"]):
            roles.NativeEstimator.from_n4me(base64.b64decode(case["n4me_base64"]))
        control = roles.NativeEstimator.from_n4me(
            base64.b64decode(case["control_base64"])
        )
        np.testing.assert_allclose(
            control.predict(X_TEST), case["control_predict"], rtol=1e-12
        )
    elif step == "output_view":
        check_output_views(case)
    elif step == "export":
        est = cls(**case["params"]).fit(X, target(case))
        assert est.contains_training_rows_ is case["contains_training_rows"]
        with pytest.raises(N4MError, match=case["mentions"]):
            est.to_n4me()
        assert (
            len(est.to_n4me(allow_training_rows=True))
            == case["export_size_with_opt_in"]
        )
    else:
        assert step == "refit"
        est = cls(**case["params"]).fit(X, target(case))
        with pytest.raises(N4MError, match=case["mentions"]):
            est.fit(X, np.asarray(case["refit_labels"]))
        np.testing.assert_array_equal(est.classes_, case["classes"])
        np.testing.assert_array_equal(est.predict(X_TEST), case["predict_labels"])


def check_output_views(case: dict) -> None:
    """A caller's output view of another width is refused before any write."""
    est = roles.NativeEstimator.from_n4me(base64.b64decode(case["n4me_base64"]))
    assert est.transform(X_TEST).shape[1] == case["transform_cols"]
    assert est.predict(X_TEST).shape == (X_TEST.shape[0],)
    x_view = numpy_to_view(X_TEST)
    for symbol, cols in (
        ("n4m_estimator_transform", case["transform_cols"] - 1),
        ("n4m_estimator_transform", case["transform_cols"] + 1),
        ("n4m_estimator_predict", case["predict_cols"] + 1),
    ):
        out = np.full((X_TEST.shape[0], cols), -7.0)
        out_view = numpy_to_view(out)
        with _Context() as ctx, pytest.raises(N4MError, match=case["mentions"]):
            ctx.check(
                getattr(lib, symbol)(
                    ctx.handle,
                    est._handle(),
                    ctypes.byref(x_view),
                    ctypes.byref(out_view),
                ),
                symbol,
            )
        assert np.all(out == -7.0)


def test_target_shapes_are_never_reinterpreted():
    y = X[:, 0]
    reg = roles.PLSRegression(n_components=2)
    # (n,), (n, 1) and (n, q) are the accepted shapes.
    assert reg.fit(X, y).predict(X_TEST).shape == (X_TEST.shape[0],)
    assert reg.fit(X, y[:, None]).predict(X_TEST).shape == (X_TEST.shape[0], 1)
    assert reg.fit(X, X[:, :2]).predict(X_TEST).shape == (X_TEST.shape[0], 2)
    for bad in (y.reshape(2, -1), y[:, None, None], y[:, None][:, :0], np.float64(7.0)):
        with pytest.raises(ValueError, match="y must have shape"):
            reg.fit(X, bad)
    with pytest.raises(ValueError, match="one per row"):
        roles.PLSRegression().fit(X, y, sample_weight=np.ones((2, 20)))
    # Class labels: 1-D or one column, one per row.
    lda = roles.PLSLDA()
    labels = (y > 0).astype(int)
    lda.fit(X, labels[:, None])
    with pytest.raises(ValueError, match="class labels"):
        lda.fit(X, labels.reshape(2, -1))
    # Sample filters and procedures check their target the same way.
    filt = roles.YOutlierFilter().fit(X, y)
    with pytest.raises(ValueError, match="y must have shape"):
        filt.get_mask(X, y[:-1])
    with pytest.raises(ValueError, match="y must have shape"):
        roles.KennardStone().split(X, y.reshape(2, -1)).__next__()


def test_failed_refit_keeps_labels_shape_and_feature_names():
    pd = pytest.importorskip("pandas")
    y = X[:, 0]
    names = np.where(y > 0, "healthy", "diseased")
    lda = roles.PLSLDA().fit(
        pd.DataFrame(X, columns=[f"w{j}" for j in range(X.shape[1])]), names
    )
    before = lda.predict(X_TEST)
    with pytest.raises(N4MError, match="two classes"):
        lda.fit(X, np.repeat("other", X.shape[0]))
    np.testing.assert_array_equal(lda.predict(X_TEST), before)
    assert list(lda.classes_) == ["diseased", "healthy"]
    assert list(lda.feature_names_in_) == [f"w{j}" for j in range(X.shape[1])]
    # A successful refit on an array drops the stale column names.
    lda.fit(X, names)
    assert not hasattr(lda, "feature_names_in_")

    reg = roles.PLSRegression(n_components=2).fit(X, X[:, :2])
    before = reg.predict(X_TEST)
    with pytest.raises(ValueError):
        reg.fit(X, y[:10])
    np.testing.assert_array_equal(reg.predict(X_TEST), before)
    assert reg.predict(X_TEST).shape == (X_TEST.shape[0], 2)


def test_seeds_are_optional_and_unset_means_zero():
    cls = roles.GaussianNoise
    assert cls().seed is None
    param = next(
        p
        for m in roles.manifest()["methods"]
        if m["method_id"] == cls._method_id
        for p in m["params"]
        if p["name"] == "seed"
    )
    assert param["default"] is None and param["required"] is False
    np.testing.assert_array_equal(cls().augment(X), cls(seed=0).augment(X))
    assert not np.array_equal(cls().augment(X), cls(seed=1).augment(X))
    # A fitted state records the seed it ran with.
    bag = roles.BaggingPLS(n_estimators=3).fit(X, X[:, 0])
    assert roles.NativeEstimator.from_n4me(bag.to_n4me()).get_params()["seed"] == 0

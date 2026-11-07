"""SNV owner parameter fidelity: typed admission, lifecycle and domain."""

from __future__ import annotations

import ctypes
import enum
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from n4m._ffi import lib
from n4m._matrix import numpy_to_view
from n4m.roles import SNV as RoleSNV
from n4m.roles import RolePipeline
from n4m.transform.scatter import SNV

ROOT = Path(__file__).resolve().parents[3]
DOC = json.loads((ROOT / "parity/fixtures/snv_parameter_contract_v1.json").read_text())
X = np.asarray(DOC["matrix"], dtype=np.float64)
assert set(DOC) == {
    "schema",
    "matrix",
    "references",
    "domains",
    "typed_values",
    "default_bits",
    "provenance",
}
assert DOC["schema"] == "n4m.snv-parameter-contract.v1"
assert all(set(r) == {"id", "params", "values"} for r in DOC["references"])
assert all(
    set(r) == {"id", "params", "create_status", "transform_status"}
    for r in DOC["domains"]
)
assert all(
    set(r) == {"id", "parameter", "tag", "value", "admit"} for r in DOC["typed_values"]
)


assert all(
    set(r["params"]) == {"with_mean", "with_std", "ddof"}
    for r in DOC["references"] + DOC["domains"]
)
assert all(
    type(r["create_status"]) is int and type(r["transform_status"]) is int
    for r in DOC["domains"]
)
assert all(
    set(r["admit"]) == {"python", "javascript", "r", "octave"}
    for r in DOC["typed_values"]
)


class Degree(enum.IntEnum):
    ONE = 1


def value(row):
    tag, val = row["tag"], row["value"]
    if tag == "literal":
        return val
    if tag == "nan":
        return float("nan")
    if tag == "infinity":
        return float("inf")
    if tag == "np.timedelta64":
        return np.timedelta64(1)
    if tag == "np.timedelta64[s]":
        return np.timedelta64(1, "s")
    if tag == "IntEnum":
        return Degree.ONE
    if tag == "np.int64":
        return np.int64(val)
    if tag == "np.uint32":
        return np.uint32(val)
    if tag == "np.bool_":
        return np.bool_(val)
    raise AssertionError(f"unconsumed Python tag: {tag}")


@pytest.fixture
def calls(monkeypatch):
    seen = []
    names = [
        "n4m_context_create",
        "n4m_method_find",
        "n4m_params_create",
        "n4m_params_destroy",
        "n4m_params_set_bool",
        "n4m_params_set_int",
        "n4m_params_validate",
        "n4m_estimator_create",
        "n4m_estimator_fit",
        "n4m_estimator_transform",
        "n4m_role_pipeline_create",
        "n4m_role_pipeline_fit",
        "n4m_role_pipeline_transform",
    ]
    for name in names:
        original = getattr(lib, name)

        def wrapped(*args, _name=name, _original=original):
            seen.append(_name)
            return _original(*args)

        monkeypatch.setattr(lib, name, wrapped)
    return seen


@pytest.mark.parametrize("cls", [SNV, RoleSNV])
@pytest.mark.parametrize("row", DOC["references"], ids=lambda r: r["id"])
def test_literal_references(cls, row):
    got = cls(**row["params"]).fit(X).transform(X)
    np.testing.assert_allclose(got, row["values"], rtol=0, atol=1e-12)
    if row["id"] == "mean1-std1-ddof0":
        assert got.astype("<f8").tobytes().hex() == DOC["default_bits"]["native_e7"]
    elif row["id"] != "mean1-std1-ddof1":
        default = np.frombuffer(
            bytes.fromhex(DOC["default_bits"]["native_e7"]), dtype="<f8"
        ).reshape(X.shape)
        assert np.max(np.abs(got - default)) > 1e-10


ROWS = [r for r in DOC["typed_values"] if r["admit"]["python"] is not None]


@pytest.mark.parametrize("cls", [SNV, RoleSNV])
@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_typed_admission(cls, row, calls):
    kw = {row["parameter"]: value(row)}
    if row["admit"]["python"]:
        got = cls(**kw).fit(X).transform(X)
        ref = next(
            r
            for r in DOC["references"]
            if r["params"]["with_mean"] is True
            and r["params"]["with_std"]
            == (row["parameter"] != "with_std" or bool(kw["with_std"]))
            and r["params"]["ddof"] == (int(kw["ddof"]) if "ddof" in kw else 0)
        )
        np.testing.assert_allclose(got, ref["values"], rtol=0, atol=1e-12)
    else:
        with pytest.raises((TypeError, ValueError)):
            cls(**kw).fit(X).transform(X)
        for name in [
            "n4m_method_find",
            "n4m_params_create",
            "n4m_params_set_int",
            "n4m_params_set_bool",
            "n4m_params_validate",
            "n4m_estimator_create",
            "n4m_estimator_fit",
            "n4m_estimator_transform",
        ]:
            assert calls.count(name) == 0, (name, calls)


@pytest.mark.parametrize("row", DOC["domains"], ids=lambda r: r["id"])
def test_owner_extent(row):
    h = ctypes.c_void_p()
    args = row["params"]
    status = lib.n4m_transform_snv_create(
        ctypes.byref(h), int(args["with_mean"]), int(args["with_std"]), args["ddof"]
    )
    assert status == row["create_status"]
    if status:
        assert not h.value
        return
    try:
        out = np.full_like(X, -12345.0)
        status = lib.n4m_transform_snv_transform(
            h, numpy_to_view(X), numpy_to_view(out)
        )
        assert status == row["transform_status"]
        if status:
            np.testing.assert_array_equal(out, np.full_like(X, -12345.0))
    finally:
        lib.n4m_transform_snv_destroy(h)


@pytest.mark.parametrize("cls", [SNV, RoleSNV])
@pytest.mark.parametrize(
    "params",
    [
        {"with_std": np.array(True)},
        {"with_std": [True]},
        {"with_mean": object()},
        {"ddof": np.array(1)},
        {"ddof": [1]},
        {"ddof": object()},
    ],
)
def test_nonscalar_and_object_parameters(cls, params, calls):
    with pytest.raises(TypeError):
        cls(**params).fit(X).transform(X)
    assert not any(name not in {"n4m_context_create"} for name in calls), calls


def test_counter_positive_and_mixed_fields(calls):
    RoleSNV().fit(X).transform(X)
    assert calls.count("n4m_estimator_fit") == 1
    assert calls.count("n4m_estimator_transform") == 1
    calls.clear()
    with pytest.raises((TypeError, ValueError)):
        RoleSNV(with_mean=True, ddof=1.9).fit(X)
    assert not any(n != "n4m_context_create" for n in calls), calls


@pytest.mark.parametrize("last", [False, True])
def test_pipeline_malformed_cleanup(last, calls):
    # The last-SNV construction gate tests cleanup, not admission of a
    # transformer-only recipe. A genuine valid recipe is exercised below.
    bad = ("preprocessing.scatter.snv", {"ddof": -1})
    steps = (
        ["preprocessing.scaling.standard_scale", bad]
        if last
        else [bad, "models.regularized.ridge"]
    )
    with pytest.raises((TypeError, ValueError)):
        RolePipeline.from_steps(steps)
    assert calls.count("n4m_params_create") == calls.count("n4m_params_destroy"), calls
    assert calls.count("n4m_role_pipeline_create") == 0
    assert calls.count("n4m_role_pipeline_fit") == 0
    assert calls.count("n4m_params_set_int") == 0


def test_valid_pipeline_and_nondefault_forwarding(calls):
    from n4m.roles import Ridge

    y = X[:, 0] + X[:, 2]
    pipeline = RolePipeline.from_steps(
        [("preprocessing.scatter.snv", {"with_std": False}), "models.regularized.ridge"]
    ).fit(X, y)
    centered = RoleSNV(with_std=False).fit(X).transform(X)
    expected = Ridge().fit(centered, y).predict(X - X.mean(axis=1, keepdims=True))
    np.testing.assert_array_equal(pipeline.predict(X), expected)
    assert calls.count("n4m_role_pipeline_create") > 0
    assert calls.count("n4m_params_create") == calls.count("n4m_params_destroy")
    pipeline.close()


def test_direct_mutation_and_atomic_set_params():
    s = SNV().fit(X)
    s.with_std = False
    np.testing.assert_array_equal(s.transform(X), X - X.mean(axis=1, keepdims=True))
    s.with_std = True
    s.ddof = 1
    np.testing.assert_allclose(
        s.fit(X).transform(X), DOC["references"][1]["values"], rtol=0, atol=1e-12
    )
    before = s.get_params().copy()
    with pytest.raises(ValueError):
        s.set_params(with_std=False, unknown=1)
    assert s.get_params() == before
    s.ddof = "x"
    with pytest.raises(TypeError):
        s.transform(X)
    s.set_params(ddof=0, with_std=False)
    np.testing.assert_array_equal(s.transform(X), X - X.mean(axis=1, keepdims=True))


def test_sklearn_clone():
    from sklearn.base import clone

    original = SNV(with_std=np.bool_(False), ddof=np.int64(1))
    np.testing.assert_array_equal(
        clone(original).fit(X).transform(X), X - X.mean(axis=1, keepdims=True)
    )


def test_no_sklearn_fallback():
    code = """
import importlib, importlib.abc, sys, numpy as np
from n4m._impl import compat, estimator_base, preprocessing
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn.base": raise ImportError("fallback control")
sys.meta_path.insert(0, Block()); sys.modules.pop("sklearn.base", None)
importlib.reload(compat); importlib.reload(estimator_base); importlib.reload(preprocessing)
assert not compat._SKLEARN_AVAILABLE
x=np.array([[1.,3.,8.,20.],[5.,5.,5.,5.]])
s=preprocessing.SNV().fit(x); s.with_std=False
np.testing.assert_array_equal(s.transform(x),x-x.mean(axis=1,keepdims=True))
old=s.get_params().copy()
try: s.set_params(ddof=1,unknown=1)
except ValueError: pass
else: raise AssertionError("unknown parameter admitted")
assert s.get_params()==old
s.ddof="x"
try: s.transform(x)
except TypeError: pass
else: raise AssertionError("invalid direct mutation admitted")
print("FALLBACK_NATIVE_POSITIVE",compat.__file__,compat._SKLEARN_AVAILABLE)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0


def test_no_unused_native_snv():
    from n4m._impl import native

    assert not hasattr(native, "snv")


def test_identity_and_fixture_copies():
    actual = hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
    expected = os.environ.get("N4M_EXPECTED_LIBRARY_SHA256")
    if expected is not None:
        assert actual == expected, "native library hash differs from explicit expected hash"
    import n4m

    assert Path(n4m.__file__).resolve().is_relative_to(ROOT)
    assert (
        ROOT / "bindings/rust/n4m/src/testdata/snv_parameter_contract_v1.json"
    ).read_bytes() == (
        ROOT / "parity/fixtures/snv_parameter_contract_v1.json"
    ).read_bytes()
    print("OWNER_ORIGIN", n4m.__file__, "LIBRARY", lib._name, actual)


def _numpy_snv_reference(*, with_std=True, ddof=0):
    centered = X - X.mean(axis=1, keepdims=True)
    if not with_std:
        return centered
    scale = X.std(axis=1, ddof=ddof, keepdims=True)
    return centered / np.where(scale == 0, 1.0, scale)


@pytest.mark.parametrize("route", ["direct", "set_params"])
@pytest.mark.parametrize("changed", [{"ddof": 1}, {"with_std": False}])
def test_sklearn_pipeline_retains_fit_after_parameter_change(route, changed):
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    pipeline = Pipeline([("s", SNV())]).fit(X)
    operator = pipeline.named_steps["s"]
    if route == "direct":
        for name, val in changed.items():
            setattr(operator, name, val)
    else:
        pipeline.set_params(**{f"s__{name}": val for name, val in changed.items()})
    expected = _numpy_snv_reference(**changed)
    assert not np.allclose(expected, _numpy_snv_reference())
    for _ in range(2):
        np.testing.assert_allclose(pipeline.transform(X), expected, rtol=0, atol=1e-12)
    check_is_fitted(operator)
    check_is_fitted(pipeline)


def test_sklearn_pipeline_default_stays_fitted():
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    pipeline = Pipeline([("s", SNV())]).fit(X)
    for _ in range(2):
        got = pipeline.transform(X)
        assert got.astype("<f8").tobytes().hex() == DOC["default_bits"]["native_e7"]
        check_is_fitted(pipeline)
        check_is_fitted(pipeline.named_steps["s"])


@pytest.mark.parametrize("route", ["direct", "set_params"])
def test_snv_unfitted_and_clone_remain_unfitted_after_parameter_change(route):
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    fitted = SNV().fit(X)
    for operator in [SNV(), clone(fitted)]:
        with pytest.raises(NotFittedError):
            check_is_fitted(operator)
        if route == "direct":
            operator.ddof = 1
        else:
            operator.set_params(ddof=1)
        for _ in range(2):
            np.testing.assert_allclose(
                operator.transform(X), _numpy_snv_reference(ddof=1), rtol=0, atol=1e-12
            )
            with pytest.raises(NotFittedError):
                check_is_fitted(operator)
        operator.fit(X)
        check_is_fitted(operator)
    check_is_fitted(fitted)


@pytest.mark.parametrize("recovery", ["direct", "set_params"])
def test_sklearn_pipeline_invalid_direct_mutation_then_recovery(recovery):
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    pipeline = Pipeline([("s", SNV())]).fit(X)
    operator = pipeline.named_steps["s"]
    operator.ddof = 1.9
    for _ in range(2):
        with pytest.raises(TypeError, match="ddof"):
            pipeline.transform(X)
        check_is_fitted(operator)
    if recovery == "direct":
        operator.ddof = 1
    else:
        pipeline.set_params(s__ddof=1)
    for _ in range(2):
        np.testing.assert_allclose(
            pipeline.transform(X), _numpy_snv_reference(ddof=1), rtol=0, atol=1e-12
        )
        check_is_fitted(operator)
        check_is_fitted(pipeline)


@pytest.mark.parametrize(
    ("rejected", "error"),
    [
        ({"ddof": 1.9, "with_std": False}, TypeError),
        ({"with_std": "false", "ddof": 1}, TypeError),
        ({"unknown": 1, "ddof": 1}, ValueError),
    ],
)
def test_sklearn_pipeline_rejected_set_params_is_atomic_and_keeps_fit(rejected, error):
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    pipeline = Pipeline([("s", SNV())]).fit(X)
    operator = pipeline.named_steps["s"]
    before = operator.get_params().copy()
    with pytest.raises(error):
        pipeline.set_params(**{f"s__{name}": val for name, val in rejected.items()})
    assert operator.get_params() == before
    for _ in range(2):
        np.testing.assert_allclose(
            pipeline.transform(X), _numpy_snv_reference(), rtol=0, atol=1e-12
        )
        check_is_fitted(operator)
        check_is_fitted(pipeline)


def test_snv_rebuild_balances_real_handles_and_explicit_destroy_unfits(monkeypatch):
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    counts = {"create": 0, "destroy": 0}
    create = lib.n4m_transform_snv_create
    destroy = lib.n4m_transform_snv_destroy

    def tracked_create(*args):
        status = create(*args)
        assert status == 0
        counts["create"] += 1
        return status

    def tracked_destroy(*args):
        counts["destroy"] += 1
        return destroy(*args)

    monkeypatch.setattr(lib, "n4m_transform_snv_create", tracked_create)
    monkeypatch.setattr(lib, "n4m_transform_snv_destroy", tracked_destroy)
    pipeline = Pipeline([("s", SNV())]).fit(X)
    operator = pipeline.named_steps["s"]
    try:
        operator.ddof = 1
        for _ in range(2):
            np.testing.assert_allclose(
                pipeline.transform(X), _numpy_snv_reference(ddof=1), rtol=0, atol=1e-12
            )
        pipeline.set_params(s__with_std=False)
        for _ in range(2):
            np.testing.assert_array_equal(
                pipeline.transform(X), _numpy_snv_reference(with_std=False)
            )
        check_is_fitted(operator)
    finally:
        operator._destroy()
    assert counts == {"create": 3, "destroy": 3}
    with pytest.raises(NotFittedError):
        check_is_fitted(operator)

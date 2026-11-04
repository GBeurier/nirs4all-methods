"""Executed C ABI contracts for optional static analysis engines."""

import ctypes as ct
import json
import os

import numpy as np
import pytest
from n4m import component_analysis as ca
from n4m._errors import N4MError
from n4m._ffi import lib
from n4m._types import (
    FanovaOptionsV1,
    FanovaSummaryV1,
    MixedEffectsOptionsV1,
    MixedEffectsSummaryV1,
)


def test_analysis_abi_layouts():
    assert [
        ct.sizeof(cls)
        for cls in [
            MixedEffectsOptionsV1,
            MixedEffectsSummaryV1,
            FanovaOptionsV1,
            FanovaSummaryV1,
        ]
    ] == [48, 112, 48, 96]


def test_native_normalization_and_atomic_alias():
    np.testing.assert_equal(ca.normalize_loss([0, 3, 8], [2, 2, 4]), [0, 1.5, 2])
    data = np.array([2.0, -1.0, 3.0])
    baseline = np.ones(3)
    original = data.copy()
    ptr = ct.POINTER(ct.c_double)
    assert (
        lib.n4m_component_normalize_loss_v1(
            data.ctypes.data_as(ptr),
            baseline.ctypes.data_as(ptr),
            3,
            data.ctypes.data_as(ptr),
        )
        != 0
    )
    np.testing.assert_equal(data, original)
    with pytest.raises(N4MError):
        ca.normalize_loss([1.0], [0.0])
    with pytest.raises(N4MError):
        ca.normalize_loss([1e308], [1e-308])


def test_mixed_binding_analytic_and_refusal():
    # Balanced random intercept with centered orthogonal within-group predictor.
    g = np.repeat(np.arange(5, dtype=np.int64), 4)
    x = np.tile([-1.5, -0.5, 0.5, 1.5], 5)
    y = (
        3
        + 0.75 * x
        + np.repeat([-2.0, -1.0, 0.0, 1.0, 2.0], 4)
        + np.tile([0.1, -0.1, -0.1, 0.1], 5)
    )
    X = np.column_stack([np.ones(20), x])
    result = ca.mixed_effects(X, y, g)
    if not lib.n4m_mixed_effects_backend_available_v1():
        assert result["reason"] == "mixed_effects_backend_not_compiled"
        return
    assert result["state"] == "ready", result
    np.testing.assert_allclose(result["beta"], [3, 0.75], atol=1e-7)
    assert len(result["variances"]) == 2
    assert result["standard_errors"][1] > 0
    json.dumps(result, allow_nan=False)
    # Duplicate fixed columns cannot produce an authoritative effect.
    refused = ca.mixed_effects(np.column_stack([X, X[:, 1]]), y, g)
    assert refused["state"] == "unavailable" and refused["reason"] == "fixed_rank"
    assert all(v is None for v in refused["beta"])
    repeated = ca.mixed_effects(X[::-1], y[::-1], g[::-1])
    np.testing.assert_allclose(repeated["beta"], result["beta"], atol=1e-7)


def test_fanova_binding_additive_interaction_and_constant():
    # Complete categorical factorial gives an exact ANOVA decomposition oracle.
    X = np.tile(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), (16, 1))
    opts = {"n_trees": 12, "min_samples_leaf": 1, "max_depth": 4}
    result = ca.fanova(X, X[:, 0] + 2 * X[:, 1], [2, 2], [[0, 1], [0, 1]], **opts)
    if not lib.n4m_fanova_backend_available_v1():
        assert result["reason"] == "fanova_backend_not_compiled"
        return
    assert result["state"] == "ready", result
    np.testing.assert_allclose(
        [r["fraction"] for r in result["feature_importance"]], [0.2, 0.8], atol=0.025
    )
    assert result["interactions"][0]["fraction"] < 0.025
    interaction = ca.fanova(
        X, (2 * X[:, 0] - 1) * (2 * X[:, 1] - 1), [2, 2], [[0, 1], [0, 1]], **opts
    )
    assert interaction["state"] == "ready", interaction
    assert interaction["interactions"][0]["fraction"] > 0.95
    constant = ca.fanova(X, np.ones(len(X)), [2, 2], [[0, 1], [0, 1]], **opts)
    assert constant["state"] == "unavailable"
    assert all(r["fraction"] is None for r in constant["feature_importance"])
    validated = ca.fanova(
        X,
        X[:, 0] + 2 * X[:, 1],
        [2, 2],
        [[0, 1], [0, 1]],
        X_valid=X,
        y_valid=X[:, 0] + 2 * X[:, 1],
        **opts,
    )
    assert validated["diagnostics"]["validation_rmse"] is not None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        ca.fanova(X, np.ones(len(X)), [2, 2], [[0, 1], [0, 1]], X_valid=X)


@pytest.mark.parametrize("types", [[-4294967296, 2], [-4294967294, 2]])
def test_fanova_rejects_negative_types_before_unsigned_conversion(types):
    if not lib.n4m_fanova_backend_available_v1():
        pytest.skip("requires the optional enabled backend")
    with pytest.raises(ValueError):
        ca.fanova([[0, 0], [1, 1]], [0, 1], types, [[0, 1], [0, 1]])


def test_unaligned_input_buffers_are_marshaled_safely():
    raw = bytearray(1 + 16)
    unaligned = np.ndarray((2,), dtype=np.float64, buffer=raw, offset=1)
    unaligned[:] = [2, 6]
    np.testing.assert_equal(ca.normalize_loss(unaligned, [4, 3]), [0.5, 2])
    if lib.n4m_fanova_backend_available_v1():
        types = np.ndarray((1,), dtype=np.uint32, buffer=bytearray(5), offset=1)
        types[:] = 2
        result = ca.fanova(
            [[0], [1], [0], [1]],
            [0, 1, 0, 1],
            types,
            [[0, 1]],
            n_trees=4,
            min_samples_leaf=1,
        )
        assert result["state"] == "ready", result


def test_release_wheel_contains_enabled_backends_when_requested():
    if os.environ.get("N4M_WHEEL_ANALYSIS") == "1":
        assert lib.n4m_mixed_effects_backend_available_v1()
        assert lib.n4m_fanova_backend_available_v1()

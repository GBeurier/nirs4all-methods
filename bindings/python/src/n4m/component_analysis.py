# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Thin bindings for statically compiled component analysis engines.

Design construction, labels, matching, domains and dependence policy belong to
callers. All fitting, loss normalization and variance attribution run natively.
"""

from __future__ import annotations

import ctypes as ct
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._errors import N4MError
from ._ffi import lib
from ._types import (
    FanovaOptionsV1,
    FanovaSummaryV1,
    MatrixView,
    MixedEffectsOptionsV1,
    MixedEffectsSummaryV1,
)

_DOUBLE = ct.POINTER(ct.c_double)


def _check(status: int) -> None:
    if status:
        raise N4MError(status, "native component analysis rejected the input")


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _matrix(values: ArrayLike, *, integer: bool = False, vector: bool = False):
    raw = np.asarray(values)
    if integer and (
        raw.dtype.kind not in "iu"
        or raw.dtype.kind == "u"
        and np.any(raw > np.iinfo(np.int64).max)
    ):
        raise ValueError("group IDs must be exact nonnegative int64 values")
    x = np.asarray(raw, dtype=np.int64 if integer else np.float64)
    if vector and x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("expected a nonempty matrix")
    if any(stride < 0 or stride % x.itemsize for stride in x.strides):
        x = np.ascontiguousarray(x)
    return x, MatrixView(
        x.ctypes.data,
        *x.shape,
        x.strides[0] // x.itemsize,
        x.strides[1] // x.itemsize,
        4 if integer else 1,
        0,
    )


def _options(cls, init, values: dict[str, Any]):
    result = cls()
    _check(init(ct.byref(result)))
    fields = dict(cls._fields_)
    allowed = fields.keys() - {"struct_size", "reserved"}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"Unsupported component analysis option: {key}")
        if fields[key] is not ct.c_double:
            if key == "reml" and isinstance(value, (bool, np.bool_)):
                value = int(value)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{key} must be an integer")
            bits = ct.sizeof(fields[key]) * 8
            lower, upper = (
                (0, 2**bits - 1)
                if fields[key] is ct.c_uint64
                else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            )
            if not lower <= int(value) <= upper:
                raise ValueError(f"{key} exceeds its ABI range")
        setattr(result, key, value)
    resolved = {key: getattr(result, key) for key in allowed}
    return result, resolved


def normalize_loss(values: ArrayLike, baselines: ArrayLike) -> np.ndarray:
    """Return native dimensionless nonnegative loss/baseline ratios."""
    x = np.require(values, dtype=np.float64, requirements=["C", "A"])
    b = np.require(baselines, dtype=np.float64, requirements=["C", "A"])
    if x.shape != b.shape or x.size == 0:
        raise ValueError("values and baselines must have the same nonempty shape")
    out = np.empty_like(x)
    _check(
        lib.n4m_component_normalize_loss_v1(
            x.ctypes.data_as(_DOUBLE),
            b.ctypes.data_as(_DOUBLE),
            x.size,
            out.ctypes.data_as(_DOUBLE),
        )
    )
    return out


def mixed_effects(
    X: ArrayLike, y: ArrayLike, groups: ArrayLike, **options: Any
) -> dict[str, Any]:
    """Gaussian ML/REML with explicit fixed X and random-intercept group IDs.

    Include any intercept explicitly in X. Estimates preserve column order.
    SEs are conditional plug-in values; variance-parameter uncertainty is omitted.
    """
    opt, resolved = _options(
        MixedEffectsOptionsV1, lib.n4m_mixed_effects_options_init_v1, options
    )
    if not lib.n4m_mixed_effects_backend_available_v1():
        return {
            "schema_version": "n4m.mixed_effects/1",
            "state": "unavailable",
            "reason": "mixed_effects_backend_not_compiled",
            "options": resolved,
            "beta": None,
            "standard_errors": None,
            "covariance": None,
            "variances": None,
            "uncertainty_kind": "conditional_plugin_standard_error",
            "diagnostics": {},
            "warnings": [],
        }
    x, xv = _matrix(X)
    _yy, yv = _matrix(y, vector=True)
    g, gv = _matrix(groups, integer=True, vector=True)
    p, k = x.shape[1], g.shape[1]
    beta, cov, se, var = np.empty(p), np.empty((p, p)), np.empty(p), np.empty(k + 1)
    summary = MixedEffectsSummaryV1()
    summary.struct_size = ct.sizeof(summary)
    _check(
        lib.n4m_mixed_effects_fit_v1(
            ct.byref(xv),
            ct.byref(yv),
            ct.byref(gv),
            ct.byref(opt),
            beta.ctypes.data_as(_DOUBLE),
            p,
            cov.ctypes.data_as(_DOUBLE),
            p * p,
            se.ctypes.data_as(_DOUBLE),
            p,
            var.ctypes.data_as(_DOUBLE),
            k + 1,
            ct.byref(summary),
        )
    )
    states = [
        "fitted",
        "fitted_boundary",
        "rank_deficient",
        "unidentifiable",
        "nonconverged",
        "numerical_refusal",
    ]
    reasons = [
        None,
        "fixed_rank",
        "no_residual_df",
        "constant_response",
        "insufficient_group_levels",
        "confounded_variances",
        "zero_residual",
        "optimizer_budget",
        "optimizer_roundoff",
        "optimizer_failure",
        "nonfinite_result",
        "engine_rejected",
    ]
    diagnostics = {
        name: _nullable(getattr(summary, name))
        if typ is ct.c_double
        else getattr(summary, name)
        for name, typ in summary._fields_
        if name != "struct_size"
    }
    diagnostics["fit_state"] = states[summary.fit_state]
    return {
        "schema_version": "n4m.mixed_effects/1",
        "state": "ready" if summary.fit_state <= 1 else "unavailable",
        "reason": reasons[summary.reason],
        "beta": [_nullable(v) for v in beta],
        "standard_errors": [_nullable(v) for v in se],
        "covariance": [[_nullable(v) for v in row] for row in cov],
        "variances": [_nullable(v) for v in var],
        "diagnostics": diagnostics,
        "options": resolved,
        "uncertainty_kind": "conditional_plugin_standard_error",
        "warnings": [
            "Gaussian random-intercept model; estimates are conditional associations, not causal effects.",
            "Plug-in SE excludes variance-parameter uncertainty and does not establish independent folds.",
        ]
        + (
            ["Boundary variance estimate: inspect identifiability and model adequacy."]
            if summary.boundary
            else []
        ),
    }


def fanova(
    X: ArrayLike,
    y: ArrayLike,
    feature_types: ArrayLike,
    bounds: ArrayLike,
    *,
    n_trees: int = 32,
    max_depth: int = 10,
    min_samples_leaf: int = 2,
    max_features: int = 0,
    seed: int = 42,
    max_order: int = 2,
    max_marginal_evaluations: int = 1_000_000,
    X_valid: ArrayLike | None = None,
    y_valid: ArrayLike | None = None,
) -> dict[str, Any]:
    """Exact uniform-domain per-tree fANOVA of a newly fitted regression forest."""
    opt, resolved = _options(
        FanovaOptionsV1,
        lib.n4m_fanova_options_init_v1,
        {
            "num_trees": n_trees,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "seed": seed,
            "max_order": max_order,
            "max_marginal_evaluations": max_marginal_evaluations,
        },
    )
    base = {
        "schema_version": "n4m.fanova/1",
        "options": resolved,
        "domain_measure": "uniform_product",
        "aggregation": "mean_per_tree_normalized_components",
        "uncertainty_kind": "tree_dispersion_not_confidence",
        "warnings": [
            "Surrogate variance attribution under independent uniform domains is not a causal effect.",
            "Tree dispersion is not a confidence interval.",
        ],
    }
    if (X_valid is None) != (y_valid is None):
        raise ValueError("X_valid and y_valid must be supplied together")
    if not lib.n4m_fanova_backend_available_v1():
        return {
            **base,
            "state": "unavailable",
            "reason": "fanova_backend_not_compiled",
            "feature_importance": [],
            "interactions": [],
            "diagnostics": {},
        }
    x, xv = _matrix(X)
    _yy, yv = _matrix(y, vector=True)
    _b, bv = _matrix(bounds)
    typesraw = np.asarray(feature_types)
    if (
        typesraw.ndim != 1
        or typesraw.size != x.shape[1]
        or typesraw.dtype.kind not in "iu"
        or np.any(typesraw < 0)
        or np.any(typesraw > 128)
    ):
        raise ValueError(
            "feature_types must contain one integer code 0..128 per feature"
        )
    types = np.require(typesraw, dtype=np.uint32, requirements=["C", "A"])
    if X_valid is not None:
        _vx, vxv = _matrix(X_valid)
        _vy, vyv = _matrix(y_valid, vector=True)
    else:
        vxv = vyv = None
    p = x.shape[1]
    mains = np.empty((p, 2))
    pairs = np.empty((p, p, 2))
    summary = FanovaSummaryV1()
    summary.struct_size = ct.sizeof(summary)
    _check(
        lib.n4m_fanova_fit_v1(
            ct.byref(xv),
            ct.byref(yv),
            ct.byref(bv),
            types.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            ct.byref(vxv) if vxv is not None else None,
            ct.byref(vyv) if vyv is not None else None,
            ct.byref(opt),
            mains.ctypes.data_as(_DOUBLE),
            mains.size,
            pairs.ctypes.data_as(_DOUBLE),
            pairs.size,
            ct.byref(summary),
        )
    )
    return {
        **base,
        "state": "ready" if summary.fit_state == 0 else "unavailable",
        "reason": [
            None,
            "zero_surrogate_variance",
            "zero_surrogate_variance",
            "numerical_refusal",
        ][summary.fit_state],
        "feature_importance": [
            {"index": i, "fraction": _nullable(row[0]), "tree_sd": _nullable(row[1])}
            for i, row in enumerate(mains)
        ],
        "interactions": [
            {
                "indices": [i, j],
                "fraction": _nullable(pairs[i, j, 0]),
                "tree_sd": _nullable(pairs[i, j, 1]),
            }
            for i in range(p)
            for j in range(i + 1, p)
        ]
        if max_order == 2
        else [],
        "diagnostics": {
            "fit_state": summary.fit_state,
            "observations": summary.observations,
            "features": summary.features,
            "trees": summary.trees,
            "informative_trees": summary.informative_trees,
            "total_variance_mean": _nullable(summary.mean_tree_variance),
            "remaining_fraction": _nullable(summary.remainder_fraction),
            "train_rmse": _nullable(summary.train_rmse),
            "validation_rmse": _nullable(summary.validation_rmse),
            "validation_r2": _nullable(summary.validation_r2),
            "marginal_evaluations": summary.marginal_evaluations,
        },
    }

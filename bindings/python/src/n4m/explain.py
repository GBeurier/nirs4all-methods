# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Model explanation (attribution) bindings over ``n4m/explain.h``.

Every attribution is computed by libn4m. This module marshals inputs, drives
the native ask/tell session (the caller's ``predict_batch`` predicts the
perturbed rows the session requests, with any backend) and reports the native
results. The explained feature space and the output space are whatever the
caller's predictor consumes and returns; nothing is relabelled here.

Model-agnostic explainers (:func:`explain`):

* ``exact_shap`` -- exact group Shapley values (2^G coalitions, G <= 16).
* ``permutation_shap`` -- antithetic permutation sampling; additive by
  construction, with a Monte-Carlo standard error.
* ``kernel_shap`` -- Shapley-kernel weighted least squares with the
  efficiency constraint enforced (experimental).
* ``lime`` -- local ridge surrogate over binary group masks; not additive
  (experimental).
* ``occlusion`` -- per-group ablation against the baseline; not additive.

Model-specific exact explainers: :func:`linear_shap` (affine models such as
PLS or Ridge coefficients) and :func:`tree_shap` (path-dependent TreeSHAP over
a generic flat :class:`TreeEnsemble`; converting a specific library's trees to
that layout belongs to the host).
"""

from __future__ import annotations

import ctypes as ct
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._context import Context
from ._errors import N4MError
from ._ffi import lib
from ._types import ExplainOptionsV1, MatrixView, Status

_DOUBLE = ct.POINTER(ct.c_double)
_INT64 = ct.POINTER(ct.c_int64)
_UINT8 = ct.POINTER(ct.c_uint8)

_METHODS = {
    "exact_shap": 0,
    "permutation_shap": 1,
    "kernel_shap": 2,
    "lime": 3,
    "occlusion": 4,
}
_BASELINES = {"background": 0, "mean": 1, "interpolate": 2}
_EXACTNESS = {
    "exact_shap": "exact",
    "permutation_shap": "sampled",
    "kernel_shap": "sampled",
    "lime": "surrogate",
    "occlusion": "perturbation",
}
_NON_ADDITIVE = frozenset({"lime", "occlusion"})


@dataclass(frozen=True)
class AttributionResult:
    """Native attribution result.

    Attributes:
        kind: Explainer kind (``exact_shap``, ``permutation_shap``,
            ``kernel_shap``, ``lime``, ``occlusion``, ``linear_shap`` or
            ``tree_shap``).
        baseline: Perturbation baseline, ``None`` for linear/tree.
        exactness: ``exact``, ``sampled``, ``surrogate`` or ``perturbation``.
        additive: Whether ``expected + attributions.sum(axis=1)`` equals the
            prediction by construction.
        group_aggregation: ``group_game`` (groups are coalition players),
            ``sum_of_feature_values`` (tree: per-feature values summed per
            group) or ``none`` (per-feature, ungrouped linear/tree).
        attributions: ``(n_rows, n_groups, n_outputs)`` attributions.
        expected: ``(n_rows, n_outputs)`` expected value (LIME: local
            intercept; occlusion: baseline prediction).
        predictions: ``(n_rows, n_outputs)`` model output for each row.
        std_error: Monte-Carlo standard error (permutation only), else
            ``None``.
        additivity_max_abs_error: ``max |prediction - expected - sum|``
            diagnostic, ``None`` for non-additive explainers.
        evaluations: Model rows evaluated by ``predict_batch`` (0 for
            linear/tree).
        settings: Effective settings, keyed as in ``nirs4all.explanation.v1``.
        engine: ``{"name": "nirs4all-methods", "version": ..., "abi": ...}``.
    """

    kind: str
    baseline: str | None
    exactness: str
    additive: bool
    group_aggregation: str
    attributions: np.ndarray
    expected: np.ndarray
    predictions: np.ndarray
    std_error: np.ndarray | None
    additivity_max_abs_error: float | None
    evaluations: int
    settings: dict[str, Any]
    engine: dict[str, str]


@dataclass(frozen=True)
class TreeEnsemble:
    """Generic flat tree ensemble (see ``n4m_explain_tree_path_dependent_v1``).

    Nodes of all trees are concatenated; tree ``t`` owns nodes
    ``[tree_offsets[t], tree_offsets[t + 1])`` and child indices are LOCAL to
    the tree (``-1`` for none). ``feature`` is ``-1`` for leaves; ``value``
    holds ``n_nodes * n_outputs`` leaf values (internal values ignored).
    ``decision_type`` 0 sends ``x <= threshold`` left (scikit-learn,
    LightGBM), 1 sends ``x < threshold`` left (XGBoost). The ensemble output is
    ``base_offset + tree_scale * sum_t leaf_t`` (``tree_scale = 1 / n_trees``
    for averaging forests, 1 for boosting). ``default_left`` (optional) routes
    NaN inputs; without it NaN input is rejected.
    """

    tree_offsets: ArrayLike
    feature: ArrayLike
    threshold: ArrayLike
    left: ArrayLike
    right: ArrayLike
    cover: ArrayLike
    value: ArrayLike
    n_outputs: int
    decision_type: int = 0
    default_left: ArrayLike | None = None
    base_offset: ArrayLike | None = None
    tree_scale: float = 1.0


# ---------------------------------------------------------------------------
# Marshalling helpers
# ---------------------------------------------------------------------------


def _check(ctx: Context | None, status: int, where: str) -> None:
    if status != Status.OK:
        text = ""
        if ctx is not None:
            detail = lib.n4m_context_last_error(ctx.handle)
            text = ct.string_at(detail).decode() if detail else ""
        raise N4MError(status, f"{where}: {text}" if text else where)


def _matrix(name: str, values: ArrayLike, *, allow_empty_rows: bool = False):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or x.shape[0] < (0 if allow_empty_rows else 1):
        raise ValueError(f"{name} must be a nonempty 2-D array")
    # Borrow positive-stride views without a layout copy (strided matrix view).
    if any(stride <= 0 or stride % x.itemsize for stride in x.strides):
        x = np.ascontiguousarray(x)
    view = MatrixView(
        x.ctypes.data,
        x.shape[0],
        x.shape[1],
        x.strides[0] // x.itemsize if x.shape[0] > 1 else x.shape[1],
        x.strides[1] // x.itemsize if x.shape[1] > 1 else 1,
        1,
        0,
    )
    return x, view


def _int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _int64_array(name: str, values: ArrayLike, length: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or (length is not None and raw.shape[0] != length):
        expected = f" of length {length}" if length is not None else ""
        raise ValueError(f"{name} must be a 1-D integer array{expected}")
    if raw.dtype.kind not in "iu":
        if raw.size and (raw.dtype.kind != "f" or not np.all(np.isfinite(raw))
                         or np.any(raw != np.round(raw))):
            raise ValueError(f"{name} must contain integers")
    return np.ascontiguousarray(raw, dtype=np.int64)


def _f64_vector(name: str, values: ArrayLike, length: int) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if out.shape[0] != length:
        raise ValueError(f"{name} must hold {length} values")
    return out


def _grouping(
    feature_group: ArrayLike | None, n_groups: int | None, n_features: int
) -> tuple[np.ndarray | None, int]:
    if feature_group is None:
        if n_groups is not None and n_groups != n_features:
            raise ValueError("n_groups requires feature_group")
        return None, n_features
    groups = _int64_array("feature_group", feature_group, n_features)
    if n_groups is None:
        n_groups = int(groups.max()) + 1 if groups.size else 0
    return groups, _int("n_groups", n_groups, 1)


def _ptr(array: np.ndarray | None, kind):
    return None if array is None else array.ctypes.data_as(kind)


def _engine() -> dict[str, str]:
    raw = ct.c_char_p(lib.n4m_get_version_string()).value
    abi = (
        f"{int(lib.n4m_get_abi_version_major())}."
        f"{int(lib.n4m_get_abi_version_minor())}."
        f"{int(lib.n4m_get_abi_version_patch())}"
    )
    return {
        "name": "nirs4all-methods",
        "version": raw.decode("utf-8") if raw else "",
        "abi": abi,
    }


def _additivity_error(
    attributions: np.ndarray, expected: np.ndarray, predictions: np.ndarray
) -> float:
    # Reported diagnostic only: max |prediction - expected - sum_g attribution|.
    if attributions.size == 0:
        return 0.0
    residual = predictions - expected - attributions.sum(axis=1)
    return float(np.max(np.abs(residual)))


def _kernel_enumerates(n_coalitions: int, n_groups: int) -> bool:
    """Whether the native kernel budget covers all 2^G - 2 proper coalitions.

    Mirrors the documented auto budget of ``n4m_explain_options_v1_t``
    (``0`` = ``min(2^G - 2, 2G + 2048)``); the weighted least squares is then
    the exact Shapley solve.
    """
    if n_groups > 52:
        return False
    budget = n_coalitions if n_coalitions > 0 else 2 * n_groups + 2048
    return budget >= 2**n_groups - 2


def _predictions(raw: ArrayLike, rows: int, n_outputs: int) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float64)
    if out.ndim == 1 and n_outputs == 1:
        out = out.reshape(-1, 1)
    if out.shape != (rows, n_outputs):
        raise ValueError(
            f"predict_batch returned shape {np.shape(raw)}, expected ({rows},) or "
            f"({rows}, {n_outputs})"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError("predict_batch returned a nonfinite prediction")
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Ask/tell session
# ---------------------------------------------------------------------------


def explain(
    predict_batch: Callable[[np.ndarray], ArrayLike],
    X: ArrayLike,
    *,
    background: ArrayLike | None = None,
    feature_group: ArrayLike | None = None,
    n_groups: int | None = None,
    method: str = "permutation_shap",
    baseline: str = "background",
    n_outputs: int | None = None,
    n_permutations: int = 10,
    n_coalitions: int = 0,
    lime_kernel_width: float = 0.0,
    lime_ridge: float = 1.0,
    max_batch_rows: int = 4096,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> AttributionResult:
    """Explain any predictor through a native ask/tell session.

    Args:
        predict_batch: Receives a C-contiguous float64 ``(rows, n_features)``
            array of perturbed rows in the explained feature space and returns
            ``(rows,)`` or ``(rows, n_outputs)`` finite predictions.
        X: ``(n_rows, n_features)`` rows to explain.
        background: Background rows; required for the ``background`` and
            ``mean`` baselines, optional for ``interpolate``.
        feature_group: Optional ``(n_features,)`` partition into groups
            ``[0, n_groups)``; ``None`` means one group per feature.
        n_groups: Number of groups (default ``max(feature_group) + 1``).
        method: ``exact_shap``, ``permutation_shap``, ``kernel_shap``,
            ``lime`` or ``occlusion``.
        baseline: ``background``, ``mean`` or ``interpolate``.
        n_outputs: Prediction columns; inferred by predicting ``X[:1]`` once
            when ``None`` (that probe row counts in ``evaluations``).
        n_permutations: Antithetic permutation pairs per row.
        n_coalitions: Kernel/LIME coalitions per row (0 = native auto).
        lime_kernel_width: LIME kernel width (<= 0 = native auto).
        lime_ridge: LIME ridge penalty.
        max_batch_rows: Upper bound of rows per ``predict_batch`` call.
        seed: Sampling seed (nonnegative integer).
        progress: Optional ``progress(done_rows, total_rows)`` callback.

    Returns:
        The native :class:`AttributionResult`.

    Raises:
        ValueError, TypeError: Invalid arguments or predictions.
        N4MError: The native session rejected the input.
    """
    if not callable(predict_batch):
        raise TypeError("predict_batch must be callable")
    if method not in _METHODS:
        raise ValueError(f"Unsupported explain method: {method}")
    if baseline not in _BASELINES:
        raise ValueError(f"Unsupported explain baseline: {baseline}")
    n_permutations = _int("n_permutations", n_permutations, 1)
    n_coalitions = _int("n_coalitions", n_coalitions, 0)
    max_batch_rows = _int("max_batch_rows", max_batch_rows, 1)
    seed = _int("seed", seed, 0)
    if seed >= 2**64:
        raise ValueError("seed must fit in uint64")
    for name, value in (("lime_kernel_width", lime_kernel_width), ("lime_ridge", lime_ridge)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    x, x_view = _matrix("X", X)
    n_features = x.shape[1]
    if background is None:
        if baseline != "interpolate":
            raise ValueError(f"the {baseline} baseline requires background rows")
        bg = np.empty((0, n_features), dtype=np.float64)
    else:
        bg = background
    bg, bg_view = _matrix("background", bg, allow_empty_rows=True)
    if bg.shape[1] != n_features:
        raise ValueError("background must have the same number of columns as X")
    groups, group_count = _grouping(feature_group, n_groups, n_features)

    evaluations = 0
    inferred = n_outputs is None
    if n_outputs is None:
        probe = np.asarray(predict_batch(np.ascontiguousarray(x[:1])), dtype=np.float64)
        evaluations += 1
        if probe.ndim == 1 and probe.shape[0] == 1:
            n_outputs = 1
        elif probe.ndim == 2 and probe.shape[0] == 1:
            n_outputs = probe.shape[1]
        else:
            raise ValueError("predict_batch must return (1,) or (1, n_outputs) for one row")
    n_outputs = _int("n_outputs", n_outputs, 1)

    options = ExplainOptionsV1()
    with Context() as ctx:
        _check(ctx, lib.n4m_explain_options_init_v1(ct.byref(options)), "n4m_explain_options_init_v1")
        options.method = _METHODS[method]
        options.baseline = _BASELINES[baseline]
        options.n_outputs = n_outputs
        options.n_permutations = n_permutations
        options.n_coalitions = n_coalitions
        options.lime_kernel_width = float(lime_kernel_width)
        options.lime_ridge = float(lime_ridge)
        options.max_batch_rows = max_batch_rows
        options.seed = seed
        session = ct.c_void_p()
        _check(
            ctx,
            lib.n4m_explain_session_create_v1(
                ctx.handle,
                ct.byref(x_view),
                ct.byref(bg_view),
                _ptr(groups, _INT64),
                group_count,
                ct.byref(options),
                ct.byref(session),
            ),
            "n4m_explain_session_create_v1",
        )
        try:
            dims = [ct.c_int64() for _ in range(5)]
            _check(
                ctx,
                lib.n4m_explain_session_info_v1(session, *(ct.byref(d) for d in dims)),
                "n4m_explain_session_info_v1",
            )
            n_rows, _, n_groups_out, _, total = (int(d.value) for d in dims)
            capacity = max(1, min(max_batch_rows, total))
            written = ct.c_int64()
            done = 0
            while True:
                # Fresh buffer per batch: the caller may keep the rows it received.
                batch = np.empty((capacity, n_features), dtype=np.float64)
                _check(
                    ctx,
                    lib.n4m_explain_session_next_batch_v1(
                        ctx.handle, session, batch.ctypes.data_as(_DOUBLE), capacity,
                        ct.byref(written),
                    ),
                    "n4m_explain_session_next_batch_v1",
                )
                rows = int(written.value)
                if rows == 0:
                    break
                predicted = _predictions(predict_batch(batch[:rows]), rows, n_outputs)
                _check(
                    ctx,
                    lib.n4m_explain_session_submit_v1(
                        ctx.handle, session, predicted.ctypes.data_as(_DOUBLE), rows
                    ),
                    "n4m_explain_session_submit_v1",
                )
                done += rows
                evaluations += rows
                if progress is not None:
                    progress(done, total)

            attributions = np.empty((n_rows, n_groups_out, n_outputs), dtype=np.float64)
            expected = np.empty((n_rows, n_outputs), dtype=np.float64)
            predictions = np.empty((n_rows, n_outputs), dtype=np.float64)
            std_error = (
                np.empty_like(attributions) if method == "permutation_shap" else None
            )
            _check(
                ctx,
                lib.n4m_explain_session_result_v1(
                    ctx.handle,
                    session,
                    attributions.ctypes.data_as(_DOUBLE),
                    attributions.size,
                    expected.ctypes.data_as(_DOUBLE),
                    expected.size,
                    predictions.ctypes.data_as(_DOUBLE),
                    predictions.size,
                    _ptr(std_error, _DOUBLE),
                    0 if std_error is None else std_error.size,
                ),
                "n4m_explain_session_result_v1",
            )
        finally:
            lib.n4m_explain_session_destroy(session)

    additive = method not in _NON_ADDITIVE
    exactness = _EXACTNESS[method]
    if method == "kernel_shap" and _kernel_enumerates(n_coalitions, n_groups_out):
        exactness = "exact"
    return AttributionResult(
        kind=method,
        baseline=baseline,
        exactness=exactness,
        additive=additive,
        group_aggregation="group_game",
        attributions=attributions,
        expected=expected,
        predictions=predictions,
        std_error=std_error,
        additivity_max_abs_error=(
            _additivity_error(attributions, expected, predictions) if additive else None
        ),
        evaluations=evaluations,
        settings={
            "method": method,
            "baseline": baseline,
            # The interpolate baseline ignores background rows.
            "n_background": 0 if baseline == "interpolate" else int(bg.shape[0]),
            "n_permutations": n_permutations if method == "permutation_shap" else None,
            "n_coalitions": n_coalitions if method in ("kernel_shap", "lime") else None,
            "lime_kernel_width": float(lime_kernel_width) if method == "lime" else None,
            "lime_ridge": float(lime_ridge) if method == "lime" else None,
            "max_batch_rows": max_batch_rows,
            "seed": seed,
            "n_outputs_inferred": inferred,
            "total_evaluations": total,
        },
        engine=_engine(),
    )


# ---------------------------------------------------------------------------
# Model-specific exact explainers
# ---------------------------------------------------------------------------


def linear_shap(
    X: ArrayLike,
    coefficients: ArrayLike,
    *,
    reference: ArrayLike,
    intercept: ArrayLike | None = None,
    feature_group: ArrayLike | None = None,
    n_groups: int | None = None,
) -> AttributionResult:
    """Exact interventional SHAP of ``f(z) = z @ coefficients + intercept``.

    Args:
        X: ``(n_rows, n_features)`` rows in the feature space of the
            coefficients.
        coefficients: ``(n_features, n_outputs)`` (or ``(n_features,)``).
        reference: ``(n_features,)`` background column means.
        intercept: Optional ``(n_outputs,)`` intercept (default zero).
        feature_group: Optional partition; linear group sums are exact group
            Shapley values.
        n_groups: Number of groups (default ``max(feature_group) + 1``).

    Returns:
        The native :class:`AttributionResult` (``evaluations == 0``).
    """
    x, x_view = _matrix("X", X)
    coef = np.asarray(coefficients, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    coef, coef_view = _matrix("coefficients", coef)
    n_rows, n_features = x.shape
    n_outputs = coef.shape[1]
    if coef.shape[0] != n_features:
        raise ValueError("coefficients must be (n_features, n_outputs)")
    ref = _f64_vector("reference", reference, n_features)
    icpt = None if intercept is None else _f64_vector("intercept", intercept, n_outputs)
    groups, group_count = _grouping(feature_group, n_groups, n_features)
    attributions = np.empty((n_rows, group_count, n_outputs), dtype=np.float64)
    expected = np.empty((n_rows, n_outputs), dtype=np.float64)
    predictions = np.empty((n_rows, n_outputs), dtype=np.float64)
    with Context() as ctx:
        _check(
            ctx,
            lib.n4m_explain_linear_v1(
                ctx.handle,
                ct.byref(x_view),
                ct.byref(coef_view),
                _ptr(icpt, _DOUBLE),
                ref.ctypes.data_as(_DOUBLE),
                _ptr(groups, _INT64),
                group_count,
                attributions.ctypes.data_as(_DOUBLE),
                attributions.size,
                expected.ctypes.data_as(_DOUBLE),
                expected.size,
                predictions.ctypes.data_as(_DOUBLE),
                predictions.size,
            ),
            "n4m_explain_linear_v1",
        )
    return AttributionResult(
        kind="linear_shap",
        baseline=None,
        exactness="exact",
        additive=True,
        group_aggregation="none" if groups is None else "group_game",
        attributions=attributions,
        expected=expected,
        predictions=predictions,
        std_error=None,
        additivity_max_abs_error=_additivity_error(attributions, expected, predictions),
        evaluations=0,
        settings={"method": "linear_shap", "has_intercept": icpt is not None},
        engine=_engine(),
    )


def tree_shap(
    X: ArrayLike,
    ensemble: TreeEnsemble,
    *,
    feature_group: ArrayLike | None = None,
    n_groups: int | None = None,
) -> AttributionResult:
    """Exact path-dependent TreeSHAP over a generic flat tree ensemble.

    Grouped values are SUMS of per-feature Shapley values
    (``group_aggregation == "sum_of_feature_values"``), not Shapley values of
    the group game.
    """
    if not isinstance(ensemble, TreeEnsemble):
        raise TypeError("ensemble must be a TreeEnsemble")
    x, x_view = _matrix("X", X)
    n_rows, n_features = x.shape
    n_outputs = _int("n_outputs", ensemble.n_outputs, 1)
    offsets = _int64_array("tree_offsets", ensemble.tree_offsets)
    n_trees = offsets.shape[0] - 1
    feature = _int64_array("feature", ensemble.feature)
    n_nodes = feature.shape[0]
    if n_trees < 1 or offsets[0] != 0 or offsets[-1] != n_nodes or np.any(np.diff(offsets) <= 0):
        raise ValueError(
            "tree_offsets must hold n_trees + 1 strictly increasing offsets from 0 to the node count"
        )
    left = _int64_array("left", ensemble.left, n_nodes)
    right = _int64_array("right", ensemble.right, n_nodes)
    threshold = _f64_vector("threshold", ensemble.threshold, n_nodes)
    cover = _f64_vector("cover", ensemble.cover, n_nodes)
    value = _f64_vector("value", ensemble.value, n_nodes * n_outputs)
    default_left = None
    if ensemble.default_left is not None:
        default_left = np.ascontiguousarray(np.asarray(ensemble.default_left), dtype=np.uint8)
        if default_left.shape != (n_nodes,):
            raise ValueError("default_left must hold one flag per node")
    base = (
        None
        if ensemble.base_offset is None
        else _f64_vector("base_offset", ensemble.base_offset, n_outputs)
    )
    if ensemble.decision_type not in (0, 1):
        raise ValueError("decision_type must be 0 (<=) or 1 (<)")
    tree_scale = float(ensemble.tree_scale)
    if not math.isfinite(tree_scale):
        raise ValueError("tree_scale must be finite")
    groups, group_count = _grouping(feature_group, n_groups, n_features)
    attributions = np.empty((n_rows, group_count, n_outputs), dtype=np.float64)
    expected = np.empty((n_rows, n_outputs), dtype=np.float64)
    predictions = np.empty((n_rows, n_outputs), dtype=np.float64)
    with Context() as ctx:
        _check(
            ctx,
            lib.n4m_explain_tree_path_dependent_v1(
                ctx.handle,
                ct.byref(x_view),
                n_trees,
                offsets.ctypes.data_as(_INT64),
                feature.ctypes.data_as(_INT64),
                threshold.ctypes.data_as(_DOUBLE),
                left.ctypes.data_as(_INT64),
                right.ctypes.data_as(_INT64),
                _ptr(default_left, _UINT8),
                cover.ctypes.data_as(_DOUBLE),
                value.ctypes.data_as(_DOUBLE),
                n_outputs,
                int(ensemble.decision_type),
                _ptr(base, _DOUBLE),
                tree_scale,
                _ptr(groups, _INT64),
                group_count,
                attributions.ctypes.data_as(_DOUBLE),
                attributions.size,
                expected.ctypes.data_as(_DOUBLE),
                expected.size,
                predictions.ctypes.data_as(_DOUBLE),
                predictions.size,
            ),
            "n4m_explain_tree_path_dependent_v1",
        )
    return AttributionResult(
        kind="tree_shap",
        baseline=None,
        exactness="exact",
        additive=True,
        group_aggregation="none" if groups is None else "sum_of_feature_values",
        attributions=attributions,
        expected=expected,
        predictions=predictions,
        std_error=None,
        additivity_max_abs_error=_additivity_error(attributions, expected, predictions),
        evaluations=0,
        settings={
            "method": "tree_shap",
            "feature_perturbation": "tree_path_dependent",
            "n_trees": n_trees,
            "decision_type": int(ensemble.decision_type),
            "tree_scale": tree_scale,
        },
        engine=_engine(),
    )


# ---------------------------------------------------------------------------
# Grouping and summary helpers
# ---------------------------------------------------------------------------


def groups_fixed_width(n_features: int, width: int) -> tuple[np.ndarray, int]:
    """Contiguous fixed-width partition: feature ``j`` -> group ``j // width``.

    Returns:
        ``(feature_group, n_groups)`` with an int64 ``(n_features,)`` array.
    """
    n_features = _int("n_features", n_features, 1)
    width = _int("width", width, 1)
    groups = np.empty(n_features, dtype=np.int64)
    count = ct.c_int64()
    _check(
        None,
        lib.n4m_explain_groups_fixed_width_v1(
            n_features, width, groups.ctypes.data_as(_INT64), n_features, ct.byref(count)
        ),
        "n4m_explain_groups_fixed_width_v1",
    )
    return groups, int(count.value)


def groups_from_ranges(
    n_features: int, ranges: Sequence[Sequence[int]] | ArrayLike
) -> tuple[np.ndarray, int, np.ndarray]:
    """Partition from half-open ``[start, end)`` feature-index ranges.

    Range ``r`` becomes group ``r``; each maximal run of uncovered features
    becomes one further group, numbered after the ranges in feature order.

    Returns:
        ``(feature_group, n_groups, range_of_group)``; ``range_of_group[g]`` is
        the source range index or ``-1`` for an uncovered run.
    """
    n_features = _int("n_features", n_features, 1)
    raw = np.asarray(ranges)
    if raw.size == 0:
        raw = raw.reshape(0, 2)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("ranges must be a sequence of (start, end) pairs")
    flat = _int64_array("ranges", raw.reshape(-1))
    n_ranges = raw.shape[0]
    groups = np.empty(n_features, dtype=np.int64)
    range_of_group = np.empty(n_ranges + n_features, dtype=np.int64)
    count = ct.c_int64()
    _check(
        None,
        lib.n4m_explain_groups_from_ranges_v1(
            n_features,
            flat.ctypes.data_as(_INT64),
            n_ranges,
            groups.ctypes.data_as(_INT64),
            n_features,
            range_of_group.ctypes.data_as(_INT64),
            range_of_group.size,
            ct.byref(count),
        ),
        "n4m_explain_groups_from_ranges_v1",
    )
    n_groups = int(count.value)
    return groups, n_groups, range_of_group[:n_groups].copy()


def summarize(attributions: np.ndarray, output_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-group mean ``|attribution|`` and mean signed attribution over rows.

    Args:
        attributions: ``(n_rows, n_groups, n_outputs)`` attributions.
        output_index: Output column to summarize.

    Returns:
        ``(mean_abs, mean_signed)``, each ``(n_groups,)``.
    """
    a = np.ascontiguousarray(np.asarray(attributions, dtype=np.float64))
    if a.ndim != 3:
        raise ValueError("attributions must be (n_rows, n_groups, n_outputs)")
    n_rows, n_groups, n_outputs = a.shape
    output_index = _int("output_index", output_index, 0)
    if output_index >= n_outputs:
        raise ValueError("output_index out of range")
    mean_abs = np.empty(n_groups, dtype=np.float64)
    mean_signed = np.empty(n_groups, dtype=np.float64)
    _check(
        None,
        lib.n4m_explain_summarize_v1(
            a.ctypes.data_as(_DOUBLE),
            n_rows,
            n_groups,
            n_outputs,
            output_index,
            mean_abs.ctypes.data_as(_DOUBLE),
            mean_signed.ctypes.data_as(_DOUBLE),
            n_groups,
        ),
        "n4m_explain_summarize_v1",
    )
    return mean_abs, mean_signed


__all__ = [
    "AttributionResult",
    "TreeEnsemble",
    "explain",
    "groups_fixed_width",
    "groups_from_ranges",
    "linear_shap",
    "summarize",
    "tree_shap",
]

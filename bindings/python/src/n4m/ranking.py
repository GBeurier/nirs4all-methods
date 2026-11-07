# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Native comparison-block rankings for pipelines or arbitrary components.

Rows must be matched task/dataset/split blocks; columns may represent complete
pipelines, models, preprocessors, parameter settings, or split strategies.
Grouping and causal interpretation belong to the caller. Repeated folds are
not independent studies. The Bayesian methods replay all within-block pairs;
they do not implement multiplayer/team inference and their posterior SDs do
not account for correlated comparisons. Online methods retain input order.
"""

from __future__ import annotations

import ctypes as ct
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._errors import N4MError
from ._ffi import lib
from ._types import MatrixView, RankingOptionsV1

_METHODS = {
    "average_rank": 0,
    "elo": 1,
    "bradley_terry": 2,
    "trueskill": 3,
    "trueskill_1v1": 3,
    "openskill": 4,
    "weng_lin_bradley_terry": 4,
}
_VARIANTS = [
    "average_rank",
    "elo",
    "bradley_terry",
    "trueskill_1v1",
    "weng_lin_bradley_terry",
]
_DOUBLE = ct.POINTER(ct.c_double)
_BYTE = ct.POINTER(ct.c_uint8)


def _check(status: int) -> None:
    if status:
        raise N4MError(status, "native ranking rejected the input")


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _mask(present: ArrayLike | None, shape: tuple[int, ...]) -> np.ndarray:
    if present is None:
        return np.ones(shape, dtype=np.uint8)
    raw = np.asarray(present)
    if raw.shape != shape:
        raise ValueError("present must have the same shape as scores/losses")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("present must contain only boolean or 0/1 entries")
    return np.ascontiguousarray(raw, dtype=np.uint8)


def rank(
    scores: ArrayLike,
    *,
    method: str = "average_rank",
    present: ArrayLike | None = None,
    higher_is_better: bool = False,
    tie_tolerance: float = 0.0,
    **method_options: Any,
) -> dict[str, Any]:
    """Rank a blocks-by-candidates matrix through the stable C ABI.

    Missingness must be explicit in ``present``; unmasked NaN/Inf is rejected.
    Positive-stride NumPy views are borrowed without a layout copy. Score
    direction is independent of metric direction. Global ranks are unavailable
    for disconnected graphs and nonconverged Bradley-Terry fits. Friedman uses
    complete blocks only, whereas displayed mean ranks use every observed row.
    Its chi-square approximate p-value requires independent comparable blocks.

    Method options mirror native names: ``initial_mean``, ``k_factor``,
    ``elo_scale``, ``prior_sigma``, ``beta``, ``dynamics``, ``draw_margin``,
    ``regularization``, ``convergence_tolerance``, and ``max_iterations``.
    Unknown methods/variants/options are rejected rather than approximated.
    """
    if method not in _METHODS:
        raise ValueError(f"Unsupported ranking method: {method}")
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better must be boolean")
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("scores must be a nonempty 2-D blocks-by-candidates matrix")
    if any(stride < 0 or stride % x.itemsize for stride in x.strides):
        x = np.ascontiguousarray(x)
    mask = _mask(present, x.shape)
    options = RankingOptionsV1()
    _check(lib.n4m_ranking_options_init_v1(ct.byref(options)))
    options.method = _METHODS[method]
    options.higher_is_better = int(higher_is_better)
    options.tie_tolerance = tie_tolerance
    allowed = {name for name, _ in RankingOptionsV1._fields_} - {
        "struct_size",
        "method",
        "higher_is_better",
        "tie_tolerance",
    }
    for key, value in method_options.items():
        if key not in allowed:
            raise ValueError(f"Unsupported ranking option: {key}")
        if key == "max_iterations":
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError("max_iterations must be an integer")
            if not 1 <= int(value) <= 1_000_000:
                raise ValueError("max_iterations must be between 1 and 1000000")
        setattr(options, key, value)
    view = MatrixView(
        x.ctypes.data,
        x.shape[0],
        x.shape[1],
        x.strides[0] // x.itemsize,
        x.strides[1] // x.itemsize,
        1,
        0,
    )
    candidates = np.empty((x.shape[1], 10), dtype=np.float64)
    summary: NDArray[np.float64] = np.empty(9, dtype=np.float64)
    _check(
        lib.n4m_ranking_compute_v1(
            ct.byref(view),
            mask.ctypes.data_as(_BYTE),
            ct.byref(options),
            candidates.ctypes.data_as(_DOUBLE),
            candidates.size,
            summary.ctypes.data_as(_DOUBLE),
            summary.size,
        )
    )
    names = [
        "score",
        "uncertainty",
        "average_rank",
        "observations",
        "matches",
        "component",
        "wins",
        "ties",
        "losses",
        "rank",
    ]
    integer_fields = {"observations", "matches", "component", "wins", "ties", "losses"}
    rows = []
    for index, row in enumerate(candidates):
        entry = {
            name: int(value) if name in integer_fields else _nullable(value)
            for name, value in zip(names, row, strict=True)
        }
        rows.append({"index": index, **entry})
    warnings = [
        "Mean ranks use all observed blocks; Friedman uses complete blocks only.",
        "Friedman p-value uses a chi-square approximation and assumes independent blocks.",
    ]
    if options.method >= 3:
        warnings.append(
            "Pairwise replay within each block is a composite approximation; "
            "posterior SD does not account for dependence between comparisons or repeated folds."
        )
    if options.method == 0 and summary[1] < summary[0]:
        warnings.append(
            "Incomplete design: mean ranks are descriptive and average_rank global ranks are "
            "unavailable; use bradley_terry or trueskill for incomplete comparisons."
        )
    if summary[3] > 1:
        warnings.append("Disconnected comparison graph: global ranks are unavailable.")
    if not summary[6]:
        warnings.append("Bradley-Terry did not converge; global ranks are unavailable.")
    return {
        "schema_version": "n4m.ranking/1",
        "method": method,
        "variant": _VARIANTS[options.method],
        "options": {
            name: bool(getattr(options, name))
            if name == "higher_is_better"
            else getattr(options, name)
            for name, _ in RankingOptionsV1._fields_
            if name not in {"struct_size", "method"}
        },
        "score_direction": "lower_is_better"
        if options.method == 0
        else "higher_is_better",
        "uncertainty_kind": "approximate_posterior_sd" if options.method >= 3 else None,
        "candidates": rows,
        "coverage": {
            "blocks": int(summary[0]),
            "complete_blocks": int(summary[1]),
            "paired_comparisons": int(summary[2]),
            "connected_components": int(summary[3]),
        },
        "inference": {
            "friedman_statistic": _nullable(summary[4]),
            "degrees_of_freedom": _nullable(summary[5]),
            "friedman_p_value": _nullable(summary[8]),
            "p_value_method": "chi_square_approximation",
            "reason": "chi_square_approximation_on_complete_blocks; host_must_assess_independence",
        },
        "convergence": {"converged": bool(summary[6]), "iterations": int(summary[7])},
        "warnings": warnings,
    }


def paired_relative_gain(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    present: ArrayLike | None = None,
    variances: ArrayLike | None = None,
) -> dict[str, Any]:
    """Summarize paired positive-loss gains, optionally with DL random effects.

    ``variances`` must be sampling variances of the relative effects, including
    paired covariance. No variances are estimated from folds or raw losses.
    The caller must establish independent studies for random-effects inference.
    The sample SD is descriptive effect dispersion, never a standard error.
    """
    a, b = (
        np.ascontiguousarray(baseline, dtype=np.float64),
        np.ascontiguousarray(candidate, dtype=np.float64),
    )
    if a.ndim != 1 or b.shape != a.shape or not a.size:
        raise ValueError("losses must be nonempty 1-D arrays of the same shape")
    mask = _mask(present, a.shape)
    var = (
        None if variances is None else np.ascontiguousarray(variances, dtype=np.float64)
    )
    if var is not None and var.shape != a.shape:
        raise ValueError("variances must have the same shape as losses")
    out: NDArray[np.float64] = np.empty(8, dtype=np.float64)
    _check(
        lib.n4m_ranking_relative_gain_v1(
            a.ctypes.data_as(_DOUBLE),
            b.ctypes.data_as(_DOUBLE),
            mask.ctypes.data_as(_BYTE),
            None if var is None else var.ctypes.data_as(_DOUBLE),
            a.size,
            out.ctypes.data_as(_DOUBLE),
            out.size,
        )
    )
    names = [
        "n",
        "mean_gain",
        "sample_sd",
        "pooled_gain",
        "standard_error",
        "tau_squared",
        "q",
        "i_squared",
    ]
    result = {name: _nullable(value) for name, value in zip(names, out, strict=True)}
    result["n"] = int(out[0])
    return {
        "schema_version": "n4m.relative_gain/1",
        "effect": "(baseline-candidate)/baseline",
        "meta_analysis": "dersimonian_laird"
        if var is not None and out[0] >= 2
        else None,
        "variance_source": "caller_supplied_effect_variance"
        if var is not None
        else None,
        "inference_assumption": "independent_study_effects; host_must_assess",
        **result,
    }

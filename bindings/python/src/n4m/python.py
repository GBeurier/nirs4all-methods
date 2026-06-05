# SPDX-License-Identifier: CECILL-2.1
"""ABI-close Python functions over ``libn4m``.

This module deliberately stays close to the C ABI: functions allocate the
matching C handle, run the requested fit/transform/apply call, return NumPy
arrays, and destroy the handle. The scikit-learn compatible estimators live in
``n4m.sklearn``.
"""
from __future__ import annotations

import ctypes
import csv
import hashlib
import json
import time
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import numpy as np

from ._errors import check
from ._ffi import lib, library_path
from ._matrix import as_f64_2d, empty_like_f64, empty_like_i32, numpy_to_view
from ._rng import PCG64
from ._types import FilterStats, SplitResult, TransferMetrics

_LSNV_PAD_MODES = {"reflect": 0, "edge": 1, "constant": 2, 0: 0, 1: 1, 2: 2}
_AREA_METHODS = {"sum": 0, "abs_sum": 1, "trapz": 2, 0: 0, 1: 1, 2: 2}
_SAVGOL_MODES = {"mirror": 0, "constant": 1, "nearest": 2, "wrap": 3, "interp": 4, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
_GAUSSIAN_MODES = {"reflect": 0, "constant": 1, "nearest": 2, "mirror": 3, "wrap": 4, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
_WAVELET_FAMILIES = {"haar": 0, "db4": 1, "sym4": 2, "coif1": 3, 0: 0, 1: 1, 2: 2, 3: 3}
_WAVELET_BOUNDARIES = {"periodization": 0, "symmetric": 1, "zero": 2, 0: 0, 1: 1, 2: 2}
_WAVELET_THRESHOLDS = {"soft": 0, "hard": 1, 0: 0, 1: 1}
_WAVELET_NOISE = {"median": 0, "std": 1, 0: 0, 1: 1}
_WAVELET_FEATURE_ENTROPY = {"energy": 0, "histogram": 1, 0: 0, 1: 1}
_Y_METHODS = {"iqr": 0, "zscore": 1, "percentile": 2, "mad": 3, 0: 0, 1: 1, 2: 2, 3: 3}
_X_METHODS = {"mahalanobis": 0, "robust_mahalanobis": 1, "pca_residual": 2, "pca_leverage": 3, "isolation_forest": 4, "lof": 5, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
_LEVERAGE_METHODS = {"hat": 0, "pca": 1, 0: 0, 1: 1}
_COMPOSITE_MODES = {"any": 0, "all": 1, 0: 0, 1: 1}
_RANDOM_X_OPS = {"multiply": 0, "add": 1, "subtract": 2, 0: 0, 1: 1, 2: 2}
_AOM_ROBUST_HPO_PROFILES = {"compact": 0, "wide": 1, 0: 0, 1: 1}
_AOM_ROBUST_HPO_HEAD_BITS = {"ridge": 1, "pls": 2}
_SWEEP_HEAD_BITS = {"ridge": 1, "pls": 2}
_N4M_BACKEND_CUDA = 5
_MOMENT_SCREEN_GPU_CROSSOVER_MIN_PRODUCT = 512 * 512
_MOMENT_SCREEN_CUDA_PLS_DEVICE_COMPONENT_MIN_FEATURES = 1024
_MOMENT_SCREEN_CUDA_PLS_FOLD_WORKSPACE_MIN_FEATURES = 1024
_MOMENT_SCREEN_GPU_CROSSOVER_POLICY = "n4m.moment_gpu_crossover.v4"
_AOM_MOMENT_POLICIES = {
    "auto": 0,
    "moments": 0,
    "operator_moments": 0,
    "materialized": 1,
    "materialised": 1,
    "legacy": 1,
    "force": 2,
    "force_moments": 2,
    "moments_only": 2,
    "operator_moments_only": 2,
    "strict_moments": 2,
    0: 0,
    1: 1,
    2: 2,
}
_AOM_PLS_SCORE_MODES = {
    "cv": 0,
    "exact_cv": 0,
    "gcv_proxy": 1,
    "pls1_gcv_proxy": 1,
    "proxy": 1,
    0: 0,
    1: 1,
}
_AOM_PLS_SCORE_MODE_NAMES = {
    0: "cv",
    1: "gcv_proxy",
}
_AOM_GATING_MODES = {
    "hard": 0,
    "winner": 0,
    "soft": 1,
    "average": 1,
    0: 0,
    1: 1,
}
_AOM_SPLIT_HEAD_SCORING_MODES = {
    "off": "off",
    "none": "off",
    "false": "off",
    "0": "off",
    "auto": "auto",
    "split": "auto",
    "true": "auto",
    "1": "auto",
    "force": "force",
    "always": "force",
}
_AOM_STRICT_OPERATOR_KINDS = {
    "identity": 0,
    "raw": 0,
    "detrend": 7,
    "detrend_poly": 7,
    "savgol": 8,
    "savgol_smooth": 8,
    "savgol_derivative": 9,
    "savgol_deriv": 9,
    "norris_williams": 10,
    "nw": 10,
    "finite_difference": 15,
    "finite_diff": 15,
    "diff": 15,
    "whittaker": 16,
    "fck": 17,
    "gaussian": 18,
    "gaussian_smooth": 18,
    0: 0,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
}
_AOM_STRICT_OPERATOR_NAMES = {
    0: "identity",
    7: "detrend_poly",
    8: "savgol_smooth",
    9: "savgol_derivative",
    10: "norris_williams",
    15: "finite_difference",
    16: "whittaker",
    17: "fck",
    18: "gaussian",
}
_AOM_SELECTOR_DEFAULT_OPERATORS = (
    "identity",
    ("detrend_poly", [1]),
    ("detrend_poly", [2]),
    ("savgol_smooth", [5, 2]),
    ("savgol_smooth", [7, 2]),
    ("savgol_derivative", [7, 2, 1]),
    ("savgol_derivative", [11, 2, 2]),
    ("norris_williams", [5, 5, 1]),
    ("finite_difference", [1]),
)
_AOM_PREPROCESS_MATRICES = (
    "transformed",
    "operator_outputs",
    "weights",
)
_AOM_PREPROCESS_SCALARS = (
    "n_operators",
    "n_samples",
    "n_features",
    "mode",
)
_MOMENT_MATRICES = (
    "x_sum",
    "y_sum",
    "xtx",
    "xty",
    "yty",
    "x_mean",
    "y_mean",
    "cxx",
    "cxy",
    "cyy",
)
_MOMENT_SCALARS = ("n_samples", "n_features", "n_targets")
_SWEEP_MATRICES = (
    "candidate_scores",
    "oof_predictions",
    "predictions",
    "coefficients",
    "intercept",
    "x_mean",
    "x_scale",
    "y_mean",
)
_AOM_SWEEP_MATRICES = (
    "candidate_scores",
    "chain_params",
    "oof_predictions",
    "predictions",
    "coefficients",
    "input_coefficients",
    "intercept",
    "x_mean",
    "x_scale",
    "y_mean",
)
_SWEEP_SCALARS = (
    "selected_candidate_id",
    "selected_head_id",
    "selected_param",
    "selected_cv_rmse",
    "n_candidates",
    "n_ridge_moment_candidates",
    "n_ridge_dual_materialized_candidates",
    "n_ridge_moment_cv_fits",
    "n_ridge_dual_materialized_cv_fits",
    "n_ridge_dual_cross_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
    "n_ridge_moment_final_fits",
    "n_ridge_dual_materialized_final_fits",
    "n_pls_moment_candidates",
    "n_pls_gcv_proxy_candidates",
    "n_pls_moment_cv_fits",
    "n_pls_moment_host_cv_fits",
    "n_pls_moment_cuda_device_cv_fits",
    "n_pls_moment_cuda_parallel_fold_batches",
    "n_pls_moment_cuda_parallel_fold_jobs",
    "n_pls_moment_score_batch_calls",
    "n_pls_moment_score_batch_jobs",
    "n_pls_materialized_cv_fits",
    "n_pls_gcv_proxy_fits",
    "n_pls_gcv_proxy_batch_calls",
    "n_pls_gcv_proxy_batch_jobs",
    "n_pls_moment_final_fits",
    "n_pls_moment_host_final_fits",
    "n_pls_moment_cuda_device_final_fits",
    "n_pls_materialized_final_fits",
    "score_only",
    "cv",
    "n_samples",
    "n_features",
    "n_targets",
)
_AOM_SWEEP_SCALARS = (
    "selected_candidate_id",
    "selected_chain_id",
    "selected_sweep_candidate_id",
    "selected_head_id",
    "selected_param",
    "selected_cv_rmse",
    "n_candidates",
    "n_operator_moment_candidates",
    "n_ridge_operator_moment_candidates",
    "n_pls_operator_moment_candidates",
    "n_banded_operator_moment_candidates",
    "n_structured_operator_moment_candidates",
    "n_dense_operator_moment_candidates",
    "n_materialized_candidates",
    "n_ridge_materialized_candidates",
    "n_pls_materialized_candidates",
    "n_moment_prefix_cache_hits",
    "n_moment_prefix_cache_misses",
    "n_ridge_moment_cv_fits",
    "n_ridge_dual_materialized_cv_fits",
    "n_ridge_dual_cross_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
    "n_ridge_moment_final_fits",
    "n_ridge_dual_materialized_final_fits",
    "n_pls_moment_cv_fits",
    "n_pls_moment_host_cv_fits",
    "n_pls_moment_cuda_device_cv_fits",
    "n_pls_moment_cuda_parallel_fold_batches",
    "n_pls_moment_cuda_parallel_fold_jobs",
    "n_pls_moment_score_batch_calls",
    "n_pls_moment_score_batch_jobs",
    "n_pls_materialized_cv_fits",
    "n_pls_gcv_proxy_candidates",
    "n_pls_gcv_proxy_fits",
    "n_pls_gcv_proxy_batch_calls",
    "n_pls_gcv_proxy_batch_jobs",
    "n_pls_moment_final_fits",
    "n_pls_moment_host_final_fits",
    "n_pls_moment_cuda_device_final_fits",
    "n_pls_materialized_final_fits",
    "aom_pls_score_mode",
    "score_only",
    "n_chains",
    "profile",
    "cv",
    "n_samples",
    "n_features",
    "n_targets",
)
_AOM_RIDGE_BLENDER_MATRICES = (
    "candidate_scores",
    "weights",
    "oof_predictions",
    "predictions",
    "input_coefficients",
    "intercept",
    "oof_candidate_predictions",
    "candidate_predictions",
)
_AOM_RIDGE_BLENDER_SCALARS = (
    "selected_candidate_id",
    "selected_chain_id",
    "selected_param",
    "selected_cv_rmse",
    "blend_oof_rmse",
    "regularizer",
    "n_candidates",
    "n_chains",
    "profile",
    "cv",
    "n_samples",
    "n_features",
    "n_targets",
)
_AOM_OPERATOR_PLS_STACK_MATRICES = (
    "candidate_scores",
    "fold_scores",
    "oof_predictions",
    "predictions",
    "stack_features",
    "coefficients",
    "intercept",
    "input_coefficients",
    "input_intercept",
)
_AOM_OPERATOR_PLS_STACK_SCALARS = (
    "selected_spec_id",
    "selected_components",
    "selected_alpha",
    "selected_oof_rmse",
    "selected_train_rmse",
    "selected_criterion",
    "std_penalty",
    "gap_penalty",
    "n_operator_features",
    "n_specs",
    "n_operators",
    "profile",
    "cv",
    "n_samples",
    "n_features",
    "n_targets",
)


def _enum(table: dict, value, name: str) -> int:
    try:
        return int(table[value])
    except KeyError as exc:
        raise ValueError(f"unknown {name}: {value!r}") from exc


def _create(prefix: str, *args) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check(getattr(lib, f"{prefix}_create")(ctypes.byref(handle), *args), f"{prefix}_create")
    return handle


def _create_ex(prefix: str, *args) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check(getattr(lib, f"{prefix}_create_ex")(ctypes.byref(handle), *args), f"{prefix}_create_ex")
    return handle


def _destroy(prefix: str, handle: ctypes.c_void_p) -> None:
    getattr(lib, f"{prefix}_destroy")(handle)


def _as_f64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(arr)


def _f64_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _null_f64_ptr():
    return ctypes.POINTER(ctypes.c_double)()


def _method_result_double_matrix(result: ctypes.c_void_p, name: str) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_double)()
    rows = ctypes.c_int64()
    cols = ctypes.c_int64()
    check(
        lib.n4m_method_result_get_double_matrix(
            result,
            name.encode("utf-8"),
            ctypes.byref(data),
            ctypes.byref(rows),
            ctypes.byref(cols),
        ),
        f"n4m_method_result_get_double_matrix({name})",
    )
    shape = (int(rows.value), int(cols.value))
    if shape[0] * shape[1] == 0:
        return np.empty(shape, dtype=np.float64)
    return np.ctypeslib.as_array(data, shape=shape).copy()


def _method_result_scalar(result: ctypes.c_void_p, name: str) -> float:
    value = ctypes.c_double()
    check(
        lib.n4m_method_result_get_scalar(
            result,
            name.encode("utf-8"),
            ctypes.byref(value),
        ),
        f"n4m_method_result_get_scalar({name})",
    )
    return float(value.value)


def _method_result_int_vector(result: ctypes.c_void_p, name: str) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_int32)()
    size = ctypes.c_int32()
    check(
        lib.n4m_method_result_get_int_vector(
            result,
            name.encode("utf-8"),
            ctypes.byref(data),
            ctypes.byref(size),
        ),
        f"n4m_method_result_get_int_vector({name})",
    )
    n = int(size.value)
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    return np.ctypeslib.as_array(data, shape=(n,)).copy()


def _method_result_int64_vector(result: ctypes.c_void_p, name: str) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_int64)()
    size = ctypes.c_int64()
    check(
        lib.n4m_method_result_get_int64_vector(
            result,
            name.encode("utf-8"),
            ctypes.byref(data),
            ctypes.byref(size),
        ),
        f"n4m_method_result_get_int64_vector({name})",
    )
    n = int(size.value)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    return np.ctypeslib.as_array(data, shape=(n,)).copy()


def _as_y_matrix(y, n_samples: int) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    elif y_arr.ndim == 2:
        y_arr = np.ascontiguousarray(y_arr)
    else:
        raise ValueError("y must be 1-D or 2-D")
    if y_arr.shape[0] != n_samples:
        raise ValueError("X and y have incompatible lengths")
    if not y_arr.flags.c_contiguous:
        y_arr = np.ascontiguousarray(y_arr)
    return y_arr


def _set_config_bool(cfg: ctypes.c_void_p, name: str, value: bool | None) -> None:
    if value is None:
        return
    check(
        getattr(lib, f"n4m_config_set_{name}")(cfg, ctypes.c_int32(int(bool(value)))),
        f"n4m_config_set_{name}",
    )


def _set_config_positive_int(
    cfg: ctypes.c_void_p,
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return
    value_i = int(value)
    if value_i < 1:
        raise ValueError(f"{name} must be positive")
    check(
        getattr(lib, f"n4m_config_set_{name}")(cfg, ctypes.c_int32(value_i)),
        f"n4m_config_set_{name}",
    )


def _set_model_config(
    cfg: ctypes.c_void_p,
    *,
    n_components: int | None = None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> None:
    if n_components is not None:
        check(
            lib.n4m_config_set_n_components(cfg, ctypes.c_int32(int(n_components))),
            "n4m_config_set_n_components",
        )
    _set_config_bool(cfg, "center_x", center_x)
    _set_config_bool(cfg, "scale_x", scale_x)
    _set_config_bool(cfg, "center_y", center_y)
    _set_config_bool(cfg, "scale_y", scale_y)


def _set_aom_moment_policy(cfg: ctypes.c_void_p, moment_policy) -> None:
    try:
        policy_id = _AOM_MOMENT_POLICIES[moment_policy]
    except KeyError as exc:
        raise ValueError(f"unknown AOM moment_policy: {moment_policy!r}") from exc
    check(
        lib.n4m_config_set_aom_moment_policy(cfg, ctypes.c_int(int(policy_id))),
        "n4m_config_set_aom_moment_policy",
    )


def _aom_pls_score_mode_id(pls_score_mode) -> int:
    try:
        return int(_AOM_PLS_SCORE_MODES[pls_score_mode])
    except KeyError as exc:
        raise ValueError(
            f"unknown AOM pls_score_mode: {pls_score_mode!r}"
        ) from exc


def _aom_pls_score_mode_name(pls_score_mode) -> str:
    mode_id = _aom_pls_score_mode_id(pls_score_mode)
    return _AOM_PLS_SCORE_MODE_NAMES.get(mode_id, f"mode_{mode_id}")


def _normalize_aom_split_head_scoring(value) -> str:
    key = str(value).strip().lower()
    if key not in _AOM_SPLIT_HEAD_SCORING_MODES:
        raise ValueError("split_head_scoring must be 'off', 'auto', or 'force'")
    return _AOM_SPLIT_HEAD_SCORING_MODES[key]


def _set_aom_pls_score_mode(cfg: ctypes.c_void_p, pls_score_mode) -> None:
    mode_id = _aom_pls_score_mode_id(pls_score_mode)
    check(
        lib.n4m_config_set_aom_pls_score_mode(cfg, ctypes.c_int(int(mode_id))),
        "n4m_config_set_aom_pls_score_mode",
    )


def _aom_gating_mode_id(gating_mode) -> int:
    try:
        return int(_AOM_GATING_MODES[gating_mode])
    except KeyError as exc:
        raise ValueError(f"unknown AOM gating_mode: {gating_mode!r}") from exc


def _method_result_dict(
    result: ctypes.c_void_p,
    *,
    matrices: Sequence[str],
    scalars: Sequence[str],
    int_vectors: Sequence[str] = (),
    int64_vectors: Sequence[str] = (),
) -> dict[str, np.ndarray | float]:
    out: dict[str, np.ndarray | float] = {}
    for name in matrices:
        out[name] = _method_result_double_matrix(result, name)
    for name in int_vectors:
        out[name] = _method_result_int_vector(result, name)
    for name in int64_vectors:
        out[name] = _method_result_int64_vector(result, name)
    for name in scalars:
        out[name] = _method_result_scalar(result, name)
    return out


def _fit_method_result(
    symbol: str,
    X,
    y,
    *extra_args,
    matrices: Sequence[str],
    scalars: Sequence[str],
    n_components: int | None = None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            n_components=n_components,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        check(
            getattr(lib, symbol)(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                *extra_args,
                ctypes.byref(result),
            ),
            symbol,
        )
        return _method_result_dict(result, matrices=matrices, scalars=scalars)
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def _moments_from_views(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    row_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    ctx = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        if row_indices is None:
            check(
                lib.n4m_moments_compute(
                    ctx,
                    ctypes.byref(Xv),
                    ctypes.byref(Yv),
                    ctypes.byref(result),
                ),
                "n4m_moments_compute",
            )
        else:
            idx = np.ascontiguousarray(row_indices, dtype=np.int64).reshape(-1)
            check(
                lib.n4m_moments_subset_compute(
                    ctx,
                    ctypes.byref(Xv),
                    ctypes.byref(Yv),
                    idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(idx.size),
                    ctypes.byref(result),
                ),
                "n4m_moments_subset_compute",
            )
        return _method_result_dict(
            result,
            matrices=_MOMENT_MATRICES,
            scalars=_MOMENT_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def moments(X, y, *, row_indices=None) -> dict[str, np.ndarray | float]:
    """Compute native raw and centered moment sufficient statistics."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    idx = None if row_indices is None else np.asarray(row_indices, dtype=np.int64)
    return _moments_from_views(X_arr, y_arr, row_indices=idx)


def moments_train_from_heldout(X, y, heldout_indices) -> dict[str, np.ndarray | float]:
    """Compute train moments as all-row moments minus held-out-row moments."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    heldout = np.ascontiguousarray(heldout_indices, dtype=np.int64).reshape(-1)
    if heldout.size == 0:
        raise ValueError("heldout_indices must not be empty")

    ctx = ctypes.c_void_p()
    all_result = ctypes.c_void_p()
    heldout_result = ctypes.c_void_p()
    train_result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        check(
            lib.n4m_moments_compute(
                ctx,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.byref(all_result),
            ),
            "n4m_moments_compute",
        )
        check(
            lib.n4m_moments_subset_compute(
                ctx,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                heldout.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(heldout.size),
                ctypes.byref(heldout_result),
            ),
            "n4m_moments_subset_compute",
        )
        check(
            lib.n4m_moments_subtract(
                ctx,
                all_result,
                heldout_result,
                ctypes.byref(train_result),
            ),
            "n4m_moments_subtract",
        )
        return _method_result_dict(
            train_result,
            matrices=_MOMENT_MATRICES,
            scalars=_MOMENT_SCALARS,
        )
    finally:
        if train_result.value:
            lib.n4m_method_result_destroy(train_result)
        if heldout_result.value:
            lib.n4m_method_result_destroy(heldout_result)
        if all_result.value:
            lib.n4m_method_result_destroy(all_result)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def _sweep_heads_mask(heads) -> int:
    if isinstance(heads, str):
        heads = (heads,)
    mask = 0
    for head in heads:
        try:
            mask |= _SWEEP_HEAD_BITS[head]
        except KeyError as exc:
            raise ValueError(f"unknown sweep head: {head!r}") from exc
    if mask == 0:
        raise ValueError("heads must include ridge and/or pls")
    return mask


def _is_aom_op_spec(value) -> bool:
    if isinstance(value, (str, int, np.integer, dict)):
        return True
    if isinstance(value, (tuple, list)) and value:
        return isinstance(value[0], (str, int, np.integer))
    return False


def _aom_op_kind_params(op) -> tuple[int, np.ndarray]:
    if isinstance(op, dict):
        kind = op.get("kind", op.get("op"))
        if kind is None:
            raise ValueError("AOM operator dict must contain 'kind' or 'op'")
        params = op.get("params", ())
    elif isinstance(op, (str, int, np.integer)):
        kind = op
        params = ()
    elif isinstance(op, (tuple, list)) and op:
        kind = op[0]
        if len(op) == 2 and isinstance(op[1], (tuple, list, np.ndarray)):
            params = op[1]
        else:
            params = op[1:]
    else:
        raise ValueError(f"invalid AOM operator spec: {op!r}")
    kind_id = _enum(_AOM_STRICT_OPERATOR_KINDS, kind, "AOM strict operator")
    param_arr = np.asarray(params, dtype=np.float64).reshape(-1)
    return kind_id, np.ascontiguousarray(param_arr)


def _flatten_aom_chains(chains):
    chain_offsets = [0]
    op_kinds = []
    param_offsets = [0]
    params = []
    n_chains = 0
    for chain in chains:
        ops = [chain] if _is_aom_op_spec(chain) else list(chain)
        if not ops:
            raise ValueError("AOM chains must not be empty; use 'identity'")
        for op in ops:
            kind_id, param_arr = _aom_op_kind_params(op)
            op_kinds.append(kind_id)
            params.extend(float(v) for v in param_arr)
            param_offsets.append(len(params))
        chain_offsets.append(len(op_kinds))
        n_chains += 1
    if n_chains == 0:
        raise ValueError("chains must contain at least one AOM chain")
    return (
        np.ascontiguousarray(chain_offsets, dtype=np.int32),
        np.ascontiguousarray(op_kinds, dtype=np.int32),
        np.ascontiguousarray(param_offsets, dtype=np.int32),
        np.ascontiguousarray(params, dtype=np.float64),
    )


_AOM_NATIVE_COMPACT_CHAINS = (
    (("identity", ()),),
    (("detrend_poly", (1.0,)),),
    (("detrend_poly", (2.0,)),),
    (("savgol_smooth", (5.0, 2.0)),),
    (("savgol_smooth", (7.0, 2.0)),),
    (("savgol_derivative", (7.0, 2.0, 1.0)),),
    (("savgol_derivative", (11.0, 2.0, 2.0)),),
    (("norris_williams", (5.0, 5.0, 1.0)),),
    (("finite_difference", (1.0,)),),
    (("detrend_poly", (1.0,)), ("savgol_derivative", (7.0, 2.0, 1.0))),
    (("detrend_poly", (1.0,)), ("norris_williams", (5.0, 5.0, 1.0))),
    (("savgol_smooth", (5.0, 2.0)), ("finite_difference", (1.0,))),
)
_AOM_NATIVE_WIDE_EXTRA_CHAINS = (
    (("savgol_smooth", (9.0, 2.0)),),
    (("savgol_smooth", (11.0, 3.0)),),
    (("savgol_smooth", (15.0, 3.0)),),
    (("savgol_derivative", (9.0, 2.0, 1.0)),),
    (("savgol_derivative", (15.0, 3.0, 1.0)),),
    (("savgol_derivative", (21.0, 3.0, 1.0)),),
    (("savgol_derivative", (15.0, 3.0, 2.0)),),
    (("norris_williams", (7.0, 7.0, 1.0)),),
    (("norris_williams", (11.0, 11.0, 1.0)),),
    (("norris_williams", (7.0, 7.0, 2.0)),),
    (("finite_difference", (2.0,)),),
    (("gaussian", (1.0,)),),
    (("gaussian", (2.0,)),),
    (("fck", (0.0,)),),
    (("fck", (1.0,)),),
    (("whittaker", (100.0,)),),
    (("whittaker", (1000.0,)),),
    (("detrend_poly", (2.0,)), ("savgol_derivative", (11.0, 2.0, 1.0))),
    (("whittaker", (100.0,)), ("savgol_derivative", (7.0, 2.0, 1.0))),
)
_AOM_LAB_TEMPLATES = (
    ("identity",),
    ("detrend_poly",),
    ("savgol_smooth",),
    ("savgol_derivative",),
    ("norris_williams",),
    ("finite_difference",),
    ("gaussian",),
    ("fck",),
    ("whittaker",),
    ("detrend_poly", "savgol_smooth"),
    ("detrend_poly", "savgol_derivative"),
    ("detrend_poly", "norris_williams"),
    ("detrend_poly", "finite_difference"),
    ("savgol_smooth", "finite_difference"),
    ("savgol_smooth", "savgol_derivative"),
    ("gaussian", "finite_difference"),
    ("fck", "finite_difference"),
    ("whittaker", "savgol_derivative"),
    ("whittaker", "finite_difference"),
)
_AOM_ROUTE_COUNTER_KEYS = (
    "n_operator_moment_candidates",
    "n_ridge_operator_moment_candidates",
    "n_pls_operator_moment_candidates",
    "n_banded_operator_moment_candidates",
    "n_structured_operator_moment_candidates",
    "n_dense_operator_moment_candidates",
    "n_materialized_candidates",
    "n_ridge_materialized_candidates",
    "n_pls_materialized_candidates",
    "n_moment_prefix_cache_hits",
    "n_moment_prefix_cache_misses",
    "n_ridge_moment_cv_fits",
    "n_ridge_dual_materialized_cv_fits",
    "n_ridge_dual_cross_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
    "n_ridge_moment_final_fits",
    "n_ridge_dual_materialized_final_fits",
    "n_pls_moment_cv_fits",
    "n_pls_moment_host_cv_fits",
    "n_pls_moment_cuda_device_cv_fits",
    "n_pls_moment_cuda_parallel_fold_batches",
    "n_pls_moment_cuda_parallel_fold_jobs",
    "n_pls_moment_score_batch_calls",
    "n_pls_moment_score_batch_jobs",
    "n_pls_materialized_cv_fits",
    "n_pls_gcv_proxy_candidates",
    "n_pls_gcv_proxy_fits",
    "n_pls_gcv_proxy_batch_calls",
    "n_pls_gcv_proxy_batch_jobs",
    "n_pls_moment_final_fits",
    "n_pls_moment_host_final_fits",
    "n_pls_moment_cuda_device_final_fits",
    "n_pls_materialized_final_fits",
)
_AOM_CANDIDATE_INT_FIELDS = {
    "candidate_id",
    "global_candidate_id",
    "chunk_index",
    "local_chain_id",
    "chain_id",
    "head_id",
    "score_route_id",
    "source_index",
    "cv_rank",
    "eval_rank",
    "rank_delta",
    "n_operator_moment_candidates",
    "n_ridge_operator_moment_candidates",
    "n_pls_operator_moment_candidates",
    "n_banded_operator_moment_candidates",
    "n_structured_operator_moment_candidates",
    "n_dense_operator_moment_candidates",
    "n_materialized_candidates",
    "n_ridge_materialized_candidates",
    "n_pls_materialized_candidates",
    "n_moment_prefix_cache_hits",
    "n_moment_prefix_cache_misses",
    "n_ridge_moment_cv_fits",
    "n_ridge_dual_materialized_cv_fits",
    "n_ridge_dual_cross_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
    "n_ridge_moment_final_fits",
    "n_ridge_dual_materialized_final_fits",
    "n_pls_moment_cv_fits",
    "n_pls_moment_host_cv_fits",
    "n_pls_moment_cuda_device_cv_fits",
    "n_pls_moment_cuda_parallel_fold_batches",
    "n_pls_moment_cuda_parallel_fold_jobs",
    "n_pls_materialized_cv_fits",
    "n_pls_gcv_proxy_candidates",
    "n_pls_gcv_proxy_fits",
    "n_pls_moment_final_fits",
    "n_pls_moment_host_final_fits",
    "n_pls_moment_cuda_device_final_fits",
    "n_pls_materialized_final_fits",
    "refit_group_n_ridge_moment_cv_fits",
    "refit_group_n_ridge_dual_materialized_cv_fits",
    "refit_group_n_ridge_dual_cross_cv_fits",
    "refit_group_n_ridge_moment_score_batch_calls",
    "refit_group_n_ridge_moment_score_batch_jobs",
    "refit_group_n_pls_moment_host_cv_fits",
    "refit_group_n_pls_moment_cuda_device_cv_fits",
    "refit_group_n_pls_moment_cuda_parallel_fold_batches",
    "refit_group_n_pls_moment_cuda_parallel_fold_jobs",
}
_AOM_CANDIDATE_FLOAT_FIELDS = {
    "param",
    "cv_rmse",
    "screen_score",
    "screen_cv_rmse",
    "refit_cv_rmse",
    "selected_cv_rmse",
    "train_rmse",
    "oof_rmse",
    "eval_rmse",
    "eval_r2",
    "elapsed_ms",
}


def _canonical_aom_op_spec(op) -> tuple[str, tuple[float, ...]]:
    kind_id, params = _aom_op_kind_params(op)
    return (
        _AOM_STRICT_OPERATOR_NAMES.get(kind_id, f"operator_{kind_id}"),
        tuple(float(v) for v in params),
    )


def _chain_tuple_to_list(chain) -> list[tuple[str, tuple[float, ...]]]:
    return [(name, tuple(params)) for name, params in chain]


def _canonicalize_aom_chain(chain) -> list[tuple[str, tuple[float, ...]]]:
    ops = [chain] if _is_aom_op_spec(chain) else list(chain)
    return _chain_tuple_to_list(tuple(_canonical_aom_op_spec(op) for op in ops))


def _sweep_head_name(head_id: int) -> str:
    if head_id == 0:
        return "ridge"
    if head_id == 1:
        return "pls"
    return f"head_{head_id}"


_AOM_SCORE_ROUTE_NAMES = {
    0: "materialized",
    1: "dense_operator_moment",
    2: "banded_operator_moment",
    3: "structured_operator_moment",
}


def _aom_score_route_name(route_id: int) -> str:
    return _AOM_SCORE_ROUTE_NAMES.get(int(route_id), f"route_{int(route_id)}")


def _normalize_sweep_head_name(head) -> str:
    if head in (0, "0", "ridge"):
        return "ridge"
    if head in (1, "1", "pls"):
        return "pls"
    raise ValueError("AOM candidate head must be 'ridge' or 'pls'")


def moment_screen_backend_recommendation(
    n_samples: int,
    n_features: int,
    *,
    head: str = "pls",
    cuda_available: bool | None = None,
    min_cuda_product: int = _MOMENT_SCREEN_GPU_CROSSOVER_MIN_PRODUCT,
    cuda_pls_min_device_features: int = (
        _MOMENT_SCREEN_CUDA_PLS_DEVICE_COMPONENT_MIN_FEATURES
    ),
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, object]:
    """Recommend the CPU or CUDA build for exact moment score screens.

    The Python binding loads one ``libn4m`` shared object per process, so this
    is a launch-planning helper: choose the CPU or CUDA build before importing
    ``n4m``. The policy uses only source-free dataset shape metadata and the
    requested linear head; it does not inspect dataset identity or labels.
    """

    head_name = _normalize_sweep_head_name(head)
    n_samples_i = int(n_samples)
    n_features_i = int(n_features)
    min_cuda_product_i = int(min_cuda_product)
    cuda_pls_min_device_features_i = int(cuda_pls_min_device_features)
    if n_samples_i <= 1:
        raise ValueError("n_samples must be greater than 1")
    if n_features_i <= 0:
        raise ValueError("n_features must be positive")
    if min_cuda_product_i <= 0:
        raise ValueError("min_cuda_product must be positive")
    if cuda_pls_min_device_features_i <= 0:
        raise ValueError("cuda_pls_min_device_features must be positive")

    work_product = n_samples_i * n_features_i
    loaded_cuda_available = bool(lib.n4m_backend_is_available(_N4M_BACKEND_CUDA))
    if cuda_available is None:
        effective_cuda_available = loaded_cuda_available
        availability_source = "loaded_libn4m"
    else:
        effective_cuda_available = bool(cuda_available)
        availability_source = "caller_override"

    if not effective_cuda_available:
        recommended_backend = "cpu"
        reason = "cuda_unavailable"
    elif work_product < min_cuda_product_i:
        recommended_backend = "cpu"
        reason = "below_live_crossover"
    else:
        recommended_backend = "cuda"
        reason = "at_or_above_live_crossover"

    loaded_backend_matches = (
        loaded_cuda_available
        if recommended_backend == "cuda"
        else not loaded_cuda_available
    )
    uses_cuda_pls_device_component_loop = (
        recommended_backend == "cuda"
        and head_name == "pls"
        and n_features_i >= cuda_pls_min_device_features_i
    )
    uses_cuda_pls_fold_workspace = (
        recommended_backend == "cuda"
        and head_name == "pls"
        and n_features_i >= cuda_pls_min_device_features_i
    )
    uses_cuda_pls_many_batched = (
        uses_cuda_pls_device_component_loop
        and bool(cuda_pls_many_batched)
    )
    return {
        "recommended_backend": recommended_backend,
        "reason": reason,
        "head": head_name,
        "n_samples": n_samples_i,
        "n_features": n_features_i,
        "work_product": work_product,
        "min_cuda_product": min_cuda_product_i,
        "cuda_available": effective_cuda_available,
        "cuda_available_source": availability_source,
        "loaded_cuda_available": loaded_cuda_available,
        "loaded_backend_matches_recommendation": loaded_backend_matches,
        "requires_fresh_process": not loaded_backend_matches,
        "uses_cuda_pls_device_component_loop": uses_cuda_pls_device_component_loop,
        "cuda_pls_device_component_min_features": (
            cuda_pls_min_device_features_i
        ),
        "uses_cuda_pls_fold_workspace": uses_cuda_pls_fold_workspace,
        "cuda_pls_fold_workspace_min_features": (
            cuda_pls_min_device_features_i
        ),
        "uses_cuda_pls_many_batched": uses_cuda_pls_many_batched,
        "cuda_pls_many_batched": (
            None
            if cuda_pls_many_batched is None
            else bool(cuda_pls_many_batched)
        ),
        "policy_source": _MOMENT_SCREEN_GPU_CROSSOVER_POLICY,
        "policy_inputs": (
            "n_samples",
            "n_features",
            "head",
            "cuda_available",
            "min_cuda_product",
            "cuda_pls_min_device_features",
            "cuda_pls_many_batched",
        ),
    }


def _candidate_chain_head_param(candidate: dict):
    if not isinstance(candidate, dict):
        raise ValueError("AOM candidates must be dictionaries")
    if "chain" not in candidate:
        raise ValueError("AOM candidate must contain a decoded 'chain'")
    if "param" not in candidate:
        raise ValueError("AOM candidate must contain 'param'")
    if "head" in candidate:
        head = _normalize_sweep_head_name(candidate["head"])
    elif "head_id" in candidate:
        head = _normalize_sweep_head_name(candidate["head_id"])
    else:
        raise ValueError("AOM candidate must contain 'head' or 'head_id'")
    return _canonicalize_aom_chain(candidate["chain"]), head, float(candidate["param"])


def _regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _regression_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    centered = yt - float(np.mean(yt))
    denom = float(centered @ centered)
    if denom <= 1e-12:
        return 0.0
    residual = yt - yp
    return float(1.0 - float(residual @ residual) / denom)


def _hash_array(hasher, name: str, array) -> None:
    if array is None:
        hasher.update(name.encode("utf-8"))
        hasher.update(b":none")
        return
    arr = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(json.dumps(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    hasher.update(memoryview(arr).cast("B"))


def _hash_jsonable(hasher, name: str, value) -> None:
    hasher.update(name.encode("utf-8"))
    hasher.update(
        json.dumps(
            _jsonable_value(value, include_arrays=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _aom_campaign_fingerprint(
    *,
    X: np.ndarray,
    y: np.ndarray,
    fold_ids,
    chain_list,
    chunk_size: int,
    top_k: int,
    cv: int,
    ridge_lambdas,
    pls_components,
    heads,
    center_x,
    scale_x,
    center_y,
    scale_y,
    moment_policy,
    pls_score_mode,
    chain_ordering,
    split_head_scoring,
    cuda_pls_parallel_folds,
    cuda_pls_min_device_features,
    cuda_pls_many_batched,
) -> str:
    hasher = hashlib.blake2b(digest_size=24)
    _hash_jsonable(hasher, "schema", "n4m.aom_chain_score_campaign.v1")
    _hash_array(hasher, "X", X)
    _hash_array(hasher, "y", y)
    _hash_array(
        hasher,
        "fold_ids",
        None if fold_ids is None else np.asarray(fold_ids, dtype=np.int32).reshape(-1),
    )
    _hash_jsonable(hasher, "chains", chain_list)
    _hash_jsonable(hasher, "chunk_size", int(chunk_size))
    _hash_jsonable(hasher, "top_k", int(top_k))
    _hash_jsonable(hasher, "cv", int(cv))
    _hash_jsonable(hasher, "ridge_lambdas", tuple(float(v) for v in ridge_lambdas))
    _hash_jsonable(hasher, "pls_components", tuple(int(v) for v in pls_components))
    _hash_jsonable(hasher, "heads", tuple(str(v) for v in heads))
    _hash_jsonable(hasher, "center_x", center_x)
    _hash_jsonable(hasher, "scale_x", scale_x)
    _hash_jsonable(hasher, "center_y", center_y)
    _hash_jsonable(hasher, "scale_y", scale_y)
    _hash_jsonable(hasher, "moment_policy", moment_policy)
    _hash_jsonable(hasher, "pls_score_mode", _aom_pls_score_mode_name(pls_score_mode))
    _hash_jsonable(hasher, "chain_ordering", chain_ordering)
    _hash_jsonable(hasher, "split_head_scoring", split_head_scoring)
    _hash_jsonable(
        hasher,
        "cuda_pls_parallel_folds",
        None if cuda_pls_parallel_folds is None else bool(cuda_pls_parallel_folds),
    )
    _hash_jsonable(
        hasher,
        "cuda_pls_min_device_features",
        None
        if cuda_pls_min_device_features is None
        else int(cuda_pls_min_device_features),
    )
    _hash_jsonable(
        hasher,
        "cuda_pls_many_batched",
        None if cuda_pls_many_batched is None else bool(cuda_pls_many_batched),
    )
    return hasher.hexdigest()


def _aom_campaign_aggregate(chunks) -> tuple[dict[str, int], int, float]:
    aggregate = {key: 0 for key in _AOM_ROUTE_COUNTER_KEYS}
    total_candidates = 0
    total_elapsed_ms = 0.0
    for chunk in chunks:
        total_candidates += int(chunk.get("n_candidates", 0))
        total_elapsed_ms += float(chunk.get("elapsed_ms", 0.0))
        for key in _AOM_ROUTE_COUNTER_KEYS:
            aggregate[key] += int(chunk.get(key, 0))
    return aggregate, total_candidates, total_elapsed_ms


def _aom_campaign_performance_metrics(
    *,
    n_chains: int,
    n_candidates: int,
    elapsed_ms: float,
    counters,
) -> dict[str, float]:
    elapsed = float(elapsed_ms)
    seconds = elapsed / 1000.0 if elapsed > 0.0 else 0.0
    candidates = int(n_candidates)
    chains = int(n_chains)

    def fraction(key: str) -> float:
        if candidates <= 0:
            return 0.0
        return float(counters.get(key, 0)) / float(candidates)

    return {
        "chains_per_second": float(chains) / seconds if seconds > 0.0 else 0.0,
        "candidates_per_second": (
            float(candidates) / seconds if seconds > 0.0 else 0.0
        ),
        "projected_200k_chains_seconds": (
            200000.0 / (float(chains) / seconds)
            if seconds > 0.0 and chains > 0
            else 0.0
        ),
        "projected_200k_chains_minutes": (
            (200000.0 / (float(chains) / seconds)) / 60.0
            if seconds > 0.0 and chains > 0
            else 0.0
        ),
        "ms_per_chain": elapsed / float(chains) if chains > 0 else 0.0,
        "ms_per_candidate": (
            elapsed / float(candidates) if candidates > 0 else 0.0
        ),
        "operator_moment_candidate_fraction": fraction(
            "n_operator_moment_candidates"
        ),
        "materialized_candidate_fraction": fraction("n_materialized_candidates"),
        "ridge_operator_moment_candidate_fraction": fraction(
            "n_ridge_operator_moment_candidates"
        ),
        "pls_operator_moment_candidate_fraction": fraction(
            "n_pls_operator_moment_candidates"
        ),
        "dense_operator_moment_candidate_fraction": fraction(
            "n_dense_operator_moment_candidates"
        ),
        "banded_operator_moment_candidate_fraction": fraction(
            "n_banded_operator_moment_candidates"
        ),
        "structured_operator_moment_candidate_fraction": fraction(
            "n_structured_operator_moment_candidates"
        ),
        "moment_prefix_cache_hit_fraction": (
            float(counters.get("n_moment_prefix_cache_hits", 0))
            / float(
                counters.get("n_moment_prefix_cache_hits", 0)
                + counters.get("n_moment_prefix_cache_misses", 0)
            )
            if (
                counters.get("n_moment_prefix_cache_hits", 0)
                + counters.get("n_moment_prefix_cache_misses", 0)
            ) > 0
            else 0.0
        ),
        "ridge_cv_fits_per_chain": (
            float(
                counters.get("n_ridge_moment_cv_fits", 0)
                + counters.get("n_ridge_dual_materialized_cv_fits", 0)
                + counters.get("n_ridge_dual_cross_cv_fits", 0)
            )
            / float(chains)
            if chains > 0
            else 0.0
        ),
        "ridge_cv_fits_per_candidate": (
            float(
                counters.get("n_ridge_moment_cv_fits", 0)
                + counters.get("n_ridge_dual_materialized_cv_fits", 0)
                + counters.get("n_ridge_dual_cross_cv_fits", 0)
            )
            / float(candidates)
            if candidates > 0
            else 0.0
        ),
        "pls_cv_fits_per_chain": (
            float(
                counters.get("n_pls_moment_cv_fits", 0)
                + counters.get("n_pls_materialized_cv_fits", 0)
            )
            / float(chains)
            if chains > 0
            else 0.0
        ),
        "pls_cv_fits_per_candidate": (
            float(
                counters.get("n_pls_moment_cv_fits", 0)
                + counters.get("n_pls_materialized_cv_fits", 0)
            )
            / float(candidates)
            if candidates > 0
            else 0.0
        ),
        "pls_gcv_proxy_fits_per_chain": (
            float(counters.get("n_pls_gcv_proxy_fits", 0)) / float(chains)
            if chains > 0
            else 0.0
        ),
        "pls_gcv_proxy_fits_per_candidate": (
            float(counters.get("n_pls_gcv_proxy_fits", 0)) / float(candidates)
            if candidates > 0
            else 0.0
        ),
    }


def _aom_campaign_backend_recommendations(
    *,
    n_samples: int,
    n_features: int,
    heads,
    cuda_available: bool | None,
    backend_min_cuda_product: int | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, dict[str, object]]:
    head_names = sorted({_normalize_sweep_head_name(head) for head in heads})
    min_product = (
        _MOMENT_SCREEN_GPU_CROSSOVER_MIN_PRODUCT
        if backend_min_cuda_product is None
        else int(backend_min_cuda_product)
    )
    if min_product <= 0:
        raise ValueError("backend_min_cuda_product must be positive")
    threshold = (
        _MOMENT_SCREEN_CUDA_PLS_DEVICE_COMPONENT_MIN_FEATURES
        if cuda_pls_min_device_features is None
        else int(cuda_pls_min_device_features)
    )
    return {
        head: moment_screen_backend_recommendation(
            n_samples,
            n_features,
            head=head,
            cuda_available=cuda_available,
            min_cuda_product=min_product,
            cuda_pls_min_device_features=threshold,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )
        for head in head_names
    }


def _write_json_atomic(path: Path, payload) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _jsonable_value(value, *, include_arrays: bool = False):
    if isinstance(value, np.ndarray):
        return value.tolist() if include_arrays else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable_value(v, include_arrays=include_arrays) for v in value]
    if isinstance(value, list):
        return [_jsonable_value(v, include_arrays=include_arrays) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            converted = _jsonable_value(item, include_arrays=include_arrays)
            if converted is not None:
                out[str(key)] = converted
        return out
    return value


def _candidate_report_rows(report_or_rows):
    if isinstance(report_or_rows, dict):
        if "rows" in report_or_rows:
            return list(report_or_rows["rows"])
        if "top_candidates" in report_or_rows:
            return list(report_or_rows["top_candidates"])
    return list(report_or_rows)


def _parse_aom_candidate_field(key: str, value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if key in {"chain", "chain_json"}:
            return json.loads(text)
        if key in _AOM_CANDIDATE_INT_FIELDS:
            return int(float(text))
        if key in _AOM_CANDIDATE_FLOAT_FIELDS:
            return float(text)
        if text in {"true", "True"}:
            return True
        if text in {"false", "False"}:
            return False
        if text in {"null", "None"}:
            return None
        return value
    if key in _AOM_CANDIDATE_INT_FIELDS:
        return int(value)
    if key in _AOM_CANDIDATE_FLOAT_FIELDS:
        return float(value)
    return value


def _restore_aom_candidate_record(record: dict) -> dict[str, object]:
    restored: dict[str, object] = {}
    for key, value in record.items():
        converted = _parse_aom_candidate_field(str(key), value)
        if converted is not None:
            restored[str(key)] = converted
    if "chain" not in restored and "chain_json" in restored:
        restored["chain"] = restored["chain_json"]
    if "chain" in restored:
        restored["chain"] = _canonicalize_aom_chain(restored["chain"])
        restored["chain_json"] = json.dumps(restored["chain"], separators=(",", ":"))
    return restored


def _aom_candidate_score_key(rows, score_key: str | None) -> str:
    if score_key is not None:
        return str(score_key)
    for key in ("eval_rmse", "cv_rmse", "refit_cv_rmse", "screen_cv_rmse"):
        if any(key in row for row in rows):
            return key
    raise ValueError(
        "could not infer score_key; expected eval_rmse, cv_rmse, "
        "refit_cv_rmse or screen_cv_rmse"
    )


def _aom_candidate_analysis_rows(report_or_rows, score_key: str | None):
    if (
        isinstance(report_or_rows, dict)
        and "rows" not in report_or_rows
        and "top_candidates" not in report_or_rows
    ):
        raw_rows = []
    else:
        raw_rows = _candidate_report_rows(report_or_rows)
    rows = [_restore_aom_candidate_record(dict(row)) for row in raw_rows]
    key = _aom_candidate_score_key(rows, score_key)
    parsed = []
    for source_index, row in enumerate(rows):
        if key not in row:
            continue
        try:
            score = float(row[key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        chain = _canonicalize_aom_chain(row.get("chain", ()))
        head = (
            _normalize_sweep_head_name(row["head"])
            if "head" in row
            else _normalize_sweep_head_name(row["head_id"])
        )
        parsed.append({
            "source_index": int(source_index),
            "row": row,
            "score": score,
            "chain": chain,
            "head": head,
        })
    if not parsed:
        raise ValueError(f"no finite candidate scores found for {key!r}")
    parsed.sort(key=lambda item: item["score"])
    for rank, item in enumerate(parsed, start=1):
        item["rank"] = int(rank)
    return key, parsed


def _aom_group_summary(name: str, items) -> dict[str, object]:
    scores = np.asarray([item["score"] for item in items], dtype=np.float64)
    ranks = np.asarray([item["rank"] for item in items], dtype=np.float64)
    best_item = min(items, key=lambda item: item["score"])
    return {
        "group": name,
        "n_candidates": int(len(items)),
        "best_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "best_rank": int(best_item["rank"]),
        "mean_rank": float(np.mean(ranks)),
        "best_candidate": best_item["row"],
    }


def _aom_operator_stage(name: str) -> str:
    op = str(name).lower()
    if op == "identity":
        return "identity"
    if "snv" in op or "msc" in op or "emsc" in op:
        return "scatter"
    if "derivative" in op or "difference" in op or "norris" in op:
        return "derivative"
    if "detrend" in op or "baseline" in op or "whittaker" in op:
        return "baseline"
    if "smooth" in op or "savgol" in op or "gaussian" in op:
        return "smooth"
    return "advanced"


def _aom_operator_option_label(name: str, params) -> str:
    values = tuple(float(v) for v in params)
    if not values:
        return str(name)

    def fmt(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    return f"{name}({','.join(fmt(v) for v in values)})"


def _infer_score_key(rows, explicit_key, candidates) -> str:
    if explicit_key is not None:
        return str(explicit_key)
    for key in candidates:
        if any(key in row for row in rows):
            return key
    raise ValueError(f"could not infer score key from {tuple(candidates)!r}")


def _rank_from_scores(values) -> dict[int, int]:
    order = sorted(range(len(values)), key=lambda ix: (float(values[ix]), ix))
    return {int(ix): int(rank) for rank, ix in enumerate(order, start=1)}


def _spearman_from_ranks(left, right) -> float:
    n = len(left)
    if n <= 1:
        return 1.0
    lx = np.asarray(left, dtype=np.float64)
    rx = np.asarray(right, dtype=np.float64)
    lx -= float(np.mean(lx))
    rx -= float(np.mean(rx))
    denom = float(np.sqrt(float(lx @ lx) * float(rx @ rx)))
    if denom <= 1e-12:
        return 0.0
    return float((lx @ rx) / denom)


def _dedupe_aom_chains(chains):
    seen = set()
    out = []
    for chain in chains:
        key = tuple((name, tuple(params)) for name, params in chain)
        if key not in seen:
            seen.add(key)
            out.append(_chain_tuple_to_list(key))
    return out


def _normalize_aom_chain_ordering(chain_ordering) -> str:
    value = "input" if chain_ordering is None else str(chain_ordering)
    normalized = value.lower().replace("-", "_")
    aliases = {
        "none": "input",
        "stable": "input",
        "as_input": "input",
        "prefix_cache": "prefix",
        "cache": "prefix",
        "cached_prefix": "prefix",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"input", "prefix"}:
        raise ValueError("chain_ordering must be 'input' or 'prefix'")
    return normalized


def _aom_chain_order_key(chain):
    return tuple((str(name), tuple(float(v) for v in params)) for name, params in chain)


def _ordered_aom_chain_records(chain_list, chain_ordering: str):
    records = [(int(index), chain) for index, chain in enumerate(chain_list)]
    if chain_ordering == "input":
        return records
    return sorted(records, key=lambda item: (_aom_chain_order_key(item[1]), item[0]))


def _default_aom_lab_families():
    return {
        "identity": [("identity", ())],
        "detrend_poly": [
            ("detrend_poly", (1.0,)),
            ("detrend_poly", (2.0,)),
        ],
        "savgol_smooth": [
            ("savgol_smooth", (5.0, 2.0)),
            ("savgol_smooth", (7.0, 2.0)),
            ("savgol_smooth", (9.0, 2.0)),
            ("savgol_smooth", (11.0, 3.0)),
            ("savgol_smooth", (15.0, 3.0)),
            ("savgol_smooth", (21.0, 3.0)),
        ],
        "savgol_derivative": [
            ("savgol_derivative", (7.0, 2.0, 1.0)),
            ("savgol_derivative", (9.0, 2.0, 1.0)),
            ("savgol_derivative", (11.0, 2.0, 1.0)),
            ("savgol_derivative", (15.0, 3.0, 1.0)),
            ("savgol_derivative", (21.0, 3.0, 1.0)),
            ("savgol_derivative", (15.0, 3.0, 2.0)),
        ],
        "norris_williams": [
            ("norris_williams", (5.0, 5.0, 1.0)),
            ("norris_williams", (7.0, 7.0, 1.0)),
            ("norris_williams", (11.0, 11.0, 1.0)),
            ("norris_williams", (7.0, 7.0, 2.0)),
        ],
        "finite_difference": [
            ("finite_difference", (1.0,)),
            ("finite_difference", (2.0,)),
        ],
        "gaussian": [
            ("gaussian", (0.75,)),
            ("gaussian", (1.0,)),
            ("gaussian", (2.0,)),
        ],
        "fck": [
            ("fck", (0.0,)),
            ("fck", (1.0,)),
            ("fck", (1.5,)),
        ],
        "whittaker": [
            ("whittaker", (100.0,)),
            ("whittaker", (1000.0,)),
            ("whittaker", (10000.0,)),
        ],
    }


def _iter_aom_strict_chain_grid_entries(
    profile: str = "compact",
    *,
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    include_identity: bool = True,
    max_chains: int | None = None,
):
    normalized = str(profile).lower().replace("-", "_")
    aliases = {
        "native_compact": "compact",
        "native_wide": "wide",
        "moment_lab": "lab",
        "large": "lab",
        "cartesian": "lab",
    }
    normalized = aliases.get(normalized, normalized)
    max_count = None
    if max_chains is not None:
        max_count = int(max_chains)
        if max_count < 1:
            raise ValueError("max_chains must be positive when provided")

    seen = set()
    emitted = 0

    def emit(chain):
        nonlocal emitted
        key = tuple((str(name), tuple(float(v) for v in params)) for name, params in chain)
        if key in seen:
            return None
        seen.add(key)
        decoded = _chain_tuple_to_list(key)
        if not include_identity and decoded == [("identity", ())]:
            return None
        if max_count is not None and emitted >= max_count:
            return None
        chain_id = emitted
        emitted += 1
        return chain_id, decoded

    if families is None and templates is None and normalized in {"compact", "wide"}:
        chains = list(_AOM_NATIVE_COMPACT_CHAINS)
        if normalized == "wide":
            chains.extend(_AOM_NATIVE_WIDE_EXTRA_CHAINS)
        for chain in chains:
            if max_count is not None and emitted >= max_count:
                return
            item = emit(chain)
            if item is not None:
                yield item
        return

    if families is None:
        if normalized not in {"lab", "wide", "compact"}:
            raise ValueError("profile must be compact, wide, lab, or cartesian")
        family_map = _default_aom_lab_families()
    else:
        family_map = {}
        for name, specs in families.items():
            values = [_canonical_aom_op_spec(spec) for spec in specs]
            if not values:
                raise ValueError(f"AOM family {name!r} must contain at least one operator")
            family_map[str(name)] = values

    if templates is None:
        template_list = _AOM_LAB_TEMPLATES
    else:
        template_list = tuple(tuple(str(name) for name in template) for template in templates)
    if not template_list:
        raise ValueError("templates must contain at least one template")

    for template in template_list:
        if not template:
            raise ValueError("AOM chain templates must not be empty")
        try:
            choices = [family_map[name] for name in template]
        except KeyError as exc:
            raise ValueError(f"unknown AOM family in template: {exc.args[0]!r}") from exc
        for ops in product(*choices):
            if max_count is not None and emitted >= max_count:
                return
            item = emit(ops)
            if item is not None:
                yield item


def build_aom_strict_chain_grid(
    profile: str = "compact",
    *,
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    include_identity: bool = True,
    max_chains: int | None = None,
) -> list[list[tuple[str, tuple[float, ...]]]]:
    """Build strict-linear AOM chains for native moment/campaign sweeps.

    ``compact`` and ``wide`` reproduce the native built-in banks. ``lab`` /
    ``cartesian`` uses deterministic family templates with broader SavGol,
    Norris-Williams, finite-difference and Whittaker diversity. Custom
    ``families`` plus ``templates`` can define larger cartesian screens.
    """
    decoded = [
        chain
        for _, chain in _iter_aom_strict_chain_grid_entries(
            profile,
            families=families,
            templates=templates,
            include_identity=include_identity,
            max_chains=max_chains,
        )
    ]
    if not decoded:
        raise ValueError("AOM strict chain grid is empty")
    return decoded


def iter_aom_strict_chain_grid(
    profile: str = "compact",
    *,
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    include_identity: bool = True,
    max_chains: int | None = None,
    start: int = 0,
    stop: int | None = None,
    chunk_size: int | None = None,
    with_ids: bool = False,
):
    """Iterate strict-linear AOM chains with stable global chain ids.

    This yields the same post-dedup, post-``include_identity`` order as
    ``build_aom_strict_chain_grid`` without requiring callers to materialize the
    full cartesian bank. ``start`` and ``stop`` slice by the stable global chain
    ids, and ``chunk_size`` yields lists suitable for incremental campaigns.
    """
    start_i = int(start)
    if start_i < 0:
        raise ValueError("start must be non-negative")
    stop_i = None if stop is None else int(stop)
    if stop_i is not None and stop_i < start_i:
        raise ValueError("stop must be greater than or equal to start")
    chunk_n = None if chunk_size is None else int(chunk_size)
    if chunk_n is not None and chunk_n < 1:
        raise ValueError("chunk_size must be positive when provided")

    def rows():
        for chain_id, chain in _iter_aom_strict_chain_grid_entries(
            profile,
            families=families,
            templates=templates,
            include_identity=include_identity,
            max_chains=max_chains,
        ):
            if chain_id < start_i:
                continue
            if stop_i is not None and chain_id >= stop_i:
                break
            yield (chain_id, chain) if with_ids else chain

    if chunk_n is None:
        yield from rows()
        return

    chunk = []
    for item in rows():
        chunk.append(item)
        if len(chunk) == chunk_n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def decode_aom_chains(
    result_or_chain_offsets,
    op_kinds=None,
    param_offsets=None,
    chain_params=None,
) -> list[list[tuple[str, tuple[float, ...]]]]:
    """Decode flat native AOM chain descriptors into Python operator specs.

    Pass either a MethodResult dictionary returned by ``aom_sweep_run`` /
    ``aom_chain_sweep_run`` or the four descriptor arrays
    ``chain_offsets``, ``op_kinds``, ``param_offsets`` and ``chain_params``.
    """
    if isinstance(result_or_chain_offsets, dict):
        result = result_or_chain_offsets
        chain_offsets = result["chain_offsets"]
        op_kinds = result["op_kinds"]
        param_offsets = result["param_offsets"]
        chain_params = result["chain_params"]
    else:
        chain_offsets = result_or_chain_offsets
        if op_kinds is None or param_offsets is None or chain_params is None:
            raise ValueError(
                "op_kinds, param_offsets and chain_params are required "
                "when decoding raw descriptor arrays"
            )

    chain_offsets_arr = np.asarray(chain_offsets, dtype=np.int32).reshape(-1)
    op_kinds_arr = np.asarray(op_kinds, dtype=np.int32).reshape(-1)
    param_offsets_arr = np.asarray(param_offsets, dtype=np.int32).reshape(-1)
    params_arr = np.asarray(chain_params, dtype=np.float64).reshape(-1)

    if chain_offsets_arr.size < 2:
        raise ValueError("chain_offsets must contain at least two entries")
    if param_offsets_arr.size != op_kinds_arr.size + 1:
        raise ValueError("param_offsets length must be len(op_kinds) + 1")
    if chain_offsets_arr[0] != 0 or chain_offsets_arr[-1] != op_kinds_arr.size:
        raise ValueError("chain_offsets do not match op_kinds")
    if param_offsets_arr[0] != 0 or param_offsets_arr[-1] != params_arr.size:
        raise ValueError("param_offsets do not match chain_params")
    if np.any(np.diff(chain_offsets_arr) < 0) or np.any(np.diff(param_offsets_arr) < 0):
        raise ValueError("AOM descriptor offsets must be monotonic")

    chains = []
    for chain_i in range(chain_offsets_arr.size - 1):
        begin = int(chain_offsets_arr[chain_i])
        end = int(chain_offsets_arr[chain_i + 1])
        if begin == end:
            raise ValueError("AOM descriptors must not contain empty chains")
        chain = []
        for op_i in range(begin, end):
            kind = int(op_kinds_arr[op_i])
            p_begin = int(param_offsets_arr[op_i])
            p_end = int(param_offsets_arr[op_i + 1])
            name = _AOM_STRICT_OPERATOR_NAMES.get(kind, f"operator_{kind}")
            chain.append((name, tuple(float(v) for v in params_arr[p_begin:p_end])))
        chains.append(chain)
    return chains


def aom_candidate_table(result: dict, *, sort: bool = False) -> list[dict[str, object]]:
    """Return decoded AOM sweep candidates as plain Python dictionaries."""
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 5:
        raise ValueError("AOM candidate_scores must have shape (n_candidates, 5)")
    route_ids = np.asarray(result.get("candidate_routes", ()), dtype=np.int32).reshape(-1)
    if route_ids.size == 0:
        route_ids = np.full(scores.shape[0], -1, dtype=np.int32)
    elif route_ids.size != scores.shape[0]:
        raise ValueError("AOM candidate_routes length must match candidate_scores rows")
    chains = decode_aom_chains(result)
    result_pls_score_mode = _aom_pls_score_mode_name(
        int(result.get("aom_pls_score_mode", 0))
    )
    rows: list[dict[str, object]] = []
    for row_index, score in enumerate(scores):
        chain_id = int(score[1])
        if chain_id < 0 or chain_id >= len(chains):
            raise ValueError("candidate_scores contain an invalid chain_id")
        head_id = int(score[2])
        route_id = int(route_ids[row_index])
        score_metric = (
            "pls_gcv_proxy_rmse"
            if head_id == 1 and result_pls_score_mode == "gcv_proxy"
            else "cv_rmse"
        )
        rows.append({
            "candidate_id": int(score[0]),
            "chain_id": chain_id,
            "chain": chains[chain_id],
            "head_id": head_id,
            "head": _sweep_head_name(head_id),
            "score_route_id": route_id,
            "score_route": _aom_score_route_name(route_id),
            "score_metric": score_metric,
            "param": float(score[3]),
            "cv_rmse": float(score[4]),
        })
    if sort:
        rows.sort(key=lambda row: row["cv_rmse"])
    return rows


def aom_chain_score_campaign(
    X,
    y,
    chains=None,
    *,
    profile: str = "lab",
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    max_chains: int | None = None,
    chain_chunk_size: int = 4096,
    top_k: int = 50,
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(0.01, 0.1, 1.0, 10.0),
    pls_components=(1, 2, 4),
    heads=("ridge", "pls"),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    pls_score_mode: str | int = "cv",
    chain_ordering: str = "input",
    split_head_scoring: str = "off",
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
    backend_cuda_available: bool | None = None,
    backend_min_cuda_product: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    max_chunks_per_run: int | None = None,
) -> dict[str, object]:
    """Score a strict-linear AOM chain campaign in chunks.

    The campaign always calls ``aom_chain_sweep_run(..., score_only=True)``.
    It returns a global top-k candidate list with decoded chains plus route
    counters summed across chunks. When ``checkpoint_path`` is provided, each
    completed chunk is written to a JSON checkpoint and can be resumed later.
    """
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    fold_ids_arr = None
    if fold_ids is not None:
        fold_ids_arr = np.asarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_ids_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids must have one entry per sample")

    if chains is None:
        chain_list = build_aom_strict_chain_grid(
            profile,
            families=families,
            templates=templates,
            max_chains=max_chains,
        )
    else:
        chain_list = [_canonicalize_aom_chain(chain) for chain in chains]
        if max_chains is not None:
            chain_list = chain_list[: int(max_chains)]
    if not chain_list:
        raise ValueError("chains must contain at least one AOM chain")
    if int(chain_chunk_size) < 1:
        raise ValueError("chain_chunk_size must be positive")
    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    if max_chunks_per_run is not None and int(max_chunks_per_run) < 1:
        raise ValueError("max_chunks_per_run must be positive when provided")

    chunk_size = int(chain_chunk_size)
    keep = int(top_k)
    max_new_chunks = None if max_chunks_per_run is None else int(max_chunks_per_run)
    pls_score_mode_name = _aom_pls_score_mode_name(pls_score_mode)
    chain_ordering_name = _normalize_aom_chain_ordering(chain_ordering)
    split_head_scoring_name = _normalize_aom_split_head_scoring(split_head_scoring)
    cuda_pls_min_device_features_i = (
        None
        if cuda_pls_min_device_features is None
        else int(cuda_pls_min_device_features)
    )
    if (
        cuda_pls_min_device_features_i is not None
        and cuda_pls_min_device_features_i < 1
    ):
        raise ValueError("cuda_pls_min_device_features must be positive")
    requested_head_names = tuple(
        sorted({_normalize_sweep_head_name(head) for head in heads})
    )
    can_split_head_scoring = requested_head_names == ("pls", "ridge")
    if split_head_scoring_name == "force" and not can_split_head_scoring:
        raise ValueError(
            "split_head_scoring='force' requires heads containing both ridge and pls"
        )
    ordered_chain_records = _ordered_aom_chain_records(
        chain_list, chain_ordering_name
    )
    backend_recommendations = _aom_campaign_backend_recommendations(
        n_samples=X_arr.shape[0],
        n_features=X_arr.shape[1],
        heads=heads,
        cuda_available=backend_cuda_available,
        backend_min_cuda_product=backend_min_cuda_product,
        cuda_pls_min_device_features=cuda_pls_min_device_features_i,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )
    chunk_starts = list(range(0, len(ordered_chain_records), chunk_size))
    n_total_chunks = len(chunk_starts)
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    fingerprint = None
    loaded_checkpoint = False
    top_rows: list[dict[str, object]] = []
    top_rows_by_head: dict[str, list[dict[str, object]]] = {}
    top_rows_by_score_route: dict[str, list[dict[str, object]]] = {}
    chunk_summaries = []
    if checkpoint is not None:
        fingerprint = _aom_campaign_fingerprint(
            X=X_arr,
            y=y_arr,
            fold_ids=fold_ids_arr,
            chain_list=chain_list,
            chunk_size=chunk_size,
            top_k=keep,
            cv=cv,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=heads,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            pls_score_mode=pls_score_mode,
            chain_ordering=chain_ordering_name,
            split_head_scoring=split_head_scoring_name,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features_i,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if bool(resume) and checkpoint.exists():
            payload = json.loads(checkpoint.read_text())
            if payload.get("checkpoint_schema") != "n4m.aom_chain_score_campaign.v1":
                raise ValueError("AOM campaign checkpoint schema is not supported")
            if payload.get("fingerprint") != fingerprint:
                raise ValueError(
                    "AOM campaign checkpoint does not match this data/configuration"
                )
            top_rows = list(payload.get("top_candidates", []))
            loaded_by_head = payload.get("top_candidates_by_head", {})
            if isinstance(loaded_by_head, dict):
                top_rows_by_head = {
                    str(head): list(rows)
                    for head, rows in loaded_by_head.items()
                    if isinstance(rows, list)
                }
            else:
                top_rows_by_head = {}
            if not top_rows_by_head:
                for row in top_rows:
                    if "head" in row:
                        top_rows_by_head.setdefault(str(row["head"]), []).append(row)
            loaded_by_score_route = payload.get("top_candidates_by_score_route", {})
            if isinstance(loaded_by_score_route, dict):
                top_rows_by_score_route = {
                    str(route): list(rows)
                    for route, rows in loaded_by_score_route.items()
                    if isinstance(rows, list)
                }
            else:
                top_rows_by_score_route = {}
            if not top_rows_by_score_route:
                for row in top_rows:
                    route = row.get("score_route")
                    if route is None and "score_route_id" in row:
                        route = _aom_score_route_name(int(row["score_route_id"]))
                    if route is not None:
                        top_rows_by_score_route.setdefault(str(route), []).append(row)
            chunk_summaries = list(payload.get("chunks", []))
            loaded_checkpoint = True
    chunk_summaries.sort(key=lambda item: int(item["chunk_index"]))
    completed_chunks = {int(summary["chunk_index"]) for summary in chunk_summaries}
    if len(completed_chunks) != len(chunk_summaries):
        raise ValueError("AOM campaign checkpoint contains duplicate chunks")
    unknown_chunks = [idx for idx in completed_chunks if idx < 0 or idx >= n_total_chunks]
    if unknown_chunks:
        raise ValueError("AOM campaign checkpoint contains out-of-range chunks")
    if completed_chunks:
        top_rows = [
            row for row in top_rows
            if int(row.get("chunk_index", -1)) in completed_chunks
        ]
        top_rows_by_head = {
            head: [
                row for row in rows
                if int(row.get("chunk_index", -1)) in completed_chunks
            ]
            for head, rows in top_rows_by_head.items()
        }
        top_rows_by_head = {
            head: rows for head, rows in top_rows_by_head.items() if rows
        }
        if not top_rows_by_head:
            for row in top_rows:
                if "head" in row:
                    top_rows_by_head.setdefault(str(row["head"]), []).append(row)
        top_rows_by_score_route = {
            route: [
                row for row in rows
                if int(row.get("chunk_index", -1)) in completed_chunks
            ]
            for route, rows in top_rows_by_score_route.items()
        }
        top_rows_by_score_route = {
            route: rows for route, rows in top_rows_by_score_route.items() if rows
        }
        if not top_rows_by_score_route:
            for row in top_rows:
                route = row.get("score_route")
                if route is None and "score_route_id" in row:
                    route = _aom_score_route_name(int(row["score_route_id"]))
                if route is not None:
                    top_rows_by_score_route.setdefault(str(route), []).append(row)
    processed_this_run = 0

    def trim_top_rows_by_head() -> None:
        nonlocal top_rows_by_head
        trimmed: dict[str, list[dict[str, object]]] = {}
        for head, rows in top_rows_by_head.items():
            sorted_rows = sorted(rows, key=lambda row: row["cv_rmse"])
            trimmed[str(head)] = sorted_rows[:keep]
        top_rows_by_head = dict(sorted(trimmed.items()))

    def trim_top_rows_by_score_route() -> None:
        nonlocal top_rows_by_score_route
        trimmed: dict[str, list[dict[str, object]]] = {}
        for route, rows in top_rows_by_score_route.items():
            sorted_rows = sorted(rows, key=lambda row: row["cv_rmse"])
            trimmed[str(route)] = sorted_rows[:keep]
        top_rows_by_score_route = dict(sorted(trimmed.items()))

    def make_report(*, complete: bool) -> dict[str, object]:
        nonlocal top_rows, top_rows_by_head, top_rows_by_score_route, chunk_summaries
        top_rows.sort(key=lambda row: row["cv_rmse"])
        if len(top_rows) > keep:
            top_rows = top_rows[:keep]
        trim_top_rows_by_head()
        trim_top_rows_by_score_route()
        chunk_summaries.sort(key=lambda item: int(item["chunk_index"]))
        aggregate, total_candidates, total_elapsed_ms = _aom_campaign_aggregate(
            chunk_summaries
        )
        performance = _aom_campaign_performance_metrics(
            n_chains=len(chain_list),
            n_candidates=total_candidates,
            elapsed_ms=total_elapsed_ms,
            counters=aggregate,
        )
        best = top_rows[0] if top_rows else None
        best_by_head = {
            head: rows[0]
            for head, rows in top_rows_by_head.items()
            if rows
        }
        best_by_score_route = {
            route: rows[0]
            for route, rows in top_rows_by_score_route.items()
            if rows
        }
        n_split_head_chunks = sum(
            int(summary.get("split_head_scoring_used", 0))
            for summary in chunk_summaries
        )
        n_chunk_score_calls = sum(
            int(summary.get("chunk_score_calls", 1))
            for summary in chunk_summaries
        )
        return {
            "checkpoint_schema": "n4m.aom_chain_score_campaign.v1",
            "fingerprint": fingerprint,
            "complete": bool(complete),
            "resumed_from_checkpoint": bool(loaded_checkpoint),
            "checkpoint_path": None if checkpoint is None else str(checkpoint),
            "top_candidates": top_rows,
            "top_candidates_by_head": top_rows_by_head,
            "top_candidates_by_score_route": top_rows_by_score_route,
            "best": best,
            "best_by_head": best_by_head,
            "best_by_score_route": best_by_score_route,
            "chunks": chunk_summaries,
            "n_chunks": int(len(chunk_summaries)),
            "n_total_chunks": int(n_total_chunks),
            "n_remaining_chunks": int(n_total_chunks - len(chunk_summaries)),
            "processed_chunks_this_run": int(processed_this_run),
            "max_chunks_per_run": (
                None if max_new_chunks is None else int(max_new_chunks)
            ),
            "n_chains": int(len(chain_list)),
            "n_candidates": int(total_candidates),
            "elapsed_ms": float(total_elapsed_ms),
            "cv": int(cv),
            "top_k": keep,
            "moment_policy": moment_policy,
            "pls_score_mode": pls_score_mode_name,
            "chain_ordering": chain_ordering_name,
            "split_head_scoring": split_head_scoring_name,
            "cuda_pls_parallel_folds": (
                None
                if cuda_pls_parallel_folds is None
                else bool(cuda_pls_parallel_folds)
            ),
            "cuda_pls_min_device_features": cuda_pls_min_device_features_i,
            "cuda_pls_many_batched": (
                None
                if cuda_pls_many_batched is None
                else bool(cuda_pls_many_batched)
            ),
            "split_head_scoring_policy": (
                "single native call per chunk"
                if split_head_scoring_name == "off"
                else "separate Ridge and PLS native score-only calls for mixed chunks"
            ),
            "n_split_head_chunks": int(n_split_head_chunks),
            "n_chunk_score_calls": int(n_chunk_score_calls),
            "chain_ordering_policy": (
                "input order"
                if chain_ordering_name == "input"
                else "lexicographic operator-prefix order before chunking"
            ),
            "moment_backend_recommendations": backend_recommendations,
            "moment_backend_recommendation_policy_inputs": (
                "n_samples",
                "n_features",
                "head",
                "cuda_available",
                "min_cuda_product",
                "cuda_pls_min_device_features",
                "cuda_pls_many_batched",
            ),
            "backend_cuda_available": backend_cuda_available,
            "backend_min_cuda_product": (
                None
                if backend_min_cuda_product is None
                else int(backend_min_cuda_product)
            ),
            "library_path": library_path(),
            "abi": ".".join(str(int(fn())) for fn in (
                lib.n4m_get_abi_version_major,
                lib.n4m_get_abi_version_minor,
                lib.n4m_get_abi_version_patch,
            )),
            **aggregate,
            **performance,
        }

    def score_chunk(chunk) -> tuple[dict[str, object], int, int]:
        split_requested = (
            split_head_scoring_name != "off" and can_split_head_scoring
        )
        if not split_requested:
            return (
                aom_chain_sweep_run(
                    X_arr,
                    y_arr,
                    chunk,
                    cv=cv,
                    fold_ids=fold_ids_arr,
                    ridge_lambdas=ridge_lambdas,
                    pls_components=pls_components,
                    heads=heads,
                    center_x=center_x,
                    scale_x=scale_x,
                    center_y=center_y,
                    scale_y=scale_y,
                    moment_policy=moment_policy,
                    pls_score_mode=pls_score_mode,
                    score_only=True,
                    cuda_pls_parallel_folds=cuda_pls_parallel_folds,
                    cuda_pls_min_device_features=cuda_pls_min_device_features_i,
                    cuda_pls_many_batched=cuda_pls_many_batched,
                ),
                0,
                1,
            )

        ridge_result = aom_chain_sweep_run(
            X_arr,
            y_arr,
            chunk,
            cv=cv,
            fold_ids=fold_ids_arr,
            ridge_lambdas=ridge_lambdas,
            pls_components=(),
            heads=("ridge",),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            pls_score_mode=pls_score_mode,
            score_only=True,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features_i,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )
        pls_result = aom_chain_sweep_run(
            X_arr,
            y_arr,
            chunk,
            cv=cv,
            fold_ids=fold_ids_arr,
            ridge_lambdas=(),
            pls_components=pls_components,
            heads=("pls",),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            pls_score_mode=pls_score_mode,
            score_only=True,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features_i,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )

        result = dict(ridge_result)
        ridge_scores = np.asarray(ridge_result["candidate_scores"], dtype=np.float64)
        pls_scores = np.asarray(pls_result["candidate_scores"], dtype=np.float64)
        scores = np.vstack([ridge_scores, pls_scores])
        scores[:, 0] = np.arange(scores.shape[0], dtype=np.float64)
        result["candidate_scores"] = scores
        ridge_routes = np.asarray(
            ridge_result.get("candidate_routes", ()), dtype=np.int32
        ).reshape(-1)
        pls_routes = np.asarray(
            pls_result.get("candidate_routes", ()), dtype=np.int32
        ).reshape(-1)
        result["candidate_routes"] = np.concatenate([ridge_routes, pls_routes])

        for key in _AOM_ROUTE_COUNTER_KEYS:
            result[key] = int(ridge_result.get(key, 0)) + int(pls_result.get(key, 0))
        result["n_candidates"] = int(scores.shape[0])
        result["score_only"] = 1.0
        result["aom_pls_score_mode"] = float(
            pls_result.get(
                "aom_pls_score_mode",
                _aom_pls_score_mode_id(pls_score_mode),
            )
        )

        finite_order = np.argsort(scores[:, 4])
        best_index = int(finite_order[0])
        result["selected_candidate_id"] = best_index
        result["selected_sweep_candidate_id"] = best_index
        result["selected_chain_id"] = int(scores[best_index, 1])
        result["selected_head_id"] = int(scores[best_index, 2])
        result["selected_param"] = float(scores[best_index, 3])
        result["selected_cv_rmse"] = float(scores[best_index, 4])
        return result, 1, 2

    for chunk_index, start in enumerate(chunk_starts):
        if chunk_index in completed_chunks:
            continue
        if max_new_chunks is not None and processed_this_run >= max_new_chunks:
            break
        chunk_records = ordered_chain_records[start:start + chunk_size]
        chunk = [chain for _, chain in chunk_records]
        original_chain_ids = [int(index) for index, _ in chunk_records]
        t0 = time.perf_counter()
        result, split_used, chunk_score_calls = score_chunk(chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scores = np.asarray(result["candidate_scores"], dtype=np.float64)
        n_candidates = int(result["n_candidates"])
        candidate_offset = sum(
            int(summary.get("n_candidates", 0))
            for summary in chunk_summaries
            if int(summary["chunk_index"]) < chunk_index
        )

        decoded = decode_aom_chains(result)
        route_ids = np.asarray(
            result.get("candidate_routes", ()), dtype=np.int32
        ).reshape(-1)
        if route_ids.size == 0:
            route_ids = np.full(scores.shape[0], -1, dtype=np.int32)
        elif route_ids.size != scores.shape[0]:
            raise ValueError(
                "AOM candidate_routes length must match candidate_scores rows"
            )
        result_pls_score_mode = _aom_pls_score_mode_name(
            int(result.get("aom_pls_score_mode", _aom_pls_score_mode_id(pls_score_mode)))
        )
        if scores.size:
            built_rows: dict[int, dict[str, object]] = {}

            def build_candidate_row(row_index) -> dict[str, object]:
                row_index = int(row_index)
                if row_index in built_rows:
                    return built_rows[row_index]
                row = scores[int(row_index)]
                local_chain_id = int(row[1])
                head_id = int(row[2])
                route_id = int(route_ids[int(row_index)])
                route_name = _aom_score_route_name(route_id)
                score_metric = (
                    "pls_gcv_proxy_rmse"
                    if head_id == 1 and result_pls_score_mode == "gcv_proxy"
                    else "cv_rmse"
                )
                built_rows[row_index] = {
                    "candidate_id": int(row[0]),
                    "global_candidate_id": int(candidate_offset + int(row[0])),
                    "chunk_index": int(chunk_index),
                    "local_chain_id": local_chain_id,
                    "ordered_chain_id": int(start + local_chain_id),
                    "chain_id": int(original_chain_ids[local_chain_id]),
                    "chain": decoded[local_chain_id],
                    "head_id": head_id,
                    "head": _sweep_head_name(head_id),
                    "score_route_id": route_id,
                    "score_route": route_name,
                    "score_metric": score_metric,
                    "param": float(row[3]),
                    "cv_rmse": float(row[4]),
                }
                return built_rows[row_index]

            order = np.argsort(scores[:, 4])[:keep]
            for row_index in order:
                top_rows.append(build_candidate_row(row_index))

            head_ids = sorted({int(row[2]) for row in scores})
            for head_id in head_ids:
                head_indices = [
                    int(index)
                    for index in range(scores.shape[0])
                    if int(scores[index, 2]) == head_id
                ]
                head_indices.sort(key=lambda index: float(scores[index, 4]))
                head_name = _sweep_head_name(head_id)
                for row_index in head_indices[:keep]:
                    top_rows_by_head.setdefault(head_name, []).append(
                        build_candidate_row(row_index)
                    )

            route_id_values = sorted({int(route_id) for route_id in route_ids})
            for route_id in route_id_values:
                route_indices = [
                    int(index)
                    for index in range(scores.shape[0])
                    if int(route_ids[index]) == route_id
                ]
                route_indices.sort(key=lambda index: float(scores[index, 4]))
                route_name = _aom_score_route_name(route_id)
                for row_index in route_indices[:keep]:
                    top_rows_by_score_route.setdefault(route_name, []).append(
                        build_candidate_row(row_index)
                    )
            trim_top_rows_by_head()
            trim_top_rows_by_score_route()

        chunk_counters = {key: int(result[key]) for key in _AOM_ROUTE_COUNTER_KEYS}
        chunk_summaries.append({
            "chunk_index": int(chunk_index),
            "chain_start": int(start),
            "chain_order_start": int(start),
            "chain_order_end": int(start + len(chunk)),
            "n_chains": int(len(chunk)),
            "n_candidates": n_candidates,
            "split_head_scoring_used": int(split_used),
            "chunk_score_calls": int(chunk_score_calls),
            "selected_ordered_chain_id": int(start + int(result["selected_chain_id"])),
            "selected_chain_id": int(
                original_chain_ids[int(result["selected_chain_id"])]
            ),
            "selected_head_id": int(result["selected_head_id"]),
            "selected_param": float(result["selected_param"]),
            "selected_cv_rmse": float(result["selected_cv_rmse"]),
            "elapsed_ms": float(elapsed_ms),
            **chunk_counters,
            **_aom_campaign_performance_metrics(
                n_chains=len(chunk),
                n_candidates=n_candidates,
                elapsed_ms=elapsed_ms,
                counters=chunk_counters,
            ),
        })
        completed_chunks.add(chunk_index)
        processed_this_run += 1
        if checkpoint is not None:
            _write_json_atomic(
                checkpoint,
                make_report(complete=len(completed_chunks) == n_total_chunks),
            )

    report = make_report(complete=len(completed_chunks) == n_total_chunks)
    if checkpoint is not None:
        _write_json_atomic(checkpoint, report)
    return report


def _finalize_aom_refit_report(
    rows: list[dict[str, object]],
    *,
    sort_by: str | None,
    cv: int,
    moment_policy,
    execution_mode: str,
    execution_mode_requested: str,
    execution_mode_auto_reason: str | None,
    execution_mode_auto_plan: dict[str, dict[str, object]] | None,
    auto_max_extra_fraction: float,
    n_refit_groups: int,
    aggregate: dict[str, int],
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, object]:
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["refit_cv_rmse"]), start=1
    ):
        row["cv_rank"] = int(rank)
    finite_screen = [
        row for row in rows
        if np.isfinite(float(row.get("screen_score", np.nan)))
    ]
    for rank, row in enumerate(
        sorted(finite_screen, key=lambda item: item["screen_score"]), start=1
    ):
        row["screen_rank"] = int(rank)
        row["rank_delta"] = int(row["cv_rank"] - row["screen_rank"])

    if sort_by is not None:
        if sort_by not in {"refit_cv_rmse", "screen_score", "screen_cv_rmse"}:
            raise ValueError(
                "sort_by must be refit_cv_rmse, screen_score, screen_cv_rmse, or None"
            )
        rows.sort(key=lambda item: item[sort_by])
    best_cv = min(rows, key=lambda item: item["refit_cv_rmse"])
    return {
        "rows": rows,
        "best_cv": best_cv,
        "n_candidates": int(len(rows)),
        "cv": int(cv),
        "moment_policy": moment_policy,
        "cuda_pls_min_device_features": cuda_pls_min_device_features,
        "cuda_pls_many_batched": (
            None
            if cuda_pls_many_batched is None
            else bool(cuda_pls_many_batched)
        ),
        "execution_mode": execution_mode,
        "execution_mode_requested": execution_mode_requested,
        "execution_mode_auto_reason": execution_mode_auto_reason,
        "execution_mode_auto_plan": execution_mode_auto_plan,
        "auto_max_extra_fraction": float(auto_max_extra_fraction),
        "n_refit_groups": int(n_refit_groups),
        "n_operator_moment_candidates": int(aggregate.get("n_operator_moment_candidates", 0)),
        "n_materialized_candidates": int(aggregate.get("n_materialized_candidates", 0)),
        "n_ridge_moment_cv_fits": int(
            aggregate.get("n_ridge_moment_cv_fits", 0)
        ),
        "n_ridge_dual_materialized_cv_fits": int(
            aggregate.get("n_ridge_dual_materialized_cv_fits", 0)
        ),
        "n_ridge_dual_cross_cv_fits": int(
            aggregate.get("n_ridge_dual_cross_cv_fits", 0)
        ),
        "n_ridge_moment_score_batch_calls": int(
            aggregate.get("n_ridge_moment_score_batch_calls", 0)
        ),
        "n_ridge_moment_score_batch_jobs": int(
            aggregate.get("n_ridge_moment_score_batch_jobs", 0)
        ),
        "n_ridge_moment_final_fits": int(
            aggregate.get("n_ridge_moment_final_fits", 0)
        ),
        "n_ridge_dual_materialized_final_fits": int(
            aggregate.get("n_ridge_dual_materialized_final_fits", 0)
        ),
        "n_pls_moment_cv_fits": int(aggregate.get("n_pls_moment_cv_fits", 0)),
        "n_pls_moment_host_cv_fits": int(
            aggregate.get("n_pls_moment_host_cv_fits", 0)
        ),
        "n_pls_moment_cuda_device_cv_fits": int(
            aggregate.get("n_pls_moment_cuda_device_cv_fits", 0)
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            aggregate.get("n_pls_moment_cuda_parallel_fold_batches", 0)
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            aggregate.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
        ),
        "n_pls_materialized_cv_fits": int(
            aggregate.get("n_pls_materialized_cv_fits", 0)
        ),
        "n_pls_gcv_proxy_fits": int(aggregate.get("n_pls_gcv_proxy_fits", 0)),
        "n_refit_scored_candidates": int(
            aggregate.get("n_refit_scored_candidates", len(rows))
        ),
        "n_refit_extra_scored_candidates": int(
            max(0, int(aggregate.get("n_refit_scored_candidates", len(rows))) - len(rows))
        ),
        "library_path": library_path(),
        "abi": ".".join(str(int(fn())) for fn in (
            lib.n4m_get_abi_version_major,
            lib.n4m_get_abi_version_minor,
            lib.n4m_get_abi_version_patch,
        )),
    }


def _aom_refit_candidate_list(candidates, top_k: int | None) -> list[dict[str, object]]:
    if isinstance(candidates, dict) and "top_candidates" in candidates:
        candidate_list = list(candidates["top_candidates"])
    else:
        candidate_list = list(candidates)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive when provided")
        candidate_list = candidate_list[: int(top_k)]
    if not candidate_list:
        raise ValueError("candidates must contain at least one candidate row")
    return candidate_list


def _aom_refit_candidate_key(candidate: dict[str, object]):
    chain, head, param = _candidate_chain_head_param(candidate)
    chain_key = json.dumps(
        _jsonable_value(chain, include_arrays=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (chain_key, head, _aom_refit_param_key(head, float(param)))


def _aom_screen_refit_candidate_union(
    screen: dict[str, object],
    *,
    refit_top_k: int,
    refit_per_head_top_k: int | None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    global_rows = list(screen.get("top_candidates", ()))[: int(refit_top_k)]
    selected: list[dict[str, object]] = []
    seen = set()

    def add(row) -> bool:
        key = _aom_refit_candidate_key(row)
        if key in seen:
            return False
        seen.add(key)
        selected.append(row)
        return True

    for row in global_rows:
        add(row)

    per_head_seen = 0
    per_head_added = 0
    if refit_per_head_top_k is not None:
        keep = int(refit_per_head_top_k)
        by_head = screen.get("top_candidates_by_head", {})
        if isinstance(by_head, dict):
            for head in sorted(by_head):
                rows = list(by_head.get(head, ()))[:keep]
                per_head_seen += len(rows)
                for row in rows:
                    if add(row):
                        per_head_added += 1

    stats = {
        "n_refit_global_candidates": int(len(global_rows)),
        "n_refit_per_head_candidates": int(per_head_seen),
        "n_refit_per_head_extra_candidates": int(per_head_added),
        "n_refit_union_candidates": int(len(selected)),
    }
    return selected, stats


def _aom_refit_param_key(head: str, param: float):
    if head == "pls":
        component = int(round(param))
        if component < 1 or abs(float(component) - param) > 1e-9:
            raise ValueError("PLS candidate param must be a positive integer")
        return component
    return float(param)


def aom_screen_refit_candidate_pool(
    screen: dict[str, object],
    *,
    refit_top_k: int | None = None,
    refit_per_head_top_k: int | None = None,
) -> dict[str, object]:
    """Return the exact candidate pool retained for AOM screen/refit.

    The helper is source-free: it only inspects a screen report returned by
    ``aom_chain_score_campaign``. It mirrors the retention step used by
    ``aom_chain_screen_refit_campaign`` before exact-CV refit, so mixed
    Ridge/PLS campaigns can audit how many global and per-head rows will be
    verified without touching ``X`` or ``y``.
    """
    if not isinstance(screen, dict):
        raise ValueError("screen must be an AOM campaign dictionary")
    if not screen.get("top_candidates"):
        raise ValueError("screen campaign contains no top_candidates")
    if refit_top_k is None:
        refit_keep = int(screen.get("top_k", len(screen["top_candidates"])))
    else:
        if int(refit_top_k) < 1:
            raise ValueError("refit_top_k must be positive when provided")
        refit_keep = int(refit_top_k)
    if refit_per_head_top_k is not None and int(refit_per_head_top_k) < 1:
        raise ValueError("refit_per_head_top_k must be positive when provided")
    rows, stats = _aom_screen_refit_candidate_union(
        screen,
        refit_top_k=refit_keep,
        refit_per_head_top_k=refit_per_head_top_k,
    )
    if not rows:
        raise ValueError("screen/refit candidate pool is empty")
    return {
        "report_schema": "n4m.aom_screen_refit_candidate_pool.v1",
        "rows": rows,
        "top_candidates": rows,
        "n_candidates": int(len(rows)),
        "refit_top_k": int(refit_keep),
        "refit_per_head_top_k": (
            None if refit_per_head_top_k is None else int(refit_per_head_top_k)
        ),
        **stats,
    }


def _aom_refit_chain_groups(
    candidate_list: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    group_order: list[tuple[str, str]] = []
    for index, candidate in enumerate(candidate_list):
        chain, head, param = _candidate_chain_head_param(candidate)
        chain_key = json.dumps(
            _jsonable_value(chain, include_arrays=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        key = (chain_key, head)
        if key not in groups:
            groups[key] = {"chain": chain, "head": head, "items": []}
            group_order.append(key)
        groups[key]["items"].append((index, candidate, param))

    chain_groups: list[dict[str, object]] = []
    for key in group_order:
        group = groups[key]
        head = str(group["head"])
        items = list(group["items"])
        param_keys: list[object] = []
        seen_params: set[object] = set()
        for _, _, param in items:
            param_key = _aom_refit_param_key(head, float(param))
            if param_key not in seen_params:
                seen_params.add(param_key)
                param_keys.append(param_key)
        chain_groups.append({
            "chain": group["chain"],
            "head": head,
            "items": items,
            "param_keys": tuple(sorted(param_keys)),
        })
    return chain_groups


def _aom_refit_batches(
    chain_groups: Sequence[dict[str, object]],
    mode: str,
) -> list[dict[str, object]]:
    if mode == "grouped_score":
        return [
            {
                "head": str(group["head"]),
                "param_keys": tuple(group["param_keys"]),
                "groups": [group],
            }
            for group in chain_groups
        ]

    if mode == "batched_score":
        signature_batches: dict[
            tuple[str, tuple[object, ...]],
            list[dict[str, object]],
        ] = {}
        batch_order: list[tuple[str, tuple[object, ...]]] = []
        for group in chain_groups:
            key = (str(group["head"]), tuple(group["param_keys"]))
            if key not in signature_batches:
                signature_batches[key] = []
                batch_order.append(key)
            signature_batches[key].append(group)
        return [
            {
                "head": key[0],
                "param_keys": key[1],
                "groups": signature_batches[key],
            }
            for key in batch_order
        ]

    if mode == "union_batched_score":
        head_batches: dict[str, list[dict[str, object]]] = {}
        head_order: list[str] = []
        for group in chain_groups:
            head = str(group["head"])
            if head not in head_batches:
                head_batches[head] = []
                head_order.append(head)
            head_batches[head].append(group)
        batches = []
        for head in head_order:
            params: set[object] = set()
            for group in head_batches[head]:
                params.update(group["param_keys"])
            batches.append({
                "head": head,
                "param_keys": tuple(sorted(params)),
                "groups": head_batches[head],
            })
        return batches

    raise ValueError("unknown AOM refit score batch mode")


def _aom_refit_batch_plan(
    candidate_list: Sequence[dict[str, object]],
    mode: str,
    chain_groups: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    n_rows = int(len(candidate_list))
    if mode == "individual":
        by_head: dict[str, dict[str, int]] = {}
        for candidate in candidate_list:
            _, head, _ = _candidate_chain_head_param(candidate)
            entry = by_head.setdefault(
                str(head),
                {"n_refit_candidates": 0, "n_refit_scored_candidates": 0},
            )
            entry["n_refit_candidates"] += 1
            entry["n_refit_scored_candidates"] += 1
        return {
            "execution_mode": "individual",
            "n_refit_candidates": n_rows,
            "n_chain_head_groups": n_rows,
            "n_refit_groups": n_rows,
            "n_refit_scored_candidates": n_rows,
            "n_refit_extra_scored_candidates": 0,
            "by_head": by_head,
        }

    groups = list(chain_groups) if chain_groups is not None else _aom_refit_chain_groups(candidate_list)
    batches = _aom_refit_batches(groups, mode)
    scored = 0
    by_head: dict[str, dict[str, int]] = {}
    for batch in batches:
        head = str(batch["head"])
        n_batch_groups = len(batch["groups"])
        n_params = len(batch["param_keys"])
        batch_scored = int(n_batch_groups * n_params)
        scored += batch_scored
        entry = by_head.setdefault(
            head,
            {
                "n_refit_groups": 0,
                "n_refit_candidates": 0,
                "n_refit_scored_candidates": 0,
            },
        )
        entry["n_refit_groups"] += 1
        entry["n_refit_scored_candidates"] += batch_scored
        for group in batch["groups"]:
            entry["n_refit_candidates"] += len(group["items"])
    return {
        "execution_mode": mode,
        "n_refit_candidates": n_rows,
        "n_chain_head_groups": int(len(groups)),
        "n_refit_groups": int(len(batches)),
        "n_refit_scored_candidates": int(scored),
        "n_refit_extra_scored_candidates": int(max(0, scored - n_rows)),
        "by_head": by_head,
    }


def _aom_refit_plans_by_mode(
    candidate_list: Sequence[dict[str, object]],
    chain_groups: Sequence[dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    groups = (
        list(chain_groups)
        if chain_groups is not None
        else _aom_refit_chain_groups(candidate_list)
    )
    modes = (
        "individual",
        "grouped_score",
        "batched_score",
        "union_batched_score",
    )
    return {
        mode: _aom_refit_batch_plan(candidate_list, mode, groups)
        for mode in modes
    }


def _validate_refit_auto_max_extra_fraction(value) -> float:
    fraction = float(value)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("auto_max_extra_fraction must be finite and non-negative")
    return fraction


def _select_aom_refit_auto_mode(
    candidate_list: Sequence[dict[str, object]],
    *,
    auto_max_extra_fraction: float,
) -> tuple[str, str, dict[str, dict[str, object]]]:
    chain_groups = _aom_refit_chain_groups(candidate_list)
    plans = _aom_refit_plans_by_mode(candidate_list, chain_groups)
    batched = plans["batched_score"]
    union = plans["union_batched_score"]
    n_candidates = max(1, int(len(candidate_list)))
    max_extra = float(auto_max_extra_fraction) * float(n_candidates)
    if (
        int(union["n_refit_groups"]) < int(batched["n_refit_groups"])
        and float(union["n_refit_extra_scored_candidates"]) <= max_extra
    ):
        return (
            "union_batched_score",
            "union_reduces_native_groups_within_extra_budget",
            plans,
        )
    return (
        "batched_score",
        "batched_preserves_parameter_signatures",
        plans,
    )


def aom_refit_execution_plan(
    candidates,
    *,
    top_k: int | None = None,
    auto_max_extra_fraction: float = 1.0,
) -> dict[str, object]:
    """Plan exact-CV refit execution costs without touching ``X`` or ``y``.

    The plan uses the same decoded candidate grouping logic as
    ``aom_refit_candidates`` and reports how many native refit groups and native
    candidate scores each score-only execution mode would pay. It is an audit
    helper for choosing an execution mode; it does not rank, refit or select
    candidates.
    """
    candidate_list = _aom_refit_candidate_list(candidates, top_k)
    auto_fraction = _validate_refit_auto_max_extra_fraction(auto_max_extra_fraction)
    chain_groups = _aom_refit_chain_groups(candidate_list)
    plans_by_mode = _aom_refit_plans_by_mode(candidate_list, chain_groups)
    recommended_mode, reason, _ = _select_aom_refit_auto_mode(
        candidate_list,
        auto_max_extra_fraction=auto_fraction,
    )
    return {
        "report_schema": "n4m.aom_refit_execution_plan.v1",
        "n_refit_candidates": int(len(candidate_list)),
        "n_chain_head_groups": int(len(chain_groups)),
        "recommended_mode": recommended_mode,
        "recommendation_reason": reason,
        "auto_max_extra_fraction": auto_fraction,
        "modes": list(plans_by_mode.values()),
        "by_mode": plans_by_mode,
    }


def aom_refit_candidates(
    X_train,
    y_train,
    candidates,
    *,
    top_k: int | None = None,
    sort_by: str | None = "refit_cv_rmse",
    cv: int = 5,
    fold_ids=None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    execution_mode: str = "individual",
    auto_max_extra_fraction: float = 1.0,
    return_predictions: bool = False,
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, object]:
    """Refit decoded AOM candidates and score them by exact native CV.

    This is the train-only verification step for broad score-only screens,
    including ``pls_score_mode="gcv_proxy"`` campaigns. It never reuses proxy
    scores for selection. ``execution_mode="individual"`` preserves the full
    one-row replay with train/OOF predictions. ``"grouped_score"`` groups rows
    sharing the same chain/head and scores their parameters together with
    exact native CV. ``"batched_score"`` keeps that exact scoring rule but
    batches multiple chains that retained the same head/parameter set into one
    native call, allowing native prefix caches to span retained chains.
    ``"union_batched_score"`` batches by head with the union of retained
    parameters for that head, which may score extra chain/parameter pairs but
    maps only the requested candidates back into the refit rows. Score modes
    only report CV scores, not per-candidate prediction arrays.
    """
    X_train_arr = as_f64_2d(X_train)
    y_train_arr = _as_y_matrix(y_train, X_train_arr.shape[0])

    if isinstance(candidates, dict) and "top_candidates" in candidates:
        candidate_list = list(candidates["top_candidates"])
    else:
        candidate_list = list(candidates)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive when provided")
        candidate_list = candidate_list[: int(top_k)]
    if not candidate_list:
        raise ValueError("candidates must contain at least one candidate row")

    mode_requested = str(execution_mode).lower()
    mode = mode_requested
    auto_fraction = _validate_refit_auto_max_extra_fraction(auto_max_extra_fraction)
    auto_reason: str | None = None
    auto_plan: dict[str, dict[str, object]] | None = None
    if mode == "auto":
        if return_predictions:
            mode = "individual"
            auto_reason = "return_predictions_requires_individual"
        else:
            mode, auto_reason, auto_plan = _select_aom_refit_auto_mode(
                candidate_list,
                auto_max_extra_fraction=auto_fraction,
            )
    if mode not in {
        "individual",
        "grouped_score",
        "batched_score",
        "union_batched_score",
    }:
        raise ValueError(
            "execution_mode must be individual, grouped_score, batched_score, "
            "union_batched_score, or auto"
        )
    if return_predictions and mode != "individual":
        raise ValueError("return_predictions requires execution_mode='individual'")
    cuda_pls_min_device_features_i = (
        None
        if cuda_pls_min_device_features is None
        else int(cuda_pls_min_device_features)
    )
    if (
        cuda_pls_min_device_features_i is not None
        and cuda_pls_min_device_features_i < 1
    ):
        raise ValueError("cuda_pls_min_device_features must be positive")

    aggregate = {key: 0 for key in _AOM_ROUTE_COUNTER_KEYS}
    aggregate["n_refit_scored_candidates"] = 0

    if mode in {"grouped_score", "batched_score", "union_batched_score"}:
        chain_groups = _aom_refit_chain_groups(candidate_list)
        batches = _aom_refit_batches(chain_groups, mode)

        rows_by_index: dict[int, dict[str, object]] = {}
        for group_id, batch_spec in enumerate(batches):
            head = str(batch_spec["head"])
            param_keys = tuple(batch_spec["param_keys"])
            batch = list(batch_spec["groups"])
            chains = [group["chain"] for group in batch]

            if head == "ridge":
                ridge_lambdas = tuple(float(param) for param in param_keys)
                pls_components = ()
            else:
                ridge_lambdas = ()
                pls_components = tuple(int(param) for param in param_keys)

            result = aom_chain_sweep_run(
                X_train_arr,
                y_train_arr,
                chains,
                cv=cv,
                fold_ids=fold_ids,
                ridge_lambdas=ridge_lambdas,
                pls_components=pls_components,
                heads=(head,),
                center_x=center_x,
                scale_x=scale_x,
                center_y=center_y,
                scale_y=scale_y,
                moment_policy=moment_policy,
                pls_score_mode="cv",
                score_only=True,
                cuda_pls_parallel_folds=cuda_pls_parallel_folds,
                cuda_pls_min_device_features=cuda_pls_min_device_features_i,
                cuda_pls_many_batched=cuda_pls_many_batched,
            )
            for counter in aggregate:
                aggregate[counter] += int(result.get(counter, 0))
            aggregate["n_refit_scored_candidates"] += int(
                result.get("n_candidates", 0)
            )

            scores = np.asarray(result["candidate_scores"], dtype=np.float64)
            if scores.ndim != 2 or scores.shape[1] < 5:
                raise ValueError("AOM grouped refit returned invalid candidate_scores")
            score_by_chain_param: dict[tuple[int, object], float] = {}
            for score_row in scores:
                chain_index = int(round(float(score_row[1])))
                if head == "pls":
                    score_key = int(round(float(score_row[3])))
                else:
                    score_key = float(score_row[3])
                score_by_chain_param[(chain_index, score_key)] = float(score_row[4])

            group_counters = {
                "refit_group_n_operator_moment_candidates": int(
                    result.get("n_operator_moment_candidates", 0)
                ),
                "refit_group_n_materialized_candidates": int(
                    result.get("n_materialized_candidates", 0)
                ),
                "refit_group_n_ridge_moment_cv_fits": int(
                    result.get("n_ridge_moment_cv_fits", 0)
                ),
                "refit_group_n_ridge_dual_materialized_cv_fits": int(
                    result.get("n_ridge_dual_materialized_cv_fits", 0)
                ),
                "refit_group_n_ridge_dual_cross_cv_fits": int(
                    result.get("n_ridge_dual_cross_cv_fits", 0)
                ),
                "refit_group_n_ridge_moment_score_batch_calls": int(
                    result.get("n_ridge_moment_score_batch_calls", 0)
                ),
                "refit_group_n_ridge_moment_score_batch_jobs": int(
                    result.get("n_ridge_moment_score_batch_jobs", 0)
                ),
                "refit_group_n_pls_moment_cv_fits": int(
                    result.get("n_pls_moment_cv_fits", 0)
                ),
                "refit_group_n_pls_moment_host_cv_fits": int(
                    result.get("n_pls_moment_host_cv_fits", 0)
                ),
                "refit_group_n_pls_moment_cuda_device_cv_fits": int(
                    result.get("n_pls_moment_cuda_device_cv_fits", 0)
                ),
                "refit_group_n_pls_moment_cuda_parallel_fold_batches": int(
                    result.get("n_pls_moment_cuda_parallel_fold_batches", 0)
                ),
                "refit_group_n_pls_moment_cuda_parallel_fold_jobs": int(
                    result.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
                ),
                "refit_group_n_pls_materialized_cv_fits": int(
                    result.get("n_pls_materialized_cv_fits", 0)
                ),
                "refit_group_n_pls_gcv_proxy_fits": int(
                    result.get("n_pls_gcv_proxy_fits", 0)
                ),
            }
            for chain_index, chain_group in enumerate(batch):
                chain = chain_group["chain"]
                for index, candidate, param in chain_group["items"]:
                    param_key = int(round(param)) if head == "pls" else float(param)
                    score_key = (chain_index, param_key)
                    if score_key not in score_by_chain_param:
                        raise ValueError("AOM grouped refit did not return a candidate score")
                    refit_score = score_by_chain_param[score_key]
                    screen_score = float(candidate.get("cv_rmse", np.nan))
                    screen_metric = str(candidate.get("score_metric", "cv_rmse"))
                    row = {
                        "source_index": int(index),
                        "candidate_id": int(candidate.get("candidate_id", index)),
                        "global_candidate_id": int(candidate.get(
                            "global_candidate_id",
                            candidate.get("candidate_id", index),
                        )),
                        "chain_id": int(candidate.get("chain_id", index)),
                        "chain": chain,
                        "head": head,
                        "head_id": 0 if head == "ridge" else 1,
                        "param": float(param),
                        "screen_score": screen_score,
                        "screen_score_metric": screen_metric,
                        "screen_cv_rmse": screen_score,
                        "refit_cv_rmse": float(refit_score),
                        "refit_score_metric": "cv_rmse",
                        "train_rmse": float("nan"),
                        "oof_rmse": float(refit_score),
                        "n_operator_moment_candidates": 0,
                        "n_materialized_candidates": 0,
                        "n_ridge_moment_cv_fits": 0,
                        "n_ridge_dual_materialized_cv_fits": 0,
                        "n_ridge_dual_cross_cv_fits": 0,
                        "n_ridge_moment_score_batch_calls": 0,
                        "n_ridge_moment_score_batch_jobs": 0,
                        "n_pls_moment_cv_fits": 0,
                        "n_pls_materialized_cv_fits": 0,
                        "n_pls_gcv_proxy_fits": 0,
                        "refit_group_id": int(group_id),
                        **group_counters,
                    }
                    if "score_route" in candidate:
                        row["screen_score_route"] = str(candidate["score_route"])
                    if "score_route_id" in candidate:
                        row["screen_score_route_id"] = int(candidate["score_route_id"])
                    rows_by_index[index] = row

        rows = [rows_by_index[index] for index in range(len(candidate_list))]
        return _finalize_aom_refit_report(
            rows,
            sort_by=sort_by,
            cv=cv,
            moment_policy=moment_policy,
            execution_mode=mode,
            execution_mode_requested=mode_requested,
            execution_mode_auto_reason=auto_reason,
            execution_mode_auto_plan=auto_plan,
            auto_max_extra_fraction=auto_fraction,
            n_refit_groups=len(batches),
            aggregate=aggregate,
            cuda_pls_min_device_features=cuda_pls_min_device_features_i,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )

    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_list):
        chain, head, param = _candidate_chain_head_param(candidate)
        if head == "ridge":
            ridge_lambdas = (param,)
            pls_components = ()
        else:
            component = int(round(param))
            if component < 1 or abs(float(component) - param) > 1e-9:
                raise ValueError("PLS candidate param must be a positive integer")
            ridge_lambdas = ()
            pls_components = (component,)

        result = aom_chain_sweep_run(
            X_train_arr,
            y_train_arr,
            [chain],
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=(head,),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            pls_score_mode="cv",
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features_i,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )
        aggregate["n_refit_scored_candidates"] += int(
            result.get("n_candidates", 1)
        )
        train_predictions = np.asarray(result["predictions"], dtype=np.float64)
        oof_predictions = np.asarray(result["oof_predictions"], dtype=np.float64)
        screen_score = float(candidate.get("cv_rmse", np.nan))
        screen_metric = str(candidate.get("score_metric", "cv_rmse"))
        row = {
            "source_index": int(index),
            "candidate_id": int(candidate.get("candidate_id", index)),
            "global_candidate_id": int(candidate.get(
                "global_candidate_id",
                candidate.get("candidate_id", index),
            )),
            "chain_id": int(candidate.get("chain_id", index)),
            "chain": chain,
            "head": head,
            "head_id": 0 if head == "ridge" else 1,
            "param": param,
            "screen_score": screen_score,
            "screen_score_metric": screen_metric,
            "screen_cv_rmse": screen_score,
            "refit_cv_rmse": float(result["selected_cv_rmse"]),
            "refit_score_metric": "cv_rmse",
            "train_rmse": _regression_rmse(y_train_arr, train_predictions),
            "oof_rmse": _regression_rmse(y_train_arr, oof_predictions),
            "n_operator_moment_candidates": int(result["n_operator_moment_candidates"]),
            "n_materialized_candidates": int(result["n_materialized_candidates"]),
            "n_ridge_moment_cv_fits": int(result.get("n_ridge_moment_cv_fits", 0)),
            "n_ridge_dual_materialized_cv_fits": int(
                result.get("n_ridge_dual_materialized_cv_fits", 0)
            ),
            "n_ridge_dual_cross_cv_fits": int(
                result.get("n_ridge_dual_cross_cv_fits", 0)
            ),
            "n_ridge_moment_score_batch_calls": int(
                result.get("n_ridge_moment_score_batch_calls", 0)
            ),
            "n_ridge_moment_score_batch_jobs": int(
                result.get("n_ridge_moment_score_batch_jobs", 0)
            ),
            "n_pls_moment_cv_fits": int(result.get("n_pls_moment_cv_fits", 0)),
            "n_pls_moment_host_cv_fits": int(
                result.get("n_pls_moment_host_cv_fits", 0)
            ),
            "n_pls_moment_cuda_device_cv_fits": int(
                result.get("n_pls_moment_cuda_device_cv_fits", 0)
            ),
            "n_pls_moment_cuda_parallel_fold_batches": int(
                result.get("n_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "n_pls_moment_cuda_parallel_fold_jobs": int(
                result.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_pls_materialized_cv_fits": int(result.get("n_pls_materialized_cv_fits", 0)),
            "n_pls_gcv_proxy_fits": int(result.get("n_pls_gcv_proxy_fits", 0)),
        }
        for counter in aggregate:
            aggregate[counter] += int(result.get(counter, 0))
        if "score_route" in candidate:
            row["screen_score_route"] = str(candidate["score_route"])
        if "score_route_id" in candidate:
            row["screen_score_route_id"] = int(candidate["score_route_id"])
        if return_predictions:
            row["train_predictions"] = train_predictions.copy()
            row["oof_predictions"] = oof_predictions.copy()
        rows.append(row)

    return _finalize_aom_refit_report(
        rows,
        sort_by=sort_by,
        cv=cv,
        moment_policy=moment_policy,
        execution_mode=mode,
        execution_mode_requested=mode_requested,
        execution_mode_auto_reason=auto_reason,
        execution_mode_auto_plan=auto_plan,
        auto_max_extra_fraction=auto_fraction,
        n_refit_groups=len(rows),
        aggregate=aggregate,
        cuda_pls_min_device_features=cuda_pls_min_device_features_i,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )


def aom_chain_screen_refit_campaign(
    X,
    y,
    chains=None,
    *,
    profile: str = "lab",
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    max_chains: int | None = None,
    chain_chunk_size: int = 4096,
    top_k: int = 50,
    refit_top_k: int | None = None,
    refit_per_head_top_k: int | None = None,
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(0.01, 0.1, 1.0, 10.0),
    pls_components=(1, 2, 4),
    heads=("ridge", "pls"),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    refit_moment_policy: str | int | None = None,
    pls_score_mode: str | int = "cv",
    chain_ordering: str = "input",
    split_head_scoring: str = "off",
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
    backend_cuda_available: bool | None = None,
    backend_min_cuda_product: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    max_chunks_per_run: int | None = None,
    refit_sort_by: str | None = "refit_cv_rmse",
    refit_execution: str = "auto",
    refit_auto_max_extra_fraction: float = 1.0,
    return_predictions: bool = False,
) -> dict[str, object]:
    """Run a score-only AOM campaign, then exact-CV refit retained rows.

    The first pass is ``aom_chain_score_campaign`` and may use a proxy score
    such as ``pls_score_mode="gcv_proxy"``. The second pass replays the
    retained rows with ``aom_refit_candidates`` and exact native CV. If a
    checkpointed screen is partial, the helper refits the current available
    top rows and marks ``screen_complete=False`` in the combined report.
    """
    if refit_top_k is not None and int(refit_top_k) < 1:
        raise ValueError("refit_top_k must be positive when provided")
    if refit_per_head_top_k is not None and int(refit_per_head_top_k) < 1:
        raise ValueError("refit_per_head_top_k must be positive when provided")

    screen = aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        profile=profile,
        families=families,
        templates=templates,
        max_chains=max_chains,
        chain_chunk_size=chain_chunk_size,
        top_k=top_k,
        cv=cv,
        fold_ids=fold_ids,
        ridge_lambdas=ridge_lambdas,
        pls_components=pls_components,
        heads=heads,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
        moment_policy=moment_policy,
        pls_score_mode=pls_score_mode,
        chain_ordering=chain_ordering,
        split_head_scoring=split_head_scoring,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
        backend_cuda_available=backend_cuda_available,
        backend_min_cuda_product=backend_min_cuda_product,
        checkpoint_path=checkpoint_path,
        resume=resume,
        max_chunks_per_run=max_chunks_per_run,
    )
    if not screen.get("top_candidates"):
        raise ValueError("screen campaign returned no candidates to refit")

    refit_keep = int(top_k) if refit_top_k is None else int(refit_top_k)
    refit_candidates, refit_candidate_stats = _aom_screen_refit_candidate_union(
        screen,
        refit_top_k=refit_keep,
        refit_per_head_top_k=refit_per_head_top_k,
    )
    exact_policy = moment_policy if refit_moment_policy is None else refit_moment_policy
    refit = aom_refit_candidates(
        X,
        y,
        refit_candidates,
        top_k=None,
        sort_by=refit_sort_by,
        cv=cv,
        fold_ids=fold_ids,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
        moment_policy=exact_policy,
        execution_mode=refit_execution,
        auto_max_extra_fraction=refit_auto_max_extra_fraction,
        return_predictions=return_predictions,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )
    screen_complete = bool(screen.get("complete", True))
    report = {
        "report_schema": "n4m.aom_chain_screen_refit_campaign.v1",
        "complete": screen_complete,
        "screen_complete": screen_complete,
        "refit_complete": True,
        "screen": screen,
        "refit": refit,
        "top_candidates": screen.get("top_candidates", []),
        "rows": refit["rows"],
        "best": screen.get("best"),
        "best_screen": screen.get("best"),
        "best_cv": refit["best_cv"],
        "best_refit": refit["best_cv"],
        "n_screen_candidates": int(screen.get("n_candidates", 0)),
        "n_screen_top_candidates": int(len(screen.get("top_candidates", ()))),
        "n_refit_candidates": int(refit["n_candidates"]),
        "n_refit_scored_candidates": int(
            refit.get("n_refit_scored_candidates", refit["n_candidates"])
        ),
        "n_refit_extra_scored_candidates": int(
            refit.get("n_refit_extra_scored_candidates", 0)
        ),
        "top_k": int(top_k),
        "refit_top_k": refit_keep,
        "refit_per_head_top_k": (
            None if refit_per_head_top_k is None else int(refit_per_head_top_k)
        ),
        **refit_candidate_stats,
        "cv": int(cv),
        "moment_policy": moment_policy,
        "refit_moment_policy": exact_policy,
        "refit_execution": str(refit["execution_mode"]),
        "refit_execution_requested": str(refit.get("execution_mode_requested", refit_execution)),
        "refit_execution_auto_reason": refit.get("execution_mode_auto_reason"),
        "refit_auto_max_extra_fraction": float(
            refit.get("auto_max_extra_fraction", refit_auto_max_extra_fraction)
        ),
        "pls_score_mode": str(
            screen.get("pls_score_mode", _aom_pls_score_mode_name(pls_score_mode))
        ),
        "refit_pls_score_mode": "cv",
        "chain_ordering": screen.get("chain_ordering", "input"),
        "split_head_scoring": screen.get("split_head_scoring", "off"),
        "cuda_pls_parallel_folds": screen.get("cuda_pls_parallel_folds"),
        "cuda_pls_min_device_features": screen.get(
            "cuda_pls_min_device_features"
        ),
        "cuda_pls_many_batched": screen.get("cuda_pls_many_batched"),
        "n_split_head_chunks": int(screen.get("n_split_head_chunks", 0)),
        "n_chunk_score_calls": int(screen.get("n_chunk_score_calls", 0)),
        "moment_backend_recommendations": screen.get(
            "moment_backend_recommendations", {}
        ),
        "moment_backend_recommendation_policy_inputs": screen.get(
            "moment_backend_recommendation_policy_inputs",
            (
                "n_samples",
                "n_features",
                "head",
                "cuda_available",
                "min_cuda_product",
                "cuda_pls_min_device_features",
                "cuda_pls_many_batched",
            ),
        ),
        "backend_cuda_available": backend_cuda_available,
        "backend_min_cuda_product": screen.get("backend_min_cuda_product"),
        "checkpoint_path": screen.get("checkpoint_path"),
        "library_path": library_path(),
        "abi": ".".join(str(int(fn())) for fn in (
            lib.n4m_get_abi_version_major,
            lib.n4m_get_abi_version_minor,
            lib.n4m_get_abi_version_patch,
        )),
    }
    return report


def aom_moment_screen_refit_campaign(
    X,
    y,
    chains=None,
    *,
    profile: str = "lab",
    families: dict | None = None,
    templates: Sequence[Sequence[str]] | None = None,
    max_chains: int | None = None,
    chain_chunk_size: int = 4096,
    top_k: int = 50,
    refit_top_k: int | None = None,
    refit_per_head_top_k: int | None = 10,
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ),
    pls_components=(1, 2, 3, 4, 6, 8),
    heads=("ridge", "pls"),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "force_moments",
    refit_moment_policy: str | int | None = None,
    pls_score_mode: str | int = "gcv_proxy",
    chain_ordering: str = "prefix",
    split_head_scoring: str = "auto",
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
    backend_cuda_available: bool | None = None,
    backend_min_cuda_product: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    max_chunks_per_run: int | None = None,
    refit_sort_by: str | None = "refit_cv_rmse",
    refit_execution: str = "auto",
    refit_auto_max_extra_fraction: float = 1.0,
    return_predictions: bool = False,
) -> dict[str, object]:
    """Run the preconfigured fast AOM/moment screen-refit campaign.

    This is the functional equivalent of the sklearn
    ``NativeAOMMomentScreenRefitRegressor`` preset: strict moment routes,
    prefix-packed chain chunks, split-head mixed Ridge/PLS scoring, a PLS GCV
    proxy first pass, exact-CV refit of retained rows, and reusable final rows.
    """
    report = aom_chain_screen_refit_campaign(
        X,
        y,
        chains=chains,
        profile=profile,
        families=families,
        templates=templates,
        max_chains=max_chains,
        chain_chunk_size=chain_chunk_size,
        top_k=top_k,
        refit_top_k=refit_top_k,
        refit_per_head_top_k=refit_per_head_top_k,
        cv=cv,
        fold_ids=fold_ids,
        ridge_lambdas=ridge_lambdas,
        pls_components=pls_components,
        heads=heads,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
        moment_policy=moment_policy,
        refit_moment_policy=refit_moment_policy,
        pls_score_mode=pls_score_mode,
        chain_ordering=chain_ordering,
        split_head_scoring=split_head_scoring,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
        backend_cuda_available=backend_cuda_available,
        backend_min_cuda_product=backend_min_cuda_product,
        checkpoint_path=checkpoint_path,
        resume=resume,
        max_chunks_per_run=max_chunks_per_run,
        refit_sort_by=refit_sort_by,
        refit_execution=refit_execution,
        refit_auto_max_extra_fraction=refit_auto_max_extra_fraction,
        return_predictions=return_predictions,
    )
    report = dict(report)
    report["campaign_preset"] = "moment_fast_screen_refit"
    return report


def aom_evaluate_candidates(
    X_train,
    y_train,
    X_eval,
    y_eval,
    candidates,
    *,
    top_k: int | None = None,
    sort_by: str | None = "eval_rmse",
    cv: int = 5,
    fold_ids=None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    return_predictions: bool = False,
) -> dict[str, object]:
    """Refit decoded AOM candidates and score them on an explicit eval set.

    This is an analysis helper: the eval set is supplied by the caller and is
    never used to choose a route or alter the native fit.
    """
    X_train_arr = as_f64_2d(X_train)
    y_train_arr = _as_y_matrix(y_train, X_train_arr.shape[0])
    X_eval_arr = as_f64_2d(X_eval)
    y_eval_arr = _as_y_matrix(y_eval, X_eval_arr.shape[0])
    if X_eval_arr.shape[1] != X_train_arr.shape[1]:
        raise ValueError("X_eval must have the same number of features as X_train")
    if y_eval_arr.shape[1] != y_train_arr.shape[1]:
        raise ValueError("y_eval must have the same number of targets as y_train")

    if isinstance(candidates, dict) and "top_candidates" in candidates:
        candidate_list = list(candidates["top_candidates"])
    else:
        candidate_list = list(candidates)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive when provided")
        candidate_list = candidate_list[: int(top_k)]
    if not candidate_list:
        raise ValueError("candidates must contain at least one candidate row")

    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_list):
        chain, head, param = _candidate_chain_head_param(candidate)
        if head == "ridge":
            ridge_lambdas = (param,)
            pls_components = ()
        else:
            component = int(round(param))
            if component < 1 or abs(float(component) - param) > 1e-9:
                raise ValueError("PLS candidate param must be a positive integer")
            ridge_lambdas = ()
            pls_components = (component,)

        result = aom_chain_sweep_run(
            X_train_arr,
            y_train_arr,
            [chain],
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=(head,),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
        )
        coef = np.asarray(result["input_coefficients"], dtype=np.float64)
        intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(1, -1)
        eval_predictions = X_eval_arr @ coef + intercept
        train_predictions = np.asarray(result["predictions"], dtype=np.float64)
        oof_predictions = np.asarray(result["oof_predictions"], dtype=np.float64)

        row = {
            "source_index": int(index),
            "candidate_id": int(candidate.get("candidate_id", index)),
            "global_candidate_id": int(candidate.get(
                "global_candidate_id",
                candidate.get("candidate_id", index),
            )),
            "chain_id": int(candidate.get("chain_id", index)),
            "chain": chain,
            "head": head,
            "head_id": 0 if head == "ridge" else 1,
            "param": param,
            "screen_cv_rmse": float(candidate.get("cv_rmse", np.nan)),
            "refit_cv_rmse": float(result["selected_cv_rmse"]),
            "train_rmse": _regression_rmse(y_train_arr, train_predictions),
            "oof_rmse": _regression_rmse(y_train_arr, oof_predictions),
            "eval_rmse": _regression_rmse(y_eval_arr, eval_predictions),
            "eval_r2": _regression_r2(y_eval_arr, eval_predictions),
            "n_operator_moment_candidates": int(result["n_operator_moment_candidates"]),
            "n_materialized_candidates": int(result["n_materialized_candidates"]),
        }
        if return_predictions:
            row["train_predictions"] = train_predictions.copy()
            row["oof_predictions"] = oof_predictions.copy()
            row["eval_predictions"] = eval_predictions.copy()
        rows.append(row)

    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["refit_cv_rmse"]), start=1
    ):
        row["cv_rank"] = int(rank)
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["eval_rmse"]), start=1
    ):
        row["eval_rank"] = int(rank)
        row["rank_delta"] = int(row["eval_rank"] - row["cv_rank"])

    if sort_by is not None:
        if sort_by not in {"eval_rmse", "refit_cv_rmse", "screen_cv_rmse"}:
            raise ValueError("sort_by must be eval_rmse, refit_cv_rmse, screen_cv_rmse, or None")
        rows.sort(key=lambda item: item[sort_by])
    best_eval = min(rows, key=lambda item: item["eval_rmse"])
    best_cv = min(rows, key=lambda item: item["refit_cv_rmse"])
    return {
        "rows": rows,
        "best_eval": best_eval,
        "best_cv": best_cv,
        "n_candidates": int(len(rows)),
        "cv": int(cv),
        "moment_policy": moment_policy,
        "library_path": library_path(),
        "abi": ".".join(str(int(fn())) for fn in (
            lib.n4m_get_abi_version_major,
            lib.n4m_get_abi_version_minor,
            lib.n4m_get_abi_version_patch,
        )),
    }


def aom_candidate_report_records(
    report_or_rows,
    *,
    include_predictions: bool = False,
) -> list[dict[str, object]]:
    """Flatten AOM campaign/evaluation candidate rows for JSON/CSV export."""
    rows = []
    for row in _candidate_report_rows(report_or_rows):
        record = {}
        for key, value in row.items():
            if key.endswith("_predictions") and not include_predictions:
                continue
            converted = _jsonable_value(value, include_arrays=include_predictions)
            if converted is not None:
                record[str(key)] = converted
        if "chain" in record:
            record["chain_json"] = json.dumps(record["chain"], separators=(",", ":"))
        rows.append(record)
    return rows


def aom_candidate_operator_summary(
    report_or_rows,
    *,
    score_key: str | None = None,
    top_k: int | None = None,
) -> dict[str, object]:
    """Summarize scored AOM candidates by head and preprocessing operator."""
    key, rows = _aom_candidate_analysis_rows(report_or_rows, score_key)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive when provided")
        rows = rows[: int(top_k)]

    by_head: dict[str, list[dict[str, object]]] = {}
    by_operator: dict[str, list[dict[str, object]]] = {}
    by_operator_head: dict[str, list[dict[str, object]]] = {}
    by_chain_length: dict[str, list[dict[str, object]]] = {}
    by_score_route: dict[str, list[dict[str, object]]] = {}

    for item in rows:
        head = item["head"]
        chain = item["chain"]
        row = item["row"]
        by_head.setdefault(head, []).append(item)
        by_chain_length.setdefault(str(len(chain)), []).append(item)
        if "score_route" in row:
            by_score_route.setdefault(str(row["score_route"]), []).append(item)
        elif "score_route_id" in row:
            by_score_route.setdefault(
                _aom_score_route_name(int(row["score_route_id"])), []
            ).append(item)
        operators = sorted({name for name, _params in chain})
        for op in operators:
            by_operator.setdefault(op, []).append(item)
            by_operator_head.setdefault(f"{op}|{head}", []).append(item)

    def summarize(groups):
        out = [_aom_group_summary(name, items) for name, items in groups.items()]
        out.sort(key=lambda row: (row["best_score"], row["mean_rank"], row["group"]))
        return out

    best = rows[0]
    return {
        "score_key": key,
        "n_candidates": int(len(rows)),
        "best": best["row"],
        "best_score": float(best["score"]),
        "by_head": summarize(by_head),
        "by_operator": summarize(by_operator),
        "by_operator_head": summarize(by_operator_head),
        "by_chain_length": summarize(by_chain_length),
        "by_score_route": summarize(by_score_route),
    }


def aom_candidate_preprocessing_impact(
    report_or_rows,
    *,
    score_key: str | None = None,
    top_k: int | None = None,
    higher_is_better: bool = False,
) -> dict[str, object]:
    """Summarize preprocessing-stage impact from scored AOM candidate rows.

    The report is post-hoc and source-free: it only uses decoded candidate
    chains plus an already-computed score. For RMSE-like scores keep the default
    ``higher_is_better=False``; for R2-like scores set it to True.
    """
    raw_rows = _candidate_report_rows(report_or_rows)
    rows = [_restore_aom_candidate_record(dict(row)) for row in raw_rows]
    key = _aom_candidate_score_key(rows, score_key)
    parsed = []
    for source_index, row in enumerate(rows):
        if key not in row:
            continue
        try:
            score = float(row[key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        chain = _canonicalize_aom_chain(row.get("chain", ()))
        head = (
            _normalize_sweep_head_name(row["head"])
            if "head" in row
            else _normalize_sweep_head_name(row["head_id"])
        )
        parsed.append({
            "source_index": int(source_index),
            "row": row,
            "score": score,
            "chain": chain,
            "head": head,
        })
    if not parsed:
        raise ValueError(f"no finite candidate scores found for {key!r}")
    parsed.sort(
        key=lambda item: (
            -float(item["score"]) if higher_is_better else float(item["score"]),
            item["source_index"],
        )
    )
    for rank, item in enumerate(parsed, start=1):
        item["rank"] = int(rank)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive when provided")
        parsed = parsed[: int(top_k)]

    best_score = float(parsed[0]["score"])
    baseline_by_head: dict[str, dict[str, object]] = {}
    baseline_all = []
    for item in parsed:
        chain = item["chain"]
        if not chain or all(str(name) == "identity" for name, _params in chain):
            baseline_all.append(item)
            head = str(item["head"])
            previous = baseline_by_head.get(head)
            if previous is None:
                baseline_by_head[head] = item
                continue
            better = (
                float(item["score"]) > float(previous["score"])
                if higher_is_better
                else float(item["score"]) < float(previous["score"])
            )
            if better:
                baseline_by_head[head] = item

    baseline_item = None
    if baseline_all:
        baseline_item = max(baseline_all, key=lambda item: item["score"]) if higher_is_better else min(
            baseline_all, key=lambda item: item["score"]
        )

    def improvement_vs_baseline(score: float, head: str | None = None):
        ref = None
        if head is not None and head in baseline_by_head:
            ref = baseline_by_head[head]
        elif baseline_item is not None:
            ref = baseline_item
        if ref is None:
            return None
        baseline_score = float(ref["score"])
        return float(score - baseline_score) if higher_is_better else float(baseline_score - score)

    def loss_vs_global(score: float) -> float:
        return float(best_score - score) if higher_is_better else float(score - best_score)

    def add(grouped: dict[str, list[dict[str, object]]], key_name: str, item):
        grouped.setdefault(key_name, []).append(item)

    by_operator: dict[str, list[dict[str, object]]] = {}
    by_stage_family: dict[str, list[dict[str, object]]] = {}
    by_position: dict[str, list[dict[str, object]]] = {}
    by_stage_operator: dict[str, list[dict[str, object]]] = {}
    by_stage_option: dict[str, list[dict[str, object]]] = {}
    by_head_operator: dict[str, list[dict[str, object]]] = {}
    by_head_stage_option: dict[str, list[dict[str, object]]] = {}

    for item in parsed:
        chain = item["chain"] or [("identity", ())]
        head = str(item["head"])
        seen_ops = set()
        seen_stages = set()
        seen_positions = set()
        seen_stage_ops = set()
        seen_stage_options = set()
        for position, (name, params) in enumerate(chain):
            op = str(name)
            stage = _aom_operator_stage(op)
            option = _aom_operator_option_label(op, params)
            position_key = f"position_{position}"
            stage_operator = f"{stage}|{op}"
            stage_option = f"{stage}|{option}"
            seen_ops.add(op)
            seen_stages.add(stage)
            seen_positions.add(position_key)
            seen_stage_ops.add(stage_operator)
            seen_stage_options.add(stage_option)
        for op in seen_ops:
            add(by_operator, op, item)
            add(by_head_operator, f"{head}|{op}", item)
        for stage in seen_stages:
            add(by_stage_family, stage, item)
        for position_key in seen_positions:
            add(by_position, position_key, item)
        for stage_operator in seen_stage_ops:
            add(by_stage_operator, stage_operator, item)
        for stage_option in seen_stage_options:
            add(by_stage_option, stage_option, item)
            add(by_head_stage_option, f"{head}|{stage_option}", item)

    def summarize(groups):
        out = []
        total = len(parsed)
        for name, items in groups.items():
            scores = np.asarray([item["score"] for item in items], dtype=np.float64)
            ranks = np.asarray([item["rank"] for item in items], dtype=np.float64)
            best_item = max(items, key=lambda item: item["score"]) if higher_is_better else min(
                items, key=lambda item: item["score"]
            )
            group_best = float(best_item["score"])
            out.append({
                "group": name,
                "n_candidates": int(len(items)),
                "candidate_fraction": float(len(items) / total) if total else 0.0,
                "best_score": group_best,
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "best_rank": int(best_item["rank"]),
                "mean_rank": float(np.mean(ranks)),
                "best_loss_vs_global_best": loss_vs_global(group_best),
                "best_improvement_vs_identity": improvement_vs_baseline(
                    group_best, str(best_item["head"])
                ),
                "best_candidate": best_item["row"],
            })
        out.sort(
            key=lambda row: (
                row["best_loss_vs_global_best"],
                row["mean_rank"],
                str(row["group"]),
            )
        )
        return out

    return {
        "score_key": key,
        "higher_is_better": bool(higher_is_better),
        "n_candidates": int(len(parsed)),
        "best": parsed[0]["row"],
        "best_score": best_score,
        "identity_baseline": None if baseline_item is None else {
            "score": float(baseline_item["score"]),
            "rank": int(baseline_item["rank"]),
            "candidate": baseline_item["row"],
        },
        "identity_baseline_by_head": {
            head: {
                "score": float(item["score"]),
                "rank": int(item["rank"]),
                "candidate": item["row"],
            }
            for head, item in sorted(baseline_by_head.items())
        },
        "by_operator": summarize(by_operator),
        "by_stage_family": summarize(by_stage_family),
        "by_position": summarize(by_position),
        "by_stage_operator": summarize(by_stage_operator),
        "by_stage_option": summarize(by_stage_option),
        "by_head_operator": summarize(by_head_operator),
        "by_head_stage_option": summarize(by_head_stage_option),
    }


def aom_candidate_route_summary(report_or_rows) -> dict[str, object]:
    """Summarize materialized vs operator-moment routes in AOM candidate rows."""
    operator_routes = {
        "dense_operator_moment",
        "banded_operator_moment",
        "structured_operator_moment",
    }

    def reported_counts(report) -> dict[str, object] | None:
        if not isinstance(report, dict) or "n_candidates" not in report:
            return None
        n = int(report.get("n_candidates", 0))
        operator_n = int(report.get("n_operator_moment_candidates", 0))
        materialized_n = int(report.get("n_materialized_candidates", 0))
        dense_n = int(report.get("n_dense_operator_moment_candidates", 0))
        banded_n = int(report.get("n_banded_operator_moment_candidates", 0))
        structured_n = int(report.get("n_structured_operator_moment_candidates", 0))
        unknown_n = max(0, n - operator_n - materialized_n)
        by_score_route = {
            "dense_operator_moment": dense_n,
            "banded_operator_moment": banded_n,
            "structured_operator_moment": structured_n,
            "materialized": materialized_n,
        }
        by_score_route = {
            route: count for route, count in by_score_route.items() if count > 0
        }
        if unknown_n > 0:
            by_score_route["unknown"] = unknown_n

        def fraction(value: int) -> float:
            return float(value / n) if n else 0.0

        by_head = {
            "ridge": {
                "n_candidates": int(
                    report.get("n_ridge_operator_moment_candidates", 0)
                    + report.get("n_ridge_materialized_candidates", 0)
                ),
                "n_operator_moment_candidates": int(
                    report.get("n_ridge_operator_moment_candidates", 0)
                ),
                "n_materialized_candidates": int(
                    report.get("n_ridge_materialized_candidates", 0)
                ),
            },
            "pls": {
                "n_candidates": int(
                    report.get("n_pls_operator_moment_candidates", 0)
                    + report.get("n_pls_materialized_candidates", 0)
                ),
                "n_operator_moment_candidates": int(
                    report.get("n_pls_operator_moment_candidates", 0)
                ),
                "n_materialized_candidates": int(
                    report.get("n_pls_materialized_candidates", 0)
                ),
            },
        }
        by_head = {
            head: {
                **counts,
                "operator_moment_candidate_fraction": (
                    float(counts["n_operator_moment_candidates"] / counts["n_candidates"])
                    if counts["n_candidates"]
                    else 0.0
                ),
                "materialized_candidate_fraction": (
                    float(counts["n_materialized_candidates"] / counts["n_candidates"])
                    if counts["n_candidates"]
                    else 0.0
                ),
            }
            for head, counts in by_head.items()
            if counts["n_candidates"] > 0
        }
        return {
            "n_candidates": n,
            "n_operator_moment_candidates": operator_n,
            "n_materialized_candidates": materialized_n,
            "n_unknown_route_candidates": unknown_n,
            "n_dense_operator_moment_candidates": dense_n,
            "n_banded_operator_moment_candidates": banded_n,
            "n_structured_operator_moment_candidates": structured_n,
            "operator_moment_candidate_fraction": fraction(operator_n),
            "materialized_candidate_fraction": fraction(materialized_n),
            "unknown_route_candidate_fraction": fraction(unknown_n),
            "dense_operator_moment_candidate_fraction": fraction(dense_n),
            "banded_operator_moment_candidate_fraction": fraction(banded_n),
            "structured_operator_moment_candidate_fraction": fraction(structured_n),
            "all_operator_moment": bool(n > 0 and operator_n == n),
            "has_materialized": bool(materialized_n > 0),
            "has_unknown_route": bool(unknown_n > 0),
            "by_score_route": dict(sorted(by_score_route.items())),
            "by_head": by_head,
        }

    report_total = reported_counts(report_or_rows)
    raw_rows = _candidate_report_rows(report_or_rows)
    rows = [_restore_aom_candidate_record(dict(row)) for row in raw_rows]
    if not rows:
        if report_total is None:
            raise ValueError("candidate report contains no rows")
        return {
            "report_schema": "n4m.aom_candidate_route_summary.v1",
            "row_scope": "none",
            **reported_counts({"n_candidates": 0}),
            "n_chains": 0,
            "n_operator_moment_chains": 0,
            "n_materialized_or_unknown_chains": 0,
            "by_head": {},
            "by_chain": [],
            "materialized_or_unknown_chains": [],
            "reported_total": report_total,
            "rows_match_report_total": False,
        }

    def route_name(row: dict[str, object]) -> str:
        if "score_route" in row:
            return str(row["score_route"])
        if "score_route_id" in row:
            return _aom_score_route_name(int(row["score_route_id"]))
        return "unknown"

    def head_name(row: dict[str, object]) -> str:
        if "head" in row:
            return _normalize_sweep_head_name(row["head"])
        if "head_id" in row:
            return _normalize_sweep_head_name(row["head_id"])
        return "unknown"

    def empty_counts() -> dict[str, object]:
        return {
            "n_candidates": 0,
            "n_operator_moment_candidates": 0,
            "n_materialized_candidates": 0,
            "n_unknown_route_candidates": 0,
            "by_score_route": {},
        }

    def add_count(counts: dict[str, object], route: str) -> None:
        counts["n_candidates"] = int(counts["n_candidates"]) + 1
        by_route = counts["by_score_route"]
        by_route[route] = int(by_route.get(route, 0)) + 1
        if route in operator_routes:
            counts["n_operator_moment_candidates"] = (
                int(counts["n_operator_moment_candidates"]) + 1
            )
        elif route == "materialized":
            counts["n_materialized_candidates"] = (
                int(counts["n_materialized_candidates"]) + 1
            )
        else:
            counts["n_unknown_route_candidates"] = (
                int(counts["n_unknown_route_candidates"]) + 1
            )

    overall = empty_counts()
    by_head: dict[str, dict[str, object]] = {}
    by_chain: dict[str, dict[str, object]] = {}

    for source_index, row in enumerate(rows):
        route = route_name(row)
        head = head_name(row)
        add_count(overall, route)
        by_head.setdefault(head, empty_counts())
        add_count(by_head[head], route)

        chain = _canonicalize_aom_chain(row.get("chain", ()))
        chain_json = json.dumps(chain, separators=(",", ":"))
        chain_id = row.get("chain_id", source_index)
        chain_key = f"{chain_id}:{chain_json}" if chain else str(chain_id)
        entry = by_chain.setdefault(
            chain_key,
            {
                "chain_id": int(chain_id) if isinstance(chain_id, (int, np.integer)) else chain_id,
                "chain": chain,
                "heads": set(),
                "score_routes": set(),
                **empty_counts(),
            },
        )
        entry["heads"].add(head)
        entry["score_routes"].add(route)
        add_count(entry, route)

    def finalize_counts(counts: dict[str, object]) -> dict[str, object]:
        n = int(counts["n_candidates"])
        operator_n = int(counts["n_operator_moment_candidates"])
        materialized_n = int(counts["n_materialized_candidates"])
        unknown_n = int(counts["n_unknown_route_candidates"])
        by_route = {
            str(route): int(value)
            for route, value in sorted(counts["by_score_route"].items())
        }
        return {
            "n_candidates": n,
            "n_operator_moment_candidates": operator_n,
            "n_materialized_candidates": materialized_n,
            "n_unknown_route_candidates": unknown_n,
            "operator_moment_candidate_fraction": (
                float(operator_n / n) if n else 0.0
            ),
            "materialized_candidate_fraction": (
                float(materialized_n / n) if n else 0.0
            ),
            "unknown_route_candidate_fraction": (
                float(unknown_n / n) if n else 0.0
            ),
            "all_operator_moment": bool(n > 0 and operator_n == n),
            "has_materialized": bool(materialized_n > 0),
            "has_unknown_route": bool(unknown_n > 0),
            "by_score_route": by_route,
        }

    chain_rows = []
    for key, entry in by_chain.items():
        counts = finalize_counts(entry)
        chain_rows.append({
            "chain_key": key,
            "chain_id": entry["chain_id"],
            "chain": entry["chain"],
            "heads": sorted(entry["heads"]),
            "score_routes": sorted(entry["score_routes"]),
            **counts,
        })
    chain_rows.sort(key=lambda row: (str(row["chain_id"]), row["chain_key"]))

    materialized_chains = [
        row for row in chain_rows
        if row["has_materialized"] or row["has_unknown_route"]
    ]
    row_scope = "rows"
    if isinstance(report_or_rows, dict):
        if "rows" in report_or_rows:
            row_scope = "rows"
        elif "top_candidates" in report_or_rows:
            row_scope = "top_candidates"
    out = {
        "report_schema": "n4m.aom_candidate_route_summary.v1",
        "row_scope": row_scope,
        **finalize_counts(overall),
        "n_chains": int(len(chain_rows)),
        "n_operator_moment_chains": int(
            sum(1 for row in chain_rows if row["all_operator_moment"])
        ),
        "n_materialized_or_unknown_chains": int(len(materialized_chains)),
        "by_head": {
            head: finalize_counts(counts)
            for head, counts in sorted(by_head.items())
        },
        "by_chain": chain_rows,
        "materialized_or_unknown_chains": materialized_chains,
    }
    if report_total is not None:
        out["reported_total"] = report_total
        out["rows_match_report_total"] = (
            int(out["n_candidates"]) == int(report_total["n_candidates"])
        )
    return out


def aom_candidate_rank_diagnostics(
    report_or_rows,
    *,
    screen_score_key: str | None = None,
    eval_score_key: str | None = None,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, object]:
    """Measure screen-vs-eval rank agreement for scored AOM candidate rows."""
    raw_rows = _candidate_report_rows(report_or_rows)
    rows = [_restore_aom_candidate_record(dict(row)) for row in raw_rows]
    screen_key = _infer_score_key(
        rows,
        screen_score_key,
        ("screen_cv_rmse", "refit_cv_rmse", "cv_rmse"),
    )
    eval_key = _infer_score_key(rows, eval_score_key, ("eval_rmse",))

    parsed = []
    for index, row in enumerate(rows):
        if screen_key not in row or eval_key not in row:
            continue
        try:
            screen_score = float(row[screen_key])
            eval_score = float(row[eval_key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(screen_score) or not np.isfinite(eval_score):
            continue
        identity = row.get(
            "source_index",
            row.get("global_candidate_id", row.get("candidate_id", index)),
        )
        parsed.append({
            "index": int(index),
            "identity": str(identity),
            "screen_score": screen_score,
            "eval_score": eval_score,
            "row": row,
        })
    if not parsed:
        raise ValueError("no candidates contain finite screen and eval scores")

    screen_ranks = _rank_from_scores([item["screen_score"] for item in parsed])
    eval_ranks = _rank_from_scores([item["eval_score"] for item in parsed])
    for item_index, item in enumerate(parsed):
        item["screen_rank"] = int(screen_ranks[item_index])
        item["eval_rank"] = int(eval_ranks[item_index])
        item["rank_delta"] = int(item["eval_rank"] - item["screen_rank"])

    by_screen = sorted(parsed, key=lambda item: item["screen_rank"])
    by_eval = sorted(parsed, key=lambda item: item["eval_rank"])
    n = len(parsed)
    cut_values = sorted({int(k) for k in cutoffs if int(k) > 0})
    overlap_rows = []
    for cutoff in cut_values:
        k = min(cutoff, n)
        screen_ids = {item["identity"] for item in by_screen[:k]}
        eval_ids = {item["identity"] for item in by_eval[:k]}
        overlap = len(screen_ids & eval_ids)
        overlap_rows.append({
            "k": int(cutoff),
            "effective_k": int(k),
            "overlap_count": int(overlap),
            "overlap_fraction": float(overlap / k) if k > 0 else 0.0,
            "eval_top_k_recall": float(overlap / k) if k > 0 else 0.0,
            "screen_top_k_precision": float(overlap / k) if k > 0 else 0.0,
        })

    deltas = np.asarray([item["rank_delta"] for item in parsed], dtype=np.float64)
    abs_deltas = np.abs(deltas)
    best_screen = by_screen[0]
    best_eval = by_eval[0]
    return {
        "screen_score_key": screen_key,
        "eval_score_key": eval_key,
        "n_candidates": int(n),
        "spearman_rank_correlation": _spearman_from_ranks(
            [item["screen_rank"] for item in parsed],
            [item["eval_rank"] for item in parsed],
        ),
        "mean_abs_rank_delta": float(np.mean(abs_deltas)),
        "median_abs_rank_delta": float(np.median(abs_deltas)),
        "max_abs_rank_delta": int(np.max(abs_deltas)),
        "best_screen_eval_rank": int(best_screen["eval_rank"]),
        "best_eval_screen_rank": int(best_eval["screen_rank"]),
        "best_screen": best_screen["row"],
        "best_eval": best_eval["row"],
        "topk": overlap_rows,
    }


def aom_save_candidate_report(
    path,
    report_or_rows,
    *,
    format: str | None = None,
    include_predictions: bool = False,
) -> str:
    """Write AOM candidate rows to JSON, JSONL or CSV without pandas."""
    out_path = Path(path)
    fmt = (format or out_path.suffix.lstrip(".") or "json").lower()
    if fmt == "ndjson":
        fmt = "jsonl"
    if fmt not in {"json", "jsonl", "csv"}:
        raise ValueError("format must be json, jsonl or csv")
    records = aom_candidate_report_records(
        report_or_rows,
        include_predictions=include_predictions,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        metadata = {}
        if isinstance(report_or_rows, dict):
            for key, value in report_or_rows.items():
                if key in {
                    "rows",
                    "top_candidates",
                    "top_candidates_by_head",
                    "top_candidates_by_score_route",
                    "best",
                    "best_by_head",
                    "best_by_score_route",
                    "best_eval",
                    "best_cv",
                    "chunks",
                }:
                    continue
                converted = _jsonable_value(value, include_arrays=False)
                if converted is not None:
                    metadata[str(key)] = converted
        payload = {"metadata": metadata, "rows": records}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif fmt == "jsonl":
        with out_path.open("w", newline="") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    else:
        fieldnames = sorted({key for record in records for key in record.keys()})
        if not fieldnames:
            fieldnames = ["candidate_id"]
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in record.items()
                })
    return str(out_path)


def aom_load_candidate_report(path, *, format: str | None = None) -> list[dict[str, object]]:
    """Read JSON, JSONL or CSV AOM candidate rows as refittable dictionaries."""
    in_path = Path(path)
    fmt = (format or in_path.suffix.lstrip(".") or "json").lower()
    if fmt == "ndjson":
        fmt = "jsonl"
    if fmt not in {"json", "jsonl", "csv"}:
        raise ValueError("format must be json, jsonl or csv")
    if fmt == "json":
        payload = json.loads(in_path.read_text())
        if isinstance(payload, dict):
            rows = payload.get("rows", payload.get("top_candidates"))
            if rows is None:
                raise ValueError("JSON AOM candidate report must contain rows")
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("JSON AOM candidate report must be a dict or list")
    elif fmt == "jsonl":
        rows = [
            json.loads(line)
            for line in in_path.read_text().splitlines()
            if line.strip()
        ]
    else:
        with in_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [_restore_aom_candidate_record(row) for row in rows]


def _create_aom_operator_bank(operators) -> ctypes.c_void_p:
    ops = list(_AOM_SELECTOR_DEFAULT_OPERATORS if operators is None else operators)
    if not ops:
        raise ValueError("operators must contain at least one AOM operator")
    bank = ctypes.c_void_p()
    try:
        check(lib.n4m_operator_bank_create(ctypes.byref(bank)), "n4m_operator_bank_create")
        for op in ops:
            kind_id, param_arr = _aom_op_kind_params(op)
            params_ptr = (
                param_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                if param_arr.size
                else ctypes.POINTER(ctypes.c_double)()
            )
            check(
                lib.n4m_operator_bank_add(
                    bank,
                    ctypes.c_int(kind_id),
                    params_ptr,
                    ctypes.c_int32(param_arr.size),
                ),
                "n4m_operator_bank_add",
            )
    except Exception:
        if bank.value:
            lib.n4m_operator_bank_destroy(bank)
        raise
    return bank


def aom_preprocess(
    X,
    y=None,
    *,
    operators=None,
    gating_mode: str | int = "soft",
) -> dict[str, np.ndarray | float]:
    """Apply the native AOM operator bank and return its MethodResult fields."""
    X_arr = as_f64_2d(X)
    y_arr = None if y is None else _as_y_matrix(y, X_arr.shape[0])
    mode_id = _aom_gating_mode_id(gating_mode)

    ctx = ctypes.c_void_p()
    bank = ctypes.c_void_p()
    gate = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        bank = _create_aom_operator_bank(operators)
        check(
            lib.n4m_gating_strategy_create(
                ctypes.byref(gate),
                ctypes.c_int(mode_id),
            ),
            "n4m_gating_strategy_create",
        )
        Xv = numpy_to_view(X_arr)
        y_ptr = ctypes.POINTER(type(Xv))()
        if y_arr is not None:
            Yv = numpy_to_view(y_arr)
            y_ptr = ctypes.byref(Yv)
        check(
            lib.n4m_aom_preprocess_fit(
                ctx,
                bank,
                gate,
                ctypes.byref(Xv),
                y_ptr,
                ctypes.byref(result),
            ),
            "n4m_aom_preprocess_fit",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_PREPROCESS_MATRICES,
            int64_vectors=("operator_kinds",),
            scalars=_AOM_PREPROCESS_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if gate.value:
            lib.n4m_gating_strategy_destroy(gate)
        if bank.value:
            lib.n4m_operator_bank_destroy(bank)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def _selector_fold_ids(n_samples: int, cv: int, fold_ids) -> np.ndarray:
    if fold_ids is None:
        if cv < 2 or cv > n_samples:
            raise ValueError("cv must be in [2, n_samples] when fold_ids are not provided")
        return np.ascontiguousarray(
            [(i * int(cv)) // int(n_samples) for i in range(n_samples)],
            dtype=np.int32,
        )
    folds = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
    if folds.size != n_samples:
        raise ValueError("fold_ids length must match X.shape[0]")
    if np.any(folds < 0):
        raise ValueError("fold_ids must be non-negative")
    inferred_cv = int(folds.max()) + 1 if folds.size else 0
    actual_cv = int(cv) if cv and int(cv) > 0 else inferred_cv
    if actual_cv < 2 or actual_cv > n_samples:
        raise ValueError("inferred/provided cv must be in [2, n_samples]")
    if np.any(folds >= actual_cv):
        raise ValueError("fold_ids contain id >= cv")
    return folds


def _create_validation_plan_from_fold_ids(fold_ids: np.ndarray, cv: int) -> ctypes.c_void_p:
    n_samples = int(fold_ids.size)
    plan = ctypes.c_void_p()
    try:
        check(lib.n4m_validation_plan_create(ctypes.byref(plan)), "n4m_validation_plan_create")
        check(
            lib.n4m_validation_plan_set_n_samples(plan, ctypes.c_int64(n_samples)),
            "n4m_validation_plan_set_n_samples",
        )
        all_idx = np.arange(n_samples, dtype=np.int64)
        for fold in range(int(cv)):
            test = np.ascontiguousarray(all_idx[fold_ids == fold], dtype=np.int64)
            train = np.ascontiguousarray(all_idx[fold_ids != fold], dtype=np.int64)
            if test.size == 0 or train.size == 0:
                raise ValueError("every fold must contain test rows and leave train rows")
            check(
                lib.n4m_validation_plan_add_fold(
                    plan,
                    train.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(train.size),
                    test.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(test.size),
                ),
                "n4m_validation_plan_add_fold",
            )
    except Exception:
        if plan.value:
            lib.n4m_validation_plan_destroy(plan)
        raise
    return plan


def _copy_double_vector_from_getter(result: ctypes.c_void_p, getter_name: str) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_double)()
    size = ctypes.c_int32()
    check(
        getattr(lib, getter_name)(result, ctypes.byref(data), ctypes.byref(size)),
        getter_name,
    )
    n = int(size.value)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    return np.ctypeslib.as_array(data, shape=(n,)).copy()


def _copy_i32_vector_from_getter(result: ctypes.c_void_p, getter_name: str) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_int32)()
    size = ctypes.c_int32()
    check(
        getattr(lib, getter_name)(result, ctypes.byref(data), ctypes.byref(size)),
        getter_name,
    )
    n = int(size.value)
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    return np.ctypeslib.as_array(data, shape=(n,)).copy()


def _copy_double_matrix_from_i32_getter(
    result: ctypes.c_void_p,
    getter_name: str,
) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_double)()
    rows = ctypes.c_int32()
    cols = ctypes.c_int32()
    check(
        getattr(lib, getter_name)(
            result,
            ctypes.byref(data),
            ctypes.byref(rows),
            ctypes.byref(cols),
        ),
        getter_name,
    )
    shape = (int(rows.value), int(cols.value))
    if shape[0] * shape[1] == 0:
        return np.empty(shape, dtype=np.float64)
    return np.ctypeslib.as_array(data, shape=shape).copy()


def _copy_double_matrix_from_i64_getter(
    result: ctypes.c_void_p,
    getter_name: str,
) -> np.ndarray:
    data = ctypes.POINTER(ctypes.c_double)()
    rows = ctypes.c_int64()
    cols = ctypes.c_int64()
    check(
        getattr(lib, getter_name)(
            result,
            ctypes.byref(data),
            ctypes.byref(rows),
            ctypes.byref(cols),
        ),
        getter_name,
    )
    shape = (int(rows.value), int(cols.value))
    if shape[0] * shape[1] == 0:
        return np.empty(shape, dtype=np.float64)
    return np.ctypeslib.as_array(data, shape=shape).copy()


def _aom_global_result_dict(result: ctypes.c_void_p) -> dict[str, np.ndarray | float]:
    scalar_i32 = ctypes.c_int32()
    scalar_f64 = ctypes.c_double()
    out: dict[str, np.ndarray | float] = {}
    for key, getter in (
        ("n_operators", lib.n4m_aom_global_result_get_n_operators),
        ("max_components", lib.n4m_aom_global_result_get_max_components),
        ("selected_operator_index", lib.n4m_aom_global_result_get_selected_operator_index),
        ("selected_n_components", lib.n4m_aom_global_result_get_selected_n_components),
    ):
        check(getter(result, ctypes.byref(scalar_i32)), key)
        out[key] = float(scalar_i32.value)
    check(
        lib.n4m_aom_global_result_get_best_score(result, ctypes.byref(scalar_f64)),
        "n4m_aom_global_result_get_best_score",
    )
    out["best_score"] = float(scalar_f64.value)
    out["operator_kinds"] = _copy_i32_vector_from_getter(
        result,
        "n4m_aom_global_result_get_operator_kinds",
    )
    out["operator_scores"] = _copy_double_vector_from_getter(
        result,
        "n4m_aom_global_result_get_operator_scores",
    )
    out["rmse_curves"] = _copy_double_matrix_from_i32_getter(
        result,
        "n4m_aom_global_result_get_rmse_curves",
    )
    out["predictions"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_global_result_get_predictions",
    )
    out["coefficients"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_global_result_get_coefficients",
    )
    out["input_coefficients"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_global_result_get_input_coefficients",
    )
    out["intercept"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_global_result_get_intercept",
    )
    return out


def _aom_per_component_result_dict(result: ctypes.c_void_p) -> dict[str, np.ndarray | float]:
    scalar_i32 = ctypes.c_int32()
    scalar_f64 = ctypes.c_double()
    out: dict[str, np.ndarray | float] = {}
    for key, getter in (
        ("n_operators", lib.n4m_aom_per_component_result_get_n_operators),
        ("max_components", lib.n4m_aom_per_component_result_get_max_components),
        ("selected_n_components", lib.n4m_aom_per_component_result_get_selected_n_components),
    ):
        check(getter(result, ctypes.byref(scalar_i32)), key)
        out[key] = float(scalar_i32.value)
    check(
        lib.n4m_aom_per_component_result_get_best_score(result, ctypes.byref(scalar_f64)),
        "n4m_aom_per_component_result_get_best_score",
    )
    out["best_score"] = float(scalar_f64.value)
    out["operator_kinds"] = _copy_i32_vector_from_getter(
        result,
        "n4m_aom_per_component_result_get_operator_kinds",
    )
    out["selected_operator_indices"] = _copy_i32_vector_from_getter(
        result,
        "n4m_aom_per_component_result_get_selected_operator_indices",
    )
    out["component_scores"] = _copy_double_matrix_from_i32_getter(
        result,
        "n4m_aom_per_component_result_get_component_scores",
    )
    out["prefix_scores"] = _copy_double_vector_from_getter(
        result,
        "n4m_aom_per_component_result_get_prefix_scores",
    )
    out["predictions"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_per_component_result_get_predictions",
    )
    out["coefficients"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_per_component_result_get_coefficients",
    )
    out["input_coefficients"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_per_component_result_get_input_coefficients",
    )
    out["intercept"] = _copy_double_matrix_from_i64_getter(
        result,
        "n4m_aom_per_component_result_get_intercept",
    )
    return out


def _i32_ptr(arr: np.ndarray):
    return (
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        if arr.size
        else ctypes.POINTER(ctypes.c_int32)()
    )


def _run_aom_selector(
    X,
    y,
    *,
    symbol: str,
    destroy_symbol: str,
    result_reader,
    max_components: int = 3,
    operators=None,
    cv: int = 3,
    fold_ids=None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    max_components = int(max_components)
    if max_components < 1:
        raise ValueError("max_components must be >= 1")
    fold_arr = _selector_fold_ids(X_arr.shape[0], int(cv), fold_ids)
    actual_cv = int(cv) if int(cv) > 0 else int(fold_arr.max()) + 1

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    bank = ctypes.c_void_p()
    plan = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        check(lib.n4m_config_set_algorithm(cfg, ctypes.c_int(0)), "n4m_config_set_algorithm")
        check(lib.n4m_config_set_solver(cfg, ctypes.c_int(1)), "n4m_config_set_solver")
        check(lib.n4m_config_set_deflation(cfg, ctypes.c_int(0)), "n4m_config_set_deflation")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        bank = _create_aom_operator_bank(operators)
        plan = _create_validation_plan_from_fold_ids(fold_arr, actual_cv)
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        check(
            getattr(lib, symbol)(
                ctx,
                cfg,
                bank,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                plan,
                ctypes.c_int32(max_components),
                ctypes.byref(result),
            ),
            symbol,
        )
        out = result_reader(result)
        out["fold_ids"] = fold_arr.copy()
        out["cv"] = float(actual_cv)
        out["n_samples"] = float(X_arr.shape[0])
        out["n_features"] = float(X_arr.shape[1])
        out["n_targets"] = float(y_arr.shape[1])
        return out
    finally:
        if result.value:
            getattr(lib, destroy_symbol)(result)
        if plan.value:
            lib.n4m_validation_plan_destroy(plan)
        if bank.value:
            lib.n4m_operator_bank_destroy(bank)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_global_select(
    X,
    y,
    *,
    max_components: int = 3,
    operators=None,
    cv: int = 3,
    fold_ids=None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the native global AOM-PLS operator selector."""
    return _run_aom_selector(
        X,
        y,
        symbol="n4m_aom_global_select",
        destroy_symbol="n4m_aom_global_result_destroy",
        result_reader=_aom_global_result_dict,
        max_components=max_components,
        operators=operators,
        cv=cv,
        fold_ids=fold_ids,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
    )


def aom_per_component_select(
    X,
    y,
    *,
    max_components: int = 3,
    operators=None,
    cv: int = 3,
    fold_ids=None,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the native POP-PLS per-component AOM operator selector."""
    return _run_aom_selector(
        X,
        y,
        symbol="n4m_aom_per_component_select",
        destroy_symbol="n4m_aom_per_component_result_destroy",
        result_reader=_aom_per_component_result_dict,
        max_components=max_components,
        operators=operators,
        cv=cv,
        fold_ids=fold_ids,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
    )


aom_pls = aom_global_select
pop_pls = aom_per_component_select


def sweep_run(
    X,
    y,
    *,
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(0.01, 0.1, 1.0, 10.0),
    pls_components=None,
    heads=("ridge",),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    score_only: bool = False,
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run native moment sweep screening.

    ABI v1 supports Ridge lambdas and PLS component candidates. The result
    contains candidate scores, selected-candidate OOF predictions and final-refit
    predictions unless score_only=True, which returns ranking outputs only.
    """
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    head_mask = _sweep_heads_mask(heads)
    lambdas = np.ascontiguousarray(ridge_lambdas, dtype=np.float64).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["ridge"] and lambdas.size == 0:
        raise ValueError("ridge_lambdas must not be empty")

    fold_arr = None
    if fold_ids is not None:
        fold_arr = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids length must match X.shape[0]")

    pls_arr = None
    if pls_components is not None:
        pls_arr = np.ascontiguousarray(pls_components, dtype=np.int32).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["pls"] and (pls_arr is None or pls_arr.size == 0):
        raise ValueError("pls_components must not be empty when the pls head is enabled")

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        _set_config_bool(cfg, "aom_score_only", score_only)
        _set_config_bool(
            cfg, "cuda_pls_parallel_folds", cuda_pls_parallel_folds
        )
        _set_config_bool(cfg, "cuda_pls_many_batched", cuda_pls_many_batched)
        _set_config_positive_int(
            cfg, "cuda_pls_min_device_features", cuda_pls_min_device_features
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        fold_ptr = (
            fold_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if fold_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        fold_n = ctypes.c_int64(0 if fold_arr is None else fold_arr.size)
        pls_ptr = (
            pls_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if pls_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        pls_n = ctypes.c_int64(0 if pls_arr is None else pls_arr.size)
        lambda_ptr = (
            lambdas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if lambdas.size
            else ctypes.POINTER(ctypes.c_double)()
        )
        check(
            lib.n4m_sweep_run(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(cv)),
                fold_ptr,
                fold_n,
                lambda_ptr,
                ctypes.c_int64(lambdas.size),
                pls_ptr,
                pls_n,
                ctypes.c_int32(head_mask),
                ctypes.byref(result),
            ),
            "n4m_sweep_run",
        )
        return _method_result_dict(
            result,
            matrices=_SWEEP_MATRICES,
            int_vectors=("fold_ids",),
            scalars=_SWEEP_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_sweep_run(
    X,
    y,
    *,
    profile: str | int = "compact",
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(0.01, 0.1, 1.0, 10.0),
    pls_components=None,
    heads=("ridge",),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    pls_score_mode: str | int = "cv",
    score_only: bool = False,
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run native AOM preprocessing sweep over strict-linear chain banks."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    head_mask = _sweep_heads_mask(heads)
    lambdas = np.ascontiguousarray(ridge_lambdas, dtype=np.float64).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["ridge"] and lambdas.size == 0:
        raise ValueError("ridge_lambdas must not be empty")
    try:
        profile_id = _AOM_ROBUST_HPO_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown AOM sweep profile: {profile!r}") from exc

    fold_arr = None
    if fold_ids is not None:
        fold_arr = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids length must match X.shape[0]")

    pls_arr = None
    if pls_components is not None:
        pls_arr = np.ascontiguousarray(pls_components, dtype=np.int32).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["pls"] and (pls_arr is None or pls_arr.size == 0):
        raise ValueError("pls_components must not be empty when the pls head is enabled")

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        _set_aom_moment_policy(cfg, moment_policy)
        _set_aom_pls_score_mode(cfg, pls_score_mode)
        _set_config_bool(cfg, "aom_score_only", score_only)
        _set_config_bool(
            cfg, "cuda_pls_parallel_folds", cuda_pls_parallel_folds
        )
        _set_config_bool(cfg, "cuda_pls_many_batched", cuda_pls_many_batched)
        _set_config_positive_int(
            cfg, "cuda_pls_min_device_features", cuda_pls_min_device_features
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        fold_ptr = (
            fold_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if fold_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        fold_n = ctypes.c_int64(0 if fold_arr is None else fold_arr.size)
        pls_ptr = (
            pls_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if pls_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        pls_n = ctypes.c_int64(0 if pls_arr is None else pls_arr.size)
        lambda_ptr = (
            lambdas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if lambdas.size
            else ctypes.POINTER(ctypes.c_double)()
        )
        check(
            lib.n4m_aom_sweep_run(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(profile_id)),
                ctypes.c_int32(int(cv)),
                fold_ptr,
                fold_n,
                lambda_ptr,
                ctypes.c_int64(lambdas.size),
                pls_ptr,
                pls_n,
                ctypes.c_int32(head_mask),
                ctypes.byref(result),
            ),
            "n4m_aom_sweep_run",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_SWEEP_MATRICES,
            int_vectors=(
                "fold_ids",
                "candidate_routes",
                "chain_offsets",
                "op_kinds",
                "param_offsets",
            ),
            scalars=_AOM_SWEEP_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_chain_sweep_run(
    X,
    y,
    chains,
    *,
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(0.01, 0.1, 1.0, 10.0),
    pls_components=None,
    heads=("ridge",),
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    pls_score_mode: str | int = "cv",
    score_only: bool = False,
    cuda_pls_parallel_folds: bool | None = None,
    cuda_pls_min_device_features: int | None = None,
    cuda_pls_many_batched: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run native AOM sweep over caller-provided strict-linear chains."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    head_mask = _sweep_heads_mask(heads)
    chain_offsets, op_kinds, param_offsets, params = _flatten_aom_chains(chains)
    lambdas = np.ascontiguousarray(ridge_lambdas, dtype=np.float64).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["ridge"] and lambdas.size == 0:
        raise ValueError("ridge_lambdas must not be empty")

    fold_arr = None
    if fold_ids is not None:
        fold_arr = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids length must match X.shape[0]")

    pls_arr = None
    if pls_components is not None:
        pls_arr = np.ascontiguousarray(pls_components, dtype=np.int32).reshape(-1)
    if head_mask & _SWEEP_HEAD_BITS["pls"] and (pls_arr is None or pls_arr.size == 0):
        raise ValueError("pls_components must not be empty when the pls head is enabled")

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        _set_aom_moment_policy(cfg, moment_policy)
        _set_aom_pls_score_mode(cfg, pls_score_mode)
        _set_config_bool(cfg, "aom_score_only", score_only)
        _set_config_bool(
            cfg, "cuda_pls_parallel_folds", cuda_pls_parallel_folds
        )
        _set_config_bool(cfg, "cuda_pls_many_batched", cuda_pls_many_batched)
        _set_config_positive_int(
            cfg, "cuda_pls_min_device_features", cuda_pls_min_device_features
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        fold_ptr = (
            fold_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if fold_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        fold_n = ctypes.c_int64(0 if fold_arr is None else fold_arr.size)
        pls_ptr = (
            pls_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if pls_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        pls_n = ctypes.c_int64(0 if pls_arr is None else pls_arr.size)
        lambda_ptr = (
            lambdas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if lambdas.size
            else ctypes.POINTER(ctypes.c_double)()
        )
        param_ptr = (
            params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if params.size
            else ctypes.POINTER(ctypes.c_double)()
        )
        check(
            lib.n4m_aom_chain_sweep_run(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(cv)),
                fold_ptr,
                fold_n,
                _i32_ptr(chain_offsets),
                ctypes.c_int64(chain_offsets.size),
                _i32_ptr(op_kinds),
                ctypes.c_int64(op_kinds.size),
                _i32_ptr(param_offsets),
                ctypes.c_int64(param_offsets.size),
                param_ptr,
                ctypes.c_int64(params.size),
                lambda_ptr,
                ctypes.c_int64(lambdas.size),
                pls_ptr,
                pls_n,
                ctypes.c_int32(head_mask),
                ctypes.byref(result),
            ),
            "n4m_aom_chain_sweep_run",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_SWEEP_MATRICES,
            int_vectors=(
                "fold_ids",
                "candidate_routes",
                "chain_offsets",
                "op_kinds",
                "param_offsets",
            ),
            scalars=_AOM_SWEEP_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_chain_fixed_fit_run(
    X,
    y,
    chain,
    *,
    head: str | int = "ridge",
    param: float = 0.1,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
    moment_policy: str | int = "auto",
    cuda_pls_min_device_features: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit one already-selected AOM chain/head/param without running CV.

    This is the final model-building endpoint for candidates whose exact CV
    score was already obtained elsewhere, for example via
    ``aom_refit_candidates``. Returned ``selected_cv_rmse`` is NaN unless a
    higher-level wrapper replaces it with a verified score.
    """
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    head_name = str(head).lower()
    if head_name in {"0", "ridge"}:
        head_id = 0
    elif head_name in {"1", "pls"}:
        head_id = 1
    else:
        raise ValueError("head must be 'ridge', 'pls', 0, or 1")
    chain_offsets, op_kinds, param_offsets, params = _flatten_aom_chains([chain])

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        _set_aom_moment_policy(cfg, moment_policy)
        _set_config_positive_int(
            cfg, "cuda_pls_min_device_features", cuda_pls_min_device_features
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        param_ptr = (
            params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if params.size
            else ctypes.POINTER(ctypes.c_double)()
        )
        check(
            lib.n4m_aom_chain_fixed_fit_run(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                _i32_ptr(chain_offsets),
                ctypes.c_int64(chain_offsets.size),
                _i32_ptr(op_kinds),
                ctypes.c_int64(op_kinds.size),
                _i32_ptr(param_offsets),
                ctypes.c_int64(param_offsets.size),
                param_ptr,
                ctypes.c_int64(params.size),
                ctypes.c_int32(head_id),
                ctypes.c_double(float(param)),
                ctypes.byref(result),
            ),
            "n4m_aom_chain_fixed_fit_run",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_SWEEP_MATRICES,
            int_vectors=(
                "fold_ids",
                "candidate_routes",
                "chain_offsets",
                "op_kinds",
                "param_offsets",
            ),
            scalars=_AOM_SWEEP_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_ridge_blender(
    X,
    y,
    *,
    profile: str | int = "compact",
    cv: int = 5,
    fold_ids=None,
    ridge_lambdas=(1e-4, 1e-2, 1.0, 100.0),
    regularizer: float = 0.01,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run native strict-linear AOM Ridge OOF simplex blending."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    try:
        profile_id = _AOM_ROBUST_HPO_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown AOM Ridge blender profile: {profile!r}") from exc
    lambdas = np.ascontiguousarray(ridge_lambdas, dtype=np.float64).reshape(-1)
    if lambdas.size == 0:
        raise ValueError("ridge_lambdas must not be empty")
    if np.any(~np.isfinite(lambdas)) or np.any(lambdas <= 0.0):
        raise ValueError("ridge_lambdas must be finite and strictly positive")
    regularizer_value = float(regularizer)
    if not np.isfinite(regularizer_value) or regularizer_value < 0.0:
        raise ValueError("regularizer must be finite and non-negative")

    fold_arr = None
    if fold_ids is not None:
        fold_arr = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids length must match X.shape[0]")

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        fold_ptr = (
            fold_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if fold_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        fold_n = ctypes.c_int64(0 if fold_arr is None else fold_arr.size)
        check(
            lib.n4m_aom_ridge_blender_fit(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(profile_id)),
                ctypes.c_int32(int(cv)),
                fold_ptr,
                fold_n,
                lambdas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(lambdas.size),
                ctypes.c_double(regularizer_value),
                ctypes.byref(result),
            ),
            "n4m_aom_ridge_blender_fit",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_RIDGE_BLENDER_MATRICES,
            int_vectors=("fold_ids",),
            scalars=_AOM_RIDGE_BLENDER_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def aom_operator_pls_stack(
    X,
    y,
    *,
    profile: str | int = "compact",
    cv: int = 5,
    fold_ids=None,
    components=(2, 4, 8),
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    std_penalty: float = 0.0,
    gap_penalty: float = 0.0,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
    scale_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Run native strict-linear AOM operator PLS score stack with Ridge head."""
    X_arr = as_f64_2d(X)
    y_arr = _as_y_matrix(y, X_arr.shape[0])
    if y_arr.shape[1] != 1:
        raise ValueError("aom_operator_pls_stack currently supports one Y target")
    try:
        profile_id = _AOM_ROBUST_HPO_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown AOM operator PLS stack profile: {profile!r}") from exc

    component_arr = np.ascontiguousarray(components, dtype=np.int32).reshape(-1)
    if component_arr.size == 0 or np.any(component_arr < 1):
        raise ValueError("components must contain positive integers")
    alpha_arr = np.ascontiguousarray(alphas, dtype=np.float64).reshape(-1)
    if alpha_arr.size == 0 or np.any(~np.isfinite(alpha_arr)) or np.any(alpha_arr < 0.0):
        raise ValueError("alphas must be finite and non-negative")
    std_penalty_value = float(std_penalty)
    gap_penalty_value = float(gap_penalty)
    if (
        not np.isfinite(std_penalty_value)
        or std_penalty_value < 0.0
        or not np.isfinite(gap_penalty_value)
        or gap_penalty_value < 0.0
    ):
        raise ValueError("std_penalty and gap_penalty must be finite and non-negative")

    fold_arr = None
    if fold_ids is not None:
        fold_arr = np.ascontiguousarray(fold_ids, dtype=np.int32).reshape(-1)
        if fold_arr.size != X_arr.shape[0]:
            raise ValueError("fold_ids length must match X.shape[0]")

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        _set_model_config(
            cfg,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
        )
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        fold_ptr = (
            fold_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if fold_arr is not None
            else ctypes.POINTER(ctypes.c_int32)()
        )
        fold_n = ctypes.c_int64(0 if fold_arr is None else fold_arr.size)
        check(
            lib.n4m_aom_operator_pls_stack_fit(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(profile_id)),
                ctypes.c_int32(int(cv)),
                fold_ptr,
                fold_n,
                component_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int64(component_arr.size),
                alpha_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(alpha_arr.size),
                ctypes.c_double(std_penalty_value),
                ctypes.c_double(gap_penalty_value),
                ctypes.byref(result),
            ),
            "n4m_aom_operator_pls_stack_fit",
        )
        return _method_result_dict(
            result,
            matrices=_AOM_OPERATOR_PLS_STACK_MATRICES,
            int_vectors=("fold_ids", "operator_feature_offsets"),
            scalars=_AOM_OPERATOR_PLS_STACK_SCALARS,
        )
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def ridge(
    X,
    y,
    *,
    alpha: float = 1.0,
    center_x: bool | None = None,
    scale_x: bool | None = None,
    center_y: bool | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit native closed-form Ridge regression and return its MethodResult."""
    lambda_arr = np.ascontiguousarray([float(alpha)], dtype=np.float64)
    return _fit_method_result(
        "n4m_ridge_fit",
        X,
        y,
        _f64_ptr(lambda_arr),
        ctypes.c_int64(1),
        matrices=("coefficients", "intercept", "x_mean", "x_scale", "y_mean", "predictions"),
        scalars=("rmse", "lambda"),
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
    )


def pcr(
    X,
    y,
    *,
    n_components: int = 2,
    center_x: bool | None = True,
    scale_x: bool | None = True,
    center_y: bool | None = True,
    scale_y: bool | None = False,
) -> dict[str, np.ndarray | float]:
    """Fit native Principal Components Regression and return its MethodResult."""
    return _fit_method_result(
        "n4m_pcr_fit",
        X,
        y,
        matrices=(
            "coefficients",
            "predictions",
            "x_mean",
            "x_scale",
            "y_mean",
            "y_scale",
            "weights_w",
            "loadings_p",
            "rotations_r",
        ),
        scalars=("rmse", "n_components"),
        n_components=n_components,
        center_x=center_x,
        scale_x=scale_x,
        center_y=center_y,
        scale_y=scale_y,
    )


def cppls(
    X,
    y,
    *,
    gamma: float = 0.5,
    n_components: int = 2,
) -> dict[str, np.ndarray | float]:
    """Fit native CPPLS and return its MethodResult."""
    return _fit_method_result(
        "n4m_cppls_fit",
        X,
        y,
        ctypes.c_double(float(gamma)),
        matrices=("coefficients", "predictions", "x_mean", "y_mean"),
        scalars=("rmse", "gamma"),
        n_components=n_components,
    )


def continuum_regression(
    X,
    y,
    *,
    tau: float = 0.5,
    n_components: int = 2,
) -> dict[str, np.ndarray | float]:
    """Fit native continuum regression and return its MethodResult."""
    return _fit_method_result(
        "n4m_continuum_regression_fit",
        X,
        y,
        ctypes.c_double(float(tau)),
        matrices=("coefficients", "predictions", "x_mean", "y_mean"),
        scalars=("rmse", "tau"),
        n_components=n_components,
    )


def ecr(
    X,
    y,
    *,
    alpha: float = 0.5,
    n_components: int = 2,
) -> dict[str, np.ndarray | float]:
    """Fit native Elastic Component Regression and return its MethodResult."""
    return _fit_method_result(
        "n4m_ecr_fit",
        X,
        y,
        ctypes.c_double(float(alpha)),
        matrices=(
            "coefficients",
            "predictions",
            "x_mean",
            "y_mean",
            "x_scale",
            "y_scale",
            "weights_w",
            "loadings_p",
            "y_loadings",
            "wstar",
            "r2x",
            "r2y",
        ),
        scalars=("n_samples", "n_features", "n_targets", "n_components", "alpha", "rmse"),
        n_components=n_components,
    )


def moment_stack(X, y, **kwargs):
    """Fit the sklearn-style native moment-model OOF stack."""
    from .sklearn.native_sweeps import NativeMomentStackRegressor

    return NativeMomentStackRegressor(**kwargs).fit(X, y)


def _aom_heads_mask(heads) -> int:
    if isinstance(heads, str):
        heads = (heads,)
    mask = 0
    for head in heads:
        try:
            mask |= _AOM_ROBUST_HPO_HEAD_BITS[head]
        except KeyError as exc:
            raise ValueError(f"unknown AOM robust-HPO head: {head!r}") from exc
    if mask == 0:
        raise ValueError("heads must include ridge and/or pls")
    return mask


def _wavelength_ptr(values, n_features: int, *, required: bool = False):
    if values is None:
        if not required:
            return _null_f64_ptr(), ctypes.c_int64(0), None
        values = np.arange(n_features, dtype=np.float64)
    arr = _as_f64_1d(values, "wavelengths")
    if arr.size != n_features:
        raise ValueError("wavelengths length must match X.shape[1]")
    return _f64_ptr(arr), ctypes.c_int64(arr.size), arr


def _aug_apply(
    prefix: str,
    X,
    *create_args,
    seed: int = 0,
    wavelengths=None,
    require_wavelengths: bool = False,
    apply_wavelengths: bool = False,
):
    X_arr = as_f64_2d(X)
    rng = PCG64(seed)
    handle = ctypes.c_void_p()
    try:
        check(
            getattr(lib, f"{prefix}_create")(
                ctypes.byref(handle), rng.handle, *create_args
            ),
            f"{prefix}_create",
        )
        out = empty_like_f64(X_arr.shape)
        if apply_wavelengths:
            _, _, wl_arr = _wavelength_ptr(
                wavelengths, X_arr.shape[1], required=True
            )
            check(
                getattr(lib, f"{prefix}_apply")(
                    handle,
                    numpy_to_view(X_arr),
                    numpy_to_view(wl_arr.reshape(1, -1)),
                    numpy_to_view(out),
                ),
                f"{prefix}_apply",
            )
        else:
            check(
                getattr(lib, f"{prefix}_apply")(
                    handle, numpy_to_view(X_arr), numpy_to_view(out)
                ),
                f"{prefix}_apply",
            )
        return out
    finally:
        if handle.value:
            getattr(lib, f"{prefix}_destroy")(handle)
        rng.close()


def transform(
    prefix: str,
    X,
    *create_args,
    fit_X=None,
    out_shape: tuple[int, int] | None = None,
    out_dtype: str = "f64",
):
    """Run a shape-preserving ABI ``*_transform`` family."""
    X_arr = as_f64_2d(X)
    fit_arr = as_f64_2d(fit_X) if fit_X is not None else None
    handle = _create(prefix, *create_args)
    try:
        if fit_arr is not None:
            check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(fit_arr)), f"{prefix}_fit")
        shape = out_shape or X_arr.shape
        out = empty_like_i32(shape) if out_dtype == "i32" else empty_like_f64(shape)
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def aom_robust_hpo(
    X,
    y,
    *,
    profile: str | int = "compact",
    cv: int = 5,
    heads=("ridge", "pls"),
) -> dict[str, np.ndarray | float]:
    """Run native strict-linear AOM robust-HPO screening.

    Returns a dictionary containing ``predictions``,
    ``coefficients_transformed``, ``input_coefficients``, ``intercept``,
    ``candidate_scores`` and the scalar selection diagnostics exposed by
    ``n4m_aom_robust_hpo_fit``.
    """
    X_arr = as_f64_2d(X)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    elif y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = np.ascontiguousarray(y_arr)
    else:
        raise ValueError("y must be 1-D or a single-column 2-D array")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y have incompatible lengths")
    if not y_arr.flags.c_contiguous:
        y_arr = np.ascontiguousarray(y_arr)
    try:
        profile_id = _AOM_ROBUST_HPO_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown AOM robust-HPO profile: {profile!r}") from exc
    heads_mask = _aom_heads_mask(heads)

    ctx = ctypes.c_void_p()
    cfg = ctypes.c_void_p()
    result = ctypes.c_void_p()
    try:
        check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
        check(lib.n4m_config_create(ctypes.byref(cfg)), "n4m_config_create")
        Xv = numpy_to_view(X_arr)
        Yv = numpy_to_view(y_arr)
        check(
            lib.n4m_aom_robust_hpo_fit(
                ctx,
                cfg,
                ctypes.byref(Xv),
                ctypes.byref(Yv),
                ctypes.c_int32(int(profile_id)),
                ctypes.c_int32(int(cv)),
                ctypes.c_int32(int(heads_mask)),
                ctypes.byref(result),
            ),
            "n4m_aom_robust_hpo_fit",
        )
        out: dict[str, np.ndarray | float] = {
            "predictions": _method_result_double_matrix(result, "predictions"),
            "coefficients_transformed": _method_result_double_matrix(
                result, "coefficients_transformed"
            ),
            "input_coefficients": _method_result_double_matrix(
                result, "input_coefficients"
            ),
            "intercept": _method_result_double_matrix(result, "intercept"),
            "candidate_scores": _method_result_double_matrix(
                result, "candidate_scores"
            ),
        }
        for name in (
            "selected_chain_id",
            "selected_head_id",
            "selected_param",
            "selected_cv_rmse",
            "n_chains",
            "n_candidates",
            "profile",
            "cv",
            "n_samples",
            "n_features",
            "n_features_transformed",
            "n_targets",
        ):
            out[name] = _method_result_scalar(result, name)
        return out
    finally:
        if result.value:
            lib.n4m_method_result_destroy(result)
        if cfg.value:
            lib.n4m_config_destroy(cfg)
        if ctx.value:
            lib.n4m_context_destroy(ctx)


def _fit_y_transform(prefix: str, X, y, *create_args):
    X_arr = as_f64_2d(X)
    y_arr = _as_f64_1d(y, "y")
    handle = _create(prefix, *create_args)
    try:
        check(
            getattr(lib, f"{prefix}_fit")(
                handle, numpy_to_view(X_arr), _f64_ptr(y_arr), ctypes.c_int64(y_arr.size)
            ),
            f"{prefix}_fit",
        )
        out = empty_like_f64(X_arr.shape)
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def _paired_transfer(prefix: str, X_source, X_target, X, *create_args):
    source = as_f64_2d(X_source)
    target = as_f64_2d(X_target)
    X_arr = as_f64_2d(X)
    if source.shape != target.shape:
        raise ValueError("X_source and X_target must have the same shape")
    handle = _create(prefix, *create_args)
    try:
        check(
            getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(source), numpy_to_view(target)),
            f"{prefix}_fit",
        )
        out = empty_like_f64(X_arr.shape)
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def _split_result_to_arrays(result: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    train = np.ctypeslib.as_array(result.train_idx, shape=(int(result.n_train),)).copy()
    test = np.ctypeslib.as_array(result.test_idx, shape=(int(result.n_test),)).copy()
    lib.n4m_split_result_destroy(ctypes.byref(result))
    return train, test


def snv(X, with_mean: bool = True, with_std: bool = True, ddof: int = 0):
    return transform("n4m_pp_snv", X, ctypes.c_int(with_mean), ctypes.c_int(with_std), ctypes.c_int(ddof))


def lsnv(X, window: int = 11, pad_mode: str | int = "reflect", constant_value: float = 0.0):
    return transform("n4m_pp_lsnv", X, ctypes.c_int32(window), ctypes.c_int32(_enum(_LSNV_PAD_MODES, pad_mode, "pad_mode")), ctypes.c_double(constant_value))


def rnv(X, with_center: bool = True, with_scale: bool = True, k: float = 1.4826):
    return transform("n4m_pp_rnv", X, ctypes.c_int(with_center), ctypes.c_int(with_scale), ctypes.c_double(k))


def area_norm(X, method: str | int = "sum"):
    return transform("n4m_pp_area", X, ctypes.c_int32(_enum(_AREA_METHODS, method, "method")))


area_normalization = area_norm


def normalize(X, feature_min: float = -1.0, feature_max: float = 1.0):
    return transform("n4m_pp_normalize", X, ctypes.c_double(feature_min), ctypes.c_double(feature_max))


def simple_scale(X):
    return transform("n4m_pp_simple_scale", X)


def log_transform(
    X,
    base: float = 0.0,
    offset: float = 0.0,
    auto_offset: bool = True,
    min_value: float = 1e-8,
):
    return transform(
        "n4m_pp_log",
        X,
        ctypes.c_double(base),
        ctypes.c_double(offset),
        ctypes.c_int(1 if auto_offset else 0),
        ctypes.c_double(min_value),
        fit_X=X,
    )


def msc(X):
    return transform("n4m_pp_msc", X, fit_X=X)


def emsc(X, degree: int = 2):
    return transform("n4m_pp_emsc", X, ctypes.c_int32(degree), fit_X=X)


def baseline_center(X):
    return transform("n4m_pp_baseline", X, fit_X=X)


def derivate(X, order: int = 1, delta: float = 1.0):
    X_arr = as_f64_2d(X)
    out_cols = int(
        lib.n4m_pp_derivate_output_cols(
            ctypes.c_int32(order), ctypes.c_int64(X_arr.shape[1])
        )
    )
    if out_cols <= 0:
        raise ValueError("order must be smaller than the input width")
    return transform(
        "n4m_pp_derivate",
        X_arr,
        ctypes.c_int32(order),
        ctypes.c_double(delta),
        fit_X=X_arr,
        out_shape=(X_arr.shape[0], out_cols),
    )


def savitzky_golay(
    X,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0,
    mode: str | int = "mirror",
    cval: float = 0.0,
):
    return transform(
        "n4m_pp_savgol",
        X,
        ctypes.c_int32(window_length),
        ctypes.c_int32(polyorder),
        ctypes.c_int32(deriv),
        ctypes.c_double(delta),
        ctypes.c_int(_enum(_SAVGOL_MODES, mode, "mode")),
        ctypes.c_double(cval),
    )


def first_derivative(X, delta: float = 1.0, edge_order: int = 2):
    return transform("n4m_pp_first_derivative", X, ctypes.c_double(delta), ctypes.c_int32(edge_order))


def second_derivative(X, delta: float = 1.0, edge_order: int = 2):
    return transform("n4m_pp_second_derivative", X, ctypes.c_double(delta), ctypes.c_int32(edge_order))


def norris_williams(X, segment: int = 5, gap: int = 5, derivative_order: int = 1, delta: float = 1.0):
    return transform("n4m_pp_norris_williams", X, ctypes.c_int32(segment), ctypes.c_int32(gap), ctypes.c_int32(derivative_order), ctypes.c_double(delta))


def gaussian(X, sigma: float = 1.0, order: int = 0, mode: str | int = "reflect", cval: float = 0.0, truncate: float = 4.0):
    return transform("n4m_pp_gaussian", X, ctypes.c_double(sigma), ctypes.c_int32(order), ctypes.c_int(_enum(_GAUSSIAN_MODES, mode, "mode")), ctypes.c_double(cval), ctypes.c_double(truncate))


def to_absorbance(X, source_type: int = 0, epsilon: float = 1e-10, clip_negative: bool = True):
    return transform("n4m_pp_to_absorbance", X, ctypes.c_int(source_type), ctypes.c_double(epsilon), ctypes.c_int(clip_negative))


def from_absorbance(X, target_type: int = 0):
    return transform("n4m_pp_from_absorbance", X, ctypes.c_int(target_type))


def pct_to_frac(X):
    return transform("n4m_pp_pct_to_frac", X)


percent_to_fraction = pct_to_frac


def frac_to_pct(X):
    return transform("n4m_pp_frac_to_pct", X)


fraction_to_percent = frac_to_pct


def kubelka_munk(X, source_type: int = 0, epsilon: float = 1e-10):
    return transform("n4m_pp_kubelka_munk", X, ctypes.c_int(source_type), ctypes.c_double(epsilon))


def detrend(X, polyorder: int = 1):
    return transform("n4m_pp_detrend", X, ctypes.c_int32(polyorder))


def asls(X, lam: float = 1e6, p: float = 1e-2, max_iter: int = 50, tol: float = 1e-3):
    return transform("n4m_pp_asls", X, ctypes.c_double(lam), ctypes.c_double(p), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def airpls(X, lam: float = 1e6, max_iter: int = 50, tol: float = 1e-3):
    return transform("n4m_pp_airpls", X, ctypes.c_double(lam), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def arpls(X, lam: float = 1e5, max_iter: int = 50, tol: float = 1e-3):
    return transform("n4m_pp_arpls", X, ctypes.c_double(lam), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def modpoly(X, polyorder: int = 2, max_iter: int = 250, tol: float = 1e-3):
    return transform("n4m_pp_modpoly", X, ctypes.c_int32(polyorder), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def imodpoly(X, polyorder: int = 2, max_iter: int = 250, tol: float = 1e-3):
    return transform("n4m_pp_imodpoly", X, ctypes.c_int32(polyorder), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def snip(X, max_half_window: int = 20):
    return transform("n4m_pp_snip", X, ctypes.c_int32(max_half_window))


def rolling_ball(X, half_window: int = 20, smooth_half_window: int = 0):
    return transform("n4m_pp_rolling_ball", X, ctypes.c_int32(half_window), ctypes.c_int32(smooth_half_window))


def iasls(X, lam: float = 1e6, p: float = 1e-2, lam_1: float = 1e-4, polyorder: int = 2, diff_order: int = 2, max_iter: int = 50, tol: float = 1e-3):
    X_arr = as_f64_2d(X)
    handle = _create_ex("n4m_pp_iasls", ctypes.c_double(lam), ctypes.c_double(p), ctypes.c_double(lam_1), ctypes.c_int32(polyorder), ctypes.c_int32(diff_order), ctypes.c_int32(max_iter), ctypes.c_double(tol))
    try:
        out = empty_like_f64(X_arr.shape)
        check(lib.n4m_pp_iasls_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_iasls_transform")
        return out
    finally:
        _destroy("n4m_pp_iasls", handle)


def beads(X, lam_0: float = 100.0, lam_1: float = 0.5, lam_2: float = 0.5, max_iter: int = 50, tol: float = 1e-3):
    return transform("n4m_pp_beads", X, ctypes.c_double(lam_0), ctypes.c_double(lam_1), ctypes.c_double(lam_2), ctypes.c_int32(max_iter), ctypes.c_double(tol))


def wavelet(X, family: str | int = "haar", mode: str | int = "periodization"):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_pp_wavelet", ctypes.c_int(_enum(_WAVELET_FAMILIES, family, "family")), ctypes.c_int(_enum(_WAVELET_BOUNDARIES, mode, "mode")))
    try:
        out_cols = ctypes.c_int64()
        check(lib.n4m_pp_wavelet_output_cols(handle, ctypes.c_int64(X_arr.shape[1]), ctypes.byref(out_cols)), "n4m_pp_wavelet_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(lib.n4m_pp_wavelet_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_wavelet_transform")
        return out
    finally:
        _destroy("n4m_pp_wavelet", handle)


def haar(X):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_pp_haar")
    try:
        out_cols = ctypes.c_int64()
        check(lib.n4m_pp_haar_output_cols(ctypes.c_int64(X_arr.shape[1]), ctypes.byref(out_cols)), "n4m_pp_haar_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(lib.n4m_pp_haar_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_haar_transform")
        return out
    finally:
        _destroy("n4m_pp_haar", handle)


def wavelet_denoise(X, family: str | int = "db4", mode: str | int = "periodization", level: int = 3, threshold_mode: str | int = "soft", noise_estimator: str | int = "median"):
    return transform("n4m_pp_wavelet_denoise", X, ctypes.c_int(_enum(_WAVELET_FAMILIES, family, "family")), ctypes.c_int(_enum(_WAVELET_BOUNDARIES, mode, "mode")), ctypes.c_int32(level), ctypes.c_int(_enum(_WAVELET_THRESHOLDS, threshold_mode, "threshold_mode")), ctypes.c_int(_enum(_WAVELET_NOISE, noise_estimator, "noise_estimator")))


def wavelet_features(X, family: str | int = "haar", mode: str | int = "periodization", max_level: int = 3, entropy: str | int = "energy"):
    X_arr = as_f64_2d(X)
    handle = _create_ex("n4m_pp_wavelet_features", ctypes.c_int(_enum(_WAVELET_FAMILIES, family, "family")), ctypes.c_int(_enum(_WAVELET_BOUNDARIES, mode, "mode")), ctypes.c_int32(max_level), ctypes.c_int(_enum(_WAVELET_FEATURE_ENTROPY, entropy, "entropy")))
    try:
        out_cols = ctypes.c_int64()
        check(lib.n4m_pp_wavelet_features_output_cols(handle, ctypes.c_int64(X_arr.shape[1]), ctypes.byref(out_cols)), "n4m_pp_wavelet_features_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(lib.n4m_pp_wavelet_features_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_wavelet_features_transform")
        return out
    finally:
        _destroy("n4m_pp_wavelet_features", handle)


def _wavelet_projection(prefix: str, X, family: str | int, mode: str | int,
                        max_level: int, n_components: float):
    X_arr = as_f64_2d(X)
    handle = _create(
        prefix,
        ctypes.c_int(_enum(_WAVELET_FAMILIES, family, "family")),
        ctypes.c_int(_enum(_WAVELET_BOUNDARIES, mode, "mode")),
        ctypes.c_int32(max_level),
        ctypes.c_double(n_components),
    )
    try:
        check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(X_arr)),
              f"{prefix}_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)),
              f"{prefix}_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)),
              f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def wavelet_pca(
    X,
    family: str | int = "haar",
    mode: str | int = "periodization",
    max_level: int = 2,
    n_components: float = 5.0,
):
    return _wavelet_projection(
        "n4m_pp_wavelet_pca", X, family, mode, max_level, n_components
    )


def wavelet_svd(
    X,
    family: str | int = "haar",
    mode: str | int = "periodization",
    max_level: int = 2,
    n_components: float = 5.0,
):
    return _wavelet_projection(
        "n4m_pp_wavelet_svd", X, family, mode, max_level, n_components
    )


def osc(X, y, n_components: int = 1, scale: bool = True):
    return _fit_y_transform("n4m_pp_osc", X, y, ctypes.c_int32(n_components), ctypes.c_int(scale))


def epo(X, d, scale: bool = True):
    X_arr = as_f64_2d(X)
    d_arr = _as_f64_1d(d, "d")
    handle = _create("n4m_pp_epo", ctypes.c_int(scale))
    try:
        check(lib.n4m_pp_epo_fit(handle, numpy_to_view(X_arr), _f64_ptr(d_arr), ctypes.c_int64(d_arr.size)), "n4m_pp_epo_fit")
        out = empty_like_f64(X_arr.shape)
        check(lib.n4m_pp_epo_transform_with_d(handle, numpy_to_view(X_arr), _f64_ptr(d_arr), ctypes.c_int64(d_arr.size), numpy_to_view(out)), "n4m_pp_epo_transform_with_d")
        return out
    finally:
        _destroy("n4m_pp_epo", handle)


def flexible_pca(X, n_components: float = 5.0):
    return _flexible("n4m_pp_flex_pca", X, n_components)


def flexible_svd(X, n_components: float = 5.0):
    return _flexible("n4m_pp_flex_svd", X, n_components)


def _flexible(prefix: str, X, n_components: float):
    X_arr = as_f64_2d(X)
    handle = _create(prefix, ctypes.c_double(n_components))
    try:
        check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(X_arr)), f"{prefix}_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)), f"{prefix}_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def fck_static(X, kernel_size: int, alphas: Sequence[float], sigmas: Sequence[float]):
    X_arr = as_f64_2d(X)
    alpha_arr = _as_f64_1d(alphas, "alphas")
    sigma_arr = _as_f64_1d(sigmas, "sigmas")
    handle = _create("n4m_pp_fck_static", ctypes.c_int32(kernel_size), _f64_ptr(alpha_arr), ctypes.c_int32(alpha_arr.size), _f64_ptr(sigma_arr), ctypes.c_int32(sigma_arr.size))
    try:
        out_cols = ctypes.c_int32()
        check(lib.n4m_pp_fck_static_output_cols(ctypes.c_int32(alpha_arr.size * sigma_arr.size), ctypes.c_int32(X_arr.shape[1]), ctypes.byref(out_cols)), "n4m_pp_fck_static_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(lib.n4m_pp_fck_static_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_fck_static_transform")
        return out
    finally:
        _destroy("n4m_pp_fck_static", handle)


def direct_standardization(X_source, X_target, X=None, fit_intercept: bool = True, ridge: float = 0.0):
    return _paired_transfer("n4m_pp_direct_standardization", X_source, X_target, X_source if X is None else X, ctypes.c_int(fit_intercept), ctypes.c_double(ridge))


def robust_direct_standardization(X_source, X_target, X=None, fit_intercept: bool = True, ridge: float = 0.0, trim_quantile: float = 0.9, max_iter: int = 3):
    return _paired_transfer("n4m_pp_robust_direct_standardization", X_source, X_target, X_source if X is None else X, ctypes.c_int(fit_intercept), ctypes.c_double(ridge), ctypes.c_double(trim_quantile), ctypes.c_int32(max_iter))


def piecewise_direct_standardization(X_source, X_target, X=None, window_size: int = 5, fit_intercept: bool = True, ridge: float = 0.0):
    return _paired_transfer("n4m_pp_piecewise_direct_standardization", X_source, X_target, X_source if X is None else X, ctypes.c_int32(window_size), ctypes.c_int(fit_intercept), ctypes.c_double(ridge))


def score_augmented_projection_standardization(X_source, X_target, X=None, n_components: int = 5, score_weight: float = 1.0, fit_intercept: bool = True, ridge: float = 0.0):
    return _paired_transfer("n4m_pp_saps", X_source, X_target, X_source if X is None else X, ctypes.c_int32(n_components), ctypes.c_double(score_weight), ctypes.c_int(fit_intercept), ctypes.c_double(ridge))


def slope_bias_correction(y_source, y_target, y=None):
    source = _as_f64_1d(y_source, "y_source")
    target = _as_f64_1d(y_target, "y_target")
    values = source if y is None else _as_f64_1d(y, "y")
    if source.shape != target.shape:
        raise ValueError("y_source and y_target must have the same shape")
    out = np.empty_like(values)
    handle = _create("n4m_pp_slope_bias")
    try:
        check(lib.n4m_pp_slope_bias_fit(handle, _f64_ptr(source), _f64_ptr(target), ctypes.c_int64(source.size)), "n4m_pp_slope_bias_fit")
        check(lib.n4m_pp_slope_bias_transform(handle, _f64_ptr(values), ctypes.c_int64(values.size), _f64_ptr(out)), "n4m_pp_slope_bias_transform")
        return out
    finally:
        _destroy("n4m_pp_slope_bias", handle)


def local_centering(X_source, X_target, X=None):
    return _paired_transfer("n4m_pp_local_centering", X_source, X_target, X_source if X is None else X)


def weighted_snv(X, weights=None, ddof: int = 0, eps: float = 1e-12):
    weight_arr = None if weights is None else _as_f64_1d(weights, "weights")
    ptr = _null_f64_ptr() if weight_arr is None else _f64_ptr(weight_arr)
    n_weights = 0 if weight_arr is None else weight_arr.size
    return transform("n4m_pp_weighted_snv", X, ptr, ctypes.c_int64(n_weights), ctypes.c_int32(ddof), ctypes.c_double(eps), fit_X=X)


def variable_sorting_normalization(X, eps: float = 1e-12):
    return transform("n4m_pp_vsn", X, ctypes.c_double(eps), fit_X=X)


def piecewise_snv(X, window_size: int = 32, ddof: int = 0, eps: float = 1e-12):
    return transform("n4m_pp_piecewise_snv", X, ctypes.c_int32(window_size), ctypes.c_int32(ddof), ctypes.c_double(eps), fit_X=X)


def piecewise_msc(X, reference=None, window_size: int = 32, eps: float = 1e-12):
    ref = None if reference is None else _as_f64_1d(reference, "reference")
    ptr = _null_f64_ptr() if ref is None else _f64_ptr(ref)
    return transform("n4m_pp_piecewise_msc", X, ptr, ctypes.c_int64(0 if ref is None else ref.size), ctypes.c_int32(window_size), ctypes.c_double(eps), fit_X=X)


def localized_msc(X, reference=None, window_size: int = 32, eps: float = 1e-12):
    ref = None if reference is None else _as_f64_1d(reference, "reference")
    ptr = _null_f64_ptr() if ref is None else _f64_ptr(ref)
    return transform("n4m_pp_localized_msc", X, ptr, ctypes.c_int64(0 if ref is None else ref.size), ctypes.c_int32(window_size), ctypes.c_double(eps), fit_X=X)


def _align(prefix: str, X, reference=None, interval_size: int = 0, max_shift: int = 0):
    ref = None if reference is None else _as_f64_1d(reference, "reference")
    ptr = _null_f64_ptr() if ref is None else _f64_ptr(ref)
    return transform(prefix, X, ptr, ctypes.c_int64(0 if ref is None else ref.size), ctypes.c_int32(interval_size), ctypes.c_int32(max_shift), fit_X=X)


def cross_correlation_alignment(X, reference=None, max_shift: int = 5):
    return _align("n4m_pp_xcorr_align", X, reference, 0, max_shift)


def icoshift_alignment(X, reference=None, interval_size: int = 32, max_shift: int = 5):
    return _align("n4m_pp_icoshift_align", X, reference, interval_size, max_shift)


def dynamic_time_warping_alignment(X, reference=None):
    return _align("n4m_pp_dtw_align", X, reference, 0, 0)


def correlation_optimized_warping(X, reference=None, interval_size: int = 32, max_shift: int = 5):
    return _align("n4m_pp_cow_align", X, reference, interval_size, max_shift)


def variance_filter(X, threshold: float = 0.0, top_k: int | None = None):
    return _selector("n4m_filter_variance", X, None, threshold=threshold, top_k=top_k)


def correlation_filter(X, y, threshold: float = 0.0, top_k: int | None = None):
    return _selector("n4m_filter_correlation", X, y, threshold=threshold, top_k=top_k)


def _selector(prefix: str, X, y=None, *, threshold: float = 0.0, top_k: int | None = None):
    X_arr = as_f64_2d(X)
    handle = _create(prefix, ctypes.c_double(threshold), ctypes.c_int32(-1 if top_k is None else int(top_k)))
    try:
        if prefix == "n4m_filter_correlation":
            y_arr = _as_f64_1d(y, "y")
            check(lib.n4m_filter_correlation_fit(handle, numpy_to_view(X_arr), _f64_ptr(y_arr), ctypes.c_int64(y_arr.size)), "n4m_filter_correlation_fit")
        else:
            check(lib.n4m_filter_variance_fit(handle, numpy_to_view(X_arr)), "n4m_filter_variance_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)), f"{prefix}_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        _destroy(prefix, handle)


def interval_generator(X, interval_size: int = 32, step: int | None = None):
    X_arr = as_f64_2d(X)
    width = max(1, int(interval_size))
    stride = width if step is None else max(1, int(step))
    handle = _create("n4m_interval_generator", ctypes.c_int32(width), ctypes.c_int32(0 if step is None else stride))
    try:
        check(lib.n4m_interval_generator_fit(handle, numpy_to_view(X_arr)), "n4m_interval_generator_fit")
        out_cols = ctypes.c_int64()
        check(lib.n4m_interval_generator_output_cols(handle, ctypes.byref(out_cols)), "n4m_interval_generator_output_cols")
        out = empty_like_f64((X_arr.shape[0], int(out_cols.value)))
        check(lib.n4m_interval_generator_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_interval_generator_transform")
        blocks = []
        offset = 0
        for lo in range(0, X_arr.shape[1], stride):
            hi = min(X_arr.shape[1], lo + width)
            cols = hi - lo
            blocks.append(out[:, offset:offset + cols])
            offset += cols
        return blocks
    finally:
        _destroy("n4m_interval_generator", handle)


def crop(X, start: int, end: int):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_pp_crop", ctypes.c_int64(start), ctypes.c_int64(end))
    try:
        out_cols = int(lib.n4m_pp_crop_output_cols(handle, ctypes.c_int64(X_arr.shape[1])))
        out = empty_like_f64((X_arr.shape[0], out_cols))
        check(lib.n4m_pp_crop_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_crop_transform")
        return out
    finally:
        _destroy("n4m_pp_crop", handle)


def resample(X, num_samples: int):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_pp_resample", ctypes.c_int64(num_samples))
    try:
        out_cols = int(lib.n4m_pp_resample_output_cols(handle, ctypes.c_int64(X_arr.shape[1])))
        out = empty_like_f64((X_arr.shape[0], out_cols))
        check(lib.n4m_pp_resample_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_resample_transform")
        return out
    finally:
        _destroy("n4m_pp_resample", handle)


def resampler(X, source_wavelengths, target_wavelengths, method: int = 0, crop_min: float = 0.0, crop_max: float = 0.0, use_crop: bool = False, fill_value: float = 0.0, bounds_error: bool = False, extrapolate: bool = False):
    X_arr = as_f64_2d(X)
    source = _as_f64_1d(source_wavelengths, "source_wavelengths")
    target = _as_f64_1d(target_wavelengths, "target_wavelengths")
    handle = _create("n4m_pp_resampler", _f64_ptr(target), ctypes.c_int64(target.size), ctypes.c_int32(method), ctypes.c_double(crop_min), ctypes.c_double(crop_max), ctypes.c_int(use_crop), ctypes.c_double(fill_value), ctypes.c_int(bounds_error), ctypes.c_int(extrapolate))
    try:
        check(lib.n4m_pp_resampler_fit(handle, _f64_ptr(source), ctypes.c_int64(source.size)), "n4m_pp_resampler_fit")
        out_cols = int(lib.n4m_pp_resampler_output_cols(handle))
        out = empty_like_f64((X_arr.shape[0], out_cols))
        check(lib.n4m_pp_resampler_transform(handle, numpy_to_view(X_arr), numpy_to_view(out)), "n4m_pp_resampler_transform")
        return out
    finally:
        _destroy("n4m_pp_resampler", handle)


def range_discretizer(X, edges: Sequence[float]):
    edge_arr = _as_f64_1d(edges, "edges")
    return transform("n4m_pp_range_disc", X, _f64_ptr(edge_arr), ctypes.c_int64(edge_arr.size), out_dtype="i32")


def kbins_discretizer(X, n_bins: int = 5, strategy: str | int = "uniform"):
    if isinstance(strategy, str):
        strategy = {"uniform": 0, "quantile": 1}[strategy]
    return transform("n4m_pp_kbins_disc", X, ctypes.c_int32(n_bins), ctypes.c_int32(strategy), fit_X=X, out_dtype="i32")


def kennard_stone(X, test_size: float = 0.25):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_split_kennard_stone", ctypes.c_double(test_size))
    try:
        result = SplitResult()
        check(lib.n4m_split_kennard_stone_split(handle, numpy_to_view(X_arr), ctypes.byref(result)), "n4m_split_kennard_stone_split")
        return _split_result_to_arrays(result)
    finally:
        _destroy("n4m_split_kennard_stone", handle)


def spxy(X, y, test_size: float = 0.25):
    X_arr = as_f64_2d(X)
    y_arr = as_f64_2d(y)
    handle = _create("n4m_split_spxy", ctypes.c_double(test_size))
    try:
        result = SplitResult()
        check(lib.n4m_split_spxy_split(handle, numpy_to_view(X_arr), numpy_to_view(y_arr), ctypes.byref(result)), "n4m_split_spxy_split")
        return _split_result_to_arrays(result)
    finally:
        _destroy("n4m_split_spxy", handle)


def kbins_stratified(y, test_size: float = 0.25, seed: int = 17, n_bins: int = 5, strategy: str | int = "uniform"):
    if isinstance(strategy, str):
        strategy = {"uniform": 0, "quantile": 1}[strategy]
    y_arr = as_f64_2d(y)
    handle = _create("n4m_split_kbins_stratified", ctypes.c_double(test_size), ctypes.c_uint64(seed), ctypes.c_int32(n_bins), ctypes.c_int32(strategy))
    try:
        result = SplitResult()
        check(lib.n4m_split_kbins_stratified_split(handle, numpy_to_view(y_arr), ctypes.byref(result)), "n4m_split_kbins_stratified_split")
        return _split_result_to_arrays(result)
    finally:
        _destroy("n4m_split_kbins_stratified", handle)


def y_outlier_iqr(y, threshold: float = 1.5, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
    return y_outlier_filter(y, method="iqr", threshold=threshold, lower_percentile=lower_percentile, upper_percentile=upper_percentile)[0]


def y_outlier_filter(y, method: str | int = "iqr", threshold: float = 1.5, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
    y_arr = _as_f64_1d(y, "y")
    handle = _create("n4m_filter_y_outlier", ctypes.c_int(_enum(_Y_METHODS, method, "method")), ctypes.c_double(threshold), ctypes.c_double(lower_percentile), ctypes.c_double(upper_percentile))
    try:
        check(lib.n4m_filter_y_outlier_fit(handle, _f64_ptr(y_arr), ctypes.c_int64(y_arr.size)), "n4m_filter_y_outlier_fit")
        mask = np.empty(y_arr.size, dtype=np.uint8)
        stats = FilterStats()
        check(lib.n4m_filter_y_outlier_apply(handle, _f64_ptr(y_arr), ctypes.c_int64(y_arr.size), mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.byref(stats)), "n4m_filter_y_outlier_apply")
        return mask, stats
    finally:
        _destroy("n4m_filter_y_outlier", handle)


def x_outlier_mahalanobis(X, n_components: int = 0, contamination: float = 0.1):
    return x_outlier_filter(X, method="mahalanobis", n_components=n_components, contamination=contamination)[0]


def x_outlier_filter(X, method: str | int = "mahalanobis", use_threshold: bool = False, threshold: float = 0.0, n_components: int = 0, contamination: float = 0.1, seed: int = 0, n_estimators: int = 100, max_samples: int = 256):
    X_arr = as_f64_2d(X)
    handle = _create("n4m_filter_x_outlier", ctypes.c_int32(_enum(_X_METHODS, method, "method")), ctypes.c_int(use_threshold), ctypes.c_double(threshold), ctypes.c_int32(n_components), ctypes.c_double(contamination), ctypes.c_uint64(seed), ctypes.c_int32(n_estimators), ctypes.c_int64(max_samples))
    try:
        check(lib.n4m_filter_x_outlier_fit(handle, numpy_to_view(X_arr)), "n4m_filter_x_outlier_fit")
        mask = np.empty(X_arr.shape[0], dtype=np.uint8)
        stats = FilterStats()
        check(lib.n4m_filter_x_outlier_apply(handle, numpy_to_view(X_arr), mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.byref(stats)), "n4m_filter_x_outlier_apply")
        return mask, stats
    finally:
        _destroy("n4m_filter_x_outlier", handle)


def high_leverage_filter(
    X,
    method: str | int = "hat",
    threshold_multiplier: float = 2.0,
    absolute_threshold: float | None = None,
    n_components: int = 0,
    center: bool = True,
):
    X_arr = as_f64_2d(X)
    use_abs = absolute_threshold is not None
    handle = _create(
        "n4m_filter_leverage",
        ctypes.c_int32(_enum(_LEVERAGE_METHODS, method, "method")),
        ctypes.c_double(threshold_multiplier),
        ctypes.c_int(1 if use_abs else 0),
        ctypes.c_double(0.0 if absolute_threshold is None else absolute_threshold),
        ctypes.c_int32(n_components),
        ctypes.c_int(1 if center else 0),
    )
    try:
        check(lib.n4m_filter_leverage_fit(handle, numpy_to_view(X_arr)),
              "n4m_filter_leverage_fit")
        mask = np.empty(X_arr.shape[0], dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.n4m_filter_leverage_apply(
                handle,
                numpy_to_view(X_arr),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_leverage_apply",
        )
        return mask
    finally:
        _destroy("n4m_filter_leverage", handle)


def spectral_quality_filter(
    X,
    max_nan_ratio: float = 0.1,
    max_zero_ratio: float = 0.5,
    min_variance: float = 1e-8,
    max_value: float | None = None,
    min_value: float | None = None,
    check_inf: bool = True,
):
    X_arr = as_f64_2d(X)
    handle = _create(
        "n4m_filter_quality",
        ctypes.c_double(max_nan_ratio),
        ctypes.c_double(max_zero_ratio),
        ctypes.c_double(min_variance),
        ctypes.c_int(0 if max_value is None else 1),
        ctypes.c_double(0.0 if max_value is None else max_value),
        ctypes.c_int(0 if min_value is None else 1),
        ctypes.c_double(0.0 if min_value is None else min_value),
        ctypes.c_int(1 if check_inf else 0),
    )
    try:
        mask = np.empty(X_arr.shape[0], dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.n4m_filter_quality_apply(
                handle,
                numpy_to_view(X_arr),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_quality_apply",
        )
        return mask
    finally:
        _destroy("n4m_filter_quality", handle)


def composite_filter(X, mode: str | int = "any"):
    X_arr = as_f64_2d(X)
    leverage = _create(
        "n4m_filter_leverage",
        ctypes.c_int32(0),
        ctypes.c_double(2.0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_int32(0),
        ctypes.c_int(1),
    )
    quality = _create(
        "n4m_filter_quality",
        ctypes.c_double(0.1),
        ctypes.c_double(0.5),
        ctypes.c_double(1e-8),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_int(1),
    )
    composite = _create(
        "n4m_filter_composite",
        ctypes.c_int(_enum(_COMPOSITE_MODES, mode, "mode")),
    )
    try:
        check(lib.n4m_filter_leverage_fit(leverage, numpy_to_view(X_arr)),
              "n4m_filter_leverage_fit")
        check(lib.n4m_filter_composite_add_leverage(composite, leverage),
              "n4m_filter_composite_add_leverage")
        check(lib.n4m_filter_composite_add_quality(composite, quality),
              "n4m_filter_composite_add_quality")
        mask = np.empty(X_arr.shape[0], dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.n4m_filter_composite_apply(
                composite,
                numpy_to_view(X_arr),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_composite_apply",
        )
        return mask
    finally:
        _destroy("n4m_filter_composite", composite)
        _destroy("n4m_filter_quality", quality)
        _destroy("n4m_filter_leverage", leverage)


def hotelling_t2(X, n_components: int = 5, alpha: float = 0.05):
    X_arr = as_f64_2d(X)
    values = np.empty(X_arr.shape[0], dtype=np.float64)
    ucl = ctypes.c_double()
    check(
        lib.n4m_util_hotelling_t2(
            numpy_to_view(X_arr),
            ctypes.c_int32(n_components),
            ctypes.c_double(alpha),
            _f64_ptr(values),
            ctypes.c_int64(values.size),
            ctypes.byref(ucl),
        ),
        "n4m_util_hotelling_t2",
    )
    return values, float(ucl.value)


def q_residuals(X, n_components: int = 5, alpha: float = 0.05):
    X_arr = as_f64_2d(X)
    values = np.empty(X_arr.shape[0], dtype=np.float64)
    ucl = ctypes.c_double()
    check(
        lib.n4m_util_q_residuals(
            numpy_to_view(X_arr),
            ctypes.c_int32(n_components),
            ctypes.c_double(alpha),
            _f64_ptr(values),
            ctypes.c_int64(values.size),
            ctypes.byref(ucl),
        ),
        "n4m_util_q_residuals",
    )
    return values, float(ucl.value)


def _metric_scalar(name: str, y_true, y_pred) -> float:
    yt = _as_f64_1d(y_true, "y_true")
    yp = _as_f64_1d(y_pred, "y_pred")
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    out = ctypes.c_double()
    check(
        getattr(lib, f"n4m_metric_{name}")(
            _f64_ptr(yt), _f64_ptr(yp), ctypes.c_int64(yt.size), ctypes.byref(out)
        ),
        f"n4m_metric_{name}",
    )
    return float(out.value)


def nirs_metrics(y_true, y_pred):
    names = ("rmse", "mae", "bias", "sep", "rpd", "rpiq", "r2", "nrmse")
    return np.asarray([_metric_scalar(name, y_true, y_pred) for name in names],
                      dtype=np.float64)


def signal_type_detector(
    X,
    wavelengths=None,
    confidence_threshold: float = 0.7,
):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    type_out = ctypes.c_int()
    confidence = ctypes.c_double()
    reason = ctypes.create_string_buffer(256)
    check(
        lib.n4m_signal_detect(
            numpy_to_view(X_arr),
            wl_ptr,
            wl_n,
            ctypes.c_double(confidence_threshold),
            ctypes.byref(type_out),
            ctypes.byref(confidence),
            ctypes.cast(reason, ctypes.c_void_p),
        ),
        "n4m_signal_detect",
    )
    return np.asarray([float(type_out.value), float(confidence.value)], dtype=np.float64)


def transfer_metrics(
    X_source,
    X_target,
    n_components: int = 10,
    k_neighbors: int = 10,
    seed: int = 0,
):
    source = as_f64_2d(X_source)
    target = as_f64_2d(X_target)
    out = TransferMetrics()
    check(
        lib.n4m_transfer_metrics_compute(
            numpy_to_view(source),
            numpy_to_view(target),
            ctypes.c_int32(n_components),
            ctypes.c_int32(k_neighbors),
            ctypes.c_uint64(seed),
            ctypes.byref(out),
        ),
        "n4m_transfer_metrics_compute",
    )
    return np.asarray([float(getattr(out, name)) for name, _ctype in out._fields_],
                      dtype=np.float64)


def rng_pcg64(seed: int = 0, n: int = 1024):
    rng = PCG64(seed)
    out = np.empty(int(n), dtype=np.float64)
    try:
        check(
            lib.n4m_rng_pcg64_standard_normal_fill(
                rng.handle,
                _f64_ptr(out),
                ctypes.c_size_t(out.size),
            ),
            "n4m_rng_pcg64_standard_normal_fill",
        )
        return out
    finally:
        rng.close()


def aug_gaussian_noise(X, sigma: float = 0.01, seed: int = 0):
    return _aug_apply("n4m_aug_gaussian_noise", X, ctypes.c_double(sigma), seed=seed)


def aug_multiplicative_noise(X, sigma_gain: float = 0.01, seed: int = 0):
    return _aug_apply(
        "n4m_aug_multiplicative_noise", X, ctypes.c_double(sigma_gain), seed=seed
    )


def aug_spike_noise(
    X,
    n_spikes_min: int = 1,
    n_spikes_max: int = 3,
    amplitude_min: float = -0.1,
    amplitude_max: float = 0.1,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_spike_noise",
        X,
        ctypes.c_int32(n_spikes_min),
        ctypes.c_int32(n_spikes_max),
        ctypes.c_double(amplitude_min),
        ctypes.c_double(amplitude_max),
        seed=seed,
    )


def aug_hetero_noise(
    X,
    noise_base: float = 0.001,
    noise_signal_dep: float = 0.01,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_hetero_noise",
        X,
        ctypes.c_double(noise_base),
        ctypes.c_double(noise_signal_dep),
        seed=seed,
    )


def aug_linear_drift(
    X,
    offset_min: float = -0.05,
    offset_max: float = 0.05,
    slope_min: float = -0.01,
    slope_max: float = 0.01,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_linear_drift",
        X,
        ctypes.c_double(offset_min),
        ctypes.c_double(offset_max),
        ctypes.c_double(slope_min),
        ctypes.c_double(slope_max),
        seed=seed,
    )


def aug_poly_drift(X, degree: int = 2, coeff_min=None, coeff_max=None, seed: int = 0):
    if coeff_min is None:
        coeff_min = np.full(int(degree) + 1, -0.01, dtype=np.float64)
    if coeff_max is None:
        coeff_max = np.full(int(degree) + 1, 0.01, dtype=np.float64)
    lo = _as_f64_1d(coeff_min, "coeff_min")
    hi = _as_f64_1d(coeff_max, "coeff_max")
    return _aug_apply(
        "n4m_aug_poly_drift",
        X,
        ctypes.c_int32(degree),
        _f64_ptr(lo),
        _f64_ptr(hi),
        seed=seed,
    )


def aug_path_length(
    X,
    path_length_std: float = 0.05,
    min_path_length: float = 0.1,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_path_length",
        X,
        ctypes.c_double(path_length_std),
        ctypes.c_double(min_path_length),
        seed=seed,
    )


def aug_wavelength_shift(
    X,
    shift_lo: float = -1.0,
    shift_hi: float = 1.0,
    wavelengths=None,
    seed: int = 0,
):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_wavelength_shift",
        X_arr,
        ctypes.c_double(shift_lo),
        ctypes.c_double(shift_hi),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_wavelength_stretch(
    X,
    stretch_lo: float = 0.99,
    stretch_hi: float = 1.01,
    wavelengths=None,
    seed: int = 0,
):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_wavelength_stretch",
        X_arr,
        ctypes.c_double(stretch_lo),
        ctypes.c_double(stretch_hi),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_local_warp(
    X,
    n_control_points: int = 5,
    max_shift: float = 1.0,
    wavelengths=None,
    seed: int = 0,
):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_local_warp",
        X_arr,
        ctypes.c_int32(n_control_points),
        ctypes.c_double(max_shift),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_band_perturb(
    X,
    n_bands: int = 3,
    bw_lo: int = 5,
    bw_hi: int = 15,
    gain_lo: float = 0.9,
    gain_hi: float = 1.1,
    offset_lo: float = -0.01,
    offset_hi: float = 0.01,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_band_perturb",
        X,
        ctypes.c_int32(n_bands),
        ctypes.c_int32(bw_lo),
        ctypes.c_int32(bw_hi),
        ctypes.c_double(gain_lo),
        ctypes.c_double(gain_hi),
        ctypes.c_double(offset_lo),
        ctypes.c_double(offset_hi),
        seed=seed,
    )


def aug_band_mask(
    X,
    n_bands_lo: int = 1,
    n_bands_hi: int = 3,
    bw_lo: int = 5,
    bw_hi: int = 15,
    mode: str | int = "zero",
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_band_mask",
        X,
        ctypes.c_int32(n_bands_lo),
        ctypes.c_int32(n_bands_hi),
        ctypes.c_int32(bw_lo),
        ctypes.c_int32(bw_hi),
        ctypes.c_int32(0 if mode == "zero" else 1 if mode == "interp" else int(mode)),
        seed=seed,
    )


def aug_channel_dropout(
    X,
    dropout_prob: float = 0.05,
    mode: str | int = "zero",
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_channel_dropout",
        X,
        ctypes.c_double(dropout_prob),
        ctypes.c_int32(0 if mode == "zero" else 1 if mode == "interp" else int(mode)),
        seed=seed,
    )


def aug_gauss_jitter(
    X,
    sigma_lo: float = 0.5,
    sigma_hi: float = 1.5,
    kernel_width: int = 9,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_gauss_jitter",
        X,
        ctypes.c_double(sigma_lo),
        ctypes.c_double(sigma_hi),
        ctypes.c_int32(kernel_width),
        seed=seed,
    )


def aug_unsharp_mask(
    X,
    amount_lo: float = 0.1,
    amount_hi: float = 0.5,
    sigma: float = 1.0,
    kernel_width: int = 11,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_unsharp_mask",
        X,
        ctypes.c_double(amount_lo),
        ctypes.c_double(amount_hi),
        ctypes.c_double(sigma),
        ctypes.c_int32(kernel_width),
        seed=seed,
    )


def aug_magnitude_warp(
    X,
    n_control_points: int = 5,
    gain_lo: float = 0.9,
    gain_hi: float = 1.1,
    wavelengths=None,
    seed: int = 0,
):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_magnitude_warp",
        X_arr,
        ctypes.c_int32(n_control_points),
        ctypes.c_double(gain_lo),
        ctypes.c_double(gain_hi),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_local_clip(
    X,
    n_regions: int = 1,
    width_lo: int = 5,
    width_hi: int = 15,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_local_clip",
        X,
        ctypes.c_int32(n_regions),
        ctypes.c_int32(width_lo),
        ctypes.c_int32(width_hi),
        seed=seed,
    )


def aug_wavelength_spectral(X, wavelengths=None, seed: int = 0):
    parts = [
        aug_wavelength_shift(X, wavelengths=wavelengths, seed=seed),
        aug_wavelength_stretch(X, wavelengths=wavelengths, seed=seed),
        aug_local_warp(X, n_control_points=3, wavelengths=wavelengths, seed=seed),
        aug_band_perturb(X, seed=seed),
        aug_band_mask(X, seed=seed),
        aug_channel_dropout(X, seed=seed),
        aug_gauss_jitter(X, seed=seed),
        aug_unsharp_mask(X, seed=seed),
        aug_magnitude_warp(X, n_control_points=3, wavelengths=wavelengths, seed=seed),
        aug_local_clip(X, seed=seed),
    ]
    return np.concatenate(parts, axis=1)


def aug_mixup(X, alpha: float = 0.2, seed: int = 0):
    return _aug_apply("n4m_aug_mixup", X, ctypes.c_double(alpha), seed=seed)


def aug_local_mixup(X, alpha: float = 0.2, k_neighbors: int = 5, seed: int = 0):
    return _aug_apply(
        "n4m_aug_local_mixup",
        X,
        ctypes.c_double(alpha),
        ctypes.c_int32(k_neighbors),
        seed=seed,
    )


def aug_scatter_sim(
    X,
    a_low: float = -0.05,
    a_high: float = 0.05,
    b_low: float = 0.9,
    b_high: float = 1.1,
    seed: int = 0,
):
    return _aug_apply(
        "n4m_aug_scatter_sim",
        X,
        ctypes.c_double(a_low),
        ctypes.c_double(a_high),
        ctypes.c_double(b_low),
        ctypes.c_double(b_high),
        seed=seed,
    )


def aug_particle_size(X, wavelengths=None, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=True)
    return _aug_apply(
        "n4m_aug_particle_size",
        X_arr,
        ctypes.c_double(50.0),
        ctypes.c_double(15.0),
        ctypes.c_int(0),
        ctypes.c_double(5.0),
        ctypes.c_double(500.0),
        ctypes.c_double(50.0),
        ctypes.c_double(1.5),
        ctypes.c_double(0.1),
        ctypes.c_int(1),
        ctypes.c_double(0.5),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_emsc_distort(X, wavelengths=None, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=True)
    return _aug_apply(
        "n4m_aug_emsc_distort",
        X_arr,
        ctypes.c_double(0.9),
        ctypes.c_double(1.1),
        ctypes.c_double(-0.05),
        ctypes.c_double(0.05),
        ctypes.c_int32(2),
        ctypes.c_double(0.02),
        ctypes.c_double(0.3),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_batch_effect(X, wavelengths=None, variation_scope: int = 0, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_batch_effect",
        X_arr,
        ctypes.c_double(0.02),
        ctypes.c_double(0.01),
        ctypes.c_double(0.03),
        ctypes.c_int32(variation_scope),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_instrument_broaden(X, wavelengths=None, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=False)
    return _aug_apply(
        "n4m_aug_instrument_broaden",
        X_arr,
        ctypes.c_double(3.0),
        ctypes.c_int(0),
        ctypes.c_double(3.0),
        ctypes.c_double(8.0),
        ctypes.c_int32(0),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_dead_band(X, seed: int = 0):
    return _aug_apply(
        "n4m_aug_dead_band",
        X,
        ctypes.c_int32(1),
        ctypes.c_int32(5),
        ctypes.c_int32(10),
        ctypes.c_double(0.05),
        ctypes.c_double(1.0),
        ctypes.c_int32(0),
        seed=seed,
    )


def aug_temperature(X, wavelengths=None, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=True)
    return _aug_apply(
        "n4m_aug_temperature",
        X_arr,
        ctypes.c_double(5.0),
        ctypes.c_int(0),
        ctypes.c_double(-5.0),
        ctypes.c_double(5.0),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(1),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_moisture(X, wavelengths=None, seed: int = 0):
    X_arr = as_f64_2d(X)
    wl_ptr, wl_n, _wl = _wavelength_ptr(wavelengths, X_arr.shape[1], required=True)
    return _aug_apply(
        "n4m_aug_moisture",
        X_arr,
        ctypes.c_double(0.1),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_double(1.0),
        ctypes.c_double(0.5),
        ctypes.c_double(0.3),
        ctypes.c_double(25.0),
        ctypes.c_double(0.10),
        ctypes.c_int(1),
        ctypes.c_int(1),
        wl_ptr,
        wl_n,
        seed=seed,
    )


def aug_detector_rolloff(X, wavelengths=None, seed: int = 0):
    return _aug_apply(
        "n4m_aug_detector_rolloff",
        X,
        ctypes.c_int32(4),
        ctypes.c_double(1.0),
        ctypes.c_double(0.02),
        ctypes.c_int32(1),
        seed=seed,
        wavelengths=wavelengths,
        apply_wavelengths=True,
    )


def aug_stray_light(X, wavelengths=None, seed: int = 0):
    return _aug_apply(
        "n4m_aug_stray_light",
        X,
        ctypes.c_double(0.001),
        ctypes.c_double(2.0),
        ctypes.c_double(0.1),
        ctypes.c_int32(1),
        seed=seed,
        wavelengths=wavelengths,
        apply_wavelengths=True,
    )


def aug_edge_curve(X, wavelengths=None, curvature_type: int = 0, seed: int = 0):
    return _aug_apply(
        "n4m_aug_edge_curve",
        X,
        ctypes.c_double(0.02),
        ctypes.c_int32(curvature_type),
        ctypes.c_double(0.0),
        ctypes.c_double(0.7),
        seed=seed,
        wavelengths=wavelengths,
        apply_wavelengths=True,
    )


def aug_truncated_peak(X, wavelengths=None, seed: int = 0):
    return _aug_apply(
        "n4m_aug_truncated_peak",
        X,
        ctypes.c_double(0.5),
        ctypes.c_double(0.01),
        ctypes.c_double(0.1),
        ctypes.c_double(50.0),
        ctypes.c_double(200.0),
        ctypes.c_int32(1),
        ctypes.c_int32(1),
        seed=seed,
        wavelengths=wavelengths,
        apply_wavelengths=True,
    )


def aug_edge_artifacts(X, wavelengths=None, seed: int = 0):
    return _aug_apply(
        "n4m_aug_edge_artifacts",
        X,
        ctypes.c_int32(0xF),
        ctypes.c_double(1.0),
        ctypes.c_int32(4),
        seed=seed,
        wavelengths=wavelengths,
        apply_wavelengths=True,
    )


def aug_spline_smooth(X, seed: int = 0):
    return _aug_apply("n4m_aug_spline_smooth", X, seed=seed)


def aug_spline_x_perturb(X, seed: int = 0):
    return _aug_apply(
        "n4m_aug_spline_x_perturb",
        X,
        ctypes.c_int32(3),
        ctypes.c_double(0.05),
        ctypes.c_double(-0.1),
        ctypes.c_double(0.1),
        seed=seed,
    )


def aug_spline_y_perturb(X, seed: int = 0):
    return _aug_apply(
        "n4m_aug_spline_y_perturb",
        X,
        ctypes.c_int32(-1),
        ctypes.c_double(0.005),
        seed=seed,
    )


def aug_rotate_translate(X, seed: int = 0):
    return _aug_apply(
        "n4m_aug_rotate_translate",
        X,
        ctypes.c_double(2.0),
        ctypes.c_double(3.0),
        seed=seed,
    )


def aug_random_x_op(X, seed: int = 0):
    return _aug_apply(
        "n4m_aug_random_x_op",
        X,
        ctypes.c_int32(_enum(_RANDOM_X_OPS, "multiply", "op_kind")),
        ctypes.c_double(0.97),
        ctypes.c_double(1.03),
        seed=seed,
    )


__all__ = [
    name
    for name, value in globals().items()
    if not name.startswith("_")
    and callable(value)
    and getattr(value, "__module__", "") == __name__
]

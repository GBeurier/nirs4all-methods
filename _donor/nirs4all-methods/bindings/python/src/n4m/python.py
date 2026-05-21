# SPDX-License-Identifier: CECILL-2.1
"""ABI-close Python functions over ``libn4m``.

This module deliberately stays close to the C ABI: functions allocate the
matching C handle, run the requested fit/transform/apply call, return NumPy
arrays, and destroy the handle. The scikit-learn compatible estimators live in
``n4m.sklearn``.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence

import numpy as np

from ._errors import check
from ._ffi import lib
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

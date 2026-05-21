# SPDX-License-Identifier: CECILL-2.1
"""Sample-level outlier filters.

Each filter exposes a fit / apply API plus a NumPy ``mask`` and ``stats``
output. The mask is a 1-D ``uint8`` array of length ``n_samples`` (1 = keep,
0 = exclude).
"""
from __future__ import annotations

import ctypes
from typing import Tuple

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import FilterStats
from ._compat import BaseEstimator

# n4m_y_outlier_method_t (string -> enum int)
_Y_METHODS = {"iqr": 0, "zscore": 1, "percentile": 2, "mad": 3}
# n4m_filter_x_outlier_method_t
_X_METHODS = {
    "mahalanobis": 0,
    "robust_mahalanobis": 1,
    "pca_residual": 2,
    "pca_leverage": 3,
    "isolation_forest": 4,
    "lof": 5,
}
_LEVERAGE_METHODS = {"hat": 0, "pca": 1, 0: 0, 1: 1}
_COMPOSITE_MODES = {"any": 0, "all": 1, 0: 0, 1: 1}


def _mask_and_stats(n_samples: int) -> tuple[np.ndarray, FilterStats]:
    return np.empty(n_samples, dtype=np.uint8), FilterStats()


class YOutlierFilter(BaseEstimator):
    """Univariate outlier filter on the target vector ``y``.

    ``method`` is one of ``"iqr"``, ``"zscore"``, ``"percentile"``, ``"mad"``.
    Threshold semantics follow nirs4all's :class:`YOutlierFilter`:

    * For ``"iqr"`` / ``"zscore"`` / ``"mad"``: ``threshold`` is the
      multiplier on the IQR / σ / MAD bounds.
    * For ``"percentile"``: ``lower_percentile`` and ``upper_percentile``
      define the keep band.
    """

    def __init__(self, method: str = "iqr", threshold: float = 1.5,
                 lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        try:
            self._method_int = _Y_METHODS[method]
        except KeyError as exc:
            raise ValueError(f"Unknown YOutlierFilter method: {method!r}") from exc
        self.method = method
        self.threshold = float(threshold)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_filter_y_outlier_create(
                ctypes.byref(h),
                ctypes.c_int(self._method_int),
                ctypes.c_double(self.threshold),
                ctypes.c_double(self.lower_percentile),
                ctypes.c_double(self.upper_percentile),
            ),
            "n4m_filter_y_outlier_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, y):
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
        self._ensure_handle()
        check(
            lib.n4m_filter_y_outlier_fit(
                self._handle,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
            ),
            "n4m_filter_y_outlier_fit",
        )
        self._fitted = True
        # Keep the buffer alive until apply() in case the underlying impl
        # retains a pointer (it doesn't — fit() captures bounds — but defensive).
        self._fit_buffer = y
        return self

    def apply(self, y) -> Tuple[np.ndarray, FilterStats]:
        if not self._fitted:
            raise RuntimeError("YOutlierFilter must be fit() before apply()")
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
        mask = np.empty(y.size, dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.n4m_filter_y_outlier_apply(
                self._handle,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_y_outlier_apply",
        )
        return mask, stats

    def fit_apply(self, y) -> Tuple[np.ndarray, FilterStats]:
        return self.fit(y).apply(y)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_filter_y_outlier_destroy(self._handle)
            self._handle = None


class XOutlierFilter(BaseEstimator):
    """Multivariate outlier filter on the design matrix ``X``.

    ``method`` selects one of six scoring strategies; see
    :c:type:`n4m_filter_x_outlier_method_t`.
    """

    def __init__(self, method: str = "mahalanobis",
                 use_threshold: bool = False,
                 threshold: float = 0.0,
                 n_components: int = 0,
                 contamination: float = 0.1,
                 seed: int = 0,
                 n_estimators: int = 100,
                 max_samples: int = 256):
        try:
            self._method_int = _X_METHODS[method]
        except KeyError as exc:
            raise ValueError(f"Unknown XOutlierFilter method: {method!r}") from exc
        self.method = method
        self.use_threshold = bool(use_threshold)
        self.threshold = float(threshold)
        self.n_components = int(n_components)
        self.contamination = float(contamination)
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_filter_x_outlier_create(
                ctypes.byref(h),
                ctypes.c_int32(self._method_int),
                ctypes.c_int(1 if self.use_threshold else 0),
                ctypes.c_double(self.threshold),
                ctypes.c_int32(self.n_components),
                ctypes.c_double(self.contamination),
                ctypes.c_uint64(self.seed),
                ctypes.c_int32(self.n_estimators),
                ctypes.c_int64(self.max_samples),
            ),
            "n4m_filter_x_outlier_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, X):
        X = as_f64_2d(X)
        self._X = X  # keep buffer alive across the fit + first apply
        self._ensure_handle()
        x_view = numpy_to_view(X)
        check(
            lib.n4m_filter_x_outlier_fit(self._handle, x_view),
            "n4m_filter_x_outlier_fit",
        )
        self._fitted = True
        return self

    def apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        if not self._fitted:
            raise RuntimeError("XOutlierFilter must be fit() before apply()")
        X = as_f64_2d(X)
        mask = np.empty(X.shape[0], dtype=np.uint8)
        stats = FilterStats()
        x_view = numpy_to_view(X)
        check(
            lib.n4m_filter_x_outlier_apply(
                self._handle,
                x_view,
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_x_outlier_apply",
        )
        return mask, stats

    def fit_apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        return self.fit(X).apply(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_filter_x_outlier_destroy(self._handle)
            self._handle = None


class HighLeverageFilter(BaseEstimator):
    """Hat-matrix or PCA score-space leverage filter."""

    def __init__(
        self,
        method: str | int = "hat",
        threshold_multiplier: float = 2.0,
        absolute_threshold: float | None = None,
        n_components: int = 0,
        center: bool = True,
    ):
        try:
            self._method_int = _LEVERAGE_METHODS[method]
        except KeyError as exc:
            raise ValueError(f"Unknown HighLeverageFilter method: {method!r}") from exc
        self.method = method
        self.threshold_multiplier = float(threshold_multiplier)
        self.absolute_threshold = (
            None if absolute_threshold is None else float(absolute_threshold)
        )
        self.n_components = int(n_components)
        self.center = bool(center)
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        use_abs = self.absolute_threshold is not None
        check(
            lib.n4m_filter_leverage_create(
                ctypes.byref(h),
                ctypes.c_int32(self._method_int),
                ctypes.c_double(self.threshold_multiplier),
                ctypes.c_int(1 if use_abs else 0),
                ctypes.c_double(0.0 if self.absolute_threshold is None
                                else self.absolute_threshold),
                ctypes.c_int32(self.n_components),
                ctypes.c_int(1 if self.center else 0),
            ),
            "n4m_filter_leverage_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, X):
        X = as_f64_2d(X)
        check(
            lib.n4m_filter_leverage_fit(self._ensure_handle(), numpy_to_view(X)),
            "n4m_filter_leverage_fit",
        )
        self._fitted = True
        return self

    def apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        if not self._fitted:
            raise RuntimeError("HighLeverageFilter must be fit() before apply()")
        X = as_f64_2d(X)
        mask, stats = _mask_and_stats(X.shape[0])
        check(
            lib.n4m_filter_leverage_apply(
                self._ensure_handle(),
                numpy_to_view(X),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_leverage_apply",
        )
        return mask, stats

    def fit_apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        return self.fit(X).apply(X)

    @property
    def threshold_(self) -> float:
        return float(lib.n4m_filter_leverage_threshold(self._ensure_handle()))

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_filter_leverage_destroy(self._handle)
            self._handle = None


class SpectralQualityFilter(BaseEstimator):
    """Stateless row-level spectrum quality filter."""

    def __init__(
        self,
        max_nan_ratio: float = 0.1,
        max_zero_ratio: float = 0.5,
        min_variance: float = 1e-8,
        max_value: float | None = None,
        min_value: float | None = None,
        check_inf: bool = True,
    ):
        self.max_nan_ratio = float(max_nan_ratio)
        self.max_zero_ratio = float(max_zero_ratio)
        self.min_variance = float(min_variance)
        self.max_value = None if max_value is None else float(max_value)
        self.min_value = None if min_value is None else float(min_value)
        self.check_inf = bool(check_inf)
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_filter_quality_create(
                ctypes.byref(h),
                ctypes.c_double(self.max_nan_ratio),
                ctypes.c_double(self.max_zero_ratio),
                ctypes.c_double(self.min_variance),
                ctypes.c_int(0 if self.max_value is None else 1),
                ctypes.c_double(0.0 if self.max_value is None else self.max_value),
                ctypes.c_int(0 if self.min_value is None else 1),
                ctypes.c_double(0.0 if self.min_value is None else self.min_value),
                ctypes.c_int(1 if self.check_inf else 0),
            ),
            "n4m_filter_quality_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, X=None, y=None):
        self._ensure_handle()
        return self

    def apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        X = as_f64_2d(X)
        mask, stats = _mask_and_stats(X.shape[0])
        check(
            lib.n4m_filter_quality_apply(
                self._ensure_handle(),
                numpy_to_view(X),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_quality_apply",
        )
        return mask, stats

    def fit_apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        return self.fit(X).apply(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_filter_quality_destroy(self._handle)
            self._handle = None


class CompositeFilter(BaseEstimator):
    """Boolean composition of leverage and quality filters."""

    def __init__(self, mode: str | int = "any", filters=()):
        try:
            self._mode_int = _COMPOSITE_MODES[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown CompositeFilter mode: {mode!r}") from exc
        self.mode = mode
        self.filters = list(filters)
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_filter_composite_create(
                ctypes.byref(h), ctypes.c_int32(self._mode_int)
            ),
            "n4m_filter_composite_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
            for filt in self.filters:
                self._add_to_c_handle(filt)
        return self._handle

    def _add_to_c_handle(self, filt):
        if isinstance(filt, HighLeverageFilter):
            check(
                lib.n4m_filter_composite_add_leverage(
                    self._ensure_handle(), filt._ensure_handle()
                ),
                "n4m_filter_composite_add_leverage",
            )
        elif isinstance(filt, SpectralQualityFilter):
            check(
                lib.n4m_filter_composite_add_quality(
                    self._ensure_handle(), filt._ensure_handle()
                ),
                "n4m_filter_composite_add_quality",
            )
        else:
            raise TypeError(
                "CompositeFilter only accepts HighLeverageFilter and "
                "SpectralQualityFilter instances"
            )

    def add(self, filt):
        self.filters.append(filt)
        if self._handle is not None and self._handle.value:
            self._add_to_c_handle(filt)
        return self

    def fit(self, X):
        X = as_f64_2d(X)
        for filt in self.filters:
            if isinstance(filt, HighLeverageFilter) and not filt._fitted:
                filt.fit(X)
            elif isinstance(filt, SpectralQualityFilter):
                filt.fit(X)
        self._ensure_handle()
        return self

    def apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        X = as_f64_2d(X)
        mask, stats = _mask_and_stats(X.shape[0])
        check(
            lib.n4m_filter_composite_apply(
                self._ensure_handle(),
                numpy_to_view(X),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "n4m_filter_composite_apply",
        )
        return mask, stats

    def fit_apply(self, X) -> Tuple[np.ndarray, FilterStats]:
        return self.fit(X).apply(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_filter_composite_destroy(self._handle)
            self._handle = None


__all__ = [
    "CompositeFilter",
    "HighLeverageFilter",
    "SpectralQualityFilter",
    "XOutlierFilter",
    "YOutlierFilter",
]

# SPDX-License-Identifier: CECILL-2.1
"""Advanced sklearn-style wrappers backed by the libn4m C ABI."""
from __future__ import annotations

import ctypes

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, empty_like_f64, numpy_to_view
from ._base import _HandleEstimator


def _as_f64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(arr)


def _check_pair(X_source, X_target) -> tuple[np.ndarray, np.ndarray]:
    source = as_f64_2d(X_source)
    target = as_f64_2d(X_target)
    if source.shape != target.shape:
        raise ValueError("X_source and X_target must have the same shape")
    return source, target


def _f64_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _null_f64_ptr():
    return ctypes.POINTER(ctypes.c_double)()


def _new_handle(prefix: str, *args) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    create = getattr(lib, f"{prefix}_create")
    check(create(ctypes.byref(handle), *args), f"{prefix}_create")
    return handle


class _PairedMatrixOperator(_HandleEstimator):
    """C-backed operator fitted on paired source/target spectra."""

    def fit(self, X_source, X_target):
        source, target = _check_pair(X_source, X_target)
        check(
            getattr(lib, f"{self._C_PREFIX}_fit")(
                self._ensure_handle(), numpy_to_view(source), numpy_to_view(target)
            ),
            f"{self._C_PREFIX}_fit",
        )
        self.n_features_in_ = source.shape[1]
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X_source, X_target) -> np.ndarray:
        return self.fit(X_source, X_target).transform(X_source)


class DirectStandardization(_PairedMatrixOperator):
    """Direct standardization transfer map between paired instruments."""

    _C_PREFIX = "n4m_pp_direct_standardization"

    def __init__(self, fit_intercept: bool = True, ridge: float = 0.0):
        super().__init__()
        self.fit_intercept = bool(fit_intercept)
        self.ridge = float(ridge)

    def _create_handle(self):
        return _new_handle(
            self._C_PREFIX,
            ctypes.c_int(1 if self.fit_intercept else 0),
            ctypes.c_double(self.ridge),
        )


class RobustDirectStandardization(_PairedMatrixOperator):
    """Direct standardization with iterative residual trimming."""

    _C_PREFIX = "n4m_pp_robust_direct_standardization"

    def __init__(
        self,
        fit_intercept: bool = True,
        ridge: float = 0.0,
        trim_quantile: float = 0.9,
        max_iter: int = 3,
    ):
        super().__init__()
        self.fit_intercept = bool(fit_intercept)
        self.ridge = float(ridge)
        self.trim_quantile = float(trim_quantile)
        self.max_iter = int(max_iter)

    def _create_handle(self):
        return _new_handle(
            self._C_PREFIX,
            ctypes.c_int(1 if self.fit_intercept else 0),
            ctypes.c_double(self.ridge),
            ctypes.c_double(self.trim_quantile),
            ctypes.c_int32(self.max_iter),
        )


class PiecewiseDirectStandardization(_PairedMatrixOperator):
    """PDS: local regressions mapping source windows to target wavelengths."""

    _C_PREFIX = "n4m_pp_piecewise_direct_standardization"

    def __init__(self, window_size: int = 5, fit_intercept: bool = True,
                 ridge: float = 0.0):
        super().__init__()
        self.window_size = int(window_size)
        self.fit_intercept = bool(fit_intercept)
        self.ridge = float(ridge)

    def _create_handle(self):
        return _new_handle(
            self._C_PREFIX,
            ctypes.c_int32(self.window_size),
            ctypes.c_int(1 if self.fit_intercept else 0),
            ctypes.c_double(self.ridge),
        )


class ScoreAugmentedProjectionStandardization(_PairedMatrixOperator):
    """Score-augmented projection standardization inspired by SA-PBS."""

    _C_PREFIX = "n4m_pp_saps"

    def __init__(self, n_components: int = 5, score_weight: float = 1.0,
                 fit_intercept: bool = True, ridge: float = 0.0):
        super().__init__()
        self.n_components = int(n_components)
        self.score_weight = float(score_weight)
        self.fit_intercept = bool(fit_intercept)
        self.ridge = float(ridge)

    def _create_handle(self):
        return _new_handle(
            self._C_PREFIX,
            ctypes.c_int32(self.n_components),
            ctypes.c_double(self.score_weight),
            ctypes.c_int(1 if self.fit_intercept else 0),
            ctypes.c_double(self.ridge),
        )


class SlopeBiasCorrection(_HandleEstimator):
    """Linear slope/bias correction for transferred predictions."""

    _C_PREFIX = "n4m_pp_slope_bias"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        return _new_handle(self._C_PREFIX)

    def fit(self, y_source, y_target):
        source = _as_f64_1d(y_source, "y_source")
        target = _as_f64_1d(y_target, "y_target")
        if source.shape != target.shape:
            raise ValueError("y_source and y_target must have the same shape")
        check(
            lib.n4m_pp_slope_bias_fit(
                self._ensure_handle(), _f64_ptr(source), _f64_ptr(target),
                ctypes.c_int64(source.size)
            ),
            "n4m_pp_slope_bias_fit",
        )
        self._fitted = True
        return self

    def transform(self, y):
        if not self._fitted:
            raise RuntimeError("SlopeBiasCorrection must be fitted before transform")
        arr = _as_f64_1d(y, "y")
        out = np.empty_like(arr)
        check(
            lib.n4m_pp_slope_bias_transform(
                self._ensure_handle(), _f64_ptr(arr), ctypes.c_int64(arr.size),
                _f64_ptr(out)
            ),
            "n4m_pp_slope_bias_transform",
        )
        return out

    apply = transform


class LocalCentering(_PairedMatrixOperator):
    """Transfer by subtracting source mean and adding target mean."""

    _C_PREFIX = "n4m_pp_local_centering"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        return _new_handle(self._C_PREFIX)


class WeightedSNV(_HandleEstimator):
    """Weighted standard normal variate normalization."""

    _C_PREFIX = "n4m_pp_weighted_snv"

    def __init__(self, weights=None, ddof: int = 0, eps: float = 1e-12):
        super().__init__()
        self.weights = None if weights is None else _as_f64_1d(weights, "weights")
        self.ddof = int(ddof)
        self.eps = float(eps)

    def _create_handle(self):
        weights = self.weights
        ptr = _null_f64_ptr() if weights is None else _f64_ptr(weights)
        n_weights = 0 if weights is None else weights.size
        return _new_handle(
            self._C_PREFIX, ptr, ctypes.c_int64(n_weights),
            ctypes.c_int32(self.ddof), ctypes.c_double(self.eps)
        )

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            lib.n4m_pp_weighted_snv_fit(self._ensure_handle(), numpy_to_view(X)),
            "n4m_pp_weighted_snv_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class VariableSortingNormalization(WeightedSNV):
    """VSN-style data-derived weighted SNV."""

    _C_PREFIX = "n4m_pp_vsn"

    def __init__(self, eps: float = 1e-12):
        super().__init__(weights=None, eps=eps)

    def _create_handle(self):
        return _new_handle(self._C_PREFIX, ctypes.c_double(self.eps))

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            lib.n4m_pp_vsn_fit(self._ensure_handle(), numpy_to_view(X)),
            "n4m_pp_vsn_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self


class PiecewiseSNV(_HandleEstimator):
    """Apply SNV independently inside fixed wavelength intervals."""

    _C_PREFIX = "n4m_pp_piecewise_snv"

    def __init__(self, window_size: int = 32, ddof: int = 0, eps: float = 1e-12):
        super().__init__()
        self.window_size = int(window_size)
        self.ddof = int(ddof)
        self.eps = float(eps)

    def _create_handle(self):
        return _new_handle(
            self._C_PREFIX,
            ctypes.c_int32(self.window_size),
            ctypes.c_int32(self.ddof),
            ctypes.c_double(self.eps),
        )

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            lib.n4m_pp_piecewise_snv_fit(self._ensure_handle(), numpy_to_view(X)),
            "n4m_pp_piecewise_snv_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class _MSCBase(PiecewiseSNV):
    _C_PREFIX = ""

    def __init__(self, window_size: int = 32, reference=None, eps: float = 1e-12):
        super().__init__(window_size=window_size, ddof=0, eps=eps)
        self.reference = None if reference is None else _as_f64_1d(reference, "reference")

    def _create_handle(self):
        ref = self.reference
        ptr = _null_f64_ptr() if ref is None else _f64_ptr(ref)
        n_ref = 0 if ref is None else ref.size
        return _new_handle(
            self._C_PREFIX, ptr, ctypes.c_int64(n_ref),
            ctypes.c_int32(self.window_size),
            ctypes.c_double(self.eps),
        )

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            getattr(lib, f"{self._C_PREFIX}_fit")(
                self._ensure_handle(), numpy_to_view(X)
            ),
            f"{self._C_PREFIX}_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self


class PiecewiseMSC(_MSCBase):
    """Apply MSC independently inside fixed wavelength intervals."""

    _C_PREFIX = "n4m_pp_piecewise_msc"


class LocalizedMSC(_MSCBase):
    """Feature-wise MSC using a moving local wavelength window."""

    _C_PREFIX = "n4m_pp_localized_msc"


class _AlignmentBase(_HandleEstimator):
    _C_PREFIX = ""
    _default_interval_size = 0
    _default_max_shift = 0

    def __init__(self, reference=None, interval_size: int | None = None,
                 max_shift: int | None = None):
        super().__init__()
        self.reference = None if reference is None else _as_f64_1d(reference, "reference")
        self.interval_size = (
            self._default_interval_size if interval_size is None else int(interval_size)
        )
        self.max_shift = self._default_max_shift if max_shift is None else int(max_shift)

    def _create_handle(self):
        ref = self.reference
        ptr = _null_f64_ptr() if ref is None else _f64_ptr(ref)
        n_ref = 0 if ref is None else ref.size
        return _new_handle(
            self._C_PREFIX, ptr, ctypes.c_int64(n_ref),
            ctypes.c_int32(self.interval_size), ctypes.c_int32(self.max_shift)
        )

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            getattr(lib, f"{self._C_PREFIX}_fit")(
                self._ensure_handle(), numpy_to_view(X)
            ),
            f"{self._C_PREFIX}_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class CrossCorrelationAlignment(_AlignmentBase):
    """Whole-spectrum integer shift chosen by maximum correlation."""

    _C_PREFIX = "n4m_pp_xcorr_align"
    _default_interval_size = 0
    _default_max_shift = 5

    def __init__(self, reference=None, max_shift: int = 5):
        super().__init__(reference=reference, interval_size=0, max_shift=max_shift)


class IcoshiftAlignment(_AlignmentBase):
    """Interval correlation shifting with fixed-size intervals."""

    _C_PREFIX = "n4m_pp_icoshift_align"
    _default_interval_size = 32
    _default_max_shift = 5

    def __init__(self, reference=None, interval_size: int = 32, max_shift: int = 5):
        super().__init__(
            reference=reference, interval_size=interval_size, max_shift=max_shift
        )


class DynamicTimeWarpingAlignment(_AlignmentBase):
    """Dynamic-time-warping alignment to a fixed-length reference."""

    _C_PREFIX = "n4m_pp_dtw_align"
    _default_interval_size = 0
    _default_max_shift = 0

    def __init__(self, reference=None):
        super().__init__(reference=reference, interval_size=0, max_shift=0)


class CorrelationOptimizedWarping(_AlignmentBase):
    """Segment-wise correlation optimized warping approximation."""

    _C_PREFIX = "n4m_pp_cow_align"
    _default_interval_size = 32
    _default_max_shift = 5

    def __init__(self, reference=None, interval_size: int = 32, max_shift: int = 5):
        super().__init__(
            reference=reference, interval_size=interval_size, max_shift=max_shift
        )


class VarianceFilter(_HandleEstimator):
    """Model-agnostic feature filter by variance."""

    _C_PREFIX = "n4m_filter_variance"

    def __init__(self, threshold: float = 0.0, top_k: int | None = None):
        super().__init__()
        self.threshold = float(threshold)
        self.top_k = top_k

    def _create_handle(self):
        top_k = -1 if self.top_k is None else int(self.top_k)
        return _new_handle(
            self._C_PREFIX, ctypes.c_double(self.threshold), ctypes.c_int32(top_k)
        )

    def _output_cols(self) -> int:
        out_cols = ctypes.c_int64()
        check(
            getattr(lib, f"{self._C_PREFIX}_output_cols")(
                self._ensure_handle(), ctypes.byref(out_cols)
            ),
            f"{self._C_PREFIX}_output_cols",
        )
        return int(out_cols.value)

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        check(
            lib.n4m_filter_variance_fit(self._ensure_handle(), numpy_to_view(X)),
            "n4m_filter_variance_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X, (X.shape[0], self._output_cols()))

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_support(self):
        raise NotImplementedError("The C ABI exposes selected output columns, not masks")


class CorrelationFilter(VarianceFilter):
    """Model-agnostic feature filter by absolute correlation to ``y``."""

    _C_PREFIX = "n4m_filter_correlation"

    def fit(self, X, y):
        X = as_f64_2d(X)
        y_arr = _as_f64_1d(y, "y")
        if y_arr.size != X.shape[0]:
            raise ValueError("y length must match X rows")
        check(
            lib.n4m_filter_correlation_fit(
                self._ensure_handle(), numpy_to_view(X), _f64_ptr(y_arr),
                ctypes.c_int64(y_arr.size)
            ),
            "n4m_filter_correlation_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self


class IntervalGenerator(_HandleEstimator):
    """Generate fixed or overlapping wavelength intervals."""

    _C_PREFIX = "n4m_interval_generator"

    def __init__(self, interval_size: int = 32, step: int | None = None):
        super().__init__()
        self.interval_size = int(interval_size)
        self.step = None if step is None else int(step)

    def _create_handle(self):
        step = 0 if self.step is None else self.step
        return _new_handle(
            self._C_PREFIX, ctypes.c_int32(self.interval_size), ctypes.c_int32(step)
        )

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        width = max(1, self.interval_size)
        step = width if self.step is None else max(1, self.step)
        self.intervals_ = [
            (lo, min(X.shape[1], lo + width))
            for lo in range(0, X.shape[1], step)
            if lo < X.shape[1]
        ]
        check(
            lib.n4m_interval_generator_fit(
                self._ensure_handle(), numpy_to_view(X)
            ),
            "n4m_interval_generator_fit",
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def _output_cols(self) -> int:
        out_cols = ctypes.c_int64()
        check(
            lib.n4m_interval_generator_output_cols(
                self._ensure_handle(), ctypes.byref(out_cols)
            ),
            "n4m_interval_generator_output_cols",
        )
        return int(out_cols.value)

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("IntervalGenerator must be fitted before transform")
        X = as_f64_2d(X)
        out = empty_like_f64((X.shape[0], self._output_cols()))
        check(
            lib.n4m_interval_generator_transform(
                self._ensure_handle(), numpy_to_view(X), numpy_to_view(out)
            ),
            "n4m_interval_generator_transform",
        )
        blocks = []
        offset = 0
        for lo, hi in self.intervals_:
            width = hi - lo
            blocks.append(out[:, offset:offset + width])
            offset += width
        return blocks

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


__all__ = [
    "CorrelationFilter",
    "CorrelationOptimizedWarping",
    "CrossCorrelationAlignment",
    "DirectStandardization",
    "DynamicTimeWarpingAlignment",
    "IcoshiftAlignment",
    "IntervalGenerator",
    "LocalCentering",
    "LocalizedMSC",
    "PiecewiseDirectStandardization",
    "PiecewiseMSC",
    "PiecewiseSNV",
    "RobustDirectStandardization",
    "ScoreAugmentedProjectionStandardization",
    "SlopeBiasCorrection",
    "VariableSortingNormalization",
    "VarianceFilter",
    "WeightedSNV",
]

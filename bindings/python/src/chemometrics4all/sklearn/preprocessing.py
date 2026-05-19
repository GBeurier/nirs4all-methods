# SPDX-License-Identifier: CECILL-2.1
"""Stateless and stateful preprocessing wrappers."""
from __future__ import annotations

import ctypes

from .._errors import check
from .._ffi import lib
from ._base import StatefulOperator, StatelessOperator

# Boundary-mode lookups for the operators that accept a string mode.
_SAVGOL_MODES = {"mirror": 0, "constant": 1, "nearest": 2, "wrap": 3, "interp": 4}
_LSNV_PAD_MODES = {"reflect": 0, "edge": 1, "constant": 2}
_AREA_METHODS = {"sum": 0, "abs_sum": 1, "trapz": 2}
_GAUSSIAN_MODES = {"reflect": 0, "constant": 1, "nearest": 2, "mirror": 3, "wrap": 4}


class SNV(StatelessOperator):
    """Standard Normal Variate normalisation."""

    _C_PREFIX = "c4a_pp_snv"

    def __init__(self, with_mean: bool = True, with_std: bool = True, ddof: int = 0):
        super().__init__()
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.ddof = int(ddof)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_snv_create(
                ctypes.byref(h),
                ctypes.c_int(1 if self.with_mean else 0),
                ctypes.c_int(1 if self.with_std else 0),
                ctypes.c_int(self.ddof),
            ),
            "c4a_pp_snv_create",
        )
        return h


class LSNV(StatelessOperator):
    """Sliding-window (local) SNV."""

    _C_PREFIX = "c4a_pp_lsnv"

    def __init__(self, window: int = 11, pad_mode: str = "reflect",
                 constant_value: float = 0.0):
        super().__init__()
        self.window = int(window)
        self.pad_mode = str(pad_mode)
        self.constant_value = float(constant_value)

    def _create_handle(self):
        h = ctypes.c_void_p()
        try:
            mode = _LSNV_PAD_MODES[self.pad_mode]
        except KeyError as exc:
            raise ValueError(f"Unknown LSNV pad mode: {self.pad_mode}") from exc
        check(
            lib.c4a_pp_lsnv_create(
                ctypes.byref(h),
                ctypes.c_int32(self.window),
                ctypes.c_int32(mode),
                ctypes.c_double(self.constant_value),
            ),
            "c4a_pp_lsnv_create",
        )
        return h


class RNV(StatelessOperator):
    """Robust SNV using median + k * MAD."""

    _C_PREFIX = "c4a_pp_rnv"

    def __init__(self, with_center: bool = True, with_scale: bool = True,
                 k: float = 1.4826):
        super().__init__()
        self.with_center = bool(with_center)
        self.with_scale = bool(with_scale)
        self.k = float(k)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_rnv_create(
                ctypes.byref(h),
                ctypes.c_int(1 if self.with_center else 0),
                ctypes.c_int(1 if self.with_scale else 0),
                ctypes.c_double(self.k),
            ),
            "c4a_pp_rnv_create",
        )
        return h


class AreaNormalization(StatelessOperator):
    """Per-row area normalisation."""

    _C_PREFIX = "c4a_pp_area"

    def __init__(self, method: str = "sum"):
        super().__init__()
        self.method = str(method)

    def _create_handle(self):
        h = ctypes.c_void_p()
        try:
            method = _AREA_METHODS[self.method]
        except KeyError as exc:
            raise ValueError(f"Unknown AreaNormalization method: {self.method}") from exc
        check(
            lib.c4a_pp_area_create(ctypes.byref(h), ctypes.c_int32(method)),
            "c4a_pp_area_create",
        )
        return h


class Normalize(StatelessOperator):
    """Column-wise normalisation."""

    _C_PREFIX = "c4a_pp_normalize"

    def __init__(self, feature_min: float = -1.0, feature_max: float = 1.0):
        super().__init__()
        self.feature_min = float(feature_min)
        self.feature_max = float(feature_max)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_normalize_create(
                ctypes.byref(h),
                ctypes.c_double(self.feature_min),
                ctypes.c_double(self.feature_max),
            ),
            "c4a_pp_normalize_create",
        )
        return h


class SimpleScale(StatelessOperator):
    """Column-wise min-max scaling to ``[0, 1]``."""

    _C_PREFIX = "c4a_pp_simple_scale"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_simple_scale_create(ctypes.byref(h)),
            "c4a_pp_simple_scale_create",
        )
        return h


class LogTransform(StatefulOperator):
    """Element-wise logarithm with optional fit-time auto-offset."""

    _C_PREFIX = "c4a_pp_log"

    def __init__(
        self,
        base: float = 0.0,
        offset: float = 0.0,
        auto_offset: bool = True,
        min_value: float = 1e-8,
    ):
        super().__init__()
        self.base = float(base)
        self.offset = float(offset)
        self.auto_offset = bool(auto_offset)
        self.min_value = float(min_value)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_log_create(
                ctypes.byref(h),
                ctypes.c_double(self.base),
                ctypes.c_double(self.offset),
                ctypes.c_int(1 if self.auto_offset else 0),
                ctypes.c_double(self.min_value),
            ),
            "c4a_pp_log_create",
        )
        return h


class MSC(StatefulOperator):
    """Multiplicative Scatter Correction.

    Fit learns the per-column linear coefficients against the per-row mean of
    the training matrix.
    """

    _C_PREFIX = "c4a_pp_msc"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(lib.c4a_pp_msc_create(ctypes.byref(h)), "c4a_pp_msc_create")
        return h


class EMSC(StatefulOperator):
    """Extended Multiplicative Scatter Correction (polynomial)."""

    _C_PREFIX = "c4a_pp_emsc"

    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = int(degree)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_emsc_create(ctypes.byref(h), ctypes.c_int32(self.degree)),
            "c4a_pp_emsc_create",
        )
        return h


class BaselineCenter(StatefulOperator):
    """Column-mean baseline centering."""

    _C_PREFIX = "c4a_pp_baseline"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(lib.c4a_pp_baseline_create(ctypes.byref(h)), "c4a_pp_baseline_create")
        return h


class Derivate(StatefulOperator):
    """Finite-difference derivative along the wavelength axis."""

    _C_PREFIX = "c4a_pp_derivate"

    def __init__(self, order: int = 1, delta: float = 1.0):
        super().__init__()
        self.order = int(order)
        self.delta = float(delta)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_derivate_create(
                ctypes.byref(h),
                ctypes.c_int32(self.order),
                ctypes.c_double(self.delta),
            ),
            "c4a_pp_derivate_create",
        )
        return h

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        from .._matrix import as_f64_2d

        X = as_f64_2d(X)
        out_cols = int(lib.c4a_pp_derivate_output_cols(
            ctypes.c_int32(self.order), ctypes.c_int64(X.shape[1])
        ))
        if out_cols <= 0:
            raise ValueError(
                "Derivate output has no columns; order must be smaller than input width"
            )
        return self._call_transform(X, (X.shape[0], out_cols))


class SavitzkyGolay(StatelessOperator):
    """scipy.signal.savgol_filter parity."""

    _C_PREFIX = "c4a_pp_savgol"

    def __init__(self, window_length: int = 5, polyorder: int = 2,
                 deriv: int = 0, delta: float = 1.0,
                 mode: str = "mirror", cval: float = 0.0):
        super().__init__()
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)
        self.mode = str(mode)
        self.cval = float(cval)

    def _create_handle(self):
        h = ctypes.c_void_p()
        try:
            mode_int = _SAVGOL_MODES[self.mode]
        except KeyError as exc:
            raise ValueError(f"Unknown SavitzkyGolay mode: {self.mode}") from exc
        check(
            lib.c4a_pp_savgol_create(
                ctypes.byref(h),
                ctypes.c_int32(self.window_length),
                ctypes.c_int32(self.polyorder),
                ctypes.c_int32(self.deriv),
                ctypes.c_double(self.delta),
                ctypes.c_int(mode_int),
                ctypes.c_double(self.cval),
            ),
            "c4a_pp_savgol_create",
        )
        return h


class NorrisWilliams(StatelessOperator):
    """Segment smoothing followed by gap finite differences."""

    _C_PREFIX = "c4a_pp_norris_williams"

    def __init__(
        self,
        segment: int = 5,
        gap: int = 5,
        derivative_order: int = 1,
        delta: float = 1.0,
    ):
        super().__init__()
        self.segment = int(segment)
        self.gap = int(gap)
        self.derivative_order = int(derivative_order)
        self.delta = float(delta)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_norris_williams_create(
                ctypes.byref(h),
                ctypes.c_int32(self.segment),
                ctypes.c_int32(self.gap),
                ctypes.c_int32(self.derivative_order),
                ctypes.c_double(self.delta),
            ),
            "c4a_pp_norris_williams_create",
        )
        return h


class Gaussian(StatelessOperator):
    """SciPy-compatible 1-D Gaussian filter along the wavelength axis."""

    _C_PREFIX = "c4a_pp_gaussian"

    def __init__(
        self,
        sigma: float = 1.0,
        order: int = 0,
        mode: str = "reflect",
        cval: float = 0.0,
        truncate: float = 4.0,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.order = int(order)
        self.mode = str(mode)
        self.cval = float(cval)
        self.truncate = float(truncate)

    def _create_handle(self):
        h = ctypes.c_void_p()
        try:
            mode = _GAUSSIAN_MODES[self.mode]
        except KeyError as exc:
            raise ValueError(f"Unknown Gaussian mode: {self.mode}") from exc
        check(
            lib.c4a_pp_gaussian_create(
                ctypes.byref(h),
                ctypes.c_double(self.sigma),
                ctypes.c_int32(self.order),
                ctypes.c_int(mode),
                ctypes.c_double(self.cval),
                ctypes.c_double(self.truncate),
            ),
            "c4a_pp_gaussian_create",
        )
        return h


class FirstDerivative(StatelessOperator):
    """``np.gradient(X, delta, axis=1, edge_order=...)`` (shape-preserving)."""

    _C_PREFIX = "c4a_pp_first_derivative"

    def __init__(self, delta: float = 1.0, edge_order: int = 2):
        super().__init__()
        self.delta = float(delta)
        self.edge_order = int(edge_order)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_first_derivative_create(
                ctypes.byref(h),
                ctypes.c_double(self.delta),
                ctypes.c_int32(self.edge_order),
            ),
            "c4a_pp_first_derivative_create",
        )
        return h


class SecondDerivative(StatelessOperator):
    """Two passes of ``np.gradient`` (shape-preserving)."""

    _C_PREFIX = "c4a_pp_second_derivative"

    def __init__(self, delta: float = 1.0, edge_order: int = 2):
        super().__init__()
        self.delta = float(delta)
        self.edge_order = int(edge_order)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_second_derivative_create(
                ctypes.byref(h),
                ctypes.c_double(self.delta),
                ctypes.c_int32(self.edge_order),
            ),
            "c4a_pp_second_derivative_create",
        )
        return h


class ToAbsorbance(StatelessOperator):
    """A = -log10(max(R, epsilon)). Optional %-scaling."""

    _C_PREFIX = "c4a_pp_to_absorbance"

    def __init__(self, is_percent: bool = False, epsilon: float = 1e-10,
                 clip_negative: bool = True):
        super().__init__()
        self.is_percent = bool(is_percent)
        self.epsilon = float(epsilon)
        self.clip_negative = bool(clip_negative)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_to_absorbance_create(
                ctypes.byref(h),
                ctypes.c_int(1 if self.is_percent else 0),
                ctypes.c_double(self.epsilon),
                ctypes.c_int(1 if self.clip_negative else 0),
            ),
            "c4a_pp_to_absorbance_create",
        )
        return h


class FromAbsorbance(StatelessOperator):
    """R = 10**(-A), optionally returned as percent."""

    _C_PREFIX = "c4a_pp_from_absorbance"

    def __init__(self, is_percent: bool = False):
        super().__init__()
        self.is_percent = bool(is_percent)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_from_absorbance_create(
                ctypes.byref(h), ctypes.c_int(1 if self.is_percent else 0)
            ),
            "c4a_pp_from_absorbance_create",
        )
        return h


class PercentToFraction(StatelessOperator):
    """Convert percent reflectance/transmittance to fraction."""

    _C_PREFIX = "c4a_pp_pct_to_frac"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_pct_to_frac_create(ctypes.byref(h)),
            "c4a_pp_pct_to_frac_create",
        )
        return h


class FractionToPercent(StatelessOperator):
    """Convert fractional reflectance/transmittance to percent."""

    _C_PREFIX = "c4a_pp_frac_to_pct"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_frac_to_pct_create(ctypes.byref(h)),
            "c4a_pp_frac_to_pct_create",
        )
        return h


class KubelkaMunk(StatelessOperator):
    """KM = (1 - R)^2 / (2 R), with R guarded by epsilon."""

    _C_PREFIX = "c4a_pp_kubelka_munk"

    def __init__(self, is_percent: bool = False, epsilon: float = 1e-10):
        super().__init__()
        self.is_percent = bool(is_percent)
        self.epsilon = float(epsilon)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_kubelka_munk_create(
                ctypes.byref(h),
                ctypes.c_int(1 if self.is_percent else 0),
                ctypes.c_double(self.epsilon),
            ),
            "c4a_pp_kubelka_munk_create",
        )
        return h


__all__ = [
    "AreaNormalization",
    "BaselineCenter",
    "Derivate",
    "EMSC",
    "FirstDerivative",
    "FractionToPercent",
    "FromAbsorbance",
    "Gaussian",
    "KubelkaMunk",
    "LSNV",
    "LogTransform",
    "MSC",
    "Normalize",
    "NorrisWilliams",
    "PercentToFraction",
    "RNV",
    "SNV",
    "SavitzkyGolay",
    "SecondDerivative",
    "SimpleScale",
    "ToAbsorbance",
]

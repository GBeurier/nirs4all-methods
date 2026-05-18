# SPDX-License-Identifier: CECILL-2.1
"""Stateless and stateful preprocessing wrappers.

Covers the ten preprocessing operators included in Gate 1:

* :class:`SNV` — Standard Normal Variate (stateless).
* :class:`LSNV` — Local Standard Normal Variate (sliding window, stateless).
* :class:`RNV` — Robust Standard Normal Variate (median + k·MAD, stateless).
* :class:`MSC` — Multiplicative Scatter Correction (stateful).
* :class:`EMSC` — Extended Multiplicative Scatter Correction (stateful).
* :class:`SavitzkyGolay` — polynomial smoothing / differentiation (stateless).
* :class:`FirstDerivative`, :class:`SecondDerivative` — np.gradient (stateless).
* :class:`ToAbsorbance` — A = -log10(R) (stateless).
* :class:`KubelkaMunk` — (1-R)^2 / 2R (stateless).
"""
from __future__ import annotations

import ctypes

from .._errors import check
from .._ffi import lib
from ._base import StatefulOperator, StatelessOperator

# Boundary-mode lookups for the operators that accept a string mode.
_SAVGOL_MODES = {"mirror": 0, "constant": 1, "nearest": 2, "wrap": 3, "interp": 4}
_LSNV_PAD_MODES = {"reflect": 0, "edge": 1, "constant": 2}


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
    "EMSC",
    "FirstDerivative",
    "KubelkaMunk",
    "LSNV",
    "MSC",
    "RNV",
    "SNV",
    "SavitzkyGolay",
    "SecondDerivative",
    "ToAbsorbance",
]

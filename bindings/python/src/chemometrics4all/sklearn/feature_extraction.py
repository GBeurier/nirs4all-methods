# SPDX-License-Identifier: CECILL-2.1
"""Orthogonalization and feature-extraction wrappers."""
from __future__ import annotations

import ctypes
from collections.abc import Sequence

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from ._base import StatefulOperator, StatelessOperator


def _as_f64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


class OSC(StatefulOperator):
    """Orthogonal Signal Correction."""

    _C_PREFIX = "c4a_pp_osc"

    def __init__(self, n_components: int = 1, scale: bool = True):
        super().__init__()
        self.n_components = int(n_components)
        self.scale = bool(scale)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_osc_create(
                ctypes.byref(h),
                ctypes.c_int32(self.n_components),
                ctypes.c_int(1 if self.scale else 0),
            ),
            "c4a_pp_osc_create",
        )
        return h

    def fit(self, X, y=None):  # noqa: D401
        if y is None:
            raise ValueError("OSC.fit requires y")
        X = as_f64_2d(X)
        y_arr = _as_f64_1d(y, "y")
        check(
            lib.c4a_pp_osc_fit(
                self._ensure_handle(),
                numpy_to_view(X),
                y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y_arr.size),
            ),
            "c4a_pp_osc_fit",
        )
        self._fitted = True
        return self


class EPO(StatefulOperator):
    """External Parameter Orthogonalisation."""

    _C_PREFIX = "c4a_pp_epo"

    def __init__(self, scale: bool = True):
        super().__init__()
        self.scale = bool(scale)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_epo_create(ctypes.byref(h), ctypes.c_int(1 if self.scale else 0)),
            "c4a_pp_epo_create",
        )
        return h

    def fit(self, X, d=None):  # noqa: D401
        if d is None:
            raise ValueError("EPO.fit requires d")
        X = as_f64_2d(X)
        d_arr = _as_f64_1d(d, "d")
        check(
            lib.c4a_pp_epo_fit(
                self._ensure_handle(),
                numpy_to_view(X),
                d_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(d_arr.size),
            ),
            "c4a_pp_epo_fit",
        )
        self._fitted = True
        return self

    def fit_transform(self, X, d=None) -> np.ndarray:
        """Fit and apply the EPO projection using the external parameter ``d``."""
        if d is None:
            raise ValueError("EPO.fit_transform requires d")
        X_arr = as_f64_2d(X)
        d_arr = _as_f64_1d(d, "d")
        return self.fit(X_arr, d_arr).transform_with_d(X_arr, d_arr)

    def transform_with_d(self, X, d) -> np.ndarray:
        """Apply the fitted EPO projection when ``d`` is available."""
        if not self._fitted:
            raise RuntimeError("EPO must be fitted before transform_with_d")
        X_arr = as_f64_2d(X)
        d_arr = _as_f64_1d(d, "d")
        if d_arr.size != X_arr.shape[0]:
            raise ValueError("d length must match X rows")
        out = np.empty_like(X_arr)
        check(
            lib.c4a_pp_epo_transform_with_d(
                self._ensure_handle(),
                numpy_to_view(X_arr),
                d_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(d_arr.size),
                numpy_to_view(out),
            ),
            "c4a_pp_epo_transform_with_d",
        )
        return out


class _FlexibleProjection(StatefulOperator):
    """Shared output-width handling for FlexiblePCA/SVD."""

    def __init__(self, n_components: float = 5.0):
        super().__init__()
        self.n_components = float(n_components)

    def _create_handle(self):
        h = ctypes.c_void_p()
        create = getattr(lib, f"{self._C_PREFIX}_create")
        check(
            create(ctypes.byref(h), ctypes.c_double(self.n_components)),
            f"{self._C_PREFIX}_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        out_cols = ctypes.c_int64()
        output_cols = getattr(lib, f"{self._C_PREFIX}_output_cols")
        check(output_cols(self._ensure_handle(), ctypes.byref(out_cols)),
              f"{self._C_PREFIX}_output_cols")
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class FlexiblePCA(_FlexibleProjection):
    """PCA with integer or explained-variance component selection."""

    _C_PREFIX = "c4a_pp_flex_pca"


class FlexibleSVD(_FlexibleProjection):
    """SVD with integer component selection."""

    _C_PREFIX = "c4a_pp_flex_svd"


class FCKStaticTransformer(StatelessOperator):
    """Static fractional convolutional kernel bank transformer."""

    _C_PREFIX = "c4a_pp_fck_static"

    def __init__(
        self,
        kernel_size: int,
        alphas: Sequence[float] | None = None,
        sigmas: Sequence[float] | None = None,
        *,
        filter_orders: Sequence[float] | None = None,
        filter_scales: Sequence[float] | None = None,
    ):
        super().__init__()
        if alphas is None:
            alphas = filter_orders
        if sigmas is None:
            sigmas = filter_scales
        if alphas is None or sigmas is None:
            raise ValueError("alphas/filter_orders and sigmas/filter_scales are required")
        self.kernel_size = int(kernel_size)
        self.alphas = np.ascontiguousarray(alphas, dtype=np.float64)
        self.sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_fck_static_create(
                ctypes.byref(h),
                ctypes.c_int32(self.kernel_size),
                self.alphas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int32(self.alphas.size),
                self.sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int32(self.sigmas.size),
            ),
            "c4a_pp_fck_static_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        out_cols = ctypes.c_int32()
        check(
            lib.c4a_pp_fck_static_output_cols(
                ctypes.c_int32(self.alphas.size * self.sigmas.size),
                ctypes.c_int32(X.shape[1]),
                ctypes.byref(out_cols),
            ),
            "c4a_pp_fck_static_output_cols",
        )
        self._ensure_handle()
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


__all__ = [
    "EPO",
    "FCKStaticTransformer",
    "FlexiblePCA",
    "FlexibleSVD",
    "OSC",
]

# SPDX-License-Identifier: CECILL-2.1
"""Resampling, cropping and discretization wrappers."""
from __future__ import annotations

import ctypes
from collections.abc import Sequence

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, empty_like_i32, numpy_to_view
from ._base import StatefulOperator, StatelessOperator
from ._compat import BaseEstimator, TransformerMixin


class CropTransformer(StatelessOperator):
    """Slice wavelength columns in the half-open interval ``[start, end)``."""

    _C_PREFIX = "c4a_pp_crop"

    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = int(start)
        self.end = int(end)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_crop_create(
                ctypes.byref(h), ctypes.c_int64(self.start), ctypes.c_int64(self.end)
            ),
            "c4a_pp_crop_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = int(lib.c4a_pp_crop_output_cols(handle, ctypes.c_int64(X.shape[1])))
        if out_cols <= 0:
            raise ValueError("CropTransformer output has no columns")
        return self._call_transform(X, (X.shape[0], out_cols))


class ResampleTransformer(StatelessOperator):
    """Resize spectra to a fixed column count."""

    _C_PREFIX = "c4a_pp_resample"

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = int(num_samples)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_resample_create(ctypes.byref(h), ctypes.c_int64(self.num_samples)),
            "c4a_pp_resample_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = int(lib.c4a_pp_resample_output_cols(
            handle, ctypes.c_int64(X.shape[1])
        ))
        if out_cols <= 0:
            raise ValueError("ResampleTransformer output has no columns")
        return self._call_transform(X, (X.shape[0], out_cols))


class Resampler(BaseEstimator, TransformerMixin):
    """Interpolate spectra from a fitted source wavelength grid to a target grid."""

    _C_PREFIX = "c4a_pp_resampler"

    def __init__(
        self,
        target_wavelengths: Sequence[float] | None = None,
        method: int = 0,
        crop_min: float = 0.0,
        crop_max: float = 0.0,
        use_crop: bool = False,
        fill_value: float = 0.0,
        bounds_error: bool = False,
        extrapolate: bool = False,
        *,
        tgt_min: float | None = None,
        tgt_step: float | None = None,
        tgt_n: int | None = None,
    ):
        if target_wavelengths is None:
            if tgt_min is None or tgt_step is None or tgt_n is None:
                raise ValueError(
                    "target_wavelengths or tgt_min/tgt_step/tgt_n must be provided"
                )
            target_wavelengths = tgt_min + tgt_step * np.arange(int(tgt_n))
        self.target_wavelengths = np.ascontiguousarray(target_wavelengths, dtype=np.float64)
        self.method = int(method)
        self.crop_min = float(crop_min)
        self.crop_max = float(crop_max)
        self.use_crop = bool(use_crop)
        self.fill_value = float(fill_value)
        self.bounds_error = bool(bounds_error)
        self.extrapolate = bool(extrapolate)
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_resampler_create(
                ctypes.byref(h),
                self.target_wavelengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(self.target_wavelengths.size),
                ctypes.c_int32(self.method),
                ctypes.c_double(self.crop_min),
                ctypes.c_double(self.crop_max),
                ctypes.c_int(1 if self.use_crop else 0),
                ctypes.c_double(self.fill_value),
                ctypes.c_int(1 if self.bounds_error else 0),
                ctypes.c_int(1 if self.extrapolate else 0),
            ),
            "c4a_pp_resampler_create",
        )
        return h

    def _ensure_handle(self) -> ctypes.c_void_p:
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, X=None, y=None, *, source_wavelengths=None):
        if source_wavelengths is None:
            source_wavelengths = X
        source = np.ascontiguousarray(source_wavelengths, dtype=np.float64)
        if source.ndim != 1:
            raise ValueError("source_wavelengths must be a 1-D array")
        check(
            lib.c4a_pp_resampler_fit(
                self._ensure_handle(),
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(source.size),
            ),
            "c4a_pp_resampler_fit",
        )
        self.source_wavelengths_ = source
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Resampler must be fitted before transform")
        X = as_f64_2d(X)
        out_cols = int(lib.c4a_pp_resampler_output_cols(self._ensure_handle()))
        if out_cols <= 0:
            raise ValueError("Resampler output has no columns")
        out = np.empty((X.shape[0], out_cols), dtype=np.float64, order="C")
        check(
            lib.c4a_pp_resampler_transform(
                self._ensure_handle(), numpy_to_view(X), numpy_to_view(out)
            ),
            "c4a_pp_resampler_transform",
        )
        return out

    def fit_transform(self, X, y=None, *, source_wavelengths=None) -> np.ndarray:
        self.fit(source_wavelengths if source_wavelengths is not None else X)
        return self.transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_pp_resampler_destroy(self._handle)
            self._handle = None


class IntegerKBinsDiscretizer(StatefulOperator):
    """Per-column integer binning using uniform or quantile edges."""

    _C_PREFIX = "c4a_pp_kbins_disc"

    def __init__(self, n_bins: int = 5, strategy: str | int = 0):
        super().__init__()
        self.n_bins = int(n_bins)
        if isinstance(strategy, str):
            strategy = {"uniform": 0, "quantile": 1}[strategy]
        self.strategy = int(strategy)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_kbins_disc_create(
                ctypes.byref(h), ctypes.c_int32(self.n_bins), ctypes.c_int32(self.strategy)
            ),
            "c4a_pp_kbins_disc_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        out = empty_like_i32(X.shape)
        check(
            lib.c4a_pp_kbins_disc_transform(
                self._ensure_handle(), numpy_to_view(X), numpy_to_view(out)
            ),
            "c4a_pp_kbins_disc_transform",
        )
        return out


class RangeDiscretizer(StatelessOperator):
    """Integer binning against monotonic numeric edges."""

    _C_PREFIX = "c4a_pp_range_disc"

    def __init__(self, edges: Sequence[float] | None = None, edges_csv: str | None = None):
        super().__init__()
        if edges is None:
            if edges_csv is None:
                raise ValueError("edges or edges_csv must be provided")
            edges = [float(part) for part in edges_csv.split(",") if part.strip()]
        self.edges = np.ascontiguousarray(edges, dtype=np.float64)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_range_disc_create(
                ctypes.byref(h),
                self.edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(self.edges.size),
            ),
            "c4a_pp_range_disc_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        out = empty_like_i32(X.shape)
        check(
            lib.c4a_pp_range_disc_transform(
                self._ensure_handle(), numpy_to_view(X), numpy_to_view(out)
            ),
            "c4a_pp_range_disc_transform",
        )
        return out


Crop = CropTransformer
Resample = ResampleTransformer
KBinsDiscretizer = IntegerKBinsDiscretizer

__all__ = [
    "Crop",
    "CropTransformer",
    "IntegerKBinsDiscretizer",
    "KBinsDiscretizer",
    "RangeDiscretizer",
    "Resample",
    "ResampleTransformer",
    "Resampler",
]

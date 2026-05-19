# SPDX-License-Identifier: CECILL-2.1
"""Augmentation / wavelet wrappers."""
from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, empty_like_f64, numpy_to_view
from .._rng import PCG64
from ._base import StatefulOperator, StatelessOperator

# c4a_pp_wavelet_family_t / boundary / threshold / noise enums.
_WAVELET_FAMILIES = {"haar": 0, "db4": 1, "sym4": 2, "coif1": 3}
_WAVELET_BOUNDARIES = {"periodization": 0, "symmetric": 1, "zero": 2}
_WAVELET_THRESHOLDS = {"soft": 0, "hard": 1}
_WAVELET_NOISE = {"median": 0, "std": 1}
_WAVELET_FEATURE_ENTROPY = {"energy": 0, "histogram": 1}


def _wavelet_ids(family: str, mode: str) -> tuple[int, int]:
    try:
        return _WAVELET_FAMILIES[family], _WAVELET_BOUNDARIES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown wavelet param: {exc.args[0]!r}") from exc


class Wavelet(StatelessOperator):
    """Single-level DWT coefficient transform."""

    _C_PREFIX = "c4a_pp_wavelet"

    def __init__(self, family: str = "haar", mode: str = "periodization"):
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self._family, self._mode = _wavelet_ids(self.family, self.mode)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_wavelet_create(
                ctypes.byref(h), ctypes.c_int(self._family), ctypes.c_int(self._mode)
            ),
            "c4a_pp_wavelet_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = ctypes.c_int64()
        check(
            lib.c4a_pp_wavelet_output_cols(
                handle, ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "c4a_pp_wavelet_output_cols",
        )
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class Haar(StatelessOperator):
    """Single-level Haar DWT coefficient transform."""

    _C_PREFIX = "c4a_pp_haar"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(lib.c4a_pp_haar_create(ctypes.byref(h)), "c4a_pp_haar_create")
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        out_cols = ctypes.c_int64()
        check(
            lib.c4a_pp_haar_output_cols(
                ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "c4a_pp_haar_output_cols",
        )
        self._ensure_handle()
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class WaveletDenoise:
    """Multi-level DWT VisuShrink denoising.

    Stateless: matches PyWavelets' ``waverec(threshold(coeffs))`` pipeline.
    """

    _C_PREFIX = "c4a_pp_wavelet_denoise"

    def __init__(self, family: str = "db4", mode: str = "periodization",
                 level: int = 3, threshold_mode: str = "soft",
                 noise_estimator: str = "median"):
        try:
            self._family = _WAVELET_FAMILIES[family]
            self._mode = _WAVELET_BOUNDARIES[mode]
            self._thr = _WAVELET_THRESHOLDS[threshold_mode]
            self._noise = _WAVELET_NOISE[noise_estimator]
        except KeyError as exc:
            raise ValueError(f"Unknown wavelet param: {exc.args[0]!r}") from exc
        self.family = family
        self.mode = mode
        self.level = int(level)
        self.threshold_mode = threshold_mode
        self.noise_estimator = noise_estimator
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_wavelet_denoise_create(
                ctypes.byref(h),
                ctypes.c_int(self._family),
                ctypes.c_int(self._mode),
                ctypes.c_int32(self.level),
                ctypes.c_int(self._thr),
                ctypes.c_int(self._noise),
            ),
            "c4a_pp_wavelet_denoise_create",
        )
        return h

    def fit(self, X, y=None):
        # Stateless — fit is a no-op that materialises the handle.
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        X = as_f64_2d(X)
        out = empty_like_f64(X.shape)
        check(
            lib.c4a_pp_wavelet_denoise_transform(
                self._handle, numpy_to_view(X), numpy_to_view(out)
            ),
            "c4a_pp_wavelet_denoise_transform",
        )
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_pp_wavelet_denoise_destroy(self._handle)
            self._handle = None


class WaveletFeatures(StatelessOperator):
    """Multi-level DWT summary features."""

    _C_PREFIX = "c4a_pp_wavelet_features"

    def __init__(self, family: str = "haar", mode: str = "periodization",
                 max_level: int = 3, entropy: str = "energy"):
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.max_level = int(max_level)
        self.entropy = str(entropy)
        self._family, self._mode = _wavelet_ids(self.family, self.mode)
        try:
            self._entropy = _WAVELET_FEATURE_ENTROPY[self.entropy]
        except KeyError as exc:
            raise ValueError(f"Unknown wavelet entropy: {exc.args[0]!r}") from exc

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_wavelet_features_create_ex(
                ctypes.byref(h),
                ctypes.c_int(self._family),
                ctypes.c_int(self._mode),
                ctypes.c_int32(self.max_level),
                ctypes.c_int(self._entropy),
            ),
            "c4a_pp_wavelet_features_create_ex",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = ctypes.c_int64()
        check(
            lib.c4a_pp_wavelet_features_output_cols(
                handle, ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "c4a_pp_wavelet_features_output_cols",
        )
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class _WaveletProjection(StatefulOperator):
    """Shared WaveletPCA / WaveletSVD lifecycle."""

    def __init__(self, family: str, mode: str, max_level: int, n_components: float):
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.max_level = int(max_level)
        self.n_components = float(n_components)
        self._family, self._mode = _wavelet_ids(self.family, self.mode)

    def _create_handle(self):
        h = ctypes.c_void_p()
        create = getattr(lib, f"{self._C_PREFIX}_create")
        check(
            create(
                ctypes.byref(h),
                ctypes.c_int(self._family),
                ctypes.c_int(self._mode),
                ctypes.c_int32(self.max_level),
                ctypes.c_double(self.n_components),
            ),
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


class WaveletPCA(_WaveletProjection):
    """DWT coefficient projection through PCA scores."""

    _C_PREFIX = "c4a_pp_wavelet_pca"

    def __init__(self, family: str = "haar", mode: str = "periodization",
                 max_level: int = 2, n_components: float = 5.0):
        super().__init__(family, mode, max_level, n_components)


class WaveletSVD(_WaveletProjection):
    """DWT coefficient projection through SVD scores."""

    _C_PREFIX = "c4a_pp_wavelet_svd"

    def __init__(self, family: str = "haar", mode: str = "periodization",
                 max_level: int = 2, n_components: float = 5.0):
        super().__init__(family, mode, max_level, n_components)


class GaussianAdditiveNoise:
    """Add IID Gaussian noise to each element of ``X``.

    The augmenter holds a borrowed reference to a :class:`PCG64` handle; the
    caller is responsible for seeding the RNG and keeping it alive.
    """

    _C_PREFIX = "c4a_aug_gaussian_noise"

    def __init__(self, sigma: float = 0.01, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.sigma = float(sigma)
        if rng is None:
            rng = PCG64(seed)
        self._rng = rng
        self.seed = int(seed)
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_aug_gaussian_noise_create(
                ctypes.byref(h),
                self._rng.handle,
                ctypes.c_double(self.sigma),
            ),
            "c4a_aug_gaussian_noise_create",
        )
        return h

    def fit(self, X, y=None):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        X = as_f64_2d(X)
        out = empty_like_f64(X.shape)
        check(
            lib.c4a_aug_gaussian_noise_apply(
                self._handle, numpy_to_view(X), numpy_to_view(out)
            ),
            "c4a_aug_gaussian_noise_apply",
        )
        return out

    # Alias for sklearn parity.
    apply = transform

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_aug_gaussian_noise_destroy(self._handle)
            self._handle = None


__all__ = [
    "GaussianAdditiveNoise",
    "Haar",
    "Wavelet",
    "WaveletDenoise",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
]

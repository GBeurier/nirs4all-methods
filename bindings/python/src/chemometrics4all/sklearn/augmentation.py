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

# c4a_pp_wavelet_family_t / boundary / threshold / noise enums.
_WAVELET_FAMILIES = {"haar": 0, "db4": 1, "sym4": 2, "coif1": 3}
_WAVELET_BOUNDARIES = {"periodization": 0, "symmetric": 1, "zero": 2}
_WAVELET_THRESHOLDS = {"soft": 0, "hard": 1}
_WAVELET_NOISE = {"median": 0, "std": 1}


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


__all__ = ["GaussianAdditiveNoise", "WaveletDenoise"]

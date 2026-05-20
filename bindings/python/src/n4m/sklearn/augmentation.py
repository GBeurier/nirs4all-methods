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
from ._compat import BaseEstimator, TransformerMixin

# n4m_pp_wavelet_family_t / boundary / threshold / noise enums.
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

    _C_PREFIX = "n4m_pp_wavelet"

    def __init__(self, family: str = "haar", mode: str = "periodization"):
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self._family, self._mode = _wavelet_ids(self.family, self.mode)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_pp_wavelet_create(
                ctypes.byref(h), ctypes.c_int(self._family), ctypes.c_int(self._mode)
            ),
            "n4m_pp_wavelet_create",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = ctypes.c_int64()
        check(
            lib.n4m_pp_wavelet_output_cols(
                handle, ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "n4m_pp_wavelet_output_cols",
        )
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class Haar(StatelessOperator):
    """Single-level Haar DWT coefficient transform."""

    _C_PREFIX = "n4m_pp_haar"

    def __init__(self):
        super().__init__()

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(lib.n4m_pp_haar_create(ctypes.byref(h)), "n4m_pp_haar_create")
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        out_cols = ctypes.c_int64()
        check(
            lib.n4m_pp_haar_output_cols(
                ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "n4m_pp_haar_output_cols",
        )
        self._ensure_handle()
        return self._call_transform(X, (X.shape[0], int(out_cols.value)))


class WaveletDenoise(BaseEstimator, TransformerMixin):
    """Multi-level DWT VisuShrink denoising.

    Stateless: matches PyWavelets' ``waverec(threshold(coeffs))`` pipeline.
    """

    _C_PREFIX = "n4m_pp_wavelet_denoise"

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
            lib.n4m_pp_wavelet_denoise_create(
                ctypes.byref(h),
                ctypes.c_int(self._family),
                ctypes.c_int(self._mode),
                ctypes.c_int32(self.level),
                ctypes.c_int(self._thr),
                ctypes.c_int(self._noise),
            ),
            "n4m_pp_wavelet_denoise_create",
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
            lib.n4m_pp_wavelet_denoise_transform(
                self._handle, numpy_to_view(X), numpy_to_view(out)
            ),
            "n4m_pp_wavelet_denoise_transform",
        )
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.n4m_pp_wavelet_denoise_destroy(self._handle)
            self._handle = None


class WaveletFeatures(StatelessOperator):
    """Multi-level DWT summary features."""

    _C_PREFIX = "n4m_pp_wavelet_features"

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
            lib.n4m_pp_wavelet_features_create_ex(
                ctypes.byref(h),
                ctypes.c_int(self._family),
                ctypes.c_int(self._mode),
                ctypes.c_int32(self.max_level),
                ctypes.c_int(self._entropy),
            ),
            "n4m_pp_wavelet_features_create_ex",
        )
        return h

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        out_cols = ctypes.c_int64()
        check(
            lib.n4m_pp_wavelet_features_output_cols(
                handle, ctypes.c_int64(X.shape[1]), ctypes.byref(out_cols)
            ),
            "n4m_pp_wavelet_features_output_cols",
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

    _C_PREFIX = "n4m_pp_wavelet_pca"

    def __init__(self, family: str = "haar", mode: str = "periodization",
                 max_level: int = 2, n_components: float = 5.0):
        super().__init__(family, mode, max_level, n_components)


class WaveletSVD(_WaveletProjection):
    """DWT coefficient projection through SVD scores."""

    _C_PREFIX = "n4m_pp_wavelet_svd"

    def __init__(self, family: str = "haar", mode: str = "periodization",
                 max_level: int = 2, n_components: float = 5.0):
        super().__init__(family, mode, max_level, n_components)


_BAND_MODES = {"zero": 0, "interp": 1, 0: 0, 1: 1}
_RANDOM_X_OPS = {"multiply": 0, "add": 1, "subtract": 2, 0: 0, 1: 1, 2: 2}


def _as_f64_axis(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _null_f64_ptr():
    return ctypes.POINTER(ctypes.c_double)()


class _AugmenterBase:
    """Shared lifecycle for stochastic ``n4m_aug_*`` handles."""

    _C_PREFIX = ""
    _APPLY_WAVELENGTHS = False

    def _setup_rng(self, rng: Optional[PCG64], seed: int) -> None:
        self.seed = int(seed)
        self._rng = PCG64(self.seed) if rng is None else rng
        self._handle: ctypes.c_void_p | None = None
        self._fitted = False
        self._pending_n_features: int | None = None

    def _setup_wavelengths(self, wavelengths) -> None:
        self.wavelengths = None if wavelengths is None else _as_f64_axis(
            wavelengths, "wavelengths"
        )

    def _wavelength_args(self, *, required: bool = False):
        wavelengths = getattr(self, "wavelengths", None)
        if wavelengths is None and required:
            if self._pending_n_features is None:
                raise ValueError("wavelengths are required before the handle can be created")
            wavelengths = np.arange(self._pending_n_features, dtype=np.float64)
            self.wavelengths = wavelengths
        if wavelengths is None:
            return _null_f64_ptr(), ctypes.c_int64(0)
        return (
            wavelengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(wavelengths.size),
        )

    def _wavelength_view(self, n_features: int):
        wavelengths = getattr(self, "wavelengths", None)
        if wavelengths is None:
            wavelengths = np.arange(n_features, dtype=np.float64)
            self.wavelengths = wavelengths
        if wavelengths.size != n_features:
            raise ValueError("wavelengths length must match X.shape[1]")
        return numpy_to_view(wavelengths.reshape(1, -1))

    def _create_args(self):
        return ()

    def _create_handle(self):
        h = ctypes.c_void_p()
        create = getattr(lib, f"{self._C_PREFIX}_create")
        check(
            create(ctypes.byref(h), self._rng.handle, *self._create_args()),
            f"{self._C_PREFIX}_create",
        )
        return h

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def fit(self, X=None, y=None):
        if X is not None:
            X = as_f64_2d(X)
            self._pending_n_features = int(X.shape[1])
        self._ensure_handle()
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        self._pending_n_features = int(X.shape[1])
        out = empty_like_f64(X.shape)
        apply_fn = getattr(lib, f"{self._C_PREFIX}_apply")
        if self._APPLY_WAVELENGTHS:
            status = apply_fn(
                self._ensure_handle(),
                numpy_to_view(X),
                self._wavelength_view(X.shape[1]),
                numpy_to_view(out),
            )
        else:
            status = apply_fn(self._ensure_handle(), numpy_to_view(X), numpy_to_view(out))
        check(status, f"{self._C_PREFIX}_apply")
        return out

    apply = transform

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def __del__(self):
        if self._handle is not None and self._handle.value:
            getattr(lib, f"{self._C_PREFIX}_destroy")(self._handle)
            self._handle = None


class GaussianAdditiveNoise(_AugmenterBase):
    """Add IID Gaussian noise to each element of ``X``."""

    _C_PREFIX = "n4m_aug_gaussian_noise"

    def __init__(self, sigma: float = 0.01, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.sigma = float(sigma)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.sigma),)


class MultiplicativeNoise(_AugmenterBase):
    """Apply per-element multiplicative Gaussian noise."""

    _C_PREFIX = "n4m_aug_multiplicative_noise"

    def __init__(self, sigma_gain: float = 0.01, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.sigma_gain = float(sigma_gain)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.sigma_gain),)


class SpikeNoise(_AugmenterBase):
    """Inject random spike artifacts into spectra."""

    _C_PREFIX = "n4m_aug_spike_noise"

    def __init__(self, n_spikes_min: int = 1, n_spikes_max: int = 3,
                 amplitude_min: float = -0.1, amplitude_max: float = 0.1,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.n_spikes_min = int(n_spikes_min)
        self.n_spikes_max = int(n_spikes_max)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_int32(self.n_spikes_min),
            ctypes.c_int32(self.n_spikes_max),
            ctypes.c_double(self.amplitude_min),
            ctypes.c_double(self.amplitude_max),
        )


class HeteroscedasticNoiseAugmenter(_AugmenterBase):
    """Noise whose standard deviation depends on signal magnitude."""

    _C_PREFIX = "n4m_aug_hetero_noise"

    def __init__(self, noise_base: float = 0.001,
                 noise_signal_dep: float = 0.01,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.noise_base = float(noise_base)
        self.noise_signal_dep = float(noise_signal_dep)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.noise_base),
                ctypes.c_double(self.noise_signal_dep))


class LinearBaselineDrift(_AugmenterBase):
    """Add random offset and linear slope drift."""

    _C_PREFIX = "n4m_aug_linear_drift"

    def __init__(self, offset_min: float = -0.05, offset_max: float = 0.05,
                 slope_min: float = -0.01, slope_max: float = 0.01,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.offset_min = float(offset_min)
        self.offset_max = float(offset_max)
        self.slope_min = float(slope_min)
        self.slope_max = float(slope_max)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_double(self.offset_min),
            ctypes.c_double(self.offset_max),
            ctypes.c_double(self.slope_min),
            ctypes.c_double(self.slope_max),
        )


class PolynomialBaselineDrift(_AugmenterBase):
    """Add random polynomial baseline drift."""

    _C_PREFIX = "n4m_aug_poly_drift"

    def __init__(self, degree: int = 2, coeff_min=None, coeff_max=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.degree = int(degree)
        if coeff_min is None:
            coeff_min = np.full(self.degree + 1, -0.01, dtype=np.float64)
        if coeff_max is None:
            coeff_max = np.full(self.degree + 1, 0.01, dtype=np.float64)
        self.coeff_min = _as_f64_axis(coeff_min, "coeff_min")
        self.coeff_max = _as_f64_axis(coeff_max, "coeff_max")
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_int32(self.degree),
            self.coeff_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeff_max.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )


class PathLengthAugmenter(_AugmenterBase):
    """Simulate multiplicative path-length variation."""

    _C_PREFIX = "n4m_aug_path_length"

    def __init__(self, path_length_std: float = 0.05,
                 min_path_length: float = 0.1,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.path_length_std = float(path_length_std)
        self.min_path_length = float(min_path_length)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.path_length_std),
                ctypes.c_double(self.min_path_length))


class WavelengthShift(_AugmenterBase):
    """Random spectral shift with linear interpolation."""

    _C_PREFIX = "n4m_aug_wavelength_shift"

    def __init__(self, shift_lo: float = -1.0, shift_hi: float = 1.0,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.shift_lo = float(shift_lo)
        self.shift_hi = float(shift_hi)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (ctypes.c_double(self.shift_lo), ctypes.c_double(self.shift_hi), ptr, n)


class WavelengthStretch(_AugmenterBase):
    """Random wavelength-axis stretching."""

    _C_PREFIX = "n4m_aug_wavelength_stretch"

    def __init__(self, stretch_lo: float = 0.99, stretch_hi: float = 1.01,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.stretch_lo = float(stretch_lo)
        self.stretch_hi = float(stretch_hi)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (ctypes.c_double(self.stretch_lo), ctypes.c_double(self.stretch_hi), ptr, n)


class LocalWarpAugmenter(_AugmenterBase):
    """Random local wavelength warping."""

    _C_PREFIX = "n4m_aug_local_warp"

    def __init__(self, n_control_points: int = 5, max_shift: float = 1.0,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.n_control_points = int(n_control_points)
        self.max_shift = float(max_shift)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (ctypes.c_int32(self.n_control_points), ctypes.c_double(self.max_shift), ptr, n)


class BandPerturbationAugmenter(_AugmenterBase):
    """Random band-local gain and offset perturbations."""

    _C_PREFIX = "n4m_aug_band_perturb"

    def __init__(self, n_bands: int = 3, bw_lo: int = 5, bw_hi: int = 15,
                 gain_lo: float = 0.9, gain_hi: float = 1.1,
                 offset_lo: float = -0.01, offset_hi: float = 0.01,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.n_bands = int(n_bands)
        self.bw_lo = int(bw_lo)
        self.bw_hi = int(bw_hi)
        self.gain_lo = float(gain_lo)
        self.gain_hi = float(gain_hi)
        self.offset_lo = float(offset_lo)
        self.offset_hi = float(offset_hi)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_int32(self.n_bands), ctypes.c_int32(self.bw_lo),
            ctypes.c_int32(self.bw_hi), ctypes.c_double(self.gain_lo),
            ctypes.c_double(self.gain_hi), ctypes.c_double(self.offset_lo),
            ctypes.c_double(self.offset_hi),
        )


class BandMasking(_AugmenterBase):
    """Mask random spectral bands with zero-fill or interpolation."""

    _C_PREFIX = "n4m_aug_band_mask"

    def __init__(self, n_bands_lo: int = 1, n_bands_hi: int = 3,
                 bw_lo: int = 5, bw_hi: int = 15, mode: str | int = "zero",
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.n_bands_lo = int(n_bands_lo)
        self.n_bands_hi = int(n_bands_hi)
        self.bw_lo = int(bw_lo)
        self.bw_hi = int(bw_hi)
        try:
            self.mode = _BAND_MODES[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown band mask mode: {mode!r}") from exc
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_int32(self.n_bands_lo), ctypes.c_int32(self.n_bands_hi),
            ctypes.c_int32(self.bw_lo), ctypes.c_int32(self.bw_hi),
            ctypes.c_int32(self.mode),
        )


class ChannelDropout(_AugmenterBase):
    """Randomly drop individual wavelength channels."""

    _C_PREFIX = "n4m_aug_channel_dropout"

    def __init__(self, dropout_prob: float = 0.05, mode: str | int = "zero",
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.dropout_prob = float(dropout_prob)
        try:
            self.mode = _BAND_MODES[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown channel dropout mode: {mode!r}") from exc
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.dropout_prob), ctypes.c_int32(self.mode))


class GaussianJitter(_AugmenterBase):
    """Gaussian smoothing jitter."""

    _C_PREFIX = "n4m_aug_gauss_jitter"

    def __init__(self, sigma_lo: float = 0.5, sigma_hi: float = 1.5,
                 kernel_width: int = 9, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.sigma_lo = float(sigma_lo)
        self.sigma_hi = float(sigma_hi)
        self.kernel_width = int(kernel_width)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.sigma_lo), ctypes.c_double(self.sigma_hi),
                ctypes.c_int32(self.kernel_width))


class UnsharpMask(_AugmenterBase):
    """Random unsharp spectral mask."""

    _C_PREFIX = "n4m_aug_unsharp_mask"

    def __init__(self, amount_lo: float = 0.1, amount_hi: float = 0.5,
                 sigma: float = 1.0, kernel_width: int = 11,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.amount_lo = float(amount_lo)
        self.amount_hi = float(amount_hi)
        self.sigma = float(sigma)
        self.kernel_width = int(kernel_width)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_double(self.amount_lo), ctypes.c_double(self.amount_hi),
            ctypes.c_double(self.sigma), ctypes.c_int32(self.kernel_width),
        )


class MagnitudeWarp(_AugmenterBase):
    """Smooth multiplicative magnitude warp."""

    _C_PREFIX = "n4m_aug_magnitude_warp"

    def __init__(self, n_control_points: int = 5, gain_lo: float = 0.9,
                 gain_hi: float = 1.1, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.n_control_points = int(n_control_points)
        self.gain_lo = float(gain_lo)
        self.gain_hi = float(gain_hi)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (ctypes.c_int32(self.n_control_points), ctypes.c_double(self.gain_lo),
                ctypes.c_double(self.gain_hi), ptr, n)


class LocalClip(_AugmenterBase):
    """Clip random local spectral regions."""

    _C_PREFIX = "n4m_aug_local_clip"

    def __init__(self, n_regions: int = 1, width_lo: int = 5,
                 width_hi: int = 15, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.n_regions = int(n_regions)
        self.width_lo = int(width_lo)
        self.width_hi = int(width_hi)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.n_regions), ctypes.c_int32(self.width_lo),
                ctypes.c_int32(self.width_hi))


class MixupAugmenter(_AugmenterBase):
    """Batch-wise mixup augmentation."""

    _C_PREFIX = "n4m_aug_mixup"

    def __init__(self, alpha: float = 0.2, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.alpha = float(alpha)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.alpha),)


class LocalMixupAugmenter(_AugmenterBase):
    """Neighbor-constrained mixup augmentation."""

    _C_PREFIX = "n4m_aug_local_mixup"

    def __init__(self, alpha: float = 0.2, k_neighbors: int = 5,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.alpha = float(alpha)
        self.k_neighbors = int(k_neighbors)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.alpha), ctypes.c_int32(self.k_neighbors))


class ScatterSimulationMSC(_AugmenterBase):
    """MSC-style multiplicative/additive scatter simulation."""

    _C_PREFIX = "n4m_aug_scatter_sim"

    def __init__(self, a_low: float = -0.05, a_high: float = 0.05,
                 b_low: float = 0.9, b_high: float = 1.1,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.a_low = float(a_low)
        self.a_high = float(a_high)
        self.b_low = float(b_low)
        self.b_high = float(b_high)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.a_low), ctypes.c_double(self.a_high),
                ctypes.c_double(self.b_low), ctypes.c_double(self.b_high))


class ParticleSizeAugmenter(_AugmenterBase):
    """Particle-size and path-length scattering simulation."""

    _C_PREFIX = "n4m_aug_particle_size"

    def __init__(self, mean_size_um: float = 50.0,
                 size_variation_um: float = 15.0,
                 use_size_range: bool = False,
                 size_range_low_um: float = 5.0,
                 size_range_high_um: float = 500.0,
                 reference_size_um: float = 50.0,
                 wavelength_exponent: float = 1.5,
                 size_effect_strength: float = 0.1,
                 include_path_length: bool = True,
                 path_length_sensitivity: float = 0.5,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.mean_size_um = float(mean_size_um)
        self.size_variation_um = float(size_variation_um)
        self.use_size_range = bool(use_size_range)
        self.size_range_low_um = float(size_range_low_um)
        self.size_range_high_um = float(size_range_high_um)
        self.reference_size_um = float(reference_size_um)
        self.wavelength_exponent = float(wavelength_exponent)
        self.size_effect_strength = float(size_effect_strength)
        self.include_path_length = bool(include_path_length)
        self.path_length_sensitivity = float(path_length_sensitivity)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=True)
        return (
            ctypes.c_double(self.mean_size_um), ctypes.c_double(self.size_variation_um),
            ctypes.c_int(1 if self.use_size_range else 0),
            ctypes.c_double(self.size_range_low_um),
            ctypes.c_double(self.size_range_high_um),
            ctypes.c_double(self.reference_size_um),
            ctypes.c_double(self.wavelength_exponent),
            ctypes.c_double(self.size_effect_strength),
            ctypes.c_int(1 if self.include_path_length else 0),
            ctypes.c_double(self.path_length_sensitivity),
            ptr, n,
        )


class EMSCDistortionAugmenter(_AugmenterBase):
    """Random EMSC-like multiplicative, additive and polynomial distortion."""

    _C_PREFIX = "n4m_aug_emsc_distort"

    def __init__(self, mult_low: float = 0.9, mult_high: float = 1.1,
                 add_low: float = -0.05, add_high: float = 0.05,
                 polynomial_order: int = 2, polynomial_strength: float = 0.02,
                 correlation: float = 0.3, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.mult_low = float(mult_low)
        self.mult_high = float(mult_high)
        self.add_low = float(add_low)
        self.add_high = float(add_high)
        self.polynomial_order = int(polynomial_order)
        self.polynomial_strength = float(polynomial_strength)
        self.correlation = float(correlation)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=True)
        return (
            ctypes.c_double(self.mult_low), ctypes.c_double(self.mult_high),
            ctypes.c_double(self.add_low), ctypes.c_double(self.add_high),
            ctypes.c_int32(self.polynomial_order),
            ctypes.c_double(self.polynomial_strength),
            ctypes.c_double(self.correlation), ptr, n,
        )


class BatchEffectAugmenter(_AugmenterBase):
    """Random offset, slope and gain batch effects."""

    _C_PREFIX = "n4m_aug_batch_effect"

    def __init__(self, offset_std: float = 0.0, slope_std: float = 0.0,
                 gain_std: float = 0.0, variation_scope: int = 0,
                 wavelengths=None, rng: Optional[PCG64] = None,
                 seed: int = 0):
        self.offset_std = float(offset_std)
        self.slope_std = float(slope_std)
        self.gain_std = float(gain_std)
        self.variation_scope = int(variation_scope)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (ctypes.c_double(self.offset_std), ctypes.c_double(self.slope_std),
                ctypes.c_double(self.gain_std), ctypes.c_int32(self.variation_scope),
                ptr, n)


class InstrumentalBroadeningAugmenter(_AugmenterBase):
    """Instrumental spectral broadening via Gaussian convolution."""

    _C_PREFIX = "n4m_aug_instrument_broaden"

    def __init__(self, fwhm: float = 5.0, use_fwhm_range: bool = False,
                 fwhm_low: float = 3.0, fwhm_high: float = 8.0,
                 variation_scope: int = 0, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.fwhm = float(fwhm)
        self.use_fwhm_range = bool(use_fwhm_range)
        self.fwhm_low = float(fwhm_low)
        self.fwhm_high = float(fwhm_high)
        self.variation_scope = int(variation_scope)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=False)
        return (
            ctypes.c_double(self.fwhm),
            ctypes.c_int(1 if self.use_fwhm_range else 0),
            ctypes.c_double(self.fwhm_low), ctypes.c_double(self.fwhm_high),
            ctypes.c_int32(self.variation_scope), ptr, n,
        )


class DeadBandAugmenter(_AugmenterBase):
    """Simulate dead spectral detector bands."""

    _C_PREFIX = "n4m_aug_dead_band"

    def __init__(self, n_bands: int = 1, width_low: int = 5,
                 width_high: int = 10, noise_std: float = 0.05,
                 probability: float = 0.0, variation_scope: int = 0,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.n_bands = int(n_bands)
        self.width_low = int(width_low)
        self.width_high = int(width_high)
        self.noise_std = float(noise_std)
        self.probability = float(probability)
        self.variation_scope = int(variation_scope)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_int32(self.n_bands), ctypes.c_int32(self.width_low),
            ctypes.c_int32(self.width_high), ctypes.c_double(self.noise_std),
            ctypes.c_double(self.probability), ctypes.c_int32(self.variation_scope),
        )


class TemperatureAugmenter(_AugmenterBase):
    """Temperature-induced shift, intensity and broadening perturbations."""

    _C_PREFIX = "n4m_aug_temperature"

    def __init__(self, temperature_delta: float = 0.0,
                 use_temp_range: bool = False, temp_low: float = -5.0,
                 temp_high: float = 5.0, enable_shift: bool = True,
                 enable_intensity: bool = True, enable_broadening: bool = True,
                 region_specific: bool = True, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.temperature_delta = float(temperature_delta)
        self.use_temp_range = bool(use_temp_range)
        self.temp_low = float(temp_low)
        self.temp_high = float(temp_high)
        self.enable_shift = bool(enable_shift)
        self.enable_intensity = bool(enable_intensity)
        self.enable_broadening = bool(enable_broadening)
        self.region_specific = bool(region_specific)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=True)
        return (
            ctypes.c_double(self.temperature_delta),
            ctypes.c_int(1 if self.use_temp_range else 0),
            ctypes.c_double(self.temp_low), ctypes.c_double(self.temp_high),
            ctypes.c_int(1 if self.enable_shift else 0),
            ctypes.c_int(1 if self.enable_intensity else 0),
            ctypes.c_int(1 if self.enable_broadening else 0),
            ctypes.c_int(1 if self.region_specific else 0),
            ptr, n,
        )


class MoistureAugmenter(_AugmenterBase):
    """Water activity and moisture-content spectral perturbation."""

    _C_PREFIX = "n4m_aug_moisture"

    def __init__(self, water_activity_delta: float = 0.0,
                 use_aw_range: bool = False, aw_low: float = 0.0,
                 aw_high: float = 1.0, reference_water_activity: float = 0.5,
                 free_water_fraction: float = 0.3, bound_water_shift: float = 25.0,
                 moisture_content: float = 0.10, enable_shift: bool = True,
                 enable_intensity: bool = True, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.water_activity_delta = float(water_activity_delta)
        self.use_aw_range = bool(use_aw_range)
        self.aw_low = float(aw_low)
        self.aw_high = float(aw_high)
        self.reference_water_activity = float(reference_water_activity)
        self.free_water_fraction = float(free_water_fraction)
        self.bound_water_shift = float(bound_water_shift)
        self.moisture_content = float(moisture_content)
        self.enable_shift = bool(enable_shift)
        self.enable_intensity = bool(enable_intensity)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        ptr, n = self._wavelength_args(required=True)
        return (
            ctypes.c_double(self.water_activity_delta),
            ctypes.c_int(1 if self.use_aw_range else 0),
            ctypes.c_double(self.aw_low), ctypes.c_double(self.aw_high),
            ctypes.c_double(self.reference_water_activity),
            ctypes.c_double(self.free_water_fraction),
            ctypes.c_double(self.bound_water_shift),
            ctypes.c_double(self.moisture_content),
            ctypes.c_int(1 if self.enable_shift else 0),
            ctypes.c_int(1 if self.enable_intensity else 0),
            ptr, n,
        )


class DetectorRollOffAugmenter(_AugmenterBase):
    """Detector edge roll-off artifact."""

    _C_PREFIX = "n4m_aug_detector_rolloff"
    _APPLY_WAVELENGTHS = True

    def __init__(self, detector_model: int = 4, effect_strength: float = 1.0,
                 noise_amplification: float = 0.02,
                 include_baseline_distortion: bool = True,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.detector_model = int(detector_model)
        self.effect_strength = float(effect_strength)
        self.noise_amplification = float(noise_amplification)
        self.include_baseline_distortion = bool(include_baseline_distortion)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.detector_model),
                ctypes.c_double(self.effect_strength),
                ctypes.c_double(self.noise_amplification),
                ctypes.c_int32(1 if self.include_baseline_distortion else 0))


class StrayLightAugmenter(_AugmenterBase):
    """Stray-light edge artifact."""

    _C_PREFIX = "n4m_aug_stray_light"
    _APPLY_WAVELENGTHS = True

    def __init__(self, stray_light_fraction: float = 0.001,
                 edge_enhancement: float = 2.0, edge_width: float = 0.1,
                 include_peak_truncation: bool = True,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.stray_light_fraction = float(stray_light_fraction)
        self.edge_enhancement = float(edge_enhancement)
        self.edge_width = float(edge_width)
        self.include_peak_truncation = bool(include_peak_truncation)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.stray_light_fraction),
                ctypes.c_double(self.edge_enhancement),
                ctypes.c_double(self.edge_width),
                ctypes.c_int32(1 if self.include_peak_truncation else 0))


class EdgeCurvatureAugmenter(_AugmenterBase):
    """Curved edge response artifact."""

    _C_PREFIX = "n4m_aug_edge_curve"
    _APPLY_WAVELENGTHS = True

    def __init__(self, curvature_strength: float = 0.02,
                 curvature_type: int = 0, asymmetry: float = 0.0,
                 edge_focus: float = 0.7, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.curvature_strength = float(curvature_strength)
        self.curvature_type = int(curvature_type)
        self.asymmetry = float(asymmetry)
        self.edge_focus = float(edge_focus)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.curvature_strength),
                ctypes.c_int32(self.curvature_type),
                ctypes.c_double(self.asymmetry), ctypes.c_double(self.edge_focus))


class TruncatedPeakAugmenter(_AugmenterBase):
    """Truncated peaks near spectral edges."""

    _C_PREFIX = "n4m_aug_truncated_peak"
    _APPLY_WAVELENGTHS = True

    def __init__(self, peak_probability: float = 0.5,
                 amplitude_min: float = 0.01, amplitude_max: float = 0.1,
                 width_min: float = 50.0, width_max: float = 200.0,
                 left_edge: bool = True, right_edge: bool = True,
                 wavelengths=None, rng: Optional[PCG64] = None, seed: int = 0):
        self.peak_probability = float(peak_probability)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.left_edge = bool(left_edge)
        self.right_edge = bool(right_edge)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (
            ctypes.c_double(self.peak_probability),
            ctypes.c_double(self.amplitude_min),
            ctypes.c_double(self.amplitude_max),
            ctypes.c_double(self.width_min),
            ctypes.c_double(self.width_max),
            ctypes.c_int32(1 if self.left_edge else 0),
            ctypes.c_int32(1 if self.right_edge else 0),
        )


class EdgeArtifactsAugmenter(_AugmenterBase):
    """Combined edge artifact augmenter."""

    _C_PREFIX = "n4m_aug_edge_artifacts"
    _APPLY_WAVELENGTHS = True

    DETECTOR_ROLL_OFF = 0x1
    STRAY_LIGHT = 0x2
    EDGE_CURVATURE = 0x4
    TRUNCATED_PEAKS = 0x8

    def __init__(self, enabled_flags: int = 0xF, overall_strength: float = 1.0,
                 detector_model: int = 4, wavelengths=None,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.enabled_flags = int(enabled_flags)
        self.overall_strength = float(overall_strength)
        self.detector_model = int(detector_model)
        self._setup_wavelengths(wavelengths)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.enabled_flags),
                ctypes.c_double(self.overall_strength),
                ctypes.c_int32(self.detector_model))


class SplineSmoothingAugmenter(_AugmenterBase):
    """Spline smoothing augmenter."""

    _C_PREFIX = "n4m_aug_spline_smooth"

    def __init__(self, rng: Optional[PCG64] = None, seed: int = 0):
        self._setup_rng(rng, seed)


class SplineXPerturbationAugmenter(_AugmenterBase):
    """Spline x-axis perturbation augmenter."""

    _C_PREFIX = "n4m_aug_spline_x_perturb"

    def __init__(self, spline_degree: int = 3, perturbation_density: float = 0.05,
                 perturbation_range_min: float = -0.1,
                 perturbation_range_max: float = 0.1,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.spline_degree = int(spline_degree)
        self.perturbation_density = float(perturbation_density)
        self.perturbation_range_min = float(perturbation_range_min)
        self.perturbation_range_max = float(perturbation_range_max)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.spline_degree),
                ctypes.c_double(self.perturbation_density),
                ctypes.c_double(self.perturbation_range_min),
                ctypes.c_double(self.perturbation_range_max))


class SplineYPerturbationAugmenter(_AugmenterBase):
    """Spline y-axis perturbation augmenter."""

    _C_PREFIX = "n4m_aug_spline_y_perturb"

    def __init__(self, spline_points: int = -1, perturbation_intensity: float = 0.005,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.spline_points = int(spline_points)
        self.perturbation_intensity = float(perturbation_intensity)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.spline_points),
                ctypes.c_double(self.perturbation_intensity))


class SplineXSimplificationAugmenter(_AugmenterBase):
    """Spline x simplification stub; current C ABI returns NOT_IMPLEMENTED."""

    _C_PREFIX = "n4m_aug_spline_x_simplify"

    def __init__(self, spline_points: int = -1, uniform: bool = True,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.spline_points = int(spline_points)
        self.uniform = bool(uniform)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.spline_points),
                ctypes.c_int32(1 if self.uniform else 0))


class SplineCurveSimplificationAugmenter(SplineXSimplificationAugmenter):
    """Spline curve simplification stub; current C ABI returns NOT_IMPLEMENTED."""

    _C_PREFIX = "n4m_aug_spline_curve_simplify"


class RotateTranslateAugmenter(_AugmenterBase):
    """Random rotate/translate spectral augmentation."""

    _C_PREFIX = "n4m_aug_rotate_translate"

    def __init__(self, p_range: float = 2.0, y_factor: float = 3.0,
                 rng: Optional[PCG64] = None, seed: int = 0):
        self.p_range = float(p_range)
        self.y_factor = float(y_factor)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_double(self.p_range), ctypes.c_double(self.y_factor))


class RandomXOperation(_AugmenterBase):
    """Random element-wise multiply/add/subtract operation."""

    _C_PREFIX = "n4m_aug_random_x_op"

    def __init__(self, op_kind: str | int = "multiply",
                 operator_range_min: float = 0.97,
                 operator_range_max: float = 1.03,
                 rng: Optional[PCG64] = None, seed: int = 0):
        try:
            self.op_kind = _RANDOM_X_OPS[op_kind]
        except KeyError as exc:
            raise ValueError(f"Unknown random X operation: {op_kind!r}") from exc
        self.operator_range_min = float(operator_range_min)
        self.operator_range_max = float(operator_range_max)
        self._setup_rng(rng, seed)

    def _create_args(self):
        return (ctypes.c_int32(self.op_kind),
                ctypes.c_double(self.operator_range_min),
                ctypes.c_double(self.operator_range_max))


WavelengthShiftAugmenter = WavelengthShift
WavelengthStretchAugmenter = WavelengthStretch
LocalWavelengthWarp = LocalWarpAugmenter
BandMaskingAugmenter = BandMasking
GaussianSmoothingJitter = GaussianJitter
UnsharpSpectralMask = UnsharpMask
SmoothMagnitudeWarp = MagnitudeWarp


__all__ = [
    "BandMasking",
    "BandMaskingAugmenter",
    "BandPerturbationAugmenter",
    "BatchEffectAugmenter",
    "ChannelDropout",
    "DeadBandAugmenter",
    "DetectorRollOffAugmenter",
    "EMSCDistortionAugmenter",
    "EdgeArtifactsAugmenter",
    "EdgeCurvatureAugmenter",
    "GaussianAdditiveNoise",
    "GaussianJitter",
    "GaussianSmoothingJitter",
    "Haar",
    "HeteroscedasticNoiseAugmenter",
    "InstrumentalBroadeningAugmenter",
    "LinearBaselineDrift",
    "LocalClip",
    "LocalMixupAugmenter",
    "LocalWarpAugmenter",
    "LocalWavelengthWarp",
    "MagnitudeWarp",
    "MixupAugmenter",
    "MoistureAugmenter",
    "MultiplicativeNoise",
    "ParticleSizeAugmenter",
    "PathLengthAugmenter",
    "PolynomialBaselineDrift",
    "RandomXOperation",
    "RotateTranslateAugmenter",
    "ScatterSimulationMSC",
    "SmoothMagnitudeWarp",
    "SplineCurveSimplificationAugmenter",
    "SplineSmoothingAugmenter",
    "SplineXPerturbationAugmenter",
    "SplineXSimplificationAugmenter",
    "SplineYPerturbationAugmenter",
    "SpikeNoise",
    "StrayLightAugmenter",
    "TemperatureAugmenter",
    "TruncatedPeakAugmenter",
    "UnsharpMask",
    "UnsharpSpectralMask",
    "Wavelet",
    "WaveletDenoise",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
    "WavelengthShift",
    "WavelengthShiftAugmenter",
    "WavelengthStretch",
    "WavelengthStretchAugmenter",
]

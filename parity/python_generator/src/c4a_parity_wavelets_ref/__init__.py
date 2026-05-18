# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 6 wavelets preprocessing operators
(Wavelet, Haar, WaveletDenoise, WaveletFeatures, WaveletPCA, WaveletSVD).

Validated once against ``PyWavelets==1.6.0`` (the parity-lock pin).  After
validation these modules become the canonical parity floor for
chemometrics4all — pywt itself is no longer in the build path of the
parity fixture generator.

Each module re-implements the requested transform in NumPy only, using
filter banks vendored from PyWavelets 1.6.0 (haar / db4 / sym4 / coif1).
Three boundary modes are supported: ``periodization``, ``symmetric``
and ``zero``.

The reference covers six operators:

  - ``wavelet_transform``           : single-level DWT (cA || cD output)
  - ``haar_transform``              : haar+periodization shortcut
  - ``wavelet_denoise_transform``   : multi-level denoise with universal
                                        threshold and MAD / std sigma
  - ``wavelet_features_transform``  : multi-level + per-band statistics
  - ``wavelet_pca_fit_transform``   : DWT-flatten + PCA
  - ``wavelet_svd_fit_transform``   : DWT-flatten + Truncated SVD
"""

from .filters import (
    FILTER_BANKS,
    boundary_mode_id,
    dwt_max_level,
    dwt_output_length,
    family_id,
)
from .haar import haar_transform
from .wavelet import wavelet_transform
from .wavelet_denoise import wavelet_denoise_transform
from .wavelet_features import wavelet_features_transform
from .wavelet_pca import wavelet_pca_fit_transform
from .wavelet_svd import wavelet_svd_fit_transform

__all__ = [
    "FILTER_BANKS",
    "boundary_mode_id",
    "dwt_max_level",
    "dwt_output_length",
    "family_id",
    "haar_transform",
    "wavelet_transform",
    "wavelet_denoise_transform",
    "wavelet_features_transform",
    "wavelet_pca_fit_transform",
    "wavelet_svd_fit_transform",
]

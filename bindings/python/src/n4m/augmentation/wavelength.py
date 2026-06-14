# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    LocalWarpAugmenter,
    WavelengthShift,
    WavelengthStretch,
)
from n4m._impl import native as _native

aug_wavelength_spectral = _native.aug_wavelength_spectral

__all__ = [
    "LocalWarpAugmenter",
    "WavelengthShift",
    "WavelengthStretch",
    "aug_wavelength_spectral",
]

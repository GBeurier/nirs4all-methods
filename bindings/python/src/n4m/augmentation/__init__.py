# SPDX-License-Identifier: CECILL-2.1
"""n4m.augmentation role package."""

from . import (
    drift,
    instrument,
    mixup,
    noise,
    scattering,
    spectral,
    splines,
    wavelength,
)
from .native import native_augmentation_specs, run_native

__all__ = [
    "native_augmentation_specs",
    "run_native",
    "drift",
    "instrument",
    "mixup",
    "noise",
    "scattering",
    "spectral",
    "splines",
    "wavelength",
]

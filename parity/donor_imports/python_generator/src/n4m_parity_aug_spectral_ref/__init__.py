# SPDX-License-Identifier: CECILL-2.1
"""Frozen NumPy reference for the Phase 16 wavelength + spectral augmenters.

The ten Phase 16 operators (WavelengthShift, WavelengthStretch,
LocalWavelengthWarp, BandPerturbation, BandMasking, ChannelDropout,
GaussianSmoothingJitter, UnsharpSpectralMask, SmoothMagnitudeWarp,
LocalClipping) are reproduced here in pure NumPy + scipy.ndimage so the C
engine has a stable parity floor independent of any nirs4all releases.

Two operators (LocalWavelengthWarp, SmoothMagnitudeWarp) use linear
interpolation through `np.interp` in this reference, where the original
nirs4all path uses `scipy.interpolate.splrep / splev` (cubic spline). The
linear variant is chosen here because the cubic spline is non-trivial to
reproduce bit-exactly in C; the C engine matches this linear path at the
1e-15 PCG64 stochastic-with-seeded-RNG contract.

Random-stream order is locked: every operator draws its parameters in the
same order and shape that NumPy's `Generator.uniform / integers` calls
emit them — see the C engines under
``cpp/src/core/augmentation/{wavelength, spectral}/`` for the matching
sweep order.
"""

from .band_mask import band_mask
from .band_perturb import band_perturb
from .channel_dropout import channel_dropout
from .gauss_jitter import gauss_jitter
from .local_clip import local_clip
from .local_warp import local_warp
from .magnitude_warp import magnitude_warp
from .unsharp_mask import unsharp_mask
from .wavelength_shift import wavelength_shift
from .wavelength_stretch import wavelength_stretch

__all__ = [
    "band_mask",
    "band_perturb",
    "channel_dropout",
    "gauss_jitter",
    "local_clip",
    "local_warp",
    "magnitude_warp",
    "unsharp_mask",
    "wavelength_shift",
    "wavelength_stretch",
]

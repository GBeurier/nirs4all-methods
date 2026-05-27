# SPDX-License-Identifier: CECILL-2.1
"""Python binding for libn4m.

The package exposes two public operator surfaces:

* ``n4m.python``: ABI-close NumPy functions over ``libn4m``;
* ``n4m.sklearn``: scikit-learn-compatible estimators.

Top-level imports still re-export the sklearn estimator classes for backwards
compatibility::

    from n4m import SNV, SavitzkyGolay
"""
import ctypes

from . import python, sklearn
from ._errors import N4MError
from ._ffi import (
    ABI_VERSION_MAJOR,
    ABI_VERSION_MINOR,
    ABI_VERSION_PATCH,
    ABI_VERSION_STRING,
    lib,
    library_path,
)
from ._rng import PCG64
from .metrics import (
    bias,
    hotelling_t2,
    mae,
    nrmse,
    q_residuals,
    r2,
    rmse,
    rpd,
    rpiq,
    sep,
    transfer_metrics,
)
from .sklearn import (
    aug_wavelength_spectral,
    nirs_metrics,
    signal_type_detector,
)
from .sklearn import (
    EMSC,
    LSNV,
    MSC,
    RNV,
    SNV,
    AirPLS,
    ArPLS,
    AreaNormalization,
    AsLS,
    BEADS,
    BaselineCenter,
    Crop,
    CropTransformer,
    Derivate,
    Detrend,
    EPO,
    FCKStaticTransformer,
    FirstDerivative,
    FlexiblePCA,
    FlexibleSVD,
    FractionToPercent,
    FromAbsorbance,
    Gaussian,
    GaussianAdditiveNoise,
    Haar,
    IAsLS,
    IModPoly,
    IntegerKBinsDiscretizer,
    KBinsDiscretizer,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KubelkaMunk,
    LogTransform,
    ModPoly,
    Normalize,
    NorrisWilliams,
    OSC,
    PercentToFraction,
    RangeDiscretizer,
    RollingBall,
    Resample,
    ResampleTransformer,
    Resampler,
    SNIP,
    SavitzkyGolay,
    SecondDerivative,
    SPXYSplitter,
    SimpleScale,
    ToAbsorbance,
    Wavelet,
    WaveletDenoise,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    XOutlierFilter,
    YOutlierFilter,
)
from .sklearn import *  # noqa: F403 - re-export full sklearn-style surface
from .sklearn import __all__ as _sklearn_all


def version() -> str:
    """Return the runtime libn4m version string, e.g. ``0.1.0+abi.1.9.0``."""
    raw = ctypes.c_char_p(lib.n4m_get_version_string()).value
    if raw is None:  # pragma: no cover - defensive ABI guard
        raise RuntimeError("libn4m returned a null version string")
    return raw.decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    """Return the loaded libn4m ABI version as ``(major, minor, patch)``."""
    return (
        int(lib.n4m_get_abi_version_major()),
        int(lib.n4m_get_abi_version_minor()),
        int(lib.n4m_get_abi_version_patch()),
    )


__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "AirPLS",
    "ArPLS",
    "AreaNormalization",
    "AsLS",
    "BEADS",
    "BaselineCenter",
    "N4MError",
    "Crop",
    "CropTransformer",
    "Derivate",
    "Detrend",
    "EPO",
    "EMSC",
    "FCKStaticTransformer",
    "FirstDerivative",
    "FlexiblePCA",
    "FlexibleSVD",
    "FractionToPercent",
    "FromAbsorbance",
    "Gaussian",
    "GaussianAdditiveNoise",
    "Haar",
    "IAsLS",
    "IModPoly",
    "IntegerKBinsDiscretizer",
    "KBinsDiscretizer",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KubelkaMunk",
    "LSNV",
    "LogTransform",
    "MSC",
    "ModPoly",
    "Normalize",
    "NorrisWilliams",
    "OSC",
    "PCG64",
    "PercentToFraction",
    "python",
    "RangeDiscretizer",
    "RNV",
    "Resample",
    "ResampleTransformer",
    "Resampler",
    "SNV",
    "SPXYSplitter",
    "RollingBall",
    "SNIP",
    "SavitzkyGolay",
    "SecondDerivative",
    "SimpleScale",
    "sklearn",
    "ToAbsorbance",
    "Wavelet",
    "WaveletDenoise",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
    "XOutlierFilter",
    "YOutlierFilter",
    "__version__",
    "abi_version",
    "aug_wavelength_spectral",
    "bias",
    "hotelling_t2",
    "library_path",
    "mae",
    "nirs_metrics",
    "nrmse",
    "q_residuals",
    "r2",
    "rmse",
    "rpd",
    "rpiq",
    "sep",
    "signal_type_detector",
    "transfer_metrics",
    "version",
]

__all__ = sorted(set(__all__) | set(_sklearn_all))

# Derive __version__ from the loaded native library so it stays in sync with the
# C ABI version_string (shape "X.Y.Z+abi.A.B.C") — strip the "+abi.*" suffix.
# Mirrors the pls4all subset; avoids a hardcoded version drifting from the lib.
__version__ = version().split("+", 1)[0]

# SPDX-License-Identifier: CECILL-2.1
"""Python binding for libc4a.

Top-level imports re-export the sklearn-style operator classes so that the
public surface mirrors what users see in scikit-learn pipelines::

    from chemometrics4all import SNV, SavitzkyGolay
"""
import ctypes

from ._errors import Chemometrics4allError
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

__version__ = "0.1.0"


def version() -> str:
    """Return the runtime libc4a version string, e.g. ``0.1.0+abi.1.9.0``."""
    raw = ctypes.c_char_p(lib.c4a_get_version_string()).value
    if raw is None:  # pragma: no cover - defensive ABI guard
        raise RuntimeError("libc4a returned a null version string")
    return raw.decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    """Return the loaded libc4a ABI version as ``(major, minor, patch)``."""
    return (
        int(lib.c4a_get_abi_version_major()),
        int(lib.c4a_get_abi_version_minor()),
        int(lib.c4a_get_abi_version_patch()),
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
    "Chemometrics4allError",
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
    "bias",
    "hotelling_t2",
    "library_path",
    "mae",
    "nrmse",
    "q_residuals",
    "r2",
    "rmse",
    "rpd",
    "rpiq",
    "sep",
    "transfer_metrics",
    "version",
]

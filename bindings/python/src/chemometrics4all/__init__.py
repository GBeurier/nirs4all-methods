# SPDX-License-Identifier: CECILL-2.1
"""Python binding for libc4a.

Top-level imports re-export the sklearn-style operator classes so that the
public surface mirrors what users see in scikit-learn pipelines::

    from chemometrics4all import SNV, SavitzkyGolay
"""
from ._errors import Chemometrics4allError
from ._ffi import (
    ABI_VERSION_MAJOR,
    ABI_VERSION_MINOR,
    ABI_VERSION_PATCH,
    ABI_VERSION_STRING,
    library_path,
)
from ._rng import PCG64
from .sklearn import (
    EMSC,
    LSNV,
    MSC,
    RNV,
    SNV,
    AirPLS,
    AsLS,
    Detrend,
    FirstDerivative,
    GaussianAdditiveNoise,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KubelkaMunk,
    SavitzkyGolay,
    SecondDerivative,
    SPXYSplitter,
    ToAbsorbance,
    WaveletDenoise,
    XOutlierFilter,
    YOutlierFilter,
)

__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "AirPLS",
    "AsLS",
    "Chemometrics4allError",
    "Detrend",
    "EMSC",
    "FirstDerivative",
    "GaussianAdditiveNoise",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KubelkaMunk",
    "LSNV",
    "MSC",
    "PCG64",
    "RNV",
    "SNV",
    "SPXYSplitter",
    "SavitzkyGolay",
    "SecondDerivative",
    "ToAbsorbance",
    "WaveletDenoise",
    "XOutlierFilter",
    "YOutlierFilter",
    "library_path",
]

# SPDX-License-Identifier: CECILL-2.1
"""Scikit-learn-style wrappers around libc4a operators.

Twenty operators that are covered by Gate 1 (``binding_parity``) tests:

* Preprocessings: SNV, LSNV, RNV, MSC, EMSC, SavitzkyGolay, FirstDerivative,
  SecondDerivative, ToAbsorbance, KubelkaMunk.
* Baseline: Detrend, AsLS, AirPLS.
* Splitters: KennardStoneSplitter, SPXYSplitter, KBinsStratifiedSplitter.
* Filters: YOutlierFilter, XOutlierFilter.
* Misc: WaveletDenoise, GaussianAdditiveNoise.
"""
from .augmentation import GaussianAdditiveNoise, WaveletDenoise
from .baseline import AirPLS, AsLS, Detrend
from .filters import XOutlierFilter, YOutlierFilter
from .preprocessing import (
    EMSC,
    LSNV,
    MSC,
    RNV,
    SNV,
    FirstDerivative,
    KubelkaMunk,
    SavitzkyGolay,
    SecondDerivative,
    ToAbsorbance,
)
from .splitters import KBinsStratifiedSplitter, KennardStoneSplitter, SPXYSplitter

__all__ = [
    "AirPLS",
    "AsLS",
    "Detrend",
    "EMSC",
    "FirstDerivative",
    "GaussianAdditiveNoise",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KubelkaMunk",
    "LSNV",
    "MSC",
    "RNV",
    "SNV",
    "SPXYSplitter",
    "SavitzkyGolay",
    "SecondDerivative",
    "ToAbsorbance",
    "WaveletDenoise",
    "XOutlierFilter",
    "YOutlierFilter",
]

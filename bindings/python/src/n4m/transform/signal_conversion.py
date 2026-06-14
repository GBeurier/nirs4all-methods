# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    FractionToPercent,
    FromAbsorbance,
    KubelkaMunk,
    PercentToFraction,
    ToAbsorbance,
)
from n4m._impl import native as _native

signal_type_detector = _native.signal_type_detector

__all__ = [
    "FractionToPercent",
    "FromAbsorbance",
    "KubelkaMunk",
    "PercentToFraction",
    "ToAbsorbance",
    "signal_type_detector",
]

# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import NativeRobustPLSRegressor as RobustPLS
from n4m._impl import NativeWeightedPLSRegressor as WeightedPLS
from n4m._impl import native as _native

robust_pls = _native.robust_pls
weighted_pls = _native.weighted_pls

__all__ = [
    "RobustPLS",
    "WeightedPLS",
    "robust_pls",
    "weighted_pls",
]

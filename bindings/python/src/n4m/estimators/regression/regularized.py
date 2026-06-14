# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import NativeRidgeRegressor as Ridge
from n4m._impl import NativeRidgePLSRegressor as RidgePLS
from n4m._impl import native as _native

ridge = _native.ridge
ridge_pls = _native.ridge_pls

__all__ = [
    "Ridge",
    "RidgePLS",
    "ridge",
    "ridge_pls",
]

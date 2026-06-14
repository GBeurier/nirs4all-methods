# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    AOMOperatorPLSSpec,
    AOMOperatorPLSStack,
)
from n4m._impl import NativeAOMPLSSuperblockRegressor as AOMPLSSuperblock
from n4m._impl import NativeAOMRidgeActiveSuperblockRegressor as AOMRidgeActiveSuperblock
from n4m._impl import NativeAOMRidgeMKLSuperblockRegressor as AOMRidgeMKLSuperblock
from n4m._impl import NativeAOMRidgePLSSuperblockRegressor as AOMRidgePLSSuperblock
from n4m._impl import NativeAOMRidgeSuperblockRegressor as AOMRidgeSuperblock
from n4m._impl import NativeAOMPLSSuperblockRegressor as AOMPLSSuperblockRegressor
from n4m._impl import NativeAOMRidgeActiveSuperblockRegressor as AOMRidgeActiveSuperblockRegressor
from n4m._impl import NativeAOMRidgeMKLSuperblockRegressor as AOMRidgeMKLSuperblockRegressor
from n4m._impl import NativeAOMRidgePLSSuperblockRegressor as AOMRidgePLSSuperblockRegressor
from n4m._impl import NativeAOMRidgeSuperblockRegressor as AOMRidgeSuperblockRegressor
from n4m._impl import native as _native

aom_pls_superblock = _native.aom_pls_superblock
aom_ridge_active_superblock = _native.aom_ridge_active_superblock
aom_ridge_mkl_superblock = _native.aom_ridge_mkl_superblock
aom_ridge_pls_superblock = _native.aom_ridge_pls_superblock
aom_ridge_superblock = _native.aom_ridge_superblock

__all__ = [
    "AOMOperatorPLSSpec",
    "AOMOperatorPLSStack",
    "AOMPLSSuperblock",
    "AOMPLSSuperblockRegressor",
    "AOMRidgeActiveSuperblock",
    "AOMRidgeActiveSuperblockRegressor",
    "AOMRidgeMKLSuperblock",
    "AOMRidgeMKLSuperblockRegressor",
    "AOMRidgePLSSuperblock",
    "AOMRidgePLSSuperblockRegressor",
    "AOMRidgeSuperblock",
    "AOMRidgeSuperblockRegressor",
    "aom_pls_superblock",
    "aom_ridge_active_superblock",
    "aom_ridge_mkl_superblock",
    "aom_ridge_pls_superblock",
    "aom_ridge_superblock",
]

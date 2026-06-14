# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import NativeCPPLSRegressor as CPPLS
from n4m._impl import NativeContinuumRegressionRegressor as ContinuumRegression
from n4m._impl import NativeECRRegressor as ECR
from n4m._impl import NativePCRRegressor as PCR
from n4m._impl import NativePLSRegressor as PLS
from n4m._impl import native as _native

continuum_regression = _native.continuum_regression
cppls = _native.cppls
ecr = _native.ecr
pcr = _native.pcr
pls = _native.pls

__all__ = [
    "CPPLS",
    "ContinuumRegression",
    "ECR",
    "PCR",
    "PLS",
    "continuum_regression",
    "cppls",
    "ecr",
    "pcr",
    "pls",
]

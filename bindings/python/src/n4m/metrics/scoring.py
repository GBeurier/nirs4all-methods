# SPDX-License-Identifier: CECILL-2.1

from n4m._impl.metrics import (
    bias,
    mae,
    nrmse,
    r2,
    rmse,
    rpd,
    rpiq,
    sep,
)
from n4m._impl import native as _native

nirs_metrics = _native.nirs_metrics

__all__ = [
    "bias",
    "mae",
    "nirs_metrics",
    "nrmse",
    "r2",
    "rmse",
    "rpd",
    "rpiq",
    "sep",
]

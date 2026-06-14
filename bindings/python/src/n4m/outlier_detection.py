# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    CompositeFilter,
    HighLeverageFilter,
    SpectralQualityFilter,
    XOutlierFilter,
    YOutlierFilter,
)
from n4m._impl import native as _native

hotelling_t2 = _native.hotelling_t2
q_residuals = _native.q_residuals
x_outlier_mahalanobis = _native.x_outlier_mahalanobis
y_outlier_iqr = _native.y_outlier_iqr

__all__ = [
    "CompositeFilter",
    "HighLeverageFilter",
    "SpectralQualityFilter",
    "XOutlierFilter",
    "YOutlierFilter",
    "hotelling_t2",
    "q_residuals",
    "x_outlier_mahalanobis",
    "y_outlier_iqr",
]

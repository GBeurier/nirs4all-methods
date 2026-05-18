# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 12+ filter operators.

Validated once against ``nirs4all/operators/filters/y_outlier.py`` (the
nirs4all 0.8.x implementation). After validation these modules become the
canonical parity floor for chemometrics4all's `c4a_filter_*` ABI category
- nirs4all itself is no longer in the build path of the parity fixture
generator.

This insulation gives us two things:

  1. Upstream nirs4all releases can change implementation details
     (additional kwargs, internal restructuring) without breaking the
     chemometrics4all parity fixtures.
  2. The reference is self-contained: pure NumPy that anyone can read and
     audit against the algorithm description in
     ``docs/algorithms/y_outlier_filter.md``.

Phase 12 (1 op, 4 methods): y_outlier (iqr / zscore / percentile / mad).
Phase 13 (1 op, 6 methods): x_outlier (mahalanobis / robust_mahalanobis /
                                       pca_residual / pca_leverage /
                                       isolation_forest / lof).
"""

from .composite import composite_mask
from .high_leverage import high_leverage_leverages, high_leverage_mask
from .spectral_quality import spectral_quality_mask
from .x_outlier import x_outlier_fit_get_mask
from .y_outlier import y_outlier_fit_get_mask

__all__ = [
    "composite_mask",
    "high_leverage_leverages",
    "high_leverage_mask",
    "spectral_quality_mask",
    "x_outlier_fit_get_mask",
    "y_outlier_fit_get_mask",
]

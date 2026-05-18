# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 20 transfer metrics utility.

Mirrors ``nirs4all/analysis/transfer_metrics.py`` *but with one critical
difference*: the subsampling step in ``spread_distance`` uses our own
SplitMix64-driven Fisher-Yates permutation (helper ``_deterministic_choice``)
instead of ``numpy.random.RandomState.choice``. This is intentional — the
C engine cannot replay legacy NumPy ``RandomState`` bit-exactly, but it can
trivially mirror SplitMix64. The Python reference and the C engine are
therefore byte-identical w.r.t. their selected indices for any given seed.

The metrics themselves (PCA, CKA, RV, Grassmann, Procrustes, Trustworthiness,
Spread) are computed via NumPy ground-truth identical to scikit-learn 1.4.2
and SciPy 1.11.4 — those libraries are not in the runtime path of the
reference, only of the validation tests.
"""

from .transfer_metrics import (
    TransferMetrics,
    transfer_metrics,
)

__all__ = [
    "TransferMetrics",
    "transfer_metrics",
]

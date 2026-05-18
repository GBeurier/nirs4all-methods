# SPDX-License-Identifier: CECILL-2.1
"""
Spectral quality sample filter — frozen NumPy reference.

Per-row multi-criteria quality check. A sample is KEPT only if every check
that is enabled passes:

  1. NaN ratio          <= max_nan_ratio
  2. Inf check          (if check_inf): no element is +/- inf
  3. Zero ratio         <= max_zero_ratio    (NaN counts as non-zero)
  4. Variance           >= min_variance      (np.nanvar)
  5. Max value          <= max_value         (only if max_value is not None)
  6. Min value          >= min_value         (only if min_value is not None)

The mask is the conjunction (AND) of all enabled checks.

The reference mirrors ``nirs4all.operators.filters.SpectralQualityFilter``
verbatim — the c4a engine reproduces the same arithmetic order and
threshold semantics. Because every check reduces to a single comparison,
the c4a output is bit-exact (mask equality, not float tolerance).
"""

from __future__ import annotations

import numpy as np


def spectral_quality_mask(X: np.ndarray,
                           *,
                           max_nan_ratio: float = 0.1,
                           max_zero_ratio: float = 0.5,
                           min_variance: float = 1e-8,
                           max_value: float | None = None,
                           min_value: float | None = None,
                           check_inf: bool = True) -> np.ndarray:
    """
    Compute the keep-mask for the spectral quality filter.

    Parameters mirror ``SpectralQualityFilter``. Returns a (n,) bool array
    where True means the sample passes every enabled check.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("spectral_quality_mask requires a 2-D input")
    if not (0.0 <= max_nan_ratio <= 1.0):
        raise ValueError("max_nan_ratio must be in [0, 1]")
    if not (0.0 <= max_zero_ratio <= 1.0):
        raise ValueError("max_zero_ratio must be in [0, 1]")
    if min_variance < 0.0:
        raise ValueError("min_variance must be non-negative")

    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.ones(0, dtype=bool)
    if n_features == 0:
        # No features means we can't compute variance / ratios; treat all
        # rows as having empty content => fail (consistent with the
        # behaviour of the c4a engine when cols == 0).
        return np.zeros(n_samples, dtype=bool)

    mask = np.ones(n_samples, dtype=bool)

    # 1. NaN ratio.
    nan_counts = np.sum(np.isnan(X), axis=1)
    nan_ratios = nan_counts / n_features
    mask &= (nan_ratios <= max_nan_ratio)

    # 2. Inf check.
    if check_inf:
        has_inf = np.any(np.isinf(X), axis=1)
        mask &= ~has_inf

    # 3. Zero ratio (NaN treated as non-zero — matches nirs4all).
    X_no_nan = np.nan_to_num(X, nan=1.0)
    zero_counts = np.sum(X_no_nan == 0.0, axis=1)
    zero_ratios = zero_counts / n_features
    mask &= (zero_ratios <= max_zero_ratio)

    # 4. Variance.
    with np.errstate(all="ignore"):
        variances = np.nanvar(X, axis=1)
    # NaN variances (e.g. all-NaN row) fail the check.
    var_ok = np.where(np.isnan(variances), False, variances >= min_variance)
    mask &= var_ok

    # 5. Max value.
    if max_value is not None:
        with np.errstate(all="ignore"):
            max_values = np.nanmax(X, axis=1)
        max_ok = np.where(np.isnan(max_values), False,
                          max_values <= float(max_value))
        mask &= max_ok

    # 6. Min value.
    if min_value is not None:
        with np.errstate(all="ignore"):
            min_values = np.nanmin(X, axis=1)
        min_ok = np.where(np.isnan(min_values), False,
                          min_values >= float(min_value))
        mask &= min_ok

    return mask

# SPDX-License-Identifier: CECILL-2.1
"""
Y-based outlier filter — frozen NumPy reference.

This is the canonical parity floor for the Phase 12 `n4m_filter_y_outlier_*`
ABI. Mirrors `nirs4all.operators.filters.YOutlierFilter` bit-for-bit on
integer counts and within 1e-12 on the numeric bounds.

Four methods:

  * iqr        : ``[Q1 - threshold*IQR,   Q3 + threshold*IQR]``
                 (threshold default 1.5)
  * zscore     : ``[mu - threshold*sigma, mu + threshold*sigma]``
                 (threshold default 3.0)
  * percentile : ``[percentile(y, lower_percentile),
                    percentile(y, upper_percentile)]``
  * mad        : ``[median - threshold*MAD*1.4826,
                    median + threshold*MAD*1.4826]``
                 (threshold default 3.5)

Quantiles use NumPy 1.26.4's default ``method="linear"`` interpolation.
``np.std`` is the biased estimator (``ddof=0``, NumPy default).

Degenerate inputs (single non-NaN sample, zero scale, all-NaN, empty) are
handled identically to the nirs4all reference — see the inline comments.

Returns
-------
A tuple ``(mask, stats)``:

  * ``mask`` : ``np.ndarray`` of dtype ``uint8``, shape ``(n,)``.
               ``mask[i] == 1`` keeps sample i (within bounds, not NaN).
               ``mask[i] == 0`` excludes it.
  * ``stats`` : ``dict`` with keys ``n_samples``, ``n_kept``, ``n_excluded``,
                ``exclusion_rate`` (matches the public
                ``n4m_filter_stats_t`` field set).
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ["y_outlier_fit_get_mask"]


def _bounds_iqr(y_valid: np.ndarray, threshold: float) -> tuple[float, float]:
    q1 = np.percentile(y_valid, 25)
    q3 = np.percentile(y_valid, 75)
    iqr = q3 - q1
    return (q1 - threshold * iqr, q3 + threshold * iqr)


def _bounds_zscore(y_valid: np.ndarray, threshold: float) -> tuple[float, float]:
    mean = np.mean(y_valid)
    std = np.std(y_valid)  # ddof=0 (NumPy default)
    if std == 0.0:
        return (mean - 1e-10, mean + 1e-10)
    return (mean - threshold * std, mean + threshold * std)


def _bounds_percentile(
    y_valid: np.ndarray, lower_pct: float, upper_pct: float
) -> tuple[float, float]:
    return (
        float(np.percentile(y_valid, lower_pct)),
        float(np.percentile(y_valid, upper_pct)),
    )


def _bounds_mad(y_valid: np.ndarray, threshold: float) -> tuple[float, float]:
    median = np.median(y_valid)
    mad = np.median(np.abs(y_valid - median))
    # 1.4826 = Gaussian-consistency factor: sigma ~= 1.4826 * MAD on normal data.
    mad_scaled = mad * 1.4826
    if mad_scaled == 0.0:
        return (median - 1e-10, median + 1e-10)
    return (median - threshold * mad_scaled, median + threshold * mad_scaled)


def y_outlier_fit_get_mask(
    y: np.ndarray,
    *,
    method: str = "iqr",
    threshold: float = 1.5,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute the keep mask for `y` under one of the four outlier methods.

    Parameters
    ----------
    y : array-like, shape (n,)
        Target values. NaN entries are always excluded.
    method : {"iqr", "zscore", "percentile", "mad"}
        Detection method. Default "iqr".
    threshold : float
        Threshold for the iqr / zscore / mad methods. Positive.
    lower_percentile, upper_percentile : float
        Cutoffs for the percentile method. Must satisfy
        ``0 <= lower < upper <= 100``.

    Returns
    -------
    mask : np.ndarray of dtype uint8, shape (n,)
        ``mask[i] == 1`` keeps sample i; ``mask[i] == 0`` excludes it.
    stats : dict
        ``{"n_samples", "n_kept", "n_excluded", "exclusion_rate"}``.
    """
    valid_methods = ("iqr", "zscore", "percentile", "mad")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100, "
            f"got lower={lower_percentile}, upper={upper_percentile}"
        )

    y_flat = np.ascontiguousarray(y, dtype=np.float64).flatten()
    n = y_flat.shape[0]

    if n == 0:
        # Empty input: nothing to keep, nothing to exclude.
        stats = {
            "n_samples": 0,
            "n_kept": 0,
            "n_excluded": 0,
            "exclusion_rate": 0.0,
        }
        return np.empty((0,), dtype=np.uint8), stats

    is_nan = np.isnan(y_flat)
    y_valid = y_flat[~is_nan]

    if y_valid.size == 0:
        # All NaN: bounds neutral, NaN check below masks everything out.
        lo, hi = -np.inf, np.inf
    elif y_valid.size == 1:
        v = float(y_valid[0])
        lo, hi = v - 1e-10, v + 1e-10
    else:
        if method == "iqr":
            lo, hi = _bounds_iqr(y_valid, threshold)
        elif method == "zscore":
            lo, hi = _bounds_zscore(y_valid, threshold)
        elif method == "percentile":
            lo, hi = _bounds_percentile(y_valid, lower_percentile, upper_percentile)
        else:  # method == "mad"
            lo, hi = _bounds_mad(y_valid, threshold)

    within_bounds = (y_flat >= lo) & (y_flat <= hi)
    keep = within_bounds & ~is_nan
    mask = keep.astype(np.uint8)

    n_kept = int(mask.sum())
    n_excluded = n - n_kept
    stats = {
        "n_samples": int(n),
        "n_kept": n_kept,
        "n_excluded": n_excluded,
        "exclusion_rate": float(n_excluded) / float(n),
    }
    return mask, stats

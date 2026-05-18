# SPDX-License-Identifier: CECILL-2.1
"""
Composite sample filter — frozen NumPy reference.

The composite combines multiple sub-filter masks via either AND ("all") or
OR ("any") semantics, expressed in keep-mask space:

  * mode = "any" (default): a sample is EXCLUDED if ANY sub-filter excludes
    it.  Equivalent to ``keep = all(submask_k)``.
  * mode = "all":          a sample is EXCLUDED only if ALL sub-filters
    exclude it.  Equivalent to ``keep = any(submask_k)``.

The reference takes a list of pre-computed keep-masks (NumPy bool arrays of
identical shape) because the c4a engine doesn't replay the sub-filter
algorithms inside the composite — it holds references to the sub-filter
handles and aggregates their already-computed masks.

This matches ``nirs4all.operators.filters.CompositeFilter.get_mask``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def composite_mask(submasks: Sequence[np.ndarray],
                    *,
                    mode: str = "any") -> np.ndarray:
    """Aggregate sub-filter keep-masks via the given mode.

    Parameters
    ----------
    submasks : sequence of (n,) bool arrays
        Per-filter keep masks. Must have identical length.
        If empty, returns an all-True mask of inferred length 0.
    mode : "any" or "all"
        Mode semantics (see module docstring).

    Returns
    -------
    mask : (n,) bool array
    """
    if mode not in ("any", "all"):
        raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")
    if not submasks:
        # Empty composite keeps all samples by convention.
        return np.ones(0, dtype=bool)
    arr = np.stack([np.asarray(m, dtype=bool) for m in submasks], axis=0)
    if mode == "any":
        return np.asarray(np.all(arr, axis=0))
    return np.asarray(np.any(arr, axis=0))

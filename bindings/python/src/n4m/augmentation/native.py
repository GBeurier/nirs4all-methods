# SPDX-License-Identifier: CECILL-2.1
"""Closed, train-only X→X augmentation dispatch through the native C ABI."""

from __future__ import annotations

import ctypes

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import numpy_to_view

# Public names match n4m-R's n4m_augmentation_apply kinds. Values are the
# native kind ID and exact positional parameter count; numerical work stays C++.
_SPECS: dict[str, tuple[int, int]] = {
    "gaussian_noise": (0, 1), "multiplicative_noise": (1, 1),
    "spike_noise": (2, 4), "hetero_noise": (3, 2),
    "linear_drift": (4, 4), "path_length": (5, 2),
    "band_perturb": (6, 7), "band_mask": (7, 5),
    "channel_dropout": (8, 2), "gauss_jitter": (9, 3),
    "unsharp_mask": (10, 4), "local_clip": (11, 3),
    "rotate_translate": (12, 2), "random_x_op": (13, 3),
    "scatter_sim_msc": (14, 4), "dead_band": (15, 6),
    "batch_effect": (16, 4), "spline_smoothing": (17, 0),
    "spline_x_perturb": (18, 4), "spline_y_perturb": (19, 2),
    "spline_x_simplify": (20, 2), "spline_curve_simplify": (21, 2),
}


def native_augmentation_specs() -> dict[str, int]:
    """Return a copy of supported native kinds and exact parameter counts."""
    return {name: count for name, (_, count) in _SPECS.items()}


def run_native(kind: str, X, values=(), seed: int = 0) -> np.ndarray:
    """Apply one seeded native augmenter to training X, preserving its shape.

    This contract never alters Y or creates a portable fitted state. Kinds
    requiring label mixing or an axis are intentionally unavailable.
    """
    if not isinstance(kind, str) or kind not in _SPECS:
        raise ValueError(f"unknown native augmentation kind: {kind!r}")
    code, count = _SPECS[kind]
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not x.flags.c_contiguous:
        if x.ndim != 2 or min(x.shape) < 1:
            raise ValueError("X must be a nonempty two-dimensional matrix")
        x = np.ascontiguousarray(x)
    if not np.all(np.isfinite(x)):
        raise ValueError("X must contain only finite values")
    params = np.asarray(values, dtype=np.float64)
    if params.ndim != 1 or params.size != count or not np.all(np.isfinite(params)):
        raise ValueError("invalid native augmentation parameter vector")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**53 - 1:
        raise ValueError("seed must be an exact nonnegative integer <= 2^53-1")
    # The output allocation and view are caller-owned; the C kernel writes
    # every element. No Python/NumPy numerical augmentation is performed.
    out = np.empty_like(x)
    xv = numpy_to_view(x)
    ov = numpy_to_view(out)
    pointer = params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if count else None
    check(lib.n4m_augmentation_run(
        ctypes.c_int32(code), pointer, ctypes.c_int32(count),
        ctypes.c_uint64(seed), xv, ov), "n4m_augmentation_run")
    return out

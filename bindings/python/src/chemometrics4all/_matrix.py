# SPDX-License-Identifier: CECILL-2.1
"""Conversion helpers between :mod:`numpy` arrays and :c:struct:`c4a_matrix_view_t`.

libc4a's row-major preprocessings require contiguous F64 input. Bindings must
materialise a contiguous copy whenever the caller hands us a non-contiguous /
non-F64 array; that copy lives only for the duration of the C call.
"""
from __future__ import annotations

import numpy as np

from ._types import Dtype, MatrixView


def _resolve_dtype(arr: np.ndarray) -> int:
    if arr.dtype == np.float64:
        return Dtype.F64
    if arr.dtype == np.float32:
        return Dtype.F32
    if arr.dtype == np.int32:
        return Dtype.I32
    if arr.dtype == np.int64:
        return Dtype.I64
    raise TypeError(f"Unsupported numpy dtype for c4a_matrix_view_t: {arr.dtype}")


def as_f64_2d(arr) -> np.ndarray:
    """Coerce ``arr`` to a 2-D row-major contiguous float64 ndarray.

    Returns the original array when already conforming; otherwise a new
    contiguous copy. 1-D inputs are promoted to a single-row 2-D shape.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got ndim={a.ndim}")
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def numpy_to_view(arr: np.ndarray) -> MatrixView:
    """Build a row-major :class:`MatrixView` over ``arr``.

    The caller is responsible for keeping ``arr`` alive for the duration of
    the C call (the view does not own the buffer).
    """
    if arr.ndim != 2:
        raise ValueError(f"MatrixView requires a 2-D array, got ndim={arr.ndim}")
    if not arr.flags.c_contiguous:
        raise ValueError("MatrixView requires a C-contiguous array")
    dtype_id = _resolve_dtype(arr)
    rows, cols = arr.shape
    view = MatrixView()
    view.data = arr.ctypes.data
    view.rows = rows
    view.cols = cols
    # Row-major: col_stride = 1 element; row_stride = cols elements.
    view.row_stride = cols
    view.col_stride = 1
    view.dtype = dtype_id
    view.reserved0 = 0
    return view


def empty_like_f64(shape) -> np.ndarray:
    """Allocate an empty C-contiguous F64 array with the given ``shape``."""
    return np.empty(shape, dtype=np.float64, order="C")


def empty_like_i32(shape) -> np.ndarray:
    """Allocate an empty C-contiguous int32 array (for discretizers)."""
    return np.empty(shape, dtype=np.int32, order="C")


__all__ = [
    "as_f64_2d",
    "empty_like_f64",
    "empty_like_i32",
    "numpy_to_view",
]

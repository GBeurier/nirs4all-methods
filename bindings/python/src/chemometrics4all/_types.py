# SPDX-License-Identifier: CECILL-2.1
"""ctypes structs and enums mirroring ``c4a.h``.

Layout matters: the C ABI fixes :class:`c4a_matrix_view_t` at 48 bytes on LP64
/ LLP64 platforms. We replicate the exact field order, with the trailing
``int32_t reserved0`` to keep the size stable. :class:`c4a_filter_stats_t`,
:class:`c4a_split_result_t` and :class:`c4a_transfer_metrics_t` follow the
same hand-mirroring approach.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, Structure, c_double, c_int, c_int32, c_int64, c_void_p


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------


class Status:
    """Mirror of :c:enum:`c4a_status_t`."""

    OK = 0
    ERR_INVALID_ARGUMENT = 1
    ERR_NULL_POINTER = 2
    ERR_SHAPE_MISMATCH = 3
    ERR_DTYPE_MISMATCH = 4
    ERR_STRIDE_INVALID = 5
    ERR_NOT_FITTED = 6
    ERR_NUMERICAL_FAILURE = 7
    ERR_CONVERGENCE_FAILED = 8
    ERR_OUT_OF_MEMORY = 9
    ERR_UNSUPPORTED = 10
    ERR_NOT_IMPLEMENTED = 11
    ERR_ABI_MISMATCH = 12
    ERR_IO = 13
    ERR_CORRUPT_BUFFER = 14
    ERR_VERSION_INCOMPATIBLE = 15
    ERR_BACKEND_UNAVAILABLE = 16
    ERR_CANCELLED = 17
    ERR_INTERNAL = 255


# ---------------------------------------------------------------------------
# Numerical dtypes
# ---------------------------------------------------------------------------


class Dtype:
    """Mirror of :c:enum:`c4a_dtype_t`."""

    UNKNOWN = 0
    F64 = 1
    F32 = 2
    I32 = 3
    I64 = 4


# ---------------------------------------------------------------------------
# c4a_matrix_view_t (48 bytes on LP64 / LLP64)
# ---------------------------------------------------------------------------


class MatrixView(Structure):
    """Non-owning 2-D view passed to every libc4a transform.

    Field order matches the C struct exactly (see ``c4a.h §3``).
    """

    _fields_ = [
        ("data", c_void_p),
        ("rows", c_int64),
        ("cols", c_int64),
        ("row_stride", c_int64),
        ("col_stride", c_int64),
        ("dtype", c_int),
        ("reserved0", c_int32),
    ]


# ---------------------------------------------------------------------------
# c4a_filter_stats_t
# ---------------------------------------------------------------------------


class FilterStats(Structure):
    """Mirror of :c:struct:`c4a_filter_stats_t`."""

    _fields_ = [
        ("n_samples", c_int64),
        ("n_kept", c_int64),
        ("n_excluded", c_int64),
        ("exclusion_rate", c_double),
    ]


# ---------------------------------------------------------------------------
# c4a_split_result_t (heap-allocated by libc4a; freed via destroy)
# ---------------------------------------------------------------------------


class SplitResult(Structure):
    """Mirror of :c:struct:`c4a_split_result_t`."""

    _fields_ = [
        ("train_idx", POINTER(c_int64)),
        ("n_train", c_int64),
        ("test_idx", POINTER(c_int64)),
        ("n_test", c_int64),
        ("_owner", c_void_p),
    ]


# ---------------------------------------------------------------------------
# c4a_transfer_metrics_t
# ---------------------------------------------------------------------------


class TransferMetrics(Structure):
    """Mirror of :c:struct:`c4a_transfer_metrics_t`."""

    _fields_ = [
        ("centroid_distance", c_double),
        ("cka_similarity", c_double),
        ("grassmann_distance", c_double),
        ("rv_coefficient", c_double),
        ("procrustes_disparity", c_double),
        ("trustworthiness", c_double),
        ("spread_distance", c_double),
        ("evr_source", c_double),
        ("evr_target", c_double),
    ]


# Sanity check at import time: layout sizes must match the ABI banner.
assert ctypes.sizeof(MatrixView) == 48, (
    f"MatrixView layout mismatch: {ctypes.sizeof(MatrixView)} != 48"
)


__all__ = [
    "Dtype",
    "FilterStats",
    "MatrixView",
    "SplitResult",
    "Status",
    "TransferMetrics",
]

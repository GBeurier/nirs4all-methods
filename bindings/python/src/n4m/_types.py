# SPDX-License-Identifier: CECILL-2.1
"""ctypes structs and enums mirroring ``n4m.h``.

Layout matters: the C ABI fixes :class:`n4m_matrix_view_t` at 48 bytes on LP64
/ LLP64 platforms. We replicate the exact field order, with the trailing
``int32_t reserved0`` to keep the size stable. :class:`n4m_filter_stats_t`,
:class:`n4m_split_result_t` and :class:`n4m_transfer_metrics_t` follow the
same hand-mirroring approach.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_int,
    c_int32,
    c_int64,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------


class Status:
    """Mirror of :c:enum:`n4m_status_t`."""

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
    """Mirror of :c:enum:`n4m_dtype_t`."""

    UNKNOWN = 0
    F64 = 1
    F32 = 2
    I32 = 3
    I64 = 4


# ---------------------------------------------------------------------------
# n4m_matrix_view_t (48 bytes on LP64 / LLP64)
# ---------------------------------------------------------------------------


class MatrixView(Structure):
    """Non-owning 2-D view passed to every libn4m transform.

    Field order matches the C struct exactly (see ``n4m.h §3``).
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


class LinearPredictorSpec(Structure):
    """Mirror of :c:struct:`n4m_linear_predictor_spec_t`.

    The coefficient buffer is row-major ``(n_features, n_targets)`` and both
    buffers are borrowed only for the duration of the native import call.
    """

    _fields_ = [
        ("source_training_samples", c_int64),
        ("n_features", c_int32),
        ("n_targets", c_int32),
        ("coefficients", POINTER(c_double)),
        ("intercept", POINTER(c_double)),
    ]


class SerializedModelInfoV1(Structure):
    """Mirror of :c:struct:`n4m_serialized_model_info_v1_t`."""

    _fields_ = [
        ("schema_version", c_uint32),
        ("format_version", c_uint32),
        ("writer_abi_major", c_uint32),
        ("writer_abi_minor", c_uint32),
        ("writer_abi_patch", c_uint32),
        ("algorithm", c_int),
        ("solver", c_int),
        ("deflation", c_int),
        ("training_samples", c_int64),
        ("n_features", c_int32),
        ("n_targets", c_int32),
        ("n_components", c_int32),
        ("reserved0", c_uint32),
        ("capabilities", c_uint64),
    ]


class SerializedPipelineInfoV1(Structure):
    """Mirror of :c:struct:`n4m_serialized_pipeline_info_v1_t`."""

    _fields_ = [
        ("schema_version", c_uint32),
        ("struct_size", c_uint32),
        ("present", c_uint32),
        ("operator_count", c_uint32),
        ("operators", c_int32 * 2),
        ("savgol_window", c_int32),
        ("savgol_poly_degree", c_int32),
        ("savgol_derivative", c_int32),
        ("reserved0", c_uint32),
        ("savgol_delta", c_double),
        ("raw_n_features", c_int32),
        ("model_n_features", c_int32),
        ("fingerprint_algorithm", c_uint32),
        ("reserved1", c_uint32),
        ("fingerprint", c_uint64),
        ("reserved", c_uint8 * 24),
    ]


# ---------------------------------------------------------------------------
# n4m_filter_stats_t
# ---------------------------------------------------------------------------


class FilterStats(Structure):
    """Mirror of :c:struct:`n4m_filter_stats_t`."""

    _fields_ = [
        ("n_samples", c_int64),
        ("n_kept", c_int64),
        ("n_excluded", c_int64),
        ("exclusion_rate", c_double),
    ]


# ---------------------------------------------------------------------------
# n4m_split_result_t (heap-allocated by libn4m; freed via destroy)
# ---------------------------------------------------------------------------


class SplitResult(Structure):
    """Mirror of :c:struct:`n4m_split_result_t`."""

    _fields_ = [
        ("train_idx", POINTER(c_int64)),
        ("n_train", c_int64),
        ("test_idx", POINTER(c_int64)),
        ("n_test", c_int64),
        ("_owner", c_void_p),
    ]


# ---------------------------------------------------------------------------
# n4m_transfer_metrics_t
# ---------------------------------------------------------------------------


class TransferMetrics(Structure):
    """Mirror of :c:struct:`n4m_transfer_metrics_t`."""

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


class OptimizerOptions(Structure):
    """Mirror of :c:struct:`n4m_optimizer_options_t` (native HPO optimizer).

    Field order and native alignment match the C struct exactly. Call
    :func:`init` (which invokes ``n4m_optimizer_options_init``) to populate
    ``struct_size`` and the defaults before use.
    """

    _fields_ = [
        ("struct_size", c_uint64),
        ("sampler", c_int),
        ("pruner", c_int),
        ("direction", c_int),
        ("eval_mode", c_int),
        ("metric", c_int),
        ("liar", c_int),
        ("n_startup_trials", c_int32),
        ("seed", c_uint64),
        ("timeout_seconds", c_double),
        ("max_resource", c_int32),
        ("reduction_factor", c_int32),
        ("reserved", c_uint8 * 56),
    ]


# Sanity check at import time: layout sizes must match the ABI banner.
assert ctypes.sizeof(MatrixView) == 48, (
    f"MatrixView layout mismatch: {ctypes.sizeof(MatrixView)} != 48"
)
assert ctypes.sizeof(SerializedModelInfoV1) == 64, (
    "SerializedModelInfoV1 layout mismatch: "
    f"{ctypes.sizeof(SerializedModelInfoV1)} != 64"
)
assert ctypes.sizeof(SerializedPipelineInfoV1) == 96, (
    "SerializedPipelineInfoV1 layout mismatch: "
    f"{ctypes.sizeof(SerializedPipelineInfoV1)} != 96"
)
if ctypes.sizeof(c_void_p) == 8:
    assert ctypes.sizeof(OptimizerOptions) == 120, (
        f"OptimizerOptions layout mismatch: {ctypes.sizeof(OptimizerOptions)} != 120"
    )
    _OPTIMIZER_OPTION_OFFSETS = {
        "struct_size": 0,
        "sampler": 8,
        "n_startup_trials": 32,
        "seed": 40,
        "timeout_seconds": 48,
        "reserved": 64,
    }
    for _field, _expected_offset in _OPTIMIZER_OPTION_OFFSETS.items():
        _actual_offset = getattr(OptimizerOptions, _field).offset
        assert _actual_offset == _expected_offset, (
            f"OptimizerOptions.{_field} offset mismatch: "
            f"{_actual_offset} != {_expected_offset}"
        )


__all__ = [
    "Dtype",
    "FilterStats",
    "LinearPredictorSpec",
    "MatrixView",
    "OptimizerOptions",
    "SerializedModelInfoV1",
    "SerializedPipelineInfoV1",
    "SplitResult",
    "Status",
    "TransferMetrics",
]

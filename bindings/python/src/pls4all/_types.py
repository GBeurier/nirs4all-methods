"""Enums mirroring p4a.h. Phase 2 expands this with the full algorithm /
solver / deflation / operator-kind / gating-mode / model-array sets."""

from __future__ import annotations

from enum import IntEnum


class Status(IntEnum):
    OK                       = 0
    ERR_INVALID_ARGUMENT     = 1
    ERR_NULL_POINTER         = 2
    ERR_SHAPE_MISMATCH       = 3
    ERR_DTYPE_MISMATCH       = 4
    ERR_STRIDE_INVALID       = 5
    ERR_NOT_FITTED           = 6
    ERR_NUMERICAL_FAILURE    = 7
    ERR_CONVERGENCE_FAILED   = 8
    ERR_OUT_OF_MEMORY        = 9
    ERR_UNSUPPORTED          = 10
    ERR_NOT_IMPLEMENTED      = 11
    ERR_ABI_MISMATCH         = 12
    ERR_IO                   = 13
    ERR_CORRUPT_BUFFER       = 14
    ERR_VERSION_INCOMPATIBLE = 15
    ERR_BACKEND_UNAVAILABLE  = 16
    ERR_CANCELLED            = 17
    ERR_INTERNAL             = 255


class Backend(IntEnum):
    AUTO          = 0
    REFERENCE_CPU = 1
    NATIVE_CPU    = 2
    BLAS          = 3
    OPENMP        = 4
    CUDA          = 5
    OPENCL        = 6
    METAL         = 7


class Dtype(IntEnum):
    UNKNOWN = 0
    F64     = 1
    F32     = 2
    I32     = 3
    I64     = 4

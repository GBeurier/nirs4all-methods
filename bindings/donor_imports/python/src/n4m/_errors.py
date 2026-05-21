# SPDX-License-Identifier: CECILL-2.1
"""Exception type raised when libn4m returns a non-``OK`` status code."""
from __future__ import annotations

from ._types import Status

_STATUS_NAMES = {
    Status.OK: "OK",
    Status.ERR_INVALID_ARGUMENT: "INVALID_ARGUMENT",
    Status.ERR_NULL_POINTER: "NULL_POINTER",
    Status.ERR_SHAPE_MISMATCH: "SHAPE_MISMATCH",
    Status.ERR_DTYPE_MISMATCH: "DTYPE_MISMATCH",
    Status.ERR_STRIDE_INVALID: "STRIDE_INVALID",
    Status.ERR_NOT_FITTED: "NOT_FITTED",
    Status.ERR_NUMERICAL_FAILURE: "NUMERICAL_FAILURE",
    Status.ERR_CONVERGENCE_FAILED: "CONVERGENCE_FAILED",
    Status.ERR_OUT_OF_MEMORY: "OUT_OF_MEMORY",
    Status.ERR_UNSUPPORTED: "UNSUPPORTED",
    Status.ERR_NOT_IMPLEMENTED: "NOT_IMPLEMENTED",
    Status.ERR_ABI_MISMATCH: "ABI_MISMATCH",
    Status.ERR_IO: "IO",
    Status.ERR_CORRUPT_BUFFER: "CORRUPT_BUFFER",
    Status.ERR_VERSION_INCOMPATIBLE: "VERSION_INCOMPATIBLE",
    Status.ERR_BACKEND_UNAVAILABLE: "BACKEND_UNAVAILABLE",
    Status.ERR_CANCELLED: "CANCELLED",
    Status.ERR_INTERNAL: "INTERNAL",
}


class N4MError(RuntimeError):
    """libn4m returned a non-OK status code."""

    def __init__(self, status: int, message: str = ""):
        self.status = int(status)
        self.status_name = _STATUS_NAMES.get(self.status, f"UNKNOWN({self.status})")
        super().__init__(f"{self.status_name}: {message}" if message else self.status_name)


def check(status: int, message: str = "") -> None:
    """Raise :class:`N4MError` when ``status != Status.OK``."""
    if status != Status.OK:
        raise N4MError(status, message)


__all__ = ["N4MError", "check"]

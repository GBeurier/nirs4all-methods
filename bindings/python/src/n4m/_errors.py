# SPDX-License-Identifier: CECILL-2.1
"""Exception type raised when libn4m returns a non-``OK`` status code."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import Status

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a runtime import cycle)
    from .model_selection.optimizer import Trial

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
        super().__init__(
            f"{self.status_name}: {message}" if message else self.status_name
        )


class PartialBatchError(N4MError):
    """A best-effort batch committed trials, then failed non-benignly.

    Raised by :meth:`n4m.model_selection.Optimizer.ask_batch` when the native
    batch dispenses one or more trials and then stops on an error that is NOT a
    benign generation boundary or timeout (for example an invalid queued
    warm-start, or an out-of-memory / internal failure).

    ``partial_trials`` holds the already-committed trials in ask order. They are
    ``RUNNING`` and core-owned (borrowed wrappers that keep the optimizer alive);
    the native side never rolls them back. The caller MUST reach a terminal state
    for each — ``tell`` / ``tell_result`` — or recover them from
    :meth:`Optimizer.get_trials`. ``status`` is the exact native status code that
    stopped the batch.
    """

    def __init__(
        self, status: int, partial_trials: "list[Trial]", message: str = ""
    ) -> None:
        super().__init__(status, message)
        self.partial_trials: list[Trial] = list(partial_trials)


def check(status: int, message: str = "") -> None:
    """Raise :class:`N4MError` when ``status != Status.OK``."""
    if status != Status.OK:
        raise N4MError(status, message)


__all__ = ["N4MError", "PartialBatchError", "check"]

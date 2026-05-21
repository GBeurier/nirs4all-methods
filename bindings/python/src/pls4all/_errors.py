"""Status → exception translation."""

from __future__ import annotations

from ._ffi import lib


class Pls4allError(RuntimeError):
    """Raised when a C ABI call returns a non-zero status code.

    Carries the integer status, its name, and (when a context was attached)
    the last_error message snapshot taken at the moment of failure.
    """

    def __init__(self, status: int, last_error: str | None = None) -> None:
        name = lib.n4m_status_to_string(status).decode("utf-8")
        if last_error:
            message = f"[{name}] {last_error}"
        else:
            message = f"[{name}]"
        super().__init__(message)
        self.status = status
        self.status_name = name
        self.last_error = last_error

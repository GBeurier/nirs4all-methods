"""Pythonic wrapper around n4m_context_t."""

from __future__ import annotations

import ctypes

from ._errors import Pls4allError
from ._ffi import lib
from ._types import Backend, Status


def _check(status_int: int, ctx_handle: int | None = None) -> None:
    if status_int == Status.OK:
        return
    msg = None
    if ctx_handle is not None:
        raw = lib.n4m_context_last_error(ctx_handle)
        if raw:
            msg = raw.decode("utf-8")
    raise Pls4allError(status_int, msg)


class Context:
    """Owning handle around a n4m_context_t.

    Usable as a context manager (``with Context() as ctx: ...``) for
    deterministic cleanup. Single-threaded use only; create one Context per
    thread (the underlying n4m_context_t is documented as such).
    """

    def __init__(self) -> None:
        handle = ctypes.c_void_p(0)
        status = lib.n4m_context_create(ctypes.byref(handle))
        if status != Status.OK or not handle:
            raise Pls4allError(status, "n4m_context_create failed")
        self._h = handle

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._h:
            lib.n4m_context_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    # ---- seed ----
    @property
    def seed(self) -> int:
        out = ctypes.c_uint64(0)
        _check(lib.n4m_context_get_seed(self._h, ctypes.byref(out)), self._h.value)
        return int(out.value)

    @seed.setter
    def seed(self, value: int) -> None:
        _check(lib.n4m_context_set_seed(self._h, ctypes.c_uint64(int(value))), self._h.value)

    # ---- backend ----
    @property
    def backend(self) -> Backend:
        out = ctypes.c_int(0)
        _check(lib.n4m_context_get_backend(self._h, ctypes.byref(out)), self._h.value)
        return Backend(out.value)

    @backend.setter
    def backend(self, value: Backend) -> None:
        _check(lib.n4m_context_set_backend(self._h, int(value)), self._h.value)

    # ---- num_threads ----
    @property
    def num_threads(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_context_get_num_threads(self._h, ctypes.byref(out)), self._h.value)
        return int(out.value)

    @num_threads.setter
    def num_threads(self, value: int) -> None:
        _check(lib.n4m_context_set_num_threads(self._h, ctypes.c_int32(int(value))), self._h.value)

    # ---- error buffer ----
    @property
    def last_error(self) -> str:
        raw = lib.n4m_context_last_error(self._h)
        return raw.decode("utf-8") if raw else ""

    def clear_error(self) -> None:
        lib.n4m_context_clear_error(self._h)

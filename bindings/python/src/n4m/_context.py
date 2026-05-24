# SPDX-License-Identifier: CECILL-2.1
"""Thin wrapper around :c:type:`n4m_context_t`.

Most operators in libn4m don't currently consume a context (they take their
parameters at create-time and live independently). The context type is
exposed mainly for completeness of the FFI surface and for future use by
backend / seed configuration calls.
"""
from __future__ import annotations

import ctypes

from ._errors import check
from ._ffi import lib


class Context:
    """Opaque libn4m context.

    Created lazily; released by ``__del__`` or :meth:`close`. Not thread-safe.
    """

    def __init__(self) -> None:
        handle = ctypes.c_void_p()
        status = lib.n4m_context_create(ctypes.byref(handle))
        check(status, "n4m_context_create")
        self._handle = handle

    def set_seed(self, seed: int) -> None:
        check(lib.n4m_context_set_seed(self._handle, ctypes.c_uint64(seed)),
              "n4m_context_set_seed")

    @property
    def num_threads(self) -> int:
        out = ctypes.c_int32()
        check(lib.n4m_context_get_num_threads(self._handle, ctypes.byref(out)),
              "n4m_context_get_num_threads")
        return int(out.value)

    @num_threads.setter
    def num_threads(self, value: int) -> None:
        check(
            lib.n4m_context_set_num_threads(self._handle, ctypes.c_int32(int(value))),
            "n4m_context_set_num_threads",
        )

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None and self._handle.value:
            lib.n4m_context_destroy(self._handle)
            self._handle = ctypes.c_void_p(0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._handle


__all__ = ["Context"]

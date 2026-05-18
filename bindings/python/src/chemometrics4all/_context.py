# SPDX-License-Identifier: CECILL-2.1
"""Thin wrapper around :c:type:`c4a_context_t`.

Most operators in libc4a don't currently consume a context (they take their
parameters at create-time and live independently). The context type is
exposed mainly for completeness of the FFI surface and for future use by
backend / seed configuration calls.
"""
from __future__ import annotations

import ctypes

from ._errors import check
from ._ffi import lib


class Context:
    """Opaque libc4a context.

    Created lazily; released by ``__del__`` or :meth:`close`. Not thread-safe.
    """

    def __init__(self) -> None:
        handle = ctypes.c_void_p()
        status = lib.c4a_context_create(ctypes.byref(handle))
        check(status, "c4a_context_create")
        self._handle = handle

    def set_seed(self, seed: int) -> None:
        check(lib.c4a_context_set_seed(self._handle, ctypes.c_uint64(seed)),
              "c4a_context_set_seed")

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None and self._handle.value:
            lib.c4a_context_destroy(self._handle)
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

"""Pythonic wrapper around p4a_config_t. Phase 0 surfaces a small subset of
setters (algorithm, n_components, tol); Phase 2 expands to the full set."""

from __future__ import annotations

import ctypes

from ._errors import Pls4allError
from ._ffi import lib
from ._types import Status


def _check(status: int) -> None:
    if status != Status.OK:
        raise Pls4allError(status, None)


class Config:
    """Phase 0 wrapper around p4a_config_t."""

    def __init__(self) -> None:
        h = ctypes.c_void_p(0)
        status = lib.p4a_config_create(ctypes.byref(h))
        if status != Status.OK or not h:
            raise Pls4allError(status, "p4a_config_create failed")
        self._h = h

    def __enter__(self) -> "Config":
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
            lib.p4a_config_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def n_components(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_config_get_n_components(self._h, ctypes.byref(out)))
        return int(out.value)

    @n_components.setter
    def n_components(self, value: int) -> None:
        _check(lib.p4a_config_set_n_components(self._h, ctypes.c_int32(int(value))))

    @property
    def tol(self) -> float:
        out = ctypes.c_double(0.0)
        _check(lib.p4a_config_get_tol(self._h, ctypes.byref(out)))
        return float(out.value)

    @tol.setter
    def tol(self, value: float) -> None:
        _check(lib.p4a_config_set_tol(self._h, ctypes.c_double(float(value))))

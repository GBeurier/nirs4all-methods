"""Pythonic wrapper around p4a_config_t.

The ctypes binding still exposes a small subset of setters; Phase 2 expands it
to the full model-estimator surface.
"""

from __future__ import annotations

import ctypes

from ._errors import Pls4allError
from ._ffi import lib
from ._types import Algorithm, Deflation, Solver, Status


def _check(status: int) -> None:
    if status != Status.OK:
        raise Pls4allError(status, None)


class Config:
    """Thin wrapper around p4a_config_t."""

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
    def algorithm(self) -> Algorithm:
        out = ctypes.c_int(0)
        _check(lib.p4a_config_get_algorithm(self._h, ctypes.byref(out)))
        return Algorithm(int(out.value))

    @algorithm.setter
    def algorithm(self, value: Algorithm | int) -> None:
        _check(lib.p4a_config_set_algorithm(self._h, ctypes.c_int(int(value))))

    @property
    def solver(self) -> Solver:
        out = ctypes.c_int(0)
        _check(lib.p4a_config_get_solver(self._h, ctypes.byref(out)))
        return Solver(int(out.value))

    @solver.setter
    def solver(self, value: Solver | int) -> None:
        _check(lib.p4a_config_set_solver(self._h, ctypes.c_int(int(value))))

    @property
    def deflation(self) -> Deflation:
        out = ctypes.c_int(0)
        _check(lib.p4a_config_get_deflation(self._h, ctypes.byref(out)))
        return Deflation(int(out.value))

    @deflation.setter
    def deflation(self, value: Deflation | int) -> None:
        _check(lib.p4a_config_set_deflation(self._h, ctypes.c_int(int(value))))

    @property
    def tol(self) -> float:
        out = ctypes.c_double(0.0)
        _check(lib.p4a_config_get_tol(self._h, ctypes.byref(out)))
        return float(out.value)

    @tol.setter
    def tol(self, value: float) -> None:
        _check(lib.p4a_config_set_tol(self._h, ctypes.c_double(float(value))))

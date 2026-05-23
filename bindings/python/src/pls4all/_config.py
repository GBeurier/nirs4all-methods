"""Pythonic wrapper around n4m_config_t.

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
    """Thin wrapper around n4m_config_t."""

    def __init__(self) -> None:
        h = ctypes.c_void_p(0)
        status = lib.n4m_config_create(ctypes.byref(h))
        if status != Status.OK or not h:
            raise Pls4allError(status, "n4m_config_create failed")
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
            lib.n4m_config_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def n_components(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_n_components(self._h, ctypes.byref(out)))
        return int(out.value)

    @n_components.setter
    def n_components(self, value: int) -> None:
        _check(lib.n4m_config_set_n_components(self._h, ctypes.c_int32(int(value))))

    @property
    def algorithm(self) -> Algorithm:
        out = ctypes.c_int(0)
        _check(lib.n4m_config_get_algorithm(self._h, ctypes.byref(out)))
        return Algorithm(int(out.value))

    @algorithm.setter
    def algorithm(self, value: Algorithm | int) -> None:
        _check(lib.n4m_config_set_algorithm(self._h, ctypes.c_int(int(value))))

    @property
    def solver(self) -> Solver:
        out = ctypes.c_int(0)
        _check(lib.n4m_config_get_solver(self._h, ctypes.byref(out)))
        return Solver(int(out.value))

    @solver.setter
    def solver(self, value: Solver | int) -> None:
        _check(lib.n4m_config_set_solver(self._h, ctypes.c_int(int(value))))

    @property
    def deflation(self) -> Deflation:
        out = ctypes.c_int(0)
        _check(lib.n4m_config_get_deflation(self._h, ctypes.byref(out)))
        return Deflation(int(out.value))

    @deflation.setter
    def deflation(self, value: Deflation | int) -> None:
        _check(lib.n4m_config_set_deflation(self._h, ctypes.c_int(int(value))))

    @property
    def tol(self) -> float:
        out = ctypes.c_double(0.0)
        _check(lib.n4m_config_get_tol(self._h, ctypes.byref(out)))
        return float(out.value)

    @tol.setter
    def tol(self, value: float) -> None:
        _check(lib.n4m_config_set_tol(self._h, ctypes.c_double(float(value))))

    # ---- center / scale / store_scores ----
    def _set_int32(self, setter, value: int | bool) -> None:
        _check(setter(self._h, ctypes.c_int32(int(bool(value)))))

    @property
    def center_x(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_center_x(self._h, ctypes.byref(out)))
        return int(out.value)

    @center_x.setter
    def center_x(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_center_x, value)

    @property
    def scale_x(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_scale_x(self._h, ctypes.byref(out)))
        return int(out.value)

    @scale_x.setter
    def scale_x(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_scale_x, value)

    @property
    def center_y(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_center_y(self._h, ctypes.byref(out)))
        return int(out.value)

    @center_y.setter
    def center_y(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_center_y, value)

    @property
    def scale_y(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_scale_y(self._h, ctypes.byref(out)))
        return int(out.value)

    @scale_y.setter
    def scale_y(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_scale_y, value)

    @property
    def store_scores(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_store_scores(self._h, ctypes.byref(out)))
        return int(out.value)

    @store_scores.setter
    def store_scores(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_store_scores, value)

    @property
    def sparse_simpls_legacy(self) -> int:
        """Switch sparse_simpls_fit to the pre-0.97.4 absolute soft-threshold
        algorithm. 0 (default) uses Chun & Keles 2010 spls (matches R `spls`)."""
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_sparse_simpls_legacy(self._h, ctypes.byref(out)))
        return int(out.value)

    @sparse_simpls_legacy.setter
    def sparse_simpls_legacy(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_sparse_simpls_legacy, value)

    @property
    def robust_pls_legacy(self) -> int:
        """Switch robust_pls_fit to the pre-0.97.4 Huber-IRLS over weighted
        SIMPLS algorithm. 0 (default) uses Partial Robust M-regression
        (Serneels et al. 2005) matching R `chemometrics::prm` bit-for-bit."""
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_robust_pls_legacy(self._h, ctypes.byref(out)))
        return int(out.value)

    @robust_pls_legacy.setter
    def robust_pls_legacy(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_robust_pls_legacy, value)

    @property
    def approximate_press_legacy(self) -> int:
        """Switch approximate_press_compute to the pre-0.97.4 Eastment-
        Krzanowski leverage-inflated training-residual approximation.
        0 (default) runs true leave-one-out PRESS over the SIMPLS kernel
        and matches R `pls::plsr(validation='LOO', method='simpls',
        scale=FALSE)$validation$PRESS` bit-for-bit."""
        out = ctypes.c_int32(0)
        _check(lib.n4m_config_get_approximate_press_legacy(
            self._h, ctypes.byref(out)))
        return int(out.value)

    @approximate_press_legacy.setter
    def approximate_press_legacy(self, value: int | bool) -> None:
        self._set_int32(lib.n4m_config_set_approximate_press_legacy, value)

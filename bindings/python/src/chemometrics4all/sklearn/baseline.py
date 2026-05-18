# SPDX-License-Identifier: CECILL-2.1
"""Baseline correction wrappers (Phase 5a operators)."""
from __future__ import annotations

import ctypes

from .._errors import check
from .._ffi import lib
from ._base import StatelessOperator


class Detrend(StatelessOperator):
    """Polynomial baseline subtraction."""

    _C_PREFIX = "c4a_pp_detrend"

    def __init__(self, polyorder: int = 1):
        super().__init__()
        self.polyorder = int(polyorder)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_detrend_create(ctypes.byref(h), ctypes.c_int32(self.polyorder)),
            "c4a_pp_detrend_create",
        )
        return h


class AsLS(StatelessOperator):
    """Asymmetric Least Squares (Eilers & Boelens 2005)."""

    _C_PREFIX = "c4a_pp_asls"

    def __init__(self, lam: float = 1e6, p: float = 1e-2,
                 max_iter: int = 50, tol: float = 1e-3):
        super().__init__()
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_asls_create(
                ctypes.byref(h),
                ctypes.c_double(self.lam),
                ctypes.c_double(self.p),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_asls_create",
        )
        return h


class AirPLS(StatelessOperator):
    """Adaptive iteratively reweighted PLS (Zhang 2010)."""

    _C_PREFIX = "c4a_pp_airpls"

    def __init__(self, lam: float = 1e6, max_iter: int = 50, tol: float = 1e-3):
        super().__init__()
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_airpls_create(
                ctypes.byref(h),
                ctypes.c_double(self.lam),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_airpls_create",
        )
        return h


__all__ = ["AirPLS", "AsLS", "Detrend"]

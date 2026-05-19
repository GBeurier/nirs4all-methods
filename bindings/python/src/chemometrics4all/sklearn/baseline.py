# SPDX-License-Identifier: CECILL-2.1
"""Baseline correction wrappers."""
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


class ArPLS(StatelessOperator):
    """Asymmetrically reweighted penalized least squares."""

    _C_PREFIX = "c4a_pp_arpls"

    def __init__(self, lam: float = 1e5, max_iter: int = 50, tol: float = 1e-3):
        super().__init__()
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_arpls_create(
                ctypes.byref(h),
                ctypes.c_double(self.lam),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_arpls_create",
        )
        return h


class ModPoly(StatelessOperator):
    """Modified polynomial baseline correction."""

    _C_PREFIX = "c4a_pp_modpoly"

    def __init__(self, polyorder: int = 2, max_iter: int = 250, tol: float = 1e-3):
        super().__init__()
        self.polyorder = int(polyorder)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_modpoly_create(
                ctypes.byref(h),
                ctypes.c_int32(self.polyorder),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_modpoly_create",
        )
        return h


class IModPoly(StatelessOperator):
    """Improved modified polynomial baseline correction."""

    _C_PREFIX = "c4a_pp_imodpoly"

    def __init__(self, polyorder: int = 2, max_iter: int = 250, tol: float = 1e-3):
        super().__init__()
        self.polyorder = int(polyorder)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_imodpoly_create(
                ctypes.byref(h),
                ctypes.c_int32(self.polyorder),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_imodpoly_create",
        )
        return h


class SNIP(StatelessOperator):
    """Statistics-sensitive nonlinear iterative peak-clipping baseline."""

    _C_PREFIX = "c4a_pp_snip"

    def __init__(self, max_half_window: int = 20):
        super().__init__()
        self.max_half_window = int(max_half_window)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_snip_create(
                ctypes.byref(h), ctypes.c_int32(self.max_half_window)
            ),
            "c4a_pp_snip_create",
        )
        return h


class RollingBall(StatelessOperator):
    """Rolling-ball morphological baseline correction."""

    _C_PREFIX = "c4a_pp_rolling_ball"

    def __init__(self, half_window: int = 20, smooth_half_window: int = 0):
        super().__init__()
        self.half_window = int(half_window)
        self.smooth_half_window = int(smooth_half_window)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_rolling_ball_create(
                ctypes.byref(h),
                ctypes.c_int32(self.half_window),
                ctypes.c_int32(self.smooth_half_window),
            ),
            "c4a_pp_rolling_ball_create",
        )
        return h


class IAsLS(StatelessOperator):
    """Improved asymmetric least-squares baseline correction."""

    _C_PREFIX = "c4a_pp_iasls"

    def __init__(
        self,
        lam: float = 1e6,
        p: float = 1e-2,
        lam_1: float = 1e-4,
        polyorder: int = 2,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
    ):
        super().__init__()
        self.lam = float(lam)
        self.p = float(p)
        self.lam_1 = float(lam_1)
        self.polyorder = int(polyorder)
        self.diff_order = int(diff_order)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_iasls_create_ex(
                ctypes.byref(h),
                ctypes.c_double(self.lam),
                ctypes.c_double(self.p),
                ctypes.c_double(self.lam_1),
                ctypes.c_int32(self.polyorder),
                ctypes.c_int32(self.diff_order),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_iasls_create_ex",
        )
        return h


class BEADS(StatelessOperator):
    """Baseline estimation and denoising with sparsity."""

    _C_PREFIX = "c4a_pp_beads"

    def __init__(
        self,
        lam_0: float = 100.0,
        lam_1: float = 0.5,
        lam_2: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-3,
    ):
        super().__init__()
        self.lam_0 = float(lam_0)
        self.lam_1 = float(lam_1)
        self.lam_2 = float(lam_2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_pp_beads_create(
                ctypes.byref(h),
                ctypes.c_double(self.lam_0),
                ctypes.c_double(self.lam_1),
                ctypes.c_double(self.lam_2),
                ctypes.c_int32(self.max_iter),
                ctypes.c_double(self.tol),
            ),
            "c4a_pp_beads_create",
        )
        return h


__all__ = [
    "AirPLS",
    "ArPLS",
    "AsLS",
    "BEADS",
    "Detrend",
    "IAsLS",
    "IModPoly",
    "ModPoly",
    "RollingBall",
    "SNIP",
]

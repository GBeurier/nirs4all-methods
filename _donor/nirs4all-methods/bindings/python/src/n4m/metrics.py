# SPDX-License-Identifier: CECILL-2.1
"""NIRS metrics and diagnostics backed by libn4m."""
from __future__ import annotations

import ctypes

import numpy as np

from ._errors import check
from ._ffi import lib
from ._matrix import as_f64_2d, numpy_to_view
from ._types import TransferMetrics


def _as_f64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _metric(name: str, y_true, y_pred) -> float:
    yt = _as_f64_1d(y_true, "y_true")
    yp = _as_f64_1d(y_pred, "y_pred")
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    out = ctypes.c_double()
    fn = getattr(lib, f"n4m_metric_{name}")
    check(
        fn(
            yt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            yp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(yt.size),
            ctypes.byref(out),
        ),
        f"n4m_metric_{name}",
    )
    return float(out.value)


def rmse(y_true, y_pred) -> float:
    return _metric("rmse", y_true, y_pred)


def mae(y_true, y_pred) -> float:
    return _metric("mae", y_true, y_pred)


def bias(y_true, y_pred) -> float:
    return _metric("bias", y_true, y_pred)


def sep(y_true, y_pred) -> float:
    return _metric("sep", y_true, y_pred)


def rpd(y_true, y_pred) -> float:
    return _metric("rpd", y_true, y_pred)


def rpiq(y_true, y_pred) -> float:
    return _metric("rpiq", y_true, y_pred)


def r2(y_true, y_pred) -> float:
    return _metric("r2", y_true, y_pred)


def nrmse(y_true, y_pred) -> float:
    return _metric("nrmse", y_true, y_pred)


def hotelling_t2(X, n_components: int, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    X = as_f64_2d(X)
    values = np.empty(X.shape[0], dtype=np.float64, order="C")
    ucl = ctypes.c_double()
    check(
        lib.n4m_util_hotelling_t2(
            numpy_to_view(X),
            ctypes.c_int32(int(n_components)),
            ctypes.c_double(float(alpha)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(values.size),
            ctypes.byref(ucl),
        ),
        "n4m_util_hotelling_t2",
    )
    return values, float(ucl.value)


def q_residuals(X, n_components: int, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    X = as_f64_2d(X)
    values = np.empty(X.shape[0], dtype=np.float64, order="C")
    ucl = ctypes.c_double()
    check(
        lib.n4m_util_q_residuals(
            numpy_to_view(X),
            ctypes.c_int32(int(n_components)),
            ctypes.c_double(float(alpha)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(values.size),
            ctypes.byref(ucl),
        ),
        "n4m_util_q_residuals",
    )
    return values, float(ucl.value)


def transfer_metrics(
    X_source,
    X_target,
    n_components: int = 10,
    k_neighbors: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    source = as_f64_2d(X_source)
    target = as_f64_2d(X_target)
    out = TransferMetrics()
    check(
        lib.n4m_transfer_metrics_compute(
            numpy_to_view(source),
            numpy_to_view(target),
            ctypes.c_int32(int(n_components)),
            ctypes.c_int32(int(k_neighbors)),
            ctypes.c_uint64(int(seed)),
            ctypes.byref(out),
        ),
        "n4m_transfer_metrics_compute",
    )
    return {name: float(getattr(out, name)) for name, _ctype in TransferMetrics._fields_}


__all__ = [
    "bias",
    "hotelling_t2",
    "mae",
    "nrmse",
    "q_residuals",
    "r2",
    "rmse",
    "rpd",
    "rpiq",
    "sep",
    "transfer_metrics",
]

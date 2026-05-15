"""Python wrappers for the universal method-result handle and the
per-method fit entry points exposed in p4a.h §16.

Each fit function returns a `MethodResult` object whose `.matrix(name)`,
`.vector(name)` and `.scalar(name)` accessors yield NumPy arrays /
scalars copied out of the C-owned storage. The underlying handle is
freed when the wrapper is closed or garbage-collected.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ._config import Config
from ._context import Context
from ._errors import Pls4allError
from ._ffi import MatrixView, lib
from ._model import _matrix_view, _as_float64_contiguous
from ._types import Status


def _check(status_int: int, ctx: Context | None = None) -> None:
    if status_int == Status.OK:
        return
    msg = None
    if ctx is not None:
        raw = lib.p4a_context_last_error(ctx.handle)
        if raw:
            msg = raw.decode("utf-8")
    raise Pls4allError(status_int, msg)


# ---- FFI signatures for the universal handle + fit entry points ----------

lib.p4a_method_result_destroy.restype = None
lib.p4a_method_result_destroy.argtypes = [ctypes.c_void_p]

lib.p4a_method_result_get_double_matrix.restype = ctypes.c_int
lib.p4a_method_result_get_double_matrix.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]

lib.p4a_method_result_get_int_vector.restype = ctypes.c_int
lib.p4a_method_result_get_int_vector.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ctypes.POINTER(ctypes.c_int32),
]

lib.p4a_method_result_get_scalar.restype = ctypes.c_int
lib.p4a_method_result_get_scalar.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_double),
]

lib.p4a_sparse_simpls_fit.restype = ctypes.c_int
lib.p4a_sparse_simpls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_di_pls_fit.restype = ctypes.c_int
lib.p4a_di_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_recursive_pls_run.restype = ctypes.c_int
lib.p4a_recursive_pls_run.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_cppls_fit.restype = ctypes.c_int
lib.p4a_cppls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_weighted_pls_fit.restype = ctypes.c_int
lib.p4a_weighted_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_robust_pls_fit.restype = ctypes.c_int
lib.p4a_robust_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_ridge_pls_fit.restype = ctypes.c_int
lib.p4a_ridge_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_continuum_regression_fit.restype = ctypes.c_int
lib.p4a_continuum_regression_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_n_pls_fit.restype = ctypes.c_int
lib.p4a_n_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_kernel_pls_fit.restype = ctypes.c_int
lib.p4a_kernel_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_o2pls_fit.restype = ctypes.c_int
lib.p4a_o2pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.p4a_approximate_press_compute.restype = ctypes.c_int
lib.p4a_approximate_press_compute.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]


class MethodResult:
    """Owning wrapper around `p4a_method_result_t`."""

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._h = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __copy__(self):
        raise TypeError("MethodResult is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("MethodResult is not copyable")

    def close(self) -> None:
        if self._h:
            lib.p4a_method_result_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    def matrix(self, name: str) -> np.ndarray:
        data_ptr = ctypes.POINTER(ctypes.c_double)()
        rows = ctypes.c_int64(0)
        cols = ctypes.c_int64(0)
        _check(lib.p4a_method_result_get_double_matrix(
            self._h, name.encode("utf-8"),
            ctypes.byref(data_ptr), ctypes.byref(rows), ctypes.byref(cols),
        ))
        n = int(rows.value) * int(cols.value)
        if n == 0:
            return np.empty((int(rows.value), int(cols.value)),
                            dtype=np.float64)
        buf_type = ctypes.c_double * n
        buffer = buf_type.from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buffer, dtype=np.float64, count=n)
        return np.array(arr.reshape(int(rows.value), int(cols.value)),
                        copy=True)

    def vector_int(self, name: str) -> np.ndarray:
        data_ptr = ctypes.POINTER(ctypes.c_int32)()
        size = ctypes.c_int32(0)
        _check(lib.p4a_method_result_get_int_vector(
            self._h, name.encode("utf-8"),
            ctypes.byref(data_ptr), ctypes.byref(size),
        ))
        n = int(size.value)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        buf_type = ctypes.c_int32 * n
        buffer = buf_type.from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buffer, dtype=np.int32, count=n)
        return np.array(arr, copy=True)

    def scalar(self, name: str) -> float:
        out = ctypes.c_double(0.0)
        _check(lib.p4a_method_result_get_scalar(
            self._h, name.encode("utf-8"), ctypes.byref(out),
        ))
        return float(out.value)


def _resolve_handle(out: ctypes.c_void_p, ctx: Context, fn_name: str) -> MethodResult:
    if not out:
        raise Pls4allError(int(Status.ERR_INTERNAL),
                           f"{fn_name} returned a NULL handle")
    try:
        return MethodResult(out)
    except BaseException:
        lib.p4a_method_result_destroy(out)
        raise


def sparse_simpls_fit(ctx: Context, cfg: Config,
                       X: Any, Y: Any,
                       sparsity_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_sparse_simpls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(sparsity_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_sparse_simpls_fit")


def di_pls_fit(ctx: Context, cfg: Config,
                X_source: Any, Y_source: Any, X_target: Any,
                di_lambda: float) -> MethodResult:
    Xs = _as_float64_contiguous(X_source)
    Ys = _as_float64_contiguous(Y_source)
    Xt = _as_float64_contiguous(X_target)
    xs_view = _matrix_view(Xs)
    ys_view = _matrix_view(Ys)
    xt_view = _matrix_view(Xt)
    out = ctypes.c_void_p(0)
    status = lib.p4a_di_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(xs_view), ctypes.byref(ys_view), ctypes.byref(xt_view),
        ctypes.c_double(float(di_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_di_pls_fit")


def recursive_pls_run(ctx: Context, cfg: Config,
                       X: Any, Y: Any,
                       window_size: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_recursive_pls_run(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(window_size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_recursive_pls_run")


def cppls_fit(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               gamma: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_cppls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(gamma)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_cppls_fit")


def weighted_pls_fit(ctx: Context, cfg: Config,
                      X: Any, Y: Any,
                      sample_weights: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    w_arr = np.ascontiguousarray(sample_weights, dtype=np.float64).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_weighted_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(w_arr.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_weighted_pls_fit")


def robust_pls_fit(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    huber_k: float,
                    max_irls_iter: int = 5) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_robust_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(huber_k)),
        ctypes.c_int32(int(max_irls_iter)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_robust_pls_fit")


def ridge_pls_fit(ctx: Context, cfg: Config,
                   X: Any, Y: Any,
                   ridge_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_ridge_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(ridge_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_ridge_pls_fit")


def continuum_regression_fit(ctx: Context, cfg: Config,
                              X: Any, Y: Any,
                              tau: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_continuum_regression_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(tau)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_continuum_regression_fit")


def n_pls_fit(ctx: Context, cfg: Config,
               X_flat: Any, mode_j: int, mode_k: int,
               Y: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X_flat)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_n_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        ctypes.c_int32(int(mode_j)), ctypes.c_int32(int(mode_k)),
        ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_n_pls_fit")


def kernel_pls_fit(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    kernel_type: int,
                    gamma: float = 0.0,
                    coef0: float = 1.0,
                    degree: int = 3) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_kernel_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.c_int32(int(kernel_type)),
        ctypes.c_double(float(gamma)), ctypes.c_double(float(coef0)),
        ctypes.c_int32(int(degree)),
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_kernel_pls_fit")


def o2pls_fit(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               n_predictive: int,
               n_x_orthogonal: int = 0,
               n_y_orthogonal: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_o2pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_predictive)),
        ctypes.c_int32(int(n_x_orthogonal)),
        ctypes.c_int32(int(n_y_orthogonal)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_o2pls_fit")


def approximate_press_compute(ctx: Context, cfg: Config,
                                X: Any, Y: Any,
                                max_components: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.p4a_approximate_press_compute(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(max_components)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "p4a_approximate_press_compute")


__all__ = [
    "MethodResult",
    "sparse_simpls_fit",
    "di_pls_fit",
    "recursive_pls_run",
    "cppls_fit",
    "weighted_pls_fit",
    "robust_pls_fit",
    "ridge_pls_fit",
    "continuum_regression_fit",
    "n_pls_fit",
    "kernel_pls_fit",
    "o2pls_fit",
    "approximate_press_compute",
]

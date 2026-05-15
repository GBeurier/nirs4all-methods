"""Python wrappers for the AOM/POP public C ABI."""

from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import Sequence

from ._config import Config
from ._context import Context
from ._errors import Pls4allError
from ._ffi import MatrixView, lib
from ._types import Dtype, Status


def _check(status_int: int, ctx: Context | None = None) -> None:
    if status_int == Status.OK:
        return
    msg = None
    if ctx is not None:
        raw = lib.p4a_context_last_error(ctx.handle)
        if raw:
            msg = raw.decode("utf-8")
    raise Pls4allError(status_int, msg)


class OperatorKind(IntEnum):
    IDENTITY            = 0
    CENTER              = 1
    AUTOSCALE           = 2
    PARETO_SCALE        = 3
    SNV                 = 4
    MSC                 = 5
    EMSC                = 6
    DETREND_POLY        = 7
    SAVGOL_SMOOTH       = 8
    SAVGOL_DERIVATIVE   = 9
    NORRIS_WILLIAMS     = 10
    ASLS_BASELINE       = 11
    OSC                 = 12
    EPO                 = 13
    WAVELET_DENOISE     = 14
    FINITE_DIFFERENCE   = 15
    WHITTAKER           = 16
    FCK                 = 17


class _OwningHandle:
    """Base class for non-copyable owning handles over a C resource."""

    def __copy__(self):
        raise TypeError(f"{type(self).__name__} is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError(f"{type(self).__name__} is not copyable")


class OperatorBank(_OwningHandle):
    """Owning handle around p4a_operator_bank_t."""

    def __init__(self) -> None:
        handle = ctypes.c_void_p(0)
        status = lib.p4a_operator_bank_create(ctypes.byref(handle))
        if status != Status.OK or not handle:
            raise Pls4allError(status, "p4a_operator_bank_create failed")
        self._h = handle

    def __enter__(self) -> "OperatorBank":
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
            lib.p4a_operator_bank_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    def add(self, kind: OperatorKind | int, params: Sequence[float] | None = None) -> None:
        params_seq = list(params) if params is not None else []
        n = len(params_seq)
        buf = (ctypes.c_double * n)(*params_seq) if n > 0 else None
        buf_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_double)) if buf else None
        _check(lib.p4a_operator_bank_add(self._h, int(kind), buf_ptr, ctypes.c_int32(n)))

    def __len__(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_operator_bank_size(self._h, ctypes.byref(out)))
        return int(out.value)


class ValidationPlan(_OwningHandle):
    """Owning handle around p4a_validation_plan_t."""

    def __init__(self) -> None:
        handle = ctypes.c_void_p(0)
        status = lib.p4a_validation_plan_create(ctypes.byref(handle))
        if status != Status.OK or not handle:
            raise Pls4allError(status, "p4a_validation_plan_create failed")
        self._h = handle

    def __enter__(self) -> "ValidationPlan":
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
            lib.p4a_validation_plan_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def n_samples(self) -> int:
        out = ctypes.c_int64(0)
        _check(lib.p4a_validation_plan_get_n_samples(self._h, ctypes.byref(out)))
        return int(out.value)

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        _check(lib.p4a_validation_plan_set_n_samples(self._h, ctypes.c_int64(int(value))))

    def __len__(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_validation_plan_get_n_folds(self._h, ctypes.byref(out)))
        return int(out.value)

    def add_fold(self, train: Sequence[int], test: Sequence[int]) -> None:
        train_arr = (ctypes.c_int64 * len(train))(*train)
        test_arr = (ctypes.c_int64 * len(test))(*test)
        _check(lib.p4a_validation_plan_add_fold(
            self._h,
            train_arr, ctypes.c_int64(len(train)),
            test_arr, ctypes.c_int64(len(test)),
        ))


def _row_major_view(data: ctypes.Array, rows: int, cols: int) -> MatrixView:
    view = MatrixView()
    view.data = ctypes.cast(data, ctypes.c_void_p)
    view.rows = rows
    view.cols = cols
    view.row_stride = cols if cols > 0 else 1
    view.col_stride = 1
    view.dtype = int(Dtype.F64)
    view.reserved0 = 0
    return view


def _make_view_from_seq(values: Sequence[float], rows: int, cols: int) -> tuple[MatrixView, ctypes.Array]:
    """Build a row-major MatrixView over a contiguous f64 buffer.

    Returns the view and the backing buffer; the caller MUST keep the buffer
    alive for as long as the view is being used.
    """
    if len(values) != rows * cols:
        raise ValueError(
            f"values length {len(values)} does not match shape ({rows}, {cols})")
    buf = (ctypes.c_double * (rows * cols))(*values)
    return _row_major_view(buf, rows, cols), buf


class _AomResultBase(_OwningHandle):
    """Common destroy + handle plumbing for AOM/POP result wrappers."""

    _destroy_fn = None  # subclass overrides

    def __init__(self, raw_handle: ctypes.c_void_p) -> None:
        self._h = raw_handle

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._h and self._destroy_fn is not None:
            self._destroy_fn(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h


def _read_int32(getter, handle) -> int:
    out = ctypes.c_int32(0)
    _check(getter(handle, ctypes.byref(out)))
    return int(out.value)


def _read_double(getter, handle) -> float:
    out = ctypes.c_double(0.0)
    _check(getter(handle, ctypes.byref(out)))
    return float(out.value)


def _read_buf_int(getter, handle, ctype) -> list[int]:
    ptr = ctypes.POINTER(ctype)()
    size = ctypes.c_int32(0)
    _check(getter(handle, ctypes.byref(ptr), ctypes.byref(size)))
    return [int(ptr[i]) for i in range(size.value)] if size.value > 0 else []


def _read_buf_double(getter, handle) -> list[float]:
    ptr = ctypes.POINTER(ctypes.c_double)()
    size = ctypes.c_int32(0)
    _check(getter(handle, ctypes.byref(ptr), ctypes.byref(size)))
    return [float(ptr[i]) for i in range(size.value)] if size.value > 0 else []


def _read_matrix_double_i32(getter, handle) -> tuple[list[float], int, int]:
    ptr = ctypes.POINTER(ctypes.c_double)()
    rows = ctypes.c_int32(0)
    cols = ctypes.c_int32(0)
    _check(getter(handle, ctypes.byref(ptr), ctypes.byref(rows), ctypes.byref(cols)))
    n = int(rows.value) * int(cols.value)
    values = [float(ptr[i]) for i in range(n)] if n > 0 else []
    return values, int(rows.value), int(cols.value)


def _read_matrix_double_i64(getter, handle) -> tuple[list[float], int, int]:
    ptr = ctypes.POINTER(ctypes.c_double)()
    rows = ctypes.c_int64(0)
    cols = ctypes.c_int64(0)
    _check(getter(handle, ctypes.byref(ptr), ctypes.byref(rows), ctypes.byref(cols)))
    n = int(rows.value) * int(cols.value)
    values = [float(ptr[i]) for i in range(n)] if n > 0 else []
    return values, int(rows.value), int(cols.value)


class AomGlobalResult(_AomResultBase):
    _destroy_fn = staticmethod(lib.p4a_aom_global_result_destroy)

    @property
    def n_operators(self) -> int:
        return _read_int32(lib.p4a_aom_global_result_get_n_operators, self._h)

    @property
    def max_components(self) -> int:
        return _read_int32(lib.p4a_aom_global_result_get_max_components, self._h)

    @property
    def selected_operator_index(self) -> int:
        return _read_int32(lib.p4a_aom_global_result_get_selected_operator_index, self._h)

    @property
    def selected_n_components(self) -> int:
        return _read_int32(lib.p4a_aom_global_result_get_selected_n_components, self._h)

    @property
    def best_score(self) -> float:
        return _read_double(lib.p4a_aom_global_result_get_best_score, self._h)

    @property
    def operator_kinds(self) -> list[int]:
        return _read_buf_int(lib.p4a_aom_global_result_get_operator_kinds,
                             self._h, ctypes.c_int)

    @property
    def operator_scores(self) -> list[float]:
        return _read_buf_double(lib.p4a_aom_global_result_get_operator_scores,
                                self._h)

    @property
    def rmse_curves(self) -> tuple[list[float], int, int]:
        return _read_matrix_double_i32(
            lib.p4a_aom_global_result_get_rmse_curves, self._h)

    @property
    def predictions(self) -> tuple[list[float], int, int]:
        return _read_matrix_double_i64(
            lib.p4a_aom_global_result_get_predictions, self._h)


class AomPerComponentResult(_AomResultBase):
    _destroy_fn = staticmethod(lib.p4a_aom_per_component_result_destroy)

    @property
    def n_operators(self) -> int:
        return _read_int32(lib.p4a_aom_per_component_result_get_n_operators, self._h)

    @property
    def max_components(self) -> int:
        return _read_int32(lib.p4a_aom_per_component_result_get_max_components, self._h)

    @property
    def selected_n_components(self) -> int:
        return _read_int32(
            lib.p4a_aom_per_component_result_get_selected_n_components, self._h)

    @property
    def best_score(self) -> float:
        return _read_double(
            lib.p4a_aom_per_component_result_get_best_score, self._h)

    @property
    def operator_kinds(self) -> list[int]:
        return _read_buf_int(
            lib.p4a_aom_per_component_result_get_operator_kinds,
            self._h, ctypes.c_int)

    @property
    def selected_operator_indices(self) -> list[int]:
        return _read_buf_int(
            lib.p4a_aom_per_component_result_get_selected_operator_indices,
            self._h, ctypes.c_int32)

    @property
    def component_scores(self) -> tuple[list[float], int, int]:
        return _read_matrix_double_i32(
            lib.p4a_aom_per_component_result_get_component_scores, self._h)

    @property
    def prefix_scores(self) -> list[float]:
        return _read_buf_double(
            lib.p4a_aom_per_component_result_get_prefix_scores, self._h)

    @property
    def predictions(self) -> tuple[list[float], int, int]:
        return _read_matrix_double_i64(
            lib.p4a_aom_per_component_result_get_predictions, self._h)


def aom_global_select(
    ctx: Context,
    cfg: Config,
    bank: OperatorBank,
    X: Sequence[float],
    Y: Sequence[float],
    plan: ValidationPlan,
    max_components: int,
    *,
    x_rows: int,
    x_cols: int,
    y_rows: int,
    y_cols: int,
) -> AomGlobalResult:
    x_view, _x_buf = _make_view_from_seq(X, x_rows, x_cols)
    y_view, _y_buf = _make_view_from_seq(Y, y_rows, y_cols)
    out = ctypes.c_void_p(0)
    status = lib.p4a_aom_global_select(
        ctx.handle, cfg.handle, bank.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan.handle, ctypes.c_int32(int(max_components)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    if not out:
        raise Pls4allError(int(Status.ERR_INTERNAL),
                           "p4a_aom_global_select returned a NULL handle")
    try:
        return AomGlobalResult(out)
    except BaseException:
        lib.p4a_aom_global_result_destroy(out)
        raise


def aom_per_component_select(
    ctx: Context,
    cfg: Config,
    bank: OperatorBank,
    X: Sequence[float],
    Y: Sequence[float],
    plan: ValidationPlan,
    max_components: int,
    *,
    x_rows: int,
    x_cols: int,
    y_rows: int,
    y_cols: int,
) -> AomPerComponentResult:
    x_view, _x_buf = _make_view_from_seq(X, x_rows, x_cols)
    y_view, _y_buf = _make_view_from_seq(Y, y_rows, y_cols)
    out = ctypes.c_void_p(0)
    status = lib.p4a_aom_per_component_select(
        ctx.handle, cfg.handle, bank.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan.handle, ctypes.c_int32(int(max_components)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    if not out:
        raise Pls4allError(int(Status.ERR_INTERNAL),
                           "p4a_aom_per_component_select returned a NULL handle")
    try:
        return AomPerComponentResult(out)
    except BaseException:
        lib.p4a_aom_per_component_result_destroy(out)
        raise


__all__ = [
    "AomGlobalResult",
    "AomPerComponentResult",
    "OperatorBank",
    "OperatorKind",
    "ValidationPlan",
    "aom_global_select",
    "aom_per_component_select",
]

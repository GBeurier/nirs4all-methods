"""Python wrappers for `p4a_model_fit`, `p4a_model_predict_alloc`,
`p4a_model_get_array` and the `p4a_array_t` accessors.

Uses NumPy zero-copy `p4a_matrix_view_t` when an ndarray is contiguous and
float64. Falls back to `numpy.ascontiguousarray` (one copy) otherwise.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import Any

import numpy as np

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


def _as_float64_contiguous(array_like: Any) -> np.ndarray:
    arr = np.ascontiguousarray(array_like, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D ndarray, got ndim={arr.ndim}")
    return arr


def _matrix_view(array: np.ndarray) -> MatrixView:
    if not array.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")
    if array.dtype != np.float64:
        raise ValueError(f"array dtype must be float64, got {array.dtype}")
    view = MatrixView()
    view.data = array.ctypes.data if array.size > 0 else 0
    view.rows = int(array.shape[0])
    view.cols = int(array.shape[1]) if array.ndim > 1 else 1
    view.row_stride = view.cols if view.cols > 0 else 1
    view.col_stride = 1
    view.dtype = int(Dtype.F64)
    view.reserved0 = 0
    return view


class ModelArrayKind(IntEnum):
    COEFFICIENTS = 0
    INTERCEPT = 1
    X_MEAN = 2
    X_SCALE = 3
    Y_MEAN = 4
    Y_SCALE = 5
    WEIGHTS_W = 6
    LOADINGS_P = 7
    Y_LOADINGS_Q = 8
    ROTATIONS_R = 9
    SCORES_T = 10
    Y_SCORES_U = 11


class _Array:
    """Owning wrapper for `p4a_array_t*`.

    Provides a NumPy view that shares the array's underlying buffer; the
    caller must keep the `_Array` alive while the NumPy view is in use.
    """

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._h = handle

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __copy__(self):
        raise TypeError("_Array is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("_Array is not copyable")

    def close(self) -> None:
        if self._h:
            lib.p4a_array_free(self._h)
            self._h = ctypes.c_void_p(0)

    def shape(self) -> tuple[int, int]:
        rows = ctypes.c_int64(0)
        cols = ctypes.c_int64(0)
        _check(lib.p4a_array_shape(self._h, ctypes.byref(rows), ctypes.byref(cols)))
        return int(rows.value), int(cols.value)

    def view(self) -> np.ndarray:
        mv = MatrixView()
        _check(lib.p4a_array_view(self._h, ctypes.byref(mv)))
        if mv.rows == 0 or mv.cols == 0:
            return np.empty((mv.rows, mv.cols), dtype=np.float64)
        # numpy.ctypeslib.as_array borrows the buffer; we re-cast to float64.
        buf_type = ctypes.c_double * (mv.rows * mv.cols)
        buffer = buf_type.from_address(int(mv.data))
        arr = np.frombuffer(buffer, dtype=np.float64,
                            count=mv.rows * mv.cols)
        return arr.reshape(int(mv.rows), int(mv.cols))

    def copy(self) -> np.ndarray:
        return np.array(self.view(), copy=True)


class Model:
    """Owning handle around `p4a_model_t`. Constructed by `Model.fit(...)`.

    Stores a reference to the input arrays during fit so the NumPy buffers
    remain valid for the duration of the C call.
    """

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._h = handle

    @classmethod
    def fit(cls, ctx: Context, cfg: Config,
            X: Any, Y: Any) -> "Model":
        X_arr = _as_float64_contiguous(X)
        Y_arr = _as_float64_contiguous(Y)
        x_view = _matrix_view(X_arr)
        y_view = _matrix_view(Y_arr)
        out = ctypes.c_void_p(0)
        status = lib.p4a_model_fit(
            ctx.handle, cfg.handle,
            ctypes.byref(x_view), ctypes.byref(y_view),
            ctypes.byref(out),
        )
        _check(status, ctx)
        if not out:
            raise Pls4allError(int(Status.ERR_INTERNAL),
                               "p4a_model_fit returned NULL handle")
        try:
            return cls(out)
        except BaseException:
            lib.p4a_model_destroy(out)
            raise

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __copy__(self):
        raise TypeError("Model is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("Model is not copyable")

    def close(self) -> None:
        if self._h:
            lib.p4a_model_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def n_components(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_model_get_n_components(self._h, ctypes.byref(out)))
        return int(out.value)

    @property
    def n_features(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_model_get_n_features(self._h, ctypes.byref(out)))
        return int(out.value)

    @property
    def n_targets(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.p4a_model_get_n_targets(self._h, ctypes.byref(out)))
        return int(out.value)

    def predict(self, ctx: Context, X: Any) -> np.ndarray:
        X_arr = _as_float64_contiguous(X)
        x_view = _matrix_view(X_arr)
        out_handle = ctypes.c_void_p(0)
        _check(
            lib.p4a_model_predict_alloc(
                ctx.handle, self._h,
                ctypes.byref(x_view), ctypes.byref(out_handle),
            ),
            ctx,
        )
        wrapper = _Array(out_handle)
        try:
            return wrapper.copy()
        finally:
            wrapper.close()

    def transform(self, ctx: Context, X: Any) -> np.ndarray:
        X_arr = _as_float64_contiguous(X)
        x_view = _matrix_view(X_arr)
        out_handle = ctypes.c_void_p(0)
        _check(
            lib.p4a_model_transform_alloc(
                ctx.handle, self._h,
                ctypes.byref(x_view), ctypes.byref(out_handle),
            ),
            ctx,
        )
        wrapper = _Array(out_handle)
        try:
            return wrapper.copy()
        finally:
            wrapper.close()

    def get_array(self, ctx: Context, kind: ModelArrayKind) -> np.ndarray:
        out_handle = ctypes.c_void_p(0)
        _check(
            lib.p4a_model_get_array(
                ctx.handle, self._h, int(kind),
                ctypes.byref(out_handle),
            ),
            ctx,
        )
        wrapper = _Array(out_handle)
        try:
            return wrapper.copy()
        finally:
            wrapper.close()

    @property
    def coefficients(self) -> np.ndarray:
        with Context() as ctx:
            return self.get_array(ctx, ModelArrayKind.COEFFICIENTS)

    def to_bytes(self) -> bytes:
        """Serialize the fitted model to a portable byte buffer (.n4a wire
        format, P4A_SERIALIZATION_FORMAT_VERSION = 1). The buffer is
        forward-compatible across pls4all minor versions; round-trip is
        bit-exact for all model arrays."""
        size = ctypes.c_size_t(0)
        _check(lib.p4a_model_export_size(self._h, ctypes.byref(size)))
        if size.value == 0:
            raise Pls4allError(int(Status.ERR_INTERNAL),
                                "p4a_model_export_size returned 0")
        buf = (ctypes.c_uint8 * int(size.value))()
        written = ctypes.c_size_t(0)
        _check(lib.p4a_model_export_to_buffer(
            self._h, buf, size, ctypes.byref(written)))
        return bytes(bytearray(buf)[: int(written.value)])

    @classmethod
    def from_bytes(cls, ctx: Context, payload: bytes) -> "Model":
        """Deserialize a model from a `.n4a` wire-format buffer produced
        by `to_bytes()`. Raises Pls4allError on ABI MAJOR mismatch or
        when the writer ABI MINOR exceeds the reader's."""
        if not payload:
            raise Pls4allError(int(Status.ERR_INVALID_ARGUMENT),
                                "from_bytes: empty payload")
        buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        out = ctypes.c_void_p(0)
        _check(lib.p4a_model_import_from_buffer(
            ctx.handle, buf, ctypes.c_size_t(len(payload)),
            ctypes.byref(out)), ctx)
        if not out:
            raise Pls4allError(int(Status.ERR_INTERNAL),
                                "p4a_model_import_from_buffer returned NULL")
        try:
            return cls(out)
        except BaseException:
            lib.p4a_model_destroy(out)
            raise


__all__ = [
    "Model",
    "ModelArrayKind",
    "_as_float64_contiguous",
    "_matrix_view",
]

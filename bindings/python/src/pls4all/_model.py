"""Python wrappers for `n4m_model_fit`, `n4m_model_predict_alloc`,
`n4m_model_get_array` and the `n4m_array_t` accessors.

Uses NumPy zero-copy `n4m_matrix_view_t` when an ndarray is contiguous and
float64. Falls back to `numpy.ascontiguousarray` (one copy) otherwise.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from ._config import Config
from ._context import Context
from ._errors import Pls4allError
from ._ffi import MatrixView, SerializedModelInfoV1, SerializedPipelineInfoV1, lib
from ._types import Dtype, Status


def _check(status_int: int, ctx: Context | None = None) -> None:
    if status_int == Status.OK:
        return
    msg = None
    if ctx is not None:
        raw = lib.n4m_context_last_error(ctx.handle)
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


SERIALIZED_MODEL_CAPABILITY_PREDICT = 1 << 0
SERIALIZED_MODEL_CAPABILITY_TRANSFORM = 1 << 1
SERIALIZED_MODEL_CAPABILITY_AFFINE = 1 << 2
SERIALIZED_MODEL_CAPABILITY_PIPELINE = 1 << 3


@dataclass(frozen=True)
class SerializedPipelineInfo:
    """Validated bounded preprocessing plan embedded in N4MM v2."""

    schema_version: int
    operators: tuple[int, ...]
    savgol_window: int
    savgol_poly_degree: int
    savgol_derivative: int
    semantic_profile: int
    savgol_delta: float
    raw_n_features: int
    model_n_features: int
    fingerprint_algorithm: int
    fingerprint: int
    snv_axis: int
    snv_with_mean: bool
    snv_with_std: bool
    snv_ddof: int
    savgol_mode: int
    savgol_cval: float


@dataclass(frozen=True)
class SerializedModelInfo:
    """Authoritative metadata from a fully validated N4MM payload."""

    schema_version: int
    format_version: int
    writer_abi: tuple[int, int, int]
    algorithm: int
    solver: int
    deflation: int
    training_samples: int
    n_features: int
    n_targets: int
    n_components: int
    capabilities: int
    pipeline: SerializedPipelineInfo | None


def inspect_n4mm(payload: bytes) -> SerializedModelInfo:
    """Validate and inspect N4MM bytes without importing model state."""

    if not payload:
        raise Pls4allError(int(Status.ERR_CORRUPT_BUFFER))
    buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    raw = SerializedModelInfoV1()
    raw_pipeline = SerializedPipelineInfoV1()
    _check(
        lib.n4m_serialization_inspect_model_v1(
            buffer, ctypes.c_size_t(len(payload)), ctypes.byref(raw)
        )
    )
    _check(
        lib.n4m_serialization_inspect_pipeline_v1(
            buffer,
            ctypes.c_size_t(len(payload)),
            ctypes.byref(raw_pipeline),
            ctypes.c_size_t(ctypes.sizeof(raw_pipeline)),
        )
    )
    pipeline = None
    if raw_pipeline.present:
        pipeline = SerializedPipelineInfo(
            schema_version=int(raw_pipeline.schema_version),
            operators=tuple(
                int(raw_pipeline.operators[index])
                for index in range(int(raw_pipeline.operator_count))
            ),
            savgol_window=int(raw_pipeline.savgol_window),
            savgol_poly_degree=int(raw_pipeline.savgol_poly_degree),
            savgol_derivative=int(raw_pipeline.savgol_derivative),
            semantic_profile=int(raw_pipeline.semantic_profile),
            savgol_delta=float(raw_pipeline.savgol_delta),
            raw_n_features=int(raw_pipeline.raw_n_features),
            model_n_features=int(raw_pipeline.model_n_features),
            fingerprint_algorithm=int(raw_pipeline.fingerprint_algorithm),
            fingerprint=int(raw_pipeline.fingerprint),
            snv_axis=int(raw_pipeline.snv_axis),
            snv_with_mean=bool(raw_pipeline.snv_with_mean),
            snv_with_std=bool(raw_pipeline.snv_with_std),
            snv_ddof=int(raw_pipeline.snv_ddof),
            savgol_mode=int(raw_pipeline.savgol_mode),
            savgol_cval=float(raw_pipeline.savgol_cval),
        )
    return SerializedModelInfo(
        schema_version=int(raw.schema_version),
        format_version=int(raw.format_version),
        writer_abi=(
            int(raw.writer_abi_major),
            int(raw.writer_abi_minor),
            int(raw.writer_abi_patch),
        ),
        algorithm=int(raw.algorithm),
        solver=int(raw.solver),
        deflation=int(raw.deflation),
        training_samples=int(raw.training_samples),
        n_features=int(raw.n_features),
        n_targets=int(raw.n_targets),
        n_components=int(raw.n_components),
        capabilities=int(raw.capabilities),
        pipeline=pipeline,
    )


class _Array:
    """Owning wrapper for `n4m_array_t*`.

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
            lib.n4m_array_free(self._h)
            self._h = ctypes.c_void_p(0)

    def shape(self) -> tuple[int, int]:
        rows = ctypes.c_int64(0)
        cols = ctypes.c_int64(0)
        _check(lib.n4m_array_shape(self._h, ctypes.byref(rows), ctypes.byref(cols)))
        return int(rows.value), int(cols.value)

    def view(self) -> np.ndarray:
        mv = MatrixView()
        _check(lib.n4m_array_view(self._h, ctypes.byref(mv)))
        if mv.rows == 0 or mv.cols == 0:
            return np.empty((mv.rows, mv.cols), dtype=np.float64)
        # numpy.ctypeslib.as_array borrows the buffer; we re-cast to float64.
        buf_type = ctypes.c_double * (mv.rows * mv.cols)
        buffer = buf_type.from_address(int(mv.data))
        arr = np.frombuffer(buffer, dtype=np.float64, count=mv.rows * mv.cols)
        return arr.reshape(int(mv.rows), int(mv.cols))

    def copy(self) -> np.ndarray:
        return np.array(self.view(), copy=True)


class Model:
    """Owning handle around `n4m_model_t`. Constructed by `Model.fit(...)`.

    Stores a reference to the input arrays during fit so the NumPy buffers
    remain valid for the duration of the C call.
    """

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._h = handle

    @classmethod
    def fit(cls, ctx: Context, cfg: Config, X: Any, Y: Any) -> "Model":
        X_arr = _as_float64_contiguous(X)
        Y_arr = _as_float64_contiguous(Y)
        x_view = _matrix_view(X_arr)
        y_view = _matrix_view(Y_arr)
        out = ctypes.c_void_p(0)
        status = lib.n4m_model_fit(
            ctx.handle,
            cfg.handle,
            ctypes.byref(x_view),
            ctypes.byref(y_view),
            ctypes.byref(out),
        )
        _check(status, ctx)
        if not out:
            raise Pls4allError(
                int(Status.ERR_INTERNAL), "n4m_model_fit returned NULL handle"
            )
        try:
            return cls(out)
        except BaseException:
            lib.n4m_model_destroy(out)
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
            lib.n4m_model_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def n_components(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_model_get_n_components(self._h, ctypes.byref(out)))
        return int(out.value)

    @property
    def n_features(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_model_get_n_features(self._h, ctypes.byref(out)))
        return int(out.value)

    @property
    def n_targets(self) -> int:
        out = ctypes.c_int32(0)
        _check(lib.n4m_model_get_n_targets(self._h, ctypes.byref(out)))
        return int(out.value)

    def predict(self, ctx: Context, X: Any) -> np.ndarray:
        X_arr = _as_float64_contiguous(X)
        x_view = _matrix_view(X_arr)
        out_handle = ctypes.c_void_p(0)
        _check(
            lib.n4m_model_predict_alloc(
                ctx.handle,
                self._h,
                ctypes.byref(x_view),
                ctypes.byref(out_handle),
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
            lib.n4m_model_transform_alloc(
                ctx.handle,
                self._h,
                ctypes.byref(x_view),
                ctypes.byref(out_handle),
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
            lib.n4m_model_get_array(
                ctx.handle,
                self._h,
                int(kind),
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
        """Serialize the fitted model to an N4MM v1 or v2 buffer.

        This is a raw fitted-model payload, not a nirs4all ``.n4a`` pipeline
        bundle. The raw payload does not currently have a canonical filename
        extension.
        """
        size = ctypes.c_size_t(0)
        _check(lib.n4m_model_export_size(self._h, ctypes.byref(size)))
        if size.value == 0:
            raise Pls4allError(
                int(Status.ERR_INTERNAL), "n4m_model_export_size returned 0"
            )
        buf = (ctypes.c_uint8 * int(size.value))()
        written = ctypes.c_size_t(0)
        _check(
            lib.n4m_model_export_to_buffer(self._h, buf, size, ctypes.byref(written))
        )
        return bytes(bytearray(buf)[: int(written.value)])

    @classmethod
    def from_bytes(cls, ctx: Context, payload: bytes) -> "Model":
        """Deserialize an N4MM buffer produced by :meth:`to_bytes`.

        Corrupt buffers and unsupported wire-format versions fail. The current
        v1/v2 importer accepts writer ABI differences and records a
        compatibility warning on ``ctx`` instead of rejecting the payload.
        """
        if not payload:
            raise Pls4allError(
                int(Status.ERR_INVALID_ARGUMENT), "from_bytes: empty payload"
            )
        buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        out = ctypes.c_void_p(0)
        _check(
            lib.n4m_model_import_from_buffer(
                ctx.handle, buf, ctypes.c_size_t(len(payload)), ctypes.byref(out)
            ),
            ctx,
        )
        if not out:
            raise Pls4allError(
                int(Status.ERR_INTERNAL), "n4m_model_import_from_buffer returned NULL"
            )
        try:
            return cls(out)
        except BaseException:
            lib.n4m_model_destroy(out)
            raise


__all__ = [
    "Model",
    "ModelArrayKind",
    "SERIALIZED_MODEL_CAPABILITY_AFFINE",
    "SERIALIZED_MODEL_CAPABILITY_PREDICT",
    "SERIALIZED_MODEL_CAPABILITY_TRANSFORM",
    "SerializedModelInfo",
    "_as_float64_contiguous",
    "_matrix_view",
    "inspect_n4mm",
]

# SPDX-License-Identifier: CECILL-2.1
"""Fitted, ordered preprocessing through libn4m's native pipeline ABI.

Operator parameters use the positional C ABI contract. Numerical fitting,
transformation, and N4MP serialization are all performed by libn4m.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from n4m._context import Context
from n4m._errors import check
from n4m._ffi import lib
from n4m._impl.compat import BaseEstimator, TransformerMixin
from n4m._matrix import numpy_to_view

_KINDS = {
    "identity": 0, "center": 1, "autoscale": 2, "pareto_scale": 3,
    "snv": 4, "msc": 5, "emsc": 6, "detrend_poly": 7,
    "savgol_smooth": 8, "savgol_derivative": 9, "norris_williams": 10,
    "asls_baseline": 11, "osc": 12, "epo": 13, "wavelet_denoise": 14,
}
_NAMES = {code: name for name, code in _KINDS.items()}


@dataclass(frozen=True)
class PreprocessingOperatorSpec:
    """One native operator kind and its positional C ABI parameters."""

    kind: str | int
    params: Sequence[float] = ()


def _float_matrix(value: Any, name: str, *, target: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1) if target else matrix.reshape(1, -1)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D matrix")
    return np.ascontiguousarray(matrix)


def _operator(value: Any) -> tuple[int, np.ndarray]:
    if isinstance(value, PreprocessingOperatorSpec):
        kind, params = value.kind, value.params
    elif isinstance(value, (str, Integral)):
        kind, params = value, ()
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        kind, params = value
    else:
        raise ValueError("operators must contain native preprocessing specifications")
    if isinstance(kind, str):
        try:
            code = _KINDS[kind]
        except KeyError as exc:
            raise ValueError(f"unknown or unsupported native preprocessing kind {kind!r}") from exc
    elif isinstance(kind, Integral) and not isinstance(kind, (bool, np.bool_)):
        code = int(kind)
        if not 0 <= code <= 14:
            raise ValueError(f"unknown or unsupported native preprocessing kind {kind!r}")
    else:
        raise ValueError("native preprocessing kind must be a name or integer code")  # noqa: TRY004
    try:
        values = np.asarray(params, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("operator params must be a finite numeric vector") from exc
    if values.ndim != 1 or values.size > np.iinfo(np.int32).max or not np.isfinite(values).all():
        raise ValueError("operator params must be a finite numeric vector")
    return code, np.ascontiguousarray(values)


def _read_steps(handle: ctypes.c_void_p, count: int) -> tuple[PreprocessingOperatorSpec, ...]:
    """Recover the original ordered plan from the fitted native handle."""
    steps = []
    for index in range(count):
        kind = ctypes.c_int()
        n_params = ctypes.c_int32()
        check(lib.n4m_pipeline_get_operator(
            handle, ctypes.c_int32(index), ctypes.byref(kind), None,
            ctypes.c_int32(0), ctypes.byref(n_params)),
            "n4m_pipeline_get_operator")
        try:
            name = _NAMES[kind.value]
        except KeyError as exc:
            raise RuntimeError("native pipeline contains an unsupported operator kind") from exc
        if n_params.value < 0:
            raise RuntimeError("native pipeline reported a negative parameter count")
        params = (ctypes.c_double * n_params.value)()
        if n_params.value:
            check(lib.n4m_pipeline_get_operator(
                handle, ctypes.c_int32(index), ctypes.byref(kind), params,
                ctypes.c_int32(n_params.value), ctypes.byref(n_params)),
                "n4m_pipeline_get_operator")
        steps.append(PreprocessingOperatorSpec(name, tuple(params)))
    return tuple(steps)


def _canonical_steps(values: Sequence[Any]) -> tuple[tuple[int, tuple[float, ...]], ...]:
    return tuple((kind, tuple(params)) for kind, params in map(_operator, values))


class NativePreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Sklearn-style native fit/transform with portable fitted N4MP state.

    ``operators`` is an ordered sequence of :class:`PreprocessingOperatorSpec`,
    operator names, or ``(kind, params)`` pairs. OSC and EPO need ``y`` at fit.
    Imported N4MP instances recover their ordered operator plan from the
    native payload and can be refitted explicitly with that plan.
    """

    def __init__(self, operators: Sequence[Any] = ()) -> None:
        self.operators = operators
        self._handle = ctypes.c_void_p()

    def __sklearn_is_fitted__(self) -> bool:
        return bool(self._handle.value)

    def _require_fitted(self) -> ctypes.c_void_p:
        if not self._handle.value:
            raise RuntimeError("NativePreprocessingPipeline is not fitted")
        return self._handle

    def fit(self, X: Any, y: Any = None) -> NativePreprocessingPipeline:
        X_arr = _float_matrix(X, "X")
        Y_arr = None if y is None else _float_matrix(y, "y", target=True)
        if Y_arr is not None and Y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y rows must match X rows")
        if not self.operators:
            raise ValueError("operators must contain at least one native preprocessing step")
        handle = ctypes.c_void_p()
        check(lib.n4m_pipeline_create(ctypes.byref(handle)), "n4m_pipeline_create")
        try:
            for value in self.operators:
                kind, params = _operator(value)
                pointer = (params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                           if params.size else None)
                check(lib.n4m_pipeline_add_operator(
                    handle, ctypes.c_int(kind), pointer, ctypes.c_int32(params.size)),
                    "n4m_pipeline_add_operator")
            x_view = numpy_to_view(X_arr)
            y_view = numpy_to_view(Y_arr) if Y_arr is not None else None
            with Context() as ctx:
                check(lib.n4m_pipeline_fit(
                    ctx.handle, handle, ctypes.byref(x_view),
                    ctypes.byref(y_view) if y_view is not None else None),
                    "n4m_pipeline_fit")
            width = ctypes.c_int64()
            count = ctypes.c_int32()
            check(lib.n4m_pipeline_get_info(
                handle, ctypes.byref(width), ctypes.byref(count)),
                "n4m_pipeline_get_info")
            steps = _read_steps(handle, count.value)
        except Exception:
            lib.n4m_pipeline_destroy(handle)
            raise
        self.close()
        self._handle = handle
        self.n_features_in_ = int(width.value)
        self.n_operators_ = int(count.value)
        self.steps_ = steps
        return self

    def transform(self, X: Any) -> np.ndarray:
        handle = self._require_fitted()
        X_arr = _float_matrix(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X feature width must match the fitted pipeline")
        result = np.empty_like(X_arr)
        x_view = numpy_to_view(X_arr)
        out_view = numpy_to_view(result)
        with Context() as ctx:
            check(lib.n4m_pipeline_transform(
                ctx.handle, handle, ctypes.byref(x_view), ctypes.byref(out_view)),
                "n4m_pipeline_transform")
        return result

    def to_bytes(self) -> bytes:
        """Export the fitted native pipeline as an N4MP blob."""
        handle = self._require_fitted()
        size = ctypes.c_size_t()
        check(lib.n4m_pipeline_export_size(handle, ctypes.byref(size)),
              "n4m_pipeline_export_size")
        buffer = ctypes.create_string_buffer(size.value)
        written = ctypes.c_size_t()
        check(lib.n4m_pipeline_export_to_buffer(
            handle, buffer, size.value, ctypes.byref(written)),
            "n4m_pipeline_export_to_buffer")
        return buffer.raw[:written.value]

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, expected_operators: Sequence[Any] | None = None,
    ) -> NativePreprocessingPipeline:
        """Import N4MP, optionally verifying an external recipe manifest."""
        if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
            raise ValueError("payload must be nonempty N4MP bytes")
        data = ctypes.create_string_buffer(bytes(payload))
        handle = ctypes.c_void_p()
        with Context() as ctx:
            check(lib.n4m_pipeline_import_from_buffer(
                ctx.handle, data, len(payload), ctypes.byref(handle)),
                "n4m_pipeline_import_from_buffer")
        try:
            width = ctypes.c_int64()
            count = ctypes.c_int32()
            check(lib.n4m_pipeline_get_info(
                handle, ctypes.byref(width), ctypes.byref(count)),
                "n4m_pipeline_get_info")
            steps = _read_steps(handle, count.value)
            if (expected_operators is not None and
                    _canonical_steps(expected_operators) != _canonical_steps(steps)):
                raise ValueError("expected preprocessing operators differ from native N4MP plan")
        except Exception:
            lib.n4m_pipeline_destroy(handle)
            raise
        instance = cls(steps)
        instance._handle = handle
        instance.n_features_in_ = int(width.value)
        instance.n_operators_ = int(count.value)
        instance.steps_ = steps
        return instance

    def close(self) -> None:
        """Release the native handle; repeated calls are safe."""
        if getattr(self, "_handle", None) is not None and self._handle.value:
            lib.n4m_pipeline_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            return


__all__ = ["NativePreprocessingPipeline", "PreprocessingOperatorSpec"]

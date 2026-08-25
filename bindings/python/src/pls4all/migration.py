"""Verified PREDICT-only N4MM construction for migration tooling.

The helper accepts an already-attested affine equation rather than a foreign
serialized model.  It never loads pickle/joblib data and never retrains.  A
caller must prove that its source estimator is exactly ``intercept + X @
coefficients`` before invoking this public ``pls4all`` API.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence

from ._context import Context
from ._errors import Pls4allError
from ._ffi import LinearPredictorSpec, lib
from ._types import Status


def _check(status: int, context: Context) -> None:
    if status == Status.OK:
        return
    raw = lib.n4m_context_last_error(context.handle)
    message = raw.decode("utf-8") if raw else None
    raise Pls4allError(status, message)


def export_linear_predictor_n4mm(
    coefficients: Sequence[Sequence[float]],
    intercept: Sequence[float],
    *,
    source_training_samples: int = 0,
) -> bytes:
    """Create a PREDICT-only N4MM from a verified affine predictor.

    ``coefficients`` is row-major with shape ``(n_features, n_targets)``;
    ``intercept`` has length ``n_targets``.  The resulting model intentionally
    has no latent PLS transform state: calling its transform operation is
    unsupported by the native ABI.
    """

    if isinstance(source_training_samples, bool) or not isinstance(source_training_samples, int):
        raise TypeError("source_training_samples must be an integer")
    if source_training_samples < 0:
        raise ValueError("source_training_samples must be non-negative")
    rows = [tuple(float(value) for value in row) for row in coefficients]
    targets = tuple(float(value) for value in intercept)
    if not rows or not targets or any(len(row) != len(targets) for row in rows):
        raise ValueError(
            "coefficients must be a non-empty rectangular (features, targets) matrix"
        )
    if (
        source_training_samples > (2**63 - 1)
        or len(rows) > (2**31 - 1)
        or len(targets) > (2**31 - 1)
    ):
        raise ValueError("linear predictor dimensions exceed the C ABI range")
    values = tuple(value for row in rows for value in row)
    if not all(math.isfinite(value) for value in (*values, *targets)):
        raise ValueError("coefficients and intercept must be finite")

    coefficient_buffer = (ctypes.c_double * len(values))(*values)
    intercept_buffer = (ctypes.c_double * len(targets))(*targets)
    spec = LinearPredictorSpec(
        source_training_samples,
        len(rows),
        len(targets),
        coefficient_buffer,
        intercept_buffer,
    )
    with Context() as context:
        model = ctypes.c_void_p()
        try:
            _check(
                lib.n4m_model_import_linear_predictor(
                    context.handle, ctypes.byref(spec), ctypes.byref(model)
                ),
                context,
            )
            size = ctypes.c_size_t()
            _check(lib.n4m_model_export_size(model, ctypes.byref(size)), context)
            if size.value == 0:
                raise RuntimeError("native linear predictor export was empty")
            payload = (ctypes.c_ubyte * size.value)()
            written = ctypes.c_size_t()
            _check(
                lib.n4m_model_export_to_buffer(
                    model, payload, size.value, ctypes.byref(written)
                ),
                context,
            )
            if written.value != size.value:
                raise RuntimeError("native linear predictor export wrote an unexpected size")
            return bytes(payload)
        finally:
            if model.value:
                lib.n4m_model_destroy(model)


__all__ = ["export_linear_predictor_n4mm"]

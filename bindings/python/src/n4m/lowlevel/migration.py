# SPDX-License-Identifier: CECILL-2.1
"""Verified PREDICT-only N4MM construction for migration tooling.

This module intentionally accepts an already-attested affine equation rather
than a foreign serialized model.  It never loads joblib/pickle and never
retrains.  Callers remain responsible for proving that their source estimator
is exactly represented by ``intercept + X @ coefficients``.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence

from n4m._context import Context
from n4m._errors import check
from n4m._ffi import lib
from n4m._types import LinearPredictorSpec


def export_linear_predictor_n4mm(
    coefficients: Sequence[Sequence[float]],
    intercept: Sequence[float],
    *,
    source_training_samples: int = 0,
) -> bytes:
    """Create a PREDICT-only N4MM for a verified affine predictor.

    ``coefficients`` must be row-major in native ``(features, targets)``
    orientation.  The result preserves the numerical prediction equation but
    deliberately has no latent PLS transform semantics.
    """

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
        int(source_training_samples),
        len(rows),
        len(targets),
        coefficient_buffer,
        intercept_buffer,
    )
    context = Context()
    model = ctypes.c_void_p()
    try:
        check(
            lib.n4m_model_import_linear_predictor(
                context.handle, ctypes.byref(spec), ctypes.byref(model)
            ),
            "n4m_model_import_linear_predictor",
        )
        size = ctypes.c_size_t()
        check(
            lib.n4m_model_export_size(model, ctypes.byref(size)),
            "n4m_model_export_size",
        )
        if size.value == 0:
            raise RuntimeError("native linear predictor export was empty")
        payload = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        check(
            lib.n4m_model_export_to_buffer(
                model, payload, size.value, ctypes.byref(written)
            ),
            "n4m_model_export_to_buffer",
        )
        if written.value != size.value:
            raise RuntimeError(
                "native linear predictor export wrote an unexpected size"
            )
        return bytes(payload)
    finally:
        if model.value:
            lib.n4m_model_destroy(model)
        context.close()


__all__ = ["export_linear_predictor_n4mm"]

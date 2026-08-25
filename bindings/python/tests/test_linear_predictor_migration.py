# SPDX-License-Identifier: CECILL-2.1
"""Exact affine-import coverage for the migration-only N4MM helper."""

from __future__ import annotations

import ctypes

import pytest

from n4m import Context
from n4m._errors import check
from n4m._ffi import lib
from n4m._types import Dtype, MatrixView
from n4m.lowlevel.migration import export_linear_predictor_n4mm


def _row_major(buffer, rows: int, cols: int) -> MatrixView:
    return MatrixView(
        ctypes.cast(buffer, ctypes.c_void_p),
        rows,
        cols,
        cols,
        1,
        Dtype.F64,
        0,
    )


def test_linear_predictor_export_is_predict_exact_and_round_trips() -> None:
    n4mm = export_linear_predictor_n4mm(
        ((2.0, 0.5), (-1.0, 3.0)),
        (1.5, -2.0),
        source_training_samples=17,
    )
    assert n4mm.startswith(b"N4MM")
    context = Context()
    model = ctypes.c_void_p()
    try:
        raw = (ctypes.c_ubyte * len(n4mm)).from_buffer_copy(n4mm)
        check(
            lib.n4m_model_import_from_buffer(
                context.handle, raw, len(n4mm), ctypes.byref(model)
            ),
            "n4m_model_import_from_buffer",
        )
        x = (ctypes.c_double * 4)(1.0, 4.0, -2.0, 3.0)
        actual = (ctypes.c_double * 4)()
        x_view = _row_major(x, 2, 2)
        actual_view = _row_major(actual, 2, 2)
        check(
            lib.n4m_model_predict(
                context.handle, model, ctypes.byref(x_view), ctypes.byref(actual_view)
            ),
            "n4m_model_predict",
        )
        assert list(actual) == [-0.5, 10.5, -5.5, 6.0]
    finally:
        if model.value:
            lib.n4m_model_destroy(model)
        context.close()


def test_linear_predictor_export_refuses_non_finite_or_non_rectangular_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        export_linear_predictor_n4mm(((float("nan"),),), (0.0,))
    with pytest.raises(ValueError, match="rectangular"):
        export_linear_predictor_n4mm(((1.0,), (2.0, 3.0)), (0.0,))

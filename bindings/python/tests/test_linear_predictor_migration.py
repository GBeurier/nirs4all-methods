# SPDX-License-Identifier: CECILL-2.1
"""Exact affine-import coverage for the migration-only N4MM helper."""

from __future__ import annotations

import numpy as np
import pytest

from pls4all import Context, Model, export_linear_predictor_n4mm


def test_linear_predictor_export_is_predict_exact_and_round_trips() -> None:
    n4mm = export_linear_predictor_n4mm(
        ((2.0, 0.5), (-1.0, 3.0)),
        (1.5, -2.0),
        source_training_samples=17,
    )
    assert n4mm.startswith(b"N4MM")
    with Context() as context:
        model = Model.from_bytes(context, n4mm)
        try:
            actual = model.predict(context, np.array([[1.0, 4.0], [-2.0, 3.0]]))
            np.testing.assert_allclose(actual, [[-0.5, 10.5], [-5.5, 6.0]])
        finally:
            model.close()


def test_linear_predictor_export_refuses_non_finite_or_non_rectangular_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        export_linear_predictor_n4mm(((float("nan"),),), (0.0,))
    with pytest.raises(ValueError, match="rectangular"):
        export_linear_predictor_n4mm(((1.0,), (2.0, 3.0)), (0.0,))

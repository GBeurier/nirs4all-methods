# SPDX-License-Identifier: CECILL-2.1
"""Exact affine-import coverage for the migration-only N4MM helper."""

from __future__ import annotations

import numpy as np
import pytest
from n4m.lowlevel.migration import inspect_n4mm as inspect_n4mm_full

from pls4all import (
    SERIALIZED_MODEL_CAPABILITY_AFFINE,
    SERIALIZED_MODEL_CAPABILITY_PREDICT,
    Context,
    Model,
    Pls4allError,
    export_linear_predictor_n4mm,
    inspect_n4mm,
)


def test_linear_predictor_export_is_predict_exact_and_round_trips() -> None:
    n4mm = export_linear_predictor_n4mm(
        ((2.0, 0.5), (-1.0, 3.0)),
        (1.5, -2.0),
        source_training_samples=17,
    )
    assert n4mm.startswith(b"N4MM")
    info = inspect_n4mm(n4mm)
    assert info.schema_version == 1
    assert info.format_version == 1
    assert info.writer_abi == (2, 4, 0)
    assert info.algorithm == 11
    assert info.solver == 0
    assert info.deflation == 0
    assert info.training_samples == 17
    assert (info.n_features, info.n_targets, info.n_components) == (2, 2, 0)
    assert info.capabilities == (
        SERIALIZED_MODEL_CAPABILITY_PREDICT | SERIALIZED_MODEL_CAPABILITY_AFFINE
    )
    assert inspect_n4mm_full(n4mm).capabilities == info.capabilities

    corrupt = bytearray(n4mm)
    corrupt[80] ^= 1
    with pytest.raises(Pls4allError) as exc_info:
        inspect_n4mm(bytes(corrupt))
    assert exc_info.value.status == 14
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

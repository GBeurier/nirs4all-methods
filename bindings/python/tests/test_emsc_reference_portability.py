"""EMSC degree and reference reconstruct the native fitted state."""

import numpy as np
import pytest

from n4m._impl.preprocessing import EMSC


def test_emsc_reference_roundtrip_without_validation_refit():
    train = np.array(
        [
            [
                1.5 + 0.08 * i + 0.04 * j + 0.11 * np.sin((i + 1) * (j + 2))
                for j in range(9)
            ]
            for i in range(6)
        ],
        dtype=np.float64,
    )
    test = np.array(
        [
            [
                1.3 + 0.13 * i + 0.05 * j + 0.09 * np.cos((i + 2) * (j + 1))
                for j in range(9)
            ]
            for i in range(3)
        ],
        dtype=np.float64,
    )
    fitted = EMSC(degree=2).fit(train)
    reference = fitted.reference_
    np.testing.assert_allclose(reference, train.mean(axis=0), atol=1e-14, rtol=0)
    restored = EMSC(degree=2).restore_reference(reference.copy())
    np.testing.assert_array_equal(restored.transform(test), fitted.transform(test))
    reference[0] += 1
    np.testing.assert_array_equal(restored.transform(test), fitted.transform(test))
    with pytest.raises(ValueError, match="finite"):
        EMSC(degree=2).restore_reference([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(RuntimeError, match="fitted"):
        _ = EMSC().reference_

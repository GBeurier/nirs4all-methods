# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Optional count admission through the generated generic KS role."""

import numpy as np
import pytest
from n4m import roles


@pytest.mark.parametrize("rows, count", [(26, 20), (25, 19)])
def test_kennard_stone_count_and_default_omission(rows, count):
    X = np.arange(rows, dtype=np.float64).reshape(-1, 1)
    default = roles.KennardStone()
    assert default.get_params() == {"test_size": None, "n_train": None}
    implicit = next(default.split(X))
    explicit = next(roles.KennardStone(test_size=0.25).split(X))
    for a, b in zip(implicit, explicit, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len(implicit[0]) == rows - 7
    counted = next(roles.KennardStone(n_train=count).split(X))
    assert (len(counted[0]), len(counted[1])) == (count, rows - count)
    np.testing.assert_array_equal(counted[0][: rows - 7], implicit[0])
    np.testing.assert_array_equal(np.sort(np.concatenate(counted)), np.arange(rows))
    assert np.all(np.diff(counted[1]) > 0)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        next(roles.KennardStone(n_train=count, test_size=0.25).split(X))


@pytest.mark.parametrize("value", [2.5, "20", 2**64 + 20])
def test_kennard_stone_count_does_not_truncate_or_wrap(value):
    X = np.arange(26, dtype=np.float64).reshape(-1, 1)
    with pytest.raises((TypeError, OverflowError)):
        next(roles.KennardStone(n_train=value).split(X))


@pytest.mark.parametrize("scalar", [float, np.float64])
def test_generic_integer_parameter_accepts_integral_floats(scalar):
    X = np.arange(26, dtype=np.float64).reshape(-1, 1)
    expected = next(roles.KennardStone(n_train=5).split(X))
    actual = next(roles.KennardStone(n_train=scalar(5.0)).split(X))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(TypeError, match="n_train must be an integer"):
        next(roles.KennardStone(n_train=scalar(5.5)).split(X))

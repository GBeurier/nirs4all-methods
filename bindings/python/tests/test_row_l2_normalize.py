# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Generated constructor and shared native-state transport smoke."""

import numpy as np
from n4m.roles import NativeEstimator, RowL2Normalize


def test_row_l2_constructor_and_state() -> None:
    X = np.array([[3.0, 4.0], [0.0, 0.0]], order="F")
    est = RowL2Normalize().fit(X)
    actual = est.transform(X)
    np.testing.assert_allclose(actual, [[0.6, 0.8], [0.0, 0.0]], rtol=0, atol=4e-15)
    replay = NativeEstimator.from_n4me(est.to_n4me())
    np.testing.assert_array_equal(replay.transform(X), actual)

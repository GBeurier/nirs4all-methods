"""The Python C-ABI binding must expose effective group sparsity."""

import numpy as np
import pls4all
import pytest


def test_group_penalty_changes_exported_predictor():
    x = np.array(
        [
            [1, 0, 2, 3],
            [2, 1, 0, 2],
            [3, 2, 1, 0],
            [4, 1, 3, 1],
            [5, 3, 2, 4],
            [6, 2, 4, 2],
            [7, 4, 1, 5],
            [8, 3, 5, 3],
        ],
        dtype=np.float64,
    )
    y = np.column_stack(
        [
            2 * x[:, 0] + 0.3 * x[:, 1] - x[:, 2] + 1,
            -x[:, 0] + 0.5 * x[:, 2] + 0.7 * x[:, 3] - 2,
        ]
    )
    groups = np.array([2, 2, 9, 9], dtype=np.int32)
    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.n_components = 2
        with pls4all.group_sparse_pls_fit(ctx, cfg, x, y, groups, 0.0) as raw:
            b0 = raw.matrix("coefficients")
            p0 = raw.matrix("predictions")
        with pls4all.group_sparse_pls_fit(ctx, cfg, x, y, groups, 0.2) as fit:
            b = fit.matrix("coefficients")
            pred = fit.matrix("predictions")
            np.testing.assert_allclose(
                pred, (x - x.mean(axis=0)) @ b + y.mean(axis=0), atol=1e-10
            )
            np.testing.assert_allclose(
                pred[:2].ravel(order="F"),
                [
                    2.211247731382924,
                    5.098353536109149,
                    -1.696732182318895,
                    -2.609927236516501,
                ],
                atol=1e-10,
            )
        for group in (slice(0, 2), slice(2, 4)):
            norm = np.linalg.norm(b0[group, :])
            factor = max(0.0, 1.0 - 0.2 / norm) if norm else 0.0
            np.testing.assert_allclose(b[group, :], b0[group, :] * factor, atol=1e-10)
        assert np.max(np.abs(pred - p0)) > 1e-6
        with pls4all.group_sparse_pls_fit(ctx, cfg, x, y, groups, 1e6) as fit:
            np.testing.assert_array_equal(fit.matrix("coefficients"), np.zeros_like(b))
            np.testing.assert_allclose(
                fit.matrix("predictions"),
                np.tile(y.mean(axis=0), (x.shape[0], 1)),
                atol=1e-10,
            )
        with pytest.raises(pls4all.Pls4allError):
            pls4all.group_sparse_pls_fit(ctx, cfg, x, y, groups, -0.1)

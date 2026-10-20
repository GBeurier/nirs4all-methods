"""Component resolution and unresolved flags of PLS regression and PCR.

With the centred, orthogonal a = (-3, -1, 1, 3), b = (1, -3, 3, -1), X = a and
Y = b + 2^-50 a, the exact PCR Y loading is 2^-50 and U = 2^50 Y, but the
cross-covariance 20 * 2^-50 does not exceed the PCR resolution
5 eps sqrt(20) ||Y||: the component is flagged unresolved, its U column is
zero, and the flags and resolution survive N4MM export/import and pickling.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pls4all import Config, Context, Model, ModelArrayKind
from pls4all._types import Algorithm, Solver
from pls4all.sklearn import PCR, PLSCanonical, PLSRegression

A = np.array([-3.0, -1.0, 1.0, 3.0])
B = np.array([1.0, -3.0, 3.0, -1.0])
EPS = np.finfo(float).eps


def _fit(
    algorithm: Algorithm, solver: Solver, X: np.ndarray, Y: np.ndarray, k: int
) -> tuple[Context, Model]:
    ctx = Context()
    cfg = Config()
    cfg.n_components = k
    cfg.algorithm = algorithm
    cfg.solver = solver
    cfg.scale_x = False
    cfg.scale_y = False
    cfg.store_scores = True
    try:
        return ctx, Model.fit(ctx, cfg, X, Y)
    finally:
        cfg.close()


def test_pcr_unresolved_component_flagged_and_round_tripped() -> None:
    X = A[:, None]
    Y = (B + 2.0**-50 * A)[:, None]
    ctx, model = _fit(Algorithm.PCR, Solver.SVD, X, Y, 1)
    expected_resolution = 5 * EPS * np.sqrt(20.0) * np.linalg.norm(Y)
    resolution, unresolved = model.component_resolution(ctx)
    assert resolution == pytest.approx(expected_resolution, rel=1e-12)
    assert unresolved.tolist() == [True]
    assert model.get_array(ctx, ModelArrayKind.Y_LOADINGS_Q)[0, 0] == pytest.approx(
        2.0**-50, rel=1e-12
    )
    np.testing.assert_array_equal(
        model.get_array(ctx, ModelArrayKind.Y_SCORES_U), np.zeros((4, 1))
    )

    payload = model.to_bytes()
    assert payload[4] == 3  # N4MM format 3 carries the resolution section
    restored = Model.from_bytes(ctx, payload)
    restored_resolution, restored_unresolved = restored.component_resolution(ctx)
    assert restored_resolution == resolution
    assert restored_unresolved.tolist() == [True]

    estimator = PCR(n_components=1, scale_x=False).fit(X, Y.ravel())
    assert estimator.component_resolution_ == resolution
    assert estimator.unresolved_components_.tolist() == [True]
    clone = pickle.loads(pickle.dumps(estimator))
    assert clone.unresolved_components_.tolist() == [True]


def test_pls_unresolved_component_refused_and_resolved_flags() -> None:
    X = A[:, None]
    Y = (B + 2.0**-50 * A)[:, None]
    with pytest.raises(Exception, match="below its numerical resolution"):
        _fit(Algorithm.PLS_REGRESSION, Solver.NIPALS, X, Y, 1)

    # 64 targets whose residuals fall below the 10 eps mask but whose
    # cross-covariance is resolved together.
    X2 = np.column_stack([A, B])
    Y2 = np.column_stack([A] + [2.0**-29 * A + 2.0**-51 * B] * 64)
    estimator = PLSRegression(n_components=2, solver="nipals", scale_x=False).fit(
        X2, Y2
    )
    expected = 6 * EPS * np.linalg.norm(X2) * np.linalg.norm(Y2)
    assert estimator.component_resolution_ == pytest.approx(expected, rel=1e-12)
    assert estimator.unresolved_components_.tolist() == [False, False]
    np.testing.assert_allclose(estimator.predict(X2), Y2, rtol=0, atol=1e-15)


def test_models_without_resolution_report_none() -> None:
    X = np.column_stack([A, B])
    Y = (A + 2.0 * B)[:, None]
    estimator = PLSCanonical(n_components=1, scale_x=False).fit(X, Y.ravel())
    assert not hasattr(estimator, "component_resolution_")

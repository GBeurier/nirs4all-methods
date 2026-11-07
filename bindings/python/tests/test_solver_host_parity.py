"""Guard canonical MethodResult solver selection across language bindings."""

import numpy as np

from n4m._impl import native


def test_continuum_uses_canonical_stone_brooks_solver():
    i = np.arange(1, 13)[:, None]
    j = np.arange(1, 9)[None, :]
    X = np.sin(i * j / 7) + i * j / 50
    y = 2 + X[:, 1] - 0.3 * X[:, 4]
    fit = native.continuum_regression(X, y, tau=0.5, n_components=2)
    np.testing.assert_allclose(
        fit["predictions"].ravel()[:3],
        [2.3604419059764936, 2.250654439880409, 2.4400245161393572],
        rtol=0,
        atol=1e-12,
    )


def test_pls_solver_canonical_names_and_kernel_alias():
    from pls4all import Solver
    from pls4all.sklearn import PLSRegression
    from pls4all.sklearn._base import _resolve_solver
    for solver in Solver:
        assert _resolve_solver(solver.name.lower()) == solver
    x = np.array([[1., 3., 8., 20.], [10., 11., 13., 18.], [-7., -3., 2., 9.], [5., 5., 5., 5.]])
    y = np.array([1., 2., 4., 3.])
    expected = PLSRegression(n_components=1, solver=Solver.KERNEL_ALGORITHM).fit(x, y).predict(x)
    for token in ("kernel_algorithm", "kernel-algorithm", "kernel"):
        np.testing.assert_array_equal(PLSRegression(n_components=1, solver=token).fit(x, y).predict(x), expected)

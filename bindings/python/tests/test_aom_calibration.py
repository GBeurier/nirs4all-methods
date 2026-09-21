"""Protocol contracts and independent held-out prediction checks."""

import pickle

import numpy as np
import pytest
from n4m.model_selection.aom_calibration import (
    AOMPLSRegressor,
    AOMRidgeRegressor,
    FastAOMPLSRegressor,
    FastAOMRidgeRegressor,
    strict_chain_bank,
)
from scipy.signal import savgol_coeffs
from sklearn.base import clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def test_chain_grammar_cardinality_and_order():
    for depth, count in ((1, 10), (2, 62), (3, 206)):
        chains = strict_chain_bank(depth)
        signatures = [">".join(name for name, _ in chain) for chain in chains]
        assert len(chains) == len(set(signatures)) == count
        assert signatures == sorted(signatures, key=lambda s: (s.count(">"), s))
        assert "identity" in signatures
        assert all("identity" not in s for s in signatures if ">" in s)


@pytest.mark.parametrize(
    "cls",
    [AOMPLSRegressor, AOMRidgeRegressor, FastAOMPLSRegressor, FastAOMRidgeRegressor],
)
def test_calibration_estimators_follow_sklearn_regressor_contract(cls):
    assert is_regressor(cls())


def _operator_matrix(name, p):
    eye = np.eye(p)
    if name == "identity":
        return eye
    if name.startswith("detrend"):
        degree = int(name[-1])
        t = np.linspace(-1, 1, p)
        q, _ = np.linalg.qr(np.vander(t, degree + 1, increasing=True))
        return eye - q @ q.T
    if name.startswith("gauss"):
        sigma = float(name[-1])
        r = round(4 * sigma)
        t = np.arange(-r, r + 1)
        kernel = np.exp(-t * t / (2 * sigma * sigma))
    else:
        _, kind, window, poly = name.split("_")
        deriv = 0 if kind == "smooth" else int(kind[1:])
        kernel = savgol_coeffs(int(window[1:]), int(poly[1:]), deriv=deriv, use="dot")
    # Independent convolution, rows of I are row spectra, gives A^T.
    half = (len(kernel) - 1) // 2
    return np.array(
        [
            np.convolve(np.pad(row, (half, half)), kernel[::-1], mode="valid")
            for row in eye
        ]
    ).T


@pytest.mark.parametrize("shape", [(48, 31), (35, 70), (90, 25)])
def test_native_ridge_matches_materialized_svd_candidates(shape):
    rng = np.random.default_rng(141)
    X = rng.normal(size=shape)
    y = X[:, 4] + rng.normal(size=len(X)) * 0.2
    V = rng.normal(size=(11, shape[1]))
    alphas = np.logspace(-2, 3, 7)
    folds = list(KFold(3, shuffle=True, random_state=77).split(X))
    ids = np.empty(len(X), dtype=np.int32)
    for i, (_, valid) in enumerate(folds):
        ids[valid] = i
    model = AOMRidgeRegressor(branches=("raw",), alphas=alphas, fold_ids=ids).fit(X, y)
    scores = []
    chains = strict_chain_bank(1)
    for chain in chains:
        A = _operator_matrix(chain[0][0], shape[1])
        errors = []
        for train, valid in folds:
            errors.append(
                [
                    np.sqrt(
                        np.mean(
                            (
                                Ridge(alpha=a, solver="svd")
                                .fit(X[train] @ A.T, y[train])
                                .predict(X[valid] @ A.T)
                                - y[valid]
                            )
                            ** 2
                        )
                    )
                    for a in alphas
                ]
            )
        scores.append(np.mean(errors, axis=0))
    np.testing.assert_allclose(model.cv_scores_[0], scores, rtol=2e-6, atol=2e-7)
    chain, alpha = np.unravel_index(np.argmin(scores), np.shape(scores))
    A = _operator_matrix(chains[chain][0][0], shape[1])
    reference = (
        Ridge(alpha=alphas[alpha], solver="svd").fit(X @ A.T, y).predict(V @ A.T)
    )
    np.testing.assert_allclose(model.predict(V), reference, rtol=2e-6, atol=2e-7)
    assert model.selected_chain_index_ == chain
    assert model.selected_parameter_index_ == alpha


@pytest.mark.parametrize(
    "cls",
    [AOMPLSRegressor, AOMRidgeRegressor, FastAOMPLSRegressor, FastAOMRidgeRegressor],
)
@pytest.mark.parametrize("branch", ["raw", "snv", "msc"])
def test_native_branch_prediction_pickle_and_fold_validation(cls, branch):
    rng = np.random.default_rng(233)
    X = rng.normal(size=(40, 31))
    y = X[:, 3] + 0.2 * rng.normal(size=len(X))
    model = clone(
        cls(branches=(branch,), max_components=5, alphas=(0.1, 1, 100), rank=16)
    ).fit(X, y)
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(model)).predict(X), model.predict(X)
    )
    assert model.get_diagnostics()["affine_raw_input"] == (branch == "raw")
    with pytest.raises(ValueError):
        model.predict(X[:, :-1])
    with pytest.raises((ValueError, RuntimeError)):
        cls(max_components=5, fold_ids=np.zeros(len(X), dtype=np.int32)).fit(X, y)


def test_ridge_rank_deficient_views_preserve_independent_predictions():
    rng = np.random.default_rng(889)
    latent = rng.normal(size=(35, 2))
    wavelengths = np.linspace(-1, 1, 67)
    X = 12 + latent[:, :1] * wavelengths + latent[:, 1:] * wavelengths**2
    X += rng.normal(size=X.shape) * 1e-9
    y = latent[:, 0] + rng.normal(size=35) * 0.01
    from n4m.model_selection.aom_calibration import _fit_single_view_ridge

    fitted = _fit_single_view_ridge(X, y, "identity", [0.1], 3)
    expected = Ridge(alpha=0.1, solver="svd").fit(X, y).predict(X)
    np.testing.assert_allclose(
        (X @ fitted["input_coefficients"] + fitted["intercept"]).ravel(),
        expected,
        rtol=2e-6,
        atol=2e-7,
    )

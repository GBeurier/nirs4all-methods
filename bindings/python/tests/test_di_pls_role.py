"""DI-PLS role API parity with the native ABI and the R n4m dispatcher."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from n4m.domain_adaptation.invariant import DIPLS, di_pls
from pls4all import Algorithm, Config, Context, Deflation, Solver, di_pls_fit
from pls4all.sklearn import DIPLSRegression
from sklearn.base import clone


@pytest.fixture
def domains():
    i = np.arange(1, 29, dtype=np.float64)[:, None]
    j = np.arange(1, 9, dtype=np.float64)[None, :]
    source = np.sin(i * j / 10) + np.cos(i / 3 + j / 8) + i * j / 110
    target_i = np.arange(1, 18, dtype=np.float64)[:, None]
    target = (
        0.85 * np.sin((target_i + 2) * j / 10)
        + np.cos(target_i / 3 + j / 8)
        + target_i * j / 110
        + j / 70
    )
    held = source[[2, 10, 22], :] + 0.047
    y = 1.2 + 0.65 * source[:, 1] - 0.3 * source[:, 5]
    return source, target, held, y


def test_di_pls_role_matches_native_abi_and_r_oracle(domains):
    source, target, held, y = domains
    raw = di_pls(source, y, X_target=target, n_components=2, di_lambda=0.7)
    with Context() as context, Config() as config:
        config.algorithm = Algorithm.PLS_REGRESSION
        config.solver = Solver.SIMPLS
        config.deflation = Deflation.REGRESSION
        config.n_components = 2
        config.center_x = True
        config.scale_x = False
        config.center_y = True
        config.scale_y = False
        native = di_pls_fit(context, config, source, y[:, None], target, 0.7)
    for key in ("coefficients", "predictions", "x_mean", "y_mean"):
        np.testing.assert_allclose(raw[key], native.matrix(key), rtol=0, atol=1e-12)

    model = DIPLS(n_components=2, di_lambda=0.7).fit(source, y, X_target=target)
    old_subset = DIPLSRegression(n_components=2, di_lambda=0.7).fit(
        source, y, X_target=target
    )
    # R n4m::n4m_method("di_pls", ..., params=list(X_target=..., di_lambda=0.7))
    # on the same deterministic source/target spectra (R n4m 1.0.21.9003).
    r_predictions = [1.26271531563482, 0.645325209345661, 0.387741594953325]
    r_coefficients = [
        0.0642696672082413,
        0.400042236464413,
        -0.104833222522444,
        0.177720715969366,
        0.027039871640705,
        -0.25637804135695,
        -0.0578247996360947,
        -0.111110888295282,
    ]
    np.testing.assert_allclose(model.predict(held), r_predictions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.coef_, r_coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        old_subset.predict(held), model.predict(held), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        model.predict(held),
        ((held - raw["x_mean"]) @ raw["coefficients"] + raw["y_mean"]).ravel(),
        rtol=0,
        atol=1e-12,
    )

    # The target cohort and the DI penalty must alter fitted coefficients.
    alternative_target = np.cos(target * 2.3)
    other = DIPLS(2, di_lambda=0.7).fit(source, y, X_target=alternative_target)
    zero = DIPLS(2, di_lambda=0.0).fit(source, y, X_target=target)
    assert np.max(np.abs(other.coef_ - model.coef_)) > 1e-3
    assert np.max(np.abs(zero.coef_ - model.coef_)) > 1e-4


def test_di_pls_clone_pickle_and_refit_contract(domains):
    source, target, held, y = domains
    model = DIPLS(2, di_lambda=0.7).fit(source, y, X_target=target)
    assert model.get_params() == {"n_components": 2, "di_lambda": 0.7}
    fresh = clone(model)
    assert not hasattr(fresh, "coef_")
    assert not any("target" in key.lower() for key in vars(model))
    np.testing.assert_array_equal(
        model.predict(held), pickle.loads(pickle.dumps(model)).predict(held)
    )
    np.testing.assert_allclose(
        fresh.fit(source, y, X_target=target).predict(held),
        model.predict(held),
        rtol=0,
        atol=1e-12,
    )
    old = model.predict(held).copy()
    with pytest.raises(ValueError, match="feature count"):
        model.fit(source, y, X_target=target[:, :-1])
    np.testing.assert_array_equal(model.predict(held), old)
    with pytest.raises(TypeError, match="X_target"):
        clone(model).fit(source, y)

    subset_model = DIPLSRegression(2, di_lambda=0.7).fit(source, y, X_target=target)
    subset_old = subset_model.predict(held).copy()
    assert not any("target" in key.lower() for key in vars(subset_model))
    with pytest.raises(ValueError, match="feature count"):
        subset_model.fit(source, y, X_target=target[:, :-1])
    np.testing.assert_array_equal(subset_model.predict(held), subset_old)


@pytest.mark.parametrize(
    "target_change, message",
    [
        (lambda target: target[:, :-1], "feature count"),
        (lambda target: target[:1, :], "at least two"),
        (lambda target: np.full_like(target, np.nan), "finite|NaN"),
    ],
)
def test_di_pls_rejects_invalid_target(domains, target_change, message):
    source, target, _, y = domains
    with pytest.raises(ValueError, match=message):
        DIPLS().fit(source, y, X_target=target_change(target))
    with pytest.raises(ValueError, match=message):
        DIPLSRegression().fit(source, y, X_target=target_change(target))


def test_di_pls_rejects_invalid_penalty_multivariate_y_and_feature_order(domains):
    import pandas as pd

    source, target, held, y = domains
    for penalty in (-0.1, np.inf, np.nan):
        with pytest.raises(ValueError, match="di_lambda"):
            DIPLS(di_lambda=penalty).fit(source, y, X_target=target)
        with pytest.raises(ValueError, match="di_lambda"):
            DIPLSRegression(di_lambda=penalty).fit(source, y, X_target=target)
    with pytest.raises(ValueError, match="one response"):
        DIPLS().fit(source, np.column_stack((y, y)), X_target=target)
    columns = [f"wl{i}" for i in range(source.shape[1])]
    named_source = pd.DataFrame(source, columns=columns)
    named_target = pd.DataFrame(target, columns=columns[::-1])
    with pytest.raises(ValueError, match="feature names and order"):
        DIPLS().fit(named_source, y, X_target=named_target)
    with pytest.raises(ValueError, match="feature names and order"):
        DIPLSRegression().fit(named_source, y, X_target=named_target)
    named_target.columns = columns
    model = DIPLS().fit(named_source, y, X_target=named_target)
    with pytest.raises(ValueError, match="feature names and order"):
        model.predict(pd.DataFrame(held, columns=columns[::-1]))

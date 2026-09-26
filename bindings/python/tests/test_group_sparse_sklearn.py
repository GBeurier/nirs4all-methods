"""Held-out GroupSparsePLS binding and sklearn contract checks."""

from __future__ import annotations

import pickle

import numpy as np
import pls4all
import pytest
from n4m.estimators.regression.sparse import GroupSparsePLS, group_sparse_pls
from pls4all.sklearn import GroupSparsePLSRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def data():
    rng = np.random.default_rng(20260926)
    X = rng.normal(size=(40, 6))
    beta = np.array([[2.0, -0.5], [0.2, 0.3], [0, 0], [0, 0], [-1, 1], [0.3, 0]])
    y = X @ beta + np.array([1.5, -2.0])
    X_test = rng.normal(size=(7, 6)) + 0.4
    return X, y, X_test


@pytest.mark.parametrize("target", [0, 2])
def test_heldout_matches_native_affine_result(data, target):
    X, y, X_test = data
    y_fit = y if target == 2 else y[:, 0]
    groups = [2, 2, 17, 17, 30, 30]
    kwargs = {"n_components": 2, "group_assignment": groups, "group_lambda": 0.2}
    n4m = GroupSparsePLS(**kwargs).fit(X, y_fit)
    legacy = GroupSparsePLSRegression(**kwargs).fit(X, y_fit)
    native = group_sparse_pls(X, y_fit, **kwargs)
    expected = (X_test - native["x_mean"].ravel()) @ native["coefficients"]
    expected += native["y_mean"].ravel()
    if target == 0:
        expected = expected.ravel()
    np.testing.assert_allclose(n4m.predict(X_test), expected, atol=1e-11)
    np.testing.assert_allclose(legacy.predict(X_test), expected, atol=1e-11)
    assert n4m.predict(X_test).shape == expected.shape
    assert legacy.predict(X_test).shape == expected.shape
    np.testing.assert_allclose(
        n4m.predict(X),
        native["predictions"].ravel() if target == 0 else native["predictions"],
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(n4m)).predict(X_test), n4m.predict(X_test)
    )
    with pls4all.Context() as ctx, pls4all.Model.from_bytes(
        ctx, n4m.export_n4mm()
    ) as exported:
        np.testing.assert_allclose(
            exported.predict(ctx, X_test).ravel() if target == 0
            else exported.predict(ctx, X_test),
            n4m.predict(X_test), atol=1e-12,
        )
    with pls4all.Context() as ctx, pls4all.Model.from_bytes(
        ctx, legacy.export_n4mm()
    ) as exported:
        np.testing.assert_allclose(
            exported.predict(ctx, X_test).ravel() if target == 0
            else exported.predict(ctx, X_test),
            legacy.predict(X_test), atol=1e-12,
        )


def test_group_penalty_changes_heldout_predictor(data):
    X, y, X_test = data
    kwargs = {"n_components": 2, "group_assignment": [2, 2, 17, 17, 30, 30]}
    unpenalized = GroupSparsePLS(**kwargs, group_lambda=0).fit(X, y)
    penalized = GroupSparsePLS(**kwargs, group_lambda=0.2).fit(X, y)
    assert (
        np.max(np.abs(unpenalized.predict(X_test) - penalized.predict(X_test))) > 1e-5
    )


@pytest.mark.parametrize(
    "groups",
    [
        None,
        [0, 1],
        [[0, 0, 1, 1, 2, 2]],
        [0, 0, 1, 1, 2, -1],
        [0, 0, 1, 1, 2, 2**31],
        [0, 0, 1, 1, 2, 1.0],
        [0, 0, 1, 1, 2, True],
        [0, 0, 1, 1, 2, "2"],
    ],
)
@pytest.mark.parametrize("cls", [GroupSparsePLS, GroupSparsePLSRegression])
def test_invalid_group_ids_rejected_before_native(data, groups, cls):
    X, y, _ = data
    with pytest.raises(ValueError, match="group_assignment"):
        cls(group_assignment=groups).fit(X, y)


@pytest.mark.parametrize("penalty", [-0.1, np.nan, np.inf, True, "bad"])
@pytest.mark.parametrize("cls", [GroupSparsePLS, GroupSparsePLSRegression])
def test_invalid_penalty_rejected_before_native(data, penalty, cls):
    X, y, _ = data
    with pytest.raises(ValueError, match="group_lambda"):
        cls(group_assignment=[0, 0, 1, 1, 2, 2], group_lambda=penalty).fit(X, y)


def test_pipeline_clone_predicts_heldout(data):
    X, y, X_test = data
    estimator = GroupSparsePLS(group_assignment=[0, 0, 1, 1, 2, 2])
    assert clone(estimator).get_params() == estimator.get_params()
    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    pipeline.fit(X, y[:, 0])
    assert pipeline.predict(X_test).shape == (len(X_test),)
    with pytest.raises(ValueError, match="features"):
        estimator.predict(X_test[:, :2])


def test_legacy_uses_corrected_native_coefficients(data):
    X, y, X_test = data
    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.n_components = 2
        with pls4all.group_sparse_pls_fit(
            ctx, cfg, X, y, np.array([2, 2, 17, 17, 30, 30], np.int32), 0.2
        ) as result:
            expected = (X_test - result.matrix("x_mean").ravel()) @ result.matrix(
                "coefficients"
            ) + result.matrix("y_mean").ravel()
    model = GroupSparsePLSRegression(
        group_assignment=[2, 2, 17, 17, 30, 30], group_lambda=0.2
    ).fit(X, y)
    np.testing.assert_allclose(model.predict(X_test), expected, atol=1e-11)


@pytest.mark.parametrize(
    ("fit_name", "parameters"),
    [
        ("fused_sparse_pls_fit", {"l1_lambda": 0.08, "fusion_lambda": 0.05}),
        ("robust_pls_fit", {"huber_k": 1.4, "max_irls_iter": 8}),
        ("ridge_pls_fit", {"ridge_lambda": 0.3}),
        ("continuum_regression_fit", {"tau": 0.4}),
        ("bagging_pls_fit", {"n_estimators": 5, "seed": 7}),
        ("boosting_pls_fit", {"n_estimators": 5, "learning_rate": 0.1}),
        ("random_subspace_pls_fit", {
            "n_estimators": 5, "features_per_subspace": 4, "seed": 7,
        }),
        ("ridge_fit", {"ridge_lambda": 0.3}),
        ("cppls_fit", {"gamma": 0.5}),
        ("sparse_simpls_fit", {"sparsity_lambda": 0.05}),
        ("ecr_fit", {"alpha": 0.5}),
        ("mir_pls_fit", {}),
        ("n_pls_fit", {"mode_j": 2, "mode_k": 3}),
        ("mb_pls_fit", {"block_sizes": [3, 3]}),
        ("di_pls_fit", {"di_lambda": 1.0}),
    ],
)
def test_verified_affine_method_results_promote_to_native_model(data, fit_name, parameters):
    X, y, X_test = data
    target_y = y if fit_name in {
        "fused_sparse_pls_fit", "ridge_fit", "mb_pls_fit",
    } else y[:, :1]
    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.n_components = 2
        if fit_name == "ridge_fit":
            cfg.scale_x = False
            cfg.scale_y = False
        if fit_name == "n_pls_fit":
            result = pls4all.n_pls_fit(
                ctx, cfg, X, parameters["mode_j"], parameters["mode_k"], target_y,
            )
        elif fit_name == "di_pls_fit":
            cfg.scale_x = False
            cfg.scale_y = False
            X_target = X + 0.013
            result = pls4all.di_pls_fit(
                ctx, cfg, X, target_y, X_target, parameters["di_lambda"],
            )
        else:
            result = getattr(pls4all, fit_name)(ctx, cfg, X, target_y, **parameters)
        with result:
            coefficients = result.matrix("coefficients")
            if fit_name in {"ridge_fit", "mb_pls_fit"}:
                expected = X_test @ coefficients + result.matrix("intercept").ravel()
            else:
                x_mean = result.matrix("x_mean").ravel()
                y_mean = result.matrix("y_mean").ravel()
                expected = (X_test - x_mean) @ coefficients + y_mean
            training_predictions = result.matrix("predictions")
            with result.to_affine_model(ctx) as model:
                payload = model.to_bytes()
                assert pls4all.inspect_n4mm(payload).training_samples == X.shape[0]
                np.testing.assert_allclose(
                    model.predict(ctx, X), training_predictions, atol=1e-10,
                )
        with pls4all.Model.from_bytes(ctx, payload) as restored:
            np.testing.assert_allclose(
                restored.predict(ctx, X_test), expected, atol=1e-10,
            )
            if fit_name == "ridge_fit":
                from sklearn.linear_model import Ridge

                reference = Ridge(alpha=parameters["ridge_lambda"]).fit(X, target_y)
                np.testing.assert_allclose(
                    restored.predict(ctx, X_test), reference.predict(X_test), atol=1e-9,
                )


@pytest.mark.parametrize(
    ("class_name", "parameters"),
    [
        ("GroupSparsePLSRegression", {"group_assignment": [0, 0, 1, 1, 2, 2]}),
        ("FusedSparsePLSRegression", {}),
        ("RobustPLSRegression", {}),
        ("RidgePLSRegression", {}),
        ("ContinuumRegression", {}),
        ("BaggingPLSRegression", {"n_estimators": 5}),
        ("BoostingPLSRegression", {"n_estimators": 5}),
        ("RandomSubspacePLSRegression", {"n_estimators": 5, "features_per_subspace": 4}),
        ("Ridge", {"alpha": 0.3}),
        ("CPPLSRegression", {}),
        ("SparseSimplsRegression", {}),
        ("ECRegression", {}),
        ("MIRPLSRegression", {}),
        ("NPLSRegression", {"mode_j": 2, "mode_k": 3}),
        ("MBPLSRegression", {"block_sizes": [3, 3], "scale_x": False,
                             "scale_y": False}),
        ("DIPLSRegression", {}),
    ],
)
def test_verified_sklearn_wrappers_use_native_heldout_n4mm(data, class_name, parameters):
    import pls4all.sklearn as sklearn_wrappers

    X, y, X_test = data
    estimator = getattr(sklearn_wrappers, class_name)(**parameters)
    if class_name == "DIPLSRegression":
        estimator.fit(X, y[:, 0], X_target=X + 0.013)
    else:
        estimator.fit(X, y[:, 0])
    assert estimator._native_affine_model
    assert pls4all.inspect_n4mm(estimator.export_n4mm()).training_samples == X.shape[0]
    predicted = estimator.predict(X_test)
    assert predicted.shape == (X_test.shape[0],)
    with pls4all.Context() as ctx, pls4all.Model.from_bytes(
        ctx, estimator.export_n4mm()
    ) as restored:
        np.testing.assert_allclose(predicted, restored.predict(ctx, X_test).ravel(),
                                   rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(estimator)).predict(X_test), predicted,
        rtol=0, atol=1e-11,
    )

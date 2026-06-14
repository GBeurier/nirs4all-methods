from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import n4m
from n4m._impl import native
import n4m._impl as native_sklearn
from n4m._impl import (
    NativeAOMChainRidgePLSRegressor,
    NativeAOMChainSweepRegressor,
    NativeAOMFixedCandidateRegressor,
    NativeAOMMomentScreenRefitRegressor,
    NativeAOMMomentPLSScreenRefitRegressor,
    NativeAOMMomentPLSExactScreenRefitRegressor,
    NativeAOMMomentRidgeScreenRefitRegressor,
    NativeAOMOperatorPLSStackRegressor,
    NativeAOMPLSRegressor,
    NativeAOMPLSSuperblockRegressor,
    NativeAOMRidgePLSSuperblockRegressor,
    NativeAOMRidgeBlenderRegressor,
    NativeAOMRidgeActiveSuperblockRegressor,
    NativeAOMRidgeGlobalRegressor,
    NativeAOMRidgeMKLSuperblockRegressor,
    NativeAOMRidgeSuperblockRegressor,
    NativeAOMRobustHPORegressor,
    NativeAOMScreenRefitRegressor,
    NativeAOMSweepRegressor,
    NativeContinuumRegressionRegressor,
    NativeCPPLSRegressor,
    NativeECRRegressor,
    NativeMomentStackRegressor,
    NativeMomentSweepRegressor,
    NativePCRRegressor,
    NativePLSRegressor,
    NativePOPPLSRegressor,
    NativeRidgeRegressor,
    NativeRidgePLSRegressor,
    NativeRobustPLSRegressor,
    NativeWeightedPLSRegressor,
)


def _dataset():
    rng = np.random.default_rng(44)
    X = rng.standard_normal((28, 10))
    y = X[:, 0] - 0.5 * X[:, 2] + 0.05 * rng.standard_normal(28)
    return X, y


def _aom_dataset():
    rng = np.random.default_rng(45)
    X = rng.standard_normal((24, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    return X, y


def _assert_method_result(res, n_samples: int, n_targets: int = 1):
    assert res["predictions"].shape == (n_samples, n_targets)
    assert res["coefficients"].shape[1] == n_targets
    assert np.all(np.isfinite(res["predictions"]))
    assert np.isfinite(res["rmse"])


def test_pls_cross_validate_reference_matches_pls_sweep():
    rng = np.random.default_rng(1729)
    X = np.ascontiguousarray(rng.normal(size=(12, 5)), dtype=np.float64)
    y = np.ascontiguousarray(
        0.6 * X[:, 0] - 0.25 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0]),
        dtype=np.float64,
    )
    folds = np.arange(X.shape[0], dtype=np.int32) % 3
    components = np.asarray([1, 2, 3], dtype=np.int32)

    got = native.pls_cross_validate(
        X,
        y,
        cv=3,
        fold_ids=folds,
        component_grid=components,
    )
    expected = native.sweep_run(
        X,
        y,
        cv=3,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=components,
        heads=("pls",),
    )
    np.testing.assert_allclose(
        got["candidate_scores"],
        expected["candidate_scores"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        got["oof_predictions"],
        expected["oof_predictions"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert got["candidate_scores"].shape == (3, 4)
    assert np.all(got["candidate_scores"][:, 1] == 1.0)
    assert got["n_pls_moment_cv_fits"] == expected["n_pls_moment_cv_fits"]
    assert got["n_pls_materialized_cv_fits"] == expected["n_pls_materialized_cv_fits"]


def test_pls_moment_fallback_builds_fold_designs_on_demand():
    X = np.ascontiguousarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0],
        ],
        dtype=np.float64,
    )
    y = 0.5 * X[:, 0] - X[:, 1]
    folds = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
    components = np.asarray([1, 2], dtype=np.int32)

    for score_only in (True, False):
        got = native.sweep_run(
            X,
            y,
            cv=2,
            fold_ids=folds,
            ridge_lambdas=[],
            pls_components=components,
            heads=("pls",),
            score_only=score_only,
        )
        assert got["candidate_scores"].shape == (2, 4)
        assert np.isfinite(got["candidate_scores"][0, 3])
        assert np.isinf(got["candidate_scores"][1, 3])
        assert got["n_pls_moment_cv_fits"] > 0
        assert got["n_pls_materialized_cv_fits"] == 0

    ref = native.pls_cross_validate(
        X,
        y,
        cv=2,
        fold_ids=folds,
        component_grid=components,
        score_only=True,
    )
    assert ref["candidate_scores"].shape == (2, 4)
    assert np.isfinite(ref["candidate_scores"][0, 3])
    assert np.isinf(ref["candidate_scores"][1, 3])
    assert ref["n_pls_moment_cv_fits"] > 0
    assert ref["n_pls_materialized_cv_fits"] == 0


def test_aom_pls_moment_batch_degenerate_components_do_not_abort_screen():
    n = 24
    t = np.arange(n, dtype=np.float64)
    X = np.ascontiguousarray(
        np.column_stack([t, t + 1.0, 2.0 * t + 3.0]),
        dtype=np.float64,
    )
    y = 0.5 * X[:, 0] - X[:, 1]
    folds = np.arange(n, dtype=np.int32) % 2
    chains = [["identity"], ["identity"]]

    got = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=2,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )

    scores = np.asarray(got["candidate_scores"], dtype=float)
    assert scores.shape == (4, 5)
    assert np.all(np.isfinite(scores[scores[:, 3] == 1.0, 4]))
    assert np.all(np.isinf(scores[scores[:, 3] == 2.0, 4]))
    assert np.isfinite(got["selected_cv_rmse"])
    assert got["n_materialized_candidates"] == 0.0
    assert got["n_pls_materialized_cv_fits"] == 0.0
    assert got["n_pls_moment_score_batch_calls"] == 1.0
    assert got["n_pls_moment_score_batch_jobs"] == len(chains) * 2
    assert got["n_pls_moment_cv_fits"] == len(chains) * 3


def test_aom_pls_force_moments_bypasses_cpu_wide_materialization_heuristic():
    rng = np.random.default_rng(20260608)
    X = rng.standard_normal((120, 32))
    y = 0.7 * X[:, 0] - 0.2 * X[:, 11] + 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    got = native.aom_chain_sweep_run(
        X,
        y,
        [[("identity", ())]],
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2, 3],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="cv",
        score_only=True,
    )

    assert got["n_materialized_candidates"] == 0.0
    assert got["n_pls_materialized_cv_fits"] == 0.0
    assert got["n_pls_moment_score_batch_calls"] == 1.0
    assert got["n_pls_moment_score_batch_jobs"] == 4.0
    assert got["n_candidates"] == 3.0
    assert got["n_pls_operator_moment_candidates"] == 3.0
    assert np.isfinite(got["selected_cv_rmse"])


def test_aom_ridge_force_moments_bypasses_cpu_wide_materialization_heuristic():
    rng = np.random.default_rng(20260609)
    X = rng.standard_normal((40, 64))
    y = 0.9 * X[:, 0] - 0.4 * X[:, 5] + 0.2 * X[:, 13]
    y += 0.04 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    lambdas = [0.1, 1.0]

    forced = native.aom_chain_sweep_run(
        X,
        y,
        [[("identity", ())]],
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )
    materialized = native.aom_chain_sweep_run(
        X,
        y,
        [[("identity", ())]],
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="materialized",
        score_only=True,
    )

    assert forced["n_materialized_candidates"] == 0.0
    assert forced["n_ridge_operator_moment_candidates"] == forced["n_candidates"]
    assert forced["n_ridge_moment_score_batch_calls"] == 1.0
    assert forced["n_ridge_moment_score_batch_jobs"] == len(lambdas) * 4
    assert forced["n_ridge_moment_cv_fits"] == len(lambdas) * 4
    assert forced["n_ridge_moment_final_fits"] == 0.0
    assert np.all(np.isfinite(forced["candidate_scores"][:, 4]))
    assert "materialized" not in {
        row["score_route"] for row in native.aom_candidate_table(forced)
    }
    np.testing.assert_allclose(
        forced["candidate_scores"][:, 4],
        materialized["candidate_scores"][:, 4],
        rtol=1e-10,
        atol=1e-10,
    )


def test_aom_ridge_force_moments_extends_wide_banded_cap():
    rng = np.random.default_rng(20260610)
    X = rng.standard_normal((40, 320))
    y = 0.8 * X[:, 2] - 0.35 * X[:, 17] + 0.25 * X[:, 101]
    y += 0.04 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chain = [[("finite_difference", (1,))]]
    lambdas = [0.1, 1.0]

    forced = native.aom_chain_sweep_run(
        X,
        y,
        chain,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )
    materialized = native.aom_chain_sweep_run(
        X,
        y,
        chain,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="materialized",
        score_only=True,
    )
    auto = native.aom_chain_sweep_run(
        X,
        y,
        chain,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="auto",
        score_only=True,
    )
    full = native.aom_chain_sweep_run(
        X,
        y,
        chain,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
    )

    assert forced["n_candidates"] == float(len(lambdas))
    assert forced["n_materialized_candidates"] == 0.0
    assert forced["n_banded_operator_moment_candidates"] == forced["n_candidates"]
    assert forced["n_ridge_moment_score_batch_calls"] == 1.0
    assert forced["n_ridge_moment_score_batch_jobs"] == len(lambdas) * 4
    assert forced["n_ridge_moment_final_fits"] == 0.0
    np.testing.assert_allclose(
        forced["candidate_scores"][:, 4],
        materialized["candidate_scores"][:, 4],
        rtol=1e-10,
        atol=1e-10,
    )
    assert full["n_materialized_candidates"] == 0.0
    assert full["n_banded_operator_moment_candidates"] == full["n_candidates"]
    assert full["predictions"].shape == (X.shape[0], 1)
    assert full["input_coefficients"].shape == (X.shape[1], 1)
    np.testing.assert_allclose(
        full["candidate_scores"][:, 4],
        forced["candidate_scores"][:, 4],
        rtol=1e-10,
        atol=1e-10,
    )
    assert auto["n_materialized_candidates"] == auto["n_candidates"]
    np.testing.assert_allclose(
        auto["candidate_scores"][:, 4],
        materialized["candidate_scores"][:, 4],
        rtol=1e-10,
        atol=1e-10,
    )


def _assert_aom_route_partitions(res):
    assert (
        res["n_ridge_operator_moment_candidates"]
        + res["n_pls_operator_moment_candidates"]
        == res["n_operator_moment_candidates"]
    )
    assert (
        res["n_ridge_materialized_candidates"]
        + res["n_pls_materialized_candidates"]
        == res["n_materialized_candidates"]
    )
    routes = np.asarray(res["candidate_routes"], dtype=np.int32)
    assert routes.shape == (int(res["n_candidates"]),)
    assert np.count_nonzero(routes == 0) == int(res["n_materialized_candidates"])
    assert np.count_nonzero(routes == 1) == int(
        res["n_dense_operator_moment_candidates"]
    )
    assert np.count_nonzero(routes == 2) == int(
        res["n_banded_operator_moment_candidates"]
    )
    assert np.count_nonzero(routes == 3) == int(
        res["n_structured_operator_moment_candidates"]
    )


def test_moment_screen_backend_recommendation_live_crossover_policy():
    assert hasattr(native, "moment_screen_backend_recommendation")

    small = native.moment_screen_backend_recommendation(
        260,
        48,
        head="pls",
        cuda_available=True,
    )
    assert small["recommended_backend"] == "cpu"
    assert small["reason"] == "below_live_crossover"
    assert "dataset" not in small["policy_inputs"]
    assert "min_cuda_product" in small["policy_inputs"]

    medium_ridge = native.moment_screen_backend_recommendation(
        260,
        256,
        head="ridge",
        cuda_available=True,
    )
    assert medium_ridge["recommended_backend"] == "cpu"
    assert medium_ridge["reason"] == "below_live_crossover"
    assert medium_ridge["work_product"] == 260 * 256

    wide_ridge = native.moment_screen_backend_recommendation(
        512,
        512,
        head="ridge",
        cuda_available=True,
    )
    assert wide_ridge["recommended_backend"] == "cuda"
    assert wide_ridge["reason"] == "at_or_above_live_crossover"
    assert wide_ridge["work_product"] == 512 * 512
    assert wide_ridge["uses_cuda_pls_device_component_loop"] is False

    wide_pls = native.moment_screen_backend_recommendation(
        256,
        1024,
        head="pls",
        cuda_available=True,
    )
    assert wide_pls["recommended_backend"] == "cuda"
    assert wide_pls["uses_cuda_pls_device_component_loop"] is True
    assert wide_pls["cuda_pls_device_component_min_features"] == 1024
    assert wide_pls["uses_cuda_pls_fold_workspace"] is True
    assert wide_pls["cuda_pls_fold_workspace_min_features"] == 1024
    assert wide_pls["uses_cuda_pls_many_batched"] is False
    assert wide_pls["cuda_pls_many_batched"] is None

    medium_pls_forced = native.moment_screen_backend_recommendation(
        260,
        256,
        head="pls",
        cuda_available=True,
        min_cuda_product=260 * 256,
        cuda_pls_min_device_features=256,
        cuda_pls_many_batched=True,
    )
    assert medium_pls_forced["recommended_backend"] == "cuda"
    assert medium_pls_forced["uses_cuda_pls_device_component_loop"] is True
    assert medium_pls_forced["cuda_pls_device_component_min_features"] == 256
    assert medium_pls_forced["uses_cuda_pls_many_batched"] is True
    assert medium_pls_forced["cuda_pls_many_batched"] is True

    unavailable = native.moment_screen_backend_recommendation(
        512,
        512,
        head=1,
        cuda_available=False,
    )
    assert unavailable["recommended_backend"] == "cpu"
    assert unavailable["reason"] == "cuda_unavailable"
    assert unavailable["uses_cuda_pls_many_batched"] is False
    assert unavailable["head"] == "pls"


def test_moment_facade_aliases_native_surface_without_shadowing_moments():
    from n4m._impl import moment_facade as moment

    assert n4m._impl.moment_facade is moment
    assert callable(native.moments)
    assert moment.moments is native.moments
    assert moment.sweep_run is native.sweep_run
    assert moment.pls_cross_validate is native.pls_cross_validate
    assert moment.aom_preprocess is native.aom_preprocess
    assert moment.aom_global_select is native.aom_global_select
    assert moment.aom_per_component_select is native.aom_per_component_select
    assert moment.aom_sweep_run is native.aom_sweep_run
    assert moment.aom_chain_sweep_run is native.aom_chain_sweep_run
    assert (
        moment.aom_moment_screen_refit_campaign
        is native.aom_moment_screen_refit_campaign
    )
    assert moment.aom_screen_refit_candidate_pool is native.aom_screen_refit_candidate_pool
    assert moment.aom_refit_execution_plan is native.aom_refit_execution_plan
    assert moment.aom_refit_candidates is native.aom_refit_candidates
    assert moment.aom_chain_fixed_fit_run is native.aom_chain_fixed_fit_run
    assert moment.aom_robust_hpo is native.aom_robust_hpo
    assert moment.aom_ridge_blender is native.aom_ridge_blender
    assert moment.aom_operator_pls_stack is native.aom_operator_pls_stack
    assert moment.build_aom_strict_chain_grid is native.build_aom_strict_chain_grid
    assert moment.iter_aom_strict_chain_grid is native.iter_aom_strict_chain_grid
    assert moment.decode_aom_chains is native.decode_aom_chains
    assert moment.aom_candidate_table is native.aom_candidate_table
    assert moment.aom_evaluate_candidates is native.aom_evaluate_candidates
    assert moment.aom_candidate_operator_summary is native.aom_candidate_operator_summary
    assert (
        moment.aom_candidate_preprocessing_impact
        is native.aom_candidate_preprocessing_impact
    )
    assert moment.aom_candidate_route_summary is native.aom_candidate_route_summary
    assert moment.aom_candidate_rank_diagnostics is native.aom_candidate_rank_diagnostics
    assert moment.aom_candidate_report_records is native.aom_candidate_report_records
    assert moment.aom_save_candidate_report is native.aom_save_candidate_report
    assert moment.aom_load_candidate_report is native.aom_load_candidate_report
    assert moment.ridge is native.ridge
    assert moment.pls is native.pls
    assert moment.pcr is native.pcr
    assert moment.cppls is native.cppls
    assert moment.weighted_pls is native.weighted_pls
    assert moment.robust_pls is native.robust_pls
    assert moment.ridge_pls is native.ridge_pls
    assert moment.continuum_regression is native.continuum_regression
    assert moment.ecr is native.ecr
    assert (
        moment.moment_screen_backend_recommendation
        is native.moment_screen_backend_recommendation
    )
    assert moment.NativeMomentSweepRegressor is NativeMomentSweepRegressor
    assert moment.NativeAOMChainSweepRegressor is NativeAOMChainSweepRegressor
    assert moment.NativeAOMFixedCandidateRegressor is NativeAOMFixedCandidateRegressor
    assert moment.NativeAOMScreenRefitRegressor is NativeAOMScreenRefitRegressor
    assert (
        moment.NativeAOMMomentScreenRefitRegressor
        is NativeAOMMomentScreenRefitRegressor
    )
    assert (
        moment.NativeAOMMomentPLSScreenRefitRegressor
        is NativeAOMMomentPLSScreenRefitRegressor
    )
    assert (
        moment.NativeAOMMomentPLSExactScreenRefitRegressor
        is NativeAOMMomentPLSExactScreenRefitRegressor
    )
    assert (
        moment.NativeAOMMomentRidgeScreenRefitRegressor
        is NativeAOMMomentRidgeScreenRefitRegressor
    )
    assert (
        moment.NativeAOMOperatorPLSStackRegressor
        is NativeAOMOperatorPLSStackRegressor
    )
    assert moment.NativeAOMPLSRegressor is NativeAOMPLSRegressor
    assert moment.NativeAOMRidgeBlenderRegressor is NativeAOMRidgeBlenderRegressor
    assert moment.NativeAOMRobustHPORegressor is NativeAOMRobustHPORegressor
    assert moment.NativeAOMSweepRegressor is NativeAOMSweepRegressor
    assert moment.NativePOPPLSRegressor is NativePOPPLSRegressor
    assert moment.NativeRidgeRegressor is NativeRidgeRegressor
    assert moment.NativePLSRegressor is NativePLSRegressor
    assert moment.NativePCRRegressor is NativePCRRegressor
    assert moment.NativeCPPLSRegressor is NativeCPPLSRegressor
    assert moment.NativeWeightedPLSRegressor is NativeWeightedPLSRegressor
    assert moment.NativeRobustPLSRegressor is NativeRobustPLSRegressor
    assert moment.NativeRidgePLSRegressor is NativeRidgePLSRegressor
    assert (
        moment.NativeContinuumRegressionRegressor
        is NativeContinuumRegressionRegressor
    )
    assert moment.NativeECRRegressor is NativeECRRegressor
    assert moment.NativeMomentStackRegressor is NativeMomentStackRegressor
    assert native_sklearn.NativeRidgeRegressor is NativeRidgeRegressor
    assert native_sklearn.NativePCRRegressor is NativePCRRegressor
    assert native_sklearn.NativeWeightedPLSRegressor is NativeWeightedPLSRegressor
    assert native_sklearn.NativeRobustPLSRegressor is NativeRobustPLSRegressor
    assert native_sklearn.NativeRidgePLSRegressor is NativeRidgePLSRegressor
    assert native_sklearn.NativeMomentStackRegressor is NativeMomentStackRegressor
    assert moment.moment_stack is native.moment_stack
    inventory = moment.available_methods()
    inventory_names = {row["name"] for row in inventory}
    assert {
        "moments",
        "moment_sweep",
        "sweep_run",
        "pls_cross_validate",
        "aom_preprocess",
        "aom_chain_sweep",
        "aom_chain_sweep_regressor",
        "aom_profile_sweep",
        "aom_profile_sweep_regressor",
        "ridge",
        "ridge_regressor",
        "pls",
        "pls_regressor",
        "pcr",
        "pcr_regressor",
        "cppls",
        "cppls_regressor",
        "weighted_pls",
        "weighted_pls_regressor",
        "robust_pls",
        "robust_pls_regressor",
        "ridge_pls",
        "ridge_pls_regressor",
        "continuum_regression",
        "continuum_regression_regressor",
        "ecr",
        "ecr_regressor",
        "moment_stack",
        "moment_stack_regressor",
        "aom_moment_screen_refit_campaign",
        "aom_screen_refit_regressor",
        "moment_mixed_screen_refit",
        "moment_pls_screen_refit",
        "moment_pls_exact_screen_refit",
        "moment_ridge_screen_refit",
        "aom_ridge_blender",
        "aom_operator_pls_stack",
        "aom_robust_hpo",
        "aom_pls",
        "pop_pls",
        "aom_screen_refit_candidate_pool",
        "aom_refit_execution_plan",
        "aom_refit_candidates",
        "aom_chain_fixed_fit",
        "fixed_candidate",
        "backend_recommendation",
        "build_aom_strict_chain_grid",
        "iter_aom_strict_chain_grid",
        "decode_aom_chains",
        "aom_candidate_table",
        "aom_evaluate_candidates",
        "aom_candidate_operator_summary",
        "aom_candidate_preprocessing_impact",
        "aom_candidate_route_summary",
        "aom_candidate_rank_diagnostics",
        "aom_candidate_report_records",
        "aom_save_candidate_report",
        "aom_load_candidate_report",
    }.issubset(inventory_names)
    assert all(row["cpu"] and row["cuda"] for row in inventory)
    assert all("config_options" in row for row in inventory)
    json.dumps(inventory)
    inventory_by_name = {row["name"]: row for row in inventory}
    assert {
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
        "score_only",
    }.issubset(inventory_by_name["sweep_run"]["config_options"])
    assert {
        "component_grid",
        "fold_ids",
        "cuda_pls_parallel_folds",
        "cuda_pls_many_batched",
        "score_only",
    }.issubset(inventory_by_name["pls_cross_validate"]["config_options"])
    assert inventory_by_name["pls_cross_validate"]["wrapper_of"] == "sweep_run"
    assert inventory_by_name["pls_cross_validate"]["catalog_role"] == (
        "campaign_helper"
    )
    assert "score_only" not in inventory_by_name["moment_sweep"]["config_options"]
    assert inventory_by_name["aom_preprocess"]["entry"] == "aom_preprocess"
    assert inventory_by_name["aom_chain_sweep"]["entry"] == "aom_chain_sweep_run"
    assert inventory_by_name["aom_chain_sweep_regressor"]["entry"] == (
        "NativeAOMChainSweepRegressor"
    )
    assert inventory_by_name["aom_profile_sweep"]["entry"] == "aom_sweep_run"
    assert inventory_by_name["aom_profile_sweep_regressor"]["entry"] == (
        "NativeAOMSweepRegressor"
    )
    assert "score_only" in inventory_by_name["aom_chain_sweep"]["config_options"]
    assert "score_only" not in inventory_by_name[
        "aom_chain_sweep_regressor"
    ]["config_options"]
    assert {
        "start",
        "stop",
        "chunk_size",
        "with_ids",
    }.issubset(inventory_by_name["iter_aom_strict_chain_grid"]["config_options"])
    assert inventory_by_name["ridge"]["config_options"] == (
        "alpha",
        "center_x",
        "scale_x",
        "center_y",
    )
    assert inventory_by_name["ridge_regressor"]["entry"] == "NativeRidgeRegressor"
    assert {
        "n_components",
        "pls_components",
        "cv",
        "fold_ids",
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
    }.issubset(inventory_by_name["pls_regressor"]["config_options"])
    assert inventory_by_name["pls"]["entry"] == "pls"
    assert inventory_by_name["pls_regressor"]["entry"] == "NativePLSRegressor"
    assert inventory_by_name["pcr"]["config_options"] == (
        "n_components",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
    )
    assert inventory_by_name["pcr_regressor"]["entry"] == "NativePCRRegressor"
    assert inventory_by_name["weighted_pls"]["config_options"] == (
        "sample_weights",
        "n_components",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
    )
    assert inventory_by_name["weighted_pls_regressor"]["entry"] == (
        "NativeWeightedPLSRegressor"
    )
    assert inventory_by_name["weighted_pls_regressor"]["wrapper_of"] == "weighted_pls"
    assert inventory_by_name["robust_pls"]["config_options"] == (
        "huber_k",
        "max_irls_iter",
        "n_components",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
    )
    assert inventory_by_name["robust_pls_regressor"]["entry"] == (
        "NativeRobustPLSRegressor"
    )
    assert inventory_by_name["robust_pls_regressor"]["wrapper_of"] == "robust_pls"
    assert inventory_by_name["ridge_pls"]["config_options"] == (
        "ridge_lambda",
        "n_components",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
    )
    assert inventory_by_name["ridge_pls_regressor"]["entry"] == (
        "NativeRidgePLSRegressor"
    )
    assert inventory_by_name["ridge_pls_regressor"]["wrapper_of"] == "ridge_pls"
    assert "kernel_pls" not in inventory_by_name
    assert inventory_by_name["moment_stack_regressor"]["entry"] == (
        "NativeMomentStackRegressor"
    )
    assert inventory_by_name["moment_stack"]["entry"] == "moment_stack"
    assert inventory_by_name["moment_stack_regressor"]["wrapper_of"] == "moment_stack"
    assert {
        "base_models",
        "cv",
        "inner_cv",
        "meta_alpha",
        "n_components",
        "cuda_pls_min_device_features",
    }.issubset(inventory_by_name["moment_stack_regressor"]["config_options"])
    assert inventory_by_name["aom_moment_screen_refit_campaign"][
        "wrapper_of"
    ] == "aom_chain_screen_refit_campaign"
    assert inventory_by_name["moment_mixed_screen_refit"][
        "entry"
    ] == "NativeAOMMomentScreenRefitRegressor"
    assert inventory_by_name["moment_pls_screen_refit"][
        "entry"
    ] == "NativeAOMMomentPLSScreenRefitRegressor"
    assert inventory_by_name["moment_pls_exact_screen_refit"][
        "entry"
    ] == "NativeAOMMomentPLSExactScreenRefitRegressor"
    assert inventory_by_name["moment_ridge_screen_refit"][
        "entry"
    ] == "NativeAOMMomentRidgeScreenRefitRegressor"
    assert inventory_by_name["aom_screen_refit_regressor"][
        "entry"
    ] == "NativeAOMScreenRefitRegressor"
    assert inventory_by_name["aom_ridge_blender"][
        "entry"
    ] == "NativeAOMRidgeBlenderRegressor"
    assert inventory_by_name["aom_operator_pls_stack"][
        "entry"
    ] == "NativeAOMOperatorPLSStackRegressor"
    assert inventory_by_name["aom_robust_hpo"][
        "entry"
    ] == "NativeAOMRobustHPORegressor"
    assert inventory_by_name["aom_pls"]["entry"] == "NativeAOMPLSRegressor"
    assert inventory_by_name["pop_pls"]["entry"] == "NativePOPPLSRegressor"
    assert inventory_by_name["aom_screen_refit_candidate_pool"][
        "entry"
    ] == "aom_screen_refit_candidate_pool"
    assert inventory_by_name["aom_refit_execution_plan"][
        "non_catalog_reason"
    ]
    assert inventory_by_name["aom_refit_candidates"][
        "entry"
    ] == "aom_refit_candidates"
    assert inventory_by_name["aom_chain_fixed_fit"][
        "entry"
    ] == "aom_chain_fixed_fit_run"
    assert inventory_by_name["fixed_candidate"][
        "entry"
    ] == "NativeAOMFixedCandidateRegressor"
    assert {
        "split_head_scoring",
        "chain_ordering",
        "pls_score_mode",
        "refit_execution",
        "backend_min_cuda_product",
    }.issubset(
        inventory_by_name["aom_moment_screen_refit_campaign"]["config_options"]
    )
    assert {
        "split_head_scoring",
        "chain_ordering",
        "pls_score_mode",
        "refit_execution",
        "backend_min_cuda_product",
    }.issubset(inventory_by_name["moment_mixed_screen_refit"]["config_options"])
    assert {
        "fit_mode",
        "precomputed_cv_rmse",
        "moment_policy",
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
    }.issubset(inventory_by_name["fixed_candidate"]["config_options"])
    assert (
        inventory_by_name["continuum_regression_regressor"]["config_options"]
        == ("tau", "n_components")
    )
    assert {
        "n_samples",
        "n_features",
        "min_cuda_product",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
    }.issubset(inventory_by_name["backend_recommendation"]["config_options"])
    assert inventory_by_name["aom_evaluate_candidates"]["config_options"] == (
        "top_k",
        "sort_by",
        "cv",
        "fold_ids",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
        "moment_policy",
        "return_predictions",
    )
    assert inventory_by_name["aom_candidate_operator_summary"]["config_options"] == (
        "score_key",
        "top_k",
    )
    assert inventory_by_name["aom_candidate_preprocessing_impact"][
        "config_options"
    ] == (
        "score_key",
        "top_k",
        "higher_is_better",
    )
    assert inventory_by_name["aom_candidate_route_summary"]["config_options"] == ()
    assert inventory_by_name["aom_candidate_rank_diagnostics"]["config_options"] == (
        "screen_score_key",
        "eval_score_key",
        "cutoffs",
    )
    assert inventory_by_name["aom_save_candidate_report"]["config_options"] == (
        "format",
        "include_predictions",
    )
    assert inventory_by_name["aom_load_candidate_report"]["config_options"] == (
        "format",
    )
    inventory[0]["name"] = "mutated"
    assert moment.available_methods()[0]["name"] != "mutated"

    X, y = _dataset()
    stats = moment.moments(X, y)
    assert stats["xtx"].shape == (X.shape[1], X.shape[1])
    res = moment.sweep_run(
        X,
        y,
        cv=4,
        ridge_lambdas=(0.1,),
        heads=("ridge",),
        scale_x=False,
    )
    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape[0] == 1
    assert np.isfinite(res["selected_cv_rmse"])


def test_moment_screen_backend_recommendation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="head"):
        native.moment_screen_backend_recommendation(260, 256, head="elastic")
    with pytest.raises(ValueError, match="n_samples"):
        native.moment_screen_backend_recommendation(1, 256)
    with pytest.raises(ValueError, match="n_features"):
        native.moment_screen_backend_recommendation(260, 0)


def test_native_moment_model_wrappers_smoke():
    X, y = _dataset()

    _assert_method_result(native.ridge(X, y, alpha=0.1), X.shape[0])
    pls = native.pls(X, y, n_components=3, cv=4, scale_x=False)
    _assert_method_result(pls, X.shape[0])
    assert int(pls["selected_head_id"]) == 1
    assert pls["selected_param"] == pytest.approx(3.0)
    assert int(pls["n_pls_moment_cv_fits"]) > 0
    _assert_method_result(native.pcr(X, y, n_components=3), X.shape[0])
    _assert_method_result(native.cppls(X, y, gamma=0.4, n_components=3), X.shape[0])
    weights = np.linspace(0.5, 1.5, X.shape[0], dtype=np.float64)
    weighted = native.weighted_pls(
        X,
        y,
        sample_weights=weights,
        n_components=3,
        scale_x=False,
    )
    _assert_method_result(weighted, X.shape[0])
    assert "final_weights" not in weighted
    robust = native.robust_pls(
        X,
        y,
        huber_k=1.345,
        max_irls_iter=3,
        n_components=3,
        scale_x=False,
    )
    _assert_method_result(robust, X.shape[0])
    assert robust["huber_k"] == pytest.approx(1.345)
    assert "final_weights" not in robust
    ridge_pls = native.ridge_pls(
        X,
        y,
        ridge_lambda=0.1,
        n_components=3,
        scale_x=False,
    )
    _assert_method_result(ridge_pls, X.shape[0])
    assert ridge_pls["ridge_lambda"] == pytest.approx(0.1)
    _assert_method_result(
        native.continuum_regression(X, y, tau=0.25, n_components=3),
        X.shape[0],
    )
    _assert_method_result(native.ecr(X, y, alpha=0.6, n_components=3), X.shape[0])


def test_direct_weighted_robust_and_ridge_pls_validate_inputs():
    X, y = _dataset()
    with pytest.raises(ValueError, match="sample_weights length"):
        native.weighted_pls(X, y, sample_weights=np.ones(X.shape[0] - 1))
    with pytest.raises(ValueError, match="strictly positive"):
        native.weighted_pls(X, y, sample_weights=np.zeros(X.shape[0]))
    with pytest.raises(ValueError, match="huber_k"):
        native.robust_pls(X, y, huber_k=0.0)
    with pytest.raises(ValueError, match="ridge_lambda"):
        native.ridge_pls(X, y, ridge_lambda=-1.0)


@pytest.mark.parametrize(
    ("Estimator", "native_fn", "kwargs", "method_name"),
    [
        (
            NativeRidgeRegressor,
            native.ridge,
            {"alpha": 0.1, "scale_x": False},
            "ridge",
        ),
        (
            NativePLSRegressor,
            native.pls,
            {"n_components": 3, "cv": 4, "scale_x": False},
            "pls",
        ),
        (
            NativePCRRegressor,
            native.pcr,
            {"n_components": 3, "scale_x": False},
            "pcr",
        ),
        (
            NativeCPPLSRegressor,
            native.cppls,
            {"gamma": 0.4, "n_components": 3},
            "cppls",
        ),
        (
            NativeWeightedPLSRegressor,
            native.weighted_pls,
            {
                "sample_weights": np.linspace(0.5, 1.5, 28),
                "n_components": 3,
                "scale_x": False,
            },
            "weighted_pls",
        ),
        (
            NativeRobustPLSRegressor,
            native.robust_pls,
            {
                "huber_k": 1.345,
                "max_irls_iter": 5,
                "n_components": 3,
                "scale_x": False,
            },
            "robust_pls",
        ),
        (
            NativeRidgePLSRegressor,
            native.ridge_pls,
            {"ridge_lambda": 0.1, "n_components": 3, "scale_x": False},
            "ridge_pls",
        ),
        (
            NativeContinuumRegressionRegressor,
            native.continuum_regression,
            {"tau": 0.25, "n_components": 3},
            "continuum_regression",
        ),
        (
            NativeECRRegressor,
            native.ecr,
            {"alpha": 0.6, "n_components": 3},
            "ecr",
        ),
    ],
)
def test_native_direct_moment_sklearn_wrappers_replay_native_predictions(
    Estimator,
    native_fn,
    kwargs,
    method_name,
):
    X, y = _dataset()
    native = native_fn(X, y, **kwargs)

    model = Estimator(**kwargs).fit(X, y)

    assert model.n_features_in_ == X.shape[1]
    assert model.n_targets_ == 1
    assert model.coef_.shape == (X.shape[1],)
    assert np.isfinite(model.intercept_)
    assert np.isfinite(model.score(X, y))
    np.testing.assert_allclose(
        model.predict(X),
        native["predictions"].ravel(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        X @ model.input_coefficients_ + np.asarray(model.intercept_).reshape(1, -1),
        native["predictions"],
        rtol=1e-10,
        atol=1e-10,
    )

    diagnostics = model.get_diagnostics()
    assert diagnostics["method"] == method_name
    assert diagnostics["n_samples"] == X.shape[0]
    assert diagnostics["n_features"] == X.shape[1]
    assert diagnostics["n_targets"] == 1
    assert np.isfinite(diagnostics["rmse"])


def test_native_direct_moment_sklearn_wrappers_support_multi_output():
    X, y = _dataset()
    Y = np.column_stack([y, -0.3 * y + 0.1 * X[:, 1]])

    for Estimator, kwargs in (
        (NativeRidgeRegressor, {"alpha": 0.1, "scale_x": False}),
        (NativePLSRegressor, {"n_components": 2, "cv": 4, "scale_x": False}),
        (NativePCRRegressor, {"n_components": 2, "scale_x": False}),
        (NativeCPPLSRegressor, {"gamma": 0.4, "n_components": 2}),
        (
            NativeWeightedPLSRegressor,
            {
                "sample_weights": np.linspace(0.5, 1.5, X.shape[0]),
                "n_components": 2,
                "scale_x": False,
            },
        ),
        (NativeRidgePLSRegressor, {"ridge_lambda": 0.1, "n_components": 2}),
        (NativeContinuumRegressionRegressor, {"tau": 0.25, "n_components": 2}),
        (NativeECRRegressor, {"alpha": 0.6, "n_components": 2}),
    ):
        model = Estimator(**kwargs).fit(X, Y)
        pred = model.predict(X)
        assert pred.shape == Y.shape
        assert model.coef_.shape == (Y.shape[1], X.shape[1])
        assert model.n_targets_ == Y.shape[1]
        np.testing.assert_allclose(
            pred,
            model.predictions_,
            rtol=1e-10,
            atol=1e-10,
        )


def test_native_moment_stack_regressor_smoke():
    X, y = _dataset()
    model = NativeMomentStackRegressor(
        base_models=("ridge", "pcr", "pls"),
        cv=4,
        inner_cv=3,
        n_components=2,
        scale_x=False,
    ).fit(X, y)

    pred = model.predict(X)
    assert pred.shape == y.shape
    assert model.base_model_names_ == ("ridge", "pcr", "pls")
    assert model.base_oof_predictions_.shape == (X.shape[0], 3)
    assert model.oof_predictions_.shape == (X.shape[0], 1)
    assert model.meta_coefficients_.shape == (3, 1)
    assert np.isfinite(model.oof_rmse_)
    assert np.isfinite(model.rmse_)
    assert np.isfinite(model.score(X, y))

    diagnostics = model.get_diagnostics()
    assert diagnostics["method"] == "moment_stack"
    assert diagnostics["base_models"] == ("ridge", "pcr", "pls")
    assert diagnostics["cv"] == 4
    assert diagnostics["inner_cv"] == 3
    assert set(diagnostics["base_oof_rmse"]) == {"ridge", "pcr", "pls"}
    assert len(diagnostics["base_oof_diagnostics"]) == 4 * 3
    assert len(diagnostics["base_final_diagnostics"]) == 3
    assert any(
        row["phase"] == "oof"
        and row["base_model"] == "pls"
        and row["estimator"] == "NativePLSRegressor"
        and row["method"] == "pls"
        for row in diagnostics["base_oof_diagnostics"]
    )
    assert any(
        row["phase"] == "final"
        and row["base_model"] == "pls"
        and row["estimator"] == "NativePLSRegressor"
        and row["method"] == "pls"
        for row in diagnostics["base_final_diagnostics"]
    )
    assert diagnostics["n_base_oof_pls_moment_cv_fits"] >= 0
    assert diagnostics["n_base_final_pls_moment_cv_fits"] >= 0

    factory_model = native.moment_stack(
        X,
        y,
        base_models=("ridge", "pcr"),
        cv=4,
        n_components=2,
        scale_x=False,
    )
    assert isinstance(factory_model, NativeMomentStackRegressor)
    assert factory_model.predict(X).shape == y.shape


def test_native_aom_pls_and_pop_pls_selector_wrappers_smoke():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = [
        "identity",
        ("savgol_smooth", [5, 2]),
        ("finite_difference", [1]),
    ]

    aom = native.aom_pls(
        X,
        y,
        max_components=2,
        operators=operators,
        cv=4,
        fold_ids=folds,
        scale_x=False,
    )

    assert aom["predictions"].shape == (X.shape[0], 1)
    assert aom["operator_kinds"].shape == (3,)
    assert aom["operator_scores"].shape == (3,)
    assert aom["rmse_curves"].shape == (3, 2)
    assert aom["coefficients"].shape == (X.shape[1], 1)
    assert aom["input_coefficients"].shape == (X.shape[1], 1)
    assert aom["intercept"].shape == (1, 1)
    assert 0.0 <= aom["selected_operator_index"] < 3.0
    assert aom["selected_n_components"] in {1.0, 2.0}
    assert np.isfinite(aom["best_score"])
    np.testing.assert_array_equal(aom["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ aom["input_coefficients"] + aom["intercept"],
        aom["predictions"],
        rtol=1e-10,
        atol=1e-10,
    )

    pop = native.aom_per_component_select(
        X,
        y,
        max_components=2,
        operators=operators,
        cv=4,
        fold_ids=folds,
        scale_x=False,
    )

    assert pop["predictions"].shape == (X.shape[0], 1)
    assert pop["operator_kinds"].shape == (3,)
    assert pop["component_scores"].shape == (2, 3)
    assert pop["prefix_scores"].shape == (2,)
    assert pop["coefficients"].shape == (X.shape[1], 1)
    assert pop["input_coefficients"].shape == (X.shape[1], 1)
    assert pop["intercept"].shape == (1, 1)
    assert 1 <= pop["selected_operator_indices"].size <= 2
    assert np.all((pop["selected_operator_indices"] >= 0) & (pop["selected_operator_indices"] < 3))
    assert np.isfinite(pop["best_score"])
    np.testing.assert_array_equal(pop["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ pop["input_coefficients"] + pop["intercept"],
        pop["predictions"],
        rtol=1e-10,
        atol=1e-10,
    )

    alias = native.pop_pls(
        X,
        y,
        max_components=1,
        operators=["identity"],
        cv=4,
        fold_ids=folds,
        scale_x=False,
    )
    assert alias["component_scores"].shape == (1, 1)


def test_native_aom_preprocess_identity_smoke():
    from n4m._impl import aom_facade as aom

    X, y = _aom_dataset()
    res = native.aom_preprocess(X, y, operators=["identity"], gating_mode="soft")

    assert aom.aom_preprocess is native.aom_preprocess
    assert res["transformed"].shape == X.shape
    assert res["operator_outputs"].shape == (1, X.size)
    assert res["weights"].shape == (1, 1)
    assert res["operator_kinds"].dtype == np.int64
    assert res["operator_kinds"].tolist() == [0]
    assert res["n_operators"] == 1.0
    assert res["n_samples"] == float(X.shape[0])
    assert res["n_features"] == float(X.shape[1])
    assert res["mode"] == 1.0
    np.testing.assert_allclose(res["transformed"], X)
    np.testing.assert_allclose(res["operator_outputs"].reshape(X.shape), X)
    np.testing.assert_allclose(res["weights"], [[1.0]])

    hard = native.aom_preprocess(X, operators=["identity"], gating_mode="hard")
    np.testing.assert_allclose(hard["transformed"], X)
    assert hard["mode"] == 0.0


def test_native_aom_preprocess_direct_strict_linear_bank():
    X, y = _aom_dataset()
    operators = [
        "identity",
        ("detrend_poly", [1]),
        ("savgol_smooth", [5, 2]),
        ("savgol_derivative", [5, 2, 1]),
        ("norris_williams", [5, 5, 1]),
        ("finite_difference", [1]),
        ("gaussian", [1.0]),
        ("whittaker", [100.0]),
        ("fck", [1.0]),
    ]
    expected_kinds = [0, 7, 8, 9, 10, 15, 18, 16, 17]

    soft = native.aom_preprocess(X, y, operators=operators, gating_mode="soft")
    outputs = np.asarray(soft["operator_outputs"]).reshape(len(operators), *X.shape)

    assert soft["transformed"].shape == X.shape
    assert soft["operator_outputs"].shape == (len(operators), X.size)
    assert soft["weights"].shape == (1, len(operators))
    assert soft["operator_kinds"].tolist() == expected_kinds
    assert soft["n_operators"] == float(len(operators))
    np.testing.assert_allclose(
        soft["weights"], np.full((1, len(operators)), 1 / len(operators))
    )
    np.testing.assert_allclose(
        soft["transformed"], np.mean(outputs, axis=0), rtol=1e-12, atol=1e-12
    )

    hard = native.aom_preprocess(X, y, operators=operators, gating_mode="hard")
    hard_outputs = np.asarray(hard["operator_outputs"]).reshape(len(operators), *X.shape)
    np.testing.assert_allclose(
        hard["weights"], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(
        hard["transformed"], hard_outputs[0], rtol=1e-12, atol=1e-12
    )


def test_native_moment_model_wrappers_support_multi_output_y():
    X, y = _dataset()
    Y = np.column_stack([y, 0.5 * y + 0.1])

    ridge_res = native.ridge(X, Y, alpha=0.1)
    continuum_res = native.continuum_regression(X, Y, tau=0.25, n_components=3)

    _assert_method_result(ridge_res, X.shape[0], n_targets=2)
    _assert_method_result(continuum_res, X.shape[0], n_targets=2)


def _expected_moments(X, y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    x_sum = X.sum(axis=0, keepdims=True)
    y_sum = Y.sum(axis=0, keepdims=True)
    return {
        "x_sum": x_sum,
        "y_sum": y_sum,
        "xtx": X.T @ X,
        "xty": X.T @ Y,
        "yty": Y.T @ Y,
        "x_mean": x_sum / n,
        "y_mean": y_sum / n,
        "cxx": X.T @ X - x_sum.T @ x_sum / n,
        "cxy": X.T @ Y - x_sum.T @ y_sum / n,
        "cyy": Y.T @ Y - y_sum.T @ y_sum / n,
    }


def _assert_moments_close(got, expected, n_samples, n_features, n_targets):
    assert got["n_samples"] == float(n_samples)
    assert got["n_features"] == float(n_features)
    assert got["n_targets"] == float(n_targets)
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=1e-12, atol=1e-12)


def test_native_moments_compute_and_subset():
    X, y = _dataset()
    Y = np.column_stack([y, 0.5 * y + 0.1])
    idx = np.array([0, 3, 4, 9, 10], dtype=np.int64)

    all_moments = native.moments(X, Y)
    subset_moments = native.moments(X, Y, row_indices=idx)

    _assert_moments_close(all_moments, _expected_moments(X, Y), X.shape[0], X.shape[1], 2)
    _assert_moments_close(
        subset_moments,
        _expected_moments(X[idx], Y[idx]),
        idx.size,
        X.shape[1],
        2,
    )


def test_native_moments_train_from_heldout_recenters_after_subtract():
    X, y = _dataset()
    heldout = np.array([1, 2, 7, 8], dtype=np.int64)
    keep = np.ones(X.shape[0], dtype=bool)
    keep[heldout] = False

    train_moments = native.moments_train_from_heldout(X, y, heldout)

    _assert_moments_close(
        train_moments,
        _expected_moments(X[keep], y[keep]),
        int(keep.sum()),
        X.shape[1],
        1,
    )


def test_native_sweep_run_ridge_smoke_and_oof_score():
    X, y = _dataset()
    res = native.sweep_run(
        X,
        y,
        cv=4,
        ridge_lambdas=[0.01, 0.1, 1.0],
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (3, 4)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["coefficients"].shape == (X.shape[1], 1)
    assert res["fold_ids"].shape == (X.shape[0],)
    assert np.all(np.isfinite(res["candidate_scores"][:, 3]))
    assert res["n_ridge_moment_candidates"] == 3.0
    assert res["n_ridge_dual_materialized_candidates"] == 0.0
    assert res["n_ridge_moment_cv_fits"] == 12.0
    assert res["n_ridge_dual_materialized_cv_fits"] == 0.0
    assert res["n_ridge_dual_cross_cv_fits"] == 0.0
    assert res["n_ridge_moment_final_fits"] == 1.0
    assert res["n_ridge_dual_materialized_final_fits"] == 0.0

    selected_id = int(res["selected_candidate_id"])
    assert selected_id == int(np.argmin(res["candidate_scores"][:, 3]))
    oof_rmse = np.sqrt(np.mean((res["oof_predictions"][:, 0] - y) ** 2))
    np.testing.assert_allclose(oof_rmse, res["selected_cv_rmse"], rtol=1e-12, atol=1e-12)


def test_native_sweep_run_ridge_moment_scores_match_numpy_path():
    X, y = _dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    lambdas = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float64)
    expected = []
    Y = y.reshape(-1, 1)
    for lam in lambdas:
        sse = 0.0
        count = 0
        for fold in sorted(set(folds.tolist())):
            train = folds != fold
            test = folds == fold
            lhs = X[train].T @ X[train] + lam * np.eye(X.shape[1])
            rhs = X[train].T @ Y[train]
            beta = np.linalg.solve(lhs, rhs)
            pred = X[test] @ beta
            sse += float(np.sum((Y[test] - pred) ** 2))
            count += int(Y[test].size)
        expected.append(np.sqrt(sse / count))

    res = native.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas.tolist(),
        pls_components=[],
        heads=("ridge",),
        center_x=False,
        center_y=False,
        scale_x=False,
    )

    np.testing.assert_allclose(
        res["candidate_scores"][:, 3],
        np.asarray(expected),
        rtol=1e-8,
        atol=1e-10,
    )
    single = native.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[],
        heads=("ridge",),
        center_x=False,
        center_y=False,
        scale_x=False,
        score_only=True,
    )
    assert res["n_ridge_moment_eigen_path_preparations"] == 4.0
    assert res["n_ridge_moment_eigen_path_cv_fits"] == len(lambdas) * 4
    assert res["n_ridge_moment_direct_cv_fits"] == 0.0
    assert single["n_ridge_moment_eigen_path_preparations"] == 0.0
    assert single["n_ridge_moment_eigen_path_cv_fits"] == 0.0
    assert single["n_ridge_moment_direct_cv_fits"] == 4.0
    np.testing.assert_allclose(
        single["candidate_scores"][0, 3],
        res["candidate_scores"][2, 3],
        rtol=1e-10,
        atol=1e-10,
    )
    assert res["n_ridge_moment_cv_fits"] == len(lambdas) * 4


def test_native_sweep_run_blas_sse_scores_match_scalar_build():
    root = Path(__file__).resolve().parents[3]
    dev_lib = root / "build/dev-release/cpp/src/libn4m.so"
    blas_lib = root / "build/blas-on/cpp/src/libn4m.so"
    if not dev_lib.exists() or not blas_lib.exists():
        pytest.skip("requires dev-release and blas-on libn4m builds")

    code = r"""
import json
import numpy as np
from n4m._impl import native as n4m

rng = np.random.default_rng(20260606)
X = rng.standard_normal((200, 80))
beta = np.zeros(X.shape[1])
beta[[0, 7, 19, 45, 72]] = [0.7, -0.35, 0.25, -0.18, 0.12]
y = X @ beta + 0.03 * rng.standard_normal(X.shape[0])
folds = np.arange(X.shape[0], dtype=np.int32) % 4

res = n4m.sweep_run(
    X,
    y,
    cv=4,
    fold_ids=folds,
    ridge_lambdas=[0.001, 0.01, 0.1, 1.0, 10.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    center_x=False,
    center_y=False,
    scale_x=False,
    score_only=True,
)
print(json.dumps({
    "candidate_scores": np.asarray(res["candidate_scores"], dtype=float).tolist(),
    "selected_candidate_id": int(res["selected_candidate_id"]),
    "selected_cv_rmse": float(res["selected_cv_rmse"]),
    "selected_head_id": float(res["selected_head_id"]),
    "selected_param": float(res["selected_param"]),
}))
"""

    def run_with_lib(lib_path: Path) -> dict:
        env = os.environ.copy()
        env["N4M_LIB_PATH"] = str(lib_path)
        env["PYTHONPATH"] = str(root / "bindings/python/src")
        env["OPENBLAS_NUM_THREADS"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            text=True,
        )
        return json.loads(out)

    scalar = run_with_lib(dev_lib)
    blas = run_with_lib(blas_lib)

    scalar_scores = np.asarray(scalar["candidate_scores"], dtype=float)
    blas_scores = np.asarray(blas["candidate_scores"], dtype=float)
    assert blas_scores.shape == scalar_scores.shape
    np.testing.assert_allclose(blas_scores, scalar_scores, rtol=1e-10, atol=1e-10)
    assert blas["selected_candidate_id"] == scalar["selected_candidate_id"]
    assert blas["selected_head_id"] == pytest.approx(
        scalar["selected_head_id"], rel=0.0, abs=1e-12
    )
    assert blas["selected_param"] == pytest.approx(
        scalar["selected_param"], rel=0.0, abs=1e-12
    )
    assert blas["selected_cv_rmse"] == pytest.approx(
        scalar["selected_cv_rmse"], rel=1e-10, abs=1e-10
    )


def test_native_aom_pls_moment_batch_omp_scores_match_scalar_build():
    root = Path(__file__).resolve().parents[3]
    dev_lib = root / "build/dev-release/cpp/src/libn4m.so"
    omp_lib = root / "build/omp-on/cpp/src/libn4m.so"
    if not dev_lib.exists() or not omp_lib.exists():
        pytest.skip("requires dev-release and omp-on libn4m builds")

    code = r"""
import json
import numpy as np
from n4m._impl import native as n4m

rng = np.random.default_rng(20260607)
X = rng.standard_normal((96, 16))
y = (
    0.65 * X[:, 0]
    - 0.3 * X[:, 4]
    + 0.18 * X[:, 9]
    + 0.04 * rng.standard_normal(X.shape[0])
)
folds = np.arange(X.shape[0], dtype=np.int32) % 4
chains = [
    [("identity", ())],
    [("savgol_smooth", (5, 2))],
    [("savgol_derivative", (7, 2, 1))],
    [("finite_difference", (1,))],
]

res = n4m.aom_chain_sweep_run(
    X,
    y,
    chains,
    cv=4,
    fold_ids=folds,
    ridge_lambdas=[],
    pls_components=[1, 2, 3],
    heads=("pls",),
    scale_x=False,
    moment_policy="force_moments",
    pls_score_mode="cv",
    score_only=True,
)
print(json.dumps({
    "candidate_scores": np.asarray(res["candidate_scores"], dtype=float).tolist(),
    "selected_candidate_id": int(res["selected_candidate_id"]),
    "selected_chain_id": int(res["selected_chain_id"]),
    "selected_cv_rmse": float(res["selected_cv_rmse"]),
    "selected_head_id": float(res["selected_head_id"]),
    "selected_param": float(res["selected_param"]),
    "n_pls_moment_score_batch_calls": int(res["n_pls_moment_score_batch_calls"]),
    "n_pls_moment_score_batch_jobs": int(res["n_pls_moment_score_batch_jobs"]),
}))
"""

    def run_with_lib(lib_path: Path) -> dict:
        env = os.environ.copy()
        env["N4M_LIB_PATH"] = str(lib_path)
        env["PYTHONPATH"] = str(root / "bindings/python/src")
        env["OMP_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            text=True,
        )
        return json.loads(out)

    scalar = run_with_lib(dev_lib)
    omp = run_with_lib(omp_lib)

    scalar_scores = np.asarray(scalar["candidate_scores"], dtype=float)
    omp_scores = np.asarray(omp["candidate_scores"], dtype=float)
    assert omp_scores.shape == scalar_scores.shape
    np.testing.assert_allclose(omp_scores, scalar_scores, rtol=1e-10, atol=1e-10)
    assert scalar["n_pls_moment_score_batch_calls"] == 1
    assert omp["n_pls_moment_score_batch_calls"] == 1
    assert scalar["n_pls_moment_score_batch_jobs"] == 4 * 4
    assert omp["n_pls_moment_score_batch_jobs"] == 4 * 4
    assert omp["selected_candidate_id"] == scalar["selected_candidate_id"]
    assert omp["selected_chain_id"] == scalar["selected_chain_id"]
    assert omp["selected_head_id"] == pytest.approx(
        scalar["selected_head_id"], rel=0.0, abs=1e-12
    )
    assert omp["selected_param"] == pytest.approx(
        scalar["selected_param"], rel=0.0, abs=1e-12
    )
    assert omp["selected_cv_rmse"] == pytest.approx(
        scalar["selected_cv_rmse"], rel=1e-10, abs=1e-10
    )


def test_native_sweep_run_accepts_explicit_folds_multi_output_y():
    X, y = _dataset()
    Y = np.column_stack([y, 0.5 * y + 0.1])
    folds = np.arange(X.shape[0], dtype=np.int32) % 5

    res = native.sweep_run(
        X,
        Y,
        cv=5,
        fold_ids=folds,
        ridge_lambdas=[0.1, 1.0],
        scale_x=False,
    )

    assert res["oof_predictions"].shape == (X.shape[0], 2)
    assert res["coefficients"].shape == (X.shape[1], 2)
    np.testing.assert_array_equal(res["fold_ids"], folds)


def test_native_sweep_run_pls_head_smoke():
    X, y = _dataset()
    res = native.sweep_run(
        X,
        y,
        cv=4,
        ridge_lambdas=[0.1],
        pls_components=[1, 2, 3],
        heads=("pls",),
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (3, 4)
    assert np.all(res["candidate_scores"][:, 1] == 1.0)
    assert res["n_pls_moment_candidates"] == 3.0
    assert res["n_pls_moment_cv_fits"] == 4.0
    assert res["n_pls_moment_host_cv_fits"] == 4.0
    assert res["n_pls_moment_cuda_device_cv_fits"] == 0.0
    assert res["n_pls_moment_final_fits"] == 1.0
    assert res["n_pls_moment_host_final_fits"] == 1.0
    assert res["n_pls_moment_cuda_device_final_fits"] == 0.0
    assert res["n_pls_materialized_cv_fits"] == 0.0
    assert res["n_pls_materialized_final_fits"] == 0.0
    assert res["selected_head_id"] == 1.0
    assert res["selected_param"] in {1.0, 2.0, 3.0}
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    oof_rmse = np.sqrt(np.mean((res["oof_predictions"][:, 0] - y) ** 2))
    np.testing.assert_allclose(oof_rmse, res["selected_cv_rmse"], rtol=1e-12, atol=1e-12)


def test_native_sweep_run_score_only_keeps_scores_and_skips_outputs():
    X, y = _dataset()
    res = native.sweep_run(
        X,
        y,
        cv=4,
        pls_components=[1, 2, 3],
        heads=("pls",),
        scale_x=False,
        score_only=True,
    )

    assert res["candidate_scores"].shape == (3, 4)
    assert res["n_pls_moment_candidates"] == 3.0
    assert res["n_pls_moment_cv_fits"] == 4.0
    assert res["n_pls_moment_host_cv_fits"] == 4.0
    assert res["n_pls_moment_cuda_device_cv_fits"] == 0.0
    assert res["n_pls_moment_score_batch_calls"] == 0.0
    assert res["n_pls_moment_score_batch_jobs"] == 0.0
    assert res["n_pls_moment_final_fits"] == 0.0
    assert res["n_pls_moment_host_final_fits"] == 0.0
    assert res["n_pls_moment_cuda_device_final_fits"] == 0.0
    assert res["score_only"] == 1.0
    assert np.all(np.isfinite(res["candidate_scores"][:, 3]))
    assert res["oof_predictions"].shape == (0, 0)
    assert res["predictions"].shape == (0, 0)
    assert res["coefficients"].shape == (0, 0)
    assert res["intercept"].shape == (0, 0)


def test_native_sweep_run_wide_ridge_score_only_matches_full_scores():
    rng = np.random.default_rng(54)
    X = rng.standard_normal((9, 14))
    y = 0.6 * X[:, 0] - 0.2 * X[:, 5] + 0.15 * X[:, 11]
    folds = np.arange(X.shape[0], dtype=np.int32) % 3
    lambdas = [0.01, 0.4]

    full = native.sweep_run(
        X,
        y,
        cv=3,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        heads=("ridge",),
        scale_x=True,
    )
    score_only = native.sweep_run(
        X,
        y,
        cv=3,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        heads=("ridge",),
        scale_x=True,
        score_only=True,
    )

    np.testing.assert_allclose(
        score_only["candidate_scores"],
        full["candidate_scores"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert full["n_ridge_moment_candidates"] == 0.0
    assert full["n_ridge_dual_materialized_candidates"] == float(len(lambdas))
    assert full["n_ridge_moment_cv_fits"] == 0.0
    assert (
        full["n_ridge_dual_materialized_cv_fits"]
        + full["n_ridge_dual_cross_cv_fits"]
        == float(len(lambdas) * 3)
    )
    assert full["n_ridge_dual_materialized_final_fits"] == 1.0
    assert score_only["n_ridge_dual_materialized_final_fits"] == 0.0
    assert score_only["score_only"] == 1.0
    assert score_only["predictions"].shape == (0, 0)
    assert score_only["oof_predictions"].shape == (0, 0)
    assert score_only["coefficients"].shape == (0, 0)


def test_native_sweep_run_ridge_moment_score_only_matches_full_scores():
    rng = np.random.default_rng(55)
    X = rng.standard_normal((18, 6))
    y = 0.6 * X[:, 0] - 0.2 * X[:, 3] + 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 3
    lambdas = [0.01, 0.4, 2.0]

    full = native.sweep_run(
        X,
        y,
        cv=3,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        heads=("ridge",),
        scale_x=True,
    )
    score_only = native.sweep_run(
        X,
        y,
        cv=3,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        heads=("ridge",),
        scale_x=True,
        score_only=True,
    )

    np.testing.assert_allclose(
        score_only["candidate_scores"],
        full["candidate_scores"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert full["n_ridge_moment_candidates"] == float(len(lambdas))
    assert full["n_ridge_dual_materialized_candidates"] == 0.0
    assert full["n_ridge_moment_cv_fits"] == float(len(lambdas) * 3)
    assert full["n_ridge_moment_eigen_path_preparations"] == 3.0
    assert full["n_ridge_moment_eigen_path_cv_fits"] == float(len(lambdas) * 3)
    assert full["n_ridge_moment_direct_cv_fits"] == 0.0
    assert full["n_ridge_dual_materialized_cv_fits"] == 0.0
    assert full["n_ridge_dual_cross_cv_fits"] == 0.0
    assert full["n_ridge_moment_final_fits"] == 1.0
    assert score_only["n_ridge_moment_eigen_path_preparations"] == 3.0
    assert score_only["n_ridge_moment_eigen_path_cv_fits"] == float(
        len(lambdas) * 3
    )
    assert score_only["n_ridge_moment_direct_cv_fits"] == 0.0
    assert score_only["n_ridge_moment_final_fits"] == 0.0
    assert score_only["score_only"] == 1.0
    assert score_only["predictions"].shape == (0, 0)
    assert score_only["oof_predictions"].shape == (0, 0)
    assert score_only["coefficients"].shape == (0, 0)


def test_native_aom_sweep_run_compact_smoke_and_pls_only():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    res = native.aom_sweep_run(
        X,
        y,
        profile="compact",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1, 2],
        heads=("ridge", "pls"),
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (36, 5)
    assert res["n_chains"] == 12.0
    assert res["chain_offsets"].shape == (13,)
    assert res["chain_offsets"][0] == 0
    assert res["chain_offsets"][-1] == res["op_kinds"].size
    assert res["param_offsets"].shape == (res["op_kinds"].size + 1,)
    assert res["param_offsets"][0] == 0
    assert res["param_offsets"][-1] == res["chain_params"].size
    decoded = native.decode_aom_chains(res)
    assert len(decoded) == 12
    assert decoded[0] == [("identity", ())]
    assert decoded[3] == [("savgol_smooth", (5.0, 2.0))]
    selected_id = int(res["selected_candidate_id"])
    table = native.aom_candidate_table(res, sort=True)
    assert table[0]["candidate_id"] == selected_id
    assert table[0]["cv_rmse"] <= table[-1]["cv_rmse"]
    assert table[0]["chain"] == decoded[table[0]["chain_id"]]
    assert "score_route_id" in table[0]
    assert "score_route" in table[0]
    assert (
        res["n_operator_moment_candidates"] + res["n_materialized_candidates"]
        == res["n_candidates"]
    )
    _assert_aom_route_partitions(res)
    assert (
        res["n_ridge_operator_moment_candidates"]
        + res["n_ridge_materialized_candidates"]
        > 0
    )
    assert (
        res["n_pls_operator_moment_candidates"]
        + res["n_pls_materialized_candidates"]
        > 0
    )
    assert (
        res["n_banded_operator_moment_candidates"]
        + res["n_structured_operator_moment_candidates"]
        + res["n_dense_operator_moment_candidates"]
        == res["n_operator_moment_candidates"]
    )
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["coefficients"].shape == (X.shape[1], 1)
    np.testing.assert_array_equal(res["fold_ids"], folds)
    assert selected_id == int(np.argmin(res["candidate_scores"][:, 4]))
    oof_rmse = np.sqrt(np.mean((res["oof_predictions"][:, 0] - y) ** 2))
    np.testing.assert_allclose(oof_rmse, res["selected_cv_rmse"], rtol=1e-12, atol=1e-12)

    pls_only = native.aom_sweep_run(
        X,
        y,
        profile="compact",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
    )
    assert pls_only["candidate_scores"].shape == (12, 5)
    assert np.all(pls_only["candidate_scores"][:, 2] == 1.0)
    _assert_aom_route_partitions(pls_only)
    assert pls_only["n_ridge_operator_moment_candidates"] == 0.0
    assert pls_only["n_pls_operator_moment_candidates"] == pls_only["n_operator_moment_candidates"]


def test_native_aom_chain_sweep_run_custom_chains_smoke():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("detrend", [1])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]

    res = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (9, 5)
    assert res["profile"] == -1.0
    assert res["n_chains"] == 3.0
    np.testing.assert_array_equal(res["chain_offsets"], np.array([0, 1, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(res["op_kinds"], np.array([0, 7, 8, 15], dtype=np.int32))
    np.testing.assert_array_equal(res["param_offsets"], np.array([0, 0, 1, 3, 4], dtype=np.int32))
    np.testing.assert_allclose(res["chain_params"], np.array([[1.0, 5.0, 2.0, 1.0]]))
    decoded = native.decode_aom_chains(res)
    assert decoded == [
        [("identity", ())],
        [("detrend_poly", (1.0,))],
        [("savgol_smooth", (5.0, 2.0)), ("finite_difference", (1.0,))],
    ]
    assert native.decode_aom_chains(
        res["chain_offsets"],
        res["op_kinds"],
        res["param_offsets"],
        res["chain_params"],
    ) == decoded
    assert (
        res["n_operator_moment_candidates"] + res["n_materialized_candidates"]
        == res["n_candidates"]
    )
    _assert_aom_route_partitions(res)
    assert (
        res["n_ridge_operator_moment_candidates"]
        + res["n_ridge_materialized_candidates"]
        > 0
    )
    assert (
        res["n_pls_operator_moment_candidates"]
        + res["n_pls_materialized_candidates"]
        > 0
    )
    assert (
        res["n_banded_operator_moment_candidates"]
        + res["n_structured_operator_moment_candidates"]
        + res["n_dense_operator_moment_candidates"]
        == res["n_operator_moment_candidates"]
    )
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    np.testing.assert_array_equal(res["fold_ids"], folds)
    selected_id = int(res["selected_candidate_id"])
    assert selected_id == int(np.argmin(res["candidate_scores"][:, 4]))
    oof_rmse = np.sqrt(np.mean((res["oof_predictions"][:, 0] - y) ** 2))
    np.testing.assert_allclose(oof_rmse, res["selected_cv_rmse"], rtol=1e-12, atol=1e-12)

    pls_only = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
    )
    assert pls_only["candidate_scores"].shape == (3, 5)
    assert np.all(pls_only["candidate_scores"][:, 2] == 1.0)
    _assert_aom_route_partitions(pls_only)
    assert pls_only["n_ridge_operator_moment_candidates"] == 0.0
    assert pls_only["n_pls_operator_moment_candidates"] == pls_only["n_operator_moment_candidates"]


def test_native_aom_chain_fixed_fit_run_matches_single_candidate_final_fit():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chain = [("savgol_smooth", [5, 2])]

    for head, param in (("ridge", 0.1), ("pls", 1.0)):
        full = native.aom_chain_sweep_run(
            X,
            y,
            [chain],
            cv=4,
            fold_ids=folds,
            ridge_lambdas=[param] if head == "ridge" else [],
            pls_components=[int(param)] if head == "pls" else [],
            heads=(head,),
            scale_x=False,
        )
        cuda_kwargs = (
            {
                "cuda_pls_parallel_folds": True,
                "cuda_pls_min_device_features": 1,
                "cuda_pls_many_batched": True,
            }
            if head == "pls"
            else {}
        )
        fixed = native.aom_chain_fixed_fit_run(
            X,
            y,
            chain,
            head=head,
            param=param,
            scale_x=False,
            **cuda_kwargs,
        )

        assert fixed["candidate_scores"].shape == (1, 5)
        assert fixed["profile"] == -1.0
        assert fixed["cv"] == 0.0
        assert fixed["score_only"] == 0.0
        assert np.isnan(fixed["selected_cv_rmse"])
        assert fixed["oof_predictions"].shape == (0, 0)
        assert fixed["fold_ids"].shape == (0,)
        assert fixed["n_pls_moment_cv_fits"] == 0.0
        assert fixed["n_pls_materialized_cv_fits"] == 0.0
        np.testing.assert_allclose(
            fixed["predictions"],
            full["predictions"],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            fixed["coefficients"],
            full["coefficients"],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            fixed["input_coefficients"],
            full["input_coefficients"],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            fixed["intercept"],
            full["intercept"],
            rtol=1e-10,
            atol=1e-10,
        )
        if head == "pls":
            assert (
                fixed["n_pls_moment_final_fits"]
                + fixed["n_pls_materialized_final_fits"]
                == 1.0
            )


def test_native_aom_strict_chain_grid_and_score_campaign():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    compact = native.build_aom_strict_chain_grid("compact")
    wide = native.build_aom_strict_chain_grid("wide")
    assert len(compact) == 12
    assert len(wide) == 31
    assert compact[0] == [("identity", ())]
    assert [("whittaker", (100.0,))] in wide
    assert [("gaussian", (1.0,))] in wide
    assert [("gaussian", (2.0,))] in wide
    assert [("fck", (0.0,))] in wide
    assert [("fck", (1.0,))] in wide

    families = {
        "identity": ["identity"],
        "detrend_poly": [("detrend", [1])],
        "savgol_smooth": [("savgol_smooth", [5, 2])],
        "savgol_derivative": [("savgol_derivative", [7, 2, 1])],
        "finite_difference": [("finite_difference", [1])],
    }
    chains = native.build_aom_strict_chain_grid(
        "lab",
        families=families,
        templates=[
            ("identity",),
            ("detrend_poly", "savgol_derivative"),
            ("savgol_smooth", "finite_difference"),
        ],
    )
    assert chains == [
        [("identity", ())],
        [("detrend_poly", (1.0,)), ("savgol_derivative", (7.0, 2.0, 1.0))],
        [("savgol_smooth", (5.0, 2.0)), ("finite_difference", (1.0,))],
    ]
    assert list(native.iter_aom_strict_chain_grid("compact")) == compact
    assert list(native.iter_aom_strict_chain_grid("wide")) == wide
    assert list(
        native.iter_aom_strict_chain_grid(
            "lab",
            families=families,
            templates=[
                ("identity",),
                ("detrend_poly", "savgol_derivative"),
                ("savgol_smooth", "finite_difference"),
            ],
        )
    ) == chains
    assert list(
        native.iter_aom_strict_chain_grid(
            "lab",
            families=families,
            templates=[
                ("identity",),
                ("detrend_poly", "savgol_derivative"),
                ("savgol_smooth", "finite_difference"),
            ],
            start=1,
            stop=3,
            with_ids=True,
        )
    ) == [(1, chains[1]), (2, chains[2])]
    assert list(
        native.iter_aom_strict_chain_grid(
            "lab",
            families=families,
            templates=[
                ("identity",),
                ("detrend_poly", "savgol_derivative"),
                ("savgol_smooth", "finite_difference"),
            ],
            chunk_size=2,
            with_ids=True,
        )
    ) == [[(0, chains[0]), (1, chains[1])], [(2, chains[2])]]
    assert list(
        native.iter_aom_strict_chain_grid("compact", include_identity=False, max_chains=3)
    ) == native.build_aom_strict_chain_grid(
        "compact", include_identity=False, max_chains=3
    )

    campaign = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        chain_chunk_size=2,
        top_k=3,
    )
    assert campaign["n_chains"] == 3
    assert campaign["n_chunks"] == 2
    assert campaign["n_candidates"] == 6
    assert set(campaign["moment_backend_recommendations"]) == {"ridge", "pls"}
    assert campaign["backend_cuda_available"] is None
    assert campaign["backend_min_cuda_product"] is None
    assert campaign["moment_backend_recommendation_policy_inputs"] == (
        "n_samples",
        "n_features",
        "head",
        "cuda_available",
        "min_cuda_product",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
    )
    for head, recommendation in campaign["moment_backend_recommendations"].items():
        assert recommendation["head"] == head
        assert recommendation["n_samples"] == X.shape[0]
        assert recommendation["n_features"] == X.shape[1]
        assert recommendation["policy_inputs"] == (
            "n_samples",
            "n_features",
            "head",
            "cuda_available",
            "min_cuda_product",
            "cuda_pls_min_device_features",
            "cuda_pls_many_batched",
        )
        assert "dataset" not in recommendation["policy_inputs"]
    assert len(campaign["top_candidates"]) == 3
    assert campaign["best"] == campaign["top_candidates"][0]
    assert set(campaign["top_candidates_by_head"]) == {"ridge", "pls"}
    assert set(campaign["best_by_head"]) == {"ridge", "pls"}
    for head, rows in campaign["top_candidates_by_head"].items():
        assert rows
        assert len(rows) <= 3
        assert campaign["best_by_head"][head] == rows[0]
        assert all(row["head"] == head for row in rows)
        head_scores = [row["cv_rmse"] for row in rows]
        assert head_scores == sorted(head_scores)
    assert campaign["top_candidates_by_score_route"]
    assert set(campaign["best_by_score_route"]) == set(
        campaign["top_candidates_by_score_route"]
    )
    for route, rows in campaign["top_candidates_by_score_route"].items():
        assert rows
        assert len(rows) <= 3
        assert campaign["best_by_score_route"][route] == rows[0]
        assert all(row["score_route"] == route for row in rows)
        route_scores = [row["cv_rmse"] for row in rows]
        assert route_scores == sorted(route_scores)
    assert (
        campaign["n_operator_moment_candidates"]
        + campaign["n_materialized_candidates"]
        == campaign["n_candidates"]
    )
    assert (
        campaign["n_ridge_operator_moment_candidates"]
        + campaign["n_pls_operator_moment_candidates"]
        == campaign["n_operator_moment_candidates"]
    )
    assert (
        campaign["n_ridge_materialized_candidates"]
        + campaign["n_pls_materialized_candidates"]
        == campaign["n_materialized_candidates"]
    )
    assert campaign["candidates_per_second"] > 0.0
    assert campaign["chains_per_second"] > 0.0
    assert campaign["ms_per_candidate"] > 0.0
    assert campaign["ms_per_chain"] > 0.0
    np.testing.assert_allclose(
        campaign["operator_moment_candidate_fraction"]
        + campaign["materialized_candidate_fraction"],
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )
    for chunk in campaign["chunks"]:
        assert chunk["candidates_per_second"] > 0.0
        assert chunk["ms_per_candidate"] > 0.0
        np.testing.assert_allclose(
            chunk["operator_moment_candidate_fraction"]
            + chunk["materialized_candidate_fraction"],
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
    scores = [row["cv_rmse"] for row in campaign["top_candidates"]]
    assert scores == sorted(scores)
    for row in campaign["top_candidates"]:
        assert row["chain"] == chains[row["chain_id"]]
        assert row["chunk_index"] in {0, 1}
        assert row["head"] in {"ridge", "pls"}

    forced_backend = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        chain_chunk_size=2,
        top_k=3,
        backend_cuda_available=True,
        backend_min_cuda_product=1,
    )
    assert forced_backend["backend_cuda_available"] is True
    assert forced_backend["backend_min_cuda_product"] == 1
    assert forced_backend["best"]["cv_rmse"] == campaign["best"]["cv_rmse"]
    for recommendation in forced_backend["moment_backend_recommendations"].values():
        assert recommendation["min_cuda_product"] == 1
        assert recommendation["recommended_backend"] == "cuda"
        assert recommendation["cuda_available_source"] == "caller_override"

    summary = native.aom_candidate_operator_summary(campaign)
    assert summary["score_key"] == "cv_rmse"
    assert summary["n_candidates"] == len(campaign["top_candidates"])
    assert summary["best"]["chain"] == campaign["best"]["chain"]
    assert summary["best"]["head"] == campaign["best"]["head"]
    np.testing.assert_allclose(
        summary["best"]["cv_rmse"],
        campaign["best"]["cv_rmse"],
    )
    assert {row["group"] for row in summary["by_head"]} <= {"ridge", "pls"}
    assert "identity" in {row["group"] for row in summary["by_operator"]}
    assert all(row["n_candidates"] > 0 for row in summary["by_operator_head"])

    route_summary = native.aom_candidate_route_summary(campaign)
    assert route_summary["report_schema"] == "n4m.aom_candidate_route_summary.v1"
    assert route_summary["row_scope"] == "top_candidates"
    assert route_summary["n_candidates"] == len(campaign["top_candidates"])
    assert route_summary["reported_total"]["n_candidates"] == campaign["n_candidates"]
    assert (
        route_summary["reported_total"]["n_operator_moment_candidates"]
        == campaign["n_operator_moment_candidates"]
    )
    assert (
        route_summary["reported_total"]["n_materialized_candidates"]
        == campaign["n_materialized_candidates"]
    )
    assert route_summary["rows_match_report_total"] is (
        len(campaign["top_candidates"]) == campaign["n_candidates"]
    )
    assert route_summary["n_chains"] >= 1
    assert (
        route_summary["n_operator_moment_candidates"]
        + route_summary["n_materialized_candidates"]
        + route_summary["n_unknown_route_candidates"]
        == route_summary["n_candidates"]
    )
    assert set(route_summary["by_head"]) <= {"ridge", "pls"}
    assert set(route_summary["by_score_route"]) == {
        row["score_route"] for row in campaign["top_candidates"]
    }
    assert route_summary["all_operator_moment"] == (
        route_summary["n_operator_moment_candidates"]
        == route_summary["n_candidates"]
    )

    model = NativeAOMFixedCandidateRegressor.from_candidate(
        campaign["best"],
        cv=4,
        fold_ids=folds,
        scale_x=False,
    ).fit(X, y)
    assert hasattr(native_sklearn, "NativeAOMFixedCandidateRegressor")
    assert model.result_["candidate_scores"].shape == (1, 5)
    assert model.selected_candidate_id_ == 0
    assert model.result_["selected_chain_id"] == 0.0
    np.testing.assert_allclose(
        model.selected_cv_rmse_,
        campaign["best"]["cv_rmse"],
        rtol=1e-10,
        atol=1e-10,
    )

    fck_X = np.tile(X, (6, 1)) + 0.01 * np.arange(6).repeat(X.shape[0])[:, None]
    fck_y = 0.6 * fck_X[:, 0] - 0.35 * fck_X[:, 3] + 0.2 * fck_X[:, 7]
    fck_folds = np.arange(fck_X.shape[0], dtype=np.int32) % 4
    fck_campaign = native.aom_chain_score_campaign(
        fck_X,
        fck_y,
        chains=[[("fck", (0.0,))]],
        cv=4,
        fold_ids=fck_folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=1,
        top_k=2,
    )
    assert fck_campaign["n_candidates"] == 2
    assert fck_campaign["n_operator_moment_candidates"] == 2
    assert fck_campaign["n_materialized_candidates"] == 0
    assert fck_campaign["n_banded_operator_moment_candidates"] == 2

    gaussian_campaign = native.aom_chain_score_campaign(
        fck_X,
        fck_y,
        chains=[[("gaussian", (1.0,))]],
        cv=4,
        fold_ids=fck_folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=1,
        top_k=2,
    )
    assert gaussian_campaign["n_candidates"] == 2
    assert gaussian_campaign["n_operator_moment_candidates"] == 2
    assert gaussian_campaign["n_materialized_candidates"] == 0
    assert gaussian_campaign["n_banded_operator_moment_candidates"] == 2
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-8,
        atol=1e-8,
    )

    global_model = NativeAOMFixedCandidateRegressor.from_campaign(
        campaign,
        cv=4,
        fold_ids=folds,
        scale_x=False,
    ).fit(X, y)
    np.testing.assert_allclose(
        global_model.selected_cv_rmse_,
        campaign["best"]["cv_rmse"],
        rtol=1e-10,
        atol=1e-10,
    )

    for head in ("ridge", "pls"):
        head_model = NativeAOMFixedCandidateRegressor.from_campaign(
            campaign,
            head=head,
            cv=4,
            fold_ids=folds,
            scale_x=False,
        ).fit(X, y)
        assert head_model.head == head
        np.testing.assert_allclose(
            head_model.selected_cv_rmse_,
            campaign["best_by_head"][head]["cv_rmse"],
            rtol=1e-10,
            atol=1e-10,
        )

    second_pls = NativeAOMFixedCandidateRegressor.from_campaign(
        campaign,
        head="pls",
        rank=1,
        cv=4,
        fold_ids=folds,
        scale_x=False,
    )
    assert second_pls.head == "pls"
    assert second_pls.param == campaign["top_candidates_by_head"]["pls"][1]["param"]

    with pytest.raises(ValueError, match="rank is outside"):
        NativeAOMFixedCandidateRegressor.from_campaign(
            campaign,
            head="pls",
            rank=100,
        )
    with pytest.raises(ValueError, match="head"):
        NativeAOMFixedCandidateRegressor.from_campaign(
            {"top_candidates": []},
            head="ridge",
        )
    assert model.coef_.shape == (X.shape[1],)


def test_native_aom_score_campaign_prefix_ordering_preserves_scores_and_cache():
    X, y = _aom_dataset()
    X = np.tile(X, (8, 1)) + 0.002 * np.arange(8).repeat(X.shape[0])[:, None]
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("savgol_smooth", (5, 2)), ("finite_difference", (1,))],
        [("gaussian", (1.0,)), ("finite_difference", (1,))],
        [("savgol_smooth", (5, 2)), ("finite_difference", (2,))],
        [("gaussian", (1.0,)), ("finite_difference", (2,))],
    ]
    common = dict(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=2,
        top_k=4,
    )

    input_report = native.aom_chain_score_campaign(
        X, y, chain_ordering="input", **common
    )
    prefix_report = native.aom_chain_score_campaign(
        X, y, chain_ordering="prefix", **common
    )

    assert input_report["chain_ordering"] == "input"
    assert prefix_report["chain_ordering"] == "prefix"
    assert prefix_report["n_candidates"] == input_report["n_candidates"] == 4
    assert prefix_report["n_materialized_candidates"] == 0
    assert prefix_report["n_moment_prefix_cache_hits"] > input_report[
        "n_moment_prefix_cache_hits"
    ]
    assert prefix_report["moment_prefix_cache_hit_fraction"] > input_report[
        "moment_prefix_cache_hit_fraction"
    ]
    assert any(
        row["ordered_chain_id"] != row["chain_id"]
        for row in prefix_report["top_candidates"]
    )
    assert all(
        row["chain"] == chains[int(row["chain_id"])]
        for row in prefix_report["top_candidates"]
    )

    def signature(report):
        return sorted(
            (
                int(row["chain_id"]),
                row["head"],
                float(row["param"]),
                round(float(row["cv_rmse"]), 12),
            )
            for row in report["top_candidates"]
        )

    assert signature(prefix_report) == signature(input_report)


def test_native_aom_score_campaign_checkpoint_resume(tmp_path):
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = native.build_aom_strict_chain_grid("compact", max_chains=6)

    kwargs = dict(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        chain_chunk_size=2,
        top_k=4,
    )
    expected = native.aom_chain_score_campaign(X, y, **kwargs)

    checkpoint = tmp_path / "aom_campaign_checkpoint.json"
    partial = native.aom_chain_score_campaign(
        X,
        y,
        checkpoint_path=checkpoint,
        max_chunks_per_run=1,
        **kwargs,
    )
    assert checkpoint.exists()
    assert partial["complete"] is False
    assert partial["n_chunks"] == 1
    assert partial["n_total_chunks"] == 3
    assert partial["n_remaining_chunks"] == 2
    assert partial["processed_chunks_this_run"] == 1
    assert partial["resumed_from_checkpoint"] is False

    saved = native.aom_chain_score_campaign(
        X,
        y,
        checkpoint_path=checkpoint,
        max_chunks_per_run=2,
        **kwargs,
    )
    assert saved["complete"] is True
    assert saved["n_chunks"] == saved["n_total_chunks"] == 3
    assert saved["n_remaining_chunks"] == 0
    assert saved["processed_chunks_this_run"] == 2
    assert saved["resumed_from_checkpoint"] is True

    payload = json.loads(checkpoint.read_text())
    first_chunk = payload["chunks"][0]
    payload["chunks"] = [first_chunk]
    payload["top_candidates"] = [
        row for row in payload["top_candidates"] if row["chunk_index"] == 0
    ]
    payload["top_candidates_by_head"] = {
        head: [row for row in rows if row["chunk_index"] == 0]
        for head, rows in payload["top_candidates_by_head"].items()
    }
    payload["top_candidates_by_score_route"] = {
        route: [row for row in rows if row["chunk_index"] == 0]
        for route, rows in payload["top_candidates_by_score_route"].items()
    }
    payload["best"] = payload["top_candidates"][0]
    payload["complete"] = False
    payload["n_chunks"] = 1
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    resumed = native.aom_chain_score_campaign(
        X,
        y,
        checkpoint_path=checkpoint,
        backend_cuda_available=True,
        **kwargs,
    )
    assert resumed["complete"] is True
    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["n_chunks"] == resumed["n_total_chunks"] == 3
    assert resumed["n_candidates"] == expected["n_candidates"]
    assert resumed["moment_backend_recommendations"]["pls"][
        "cuda_available_source"
    ] == "caller_override"
    assert resumed["moment_backend_recommendations"]["pls"][
        "cuda_available"
    ] is True

    def signature(report):
        return [
            (
                int(row["chain_id"]),
                row["head"],
                float(row["param"]),
                round(float(row["cv_rmse"]), 12),
            )
            for row in report["top_candidates"]
        ]

    assert signature(resumed) == signature(expected)

    def by_head_signature(report):
        return {
            head: [
                (
                    int(row["chain_id"]),
                    row["head"],
                    float(row["param"]),
                    round(float(row["cv_rmse"]), 12),
                )
                for row in rows
            ]
            for head, rows in report["top_candidates_by_head"].items()
        }

    assert by_head_signature(resumed) == by_head_signature(expected)

    def by_route_signature(report):
        return {
            route: [
                (
                    int(row["chain_id"]),
                    row["head"],
                    float(row["param"]),
                    round(float(row["cv_rmse"]), 12),
                )
                for row in rows
            ]
            for route, rows in report["top_candidates_by_score_route"].items()
        }

    assert by_route_signature(resumed) == by_route_signature(expected)

    with pytest.raises(ValueError, match="checkpoint does not match"):
        native.aom_chain_score_campaign(
            X,
            y,
            checkpoint_path=checkpoint,
            top_k=5,
            **{key: value for key, value in kwargs.items() if key != "top_k"},
        )


def test_native_aom_candidate_holdout_evaluation_report():
    X, y = _aom_dataset()
    X_train = X[:16]
    y_train = y[:16]
    X_eval = X[16:]
    y_eval = y[16:]
    folds = np.arange(X_train.shape[0], dtype=np.int32) % 4
    chains = native.build_aom_strict_chain_grid("compact", max_chains=5)

    campaign = native.aom_chain_score_campaign(
        X_train,
        y_train,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        chain_chunk_size=3,
        top_k=5,
    )
    report = native.aom_evaluate_candidates(
        X_train,
        y_train,
        X_eval,
        y_eval,
        campaign,
        top_k=4,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        return_predictions=True,
    )

    assert report["n_candidates"] == 4
    assert report["best_eval"] == report["rows"][0]
    assert set(row["cv_rank"] for row in report["rows"]) == {1, 2, 3, 4}
    assert set(row["eval_rank"] for row in report["rows"]) == {1, 2, 3, 4}
    assert [row["eval_rmse"] for row in report["rows"]] == sorted(
        row["eval_rmse"] for row in report["rows"]
    )
    for row in report["rows"]:
        source = campaign["top_candidates"][row["source_index"]]
        assert row["chain"] == source["chain"]
        assert row["head"] == source["head"]
        np.testing.assert_allclose(
            row["refit_cv_rmse"],
            source["cv_rmse"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert row["eval_predictions"].shape == (X_eval.shape[0], 1)
        assert np.isfinite(row["eval_rmse"])
        assert np.isfinite(row["eval_r2"])
    summary = native.aom_candidate_operator_summary(report)
    assert summary["score_key"] == "eval_rmse"
    assert summary["best"]["chain"] == report["best_eval"]["chain"]
    assert summary["best"]["head"] == report["best_eval"]["head"]
    np.testing.assert_allclose(
        summary["best"]["eval_rmse"],
        report["best_eval"]["eval_rmse"],
    )
    assert summary["by_operator"][0]["best_score"] <= summary["by_operator"][-1]["best_score"]
    rank_diag = native.aom_candidate_rank_diagnostics(report, cutoffs=(1, 2, 4, 10))
    assert rank_diag["screen_score_key"] == "screen_cv_rmse"
    assert rank_diag["eval_score_key"] == "eval_rmse"
    assert rank_diag["n_candidates"] == 4
    assert -1.0 <= rank_diag["spearman_rank_correlation"] <= 1.0
    assert rank_diag["topk"][0]["k"] == 1
    assert rank_diag["topk"][-1]["effective_k"] == 4
    assert 1 <= rank_diag["best_screen_eval_rank"] <= 4
    assert 1 <= rank_diag["best_eval_screen_rank"] <= 4


def test_aom_candidate_preprocessing_impact_groups_options_and_baseline():
    rows = [
        {
            "chain": [("identity", ())],
            "head": "ridge",
            "cv_rmse": 1.0,
        },
        {
            "chain": [("savgol_smooth", (5, 2))],
            "head": "ridge",
            "cv_rmse": 0.8,
        },
        {
            "chain": [("savgol_smooth", (7, 2)), ("detrend_poly", (1,))],
            "head": "ridge",
            "cv_rmse": 0.7,
        },
        {
            "chain": [("finite_difference", (1,))],
            "head": "pls",
            "cv_rmse": 0.9,
        },
    ]

    impact = native.aom_candidate_preprocessing_impact(rows)

    assert impact["score_key"] == "cv_rmse"
    assert impact["n_candidates"] == 4
    assert impact["identity_baseline"]["score"] == 1.0
    assert impact["identity_baseline_by_head"]["ridge"]["score"] == 1.0
    assert impact["best"]["cv_rmse"] == 0.7

    by_operator = {row["group"]: row for row in impact["by_operator"]}
    assert by_operator["savgol_smooth"]["best_score"] == 0.7
    assert by_operator["savgol_smooth"]["best_improvement_vs_identity"] == pytest.approx(0.3)
    assert by_operator["finite_difference"]["best_improvement_vs_identity"] == pytest.approx(0.1)

    by_stage = {row["group"]: row for row in impact["by_stage_family"]}
    assert by_stage["smooth"]["best_score"] == 0.7
    assert by_stage["baseline"]["best_score"] == 0.7
    assert by_stage["derivative"]["best_score"] == 0.9

    by_option = {row["group"]: row for row in impact["by_stage_option"]}
    assert by_option["smooth|savgol_smooth(7,2)"]["best_score"] == 0.7
    assert by_option["baseline|detrend_poly(1)"]["best_score"] == 0.7

    by_head_option = {row["group"]: row for row in impact["by_head_stage_option"]}
    assert by_head_option["ridge|smooth|savgol_smooth(7,2)"]["best_rank"] == 1


def test_native_aom_candidate_report_export_helpers(tmp_path):
    X, y = _aom_dataset()
    X_train = X[:16]
    y_train = y[:16]
    X_eval = X[16:]
    y_eval = y[16:]
    folds = np.arange(X_train.shape[0], dtype=np.int32) % 4
    chains = native.build_aom_strict_chain_grid("compact", max_chains=4)

    campaign = native.aom_chain_score_campaign(
        X_train,
        y_train,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        chain_chunk_size=2,
        top_k=4,
    )
    report = native.aom_evaluate_candidates(
        X_train,
        y_train,
        X_eval,
        y_eval,
        campaign,
        top_k=3,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        return_predictions=True,
    )
    records = native.aom_candidate_report_records(report)
    assert len(records) == 3
    assert "eval_predictions" not in records[0]
    assert "chain_json" in records[0]

    json_path = tmp_path / "aom_report.json"
    csv_path = tmp_path / "aom_report.csv"
    jsonl_path = tmp_path / "aom_report.jsonl"
    assert native.aom_save_candidate_report(json_path, report) == str(json_path)
    assert native.aom_save_candidate_report(csv_path, report) == str(csv_path)
    assert native.aom_save_candidate_report(jsonl_path, report) == str(jsonl_path)

    payload = json.loads(json_path.read_text())
    assert payload["metadata"]["n_candidates"] == 3
    assert "top_candidates_by_head" not in payload["metadata"]
    assert "top_candidates_by_score_route" not in payload["metadata"]
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["chain_json"] == records[0]["chain_json"]

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert json.loads(rows[0]["chain_json"]) == records[0]["chain"]

    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(jsonl_rows) == 3
    assert jsonl_rows[0]["eval_rank"] == records[0]["eval_rank"]

    loaded_json = native.aom_load_candidate_report(json_path)
    loaded_csv = native.aom_load_candidate_report(csv_path)
    loaded_jsonl = native.aom_load_candidate_report(jsonl_path)
    assert loaded_json[0]["chain"] == loaded_csv[0]["chain"]
    assert loaded_jsonl[0]["chain"] == loaded_csv[0]["chain"]
    assert json.loads(loaded_csv[0]["chain_json"]) == records[0]["chain"]
    assert loaded_csv[0]["head"] in {"ridge", "pls"}
    assert isinstance(loaded_csv[0]["param"], float)
    loaded_summary = native.aom_candidate_operator_summary(loaded_csv)
    assert loaded_summary["score_key"] == "eval_rmse"
    assert loaded_summary["n_candidates"] == len(loaded_csv)
    assert loaded_summary["best"]["chain"] == loaded_csv[0]["chain"]
    loaded_rank_diag = native.aom_candidate_rank_diagnostics(loaded_csv, cutoffs=(1, 3))
    assert loaded_rank_diag["n_candidates"] == len(loaded_csv)
    assert loaded_rank_diag["topk"][1]["effective_k"] == 3
    assert loaded_rank_diag["best_eval"]["chain"] == loaded_csv[0]["chain"]
    refit = NativeAOMFixedCandidateRegressor.from_candidate(
        loaded_csv[0],
        cv=4,
        fold_ids=folds,
        scale_x=False,
    ).fit(X_train, y_train)
    assert refit.predict(X_eval).shape[0] == X_eval.shape[0]


def test_native_aom_sweep_materialized_policy_forces_legacy_route():
    rng = np.random.default_rng(47)
    X = rng.standard_normal((140, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("detrend", [1]), ("finite_difference", [1])],
    ]

    auto = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="auto",
    )
    materialized = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="materialized",
    )

    assert auto["n_operator_moment_candidates"] > 0
    assert materialized["n_operator_moment_candidates"] == 0.0
    assert materialized["n_materialized_candidates"] == materialized["n_candidates"]
    _assert_aom_route_partitions(auto)
    _assert_aom_route_partitions(materialized)
    assert materialized["n_ridge_materialized_candidates"] == 4.0
    assert materialized["n_pls_materialized_candidates"] == 2.0
    np.testing.assert_allclose(
        auto["candidate_scores"][:, :4],
        materialized["candidate_scores"][:, :4],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        auto["candidate_scores"][:, 4],
        materialized["candidate_scores"][:, 4],
        rtol=1e-8,
        atol=1e-8,
    )

    try:
        native.aom_chain_sweep_run(X, y, chains, moment_policy="bad-policy")
    except ValueError as exc:
        assert "moment_policy" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid moment_policy")


def test_native_aom_sweep_force_moments_policy_is_strict():
    rng = np.random.default_rng(49)
    X = rng.standard_normal((140, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("finite_difference", [1])],
    ]

    strict = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
    )

    assert strict["n_operator_moment_candidates"] == strict["n_candidates"]
    assert strict["n_materialized_candidates"] == 0.0
    _assert_aom_route_partitions(strict)
    assert strict["n_ridge_operator_moment_candidates"] == 2.0
    assert strict["n_pls_operator_moment_candidates"] == 2.0

    score_only = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )

    assert score_only["score_only"] == 1.0
    assert score_only["candidate_scores"].shape == strict["candidate_scores"].shape
    assert score_only["n_operator_moment_candidates"] == score_only["n_candidates"]
    assert score_only["n_materialized_candidates"] == 0.0
    _assert_aom_route_partitions(score_only)
    assert score_only["n_ridge_operator_moment_candidates"] == 2.0
    assert score_only["n_pls_operator_moment_candidates"] == 2.0
    assert score_only["predictions"].shape == (0, 0)
    assert score_only["oof_predictions"].shape == (0, 0)
    assert score_only["coefficients"].shape == (0, 0)
    assert score_only["input_coefficients"].shape == (0, 0)
    np.testing.assert_array_equal(score_only["fold_ids"], folds)

    Y = np.column_stack([y, 0.5 * y + 0.1 * X[:, 0]])
    try:
        native.aom_chain_sweep_run(
            X,
            Y,
            [["identity"]],
            cv=4,
            fold_ids=folds,
            ridge_lambdas=[],
            pls_components=[1],
            heads=("pls",),
            scale_x=False,
            moment_policy="force_moments",
        )
    except n4m.N4MError as exc:
        assert exc.status_name == "UNSUPPORTED"
    else:
        raise AssertionError("expected strict force_moments policy to reject fallback")


def test_native_aom_pls_gcv_proxy_score_only_is_explicit_and_moment_only():
    rng = np.random.default_rng(53)
    X = rng.standard_normal((96, 16))
    y = 0.55 * X[:, 0] - 0.30 * X[:, 8] + 0.04 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("finite_difference", [1])],
        [("detrend", [1]), ("finite_difference", [1])],
    ]

    exact = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )
    proxy = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        score_only=True,
    )

    assert exact["aom_pls_score_mode"] == 0.0
    assert proxy["aom_pls_score_mode"] == 1.0
    assert exact["score_only"] == 1.0
    assert exact["n_materialized_candidates"] == 0.0
    assert exact["n_pls_operator_moment_candidates"] == exact["n_candidates"]
    assert exact["n_pls_moment_cv_fits"] == len(chains) * 4
    assert exact["n_pls_moment_score_batch_calls"] == 1.0
    assert exact["n_pls_moment_score_batch_jobs"] == len(chains) * 4
    assert exact["n_pls_gcv_proxy_batch_calls"] == 0.0
    assert exact["n_pls_gcv_proxy_batch_jobs"] == 0.0
    assert proxy["score_only"] == 1.0
    assert proxy["n_materialized_candidates"] == 0.0
    assert proxy["n_pls_operator_moment_candidates"] == proxy["n_candidates"]
    assert proxy["n_pls_gcv_proxy_candidates"] == proxy["n_candidates"]
    assert proxy["n_pls_gcv_proxy_fits"] == len(chains)
    assert proxy["n_pls_gcv_proxy_batch_calls"] == 1.0
    assert proxy["n_pls_gcv_proxy_batch_jobs"] == len(chains)
    assert proxy["n_pls_moment_score_batch_calls"] == 0.0
    assert proxy["n_pls_moment_score_batch_jobs"] == 0.0
    assert proxy["n_pls_moment_cv_fits"] == 0.0
    assert proxy["n_pls_materialized_cv_fits"] == 0.0
    assert exact["n_pls_moment_cv_fits"] > 0.0
    assert np.all(np.isfinite(proxy["candidate_scores"][:, 4]))
    assert "materialized" not in {
        row["score_route"] for row in native.aom_candidate_table(proxy)
    }
    assert {
        row["score_metric"] for row in native.aom_candidate_table(proxy)
    } == {"pls_gcv_proxy_rmse"}

    with pytest.raises(ValueError, match="pls_score_mode"):
        native.aom_chain_sweep_run(
            X,
            y,
            chains,
            cv=4,
            fold_ids=folds,
            ridge_lambdas=[],
            pls_components=[1],
            heads=("pls",),
            scale_x=False,
            pls_score_mode="bad-mode",
            score_only=True,
        )

    with pytest.raises(n4m.N4MError) as excinfo:
        native.aom_chain_sweep_run(
            X,
            y,
            chains,
            cv=4,
            fold_ids=folds,
            ridge_lambdas=[],
            pls_components=[1],
            heads=("pls",),
            scale_x=False,
            moment_policy="force_moments",
            pls_score_mode="gcv_proxy",
            score_only=False,
        )
    assert excinfo.value.status_name == "INVALID_ARGUMENT"

    campaign = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        chain_chunk_size=2,
        top_k=3,
    )
    assert campaign["pls_score_mode"] == "gcv_proxy"
    assert campaign["n_pls_gcv_proxy_candidates"] == campaign["n_candidates"]
    assert campaign["n_pls_moment_cv_fits"] == 0
    assert campaign["pls_cv_fits_per_chain"] == 0.0
    assert campaign["pls_gcv_proxy_fits_per_chain"] > 0.0
    assert campaign["chains_per_second"] > 0.0
    assert campaign["candidates_per_second"] > 0.0
    assert campaign["projected_200k_chains_seconds"] > 0.0
    np.testing.assert_allclose(
        campaign["projected_200k_chains_minutes"],
        campaign["projected_200k_chains_seconds"] / 60.0,
        rtol=1e-12,
        atol=1e-12,
    )
    assert {
        row["score_metric"] for row in campaign["top_candidates"]
    } == {"pls_gcv_proxy_rmse"}

    verified = native.aom_refit_candidates(
        X,
        y,
        campaign,
        top_k=3,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
    )
    assert verified["n_candidates"] == 3
    assert verified["best_cv"] == verified["rows"][0]
    assert {row["screen_score_metric"] for row in verified["rows"]} == {
        "pls_gcv_proxy_rmse"
    }
    assert {row["refit_score_metric"] for row in verified["rows"]} == {"cv_rmse"}
    assert all(row["n_pls_gcv_proxy_fits"] == 0 for row in verified["rows"])
    assert all(row["n_pls_moment_cv_fits"] > 0 for row in verified["rows"])
    assert [row["refit_cv_rmse"] for row in verified["rows"]] == sorted(
        row["refit_cv_rmse"] for row in verified["rows"]
    )

    same_chain_rows = [
        row for row in native.aom_candidate_table(proxy)
        if row["chain_id"] == 0
    ]
    assert {row["param"] for row in same_chain_rows} == {1.0, 2.0}
    individual_refit = native.aom_refit_candidates(
        X,
        y,
        same_chain_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="individual",
    )
    grouped_refit = native.aom_refit_candidates(
        X,
        y,
        same_chain_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="grouped_score",
    )
    assert grouped_refit["execution_mode"] == "grouped_score"
    assert grouped_refit["n_refit_groups"] == 1
    assert grouped_refit["n_pls_moment_cv_fits"] == 4
    assert grouped_refit["n_pls_moment_cv_fits"] < individual_refit[
        "n_pls_moment_cv_fits"
    ]
    individual_by_param = {
        row["param"]: row["refit_cv_rmse"] for row in individual_refit["rows"]
    }
    grouped_by_param = {
        row["param"]: row["refit_cv_rmse"] for row in grouped_refit["rows"]
    }
    assert individual_by_param.keys() == grouped_by_param.keys()
    for param, score in individual_by_param.items():
        np.testing.assert_allclose(
            grouped_by_param[param],
            score,
            rtol=1e-12,
            atol=1e-12,
        )
    assert all(np.isnan(row["train_rmse"]) for row in grouped_refit["rows"])
    assert all(
        row["oof_rmse"] == row["refit_cv_rmse"]
        for row in grouped_refit["rows"]
    )

    all_pls_rows = native.aom_candidate_table(proxy)
    grouped_all_refit = native.aom_refit_candidates(
        X,
        y,
        all_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="grouped_score",
    )
    batched_refit = native.aom_refit_candidates(
        X,
        y,
        all_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="batched_score",
    )
    assert batched_refit["execution_mode"] == "batched_score"
    assert grouped_all_refit["n_refit_groups"] == len(chains)
    assert batched_refit["n_refit_groups"] == 1
    assert batched_refit["n_pls_moment_cv_fits"] == grouped_all_refit[
        "n_pls_moment_cv_fits"
    ]
    grouped_all_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in grouped_all_refit["rows"]
    }
    batched_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in batched_refit["rows"]
    }
    assert grouped_all_by_chain_param.keys() == batched_by_chain_param.keys()
    for key, score in grouped_all_by_chain_param.items():
        np.testing.assert_allclose(
            batched_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )

    mixed_pls_rows = [
        row for row in all_pls_rows
        if (row["chain_id"], row["param"]) in {
            (0, 1.0),
            (1, 2.0),
            (2, 1.0),
        }
    ]
    assert len(mixed_pls_rows) == 3
    plan = native.aom_refit_execution_plan(mixed_pls_rows)
    assert plan["report_schema"] == "n4m.aom_refit_execution_plan.v1"
    assert hasattr(native, "aom_refit_execution_plan")
    assert plan["by_mode"]["individual"]["n_refit_groups"] == 3
    assert plan["by_mode"]["grouped_score"]["n_refit_groups"] == 3
    assert plan["by_mode"]["batched_score"]["n_refit_groups"] == 2
    assert plan["by_mode"]["union_batched_score"]["n_refit_groups"] == 1
    assert plan["by_mode"]["union_batched_score"]["n_refit_scored_candidates"] == 6
    assert plan["by_mode"]["union_batched_score"]["n_refit_extra_scored_candidates"] == 3
    assert plan["recommended_mode"] == "union_batched_score"
    assert plan["recommendation_reason"] == (
        "union_reduces_native_groups_within_extra_budget"
    )
    conservative_plan = native.aom_refit_execution_plan(
        mixed_pls_rows,
        auto_max_extra_fraction=0.0,
    )
    assert conservative_plan["recommended_mode"] == "batched_score"
    assert conservative_plan["recommendation_reason"] == (
        "batched_preserves_parameter_signatures"
    )
    grouped_mixed_refit = native.aom_refit_candidates(
        X,
        y,
        mixed_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="grouped_score",
    )
    batched_mixed_refit = native.aom_refit_candidates(
        X,
        y,
        mixed_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="batched_score",
    )
    union_mixed_refit = native.aom_refit_candidates(
        X,
        y,
        mixed_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="union_batched_score",
    )
    auto_mixed_refit = native.aom_refit_candidates(
        X,
        y,
        mixed_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="auto",
    )
    conservative_auto_mixed_refit = native.aom_refit_candidates(
        X,
        y,
        mixed_pls_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="auto",
        auto_max_extra_fraction=0.0,
    )
    assert grouped_mixed_refit["n_refit_groups"] == 3
    assert batched_mixed_refit["n_refit_groups"] == 2
    assert union_mixed_refit["execution_mode"] == "union_batched_score"
    assert union_mixed_refit["n_refit_groups"] == 1
    assert auto_mixed_refit["execution_mode_requested"] == "auto"
    assert auto_mixed_refit["execution_mode"] == "union_batched_score"
    assert auto_mixed_refit["execution_mode_auto_reason"] == (
        "union_reduces_native_groups_within_extra_budget"
    )
    assert conservative_auto_mixed_refit["execution_mode_requested"] == "auto"
    assert conservative_auto_mixed_refit["execution_mode"] == "batched_score"
    assert conservative_auto_mixed_refit["execution_mode_auto_reason"] == (
        "batched_preserves_parameter_signatures"
    )
    assert union_mixed_refit["n_refit_scored_candidates"] == plan[
        "by_mode"
    ]["union_batched_score"]["n_refit_scored_candidates"]
    assert union_mixed_refit["n_refit_extra_scored_candidates"] == plan[
        "by_mode"
    ]["union_batched_score"]["n_refit_extra_scored_candidates"]
    grouped_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in grouped_mixed_refit["rows"]
    }
    union_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in union_mixed_refit["rows"]
    }
    auto_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in auto_mixed_refit["rows"]
    }
    assert grouped_mixed_by_chain_param.keys() == union_mixed_by_chain_param.keys()
    assert grouped_mixed_by_chain_param.keys() == auto_mixed_by_chain_param.keys()
    for key, score in grouped_mixed_by_chain_param.items():
        np.testing.assert_allclose(
            union_mixed_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            auto_mixed_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )

    two_pass = native.aom_chain_screen_refit_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        chain_ordering="prefix",
        backend_cuda_available=True,
        backend_min_cuda_product=1,
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
    )
    assert two_pass["report_schema"] == "n4m.aom_chain_screen_refit_campaign.v1"
    assert two_pass["screen_complete"] is True
    assert two_pass["pls_score_mode"] == "gcv_proxy"
    assert two_pass["refit_pls_score_mode"] == "cv"
    assert two_pass["chain_ordering"] == "prefix"
    assert two_pass["screen"]["chain_ordering"] == "prefix"
    assert set(two_pass["moment_backend_recommendations"]) == {"pls"}
    assert two_pass["moment_backend_recommendations"]["pls"][
        "cuda_available_source"
    ] == "caller_override"
    assert two_pass["screen"]["moment_backend_recommendations"] == two_pass[
        "moment_backend_recommendations"
    ]
    two_pass_plan = native.aom_refit_execution_plan(two_pass["screen"], top_k=2)
    two_pass_expected_mode = two_pass_plan["recommended_mode"]
    two_pass_expected = two_pass_plan["by_mode"][two_pass_expected_mode]
    assert two_pass["refit_execution_requested"] == "auto"
    assert two_pass["refit_execution"] == two_pass_expected_mode
    assert two_pass["refit_execution_auto_reason"] == two_pass_plan[
        "recommendation_reason"
    ]
    assert two_pass["refit"]["execution_mode_requested"] == "auto"
    assert two_pass["refit"]["execution_mode"] == two_pass_expected_mode
    assert two_pass["n_refit_candidates"] == 2
    assert two_pass["n_refit_scored_candidates"] == two_pass_expected[
        "n_refit_scored_candidates"
    ]
    assert two_pass["n_refit_extra_scored_candidates"] == two_pass_expected[
        "n_refit_extra_scored_candidates"
    ]
    assert two_pass["rows"] == two_pass["refit"]["rows"]
    assert two_pass["best_cv"] == two_pass["refit"]["best_cv"]
    assert {row["screen_score_metric"] for row in two_pass["rows"]} == {
        "pls_gcv_proxy_rmse"
    }

    verified_model = NativeAOMFixedCandidateRegressor.from_refit_report(
        verified,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
    ).fit(X, y)
    assert verified_model.head == verified["best_cv"]["head"]
    assert verified_model.param == verified["best_cv"]["param"]
    np.testing.assert_allclose(
        verified_model.selected_cv_rmse_,
        verified["best_cv"]["refit_cv_rmse"],
        rtol=1e-10,
        atol=1e-10,
    )
    diagnostics = verified_model.get_diagnostics()
    assert diagnostics["aom_pls_score_mode"] == 0
    assert diagnostics["n_pls_gcv_proxy_fits"] == 0

    two_pass_model = NativeAOMFixedCandidateRegressor.from_refit_report(
        two_pass,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
    ).fit(X, y)
    np.testing.assert_allclose(
        two_pass_model.selected_cv_rmse_,
        two_pass["best_cv"]["refit_cv_rmse"],
        rtol=1e-10,
        atol=1e-10,
    )

    screen_refit_model = NativeAOMScreenRefitRegressor(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2],
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        chain_ordering="prefix",
        backend_cuda_available=True,
        backend_min_cuda_product=1,
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
    ).fit(X, y)
    assert hasattr(native_sklearn, "NativeAOMScreenRefitRegressor")
    assert screen_refit_model.campaign_report_["report_schema"] == (
        "n4m.aom_chain_screen_refit_campaign.v1"
    )
    assert screen_refit_model.campaign_report_["moment_backend_recommendations"][
        "pls"
    ]["cuda_available_source"] == "caller_override"
    assert screen_refit_model.campaign_report_["backend_min_cuda_product"] == 1
    assert screen_refit_model.screen_report_["pls_score_mode"] == "gcv_proxy"
    assert screen_refit_model.refit_report_["n_candidates"] == 2
    assert screen_refit_model.predict(X).shape == y.shape
    np.testing.assert_allclose(
        screen_refit_model.selected_cv_rmse_,
        screen_refit_model.selected_refit_row_["refit_cv_rmse"],
        rtol=1e-10,
        atol=1e-10,
    )
    screen_refit_diag = screen_refit_model.get_diagnostics()
    assert screen_refit_diag["screen_complete"] is True
    assert screen_refit_diag["chain_ordering"] == "prefix"
    assert screen_refit_diag["n_refit_candidates"] == 2
    assert screen_refit_diag["moment_backend_recommendations"]["pls"][
        "cuda_available_source"
    ] == "caller_override"
    assert screen_refit_diag["backend_min_cuda_product"] == 1
    model_refit_plan = native.aom_refit_execution_plan(
        screen_refit_model.screen_report_,
        top_k=2,
    )
    model_expected_mode = model_refit_plan["recommended_mode"]
    model_expected = model_refit_plan["by_mode"][model_expected_mode]
    assert screen_refit_diag["n_refit_scored_candidates"] == model_expected[
        "n_refit_scored_candidates"
    ]
    assert screen_refit_diag["n_refit_extra_scored_candidates"] == model_expected[
        "n_refit_extra_scored_candidates"
    ]
    assert screen_refit_diag["refit_execution_requested"] == "auto"
    assert screen_refit_diag["refit_execution"] == model_expected_mode
    assert screen_refit_diag["refit_execution_auto_reason"] == model_refit_plan[
        "recommendation_reason"
    ]
    assert screen_refit_diag["n_pls_gcv_proxy_fits"] == len(chains)
    assert screen_refit_diag["n_refit_pls_moment_cv_fits"] > 0
    assert screen_refit_diag["final_n_candidates"] == 1
    assert screen_refit_diag["final_n_pls_gcv_proxy_fits"] == 0
    assert screen_refit_diag["final_n_pls_moment_cv_fits"] == 0
    assert (
        screen_refit_diag["final_n_pls_moment_final_fits"]
        + screen_refit_diag["final_n_pls_materialized_final_fits"]
    ) >= 1
    assert screen_refit_diag["selected_head"] == "pls"

    with pytest.raises(ValueError, match="rank is outside"):
        NativeAOMFixedCandidateRegressor.from_refit_report(verified, rank=99)


def test_native_aom_ridge_score_only_uses_batch_moment_path():
    rng = np.random.default_rng(54)
    X = rng.standard_normal((80, 12))
    y = 0.45 * X[:, 1] - 0.20 * X[:, 5] + 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("finite_difference", [1])],
        [("detrend", [1]), ("finite_difference", [1])],
    ]
    lambdas = [0.01, 0.1]

    res = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )

    assert res["score_only"] == 1.0
    assert res["n_materialized_candidates"] == 0.0
    assert res["n_ridge_operator_moment_candidates"] == res["n_candidates"]
    assert res["n_ridge_moment_cv_fits"] == len(chains) * len(lambdas) * 4
    assert res["n_ridge_moment_eigen_path_preparations"] == len(chains) * 4
    assert (
        res["n_ridge_moment_eigen_path_cv_fits"]
        == len(chains) * len(lambdas) * 4
    )
    assert res["n_ridge_moment_direct_cv_fits"] == 0.0
    assert res["n_ridge_moment_score_batch_calls"] == 1.0
    assert res["n_ridge_moment_score_batch_jobs"] == len(chains) * len(lambdas) * 4
    assert res["n_ridge_moment_final_fits"] == 0.0
    assert np.all(np.isfinite(res["candidate_scores"][:, 4]))
    assert "materialized" not in {
        row["score_route"] for row in native.aom_candidate_table(res)
    }

    campaign = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=len(chains),
        top_k=3,
    )
    assert campaign["n_ridge_moment_cv_fits"] == len(chains) * len(lambdas) * 4
    assert campaign["n_ridge_moment_eigen_path_preparations"] == len(chains) * 4
    assert (
        campaign["n_ridge_moment_eigen_path_cv_fits"]
        == len(chains) * len(lambdas) * 4
    )
    assert campaign["n_ridge_moment_direct_cv_fits"] == 0
    assert campaign["n_ridge_moment_score_batch_calls"] == 1
    assert campaign["n_ridge_moment_score_batch_jobs"] == len(chains) * len(lambdas) * 4
    assert campaign["ridge_cv_fits_per_chain"] == len(lambdas) * 4
    assert campaign["ridge_cv_fits_per_candidate"] == 4
    assert campaign["n_pls_moment_score_batch_calls"] == 0
    assert campaign["n_pls_gcv_proxy_batch_calls"] == 0


def test_native_aom_moment_screen_refit_presets_are_reusable():
    from n4m._impl import aom_facade as aom

    rng = np.random.default_rng(61)
    X = rng.standard_normal((72, 12))
    y = 0.7 * X[:, 0] - 0.25 * X[:, 4] + 0.1 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2))],
        [("savgol_derivative", (7, 2, 1))],
        [("detrend_poly", (1,))],
    ]

    mixed_model = NativeAOMMomentScreenRefitRegressor(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(0.01,),
        pls_components=(1,),
        chain_chunk_size=2,
        top_k=1,
        refit_top_k=1,
        refit_per_head_top_k=1,
        scale_x=False,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=256,
        cuda_pls_many_batched=True,
    ).fit(X, y)
    assert n4m._impl.aom_facade is aom
    assert aom.aom_screen_refit_candidate_pool is native.aom_screen_refit_candidate_pool
    assert aom.aom_refit_candidates is native.aom_refit_candidates
    assert aom.aom_chain_fixed_fit_run is native.aom_chain_fixed_fit_run
    assert aom.build_aom_strict_chain_grid is native.build_aom_strict_chain_grid
    assert aom.iter_aom_strict_chain_grid is native.iter_aom_strict_chain_grid
    assert aom.decode_aom_chains is native.decode_aom_chains
    assert aom.aom_candidate_table is native.aom_candidate_table
    assert aom.aom_evaluate_candidates is native.aom_evaluate_candidates
    assert aom.aom_candidate_operator_summary is native.aom_candidate_operator_summary
    assert aom.aom_candidate_preprocessing_impact is native.aom_candidate_preprocessing_impact
    assert aom.aom_candidate_route_summary is native.aom_candidate_route_summary
    assert aom.aom_candidate_rank_diagnostics is native.aom_candidate_rank_diagnostics
    assert aom.aom_candidate_report_records is native.aom_candidate_report_records
    assert aom.aom_save_candidate_report is native.aom_save_candidate_report
    assert aom.aom_load_candidate_report is native.aom_load_candidate_report
    assert (
        aom.NativeAOMMomentScreenRefitRegressor
        is NativeAOMMomentScreenRefitRegressor
    )
    assert (
        aom.NativeAOMMomentPLSExactScreenRefitRegressor
        is NativeAOMMomentPLSExactScreenRefitRegressor
    )
    inventory = aom.available_methods()
    inventory_names = {row["name"] for row in inventory}
    assert {
        "moment_mixed_screen_refit",
        "moment_pls_screen_refit",
        "moment_pls_exact_screen_refit",
        "moment_ridge_screen_refit",
        "moment_fast_screen_refit_campaign",
        "screen_refit_campaign",
        "aom_screen_refit_candidate_pool",
        "aom_refit_execution_plan",
        "aom_refit_candidates",
        "aom_chain_fixed_fit",
        "fixed_candidate",
        "chain_sweep",
        "build_aom_strict_chain_grid",
        "iter_aom_strict_chain_grid",
        "decode_aom_chains",
        "aom_candidate_table",
        "aom_evaluate_candidates",
        "aom_candidate_operator_summary",
        "aom_candidate_preprocessing_impact",
        "aom_candidate_rank_diagnostics",
        "aom_candidate_report_records",
        "aom_save_candidate_report",
        "aom_load_candidate_report",
        "ridge_blender",
        "operator_pls_stack",
        "robust_hpo",
        "aom_candidate_route_summary",
    }.issubset(inventory_names)
    assert all(row["cpu"] and row["cuda"] for row in inventory)
    assert all("config_options" in row for row in inventory)
    json.dumps(inventory)
    inventory_by_name = {row["name"]: row for row in inventory}
    assert {
        "chains",
        "checkpoint_path",
        "split_head_scoring",
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
        "backend_min_cuda_product",
    }.issubset(inventory_by_name["moment_mixed_screen_refit"]["config_options"])
    assert inventory_by_name["moment_fast_screen_refit_campaign"][
        "entry"
    ] == "aom_moment_screen_refit_campaign"
    assert inventory_by_name["moment_pls_exact_screen_refit"][
        "entry"
    ] == "NativeAOMMomentPLSExactScreenRefitRegressor"
    assert {
        "moment_policy",
        "pls_score_mode",
        "score_only",
    }.issubset(inventory_by_name["chain_sweep"]["config_options"])
    assert inventory_by_name["fixed_candidate"]["config_options"][:3] == (
        "chain",
        "head",
        "param",
    )
    assert {
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
    }.issubset(inventory_by_name["fixed_candidate"]["config_options"])
    assert inventory_by_name["aom_chain_fixed_fit"][
        "entry"
    ] == "aom_chain_fixed_fit_run"
    assert "include_identity" in inventory_by_name["build_aom_strict_chain_grid"][
        "config_options"
    ]
    assert {
        "start",
        "stop",
        "chunk_size",
        "with_ids",
    }.issubset(inventory_by_name["iter_aom_strict_chain_grid"]["config_options"])
    assert inventory_by_name["aom_candidate_table"]["config_options"] == ("sort",)
    assert {
        "execution_mode",
        "return_predictions",
        "cuda_pls_parallel_folds",
        "cuda_pls_many_batched",
    }.issubset(inventory_by_name["aom_refit_candidates"]["config_options"])
    assert inventory_by_name["aom_evaluate_candidates"]["config_options"] == (
        "top_k",
        "sort_by",
        "cv",
        "fold_ids",
        "center_x",
        "scale_x",
        "center_y",
        "scale_y",
        "moment_policy",
        "return_predictions",
    )
    assert inventory_by_name["aom_candidate_operator_summary"]["config_options"] == (
        "score_key",
        "top_k",
    )
    assert inventory_by_name["aom_candidate_preprocessing_impact"][
        "config_options"
    ] == (
        "score_key",
        "top_k",
        "higher_is_better",
    )
    assert "regularizer" in inventory_by_name["ridge_blender"]["config_options"]
    assert "components" in inventory_by_name["operator_pls_stack"]["config_options"]
    assert inventory_by_name["aom_candidate_route_summary"]["config_options"] == ()
    assert inventory_by_name["aom_candidate_rank_diagnostics"]["config_options"] == (
        "screen_score_key",
        "eval_score_key",
        "cutoffs",
    )
    assert inventory_by_name["aom_candidate_report_records"][
        "config_options"
    ] == ("include_predictions",)
    assert inventory_by_name["aom_save_candidate_report"]["config_options"] == (
        "format",
        "include_predictions",
    )
    assert inventory_by_name["aom_load_candidate_report"]["config_options"] == (
        "format",
    )
    assert "cuda_pls_parallel_folds" not in inventory_by_name["aom_pls"][
        "config_options"
    ]
    assert "cuda_pls_many_batched" not in inventory_by_name["aom_pls"][
        "config_options"
    ]
    inventory[0]["name"] = "mutated"
    assert aom.available_methods()[0]["name"] != "mutated"
    assert hasattr(native_sklearn, "NativeAOMMomentScreenRefitRegressor")
    assert aom.aom_moment_screen_refit_campaign is native.aom_moment_screen_refit_campaign
    assert mixed_model.heads == ("ridge", "pls")
    assert mixed_model.pls_score_mode == "gcv_proxy"
    assert mixed_model.split_head_scoring == "auto"
    assert mixed_model.cuda_pls_parallel_folds is True
    assert mixed_model.cuda_pls_min_device_features == 256
    assert mixed_model.cuda_pls_many_batched is True
    assert mixed_model.moment_policy == "force_moments"
    assert mixed_model.predict(X).shape == y.shape
    mixed_diag = mixed_model.get_diagnostics()
    assert mixed_diag["preset"] == "moment_mixed_gcv_proxy_screen_refit"
    assert mixed_diag["split_head_scoring"] == "auto"
    assert mixed_diag["cuda_pls_parallel_folds"] is True
    assert mixed_diag["cuda_pls_min_device_features"] == 256
    assert mixed_diag["cuda_pls_many_batched"] is True
    assert mixed_diag["n_split_head_chunks"] == 2
    assert mixed_diag["n_chunk_score_calls"] == 4
    assert mixed_diag["refit_per_head_top_k"] == 1
    assert mixed_diag["n_refit_global_candidates"] == 1
    assert mixed_diag["n_refit_per_head_candidates"] == 2
    assert mixed_diag["n_refit_candidates"] >= 2
    assert {row["head"] for row in mixed_model.refit_report_["rows"]} == {
        "ridge",
        "pls",
    }
    mixed_pool = native.aom_screen_refit_candidate_pool(
        mixed_model.screen_report_,
        refit_top_k=1,
        refit_per_head_top_k=1,
    )
    assert mixed_pool["report_schema"] == "n4m.aom_screen_refit_candidate_pool.v1"
    assert mixed_pool["n_refit_global_candidates"] == 1
    assert mixed_pool["n_refit_per_head_candidates"] == 2
    assert mixed_pool["n_candidates"] == mixed_diag["n_refit_candidates"]
    assert {row["head"] for row in mixed_pool["rows"]} == {"ridge", "pls"}

    pls_model = NativeAOMMomentPLSScreenRefitRegressor(
        chains=chains,
        cv=4,
        fold_ids=folds,
        pls_components=(1, 2),
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
        scale_x=False,
    ).fit(X, y)
    assert hasattr(native_sklearn, "NativeAOMMomentPLSScreenRefitRegressor")
    assert pls_model.heads == ("pls",)
    assert pls_model.ridge_lambdas == ()
    assert pls_model.pls_score_mode == "gcv_proxy"
    assert pls_model.moment_policy == "force_moments"
    assert pls_model.predict(X).shape == y.shape
    pls_diag = pls_model.get_diagnostics()
    assert pls_diag["preset"] == "moment_pls_gcv_proxy_screen_refit"
    assert pls_diag["selected_head"] == "pls"
    assert pls_diag["n_pls_gcv_proxy_fits"] == len(chains)
    assert pls_diag["final_n_pls_gcv_proxy_fits"] == 0

    pls_exact_model = NativeAOMMomentPLSExactScreenRefitRegressor(
        chains=chains,
        cv=4,
        fold_ids=folds,
        pls_components=(1,),
        chain_chunk_size=2,
        top_k=2,
        refit_top_k=1,
        scale_x=False,
    ).fit(X, y)
    assert hasattr(native_sklearn, "NativeAOMMomentPLSExactScreenRefitRegressor")
    assert pls_exact_model.heads == ("pls",)
    assert pls_exact_model.ridge_lambdas == ()
    assert pls_exact_model.pls_score_mode == "cv"
    assert pls_exact_model.moment_policy == "force_moments"
    assert pls_exact_model.predict(X).shape == y.shape
    pls_exact_diag = pls_exact_model.get_diagnostics()
    assert pls_exact_diag["preset"] == "moment_pls_exact_cv_screen_refit"
    assert pls_exact_diag["selected_head"] == "pls"
    assert pls_exact_diag["pls_score_mode"] == "cv"
    assert pls_exact_diag["refit_pls_score_mode"] == "cv"
    assert pls_exact_diag["n_pls_gcv_proxy_fits"] == 0

    ridge_model = NativeAOMMomentRidgeScreenRefitRegressor(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(0.01, 0.1, 1.0),
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
        scale_x=False,
    ).fit(X, y)
    assert hasattr(native_sklearn, "NativeAOMMomentRidgeScreenRefitRegressor")
    assert ridge_model.heads == ("ridge",)
    assert ridge_model.pls_components == ()
    assert ridge_model.pls_score_mode == "cv"
    assert ridge_model.moment_policy == "force_moments"
    assert ridge_model.predict(X).shape == y.shape
    ridge_diag = ridge_model.get_diagnostics()
    assert ridge_diag["preset"] == "moment_ridge_exact_screen_refit"
    assert ridge_diag["selected_head"] == "ridge"
    assert ridge_diag["n_pls_gcv_proxy_fits"] == 0
    assert ridge_diag["n_operator_moment_candidates"] == ridge_diag["n_candidates"]


def test_aom_moment_screen_refit_campaign_fast_defaults_match_explicit_campaign(tmp_path):
    rng = np.random.default_rng(611)
    X = rng.standard_normal((72, 12))
    y = 0.7 * X[:, 0] - 0.25 * X[:, 4] + 0.1 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2))],
        [("savgol_derivative", (7, 2, 1))],
        [("detrend_poly", (1,))],
    ]
    common = dict(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(0.01,),
        pls_components=(1,),
        chain_chunk_size=2,
        top_k=1,
        refit_top_k=1,
        refit_per_head_top_k=1,
        scale_x=False,
    )

    fast = native.aom_moment_screen_refit_campaign(X, y, **common)
    explicit = native.aom_chain_screen_refit_campaign(
        X,
        y,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        chain_ordering="prefix",
        split_head_scoring="auto",
        refit_execution="auto",
        **common,
    )

    assert fast["campaign_preset"] == "moment_fast_screen_refit"
    assert fast["report_schema"] == "n4m.aom_chain_screen_refit_campaign.v1"
    assert fast["moment_policy"] == "force_moments"
    assert fast["chain_ordering"] == "prefix"
    assert fast["split_head_scoring"] == "auto"
    assert fast["pls_score_mode"] == "gcv_proxy"
    assert fast["n_split_head_chunks"] > 0
    assert fast["screen"]["n_ridge_moment_score_batch_calls"] > 0
    assert fast["screen"]["n_pls_gcv_proxy_batch_calls"] > 0
    assert fast["screen"]["n_pls_moment_score_batch_calls"] == 0
    assert fast["n_refit_candidates"] == len(fast["rows"])
    assert all("refit_cv_rmse" in row for row in fast["rows"])
    assert fast["best_cv"]["refit_cv_rmse"] == min(
        row["refit_cv_rmse"] for row in fast["rows"]
    )

    fast_rows = [
        (row["chain_id"], row["head"], row["param"], row["refit_cv_rmse"])
        for row in fast["rows"]
    ]
    explicit_rows = [
        (row["chain_id"], row["head"], row["param"], row["refit_cv_rmse"])
        for row in explicit["rows"]
    ]
    assert fast_rows == explicit_rows

    from n4m._impl import moment_facade as moment

    fixed = moment.NativeAOMFixedCandidateRegressor.from_refit_report(
        fast["refit"],
        fit_mode="final_only",
        precomputed_cv_rmse=float(fast["best_cv"]["refit_cv_rmse"]),
        scale_x=False,
        moment_policy="force_moments",
    ).fit(X, y)
    assert fixed.predict(X).shape == y.shape
    np.testing.assert_allclose(
        fixed.selected_cv_rmse_,
        fast["best_cv"]["refit_cv_rmse"],
        rtol=1e-12,
        atol=1e-12,
    )

    summary = moment.aom_candidate_operator_summary(fast)
    assert summary["n_candidates"] >= 1
    assert summary["best"]["head"] in {"ridge", "pls"}

    records = moment.aom_candidate_report_records(fast)
    assert records
    assert "chain_json" in records[0]

    out = tmp_path / "moment_aom_candidates.json"
    assert moment.aom_save_candidate_report(out, fast) == str(out)
    loaded = moment.aom_load_candidate_report(out)
    assert loaded
    assert "chain" in loaded[0]
    assert [
        (name, tuple(params)) for name, params in loaded[0]["chain"]
    ] == [
        (name, tuple(params)) for name, params in fast["rows"][0]["chain"]
    ]


def test_aom_campaign_split_head_scoring_preserves_scores_and_enables_split():
    rng = np.random.default_rng(62)
    X = rng.standard_normal((60, 10))
    y = 0.4 * X[:, 0] - 0.3 * X[:, 5] + 0.05 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2))],
        [("savgol_derivative", (7, 2, 1))],
    ]
    ridge_lambdas = (0.01, 0.1)

    base = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=ridge_lambdas,
        pls_components=(1,),
        heads=("ridge", "pls"),
        top_k=20,
        chain_chunk_size=3,
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        split_head_scoring="off",
    )
    split = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=ridge_lambdas,
        pls_components=(1,),
        heads=("ridge", "pls"),
        top_k=20,
        chain_chunk_size=3,
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="gcv_proxy",
        split_head_scoring="auto",
    )

    def keyed_scores(report):
        out = {}
        for row in report["top_candidates"]:
            key = (row["chain_id"], row["head"], row["param"])
            out[key] = row["cv_rmse"]
        return out

    assert base["split_head_scoring"] == "off"
    assert base["n_split_head_chunks"] == 0
    assert base["n_chunk_score_calls"] == 1
    # A single mixed Ridge+PLS call uses none of the batched fast paths.
    assert base["n_ridge_moment_score_batch_calls"] == 0
    assert base["n_ridge_moment_score_batch_jobs"] == 0
    assert base["n_pls_gcv_proxy_batch_calls"] == 0
    assert base["n_pls_gcv_proxy_batch_jobs"] == 0
    assert split["split_head_scoring"] == "auto"
    assert split["n_split_head_chunks"] == 1
    assert split["n_chunk_score_calls"] == 2
    # Splitting unlocks both head-homogeneous batch paths: Ridge moment batch
    # (chains x |ridge_lambdas| x cv jobs) and the PLS GCV-proxy batch.
    assert split["n_ridge_moment_score_batch_calls"] == 1
    assert split["n_ridge_moment_score_batch_jobs"] == (
        len(chains) * len(ridge_lambdas) * 4
    )
    assert split["n_pls_gcv_proxy_batch_calls"] == 1
    assert split["n_pls_gcv_proxy_batch_jobs"] == len(chains)
    assert keyed_scores(split) == pytest.approx(keyed_scores(base))
    assert split["best"]["cv_rmse"] == pytest.approx(base["best"]["cv_rmse"])
    assert split["best"]["head"] == base["best"]["head"]


def test_aom_campaign_split_head_exact_cv_enables_pls_moment_batch():
    rng = np.random.default_rng(64)
    X = rng.standard_normal((64, 10))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 4] + 0.05 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2))],
        [("savgol_derivative", (7, 2, 1))],
    ]
    ridge_lambdas = (0.01, 0.1)
    pls_components = (1, 2)

    def run(split):
        return native.aom_chain_score_campaign(
            X,
            y,
            chains=chains,
            cv=4,
            fold_ids=folds,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=("ridge", "pls"),
            top_k=30,
            chain_chunk_size=3,
            scale_x=False,
            moment_policy="force_moments",
            pls_score_mode="cv",
            split_head_scoring=split,
        )

    base = run("off")
    split = run("auto")

    def keyed_scores(report):
        return {
            (row["chain_id"], row["head"], row["param"]): row["cv_rmse"]
            for row in report["top_candidates"]
        }

    # Exact-CV mixed single call batches nothing.
    assert base["n_split_head_chunks"] == 0
    assert base["n_ridge_moment_score_batch_calls"] == 0
    assert base["n_pls_moment_score_batch_calls"] == 0
    assert base["n_pls_gcv_proxy_batch_calls"] == 0
    # Splitting runs Ridge-only + PLS-only homogeneous calls, so both batch.
    assert split["n_split_head_chunks"] == 1
    assert split["n_chunk_score_calls"] == 2
    assert split["n_ridge_moment_score_batch_calls"] == 1
    assert split["n_ridge_moment_score_batch_jobs"] == (
        len(chains) * len(ridge_lambdas) * 4
    )
    assert split["n_pls_moment_score_batch_calls"] == 1
    assert split["n_pls_moment_score_batch_jobs"] == len(chains) * 4
    assert split["n_pls_moment_score_batch_jobs"] == split["n_pls_moment_cv_fits"]
    # Exact CV must not engage the GCV proxy batch.
    assert split["n_pls_gcv_proxy_batch_calls"] == 0
    assert split["n_pls_gcv_proxy_batch_jobs"] == 0
    # Splitting is score-preserving.
    assert keyed_scores(split) == pytest.approx(keyed_scores(base))
    assert split["best"]["cv_rmse"] == pytest.approx(base["best"]["cv_rmse"])
    assert split["best"]["head"] == base["best"]["head"]


def test_native_aom_screen_refit_defaults_to_split_head_scoring():
    rng = np.random.default_rng(78)
    X = rng.standard_normal((72, 12))
    y = 0.7 * X[:, 0] - 0.25 * X[:, 4] + 0.08 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2))],
        [("savgol_derivative", (7, 2, 1))],
        [("detrend_poly", (1,))],
    ]
    ridge_lambdas = (0.01, 0.1, 1.0)
    common = dict(
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=ridge_lambdas,
        pls_components=(1, 2),
        chain_chunk_size=2,
        top_k=8,
        refit_top_k=4,
        refit_per_head_top_k=4,
        scale_x=False,
        moment_policy="force_moments",
    )

    # The base mixed screen/refit estimator now defaults to split-head scoring.
    assert NativeAOMScreenRefitRegressor().split_head_scoring == "auto"

    default_model = NativeAOMScreenRefitRegressor(**common).fit(X, y)
    off_model = NativeAOMScreenRefitRegressor(
        split_head_scoring="off", **common
    ).fit(X, y)

    assert default_model.split_head_scoring == "auto"
    assert default_model.heads == ("ridge", "pls")

    dd = default_model.get_diagnostics()
    od = off_model.get_diagnostics()

    # Default 'auto' splits each mixed screen chunk and turns on both batched
    # head-homogeneous fast paths (Ridge moment batch + exact-CV PLS moment
    # batch) once per split chunk.
    assert dd["split_head_scoring"] == "auto"
    assert dd["n_split_head_chunks"] >= 1
    assert dd["n_ridge_moment_score_batch_calls"] == dd["n_split_head_chunks"]
    assert dd["n_ridge_moment_score_batch_jobs"] == (
        len(chains) * len(ridge_lambdas) * 4
    )
    assert dd["n_pls_moment_score_batch_calls"] == dd["n_split_head_chunks"]
    assert dd["n_pls_moment_score_batch_jobs"] == len(chains) * 4

    # Explicit 'off' keeps the legacy single mixed call: no batched fast path.
    assert od["split_head_scoring"] == "off"
    assert od["n_split_head_chunks"] == 0
    assert od["n_ridge_moment_score_batch_calls"] == 0
    assert od["n_pls_moment_score_batch_calls"] == 0

    # Splitting is score-preserving end to end: same winner, coef, predictions.
    assert default_model.selected_chain_ == off_model.selected_chain_
    assert default_model.selected_head_ == off_model.selected_head_
    np.testing.assert_allclose(
        default_model.selected_cv_rmse_,
        off_model.selected_cv_rmse_,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        default_model.predict(X),
        off_model.predict(X),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        default_model.coef_,
        off_model.coef_,
        rtol=1e-10,
        atol=1e-10,
    )


def test_cuda_pls_parallel_folds_option_is_score_preserving_on_cpu_path():
    rng = np.random.default_rng(63)
    X = rng.standard_normal((48, 9))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 3] + 0.05 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    base = native.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1, 2),
        heads=("pls",),
        scale_x=False,
        score_only=True,
        cuda_pls_parallel_folds=False,
        cuda_pls_min_device_features=256,
        cuda_pls_many_batched=False,
    )
    requested = native.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1, 2),
        heads=("pls",),
        scale_x=False,
        score_only=True,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=256,
        cuda_pls_many_batched=True,
    )

    np.testing.assert_allclose(
        requested["candidate_scores"],
        base["candidate_scores"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert requested["n_pls_moment_cuda_parallel_fold_batches"] == 0.0
    assert requested["n_pls_moment_cuda_parallel_fold_jobs"] == 0.0
    assert requested["n_pls_moment_cuda_many_batched_batches"] == 0.0
    assert requested["n_pls_moment_cuda_many_batched_jobs"] == 0.0

    chains = [[("identity", ())], [("savgol_smooth", (5, 2))]]
    campaign = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1,),
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=2,
        top_k=2,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=256,
        cuda_pls_many_batched=True,
    )
    assert campaign["cuda_pls_parallel_folds"] is True
    assert campaign["cuda_pls_min_device_features"] == 256
    assert campaign["cuda_pls_many_batched"] is True
    assert campaign["moment_backend_recommendations"]["pls"][
        "cuda_pls_many_batched"
    ] is True
    assert campaign["n_pls_moment_cuda_parallel_fold_batches"] == 0
    assert campaign["n_pls_moment_cuda_parallel_fold_jobs"] == 0
    assert campaign["n_pls_moment_cuda_many_batched_batches"] == 0
    assert campaign["n_pls_moment_cuda_many_batched_jobs"] == 0
    assert campaign["n_pls_moment_score_batch_calls"] == 1
    assert campaign["n_pls_moment_score_batch_jobs"] == len(chains) * 4


def test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides():
    root = Path(__file__).resolve().parents[3]
    cuda_lib = root / "build/cuda-on/cpp/src/libn4m.so"
    if not cuda_lib.exists():
        pytest.skip("requires cuda-on libn4m build")

    code = r"""
import json
import os
import numpy as np
from n4m._impl import native as n4m

plan = n4m.moment_screen_backend_recommendation(
    96,
    16,
    head="pls",
    cuda_available=None,
    min_cuda_product=1,
    cuda_pls_min_device_features=1,
    cuda_pls_many_batched=True,
)
if not plan["loaded_cuda_available"]:
    print(json.dumps({"skip": "CUDA runtime unavailable"}))
    raise SystemExit

rng = np.random.default_rng(20260608)
X = rng.standard_normal((96, 16))
y = 0.7 * X[:, 0] - 0.25 * X[:, 5] + 0.05 * rng.standard_normal(X.shape[0])
folds = np.arange(X.shape[0], dtype=np.int32) % 4

def run_once():
    res = n4m.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1, 2),
        heads=("pls",),
        scale_x=False,
        score_only=True,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        cuda_pls_many_batched=True,
    )
    return {
        "candidate_scores": np.asarray(res["candidate_scores"], dtype=float).tolist(),
        "selected_cv_rmse": float(res["selected_cv_rmse"]),
        "n_pls_moment_cv_fits": int(res["n_pls_moment_cv_fits"]),
        "n_pls_moment_cuda_device_cv_fits": int(
            res["n_pls_moment_cuda_device_cv_fits"]
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            res["n_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            res["n_pls_moment_cuda_parallel_fold_jobs"]
        ),
        "n_pls_moment_cuda_many_batched_batches": int(
            res["n_pls_moment_cuda_many_batched_batches"]
        ),
        "n_pls_moment_cuda_many_batched_jobs": int(
            res["n_pls_moment_cuda_many_batched_jobs"]
        ),
    }

def run_full_once():
    res = n4m.sweep_run(
        X,
        y,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1, 2),
        heads=("pls",),
        scale_x=False,
        score_only=False,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        cuda_pls_many_batched=True,
    )
    return {
        "candidate_scores": np.asarray(res["candidate_scores"], dtype=float).tolist(),
        "oof_predictions": np.asarray(res["oof_predictions"], dtype=float).tolist(),
        "predictions": np.asarray(res["predictions"], dtype=float).tolist(),
        "coefficients": np.asarray(res["coefficients"], dtype=float).tolist(),
        "intercept": np.asarray(res["intercept"], dtype=float).tolist(),
        "selected_cv_rmse": float(res["selected_cv_rmse"]),
        "n_pls_moment_cv_fits": int(res["n_pls_moment_cv_fits"]),
        "n_pls_moment_cuda_device_cv_fits": int(
            res["n_pls_moment_cuda_device_cv_fits"]
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            res["n_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            res["n_pls_moment_cuda_parallel_fold_jobs"]
        ),
        "n_pls_moment_cuda_many_batched_batches": int(
            res["n_pls_moment_cuda_many_batched_batches"]
        ),
        "n_pls_moment_cuda_many_batched_jobs": int(
            res["n_pls_moment_cuda_many_batched_jobs"]
        ),
        "n_pls_moment_final_fits": int(res["n_pls_moment_final_fits"]),
        "n_pls_moment_cuda_device_final_fits": int(
            res["n_pls_moment_cuda_device_final_fits"]
        ),
        "n_pls_moment_host_final_fits": int(
            res["n_pls_moment_host_final_fits"]
        ),
    }

many_first = run_once()
os.environ["N4M_CUDA_PLS_MANY_LEGACY"] = "1"
legacy_override = run_once()
os.environ.pop("N4M_CUDA_PLS_MANY_LEGACY", None)

full_many = run_full_once()
os.environ["N4M_CUDA_PLS_MANY_LEGACY"] = "1"
full_legacy = run_full_once()
os.environ.pop("N4M_CUDA_PLS_MANY_LEGACY", None)

campaign_chains = [
    [("identity", ())],
    [("savgol_smooth", (5, 2))],
    [("finite_difference", (1,))],
    [("savgol_smooth", (5, 2)), ("finite_difference", (1,))],
]

def campaign_signature(report):
    rows = []
    for row in report["top_candidates"]:
        rows.append((
            int(row["chain_id"]),
            str(row["head"]),
            int(round(float(row["param"]))),
            float(row["cv_rmse"]),
        ))
    return sorted(rows)

def summarize_campaign(report):
    return {
        "signature": campaign_signature(report),
        "n_chunks": int(report["n_chunks"]),
        "n_chunk_score_calls": int(report["n_chunk_score_calls"]),
        "n_candidates": int(report["n_candidates"]),
        "n_pls_moment_cv_fits": int(report["n_pls_moment_cv_fits"]),
        "n_pls_moment_cuda_device_cv_fits": int(
            report["n_pls_moment_cuda_device_cv_fits"]
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            report["n_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            report["n_pls_moment_cuda_parallel_fold_jobs"]
        ),
        "n_pls_moment_cuda_many_batched_batches": int(
            report["n_pls_moment_cuda_many_batched_batches"]
        ),
        "n_pls_moment_cuda_many_batched_jobs": int(
            report["n_pls_moment_cuda_many_batched_jobs"]
        ),
        "n_pls_moment_score_batch_calls": int(
            report["n_pls_moment_score_batch_calls"]
        ),
        "n_pls_moment_score_batch_jobs": int(
            report["n_pls_moment_score_batch_jobs"]
        ),
    }

def run_campaign():
    return n4m.aom_chain_score_campaign(
        X,
        y,
        chains=campaign_chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=(1, 2),
        heads=("pls",),
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode="cv",
        chain_chunk_size=2,
        top_k=8,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        cuda_pls_many_batched=True,
    )

campaign_many = summarize_campaign(run_campaign())
os.environ["N4M_CUDA_PLS_MANY_LEGACY"] = "1"
campaign_legacy = summarize_campaign(run_campaign())
os.environ.pop("N4M_CUDA_PLS_MANY_LEGACY", None)

t = np.arange(24, dtype=np.float64)
X_bad = np.ascontiguousarray(
    np.column_stack([t, t + 1.0, 2.0 * t + 3.0]), dtype=np.float64
)
y_bad = 0.5 * X_bad[:, 0] - X_bad[:, 1]
folds_bad = np.arange(X_bad.shape[0], dtype=np.int32) % 2
recovered = n4m.aom_chain_sweep_run(
    X_bad,
    y_bad,
    [["identity"], ["identity"]],
    cv=2,
    fold_ids=folds_bad,
    ridge_lambdas=[],
    pls_components=[1, 2],
    heads=("pls",),
    scale_x=False,
    moment_policy="force_moments",
    score_only=True,
    cuda_pls_min_device_features=1,
    cuda_pls_many_batched=True,
)
single_sweep_recovered = n4m.sweep_run(
    X_bad,
    y_bad,
    cv=2,
    fold_ids=folds_bad,
    ridge_lambdas=[],
    pls_components=[1, 2],
    heads=("pls",),
    scale_x=False,
    score_only=True,
    cuda_pls_parallel_folds=True,
    cuda_pls_min_device_features=1,
)
single_cv_recovered = n4m.pls_cross_validate(
    X_bad,
    y_bad,
    cv=2,
    fold_ids=folds_bad,
    component_grid=[1, 2],
    score_only=True,
    cuda_pls_parallel_folds=True,
    cuda_pls_min_device_features=1,
)

def summarize_single(res):
    return {
        "candidate_scores": np.asarray(res["candidate_scores"], dtype=float).tolist(),
        "n_pls_moment_cv_fits": int(res["n_pls_moment_cv_fits"]),
        "n_pls_moment_host_cv_fits": int(
            res["n_pls_moment_host_cv_fits"]
        ),
        "n_pls_moment_cuda_device_cv_fits": int(
            res["n_pls_moment_cuda_device_cv_fits"]
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            res["n_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            res["n_pls_moment_cuda_parallel_fold_jobs"]
        ),
        "n_pls_materialized_cv_fits": int(res["n_pls_materialized_cv_fits"]),
    }

print(json.dumps({
    "many_first": many_first,
    "legacy_override": legacy_override,
    "full_many": full_many,
    "full_legacy": full_legacy,
    "campaign_many": campaign_many,
    "campaign_legacy": campaign_legacy,
    "recovered": {
        "candidate_scores": np.asarray(
            recovered["candidate_scores"], dtype=float
        ).tolist(),
        "n_pls_moment_cv_fits": int(recovered["n_pls_moment_cv_fits"]),
        "n_pls_moment_host_cv_fits": int(
            recovered["n_pls_moment_host_cv_fits"]
        ),
        "n_pls_moment_cuda_device_cv_fits": int(
            recovered["n_pls_moment_cuda_device_cv_fits"]
        ),
        "n_pls_moment_score_batch_calls": int(
            recovered["n_pls_moment_score_batch_calls"]
        ),
        "n_pls_moment_score_batch_jobs": int(
            recovered["n_pls_moment_score_batch_jobs"]
        ),
        "n_pls_moment_cuda_many_batched_batches": int(
            recovered["n_pls_moment_cuda_many_batched_batches"]
        ),
        "n_pls_moment_cuda_many_batched_jobs": int(
            recovered["n_pls_moment_cuda_many_batched_jobs"]
        ),
    },
    "single_sweep_recovered": summarize_single(single_sweep_recovered),
    "single_cv_recovered": summarize_single(single_cv_recovered),
}))
"""

    env = os.environ.copy()
    env["N4M_LIB_PATH"] = str(cuda_lib)
    env["PYTHONPATH"] = str(root / "bindings/python/src")
    env.pop("N4M_CUDA_PLS_MANY_LEGACY", None)
    env.pop("N4M_CUDA_PLS_MANY_BATCHED", None)
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
    )
    payload = json.loads(out)
    if payload.get("skip"):
        pytest.skip(payload["skip"])

    many_first = payload["many_first"]
    legacy_override = payload["legacy_override"]
    n_cv = many_first["n_pls_moment_cv_fits"]
    assert n_cv == 4
    assert many_first["n_pls_moment_cuda_device_cv_fits"] == n_cv
    assert many_first["n_pls_moment_cuda_parallel_fold_batches"] == 0
    assert many_first["n_pls_moment_cuda_parallel_fold_jobs"] == 0
    assert many_first["n_pls_moment_cuda_many_batched_batches"] == 1
    assert many_first["n_pls_moment_cuda_many_batched_jobs"] == n_cv

    assert legacy_override["n_pls_moment_cuda_device_cv_fits"] == n_cv
    assert legacy_override["n_pls_moment_cuda_parallel_fold_batches"] == 1
    assert legacy_override["n_pls_moment_cuda_parallel_fold_jobs"] == n_cv
    assert legacy_override["n_pls_moment_cuda_many_batched_batches"] == 0
    assert legacy_override["n_pls_moment_cuda_many_batched_jobs"] == 0
    np.testing.assert_allclose(
        legacy_override["candidate_scores"],
        many_first["candidate_scores"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert legacy_override["selected_cv_rmse"] == pytest.approx(
        many_first["selected_cv_rmse"], rel=1e-10, abs=1e-10
    )

    full_many = payload["full_many"]
    full_legacy = payload["full_legacy"]
    n_cv = full_many["n_pls_moment_cv_fits"]
    assert n_cv == 4
    assert full_many["n_pls_moment_cuda_device_cv_fits"] == n_cv
    assert full_many["n_pls_moment_cuda_parallel_fold_batches"] == 0
    assert full_many["n_pls_moment_cuda_parallel_fold_jobs"] == 0
    assert full_many["n_pls_moment_cuda_many_batched_batches"] == 1
    assert full_many["n_pls_moment_cuda_many_batched_jobs"] == n_cv
    assert full_many["n_pls_moment_final_fits"] == 1
    assert full_many["n_pls_moment_cuda_device_final_fits"] == 1
    assert full_many["n_pls_moment_host_final_fits"] == 0

    assert full_legacy["n_pls_moment_cv_fits"] == n_cv
    assert full_legacy["n_pls_moment_cuda_device_cv_fits"] == n_cv
    assert full_legacy["n_pls_moment_cuda_parallel_fold_batches"] == 1
    assert full_legacy["n_pls_moment_cuda_parallel_fold_jobs"] == n_cv
    assert full_legacy["n_pls_moment_cuda_many_batched_batches"] == 0
    assert full_legacy["n_pls_moment_cuda_many_batched_jobs"] == 0
    assert full_legacy["n_pls_moment_final_fits"] == full_many[
        "n_pls_moment_final_fits"
    ]
    assert full_legacy["n_pls_moment_cuda_device_final_fits"] == full_many[
        "n_pls_moment_cuda_device_final_fits"
    ]
    assert full_legacy["n_pls_moment_host_final_fits"] == full_many[
        "n_pls_moment_host_final_fits"
    ]
    for key in (
        "candidate_scores",
        "oof_predictions",
        "predictions",
        "coefficients",
        "intercept",
    ):
        np.testing.assert_allclose(
            full_legacy[key],
            full_many[key],
            rtol=1e-10,
            atol=1e-10,
        )
    assert full_legacy["selected_cv_rmse"] == pytest.approx(
        full_many["selected_cv_rmse"], rel=1e-10, abs=1e-10
    )

    campaign_many = payload["campaign_many"]
    campaign_legacy = payload["campaign_legacy"]
    assert campaign_many["n_chunks"] == 2
    assert campaign_many["n_chunk_score_calls"] == 2
    assert campaign_many["n_candidates"] == 8
    assert campaign_many["n_pls_moment_score_batch_calls"] == 2
    assert campaign_many["n_pls_moment_score_batch_jobs"] == 16
    assert campaign_many["n_pls_moment_cv_fits"] == 16
    assert campaign_many["n_pls_moment_cuda_device_cv_fits"] == 16
    assert campaign_many["n_pls_moment_cuda_parallel_fold_batches"] == 0
    assert campaign_many["n_pls_moment_cuda_parallel_fold_jobs"] == 0
    assert campaign_many["n_pls_moment_cuda_many_batched_batches"] == 2
    assert campaign_many["n_pls_moment_cuda_many_batched_jobs"] == 16

    assert campaign_legacy["n_chunks"] == campaign_many["n_chunks"]
    assert campaign_legacy["n_chunk_score_calls"] == campaign_many[
        "n_chunk_score_calls"
    ]
    assert campaign_legacy["n_candidates"] == campaign_many["n_candidates"]
    assert campaign_legacy["n_pls_moment_score_batch_calls"] == campaign_many[
        "n_pls_moment_score_batch_calls"
    ]
    assert campaign_legacy["n_pls_moment_score_batch_jobs"] == campaign_many[
        "n_pls_moment_score_batch_jobs"
    ]
    assert campaign_legacy["n_pls_moment_cuda_device_cv_fits"] == campaign_many[
        "n_pls_moment_cuda_device_cv_fits"
    ]
    assert campaign_legacy["n_pls_moment_cuda_parallel_fold_batches"] == 2
    assert campaign_legacy["n_pls_moment_cuda_parallel_fold_jobs"] == 16
    assert campaign_legacy["n_pls_moment_cuda_many_batched_batches"] == 0
    assert campaign_legacy["n_pls_moment_cuda_many_batched_jobs"] == 0
    np.testing.assert_allclose(
        [row[3] for row in campaign_legacy["signature"]],
        [row[3] for row in campaign_many["signature"]],
        rtol=1e-10,
        atol=1e-10,
    )
    assert [row[:3] for row in campaign_legacy["signature"]] == [
        row[:3] for row in campaign_many["signature"]
    ]

    recovered = payload["recovered"]
    recovered_scores = np.asarray(recovered["candidate_scores"], dtype=float)
    assert np.all(np.isfinite(recovered_scores[recovered_scores[:, 3] == 1.0, 4]))
    assert np.all(np.isinf(recovered_scores[recovered_scores[:, 3] == 2.0, 4]))
    assert recovered["n_pls_moment_score_batch_calls"] == 1
    assert recovered["n_pls_moment_score_batch_jobs"] == 4
    assert recovered["n_pls_moment_cuda_many_batched_batches"] == 1
    assert recovered["n_pls_moment_cuda_many_batched_jobs"] == 4
    assert recovered["n_pls_moment_cuda_device_cv_fits"] == 4
    assert recovered["n_pls_moment_host_cv_fits"] == 2
    assert recovered["n_pls_moment_cv_fits"] == 6

    single_sweep = payload["single_sweep_recovered"]
    single_cv = payload["single_cv_recovered"]
    for row in (single_sweep, single_cv):
        scores = np.asarray(row["candidate_scores"], dtype=float)
        assert np.isfinite(scores[0, 3])
        assert np.isinf(scores[1, 3])
        assert row["n_pls_moment_cv_fits"] == 3
        assert row["n_pls_moment_host_cv_fits"] == 1
        assert row["n_pls_moment_cuda_device_cv_fits"] == 2
        assert row["n_pls_moment_cuda_parallel_fold_batches"] == 1
        assert row["n_pls_moment_cuda_parallel_fold_jobs"] == 2
        assert row["n_pls_materialized_cv_fits"] == 0
    np.testing.assert_allclose(
        single_cv["candidate_scores"],
        single_sweep["candidate_scores"],
        rtol=1e-8,
        atol=1e-6,
    )


def test_native_aom_chain_sweep_moment_prefix_cache_counters():
    rng = np.random.default_rng(55)
    X = rng.standard_normal((140, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("detrend", [1])],
        [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
        [("detrend", [1]), ("finite_difference", [1])],
        [("savgol_smooth", [5, 2])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]

    res = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )

    assert res["n_operator_moment_candidates"] == res["n_candidates"]
    assert res["n_materialized_candidates"] == 0.0
    assert res["n_moment_prefix_cache_hits"] >= 3.0
    assert res["n_moment_prefix_cache_misses"] >= 5.0
    campaign = native.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        chain_chunk_size=5,
        top_k=2,
    )
    assert campaign["n_moment_prefix_cache_hits"] >= 3
    assert campaign["moment_prefix_cache_hit_fraction"] > 0.0
    assert campaign["n_pls_moment_cv_fits"] == 20
    assert campaign["n_pls_moment_final_fits"] == 0
    assert campaign["pls_cv_fits_per_chain"] == 4.0
    assert campaign["best"]["score_route"] in {
        "banded_operator_moment",
        "structured_operator_moment",
        "dense_operator_moment",
    }
    summary = native.aom_candidate_operator_summary(campaign)
    assert summary["by_score_route"]


def test_native_aom_refit_batched_score_handles_ridge_lambdas_exactly():
    rng = np.random.default_rng(551)
    X = rng.standard_normal((132, 28))
    y = 0.45 * X[:, 0] - 0.2 * X[:, 7] + 0.1 * X[:, 15]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        ["identity"],
        [("savgol_smooth", [5, 2])],
    ]

    screen = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1, 1.0],
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy="force_moments",
        score_only=True,
    )
    rows = native.aom_candidate_table(screen)
    assert {row["param"] for row in rows} == {0.01, 0.1, 1.0}
    assert {row["chain_id"] for row in rows} == {0, 1}

    individual = native.aom_refit_candidates(
        X,
        y,
        rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="individual",
    )
    grouped = native.aom_refit_candidates(
        X,
        y,
        rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="grouped_score",
    )
    batched = native.aom_refit_candidates(
        X,
        y,
        rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="batched_score",
    )

    assert individual["execution_mode"] == "individual"
    assert grouped["execution_mode"] == "grouped_score"
    assert batched["execution_mode"] == "batched_score"
    assert individual["n_refit_groups"] == 6
    assert grouped["n_refit_groups"] == 2
    assert batched["n_refit_groups"] == 1
    assert individual["n_operator_moment_candidates"] == 6
    assert grouped["n_operator_moment_candidates"] == 6
    assert batched["n_operator_moment_candidates"] == 6
    individual_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in individual["rows"]
    }
    grouped_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in grouped["rows"]
    }
    batched_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in batched["rows"]
    }
    assert individual_by_chain_param.keys() == grouped_by_chain_param.keys()
    assert individual_by_chain_param.keys() == batched_by_chain_param.keys()
    for key, score in individual_by_chain_param.items():
        np.testing.assert_allclose(
            grouped_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            batched_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )
    assert all(np.isnan(row["train_rmse"]) for row in grouped["rows"])
    assert all(row["oof_rmse"] == row["refit_cv_rmse"] for row in grouped["rows"])
    assert all(np.isnan(row["train_rmse"]) for row in batched["rows"])
    assert all(row["oof_rmse"] == row["refit_cv_rmse"] for row in batched["rows"])

    mixed_rows = [
        row for row in rows
        if (row["chain_id"], row["param"]) in {
            (0, 0.01),
            (1, 0.1),
        }
    ]
    assert len(mixed_rows) == 2
    plan = native.aom_refit_execution_plan(mixed_rows)
    assert plan["by_mode"]["individual"]["n_refit_groups"] == 2
    assert plan["by_mode"]["batched_score"]["n_refit_groups"] == 2
    assert plan["by_mode"]["union_batched_score"]["n_refit_groups"] == 1
    assert plan["by_mode"]["union_batched_score"]["n_refit_scored_candidates"] == 4
    assert plan["by_mode"]["union_batched_score"]["n_refit_extra_scored_candidates"] == 2
    assert plan["recommended_mode"] == "union_batched_score"
    assert plan["recommendation_reason"] == (
        "union_reduces_native_groups_within_extra_budget"
    )
    conservative_plan = native.aom_refit_execution_plan(
        mixed_rows,
        auto_max_extra_fraction=0.0,
    )
    assert conservative_plan["recommended_mode"] == "batched_score"
    assert conservative_plan["recommendation_reason"] == (
        "batched_preserves_parameter_signatures"
    )
    individual_mixed = native.aom_refit_candidates(
        X,
        y,
        mixed_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="individual",
    )
    batched_mixed = native.aom_refit_candidates(
        X,
        y,
        mixed_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="batched_score",
    )
    union_mixed = native.aom_refit_candidates(
        X,
        y,
        mixed_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="union_batched_score",
    )
    auto_mixed = native.aom_refit_candidates(
        X,
        y,
        mixed_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="auto",
    )
    conservative_auto_mixed = native.aom_refit_candidates(
        X,
        y,
        mixed_rows,
        cv=4,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode="auto",
        auto_max_extra_fraction=0.0,
    )
    assert individual_mixed["n_refit_groups"] == 2
    assert batched_mixed["n_refit_groups"] == 2
    assert union_mixed["execution_mode"] == "union_batched_score"
    assert union_mixed["n_refit_groups"] == 1
    assert auto_mixed["execution_mode_requested"] == "auto"
    assert auto_mixed["execution_mode"] == "union_batched_score"
    assert auto_mixed["execution_mode_auto_reason"] == (
        "union_reduces_native_groups_within_extra_budget"
    )
    assert conservative_auto_mixed["execution_mode_requested"] == "auto"
    assert conservative_auto_mixed["execution_mode"] == "batched_score"
    assert conservative_auto_mixed["execution_mode_auto_reason"] == (
        "batched_preserves_parameter_signatures"
    )
    assert union_mixed["n_refit_scored_candidates"] == plan[
        "by_mode"
    ]["union_batched_score"]["n_refit_scored_candidates"]
    assert union_mixed["n_refit_extra_scored_candidates"] == plan[
        "by_mode"
    ]["union_batched_score"]["n_refit_extra_scored_candidates"]
    individual_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in individual_mixed["rows"]
    }
    union_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in union_mixed["rows"]
    }
    auto_mixed_by_chain_param = {
        (row["chain_id"], row["param"]): row["refit_cv_rmse"]
        for row in auto_mixed["rows"]
    }
    assert individual_mixed_by_chain_param.keys() == union_mixed_by_chain_param.keys()
    assert individual_mixed_by_chain_param.keys() == auto_mixed_by_chain_param.keys()
    for key, score in individual_mixed_by_chain_param.items():
        np.testing.assert_allclose(
            union_mixed_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            auto_mixed_by_chain_param[key],
            score,
            rtol=1e-12,
            atol=1e-12,
        )


def test_native_aom_sweep_whittaker_uses_structured_moment_route():
    rng = np.random.default_rng(46)
    X = rng.standard_normal((140, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("whittaker", [100.0])],
        [("whittaker", [1000.0]), ("savgol_smooth", [5, 2])],
    ]

    auto = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1, 2],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="auto",
    )
    materialized = native.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1, 2],
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="materialized",
    )

    assert auto["n_operator_moment_candidates"] == auto["n_candidates"]
    assert auto["n_structured_operator_moment_candidates"] == auto["n_candidates"]
    assert auto["n_banded_operator_moment_candidates"] == 0.0
    assert auto["n_dense_operator_moment_candidates"] == 0.0
    assert auto["n_materialized_candidates"] == 0.0
    assert materialized["n_operator_moment_candidates"] == 0.0
    assert materialized["n_materialized_candidates"] == materialized["n_candidates"]
    _assert_aom_route_partitions(auto)
    _assert_aom_route_partitions(materialized)
    np.testing.assert_allclose(
        auto["candidate_scores"][:, :4],
        materialized["candidate_scores"][:, :4],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        auto["candidate_scores"][:, 4],
        materialized["candidate_scores"][:, 4],
        rtol=1e-7,
        atol=1e-7,
    )


def test_native_aom_ridge_blender_compact_contract():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    res = native.aom_ridge_blender(
        X,
        y,
        profile="compact",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 1.0],
        regularizer=0.01,
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (24, 5)
    assert res["weights"].shape == (1, 24)
    assert res["n_chains"] == 12.0
    assert res["n_candidates"] == 24.0
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["oof_candidate_predictions"].shape == (X.shape[0], 24)
    assert res["candidate_predictions"].shape == (X.shape[0], 24)
    np.testing.assert_array_equal(res["fold_ids"], folds)

    weights = res["weights"].reshape(-1)
    assert np.all(weights >= -1e-12)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["candidate_scores"][:, 4], weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        res["predictions"][:, 0],
        res["candidate_predictions"] @ weights,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        res["oof_predictions"][:, 0],
        res["oof_candidate_predictions"] @ weights,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-8,
        atol=1e-8,
    )

    assert int(res["selected_candidate_id"]) == int(np.argmax(weights))
    assert hasattr(native_sklearn, "NativeAOMRidgeBlenderRegressor")
    model = NativeAOMRidgeBlenderRegressor(
        profile="compact",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 1.0],
        regularizer=0.01,
        scale_x=False,
    ).fit(X, y)
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-8,
        atol=1e-8,
    )
    assert model.coef_.shape == (X.shape[1],)
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_candidates"] == 24
    assert diagnostics["profile_name"] == "compact"
    assert diagnostics["expected_bank_size"] == 12
    # n_candidates=24, cv=4: 24*4 CV fits plus 24 final fits.
    assert diagnostics["n_ridge_blender_cv_fits"] == 96
    assert diagnostics["n_ridge_blender_final_fits"] == 24
    assert diagnostics["n_ridge_blender_fit_calls"] == 120
    assert int(res["n_ridge_blender_cv_fits"]) == 96
    assert int(res["n_ridge_blender_final_fits"]) == 24
    assert int(res["n_ridge_blender_fit_calls"]) == 120


def test_native_aom_ridge_blender_rejects_non_positive_lambda():
    X, y = _aom_dataset()
    try:
        native.aom_ridge_blender(X, y, ridge_lambdas=[0.0])
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive Ridge lambda")


def test_native_aom_ridge_global_selects_operator_and_replays_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    res = native.aom_ridge_global(
        X,
        y,
        operators=operators,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1],
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (len(operators) * 2, 5)
    assert res["n_operators"] == float(len(operators))
    assert res["n_chains"] == float(len(operators))
    assert res["selected_head_id"] == 0.0
    assert res["selected_param"] in {0.01, 0.1}
    assert res["selected_operator_kind"] in {0.0, 15.0, 8.0}
    assert res["ridge_backend"] == "native_aom_chain_sweep"
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_ridge_global_wrapper_replays_and_reports_selection():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1])]

    model = NativeAOMRidgeGlobalRegressor(
        operators=operators,
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.01, 0.1],
        scale_x=False,
    ).fit(X, y)

    assert hasattr(native, "aom_ridge_global")
    assert hasattr(native_sklearn, "NativeAOMRidgeGlobalRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["selected_operator_kind"] in {0, 15}
    assert diagnostics["ridge_backend"] == "native_aom_chain_sweep"


def test_native_aom_ridge_superblock_function_replays_input_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    res = native.aom_ridge_superblock(
        X,
        y,
        operators=operators,
        alpha=0.1,
        cv=4,
        fold_ids=folds,
        block_scaling="none",
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (1, 3)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["operator_kinds"].tolist() == [0, 15, 8]
    assert res["n_operators"] == float(len(operators))
    assert res["n_features_superblock"] == float(len(operators) * X.shape[1])
    assert res["selected_alpha"] == 0.1
    assert res["ridge_backend"] == "native"
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_ridge_superblock_cv_and_wrapper_replay():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1])]

    res = native.aom_ridge_superblock(
        X,
        y,
        operators=operators,
        alphas=[0.01, 0.1, 1.0],
        cv=4,
        fold_ids=folds,
    )
    assert res["candidate_scores"].shape == (3, 3)
    assert res["selected_alpha"] in {0.01, 0.1, 1.0}
    assert res["selected_cv_rmse"] == pytest.approx(
        np.min(res["candidate_scores"][:, 2])
    )

    assert hasattr(native_sklearn, "NativeAOMRidgeSuperblockRegressor")
    model = NativeAOMRidgeSuperblockRegressor(
        operators=operators,
        alphas=[0.01, 0.1, 1.0],
        cv=4,
        fold_ids=folds,
    ).fit(X, y)
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    assert model.selected_alpha_ in {0.01, 0.1, 1.0}
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["n_features_superblock"] == len(operators) * X.shape[1]
    assert diagnostics["ridge_backend"] == "native"


def test_native_aom_ridge_mkl_superblock_learns_weights_and_replays():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = [
        "identity",
        ("finite_difference", [1]),
        ("savgol_smooth", [5, 2]),
        ("savgol_derivative", [5, 2, 1]),
    ]

    res = native.aom_ridge_mkl_superblock(
        X,
        y,
        operators=operators,
        alphas=[0.01, 0.1],
        cv=4,
        fold_ids=folds,
        mkl_top_k=2,
        block_scaling="none",
    )

    weights = res["mkl_weights"].reshape(-1)
    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (2, 3)
    assert res["n_operators"] == float(len(operators))
    assert res["n_mkl_active_operators"] <= 2.0
    assert np.count_nonzero(weights > 0.0) == int(res["n_mkl_active_operators"])
    assert np.sum(weights) == pytest.approx(1.0)
    assert res["mkl_alignment_scores"].shape == (len(operators), 1)
    assert res["fold_mkl_weights"].shape == (2, 4, len(operators))
    assert res["selection_mode"] == "mkl_superblock"
    assert res["mkl_mode"] == "alignment"
    assert res["ridge_backend"] == "native"
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_ridge_mkl_superblock_wrapper_reports_weights():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    model = NativeAOMRidgeMKLSuperblockRegressor(
        operators=operators,
        alphas=[0.01, 0.1],
        cv=4,
        fold_ids=folds,
        mkl_top_k=2,
        block_scaling="none",
    ).fit(X, y)

    assert hasattr(native, "aom_ridge_mkl_superblock")
    assert hasattr(native_sklearn, "NativeAOMRidgeMKLSuperblockRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["n_mkl_active_operators"] <= 2
    assert diagnostics["selected_operator_indices"] == model.selected_operator_indices_.tolist()
    assert sum(diagnostics["mkl_weights"]) == pytest.approx(1.0)
    assert diagnostics["mkl_mode"] == "alignment"
    assert diagnostics["ridge_backend"] == "native"


def test_native_aom_ridge_active_superblock_screens_fold_local_and_replays():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    res = native.aom_ridge_active_superblock(
        X,
        y,
        operators=operators,
        alphas=[0.01, 0.1],
        cv=4,
        fold_ids=folds,
        active_top_m=2,
        block_scaling="none",
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (2, 3)
    assert res["n_operators"] == float(len(operators))
    assert res["n_active_operators"] == 2.0
    assert res["n_features_superblock"] == 2.0 * X.shape[1]
    assert res["selected_alpha"] in {0.01, 0.1}
    assert res["selected_cv_rmse"] == pytest.approx(
        np.min(res["candidate_scores"][:, 2])
    )
    assert res["fold_active_operator_indices"].shape == (2, 4, 2)
    assert np.all(res["fold_active_operator_counts"] == 2)
    assert res["selected_operator_indices"].shape == (2,)
    assert res["selected_operator_indices"][0] == 0
    assert res["selected_operator_kinds"][0] == 0
    assert res["active_score_method"] == "norm"
    assert res["selection_mode"] == "active_superblock"
    assert res["ridge_backend"] == "native"
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_ridge_active_superblock_wrapper_replays_and_reports_active_subset():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    model = NativeAOMRidgeActiveSuperblockRegressor(
        operators=operators,
        alphas=[0.01, 0.1],
        cv=4,
        fold_ids=folds,
        active_top_m=2,
        block_scaling="none",
    ).fit(X, y)

    assert hasattr(native, "aom_ridge_active_superblock")
    assert hasattr(native_sklearn, "NativeAOMRidgeActiveSuperblockRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["n_active_operators"] == 2
    assert diagnostics["selected_operator_indices"][0] == 0
    assert diagnostics["selected_operator_kinds"][0] == 0
    assert diagnostics["active_score_method"] == "norm"
    assert diagnostics["ridge_backend"] == "native"


def test_native_aom_pls_superblock_function_replays_input_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    res = native.aom_pls_superblock(
        X,
        y,
        operators=operators,
        pls_components=[1, 2],
        cv=4,
        fold_ids=folds,
        block_scaling="none",
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (2, 3)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["operator_kinds"].tolist() == [0, 15, 8]
    assert res["n_operators"] == float(len(operators))
    assert res["n_features_superblock"] == float(len(operators) * X.shape[1])
    assert res["n_components"] in {1.0, 2.0}
    assert res["selected_cv_rmse"] == pytest.approx(
        np.min(res["candidate_scores"][:, 2])
    )
    assert res["selection_mode"] == "superblock"
    assert res["pls_backend"] == "native"
    expected_pls_solves = len([1, 2]) * 4 + 1
    assert res["n_pls_moment_cv_fits"] == float(2 * expected_pls_solves)
    assert res["n_pls_moment_final_fits"] == float(expected_pls_solves)
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_pls_superblock_wrapper_replays_and_reports_components():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1])]

    model = NativeAOMPLSSuperblockRegressor(
        operators=operators,
        pls_components=[1, 2],
        cv=4,
        fold_ids=folds,
        block_scaling="none",
    ).fit(X, y)

    assert hasattr(native, "aom_pls_superblock")
    assert hasattr(native_sklearn, "NativeAOMPLSSuperblockRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["n_components"] in {1, 2}
    assert diagnostics["n_features_superblock"] == len(operators) * X.shape[1]
    assert diagnostics["pls_backend"] == "native"
    expected_pls_solves = len([1, 2]) * 4 + 1
    assert diagnostics["n_pls_moment_cv_fits"] == 2 * expected_pls_solves
    assert diagnostics["n_pls_moment_final_fits"] == expected_pls_solves


def test_native_aom_ridge_pls_superblock_function_replays_input_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])]

    res = native.aom_ridge_pls_superblock(
        X,
        y,
        operators=operators,
        pls_components=[1, 2],
        ridge_lambdas=[0.0, 0.1],
        cv=4,
        fold_ids=folds,
        block_scaling="none",
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (4, 4)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["operator_kinds"].tolist() == [0, 15, 8]
    assert res["n_operators"] == float(len(operators))
    assert res["n_features_superblock"] == float(len(operators) * X.shape[1])
    assert res["n_components"] in {1.0, 2.0}
    assert res["ridge_lambda"] in {0.0, 0.1}
    assert res["selected_cv_rmse"] == pytest.approx(
        np.min(res["candidate_scores"][:, 3])
    )
    assert res["selection_mode"] == "ridge_pls_superblock"
    assert res["ridge_pls_backend"] == "native"
    assert res["n_ridge_pls_fit_calls"] == float(4 * 4 + 1)
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_native_aom_ridge_pls_superblock_wrapper_replays_and_reports_grid():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("finite_difference", [1])]

    model = NativeAOMRidgePLSSuperblockRegressor(
        operators=operators,
        pls_components=[1, 2],
        ridge_lambdas=[0.0, 0.1],
        cv=4,
        fold_ids=folds,
        block_scaling="none",
    ).fit(X, y)

    assert hasattr(native, "aom_ridge_pls_superblock")
    assert hasattr(native_sklearn, "NativeAOMRidgePLSSuperblockRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_operators"] == len(operators)
    assert diagnostics["n_components"] in {1, 2}
    assert diagnostics["ridge_lambda"] in {0.0, 0.1}
    assert diagnostics["n_candidates"] == 4
    assert diagnostics["n_features_superblock"] == len(operators) * X.shape[1]
    assert diagnostics["ridge_pls_backend"] == "native"
    assert diagnostics["n_ridge_pls_fit_calls"] == 4 * 4 + 1


def test_native_aom_chain_ridge_pls_function_replays_input_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2)), ("finite_difference", (1,))],
    ]

    res = native.aom_chain_ridge_pls(
        X,
        y,
        chains=chains,
        pls_components=[1, 2],
        ridge_lambdas=[0.0, 0.1],
        cv=4,
        fold_ids=folds,
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["candidate_scores"].shape == (8, 5)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["chain_transform_matrix"].shape == (X.shape[1], X.shape[1])
    assert res["n_chains"] == 2.0
    assert res["n_operators"] in {1.0, 2.0}
    assert res["n_features_transformed"] == float(X.shape[1])
    assert res["n_components"] in {1.0, 2.0}
    assert res["ridge_lambda"] in {0.0, 0.1}
    assert res["selected_cv_rmse"] == pytest.approx(
        np.min(res["candidate_scores"][:, 4])
    )
    assert res["selection_mode"] == "chain_ridge_pls"
    assert res["ridge_pls_backend"] == "native"
    assert res["n_ridge_pls_fit_calls"] == float(8 * 4 + 1)
    np.testing.assert_array_equal(res["fold_ids"], folds)
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-8,
        atol=1e-8,
    )

    transformed_only = native.aom_chain_ridge_pls(
        X,
        y,
        chains=[chains[1]],
        pls_components=[1],
        ridge_lambdas=[0.0],
        cv=4,
        fold_ids=folds,
    )
    assert transformed_only["n_operators"] == 2.0
    assert not np.allclose(
        transformed_only["chain_transform_matrix"],
        np.eye(X.shape[1]),
    )
    np.testing.assert_allclose(
        X @ transformed_only["input_coefficients"] + transformed_only["intercept"],
        transformed_only["predictions"],
        rtol=1e-8,
        atol=1e-8,
    )


def test_native_aom_chain_ridge_pls_wrapper_replays_and_reports_grid():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    chains = [
        [("identity", ())],
        [("savgol_smooth", (5, 2)), ("finite_difference", (1,))],
    ]

    model = NativeAOMChainRidgePLSRegressor(
        chains=chains,
        pls_components=[1, 2],
        ridge_lambdas=[0.0, 0.1],
        cv=4,
        fold_ids=folds,
    ).fit(X, y)

    assert hasattr(native, "aom_chain_ridge_pls")
    assert hasattr(native_sklearn, "NativeAOMChainRidgePLSRegressor")
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-8,
        atol=1e-8,
    )
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_chains"] == 2
    assert diagnostics["n_components"] in {1, 2}
    assert diagnostics["ridge_lambda"] in {0.0, 0.1}
    assert diagnostics["n_candidates"] == 8
    assert diagnostics["n_features_transformed"] == X.shape[1]
    assert diagnostics["ridge_pls_backend"] == "native"
    assert diagnostics["n_ridge_pls_fit_calls"] == 8 * 4 + 1


def test_native_aom_operator_pls_stack_compact_contract():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    res = native.aom_operator_pls_stack(
        X,
        y,
        profile="compact",
        cv=4,
        fold_ids=folds,
        components=[1, 2],
        alphas=[0.01, 1.0],
        scale_x=False,
    )

    assert res["candidate_scores"].shape == (4, 7)
    assert res["fold_scores"].shape == (4, 4)
    assert res["oof_predictions"].shape == (X.shape[0], 1)
    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["input_intercept"].shape == (1, 1)
    assert res["n_operators"] == 12.0
    assert res["n_specs"] == 4.0
    np.testing.assert_array_equal(res["fold_ids"], folds)

    selected = int(res["selected_spec_id"])
    assert selected == int(np.argmin(res["candidate_scores"][:, 6]))
    features = res["stack_features"]
    coefs = res["coefficients"].reshape(-1)
    intercept = float(res["intercept"][0, 0])
    assert features.shape[0] == X.shape[0]
    assert features.shape[1] == int(res["n_operator_features"])
    assert coefs.shape == (features.shape[1],)
    np.testing.assert_allclose(
        res["predictions"][:, 0],
        features @ coefs + intercept,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["input_intercept"],
        res["predictions"],
        rtol=1e-8,
        atol=1e-8,
    )
    assert res["operator_feature_offsets"][0] == 0
    assert res["operator_feature_offsets"][-1] == features.shape[1]
    assert hasattr(native_sklearn, "NativeAOMOperatorPLSStackRegressor")
    model = NativeAOMOperatorPLSStackRegressor(
        profile="compact",
        cv=4,
        fold_ids=folds,
        components=[1, 2],
        alphas=[0.01, 1.0],
        scale_x=False,
    ).fit(X, y)
    np.testing.assert_allclose(
        model.predict(X),
        model.result_["predictions"].ravel(),
        rtol=1e-8,
        atol=1e-8,
    )
    assert model.coef_.shape == (X.shape[1],)
    diagnostics = model.get_diagnostics()
    assert diagnostics["n_specs"] == 4
    assert diagnostics["profile_name"] == "compact"
    assert diagnostics["expected_bank_size"] == 12
    # n_specs=4, cv=4, n_operators=12: 21 stack fits, 252 PLS fits.
    assert diagnostics["n_pls_stack_cv_fits"] == 240
    assert diagnostics["n_pls_stack_final_fits"] == 12
    assert diagnostics["n_ridge_stack_cv_fits"] == 20
    assert diagnostics["n_ridge_stack_final_fits"] == 1
    assert diagnostics["n_operator_pls_stack_fit_calls"] == 21
    assert diagnostics["n_operator_pls_stack_pls_fit_calls"] == 252
    assert diagnostics["n_operator_pls_stack_ridge_fit_calls"] == 21
    assert int(res["n_pls_stack_cv_fits"]) == 240
    assert int(res["n_pls_stack_final_fits"]) == 12
    assert int(res["n_ridge_stack_cv_fits"]) == 20
    assert int(res["n_ridge_stack_final_fits"]) == 1
    assert int(res["n_operator_pls_stack_fit_calls"]) == 21
    assert int(res["n_operator_pls_stack_pls_fit_calls"]) == 252
    assert int(res["n_operator_pls_stack_ridge_fit_calls"]) == 21


def test_native_aom_operator_pls_stack_rejects_multi_output_y():
    X, y = _aom_dataset()
    Y = np.column_stack([y, y])
    try:
        native.aom_operator_pls_stack(X, Y, components=[1], alphas=[0.1])
    except ValueError as exc:
        assert "one Y target" in str(exc)
    else:
        raise AssertionError("expected ValueError for multi-output Y")


def test_native_aom_wide_preconfigured_banks_include_fck_variants():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4

    hpo = native.aom_robust_hpo(
        X,
        y,
        profile="wide",
        cv=4,
        heads=("ridge",),
    )
    assert hpo["n_chains"] == 31.0
    assert hpo["n_candidates"] == 31.0 * 4.0
    assert hpo["candidate_scores"].shape == (124, 4)

    blender = native.aom_ridge_blender(
        X,
        y,
        profile="wide",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        regularizer=0.01,
        scale_x=False,
    )
    assert blender["n_chains"] == 31.0
    assert blender["n_candidates"] == 31.0
    assert blender["weights"].shape == (1, 31)
    blender_model = NativeAOMRidgeBlenderRegressor(
        profile="wide",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        regularizer=0.01,
        scale_x=False,
    ).fit(X, y)
    blender_diag = blender_model.get_diagnostics()
    assert blender_diag["profile_name"] == "wide"
    assert blender_diag["expected_bank_size"] == 31
    assert blender_diag["n_chains"] == 31

    stack = native.aom_operator_pls_stack(
        X,
        y,
        profile="wide",
        cv=4,
        fold_ids=folds,
        components=[1],
        alphas=[0.1],
        scale_x=False,
    )
    assert stack["n_operators"] == 31.0
    assert stack["operator_feature_offsets"].shape == (32,)
    assert stack["operator_feature_offsets"][0] == 0
    assert stack["operator_feature_offsets"][-1] == int(
        stack["n_operator_features"]
    )


def test_native_aom_robust_hpo_reusable_input_coefficients():
    X, y = _aom_dataset()
    res = native.aom_robust_hpo(
        X,
        y,
        profile="compact",
        cv=4,
        heads=("ridge", "pls"),
    )

    assert res["predictions"].shape == (X.shape[0], 1)
    assert res["coefficients_transformed"].shape[1] == 1
    assert res["input_coefficients"].shape == (X.shape[1], 1)
    assert res["intercept"].shape == (1, 1)
    assert res["n_features"] == float(X.shape[1])
    assert res["n_targets"] == 1.0
    np.testing.assert_allclose(
        X @ res["input_coefficients"] + res["intercept"],
        res["predictions"],
        rtol=1e-8,
        atol=1e-8,
    )


def _assert_native_regressor_replays_training_predictions(model, X, y):
    fitted = model.fit(X, y)
    pred = fitted.predict(X)
    expected = np.asarray(fitted.result_["predictions"], dtype=np.float64)
    if np.asarray(y).ndim == 1 and expected.ndim == 2 and expected.shape[1] == 1:
        expected = expected.ravel()
    np.testing.assert_allclose(pred, expected, rtol=1e-8, atol=1e-8)
    if fitted.coef_.ndim == 2:
        assert fitted.coef_.shape[-1] == X.shape[1]
    else:
        assert fitted.coef_.shape[0] == X.shape[1]
    assert fitted.get_diagnostics()["n_candidates"] >= 1
    return fitted


def test_native_sweep_sklearn_regressor_replays_native_result():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    assert hasattr(native_sklearn, "NativeMomentSweepRegressor")

    fitted = _assert_native_regressor_replays_training_predictions(
        NativeMomentSweepRegressor(
            cv=4,
            fold_ids=folds,
            ridge_lambdas=(0.1, 1.0),
            pls_components=(1, 2),
            heads=("ridge", "pls"),
            scale_x=False,
        ),
        X,
        y,
    )
    diagnostics = fitted.get_diagnostics()
    assert (
        diagnostics["n_ridge_moment_cv_fits"]
        + diagnostics["n_ridge_dual_materialized_cv_fits"]
        + diagnostics["n_ridge_dual_cross_cv_fits"]
    ) > 0


def test_native_aom_sklearn_regressors_replay_input_space_coefficients():
    rng = np.random.default_rng(48)
    X = rng.standard_normal((140, 24))
    y = 0.6 * X[:, 0] - 0.35 * X[:, 3] + 0.2 * X[:, 7]
    y += 0.03 * rng.standard_normal(X.shape[0])
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    assert hasattr(native_sklearn, "NativeAOMSweepRegressor")
    assert hasattr(native_sklearn, "NativeAOMChainSweepRegressor")
    assert hasattr(native_sklearn, "NativeAOMRobustHPORegressor")

    profile_model = NativeAOMSweepRegressor(
        profile="compact",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(0.1,),
        pls_components=(1, 2),
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="auto",
    )
    _assert_native_regressor_replays_training_predictions(profile_model, X, y)
    assert profile_model.result_["input_coefficients"].shape == (X.shape[1], 1)
    profile_diagnostics = profile_model.get_diagnostics()
    assert profile_diagnostics["profile_name"] == "compact"
    assert profile_diagnostics["expected_bank_size"] == 12

    chain_model = NativeAOMChainSweepRegressor(
        [
            ["identity"],
            [("detrend", [1]), ("finite_difference", [1])],
            [("whittaker", [100.0])],
        ],
        cv=4,
        fold_ids=folds,
        ridge_lambdas=(0.1,),
        pls_components=(1, 2),
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="auto",
    )
    _assert_native_regressor_replays_training_predictions(chain_model, X, y)
    diagnostics = chain_model.get_diagnostics()
    assert diagnostics["n_chains"] == 3
    assert (
        diagnostics["n_operator_moment_candidates"]
        + diagnostics["n_materialized_candidates"]
        == diagnostics["n_candidates"]
    )

    robust_model = NativeAOMRobustHPORegressor(
        profile="compact",
        cv=4,
        heads=("ridge", "pls"),
    )
    _assert_native_regressor_replays_training_predictions(robust_model, X, y)
    assert robust_model.result_["input_coefficients"].shape == (X.shape[1], 1)
    robust_diagnostics = robust_model.get_diagnostics()
    assert robust_diagnostics["n_features"] == X.shape[1]
    assert robust_diagnostics["n_targets"] == 1
    assert robust_diagnostics["selected_head"] in {"ridge", "pls"}
    assert robust_diagnostics["profile_name"] == "compact"
    assert robust_diagnostics["expected_bank_size"] == 12


def test_native_aom_pls_and_pop_pls_sklearn_regressors_replay_coefficients():
    X, y = _aom_dataset()
    folds = np.arange(X.shape[0], dtype=np.int32) % 4
    operators = ["identity", ("savgol_smooth", [5, 2])]

    for model in (
        NativeAOMPLSRegressor(
            max_components=2,
            operators=operators,
            cv=4,
            fold_ids=folds,
            scale_x=False,
        ),
        NativePOPPLSRegressor(
            max_components=2,
            operators=operators,
            cv=4,
            fold_ids=folds,
            scale_x=False,
        ),
    ):
        fitted = model.fit(X, y)
        np.testing.assert_allclose(
            fitted.predict(X),
            fitted.result_["predictions"].ravel(),
            rtol=1e-10,
            atol=1e-10,
        )
        assert fitted.result_["input_coefficients"].shape == (X.shape[1], 1)
        assert fitted.get_diagnostics()["n_operators"] == 2
        assert fitted.selected_n_components_ in {1, 2}

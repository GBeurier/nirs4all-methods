"""Tests for the staged strict-chain cartesian screen/refit orchestration.

``aom_staged_chain_campaign`` chains several score-only strict-linear screens
(compact/wide/lab profiles over Ridge/PLS/mixed heads), keeps the top global and
per-head candidates across stages, exact-CV refits that union once, and attaches
preprocessing-impact and screen-vs-refit rank diagnostics. These tests pin the
contract that matters for the benchmark workflow:

* it runs on tiny synthetic data with no dataset/source/id/name input;
* cross-stage global + per-head retention is honoured;
* the refit rows carry the fields the impact/rank diagnostics need;
* selection is exact-CV on train only and any held-out audit is offline-only.
"""
from __future__ import annotations

import numpy as np
import pytest

import n4m._impl as native_sklearn
from n4m._impl import aom_facade as aom
from n4m._impl import moment_facade as moment
from n4m._impl import native
from n4m.model_selection.aom_campaign import (
    AOMSavgolFocusRegressor as NativeAOMSavgolFocusRegressor,
    AOMStagedChainCampaignRegressor as NativeAOMStagedChainCampaignRegressor,
    AOMStrictFamilyLiteRegressor as NativeAOMStrictFamilyLiteRegressor,
)
from n4m.model_selection.aom_search import (
    AOMFixedCandidateRegressor as NativeAOMFixedCandidateRegressor,
)
from n4m.model_selection.aom_campaign import aom_staged_chain_campaign as _aom_staged_chain_campaign


def _dataset(seed: int = 71, n: int = 32, p: int = 16):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 0.7 * X[:, 0] - 0.4 * X[:, 3] + 0.2 * X[:, 9] + 0.03 * rng.standard_normal(n)
    return X, y


def _folds(n: int, k: int = 4):
    return np.arange(n, dtype=np.int32) % k


def _run(X, y, **overrides):
    folds = _folds(X.shape[0])
    kwargs = dict(
        plan="compact_wide",
        cv=4,
        fold_ids=folds,
        ridge_lambdas=[0.1, 1.0],
        pls_components=[1, 2],
        heads=("ridge", "pls"),
        max_chains=6,
        chain_chunk_size=3,
        top_k=5,
        refit_top_k=4,
        refit_per_head_top_k=2,
        scale_x=False,
    )
    kwargs.update(overrides)
    return _aom_staged_chain_campaign(X, y, **kwargs)


def _screen_chains_by_stage(report):
    return {
        stage["name"]: [
            tuple(str(op[0]) for op in row["chain"])
            for row in screen["top_candidates"]
        ]
        for stage, screen in zip(report["stages"], report["stage_screens"])
    }


def test_staged_campaign_runs_and_reports_stage_structure():
    X, y = _dataset()
    report = _run(X, y, return_stage_screens=True)

    assert report["report_schema"] == "n4m.aom_staged_chain_campaign.v1"
    assert report["plan"] == "compact_wide"
    assert report["n_stages"] == 2
    assert report["complete"] is True

    names = [stage["name"] for stage in report["stages"]]
    assert names == ["compact", "wide"]
    for stage, expected_profile in zip(report["stages"], ("compact", "wide")):
        assert stage["profile"] == expected_profile
        assert stage["heads"] == ("pls", "ridge")
        assert stage["n_chains"] >= 1
        assert stage["n_top_candidates"] >= 1
        assert stage["screen_complete"] is True
        assert stage["n_pls_moment_cv_fits"] >= 0
        assert stage["n_pls_moment_host_cv_fits"] >= 0
        assert stage["n_pls_moment_cuda_device_cv_fits"] >= 0
        assert stage["n_pls_moment_cuda_parallel_fold_jobs"] >= 0
        assert stage["n_pls_moment_score_batch_calls"] >= 0
        assert stage["n_pls_moment_score_batch_jobs"] >= 0

    # Stage screens are streamed candidate reports, not whole-grid materializations.
    assert len(report["stage_screens"]) == 2
    assert all(
        screen["checkpoint_schema"] == "n4m.aom_chain_score_campaign.v1"
        for screen in report["stage_screens"]
    )
    assert all("top_candidates" in screen for screen in report["stage_screens"])
    assert report["n_screen_pls_moment_cv_fits"] == sum(
        stage["n_pls_moment_cv_fits"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_host_cv_fits"] == sum(
        stage["n_pls_moment_host_cv_fits"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_cuda_device_cv_fits"] == sum(
        stage["n_pls_moment_cuda_device_cv_fits"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_cuda_parallel_fold_jobs"] == sum(
        stage["n_pls_moment_cuda_parallel_fold_jobs"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_score_batch_calls"] == sum(
        stage["n_pls_moment_score_batch_calls"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_score_batch_jobs"] == sum(
        stage["n_pls_moment_score_batch_jobs"] for stage in report["stages"]
    )
    assert report["n_refit_pls_moment_cv_fits"] == report["refit"][
        "n_pls_moment_cv_fits"
    ]
    assert report["n_refit_pls_moment_host_cv_fits"] == report["refit"][
        "n_pls_moment_host_cv_fits"
    ]
    assert report["n_refit_pls_moment_cuda_device_cv_fits"] == report["refit"][
        "n_pls_moment_cuda_device_cv_fits"
    ]
    assert report["n_refit_pls_moment_cuda_parallel_fold_jobs"] == report["refit"][
        "n_pls_moment_cuda_parallel_fold_jobs"
    ]
    assert report["n_refit_pls_moment_score_batch_calls"] == report[
        "refit"
    ].get("n_pls_moment_score_batch_calls", 0)
    assert report["n_refit_pls_moment_score_batch_jobs"] == report[
        "refit"
    ].get("n_pls_moment_score_batch_jobs", 0)


def test_staged_campaign_reports_split_head_screen_counters():
    X, y = _dataset(n=64, p=10)
    report = _run(
        X,
        y,
        plan="compact",
        max_chains=4,
        chain_chunk_size=2,
        top_k=6,
        refit_top_k=4,
        split_head_scoring="auto",
        return_stage_screens=True,
    )

    assert report["n_screen_split_head_chunks"] == sum(
        stage["n_split_head_chunks"] for stage in report["stages"]
    )
    assert report["n_screen_chunk_score_calls"] == sum(
        stage["n_chunk_score_calls"] for stage in report["stages"]
    )
    assert report["n_ridge_moment_cv_fits"] == sum(
        stage["n_ridge_moment_cv_fits"] for stage in report["stages"]
    )
    assert report["n_ridge_moment_eigen_path_preparations"] == sum(
        stage["n_ridge_moment_eigen_path_preparations"]
        for stage in report["stages"]
    )
    assert report["n_ridge_moment_eigen_path_cv_fits"] == sum(
        stage["n_ridge_moment_eigen_path_cv_fits"] for stage in report["stages"]
    )
    assert report["n_ridge_moment_direct_cv_fits"] == sum(
        stage["n_ridge_moment_direct_cv_fits"] for stage in report["stages"]
    )
    assert (
        report["n_ridge_moment_eigen_path_cv_fits"]
        + report["n_ridge_moment_direct_cv_fits"]
        == report["n_ridge_moment_cv_fits"]
    )
    assert report["n_ridge_moment_score_batch_calls"] == sum(
        stage["n_ridge_moment_score_batch_calls"] for stage in report["stages"]
    )
    assert report["n_ridge_moment_score_batch_jobs"] == sum(
        stage["n_ridge_moment_score_batch_jobs"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_score_batch_calls"] == sum(
        stage["n_pls_moment_score_batch_calls"] for stage in report["stages"]
    )
    assert report["n_screen_pls_moment_score_batch_jobs"] == sum(
        stage["n_pls_moment_score_batch_jobs"] for stage in report["stages"]
    )
    assert report["n_screen_split_head_chunks"] > 0
    assert report["n_screen_chunk_score_calls"] > report["n_screen_split_head_chunks"]
    assert report["n_screen_pls_moment_score_batch_calls"] == report[
        "n_screen_split_head_chunks"
    ]
    assert report["n_screen_pls_moment_score_batch_jobs"] == (
        report["n_screen_split_head_chunks"] * 2 * 4
    )
    assert {stage["split_head_scoring"] for stage in report["stages"]} == {"auto"}


def test_staged_campaign_savgol_focus_prioritizes_savgol_under_small_budget():
    X, y = _dataset(n=28, p=18)
    report = _run(
        X,
        y,
        plan="savgol_focus",
        heads=("ridge",),
        ridge_lambdas=[0.1],
        pls_components=[1],
        max_chains=3,
        chain_chunk_size=2,
        top_k=12,
        refit_top_k=6,
        refit_per_head_top_k=1,
        return_stage_screens=True,
    )

    assert report["plan"] == "savgol_focus"
    assert [stage["name"] for stage in report["stages"]] == [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "savgol_combinations",
    ]
    assert {stage["profile"] for stage in report["stages"]} <= {"compact", "lab"}
    assert report["selection_uses_test_set"] is False

    chains_by_stage = _screen_chains_by_stage(report)
    assert chains_by_stage["savgol_smooth"]
    assert all(
        chain == ("savgol_smooth",)
        for chain in chains_by_stage["savgol_smooth"]
    )
    assert chains_by_stage["savgol_derivative"]
    assert all(
        chain == ("savgol_derivative",)
        for chain in chains_by_stage["savgol_derivative"]
    )
    assert any(
        "savgol_smooth" in chain or "savgol_derivative" in chain
        for chain in chains_by_stage["savgol_combinations"]
    )


def test_staged_campaign_strict_family_focus_reaches_late_families():
    X, y = _dataset(n=28, p=18)
    report = _run(
        X,
        y,
        plan="strict_family_focus",
        heads=("ridge",),
        ridge_lambdas=[0.1],
        pls_components=[1],
        max_chains=2,
        chain_chunk_size=2,
        top_k=12,
        refit_top_k=8,
        refit_per_head_top_k=1,
        return_stage_screens=True,
    )

    stage_names = [stage["name"] for stage in report["stages"]]
    for expected in (
        "gaussian",
        "fck",
        "whittaker",
        "norris_williams",
        "finite_difference",
    ):
        assert expected in stage_names

    chains_by_stage = _screen_chains_by_stage(report)
    for family in ("gaussian", "fck", "whittaker"):
        assert chains_by_stage[family]
        assert all(chain == (family,) for chain in chains_by_stage[family])

    for stage in report["stages"]:
        assert "dataset" not in stage
        assert "source" not in stage
        assert "dataset_id" not in stage


def test_staged_campaign_global_and_per_head_retention():
    X, y = _dataset()
    report = _run(X, y, refit_top_k=3, refit_per_head_top_k=2)

    retention = report["retention"]
    assert retention["refit_top_k"] == 3
    assert retention["refit_per_head_top_k"] == 2
    assert retention["n_refit_global_candidates"] == 3
    # The exact-CV refit verifies the full retained union.
    assert report["n_refit_candidates"] == retention["n_refit_union_candidates"]
    assert len(report["rows"]) == retention["n_refit_union_candidates"]
    assert retention["n_refit_union_candidates"] >= retention["n_refit_global_candidates"]

    # Per-head retention guarantees both screened heads survive into the refit.
    heads = sorted({row["head"] for row in report["rows"]})
    assert heads == ["pls", "ridge"]
    assert set(report["best_by_head"]) == {"pls", "ridge"}
    for head, row in report["best_by_head"].items():
        assert row["head"] == head
        head_rows = [r["refit_cv_rmse"] for r in report["rows"] if r["head"] == head]
        assert row["refit_cv_rmse"] == pytest.approx(min(head_rows))


def test_staged_campaign_rows_feed_impact_and_rank_diagnostics():
    X, y = _dataset()
    report = _run(X, y)

    # Every refit row must carry the fields the downstream audits consume.
    for row in report["rows"]:
        assert set(row) >= {"chain", "head", "param", "refit_cv_rmse", "screen_cv_rmse"}
        assert isinstance(row["chain"], list) and row["chain"]
        assert row["head"] in {"ridge", "pls"}
        assert np.isfinite(row["refit_cv_rmse"])
        # Cross-stage provenance is attached (labels only, never identity).
        assert row["campaign_stage"] in {"compact", "wide"}
        assert set(row["campaign_stages"]) <= {"compact", "wide"}

    impact = report["impact"]
    assert impact is not None
    assert impact["score_key"] == "refit_cv_rmse"
    assert impact["best_score"] == pytest.approx(report["best"]["refit_cv_rmse"])
    assert any(group["group"] for group in impact["by_operator"])

    rank = report["rank_diagnostics"]
    assert rank is not None
    assert rank["screen_score_key"] == "screen_cv_rmse"
    assert rank["eval_score_key"] == "refit_cv_rmse"
    assert -1.0 <= rank["spearman_rank_correlation"] <= 1.0
    assert rank["n_candidates"] == len(report["rows"])

    # The report is consumable by the existing reusable winner estimator.
    model = NativeAOMFixedCandidateRegressor.from_refit_report(
        report, cv=4, fold_ids=_folds(X.shape[0]), scale_x=False
    ).fit(X, y)
    assert model.head == report["best"]["head"]
    assert model.param == report["best"]["param"]
    assert model.predict(X).shape == y.shape


def test_staged_campaign_sklearn_estimator_selects_train_cv_only():
    X, y = _dataset(n=30)
    folds = _folds(X.shape[0], k=3)

    model = NativeAOMStagedChainCampaignRegressor(
        plan="compact",
        cv=3,
        fold_ids=folds,
        ridge_lambdas=[0.1, 1.0],
        pls_components=[1],
        heads=("ridge", "pls"),
        max_chains=4,
        chain_chunk_size=2,
        top_k=4,
        refit_top_k=3,
        refit_per_head_top_k=1,
        scale_x=False,
    ).fit(X, y)

    pred = model.predict(X)
    assert pred.shape == y.shape
    assert model.staged_report_ is model.campaign_report_
    assert model.refit_report_ is model.campaign_report_["refit"]
    assert model.campaign_report_["audit"] is None
    assert model.campaign_report_["selection_uses_test_set"] is False
    assert model.campaign_report_["selection_policy"] == "exact_cv_refit_train_only"
    assert model.selected_refit_row_["refit_cv_rmse"] == pytest.approx(
        model.selected_cv_rmse_
    )

    diagnostics = model.get_diagnostics()
    assert diagnostics["n_stages"] == 1
    assert diagnostics["screen_complete"] is True
    assert model.split_head_scoring == "auto"
    assert diagnostics["split_head_scoring"] == "auto"
    assert diagnostics["n_screen_split_head_chunks"] > 0
    assert diagnostics["n_screen_chunk_score_calls"] > diagnostics[
        "n_screen_split_head_chunks"
    ]
    assert diagnostics["n_screen_pls_moment_score_batch_calls"] == model.campaign_report_[
        "n_screen_pls_moment_score_batch_calls"
    ]
    assert diagnostics["n_screen_pls_moment_score_batch_jobs"] == model.campaign_report_[
        "n_screen_pls_moment_score_batch_jobs"
    ]
    assert diagnostics["n_refit_pls_moment_score_batch_calls"] == model.campaign_report_[
        "n_refit_pls_moment_score_batch_calls"
    ]
    assert diagnostics["n_refit_pls_moment_score_batch_jobs"] == model.campaign_report_[
        "n_refit_pls_moment_score_batch_jobs"
    ]
    assert diagnostics["selection_uses_test_set"] is False
    assert diagnostics["n_remaining_stage_chunks_total"] == 0
    assert diagnostics["n_screen_candidates_total"] >= diagnostics["n_refit_candidates"]
    assert diagnostics["n_merged_global_candidates"] >= 1
    assert diagnostics["selected_stage"] == "compact"
    assert diagnostics["selected_head"] in {"ridge", "pls"}
    assert diagnostics["selected_refit_cv_rmse"] == pytest.approx(
        model.selected_cv_rmse_
    )


def test_staged_campaign_selects_scale_x_grid_by_train_cv_only():
    X, y = _dataset()
    common = {"split_head_scoring": "auto"}
    grid = _run(X, y, scale_x=None, scale_x_values=[False, True], **common)
    plain = _run(X, y, scale_x=False, **common)
    scaled = _run(X, y, scale_x=True, **common)

    expected = min(
        (plain, False),
        (scaled, True),
        key=lambda item: float(item[0]["best"]["refit_cv_rmse"]),
    )
    expected_report, expected_scale_x = expected

    assert grid["model_config_grid_enabled"] is True
    assert grid["model_config_selection_metric"] == "best_refit_cv_rmse"
    assert grid["scale_x_values"] == [False, True]
    assert grid["selected_model_config"]["scale_x"] is expected_scale_x
    assert grid["scale_x"] is expected_scale_x
    assert grid["best"]["refit_cv_rmse"] == pytest.approx(
        expected_report["best"]["refit_cv_rmse"], rel=1e-12, abs=1e-12
    )
    assert grid["best"]["scale_x"] is expected_scale_x
    assert grid["selection_uses_test_set"] is False
    assert grid["n_screen_candidates_total"] == (
        plain["n_screen_candidates_total"] + scaled["n_screen_candidates_total"]
    )
    assert grid["n_screen_split_head_chunks"] == (
        plain["n_screen_split_head_chunks"] + scaled["n_screen_split_head_chunks"]
    )
    assert grid["n_screen_chunk_score_calls"] == (
        plain["n_screen_chunk_score_calls"] + scaled["n_screen_chunk_score_calls"]
    )
    assert grid["n_ridge_moment_cv_fits"] == (
        plain["n_ridge_moment_cv_fits"] + scaled["n_ridge_moment_cv_fits"]
    )
    assert grid["n_ridge_moment_eigen_path_preparations"] == (
        plain["n_ridge_moment_eigen_path_preparations"]
        + scaled["n_ridge_moment_eigen_path_preparations"]
    )
    assert grid["n_ridge_moment_eigen_path_cv_fits"] == (
        plain["n_ridge_moment_eigen_path_cv_fits"]
        + scaled["n_ridge_moment_eigen_path_cv_fits"]
    )
    assert grid["n_ridge_moment_direct_cv_fits"] == (
        plain["n_ridge_moment_direct_cv_fits"]
        + scaled["n_ridge_moment_direct_cv_fits"]
    )
    assert grid["n_ridge_moment_score_batch_calls"] == (
        plain["n_ridge_moment_score_batch_calls"]
        + scaled["n_ridge_moment_score_batch_calls"]
    )
    assert grid["n_ridge_moment_score_batch_jobs"] == (
        plain["n_ridge_moment_score_batch_jobs"]
        + scaled["n_ridge_moment_score_batch_jobs"]
    )
    assert grid["n_screen_pls_moment_cv_fits"] == (
        plain["n_screen_pls_moment_cv_fits"]
        + scaled["n_screen_pls_moment_cv_fits"]
    )
    assert grid["n_screen_pls_moment_score_batch_calls"] == (
        plain["n_screen_pls_moment_score_batch_calls"]
        + scaled["n_screen_pls_moment_score_batch_calls"]
    )
    assert grid["n_screen_pls_moment_score_batch_jobs"] == (
        plain["n_screen_pls_moment_score_batch_jobs"]
        + scaled["n_screen_pls_moment_score_batch_jobs"]
    )
    assert grid["n_refit_pls_moment_score_batch_calls"] == (
        plain["n_refit_pls_moment_score_batch_calls"]
        + scaled["n_refit_pls_moment_score_batch_calls"]
    )
    assert grid["n_refit_pls_moment_score_batch_jobs"] == (
        plain["n_refit_pls_moment_score_batch_jobs"]
        + scaled["n_refit_pls_moment_score_batch_jobs"]
    )
    assert grid["n_screen_split_head_chunks"] > 0
    assert len(grid["model_config_summaries"]) == 2
    assert all("best_refit_cv_rmse" in row for row in grid["model_config_summaries"])


def test_staged_campaign_sklearn_estimator_uses_selected_scale_x():
    X, y = _dataset(n=30)
    folds = _folds(X.shape[0], k=3)

    model = NativeAOMStagedChainCampaignRegressor(
        plan="compact",
        cv=3,
        fold_ids=folds,
        ridge_lambdas=[0.1, 1.0],
        pls_components=[1],
        heads=("ridge", "pls"),
        max_chains=4,
        chain_chunk_size=2,
        top_k=4,
        refit_top_k=3,
        refit_per_head_top_k=1,
        scale_x_values=[False, True],
    ).fit(X, y)

    assert model.selected_scale_x_ is model.campaign_report_["scale_x"]
    assert model.selected_model_config_ == model.campaign_report_[
        "selected_model_config"
    ]
    diagnostics = model.get_diagnostics()
    assert diagnostics["selected_scale_x"] is model.selected_scale_x_
    assert diagnostics["selected_model_config"] == model.selected_model_config_
    np.testing.assert_allclose(
        model.predict(X),
        model.selected_model_.predict(X),
        rtol=1e-12,
        atol=1e-12,
    )


def test_savgol_focus_preset_is_reusable_train_cv_estimator():
    X, y = _dataset(n=30, p=18)
    folds = _folds(X.shape[0], k=3)

    model = NativeAOMSavgolFocusRegressor(
        cv=3,
        fold_ids=folds,
        heads=("ridge",),
        ridge_lambdas=[0.1],
        pls_components=[1],
        max_chains=2,
        chain_chunk_size=2,
        top_k=4,
        refit_top_k=3,
        refit_per_head_top_k=1,
        return_stage_screens=True,
    ).fit(X, y)

    diagnostics = model.get_diagnostics()
    assert diagnostics["plan"] == "savgol_focus"
    assert model.split_head_scoring == "auto"
    assert diagnostics["split_head_scoring"] == "auto"
    assert diagnostics["selection_uses_test_set"] is False
    assert diagnostics["selected_model_config"]["scale_x"] in {False, True}
    assert diagnostics["selected_scale_x"] in {False, True}
    assert [stage["name"] for stage in diagnostics["stages"]] == [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "savgol_combinations",
    ]
    assert model.selected_model_config_ == model.campaign_report_[
        "selected_model_config"
    ]
    np.testing.assert_allclose(
        model.predict(X),
        model.selected_model_.predict(X),
        rtol=1e-12,
        atol=1e-12,
    )


def test_strict_family_lite_preset_is_reusable_family_audit_estimator():
    X, y = _dataset(n=30, p=18)
    folds = _folds(X.shape[0], k=3)

    model = NativeAOMStrictFamilyLiteRegressor(
        cv=3,
        fold_ids=folds,
        heads=("ridge",),
        ridge_lambdas=[0.1],
        pls_components=[1],
        max_chains=1,
        chain_chunk_size=1,
        top_k=3,
        refit_top_k=3,
        refit_per_head_top_k=1,
        return_stage_screens=True,
    ).fit(X, y)

    diagnostics = model.get_diagnostics()
    assert diagnostics["plan"] == "strict_family_focus"
    assert model.split_head_scoring == "auto"
    assert diagnostics["split_head_scoring"] == "auto"
    assert diagnostics["selection_uses_test_set"] is False
    assert diagnostics["selected_scale_x"] is False
    assert model.scale_x_values is None
    assert [stage["name"] for stage in diagnostics["stages"]] == [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "norris_williams",
        "finite_difference",
        "gaussian",
        "fck",
        "whittaker",
        "strict_combinations",
    ]
    assert diagnostics["n_refit_candidates"] <= 4
    for forbidden in ("dataset", "source", "dataset_id"):
        assert all(forbidden not in stage for stage in diagnostics["stages"])
    np.testing.assert_allclose(
        model.predict(X),
        model.selected_model_.predict(X),
        rtol=1e-12,
        atol=1e-12,
    )


def test_staged_campaign_checkpoint_resume_per_stage(tmp_path):
    X, y = _dataset()
    checkpoint_dir = tmp_path / "staged"

    partial = _run(
        X,
        y,
        checkpoint_dir=checkpoint_dir,
        max_chunks_per_run=1,
        return_stage_screens=True,
    )
    assert partial["screen_complete"] is False
    assert partial["checkpoint_dir"] == str(checkpoint_dir)
    assert partial["resume"] is True
    assert partial["max_chunks_per_run"] == 1
    assert partial["n_remaining_stage_chunks_total"] > 0
    assert len(list(checkpoint_dir.glob("*.json"))) == 2
    for stage in partial["stages"]:
        assert stage["checkpoint_path"]
        assert stage["processed_chunks_this_run"] == 1
        assert stage["n_remaining_chunks"] > 0

    resumed = _run(
        X,
        y,
        checkpoint_dir=checkpoint_dir,
        max_chunks_per_run=1,
        return_stage_screens=True,
    )
    assert resumed["screen_complete"] is True
    assert resumed["n_remaining_stage_chunks_total"] == 0
    assert all(stage["resumed_from_checkpoint"] for stage in resumed["stages"])
    assert all(stage["processed_chunks_this_run"] == 1 for stage in resumed["stages"])
    assert all(screen["complete"] is True for screen in resumed["stage_screens"])

    direct = _run(X, y)
    assert resumed["best"]["chain"] == direct["best"]["chain"]
    assert resumed["best"]["head"] == direct["best"]["head"]
    assert resumed["best"]["param"] == direct["best"]["param"]
    assert resumed["best"]["refit_cv_rmse"] == pytest.approx(
        direct["best"]["refit_cv_rmse"], rel=1e-12, abs=1e-12
    )


def test_staged_campaign_selects_on_train_cv_not_identity_or_names():
    X, y = _dataset()
    report = _run(X, y)

    assert report["selection_metric"] == "refit_cv_rmse"
    assert report["selection_policy"] == "exact_cv_refit_train_only"
    assert report["selection_uses_test_set"] is False
    # Winner is the minimum exact-CV refit row.
    assert report["best"] is report["best_cv"]
    assert report["best"]["refit_cv_rmse"] == pytest.approx(
        min(row["refit_cv_rmse"] for row in report["rows"])
    )

    # Stage labels are cosmetic: renaming the stages cannot change the winner.
    base = _aom_staged_chain_campaign(
        X, y,
        stages=[
            {"name": "alpha", "profile": "compact"},
            {"name": "beta", "profile": "wide"},
        ],
        cv=4, fold_ids=_folds(X.shape[0]),
        ridge_lambdas=[0.1, 1.0], pls_components=[1, 2],
        max_chains=6, chain_chunk_size=3, top_k=5, refit_top_k=4, scale_x=False,
    )
    relabelled = _aom_staged_chain_campaign(
        X, y,
        stages=[
            {"name": "zzz_run", "profile": "compact"},
            {"name": "later", "profile": "wide"},
        ],
        cv=4, fold_ids=_folds(X.shape[0]),
        ridge_lambdas=[0.1, 1.0], pls_components=[1, 2],
        max_chains=6, chain_chunk_size=3, top_k=5, refit_top_k=4, scale_x=False,
    )
    assert relabelled["best"]["chain"] == base["best"]["chain"]
    assert relabelled["best"]["head"] == base["best"]["head"]
    assert relabelled["best"]["param"] == base["best"]["param"]
    assert relabelled["best"]["refit_cv_rmse"] == pytest.approx(
        base["best"]["refit_cv_rmse"], rel=1e-12, abs=1e-12
    )


def test_staged_campaign_audit_is_offline_only_and_does_not_drive_selection():
    X, y = _dataset(seed=12, n=44)
    X_train, y_train = X[:32], y[:32]
    X_audit, y_audit = X[32:], y[32:]

    without_audit = _run(X_train, y_train)
    with_audit = _run(
        X_train, y_train, X_audit=X_audit, y_audit=y_audit, audit_top_k=4
    )

    assert without_audit["audit"] is None
    audit = with_audit["audit"]
    assert audit is not None
    assert audit["audit_only"] is True
    assert "never used" in audit["note"]
    assert audit["n_candidates"] == 4
    assert "eval_rmse" in audit["eval"]["rows"][0]
    assert audit["best_eval"]["head"] in {"ridge", "pls"}
    if audit["rank_diagnostics"] is not None:
        assert audit["rank_diagnostics"]["eval_score_key"] == "eval_rmse"

    # The production winner is identical with or without the offline audit.
    assert with_audit["selection_uses_test_set"] is False
    assert with_audit["best"]["chain"] == without_audit["best"]["chain"]
    assert with_audit["best"]["head"] == without_audit["best"]["head"]
    assert with_audit["best"]["param"] == without_audit["best"]["param"]
    assert with_audit["best"]["refit_cv_rmse"] == pytest.approx(
        without_audit["best"]["refit_cv_rmse"], rel=1e-12, abs=1e-12
    )


def test_staged_campaign_input_validation():
    X, y = _dataset()
    with pytest.raises(ValueError) as plan_error:
        _aom_staged_chain_campaign(X, y, plan="does_not_exist")
    assert "savgol_focus" in str(plan_error.value)
    assert "strict_family_focus" in str(plan_error.value)
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, stages=[{"profile": "compact", "bogus": 1}])
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, stages=[])
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, stages=[123])
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, plan="compact", X_audit=X)
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, refit_top_k=0)
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, max_chunks_per_run=0)
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, scale_x=True, scale_x_values=[False, True])
    with pytest.raises(ValueError):
        _aom_staged_chain_campaign(X, y, scale_x_values=[])


def test_staged_campaign_facade_exports_and_inventory():
    # The single libn4m-backed callable is shared across every surface.
    from n4m.model_selection import aom_campaign as role_campaign

    assert role_campaign.aom_staged_chain_campaign is native.aom_staged_chain_campaign
    assert aom.aom_staged_chain_campaign is native.aom_staged_chain_campaign
    assert moment.aom_staged_chain_campaign is native.aom_staged_chain_campaign
    assert (
        native_sklearn.NativeAOMStagedChainCampaignRegressor
        is NativeAOMStagedChainCampaignRegressor
    )
    assert native_sklearn.NativeAOMSavgolFocusRegressor is NativeAOMSavgolFocusRegressor
    assert (
        native_sklearn.NativeAOMStrictFamilyLiteRegressor
        is NativeAOMStrictFamilyLiteRegressor
    )
    assert role_campaign.AOMStagedChainCampaignRegressor is NativeAOMStagedChainCampaignRegressor
    assert role_campaign.AOMSavgolFocusRegressor is NativeAOMSavgolFocusRegressor
    assert role_campaign.AOMStrictFamilyLiteRegressor is NativeAOMStrictFamilyLiteRegressor
    assert aom.NativeAOMStagedChainCampaignRegressor is NativeAOMStagedChainCampaignRegressor
    assert aom.NativeAOMSavgolFocusRegressor is NativeAOMSavgolFocusRegressor
    assert (
        aom.NativeAOMStrictFamilyLiteRegressor
        is NativeAOMStrictFamilyLiteRegressor
    )
    assert (
        moment.NativeAOMStagedChainCampaignRegressor
        is NativeAOMStagedChainCampaignRegressor
    )
    assert moment.NativeAOMSavgolFocusRegressor is NativeAOMSavgolFocusRegressor
    assert (
        moment.NativeAOMStrictFamilyLiteRegressor
        is NativeAOMStrictFamilyLiteRegressor
    )

    for facade in (aom, moment):
        inventory = {row["name"]: row for row in facade.available_methods()}
        row = inventory["staged_chain_campaign"]
        assert row["entry"] == "aom_staged_chain_campaign"
        assert "wrapper_of" not in row
        assert row["catalog_role"] == "catalog_binding"
        assert row["catalog_id"] == "aom_pop.aom_staged_chain_campaign"
        assert row["doc_path"] == "docs/methods/aom_staged_chain_campaign.md"
        options = row["config_options"]
        assert {
            "plan",
            "stages",
            "refit_top_k",
            "refit_per_head_top_k",
            "checkpoint_dir",
            "resume",
            "max_chunks_per_run",
            "impact",
            "rank_diagnostics",
            "X_audit",
            "y_audit",
        }.issubset(options)

        estimator_row = inventory["staged_chain_campaign_estimator"]
        assert estimator_row["entry"] == "NativeAOMStagedChainCampaignRegressor"
        assert estimator_row["kind"] == "sklearn_estimator"
        assert estimator_row["wrapper_of"] == "aom_staged_chain_campaign"
        assert estimator_row["catalog_role"] == "sklearn_wrapper"
        assert estimator_row["catalog_id"] == "aom_pop.aom_staged_chain_campaign"
        estimator_options = estimator_row["config_options"]
        assert {"plan", "final_rank", "final_moment_policy"}.issubset(
            estimator_options
        )
        assert "X_audit" not in estimator_options
        assert "y_audit" not in estimator_options

        preset_row = inventory["savgol_focus_regressor"]
        assert preset_row["entry"] == "NativeAOMSavgolFocusRegressor"
        assert preset_row["kind"] == "sklearn_estimator"
        assert preset_row["wrapper_of"] == "aom_staged_chain_campaign"
        assert preset_row["catalog_role"] == "preset_sklearn_wrapper"
        assert preset_row["catalog_id"] == "aom_pop.aom_staged_chain_campaign"
        assert preset_row["preset_plan"] == "savgol_focus"
        assert {"max_chains", "scale_x_values", "final_rank"}.issubset(
            preset_row["config_options"]
        )
        assert "plan" not in preset_row["config_options"]
        assert "stages" not in preset_row["config_options"]

        lite_row = inventory["strict_family_lite_regressor"]
        assert lite_row["entry"] == "NativeAOMStrictFamilyLiteRegressor"
        assert lite_row["kind"] == "sklearn_estimator"
        assert lite_row["wrapper_of"] == "aom_staged_chain_campaign"
        assert lite_row["catalog_role"] == "preset_sklearn_wrapper"
        assert lite_row["catalog_id"] == "aom_pop.aom_staged_chain_campaign"
        assert lite_row["preset_plan"] == "strict_family_focus"
        assert {"max_chains", "scale_x_values", "final_rank"}.issubset(
            lite_row["config_options"]
        )
        assert "plan" not in lite_row["config_options"]
        assert "stages" not in lite_row["config_options"]

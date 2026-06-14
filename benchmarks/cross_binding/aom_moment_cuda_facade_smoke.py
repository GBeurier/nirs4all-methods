#!/usr/bin/env python3
"""CUDA smoke for the logical AOM and moment Python facades.

The binding loads one libn4m per Python process, so this script spawns a child
interpreter with ``N4M_LIB_PATH`` pointing at the CUDA build. It then verifies
that ``n4m.moment`` and ``n4m.aom`` are thin aliases over the public runtime and
that a wide PLS1 moment screen uses the CUDA device CV path on one GPU.

This is a correctness/route smoke, not a timing benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_LIB = REPO / "build" / "cuda-on" / "cpp" / "src" / "libn4m.so"
DEFAULT_PYTHONPATH = REPO / "bindings" / "python" / "src"
DEFAULT_OUT = REPO / "benchmarks" / "cross_binding" / "aom_moment_cuda_facade_smoke.json"


CHILD_CODE = r"""
import json
import sys

import numpy as np

import n4m
from n4m._impl import aom_facade as aom
from n4m._impl import moment_facade as moment


def main():
    payload = json.loads(sys.stdin.read())
    n_samples = int(payload["n_samples"])
    n_features = int(payload["n_features"])
    cv = int(payload["cv"])
    seed = int(payload["seed"])

    if n_samples <= cv:
        raise ValueError("n_samples must be greater than cv")
    if n_features < 1024:
        raise ValueError("n_features must be >= 1024 to smoke the CUDA PLS1 device loop")

    # The public flat re-exports were removed in the ABI-2 namespace migration;
    # the AOM / moment surface now lives behind the n4m._impl facades, which the
    # role packages re-export. Verify the facade surface exposes the callables
    # this smoke exercises and that the facades agree with the role packages.
    from n4m.lowlevel.moments import moments as role_moments
    from n4m.model_selection.sweep import sweep_run as role_sweep_run
    from n4m.model_selection.aom_search import (
        aom_pls as role_aom_pls,
        pop_pls as role_pop_pls,
    )

    assert callable(n4m.lowlevel.moments.moments)
    assert moment.moments is role_moments
    assert moment.sweep_run is role_sweep_run
    assert aom.aom_pls is role_aom_pls
    assert aom.pop_pls is role_pop_pls
    for name in (
        "aom_moment_screen_refit_campaign", "aom_staged_chain_campaign",
        "aom_screen_refit_candidate_pool", "aom_refit_candidates",
        "aom_chain_fixed_fit_run", "build_aom_strict_chain_grid",
        "decode_aom_chains", "aom_candidate_table", "aom_evaluate_candidates",
        "aom_candidate_operator_summary", "aom_candidate_route_summary",
        "aom_candidate_rank_diagnostics", "aom_candidate_report_records",
        "aom_save_candidate_report", "aom_load_candidate_report",
        "NativeAOMFixedCandidateRegressor", "NativeAOMMomentScreenRefitRegressor",
        "NativeAOMMomentPLSScreenRefitRegressor",
        "NativeAOMMomentPLSExactScreenRefitRegressor",
        "NativeAOMMomentRidgeScreenRefitRegressor",
        "NativeAOMStagedChainCampaignRegressor", "NativeAOMSavgolFocusRegressor",
        "NativeAOMStrictFamilyLiteRegressor",
    ):
        assert hasattr(moment, name), name
    for name in (
        "aom_preprocess", "aom_chain_sweep_run", "aom_sweep_run",
        "aom_moment_screen_refit_campaign", "aom_staged_chain_campaign",
        "aom_pls", "pop_pls", "aom_ridge_blender", "aom_operator_pls_stack",
        "NativeAOMStagedChainCampaignRegressor",
        "NativeAOMMomentPLSExactScreenRefitRegressor",
        "NativeAOMSavgolFocusRegressor", "NativeAOMStrictFamilyLiteRegressor",
        "NativeAOMPLSRegressor", "NativePOPPLSRegressor",
        "NativeAOMRidgeBlenderRegressor", "NativeAOMOperatorPLSStackRegressor",
    ):
        assert hasattr(aom, name), name

    cuda_available = int(n4m.lib.n4m_backend_is_available(5))
    if cuda_available != 1:
        raise RuntimeError("CUDA backend is not available from the loaded libn4m")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.7 * X[:, 0]
        - 0.25 * X[:, min(7, n_features - 1)]
        + 0.05 * rng.standard_normal(n_samples)
    )
    folds = np.arange(n_samples, dtype=np.int32) % cv
    pre = aom.aom_preprocess(X, y, operators=["identity"], gating_mode="soft")
    if pre["transformed"].shape != X.shape:
        raise RuntimeError("aom.aom_preprocess returned an unexpected transformed shape")
    if pre["operator_kinds"].tolist() != [0]:
        raise RuntimeError("aom.aom_preprocess identity operator kind was not reported")

    moment_res = moment.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
        score_only=True,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
    )
    chains = [[("identity", ())], [("savgol_smooth", (5, 2))]]
    aom_res = aom.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
        score_only=True,
        moment_policy="force_moments",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
    )
    aom_profile_res = aom.aom_sweep_run(
        X,
        y,
        profile="compact",
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
        score_only=True,
        moment_policy="force_moments",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
    )

    moment_device_cv = int(moment_res.get("n_pls_moment_cuda_device_cv_fits", 0))
    aom_device_cv = int(aom_res.get("n_pls_moment_cuda_device_cv_fits", 0))
    aom_profile_device_cv = int(
        aom_profile_res.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    if moment_device_cv <= 0:
        raise RuntimeError("moment.sweep_run did not report CUDA device CV fits")
    if aom_device_cv <= 0:
        raise RuntimeError("aom.aom_chain_sweep_run did not report CUDA device CV fits")
    if aom_profile_device_cv <= 0:
        raise RuntimeError("aom.aom_sweep_run did not report CUDA device CV fits")

    selector_operators = ["identity", ("savgol_smooth", (5, 2))]
    aom_pls_res = aom.aom_pls(
        X,
        y,
        max_components=1,
        operators=selector_operators,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    )
    pop_pls_res = aom.pop_pls(
        X,
        y,
        max_components=1,
        operators=selector_operators,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    )
    aom_pls_model = aom.NativeAOMPLSRegressor(
        max_components=1,
        operators=selector_operators,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    ).fit(X, y)
    pop_pls_model = aom.NativePOPPLSRegressor(
        max_components=1,
        operators=selector_operators,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    ).fit(X, y)
    aom_pls_replay_error = float(
        np.max(
            np.abs(
                aom_pls_model.predict(X)
                - np.asarray(aom_pls_res["predictions"], dtype=np.float64).ravel()
            )
        )
    )
    pop_pls_replay_error = float(
        np.max(
            np.abs(
                pop_pls_model.predict(X)
                - np.asarray(pop_pls_res["predictions"], dtype=np.float64).ravel()
            )
        )
    )
    if aom_pls_replay_error > 1e-10:
        raise RuntimeError("NativeAOMPLSRegressor replay diverged from native result")
    if pop_pls_replay_error > 1e-10:
        raise RuntimeError("NativePOPPLSRegressor replay diverged from native result")

    staged = aom.NativeAOMStagedChainCampaignRegressor(
        plan="compact",
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("pls",),
        max_chains=2,
        chain_chunk_size=2,
        top_k=2,
        refit_top_k=2,
        refit_per_head_top_k=1,
        scale_x=False,
        moment_policy="auto",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        impact=False,
        rank_diagnostics=True,
    ).fit(X, y)
    staged_pred = staged.predict(X)
    if staged_pred.shape != y.shape:
        raise RuntimeError("NativeAOMStagedChainCampaignRegressor predict shape mismatch")
    staged_diag = staged.get_diagnostics()
    if staged_diag["selection_uses_test_set"] is not False:
        raise RuntimeError("staged estimator unexpectedly used test-set selection")
    staged_device_cv = int(
        staged.refit_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    staged_host_cv = int(
        staged.refit_report_.get("n_pls_moment_host_cv_fits", 0)
    )
    if staged_device_cv <= 0:
        raise RuntimeError("staged estimator did not report CUDA device CV fits")
    if staged_host_cv != 0:
        raise RuntimeError("staged estimator unexpectedly used host PLS CV fits")

    pls_exact = moment.NativeAOMMomentPLSExactScreenRefitRegressor(
        chains=[
            [("identity", ())],
            [("savgol_smooth", (5, 2))],
        ],
        cv=cv,
        fold_ids=folds,
        pls_components=[1],
        chain_chunk_size=2,
        top_k=2,
        refit_top_k=1,
        scale_x=False,
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
    ).fit(X, y)
    pls_exact_pred = pls_exact.predict(X)
    if pls_exact_pred.shape != y.shape:
        raise RuntimeError(
            "NativeAOMMomentPLSExactScreenRefitRegressor predict shape mismatch"
        )
    pls_exact_diag = pls_exact.get_diagnostics()
    if pls_exact_diag["preset"] != "moment_pls_exact_cv_screen_refit":
        raise RuntimeError("PLS exact preset reported an unexpected preset id")
    if pls_exact_diag["pls_score_mode"] != "cv":
        raise RuntimeError("PLS exact preset did not run exact-CV screen")
    if pls_exact_diag["refit_pls_score_mode"] != "cv":
        raise RuntimeError("PLS exact preset did not exact-CV refit")
    pls_exact_screen_device_cv = int(
        pls_exact.screen_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    pls_exact_screen_host_cv = int(
        pls_exact.screen_report_.get("n_pls_moment_host_cv_fits", 0)
    )
    pls_exact_refit_device_cv = int(
        pls_exact.refit_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    pls_exact_refit_host_cv = int(
        pls_exact.refit_report_.get("n_pls_moment_host_cv_fits", 0)
    )
    if pls_exact_screen_device_cv <= 0:
        raise RuntimeError("PLS exact preset did not route screen on CUDA")
    if pls_exact_screen_host_cv != 0:
        raise RuntimeError("PLS exact preset unexpectedly used host screen CV")
    if pls_exact_refit_device_cv <= 0:
        raise RuntimeError("PLS exact preset did not route refit on CUDA")
    if pls_exact_refit_host_cv != 0:
        raise RuntimeError("PLS exact preset unexpectedly used host refit CV")

    X_mixed = X[:, : min(128, X.shape[1])]
    staged_mixed_default = aom.NativeAOMStagedChainCampaignRegressor(
        plan="compact",
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("ridge", "pls"),
        max_chains=2,
        chain_chunk_size=2,
        top_k=4,
        refit_top_k=3,
        refit_per_head_top_k=1,
        scale_x=False,
        moment_policy="auto",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        impact=False,
        rank_diagnostics=True,
    ).fit(X_mixed, y)
    staged_mixed_pred = staged_mixed_default.predict(X_mixed)
    if staged_mixed_pred.shape != y.shape:
        raise RuntimeError(
            "mixed NativeAOMStagedChainCampaignRegressor predict shape mismatch"
        )
    mixed_diag = staged_mixed_default.get_diagnostics()
    if mixed_diag["selection_uses_test_set"] is not False:
        raise RuntimeError("mixed staged estimator unexpectedly used test-set selection")
    if mixed_diag["split_head_scoring"] != "auto":
        raise RuntimeError("mixed staged estimator did not default to split-head auto")
    mixed_split_chunks = int(mixed_diag["n_screen_split_head_chunks"])
    mixed_score_calls = int(mixed_diag["n_screen_chunk_score_calls"])
    mixed_screen_pls_device_cv = int(
        mixed_diag.get("n_screen_pls_moment_cuda_device_cv_fits", 0)
    )
    mixed_screen_pls_host_cv = int(
        mixed_diag.get("n_screen_pls_moment_host_cv_fits", 0)
    )
    if mixed_split_chunks <= 0:
        raise RuntimeError("mixed staged estimator did not split mixed chunks")
    if mixed_score_calls <= mixed_split_chunks:
        raise RuntimeError("mixed staged estimator did not report extra split score calls")
    if mixed_screen_pls_device_cv <= 0:
        raise RuntimeError("mixed staged estimator did not route PLS screen on CUDA")
    if mixed_screen_pls_host_cv != 0:
        raise RuntimeError("mixed staged estimator unexpectedly used host PLS screen CV")

    savgol_focus = aom.NativeAOMSavgolFocusRegressor(
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("pls",),
        max_chains=2,
        chain_chunk_size=2,
        top_k=2,
        refit_top_k=2,
        refit_per_head_top_k=1,
        scale_x_values=[False, True],
        impact=False,
        rank_diagnostics=True,
    ).fit(X, y)
    savgol_pred = savgol_focus.predict(X)
    if savgol_pred.shape != y.shape:
        raise RuntimeError("NativeAOMSavgolFocusRegressor predict shape mismatch")
    savgol_diag = savgol_focus.get_diagnostics()
    if savgol_diag["plan"] != "savgol_focus":
        raise RuntimeError("SavGol-focus preset did not run plan=savgol_focus")
    if savgol_diag["selection_uses_test_set"] is not False:
        raise RuntimeError("SavGol-focus preset unexpectedly used test-set selection")
    savgol_stages = [str(stage["name"]) for stage in savgol_diag["stages"]]
    if savgol_stages != [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "savgol_combinations",
    ]:
        raise RuntimeError("SavGol-focus preset emitted unexpected stage names")
    savgol_device_cv = int(
        savgol_focus.refit_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    savgol_host_cv = int(
        savgol_focus.refit_report_.get("n_pls_moment_host_cv_fits", 0)
    )
    if savgol_device_cv <= 0:
        raise RuntimeError("SavGol-focus preset did not report CUDA device CV fits")
    if savgol_host_cv != 0:
        raise RuntimeError("SavGol-focus preset unexpectedly used host PLS CV fits")

    strict_family_lite = aom.NativeAOMStrictFamilyLiteRegressor(
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=[1],
        heads=("pls",),
        max_chains=1,
        chain_chunk_size=1,
        top_k=1,
        refit_top_k=1,
        refit_per_head_top_k=1,
        scale_x=False,
        scale_x_values=None,
        impact=False,
        rank_diagnostics=True,
    ).fit(X, y)
    strict_pred = strict_family_lite.predict(X)
    if strict_pred.shape != y.shape:
        raise RuntimeError("NativeAOMStrictFamilyLiteRegressor predict shape mismatch")
    strict_diag = strict_family_lite.get_diagnostics()
    if strict_diag["plan"] != "strict_family_focus":
        raise RuntimeError("Strict-family lite preset did not run plan=strict_family_focus")
    if strict_diag["selection_uses_test_set"] is not False:
        raise RuntimeError("Strict-family lite preset unexpectedly used test-set selection")
    strict_stages = [str(stage["name"]) for stage in strict_diag["stages"]]
    if strict_stages != [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "norris_williams",
        "finite_difference",
        "gaussian",
        "fck",
        "whittaker",
        "strict_combinations",
    ]:
        raise RuntimeError("Strict-family lite preset emitted unexpected stage names")
    strict_device_cv = int(
        strict_family_lite.refit_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
    )
    strict_host_cv = int(
        strict_family_lite.refit_report_.get("n_pls_moment_host_cv_fits", 0)
    )
    if strict_device_cv <= 0:
        raise RuntimeError("Strict-family lite preset did not report CUDA device CV fits")
    if strict_host_cv != 0:
        raise RuntimeError("Strict-family lite preset unexpectedly used host PLS CV fits")

    print(json.dumps({
        "report_schema": "n4m.aom_moment_cuda_facade_smoke.v1",
        "library_path": n4m.library_path(),
        "abi": ".".join(str(value) for value in n4m.abi_version()),
        "cuda_available": cuda_available,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "facades": {
            "aom_aliases_top_level": True,
            "moment_aliases_top_level": True,
            "n4m_moments_stays_callable": True,
            "aom_moment_screen_refit_aliases_top_level": True,
            "aom_moment_screen_refit_estimators_alias_top_level": True,
            "aom_staged_campaign_aliases_top_level": True,
            "aom_staged_campaign_estimator_aliases_top_level": True,
            "aom_savgol_focus_estimator_aliases_top_level": True,
            "aom_strict_family_lite_estimator_aliases_top_level": True,
            "aom_pls_exact_screen_refit_estimator_aliases_top_level": True,
            "aom_moment_winner_reuse_aliases_top_level": True,
            "aom_moment_audit_report_aliases_top_level": True,
            "aom_moment_route_summary_alias_top_level": True,
            "aom_preprocess_aliases_top_level": True,
            "aom_pls_aliases_top_level": True,
            "pop_pls_aliases_top_level": True,
            "aom_diversity_aliases_top_level": True,
            "aom_diversity_estimators_alias_top_level": True,
        },
        "aom_preprocess": {
            "n_operators": int(pre["n_operators"]),
            "mode": int(pre["mode"]),
            "operator_kinds": [int(value) for value in pre["operator_kinds"]],
        },
        "moment": {
            "selected_cv_rmse": float(moment_res["selected_cv_rmse"]),
            "n_candidates": int(moment_res["n_candidates"]),
            "n_pls_moment_cuda_device_cv_fits": moment_device_cv,
            "n_pls_moment_host_cv_fits": int(moment_res.get("n_pls_moment_host_cv_fits", 0)),
            "n_pls_materialized_cv_fits": int(moment_res.get("n_pls_materialized_cv_fits", 0)),
        },
        "aom": {
            "selected_cv_rmse": float(aom_res["selected_cv_rmse"]),
            "n_candidates": int(aom_res["n_candidates"]),
            "n_pls_moment_cuda_device_cv_fits": aom_device_cv,
            "n_pls_moment_host_cv_fits": int(aom_res.get("n_pls_moment_host_cv_fits", 0)),
            "n_pls_materialized_cv_fits": int(aom_res.get("n_pls_materialized_cv_fits", 0)),
        },
        "aom_profile_sweep": {
            "selected_cv_rmse": float(aom_profile_res["selected_cv_rmse"]),
            "n_candidates": int(aom_profile_res["n_candidates"]),
            "n_pls_moment_cuda_device_cv_fits": aom_profile_device_cv,
            "n_pls_moment_host_cv_fits": int(
                aom_profile_res.get("n_pls_moment_host_cv_fits", 0)
            ),
            "n_pls_materialized_cv_fits": int(
                aom_profile_res.get("n_pls_materialized_cv_fits", 0)
            ),
        },
        "aom_pls": {
            "n_operators": int(aom_pls_res["n_operators"]),
            "selected_n_components": int(aom_pls_res["selected_n_components"]),
            "selected_operator_index": int(aom_pls_res["selected_operator_index"]),
            "prediction_replay_max_abs_error": aom_pls_replay_error,
        },
        "pop_pls": {
            "n_operators": int(pop_pls_res["n_operators"]),
            "selected_n_components": int(pop_pls_res["selected_n_components"]),
            "selected_operator_indices": [
                int(value) for value in pop_pls_res["selected_operator_indices"]
            ],
            "prediction_replay_max_abs_error": pop_pls_replay_error,
        },
        "staged_estimator": {
            "selected_cv_rmse": float(staged.selected_cv_rmse_),
            "selected_head": str(staged.selected_head_),
            "selected_stage": str(staged.selected_stage_),
            "split_head_scoring": str(staged_diag["split_head_scoring"]),
            "n_screen_split_head_chunks": int(
                staged_diag["n_screen_split_head_chunks"]
            ),
            "n_screen_chunk_score_calls": int(
                staged_diag["n_screen_chunk_score_calls"]
            ),
            "n_refit_candidates": int(staged.campaign_report_["n_refit_candidates"]),
            "screen_complete": bool(staged.campaign_report_["screen_complete"]),
            "selection_uses_test_set": bool(
                staged.campaign_report_["selection_uses_test_set"]
            ),
            "n_pls_moment_cuda_device_cv_fits": staged_device_cv,
            "n_pls_moment_host_cv_fits": staged_host_cv,
            "n_pls_moment_cuda_parallel_fold_batches": int(
                staged.refit_report_.get(
                    "n_pls_moment_cuda_parallel_fold_batches", 0
                )
            ),
            "n_pls_moment_cuda_parallel_fold_jobs": int(
                staged.refit_report_.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_pls_moment_cuda_many_batched_batches": int(
                staged.refit_report_.get(
                    "n_pls_moment_cuda_many_batched_batches", 0
                )
            ),
            "n_pls_moment_cuda_many_batched_jobs": int(
                staged.refit_report_.get("n_pls_moment_cuda_many_batched_jobs", 0)
            ),
        },
        "pls_exact_screen_refit_estimator": {
            "selected_cv_rmse": float(pls_exact.selected_cv_rmse_),
            "selected_head": str(pls_exact.selected_head_),
            "selected_param": int(pls_exact.selected_param_),
            "preset": str(pls_exact_diag["preset"]),
            "pls_score_mode": str(pls_exact_diag["pls_score_mode"]),
            "refit_pls_score_mode": str(pls_exact_diag["refit_pls_score_mode"]),
            "screen_complete": bool(pls_exact.campaign_report_["screen_complete"]),
            "selection_uses_test_set": False,
            "n_screen_candidates": int(pls_exact_diag["n_screen_candidates"]),
            "n_refit_candidates": int(pls_exact_diag["n_refit_candidates"]),
            "n_screen_pls_moment_cuda_device_cv_fits": pls_exact_screen_device_cv,
            "n_screen_pls_moment_host_cv_fits": pls_exact_screen_host_cv,
            "n_refit_pls_moment_cuda_device_cv_fits": pls_exact_refit_device_cv,
            "n_refit_pls_moment_host_cv_fits": pls_exact_refit_host_cv,
            "n_pls_gcv_proxy_fits": int(pls_exact_diag["n_pls_gcv_proxy_fits"]),
        },
        "staged_mixed_default_estimator": {
            "selected_cv_rmse": float(staged_mixed_default.selected_cv_rmse_),
            "selected_head": str(staged_mixed_default.selected_head_),
            "selected_stage": str(staged_mixed_default.selected_stage_),
            "split_head_scoring": str(mixed_diag["split_head_scoring"]),
            "n_screen_split_head_chunks": mixed_split_chunks,
            "n_screen_chunk_score_calls": mixed_score_calls,
            "n_screen_pls_moment_cuda_device_cv_fits": mixed_screen_pls_device_cv,
            "n_screen_pls_moment_host_cv_fits": mixed_screen_pls_host_cv,
            "n_ridge_moment_cv_fits": int(
                mixed_diag.get("n_ridge_moment_cv_fits", 0)
            ),
            "n_ridge_moment_score_batch_calls": int(
                mixed_diag.get("n_ridge_moment_score_batch_calls", 0)
            ),
            "n_ridge_moment_score_batch_jobs": int(
                mixed_diag.get("n_ridge_moment_score_batch_jobs", 0)
            ),
            "n_refit_candidates": int(
                staged_mixed_default.campaign_report_["n_refit_candidates"]
            ),
            "screen_complete": bool(
                staged_mixed_default.campaign_report_["screen_complete"]
            ),
            "selection_uses_test_set": bool(
                staged_mixed_default.campaign_report_["selection_uses_test_set"]
            ),
        },
        "savgol_focus_estimator": {
            "selected_cv_rmse": float(savgol_focus.selected_cv_rmse_),
            "selected_head": str(savgol_focus.selected_head_),
            "selected_stage": str(savgol_focus.selected_stage_),
            "selected_scale_x": (
                None
                if savgol_focus.selected_scale_x_ is None
                else bool(savgol_focus.selected_scale_x_)
            ),
            "selected_model_config": savgol_focus.selected_model_config_,
            "n_refit_candidates": int(
                savgol_focus.campaign_report_["n_refit_candidates"]
            ),
            "screen_complete": bool(
                savgol_focus.campaign_report_["screen_complete"]
            ),
            "selection_uses_test_set": bool(
                savgol_focus.campaign_report_["selection_uses_test_set"]
            ),
            "stage_names": savgol_stages,
            "n_screen_pls_moment_cuda_device_cv_fits": int(
                savgol_focus.campaign_report_.get(
                    "n_screen_pls_moment_cuda_device_cv_fits", 0
                )
            ),
            "n_screen_pls_moment_host_cv_fits": int(
                savgol_focus.campaign_report_.get(
                    "n_screen_pls_moment_host_cv_fits", 0
                )
            ),
            "n_pls_moment_cuda_device_cv_fits": savgol_device_cv,
            "n_pls_moment_host_cv_fits": savgol_host_cv,
            "n_pls_moment_cuda_parallel_fold_batches": int(
                savgol_focus.refit_report_.get(
                    "n_pls_moment_cuda_parallel_fold_batches", 0
                )
            ),
            "n_pls_moment_cuda_parallel_fold_jobs": int(
                savgol_focus.refit_report_.get(
                    "n_pls_moment_cuda_parallel_fold_jobs", 0
                )
            ),
            "n_pls_moment_cuda_many_batched_batches": int(
                savgol_focus.refit_report_.get(
                    "n_pls_moment_cuda_many_batched_batches", 0
                )
            ),
            "n_pls_moment_cuda_many_batched_jobs": int(
                savgol_focus.refit_report_.get(
                    "n_pls_moment_cuda_many_batched_jobs", 0
                )
            ),
        },
        "strict_family_lite_estimator": {
            "selected_cv_rmse": float(strict_family_lite.selected_cv_rmse_),
            "selected_head": str(strict_family_lite.selected_head_),
            "selected_stage": str(strict_family_lite.selected_stage_),
            "selected_scale_x": (
                None
                if strict_family_lite.selected_scale_x_ is None
                else bool(strict_family_lite.selected_scale_x_)
            ),
            "selected_model_config": strict_family_lite.selected_model_config_,
            "n_refit_candidates": int(
                strict_family_lite.campaign_report_["n_refit_candidates"]
            ),
            "screen_complete": bool(
                strict_family_lite.campaign_report_["screen_complete"]
            ),
            "selection_uses_test_set": bool(
                strict_family_lite.campaign_report_["selection_uses_test_set"]
            ),
            "stage_names": strict_stages,
            "n_screen_pls_moment_cuda_device_cv_fits": int(
                strict_family_lite.campaign_report_.get(
                    "n_screen_pls_moment_cuda_device_cv_fits", 0
                )
            ),
            "n_screen_pls_moment_host_cv_fits": int(
                strict_family_lite.campaign_report_.get(
                    "n_screen_pls_moment_host_cv_fits", 0
                )
            ),
            "n_pls_moment_cuda_device_cv_fits": strict_device_cv,
            "n_pls_moment_host_cv_fits": strict_host_cv,
            "n_pls_moment_cuda_parallel_fold_batches": int(
                strict_family_lite.refit_report_.get(
                    "n_pls_moment_cuda_parallel_fold_batches", 0
                )
            ),
            "n_pls_moment_cuda_parallel_fold_jobs": int(
                strict_family_lite.refit_report_.get(
                    "n_pls_moment_cuda_parallel_fold_jobs", 0
                )
            ),
            "n_pls_moment_cuda_many_batched_batches": int(
                strict_family_lite.refit_report_.get(
                    "n_pls_moment_cuda_many_batched_batches", 0
                )
            ),
            "n_pls_moment_cuda_many_batched_jobs": int(
                strict_family_lite.refit_report_.get(
                    "n_pls_moment_cuda_many_batched_jobs", 0
                )
            ),
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-lib", default=str(DEFAULT_CUDA_LIB))
    parser.add_argument("--pythonpath", default=str(DEFAULT_PYTHONPATH))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5150)
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env["N4M_LIB_PATH"] = str(Path(args.cuda_lib))
    env["PYTHONPATH"] = str(Path(args.pythonpath))
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    payload = {
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "cv": int(args.cv),
        "seed": int(args.seed),
    }
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(CHILD_CODE)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CUDA facade smoke failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    report = json.loads(proc.stdout)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

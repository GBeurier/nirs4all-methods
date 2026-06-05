#!/usr/bin/env python3
"""Timing smoke benchmark for n4m.aom_sweep_run.

This is a product smoke timing for the catalogued strict-linear AOM sweep. It
does not benchmark the future arbitrary-chain fused GPU grinder.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.8 * X[:, 0]
        - 0.45 * X[:, min(5, n_features - 1)]
        + 0.20 * X[:, min(17, n_features - 1)]
        + 0.04 * rng.standard_normal(n_samples)
    )
    return X, y


def balanced_folds(n_samples: int, cv: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.int32) % int(cv)


def median_ms(fn, repeats: int) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def cuda_opts(
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return {
        "cuda_pls_parallel_folds": bool(cuda_pls_parallel_folds),
        "cuda_pls_min_device_features": cuda_pls_min_device_features,
        "cuda_pls_many_batched": bool(cuda_pls_many_batched),
    }


def run_aom_sweep(
    X,
    y,
    folds,
    cv,
    profile,
    lambdas,
    components,
    moment_policy,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return n4m.aom_sweep_run(
        X,
        y,
        profile=profile,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy=moment_policy,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_sweep_ridge(
    X,
    y,
    folds,
    cv,
    profile,
    lambdas,
    moment_policy,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return n4m.aom_sweep_run(
        X,
        y,
        profile=profile,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy=moment_policy,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_sweep_pls(
    X,
    y,
    folds,
    cv,
    profile,
    components,
    moment_policy,
    pls_score_mode="cv",
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return n4m.aom_sweep_run(
        X,
        y,
        profile=profile,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=components,
        heads=("pls",),
        scale_x=False,
        moment_policy=moment_policy,
        pls_score_mode=pls_score_mode,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_sweep(
    X,
    y,
    folds,
    cv,
    lambdas,
    components,
    moment_policy,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        ["identity"],
        [("detrend", [1])],
        [("savgol_smooth", [5, 2])],
        [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy=moment_policy,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_sweep_whittaker(
    X,
    y,
    folds,
    cv,
    lambdas,
    components,
    moment_policy,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        [("whittaker", [100.0])],
        [("whittaker", [1000.0]), ("savgol_derivative", [7, 2, 1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy=moment_policy,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_sweep_fck(
    X,
    y,
    folds,
    cv,
    lambdas,
    components,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        [("fck", [0.0])],
        [("fck", [1.0])],
        [("fck", [1.0]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_sweep_gaussian(
    X,
    y,
    folds,
    cv,
    lambdas,
    components,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        [("gaussian", [1.0])],
        [("gaussian", [2.0])],
        [("gaussian", [1.0]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=("ridge", "pls"),
        scale_x=False,
        moment_policy="force_moments",
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_sweep_pls(
    X,
    y,
    folds,
    cv,
    components,
    moment_policy,
    pls_score_mode="cv",
    score_only=False,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        ["identity"],
        [("detrend", [1])],
        [("savgol_smooth", [5, 2])],
        [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=components,
        heads=("pls",),
        scale_x=False,
        moment_policy=moment_policy,
        pls_score_mode=pls_score_mode,
        score_only=score_only,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def run_aom_chain_screen_refit_pls(
    X,
    y,
    folds,
    cv,
    components,
    moment_policy,
    pls_score_mode="gcv_proxy",
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
    backend_min_cuda_product=None,
):
    chains = [
        ["identity"],
        [("detrend", [1])],
        [("savgol_smooth", [5, 2])],
        [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_screen_refit_campaign(
        X,
        y,
        chains=chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=components,
        heads=("pls",),
        scale_x=False,
        moment_policy=moment_policy,
        pls_score_mode=pls_score_mode,
        chain_chunk_size=len(chains),
        top_k=3,
        refit_top_k=2,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
        backend_min_cuda_product=backend_min_cuda_product,
    )


def run_aom_chain_fixed_fit(X, y, head, param, moment_policy):
    chain = [("savgol_smooth", [5, 2])]
    return n4m.aom_chain_fixed_fit_run(
        X,
        y,
        chain,
        head=head,
        param=param,
        scale_x=False,
        moment_policy=moment_policy,
    )


def run_aom_chain_sweep_ridge(
    X,
    y,
    folds,
    cv,
    lambdas,
    moment_policy,
    score_only=False,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    chains = [
        ["identity"],
        [("detrend", [1])],
        [("savgol_smooth", [5, 2])],
        [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
        [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
    ]
    return n4m.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=[],
        heads=("ridge",),
        scale_x=False,
        moment_policy=moment_policy,
        score_only=score_only,
        **cuda_opts(
            cuda_pls_parallel_folds,
            cuda_pls_min_device_features,
            cuda_pls_many_batched,
        ),
    )


def row(backend, profile, moment_policy, n_samples, n_features, cv, elapsed_ms, result):
    n_chains = int(result["n_chains"])
    n_candidates = int(result["n_candidates"])
    seconds = float(elapsed_ms) / 1000.0 if elapsed_ms > 0.0 else 0.0
    chains_per_second = (
        float(n_chains) / seconds if seconds > 0.0 and n_chains > 0 else 0.0
    )
    candidates_per_second = (
        float(n_candidates) / seconds
        if seconds > 0.0 and n_candidates > 0
        else 0.0
    )
    return {
        "backend": backend,
        "profile": profile,
        "moment_policy": moment_policy,
        "aom_pls_score_mode": int(result.get("aom_pls_score_mode", 0)),
        "score_only": int(result.get("score_only", 0)),
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_chains": n_chains,
        "n_candidates": n_candidates,
        "n_operator_moment_candidates": int(result["n_operator_moment_candidates"]),
        "n_ridge_operator_moment_candidates": int(
            result["n_ridge_operator_moment_candidates"]
        ),
        "n_pls_operator_moment_candidates": int(
            result["n_pls_operator_moment_candidates"]
        ),
        "n_banded_operator_moment_candidates": int(
            result["n_banded_operator_moment_candidates"]
        ),
        "n_structured_operator_moment_candidates": int(
            result["n_structured_operator_moment_candidates"]
        ),
        "n_dense_operator_moment_candidates": int(
            result["n_dense_operator_moment_candidates"]
        ),
        "n_materialized_candidates": int(result["n_materialized_candidates"]),
        "n_ridge_materialized_candidates": int(
            result["n_ridge_materialized_candidates"]
        ),
        "n_pls_materialized_candidates": int(
            result["n_pls_materialized_candidates"]
        ),
        "n_moment_prefix_cache_hits": int(
            result["n_moment_prefix_cache_hits"]
        ),
        "n_moment_prefix_cache_misses": int(
            result["n_moment_prefix_cache_misses"]
        ),
        "n_ridge_moment_cv_fits": int(
            result.get("n_ridge_moment_cv_fits", 0)
        ),
        "n_ridge_dual_materialized_cv_fits": int(
            result.get("n_ridge_dual_materialized_cv_fits", 0)
        ),
        "n_ridge_dual_cross_cv_fits": int(
            result.get("n_ridge_dual_cross_cv_fits", 0)
        ),
        "n_ridge_moment_score_batch_calls": int(
            result.get("n_ridge_moment_score_batch_calls", 0)
        ),
        "n_ridge_moment_score_batch_jobs": int(
            result.get("n_ridge_moment_score_batch_jobs", 0)
        ),
        "n_ridge_moment_final_fits": int(
            result.get("n_ridge_moment_final_fits", 0)
        ),
        "n_ridge_dual_materialized_final_fits": int(
            result.get("n_ridge_dual_materialized_final_fits", 0)
        ),
        "n_pls_moment_cv_fits": int(
            result["n_pls_moment_cv_fits"]
        ),
        "n_pls_moment_host_cv_fits": int(
            result.get("n_pls_moment_host_cv_fits", 0)
        ),
        "n_pls_moment_cuda_device_cv_fits": int(
            result.get("n_pls_moment_cuda_device_cv_fits", 0)
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            result.get("n_pls_moment_cuda_parallel_fold_batches", 0)
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            result.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
        ),
        "n_pls_moment_score_batch_calls": int(
            result.get("n_pls_moment_score_batch_calls", 0)
        ),
        "n_pls_moment_score_batch_jobs": int(
            result.get("n_pls_moment_score_batch_jobs", 0)
        ),
        "n_pls_materialized_cv_fits": int(
            result["n_pls_materialized_cv_fits"]
        ),
        "n_pls_gcv_proxy_candidates": int(
            result.get("n_pls_gcv_proxy_candidates", 0)
        ),
        "n_pls_gcv_proxy_fits": int(
            result.get("n_pls_gcv_proxy_fits", 0)
        ),
        "n_pls_gcv_proxy_batch_calls": int(
            result.get("n_pls_gcv_proxy_batch_calls", 0)
        ),
        "n_pls_gcv_proxy_batch_jobs": int(
            result.get("n_pls_gcv_proxy_batch_jobs", 0)
        ),
        "n_pls_moment_final_fits": int(
            result["n_pls_moment_final_fits"]
        ),
        "n_pls_moment_host_final_fits": int(
            result.get("n_pls_moment_host_final_fits", 0)
        ),
        "n_pls_moment_cuda_device_final_fits": int(
            result.get("n_pls_moment_cuda_device_final_fits", 0)
        ),
        "n_pls_materialized_final_fits": int(
            result["n_pls_materialized_final_fits"]
        ),
        "n_refit_candidates": int(result.get("n_refit_candidates", 0)),
        "n_refit_pls_moment_cv_fits": int(
            result.get("n_refit_pls_moment_cv_fits", 0)
        ),
        "n_refit_pls_moment_cuda_parallel_fold_batches": int(
            result.get("n_refit_pls_moment_cuda_parallel_fold_batches", 0)
        ),
        "n_refit_pls_moment_cuda_parallel_fold_jobs": int(
            result.get("n_refit_pls_moment_cuda_parallel_fold_jobs", 0)
        ),
        "n_refit_pls_materialized_cv_fits": int(
            result.get("n_refit_pls_materialized_cv_fits", 0)
        ),
        "selected_chain_id": int(result["selected_chain_id"]),
        "selected_head_id": int(result["selected_head_id"]),
        "selected_param": result["selected_param"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "elapsed_ms_median": elapsed_ms,
        "chains_per_second": chains_per_second,
        "candidates_per_second": candidates_per_second,
        "ms_per_chain": float(elapsed_ms) / float(n_chains) if n_chains > 0 else 0.0,
        "ms_per_candidate": (
            float(elapsed_ms) / float(n_candidates) if n_candidates > 0 else 0.0
        ),
        "projected_200k_chains_seconds": (
            200000.0 / chains_per_second if chains_per_second > 0.0 else 0.0
        ),
        "projected_200k_chains_minutes": (
            (200000.0 / chains_per_second) / 60.0
            if chains_per_second > 0.0
            else 0.0
        ),
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def row_screen_refit(
    backend, profile, moment_policy, n_samples, n_features, cv, elapsed_ms, report
):
    screen = dict(report["screen"])
    best = report["best_refit"]
    screen["aom_pls_score_mode"] = (
        1 if str(report.get("pls_score_mode", "cv")) == "gcv_proxy" else 0
    )
    screen["score_only"] = 0
    screen["n_refit_candidates"] = int(report["n_refit_candidates"])
    screen["n_refit_pls_moment_cv_fits"] = sum(
        int(item.get("n_pls_moment_cv_fits", 0)) for item in report["rows"]
    )
    screen["n_refit_pls_materialized_cv_fits"] = sum(
        int(item.get("n_pls_materialized_cv_fits", 0)) for item in report["rows"]
    )
    screen["selected_chain_id"] = int(best["chain_id"])
    screen["selected_head_id"] = int(best["head_id"])
    screen["selected_param"] = float(best["param"])
    screen["selected_cv_rmse"] = float(best["refit_cv_rmse"])
    return row(
        backend,
        profile,
        moment_policy,
        n_samples,
        n_features,
        cv,
        elapsed_ms,
        screen,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_sweep_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--profile", default="compact", choices=("compact", "wide"))
    parser.add_argument(
        "--cuda-pls-parallel-folds",
        action="store_true",
        help="Enable bounded stream-parallel CUDA scheduling for exact PLS1 moment fold jobs",
    )
    parser.add_argument(
        "--cuda-pls-min-device-features",
        type=int,
        default=None,
        help="Override the PLS1 moment CUDA device-route feature threshold",
    )
    parser.add_argument(
        "--cuda-pls-many-batched",
        action="store_true",
        help="Enable the experimental tiled/strided-batched CUDA route for many PLS1 moment jobs",
    )
    parser.add_argument(
        "--backend-min-cuda-product",
        type=int,
        default=None,
        help="Override campaign CPU/CUDA launch-recommendation threshold for screen/refit rows",
    )
    args = parser.parse_args()
    cuda_kwargs = {
        "cuda_pls_parallel_folds": args.cuda_pls_parallel_folds,
        "cuda_pls_min_device_features": args.cuda_pls_min_device_features,
        "cuda_pls_many_batched": args.cuda_pls_many_batched,
    }

    shapes = [(48, 64), (80, 128)]
    whittaker_shapes = [(48, 64), (64, 96)]
    # Exact PLS operator-moment force_moments rows need min_train >= 4p.
    # The common cv=4 smoke uses min_train=3n/4, so n >= ceil(16p/3).
    fck_shapes = [(192, 32), (288, 48)]
    gaussian_shapes = [(192, 32), (288, 48)]
    pls_shapes = [(60, 32), (80, 48), (96, 64)]
    exact_pls_score_shapes = [(192, 32), (288, 48)]
    screen_refit_shapes = [(192, 32), (288, 48)]
    fixed_fit_shapes = [(160, 32), (240, 48)]
    ridge_shapes = [(96, 32), (160, 64)]
    lambdas = np.asarray([0.01, 0.1, 1.0], dtype=np.float64)
    components = np.asarray([1, 2], dtype=np.int32)
    policies = ("auto", "materialized")
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=5200 + i)
        folds = balanced_folds(n_samples, args.cv)
        for policy in policies:
            elapsed, result = median_ms(
                lambda policy=policy: run_aom_sweep(
                    X, y, folds, args.cv, args.profile, lambdas, components,
                    policy, **cuda_kwargs,
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_sweep",
                    args.profile,
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

            elapsed, result = median_ms(
                lambda policy=policy: run_aom_chain_sweep(
                    X, y, folds, args.cv, lambdas, components, policy,
                    **cuda_kwargs,
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_chain_sweep",
                    "custom5",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

    for i, (n_samples, n_features) in enumerate(whittaker_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5450 + i)
        folds = balanced_folds(n_samples, args.cv)
        for policy in policies:
            elapsed, result = median_ms(
                lambda policy=policy: run_aom_chain_sweep_whittaker(
                    X, y, folds, args.cv, lambdas, components, policy,
                    **cuda_kwargs,
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_chain_sweep_whittaker",
                    "custom2_whittaker",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

    for i, (n_samples, n_features) in enumerate(fck_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5600 + i)
        folds = balanced_folds(n_samples, args.cv)
        elapsed, result = median_ms(
            lambda: run_aom_chain_sweep_fck(
                X, y, folds, args.cv, lambdas, components, **cuda_kwargs
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_aom_chain_sweep_fck",
                "custom3_fck",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for i, (n_samples, n_features) in enumerate(gaussian_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5650 + i)
        folds = balanced_folds(n_samples, args.cv)
        elapsed, result = median_ms(
            lambda: run_aom_chain_sweep_gaussian(
                X, y, folds, args.cv, lambdas, components, **cuda_kwargs
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_aom_chain_sweep_gaussian",
                "custom3_gaussian",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for i, (n_samples, n_features) in enumerate(pls_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5700 + i)
        folds = balanced_folds(n_samples, args.cv)
        for policy in policies:
            elapsed, result = median_ms(
                lambda policy=policy: run_aom_sweep_pls(
                    X, y, folds, args.cv, args.profile, components, policy,
                    **cuda_kwargs,
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_sweep_pls",
                    f"{args.profile}_pls",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

            elapsed, result = median_ms(
                lambda policy=policy: run_aom_chain_sweep_pls(
                    X, y, folds, args.cv, components, policy, **cuda_kwargs
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_chain_sweep_pls",
                    "custom5_pls",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

        elapsed, result = median_ms(
            lambda: run_aom_chain_sweep_pls(
                X,
                y,
                folds,
                args.cv,
                components,
                "force_moments",
                pls_score_mode="gcv_proxy",
                score_only=True,
                **cuda_kwargs,
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_aom_chain_sweep_pls_gcv_proxy",
                "custom5_pls_proxy",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for i, (n_samples, n_features) in enumerate(exact_pls_score_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5850 + i)
        folds = balanced_folds(n_samples, args.cv)
        elapsed, result = median_ms(
            lambda: run_aom_chain_sweep_pls(
                X,
                y,
                folds,
                args.cv,
                components,
                "force_moments",
                score_only=True,
                **cuda_kwargs,
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_aom_chain_sweep_pls_exact_score_only",
                "custom5_pls_exact_score",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for i, (n_samples, n_features) in enumerate(screen_refit_shapes):
        X, y = make_dataset(n_samples, n_features, seed=5900 + i)
        folds = balanced_folds(n_samples, args.cv)
        elapsed, result = median_ms(
            lambda: run_aom_chain_screen_refit_pls(
                X,
                y,
                folds,
                args.cv,
                components,
                "force_moments",
                pls_score_mode="gcv_proxy",
                backend_min_cuda_product=args.backend_min_cuda_product,
                **cuda_kwargs,
            ),
            args.repeats,
        )
        rows.append(
            row_screen_refit(
                "native_aom_chain_screen_refit_pls_gcv_proxy",
                "custom5_pls_proxy_refit2",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for i, (n_samples, n_features) in enumerate(fixed_fit_shapes):
        X, y = make_dataset(n_samples, n_features, seed=6050 + i)
        for head, param in (("pls", 2.0), ("ridge", 0.1)):
            elapsed, result = median_ms(
                lambda head=head, param=param: run_aom_chain_fixed_fit(
                    X, y, head, param, "auto"
                ),
                args.repeats,
            )
            rows.append(
                row(
                    f"native_aom_chain_fixed_fit_{head}",
                    "custom1_savgol_final_only",
                    "auto",
                    n_samples,
                    n_features,
                    0,
                    elapsed,
                    result,
                )
            )

    for i, (n_samples, n_features) in enumerate(ridge_shapes):
        X, y = make_dataset(n_samples, n_features, seed=6200 + i)
        folds = balanced_folds(n_samples, args.cv)
        for policy in policies:
            elapsed, result = median_ms(
                lambda policy=policy: run_aom_sweep_ridge(
                    X, y, folds, args.cv, args.profile, lambdas, policy,
                    **cuda_kwargs,
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_sweep_ridge",
                    f"{args.profile}_ridge",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

            elapsed, result = median_ms(
                lambda policy=policy: run_aom_chain_sweep_ridge(
                    X, y, folds, args.cv, lambdas, policy, **cuda_kwargs
                ),
                args.repeats,
            )
            rows.append(
                row(
                    "native_aom_chain_sweep_ridge",
                    "custom5_ridge",
                    policy,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                )
            )

        elapsed, result = median_ms(
            lambda: run_aom_chain_sweep_ridge(
                X, y, folds, args.cv, lambdas, "force_moments",
                score_only=True,
                **cuda_kwargs,
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_aom_chain_sweep_ridge_exact_score_only",
                "custom5_ridge_exact_score",
                "force_moments",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
            )
        )

    for item in rows:
        item["cuda_pls_parallel_folds"] = bool(args.cuda_pls_parallel_folds)
        item["cuda_pls_min_device_features"] = args.cuda_pls_min_device_features
        item["cuda_pls_many_batched"] = bool(args.cuda_pls_many_batched)
        item["backend_min_cuda_product"] = args.backend_min_cuda_product

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

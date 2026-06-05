#!/usr/bin/env python3
"""Screen/refit scaling benchmark for AOM preprocessing campaigns.

This benchmark isolates the current bottleneck behind broad preprocessing
screens: an explicit proxy first pass followed by exact-CV refits of retained
candidates. For PLS, it is not a fused batched IKPLS benchmark; it measures
the target that a future batched IKPLS/CUDA implementation must beat. For
Ridge, it measures how much grouped, signature-batched and union-batched
exact-CV refit avoid redundant lambda replays and Python/native call overhead.
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
        0.7 * X[:, 0]
        - 0.35 * X[:, min(7, n_features - 1)]
        + 0.22 * X[:, min(19, n_features - 1)]
        + 0.04 * rng.standard_normal(n_samples)
    )
    return X, y


def balanced_folds(n_samples: int, cv: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.int32) % int(cv)


def median_ms(fn, repeats: int) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one integer")
    if any(value < 1 for value in values):
        raise ValueError("list entries must be positive")
    return values


def run_screen(
    X,
    y,
    folds,
    *,
    cv: int,
    chains,
    head: str,
    lambdas,
    components,
    top_k: int,
    chain_chunk_size: int,
    chain_ordering: str,
    split_head_scoring: str,
    cuda_pls_parallel_folds: bool,
    cuda_pls_min_device_features: int | None,
    cuda_pls_many_batched: bool,
    backend_min_cuda_product: int | None,
):
    if head == "ridge":
        ridge_lambdas = lambdas
        pls_components = []
        pls_score_mode = "cv"
        heads = ("ridge",)
    elif head == "mixed":
        ridge_lambdas = lambdas
        pls_components = components
        pls_score_mode = "gcv_proxy"
        heads = ("ridge", "pls")
    else:
        ridge_lambdas = []
        pls_components = components
        pls_score_mode = "gcv_proxy"
        heads = ("pls",)
    return n4m.aom_chain_score_campaign(
        X,
        y,
        chains=chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=ridge_lambdas,
        pls_components=pls_components,
        heads=heads,
        scale_x=False,
        moment_policy="force_moments",
        pls_score_mode=pls_score_mode,
        chain_ordering=chain_ordering,
        split_head_scoring=split_head_scoring,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
        backend_min_cuda_product=backend_min_cuda_product,
        chain_chunk_size=chain_chunk_size,
        top_k=top_k,
    )


def parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one entry")
    return values


def run_refit(
    X,
    y,
    folds,
    *,
    cv: int,
    screen,
    top_k: int,
    refit_per_head_top_k: int | None,
    execution_mode: str,
    auto_max_extra_fraction: float,
    cuda_pls_parallel_folds: bool,
    cuda_pls_min_device_features: int | None,
    cuda_pls_many_batched: bool,
):
    pool = n4m.aom_screen_refit_candidate_pool(
        screen,
        refit_top_k=top_k,
        refit_per_head_top_k=refit_per_head_top_k,
    )
    refit = n4m.aom_refit_candidates(
        X,
        y,
        pool["rows"],
        top_k=None,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        execution_mode=execution_mode,
        auto_max_extra_fraction=auto_max_extra_fraction,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )
    refit = dict(refit)
    refit["candidate_pool"] = pool
    return refit


def run_final_fit(
    X,
    y,
    folds,
    *,
    cv: int,
    refit,
    cuda_pls_min_device_features: int | None,
):
    return n4m.NativeAOMFixedCandidateRegressor.from_refit_report(
        refit,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
        moment_policy="force_moments",
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        fit_mode="final_only",
        precomputed_cv_rmse=float(refit["best_cv"]["refit_cv_rmse"]),
    ).fit(X, y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_screen_refit_scaling.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=260)
    parser.add_argument("--n-features", type=int, default=48)
    parser.add_argument("--n-chains", type=int, default=24)
    parser.add_argument("--chain-chunk-size", type=int, default=8)
    parser.add_argument(
        "--chain-ordering",
        default="input",
        choices=("input", "prefix"),
        help="Campaign chunk ordering; prefix groups shared operator prefixes before chunking",
    )
    parser.add_argument(
        "--split-head-scoring",
        default="auto",
        choices=("off", "auto", "force"),
        help=(
            "For mixed screens, score Ridge and PLS in separate native calls so "
            "each head can use its batched native path; 'auto' (default) is "
            "score-preserving and inert for single-head screens, 'off' forces "
            "the legacy single mixed call"
        ),
    )
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
        help="Override campaign CPU/CUDA launch-recommendation threshold",
    )
    parser.add_argument("--head", default="pls", choices=("pls", "ridge", "mixed"))
    parser.add_argument("--components", default="1,2")
    parser.add_argument("--ridge-lambdas", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--refit-top-k", default="1,2,4,8,16")
    parser.add_argument(
        "--refit-per-head-top-k",
        type=int,
        default=0,
        help="Add each head's best screen rows to the exact-refit pool; 0 disables",
    )
    parser.add_argument(
        "--execution-modes",
        default="individual,grouped_score,batched_score,union_batched_score,auto",
        help="Comma-separated refit execution modes",
    )
    parser.add_argument(
        "--refit-auto-max-extra-fraction",
        type=float,
        default=1.0,
        help="Extra exact scores allowed per retained candidate when execution_mode=auto",
    )
    parser.add_argument(
        "--skip-final-fit",
        action="store_true",
        help="Do not time the reusable final-candidate fit after exact refit",
    )
    parser.add_argument("--seed", type=int, default=7300)
    args = parser.parse_args()

    components = np.asarray(parse_int_list(args.components), dtype=np.int32)
    lambdas = np.asarray(
        [float(part.strip()) for part in args.ridge_lambdas.split(",") if part.strip()],
        dtype=np.float64,
    )
    if lambdas.size == 0:
        raise ValueError("ridge-lambdas must contain at least one value")
    refit_top_k_values = parse_int_list(args.refit_top_k)
    execution_modes = parse_str_list(args.execution_modes)
    screen_top_k = max(refit_top_k_values)
    if args.refit_per_head_top_k < 0:
        raise ValueError("refit-per-head-top-k must be >= 0")
    refit_per_head_top_k = (
        None if int(args.refit_per_head_top_k) == 0
        else int(args.refit_per_head_top_k)
    )
    if args.head == "pls":
        n_params = int(components.size)
    elif args.head == "ridge":
        n_params = int(lambdas.size)
    else:
        n_params = int(components.size + lambdas.size)
    if screen_top_k > int(args.n_chains) * n_params:
        raise ValueError("max refit_top_k exceeds the number of candidates")

    X, y = make_dataset(args.n_samples, args.n_features, args.seed)
    folds = balanced_folds(args.n_samples, args.cv)
    chains = n4m.build_aom_strict_chain_grid("lab", max_chains=args.n_chains)
    if not chains:
        raise ValueError("chain grid is empty")

    screen_elapsed_ms, screen = median_ms(
        lambda: run_screen(
            X,
            y,
            folds,
            cv=args.cv,
            chains=chains,
            head=args.head,
            lambdas=lambdas,
            components=components,
            top_k=screen_top_k,
            chain_chunk_size=args.chain_chunk_size,
            chain_ordering=args.chain_ordering,
            split_head_scoring=args.split_head_scoring,
            cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=args.cuda_pls_min_device_features,
            cuda_pls_many_batched=args.cuda_pls_many_batched,
            backend_min_cuda_product=args.backend_min_cuda_product,
        ),
        args.repeats,
    )

    rows = []
    for execution_mode in execution_modes:
        if execution_mode not in {
            "individual",
            "grouped_score",
            "batched_score",
            "union_batched_score",
            "auto",
        }:
            raise ValueError(
                "execution modes must be individual, grouped_score, batched_score, "
                "union_batched_score, or auto"
            )
        for top_k in refit_top_k_values:
            plan_pool = n4m.aom_screen_refit_candidate_pool(
                screen,
                refit_top_k=top_k,
                refit_per_head_top_k=refit_per_head_top_k,
            )
            plan = n4m.aom_refit_execution_plan(
                plan_pool["rows"],
                top_k=None,
                auto_max_extra_fraction=args.refit_auto_max_extra_fraction,
            )
            planned_mode = (
                plan["recommended_mode"]
                if execution_mode == "auto"
                else execution_mode
            )
            planned = plan["by_mode"][planned_mode]
            refit_elapsed_ms, refit = median_ms(
                lambda top_k=top_k, execution_mode=execution_mode: run_refit(
                    X,
                    y,
                    folds,
                    cv=args.cv,
                    screen=screen,
                    top_k=top_k,
                    refit_per_head_top_k=refit_per_head_top_k,
                    execution_mode=execution_mode,
                    auto_max_extra_fraction=args.refit_auto_max_extra_fraction,
                    cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                    cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                    cuda_pls_many_batched=args.cuda_pls_many_batched,
                ),
                args.repeats,
            )
            pool = refit["candidate_pool"]
            n_refit_ridge_moment_cv_fits = int(
                refit.get("n_ridge_moment_cv_fits", 0)
            )
            n_refit_ridge_dual_materialized_cv_fits = int(
                refit.get("n_ridge_dual_materialized_cv_fits", 0)
            )
            n_refit_ridge_dual_cross_cv_fits = int(
                refit.get("n_ridge_dual_cross_cv_fits", 0)
            )
            n_refit_ridge_cv_fits = (
                n_refit_ridge_moment_cv_fits
                + n_refit_ridge_dual_materialized_cv_fits
                + n_refit_ridge_dual_cross_cv_fits
            )
            n_refit_ridge_moment_score_batch_calls = int(
                refit.get("n_ridge_moment_score_batch_calls", 0)
            )
            n_refit_ridge_moment_score_batch_jobs = int(
                refit.get("n_ridge_moment_score_batch_jobs", 0)
            )
            n_refit_pls_moment_cv_fits = int(
                refit.get("n_pls_moment_cv_fits", 0)
            )
            n_refit_pls_moment_host_cv_fits = int(
                refit.get("n_pls_moment_host_cv_fits", 0)
            )
            n_refit_pls_moment_cuda_device_cv_fits = int(
                refit.get("n_pls_moment_cuda_device_cv_fits", 0)
            )
            n_refit_pls_moment_cuda_parallel_fold_batches = int(
                refit.get("n_pls_moment_cuda_parallel_fold_batches", 0)
            )
            n_refit_pls_moment_cuda_parallel_fold_jobs = int(
                refit.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            )
            n_refit_pls_moment_score_batch_calls = int(
                refit.get("n_pls_moment_score_batch_calls", 0)
            )
            n_refit_pls_moment_score_batch_jobs = int(
                refit.get("n_pls_moment_score_batch_jobs", 0)
            )
            n_refit_pls_materialized_cv_fits = int(
                refit.get("n_pls_materialized_cv_fits", 0)
            )
            if args.skip_final_fit:
                final_elapsed_ms = 0.0
                final_diag = {
                    "selected_cv_rmse": float("nan"),
                    "n_candidates": 0,
                    "n_pls_moment_cv_fits": 0,
                    "n_pls_moment_host_cv_fits": 0,
                    "n_pls_moment_cuda_device_cv_fits": 0,
                    "n_pls_moment_cuda_parallel_fold_batches": 0,
                    "n_pls_moment_cuda_parallel_fold_jobs": 0,
                    "n_pls_moment_score_batch_calls": 0,
                    "n_pls_moment_score_batch_jobs": 0,
                    "n_ridge_moment_cv_fits": 0,
                    "n_ridge_dual_materialized_cv_fits": 0,
                    "n_ridge_dual_cross_cv_fits": 0,
                    "n_ridge_moment_score_batch_calls": 0,
                    "n_ridge_moment_score_batch_jobs": 0,
                    "n_ridge_moment_final_fits": 0,
                    "n_ridge_dual_materialized_final_fits": 0,
                    "n_pls_materialized_cv_fits": 0,
                    "n_pls_moment_final_fits": 0,
                    "n_pls_moment_host_final_fits": 0,
                    "n_pls_moment_cuda_device_final_fits": 0,
                    "n_pls_materialized_final_fits": 0,
                }
            else:
                final_elapsed_ms, final_model = median_ms(
                    lambda refit=refit: run_final_fit(
                        X,
                        y,
                        folds,
                        cv=args.cv,
                        refit=refit,
                        cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                    ),
                    args.repeats,
                )
                final_diag = final_model.get_diagnostics()
            best = refit["best_cv"]
            rows.append({
                "backend": "native_aom_screen_refit_scaling",
                "execution_mode": str(refit["execution_mode"]),
                "execution_mode_requested": str(
                    refit.get("execution_mode_requested", execution_mode)
                ),
                "execution_mode_auto_reason": str(
                    refit.get("execution_mode_auto_reason", "")
                ),
                "refit_auto_max_extra_fraction": float(
                    refit.get(
                        "auto_max_extra_fraction",
                        args.refit_auto_max_extra_fraction,
                    )
                ),
                "head": args.head,
                "chain_ordering": str(screen.get("chain_ordering", args.chain_ordering)),
                "split_head_scoring": str(
                    screen.get("split_head_scoring", args.split_head_scoring)
                ),
                "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
                "cuda_pls_min_device_features": args.cuda_pls_min_device_features,
                "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
                "backend_min_cuda_product": args.backend_min_cuda_product,
                "screen_backend_min_cuda_product": screen.get(
                    "backend_min_cuda_product"
                ),
                "n_split_head_chunks": int(screen.get("n_split_head_chunks", 0)),
                "n_chunk_score_calls": int(screen.get("n_chunk_score_calls", 0)),
                "screen_pls_moment_cuda_parallel_fold_batches": int(
                    screen.get("n_pls_moment_cuda_parallel_fold_batches", 0)
                ),
                "screen_pls_moment_cuda_parallel_fold_jobs": int(
                    screen.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
                ),
                "n_samples": int(args.n_samples),
                "n_features": int(args.n_features),
                "cv": int(args.cv),
                "n_chains": int(screen["n_chains"]),
                "n_components": int(components.size),
                "n_ridge_lambdas": int(lambdas.size),
                "n_params_per_chain": int(n_params),
                "n_screen_candidates": int(screen["n_candidates"]),
                "n_screen_top_candidates": int(len(screen["top_candidates"])),
                "refit_top_k": int(top_k),
                "refit_per_head_top_k": (
                    0 if refit_per_head_top_k is None else int(refit_per_head_top_k)
                ),
                "n_refit_global_candidates": int(
                    pool.get("n_refit_global_candidates", top_k)
                ),
                "n_refit_per_head_candidates": int(
                    pool.get("n_refit_per_head_candidates", 0)
                ),
                "n_refit_per_head_extra_candidates": int(
                    pool.get("n_refit_per_head_extra_candidates", 0)
                ),
                "n_refit_union_candidates": int(
                    pool.get("n_refit_union_candidates", refit["n_candidates"])
                ),
                "n_refit_candidates": int(refit["n_candidates"]),
                "n_refit_groups": int(refit.get("n_refit_groups", 0)),
                "n_refit_scored_candidates": int(
                    refit.get("n_refit_scored_candidates", refit["n_candidates"])
                ),
                "n_refit_extra_scored_candidates": int(
                    refit.get("n_refit_extra_scored_candidates", 0)
                ),
                "planned_refit_groups": int(planned["n_refit_groups"]),
                "planned_refit_scored_candidates": int(
                    planned["n_refit_scored_candidates"]
                ),
                "planned_refit_extra_scored_candidates": int(
                    planned["n_refit_extra_scored_candidates"]
                ),
                "screen_elapsed_ms_median": screen_elapsed_ms,
                "refit_elapsed_ms_median": refit_elapsed_ms,
                "screen_plus_refit_elapsed_ms": screen_elapsed_ms + refit_elapsed_ms,
                "screen_refit_final_elapsed_ms": (
                    screen_elapsed_ms + refit_elapsed_ms + final_elapsed_ms
                ),
                "final_fit_elapsed_ms_median": final_elapsed_ms,
                "screen_ms_per_chain": float(screen["ms_per_chain"]),
                "screen_ms_per_candidate": float(screen["ms_per_candidate"]),
                "screen_moment_prefix_cache_hits": int(
                    screen.get("n_moment_prefix_cache_hits", 0)
                ),
                "screen_moment_prefix_cache_misses": int(
                    screen.get("n_moment_prefix_cache_misses", 0)
                ),
                "screen_moment_prefix_cache_hit_fraction": float(
                    screen.get("moment_prefix_cache_hit_fraction", 0.0)
                ),
                "screen_chains_per_second": float(
                    screen.get("chains_per_second", 0.0)
                ),
                "screen_candidates_per_second": float(
                    screen.get("candidates_per_second", 0.0)
                ),
                "screen_projected_200k_chains_seconds": float(
                    screen.get("projected_200k_chains_seconds", 0.0)
                ),
                "screen_projected_200k_chains_minutes": float(
                    screen.get("projected_200k_chains_minutes", 0.0)
                ),
                "refit_ms_per_candidate": (
                    refit_elapsed_ms / float(max(1, refit["n_candidates"]))
                ),
                "refit_ms_per_ridge_cv_fit": (
                    refit_elapsed_ms / float(max(1, n_refit_ridge_cv_fits))
                ),
                "refit_ms_per_pls_cv_fit": (
                    refit_elapsed_ms / float(max(1, n_refit_pls_moment_cv_fits))
                ),
                "n_ridge_moment_cv_fits": int(
                    screen.get("n_ridge_moment_cv_fits", 0)
                ),
                "n_ridge_dual_materialized_cv_fits": int(
                    screen.get("n_ridge_dual_materialized_cv_fits", 0)
                ),
                "n_ridge_dual_cross_cv_fits": int(
                    screen.get("n_ridge_dual_cross_cv_fits", 0)
                ),
                "n_ridge_moment_score_batch_calls": int(
                    screen.get("n_ridge_moment_score_batch_calls", 0)
                ),
                "n_ridge_moment_score_batch_jobs": int(
                    screen.get("n_ridge_moment_score_batch_jobs", 0)
                ),
                "n_pls_gcv_proxy_candidates": int(
                    screen.get("n_pls_gcv_proxy_candidates", 0)
                ),
                "n_pls_gcv_proxy_fits": int(screen.get("n_pls_gcv_proxy_fits", 0)),
                "n_pls_gcv_proxy_batch_calls": int(
                    screen.get("n_pls_gcv_proxy_batch_calls", 0)
                ),
                "n_pls_gcv_proxy_batch_jobs": int(
                    screen.get("n_pls_gcv_proxy_batch_jobs", 0)
                ),
                "n_refit_ridge_moment_cv_fits": int(
                    n_refit_ridge_moment_cv_fits
                ),
                "n_refit_ridge_dual_materialized_cv_fits": int(
                    n_refit_ridge_dual_materialized_cv_fits
                ),
                "n_refit_ridge_dual_cross_cv_fits": int(
                    n_refit_ridge_dual_cross_cv_fits
                ),
                "n_refit_ridge_moment_score_batch_calls": int(
                    n_refit_ridge_moment_score_batch_calls
                ),
                "n_refit_ridge_moment_score_batch_jobs": int(
                    n_refit_ridge_moment_score_batch_jobs
                ),
                "n_refit_pls_moment_cv_fits": int(n_refit_pls_moment_cv_fits),
                "n_refit_pls_moment_host_cv_fits": int(
                    n_refit_pls_moment_host_cv_fits
                ),
                "n_refit_pls_moment_cuda_device_cv_fits": int(
                    n_refit_pls_moment_cuda_device_cv_fits
                ),
                "n_refit_pls_moment_cuda_parallel_fold_batches": int(
                    n_refit_pls_moment_cuda_parallel_fold_batches
                ),
                "n_refit_pls_moment_cuda_parallel_fold_jobs": int(
                    n_refit_pls_moment_cuda_parallel_fold_jobs
                ),
                "n_refit_pls_moment_score_batch_calls": int(
                    n_refit_pls_moment_score_batch_calls
                ),
                "n_refit_pls_moment_score_batch_jobs": int(
                    n_refit_pls_moment_score_batch_jobs
                ),
                "n_refit_pls_materialized_cv_fits": int(
                    n_refit_pls_materialized_cv_fits
                ),
                "final_fit_n_candidates": int(final_diag["n_candidates"]),
                "final_fit_n_ridge_moment_cv_fits": int(
                    final_diag.get("n_ridge_moment_cv_fits", 0)
                ),
                "final_fit_n_ridge_dual_materialized_cv_fits": int(
                    final_diag.get("n_ridge_dual_materialized_cv_fits", 0)
                ),
                "final_fit_n_ridge_dual_cross_cv_fits": int(
                    final_diag.get("n_ridge_dual_cross_cv_fits", 0)
                ),
                "final_fit_n_ridge_moment_score_batch_calls": int(
                    final_diag.get("n_ridge_moment_score_batch_calls", 0)
                ),
                "final_fit_n_ridge_moment_score_batch_jobs": int(
                    final_diag.get("n_ridge_moment_score_batch_jobs", 0)
                ),
                "final_fit_n_ridge_moment_final_fits": int(
                    final_diag.get("n_ridge_moment_final_fits", 0)
                ),
                "final_fit_n_ridge_dual_materialized_final_fits": int(
                    final_diag.get("n_ridge_dual_materialized_final_fits", 0)
                ),
                "final_fit_n_pls_moment_cv_fits": int(
                    final_diag.get("n_pls_moment_cv_fits", 0)
                ),
                "final_fit_n_pls_moment_host_cv_fits": int(
                    final_diag.get("n_pls_moment_host_cv_fits", 0)
                ),
                "final_fit_n_pls_moment_cuda_device_cv_fits": int(
                    final_diag.get("n_pls_moment_cuda_device_cv_fits", 0)
                ),
                "final_fit_n_pls_moment_cuda_parallel_fold_batches": int(
                    final_diag.get("n_pls_moment_cuda_parallel_fold_batches", 0)
                ),
                "final_fit_n_pls_moment_cuda_parallel_fold_jobs": int(
                    final_diag.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
                ),
                "final_fit_n_pls_moment_score_batch_calls": int(
                    final_diag.get("n_pls_moment_score_batch_calls", 0)
                ),
                "final_fit_n_pls_moment_score_batch_jobs": int(
                    final_diag.get("n_pls_moment_score_batch_jobs", 0)
                ),
                "final_fit_n_pls_materialized_cv_fits": int(
                    final_diag.get("n_pls_materialized_cv_fits", 0)
                ),
                "final_fit_n_pls_moment_final_fits": int(
                    final_diag.get("n_pls_moment_final_fits", 0)
                ),
                "final_fit_n_pls_moment_host_final_fits": int(
                    final_diag.get("n_pls_moment_host_final_fits", 0)
                ),
                "final_fit_n_pls_moment_cuda_device_final_fits": int(
                    final_diag.get("n_pls_moment_cuda_device_final_fits", 0)
                ),
                "final_fit_n_pls_materialized_final_fits": int(
                    final_diag.get("n_pls_materialized_final_fits", 0)
                ),
                "best_screen_score": float(screen["best"]["cv_rmse"]),
                "best_refit_cv_rmse": float(best["refit_cv_rmse"]),
                "final_fit_selected_cv_rmse": float(final_diag["selected_cv_rmse"]),
                "best_refit_screen_score": float(best["screen_score"]),
                "best_refit_screen_rank": int(best.get("screen_rank", 0)),
                "best_refit_cv_rank": int(best.get("cv_rank", 0)),
                "best_refit_rank_delta": int(best.get("rank_delta", 0)),
                "library_path": n4m.library_path(),
                "abi": ".".join(str(value) for value in n4m.abi_version()),
            })

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

#!/usr/bin/env python3
"""Timing smoke benchmark for ``n4m.aom_staged_chain_campaign``.

This measures the *orchestration* cost of the staged strict-chain
screen/refit campaign: several score-only strict-linear screens (the
``compact`` / ``wide`` / ``lab`` profiles), cross-stage global + per-head
retention, a single exact-CV refit of the retained union, and the post-hoc
impact / rank diagnostics.

It is deliberately end-to-end wall-clock timing of the Python orchestrator on
top of the single ``libn4m`` runtime. It is **not** a fused batched IKPLS
benchmark: the exact-refit pass it times is the explicit, per-candidate exact
CV that a future fused IKPLS/CUDA grinder must beat, not that grinder itself.
Use ``bench_aom_screen_refit_scaling.py`` for the screen-vs-refit execution-mode
breakdown; this script tracks the staged campaign's total orchestration cost.
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
    for repeat_index in range(int(repeats)):
        t0 = time.perf_counter()
        result = fn(repeat_index)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one integer")
    if any(value < 1 for value in values):
        raise ValueError("list entries must be positive")
    return values


def parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one entry")
    return values


def benchmark_checkpoint_dir(
    base: str | None, *, plan: str, repeat_index: int, repeats: int
) -> str | None:
    if base is None or int(repeats) <= 1:
        return base
    safe_plan = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(plan)
    ).strip("._")
    return str(Path(base) / f"{safe_plan or 'plan'}_repeat_{repeat_index:03d}")


def run_campaign(
    X,
    y,
    folds,
    *,
    plan: str,
    cv: int,
    heads,
    components,
    lambdas,
    max_chains: int,
    chain_chunk_size: int,
    top_k: int,
    refit_top_k: int,
    refit_per_head_top_k: int | None,
    checkpoint_dir: str | None,
    resume: bool,
    max_chunks_per_run: int | None,
    moment_policy: str,
    pls_score_mode: str,
    cuda_pls_parallel_folds: bool,
    cuda_pls_min_device_features: int | None,
    cuda_pls_many_batched: bool,
    backend_min_cuda_product: int | None,
):
    return n4m.aom_staged_chain_campaign(
        X,
        y,
        plan=plan,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        pls_components=components,
        heads=heads,
        max_chains=max_chains,
        chain_chunk_size=chain_chunk_size,
        top_k=top_k,
        refit_top_k=refit_top_k,
        refit_per_head_top_k=refit_per_head_top_k,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_chunks_per_run=max_chunks_per_run,
        scale_x=False,
        moment_policy=moment_policy,
        pls_score_mode=pls_score_mode,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
        backend_min_cuda_product=backend_min_cuda_product,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_staged_chain_campaign_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--n-features", type=int, default=48)
    parser.add_argument(
        "--plans",
        default="compact,compact_wide,compact_wide_lab",
        help="Comma-separated staged plans to time (each becomes one CSV row)",
    )
    parser.add_argument("--max-chains", type=int, default=16)
    parser.add_argument("--chain-chunk-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--refit-top-k", type=int, default=8)
    parser.add_argument(
        "--refit-per-head-top-k",
        type=int,
        default=2,
        help="Extra per-head retained rows added to the exact-refit union; 0 disables",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional staged checkpoint directory; creates one JSON checkpoint per stage",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume matching staged checkpoints when --checkpoint-dir is set",
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=None,
        help="Limit new chunks processed per stage in this call",
    )
    parser.add_argument("--heads", default="ridge,pls")
    parser.add_argument(
        "--moment-policy",
        default="auto",
        help=(
            "Screen + refit moment routing policy passed to the campaign "
            "(e.g. auto, force_moments). 'force_moments' requires a build with "
            "the moment chain-sweep path enabled"
        ),
    )
    parser.add_argument(
        "--pls-score-mode",
        choices=("cv", "gcv_proxy"),
        default="cv",
        help=(
            "PLS first-pass screen score mode. Use gcv_proxy for explicit "
            "proxy-vs-exact timing/recall campaigns; exact-CV refit is unchanged."
        ),
    )
    parser.add_argument("--components", default="1,2")
    parser.add_argument("--ridge-lambdas", default="0.01,0.1,1.0,10.0")
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
    parser.add_argument("--seed", type=int, default=7311)
    args = parser.parse_args()

    plans = parse_str_list(args.plans)
    heads = parse_str_list(args.heads)
    components = np.asarray(parse_int_list(args.components), dtype=np.int32)
    lambdas = np.asarray(
        [float(part.strip()) for part in args.ridge_lambdas.split(",") if part.strip()],
        dtype=np.float64,
    )
    if lambdas.size == 0:
        raise ValueError("ridge-lambdas must contain at least one value")
    if args.refit_top_k < 1:
        raise ValueError("refit-top-k must be positive")
    if args.refit_per_head_top_k < 0:
        raise ValueError("refit-per-head-top-k must be >= 0")
    if args.max_chunks_per_run is not None and args.max_chunks_per_run < 1:
        raise ValueError("max-chunks-per-run must be positive when provided")
    refit_per_head_top_k = (
        None if int(args.refit_per_head_top_k) == 0 else int(args.refit_per_head_top_k)
    )

    X, y = make_dataset(args.n_samples, args.n_features, args.seed)
    folds = balanced_folds(args.n_samples, args.cv)

    rows = []
    for plan in plans:
        elapsed_ms, report = median_ms(
            lambda repeat_index, plan=plan: run_campaign(
                X,
                y,
                folds,
                plan=plan,
                cv=args.cv,
                heads=heads,
                components=components,
                lambdas=lambdas,
                max_chains=args.max_chains,
                chain_chunk_size=args.chain_chunk_size,
                top_k=args.top_k,
                refit_top_k=args.refit_top_k,
                refit_per_head_top_k=refit_per_head_top_k,
                checkpoint_dir=benchmark_checkpoint_dir(
                    args.checkpoint_dir,
                    plan=plan,
                    repeat_index=repeat_index,
                    repeats=args.repeats,
                ),
                resume=not bool(args.no_resume),
                max_chunks_per_run=args.max_chunks_per_run,
                moment_policy=args.moment_policy,
                pls_score_mode=args.pls_score_mode,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
                backend_min_cuda_product=args.backend_min_cuda_product,
            ),
            args.repeats,
        )

        best = report["best"]
        retention = report["retention"]
        refit = report["refit"]
        impact = report.get("impact")
        rank = report.get("rank_diagnostics")

        # Route summary over exact-refit rows. Refit reports preserve the screen
        # route under ``screen_score_route`` so selection diagnostics stay clear;
        # normalize it back to the generic route keys for the route-summary helper.
        route_rows = []
        for refit_row in report["rows"]:
            route_row = dict(refit_row)
            if "screen_score_route" in route_row and "score_route" not in route_row:
                route_row["score_route"] = route_row["screen_score_route"]
            if (
                "screen_score_route_id" in route_row
                and "score_route_id" not in route_row
            ):
                route_row["score_route_id"] = route_row["screen_score_route_id"]
            route_rows.append(route_row)
        try:
            route = n4m.aom_candidate_route_summary(route_rows)
        except (ValueError, KeyError):
            route = {}

        rows.append({
            "backend": "native_aom_staged_chain_campaign",
            "plan": str(report["plan"]),
            "plan_requested": str(plan),
            "heads": "|".join(sorted(set(heads))),
            "n_stages": int(report["n_stages"]),
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "cv": int(report["cv"]),
            "max_chains": int(args.max_chains),
            "chain_chunk_size": int(args.chain_chunk_size),
            "top_k": int(args.top_k),
            "checkpoint_dir": "" if report["checkpoint_dir"] is None else str(report["checkpoint_dir"]),
            "resume": not bool(args.no_resume),
            "max_chunks_per_run": (
                0 if args.max_chunks_per_run is None else int(args.max_chunks_per_run)
            ),
            "screen_complete": bool(report["screen_complete"]),
            "n_remaining_stage_chunks_total": int(
                report["n_remaining_stage_chunks_total"]
            ),
            "n_components": int(components.size),
            "n_ridge_lambdas": int(lambdas.size),
            "moment_policy": str(args.moment_policy),
            "pls_score_mode": str(report.get("pls_score_mode", args.pls_score_mode)),
            "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
            "cuda_pls_min_device_features": args.cuda_pls_min_device_features,
            "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
            "backend_min_cuda_product": args.backend_min_cuda_product,
            "elapsed_ms_median": elapsed_ms,
            "n_screen_candidates_total": int(report["n_screen_candidates_total"]),
            "n_merged_global_candidates": int(report["n_merged_global_candidates"]),
            "n_refit_candidates": int(report["n_refit_candidates"]),
            "refit_ms_per_candidate": (
                elapsed_ms / float(max(1, int(report["n_refit_candidates"])))
            ),
            # Retention counts (global + per-head union).
            "retention_refit_top_k": int(retention["refit_top_k"]),
            "retention_refit_per_head_top_k": (
                0
                if retention["refit_per_head_top_k"] is None
                else int(retention["refit_per_head_top_k"])
            ),
            "n_refit_global_candidates": int(
                retention.get("n_refit_global_candidates", 0)
            ),
            "n_refit_per_head_candidates": int(
                retention.get("n_refit_per_head_candidates", 0)
            ),
            "n_refit_per_head_extra_candidates": int(
                retention.get("n_refit_per_head_extra_candidates", 0)
            ),
            "n_refit_union_candidates": int(
                retention.get("n_refit_union_candidates", 0)
            ),
            # Selected production winner (exact-CV on train only).
            "selection_metric": str(report["selection_metric"]),
            "selection_uses_test_set": bool(report["selection_uses_test_set"]),
            "selected_head": str(best["head"]),
            "selected_param": float(best["param"]),
            "selected_chain_len": int(len(best.get("chain", ()))),
            "selected_campaign_stage": str(best.get("campaign_stage", "")),
            "best_refit_cv_rmse": float(best["refit_cv_rmse"]),
            "best_screen_cv_rmse": float(best.get("screen_cv_rmse", float("nan"))),
            "best_cv_rank": int(best.get("cv_rank", 0)),
            "best_screen_rank": int(best.get("screen_rank", 0)),
            "best_rank_delta": int(best.get("rank_delta", 0)),
            "n_heads_selected": int(len(report.get("best_by_head", {}))),
            # Diagnostics availability.
            "impact_available": impact is not None,
            "impact_best_score": (
                float(impact["best_score"]) if impact is not None else float("nan")
            ),
            "impact_n_operator_groups": (
                int(len(impact.get("by_operator", ()))) if impact is not None else 0
            ),
            "rank_diagnostics_available": rank is not None,
            "spearman_rank_correlation": (
                float(rank["spearman_rank_correlation"])
                if rank is not None
                else float("nan")
            ),
            "rank_mean_abs_rank_delta": (
                float(rank["mean_abs_rank_delta"]) if rank is not None else float("nan")
            ),
            # Refit route / CUDA summary counters where available.
            "refit_execution_mode": str(refit.get("execution_mode", "")),
            "n_pls_moment_cv_fits": int(refit.get("n_pls_moment_cv_fits", 0)),
            "n_pls_moment_host_cv_fits": int(
                refit.get("n_pls_moment_host_cv_fits", 0)
            ),
            "n_pls_moment_cuda_device_cv_fits": int(
                refit.get("n_pls_moment_cuda_device_cv_fits", 0)
            ),
            "n_pls_moment_cuda_parallel_fold_batches": int(
                refit.get("n_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "n_pls_moment_cuda_parallel_fold_jobs": int(
                refit.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_pls_moment_cuda_many_batched_batches": int(
                refit.get("n_pls_moment_cuda_many_batched_batches", 0)
            ),
            "n_pls_moment_cuda_many_batched_jobs": int(
                refit.get("n_pls_moment_cuda_many_batched_jobs", 0)
            ),
            "n_ridge_moment_cv_fits": int(refit.get("n_ridge_moment_cv_fits", 0)),
            "n_ridge_moment_eigen_path_preparations": int(
                refit.get("n_ridge_moment_eigen_path_preparations", 0)
            ),
            "n_ridge_moment_eigen_path_cv_fits": int(
                refit.get("n_ridge_moment_eigen_path_cv_fits", 0)
            ),
            "n_ridge_moment_direct_cv_fits": int(
                refit.get("n_ridge_moment_direct_cv_fits", 0)
            ),
            "route_operator_moment_candidate_fraction": float(
                route.get("operator_moment_candidate_fraction", 0.0)
            ),
            "route_materialized_candidate_fraction": float(
                route.get("materialized_candidate_fraction", 0.0)
            ),
            "route_n_operator_moment_candidates": int(
                route.get("n_operator_moment_candidates", 0)
            ),
            "route_n_materialized_candidates": int(
                route.get("n_materialized_candidates", 0)
            ),
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

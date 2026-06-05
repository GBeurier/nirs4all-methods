#!/usr/bin/env python3
"""Live CPU-vs-CUDA crossover benchmark for native moment sweeps.

The Python binding loads one libn4m shared object per process, so this script
spawns separate child interpreters for the CPU and CUDA builds. It times
score-only `n4m.sweep_run` Ridge/PLS screens on identical synthetic datasets
and writes one CSV row per backend/shape/head.

Use `--compare-cuda-pls-many-batched` to run CUDA twice on the same shapes:
the default exact PLS moment route and the optional many-batched route. The CSV
then includes `speedup_vs_cuda_default` for direct one-GPU comparison.

This is intentionally narrow: it measures the current CUDA value for the
moment/AOM grinder substrate. It is not a fused batched IKPLS benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CPU_LIB = REPO / "build" / "dev-release" / "cpp" / "src" / "libn4m.so"
DEFAULT_CUDA_LIB = REPO / "build" / "cuda-on" / "cpp" / "src" / "libn4m.so"
DEFAULT_PYTHONPATH = REPO / "bindings" / "python" / "src"
DEFAULT_OUT = REPO / "benchmarks" / "cross_binding" / "moment_gpu_crossover.csv"
DEFAULT_SHAPES = "260x48,260x256,512x512,256x1024"


CHILD_CODE = r"""
import json
import statistics
import sys
import time

import numpy as np
import n4m


def make_dataset(n_samples, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.7 * X[:, 0]
        - 0.2 * X[:, min(7, n_features - 1)]
        + 0.05 * rng.standard_normal(n_samples)
    )
    return X, y


def balanced_folds(n_samples, cv):
    return np.arange(n_samples, dtype=np.int32) % int(cv)


def median_ms(fn, repeats):
    times = []
    result = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def run_case(X, y, folds, cv, head, cuda_pls_min_device_features, cuda_pls_many_batched):
    if head == "ridge":
        return n4m.sweep_run(
            X,
            y,
            cv=cv,
            fold_ids=folds,
            ridge_lambdas=[0.01, 0.1, 1.0, 10.0],
            heads=("ridge",),
            scale_x=False,
            score_only=True,
        )
    return n4m.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1, 2, 4],
        heads=("pls",),
        scale_x=False,
        score_only=True,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )


payload = json.loads(sys.stdin.read())
backend = payload["backend"]
repeats = int(payload["repeats"])
cv = int(payload["cv"])
cuda_pls_min_device_features = payload.get("cuda_pls_min_device_features")
cuda_pls_many_batched = payload.get("cuda_pls_many_batched")
cuda_pls_profile = str(payload.get("cuda_pls_profile", "default"))
rows = []
for shape in payload["shapes"]:
    n_samples = int(shape[0])
    n_features = int(shape[1])
    X, y = make_dataset(n_samples, n_features, 1000 + n_samples + n_features)
    folds = balanced_folds(n_samples, cv)
    for head in ("ridge", "pls"):
        row = {
            "backend": backend,
            "head": head,
            "n_samples": n_samples,
            "n_features": n_features,
            "cv": cv,
            "repeats": repeats,
            "library_path": n4m.library_path(),
            "abi": ".".join(str(value) for value in n4m.abi_version()),
            "cuda_available": int(n4m.lib.n4m_backend_is_available(5)),
            "cuda_pls_profile": cuda_pls_profile,
            "cuda_pls_min_device_features": (
                "" if cuda_pls_min_device_features is None
                else int(cuda_pls_min_device_features)
            ),
            "cuda_pls_many_batched": (
                "" if cuda_pls_many_batched is None
                else int(bool(cuda_pls_many_batched))
            ),
            "error": "",
        }
        try:
            elapsed_ms, result = median_ms(
                lambda: run_case(
                    X,
                    y,
                    folds,
                    cv,
                    head,
                    cuda_pls_min_device_features,
                    cuda_pls_many_batched,
                ),
                repeats,
            )
            row.update({
                "elapsed_ms_median": elapsed_ms,
                "selected_cv_rmse": float(result["selected_cv_rmse"]),
                "n_candidates": int(result["n_candidates"]),
                "n_ridge_moment_candidates": int(
                    result.get("n_ridge_moment_candidates", 0)
                ),
                "n_ridge_dual_materialized_candidates": int(
                    result.get("n_ridge_dual_materialized_candidates", 0)
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
                "n_pls_moment_cv_fits": int(
                    result.get("n_pls_moment_cv_fits", 0)
                ),
                "n_pls_moment_host_cv_fits": int(
                    result.get("n_pls_moment_host_cv_fits", 0)
                ),
                "n_pls_moment_cuda_device_cv_fits": int(
                    result.get("n_pls_moment_cuda_device_cv_fits", 0)
                ),
                "n_pls_moment_score_batch_calls": int(
                    result.get("n_pls_moment_score_batch_calls", 0)
                ),
                "n_pls_moment_score_batch_jobs": int(
                    result.get("n_pls_moment_score_batch_jobs", 0)
                ),
                "n_pls_moment_cuda_parallel_fold_batches": int(
                    result.get("n_pls_moment_cuda_parallel_fold_batches", 0)
                ),
                "n_pls_moment_cuda_parallel_fold_jobs": int(
                    result.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
                ),
                "n_pls_materialized_cv_fits": int(
                    result.get("n_pls_materialized_cv_fits", 0)
                ),
            })
        except Exception as exc:
            row.update({
                "elapsed_ms_median": float("nan"),
                "selected_cv_rmse": float("nan"),
                "n_candidates": 0,
                "n_ridge_moment_candidates": 0,
                "n_ridge_dual_materialized_candidates": 0,
                "n_ridge_moment_cv_fits": 0,
                "n_ridge_dual_materialized_cv_fits": 0,
                "n_ridge_dual_cross_cv_fits": 0,
                "n_pls_moment_cv_fits": 0,
                "n_pls_moment_host_cv_fits": 0,
                "n_pls_moment_cuda_device_cv_fits": 0,
                "n_pls_moment_score_batch_calls": 0,
                "n_pls_moment_score_batch_jobs": 0,
                "n_pls_moment_cuda_parallel_fold_batches": 0,
                "n_pls_moment_cuda_parallel_fold_jobs": 0,
                "n_pls_materialized_cv_fits": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
        rows.append(row)
print(json.dumps(rows, sort_keys=True))
"""


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for part in text.split(","):
        item = part.strip().lower()
        if not item:
            continue
        n_text, p_text = item.split("x", 1)
        n = int(n_text)
        p = int(p_text)
        if n <= 1 or p <= 0:
            raise ValueError("shapes must be n>1 and p>0")
        shapes.append((n, p))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def run_backend(
    *,
    backend: str,
    lib_path: Path,
    pythonpath: Path,
    shapes: list[tuple[int, int]],
    cv: int,
    repeats: int,
    cuda_visible_devices: str | None,
    cuda_pls_min_device_features: int | None,
    cuda_pls_many_batched: bool | None,
    cuda_pls_profile: str,
) -> list[dict[str, object]]:
    env = os.environ.copy()
    env["N4M_LIB_PATH"] = str(lib_path)
    env["PYTHONPATH"] = str(pythonpath)
    if backend == "cuda" and cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    payload = {
        "backend": backend,
        "shapes": shapes,
        "cv": int(cv),
        "repeats": int(repeats),
        "cuda_pls_min_device_features": cuda_pls_min_device_features,
        "cuda_pls_many_batched": cuda_pls_many_batched,
        "cuda_pls_profile": cuda_pls_profile,
    }
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_CODE],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} child failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return list(json.loads(proc.stdout))


def add_speedups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cpu_by_case: dict[tuple[str, int, int], float] = {}
    cuda_default_by_case: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row["backend"] != "cpu":
            if (
                row["backend"] == "cuda"
                and str(row.get("cuda_pls_profile", "")) == "default"
                and not row.get("error")
            ):
                key = (
                    str(row["head"]),
                    int(row["n_samples"]),
                    int(row["n_features"]),
                )
                try:
                    elapsed = float(row["elapsed_ms_median"])
                except (TypeError, ValueError):
                    continue
                if elapsed > 0.0:
                    cuda_default_by_case[key] = elapsed
            continue
        key = (str(row["head"]), int(row["n_samples"]), int(row["n_features"]))
        try:
            elapsed = float(row["elapsed_ms_median"])
        except (TypeError, ValueError):
            continue
        if elapsed > 0.0:
            cpu_by_case[key] = elapsed

    out: list[dict[str, object]] = []
    for row in rows:
        row = dict(row)
        key = (str(row["head"]), int(row["n_samples"]), int(row["n_features"]))
        cpu_elapsed = cpu_by_case.get(key)
        elapsed = float(row.get("elapsed_ms_median", 0.0))
        if cpu_elapsed is not None and elapsed > 0.0:
            speedup = cpu_elapsed / elapsed
        else:
            speedup = 0.0
        cuda_default_elapsed = cuda_default_by_case.get(key)
        if (
            row.get("backend") == "cuda"
            and cuda_default_elapsed is not None
            and elapsed > 0.0
        ):
            speedup_vs_cuda_default = cuda_default_elapsed / elapsed
        else:
            speedup_vs_cuda_default = 0.0
        row["speedup_vs_cpu"] = speedup
        row["speedup_vs_cuda_default"] = speedup_vs_cuda_default
        row["recommended_backend"] = (
            "cuda"
            if row["backend"] == "cuda" and speedup > 1.05 and not row.get("error")
            else "cpu"
        )
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu-lib", default=str(DEFAULT_CPU_LIB))
    parser.add_argument("--cuda-lib", default=str(DEFAULT_CUDA_LIB))
    parser.add_argument("--pythonpath", default=str(DEFAULT_PYTHONPATH))
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cuda-pls-min-device-features", type=int, default=None)
    parser.add_argument("--cuda-pls-many-batched", action="store_true")
    parser.add_argument(
        "--compare-cuda-pls-many-batched",
        action="store_true",
        help=(
            "run CUDA twice, once with the default PLS moment route and once "
            "with cuda_pls_many_batched=True; CPU baseline is run once"
        ),
    )
    args = parser.parse_args(argv)

    shapes = parse_shapes(args.shapes)
    rows = []
    rows.extend(
        run_backend(
            backend="cpu",
            lib_path=Path(args.cpu_lib),
            pythonpath=Path(args.pythonpath),
            shapes=shapes,
            cv=args.cv,
            repeats=args.repeats,
            cuda_visible_devices=None,
            cuda_pls_min_device_features=args.cuda_pls_min_device_features,
            cuda_pls_many_batched=False,
            cuda_pls_profile="cpu_baseline",
        )
    )
    cuda_runs = [
        ("many_batched", True)
        if args.cuda_pls_many_batched
        else ("default", False)
    ]
    if args.compare_cuda_pls_many_batched:
        cuda_runs = [("default", False), ("many_batched", True)]
    for cuda_pls_profile, cuda_pls_many_batched in cuda_runs:
        rows.extend(
            run_backend(
                backend="cuda",
                lib_path=Path(args.cuda_lib),
                pythonpath=Path(args.pythonpath),
                shapes=shapes,
                cv=args.cv,
                repeats=args.repeats,
                cuda_visible_devices=args.cuda_visible_devices,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=cuda_pls_many_batched,
                cuda_pls_profile=cuda_pls_profile,
            )
        )
    rows = add_speedups(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

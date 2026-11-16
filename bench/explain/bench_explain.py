#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Benchmark the native explainability kernels through the Python binding.

Measures wall time and model evaluations of ``n4m.explain`` for the session
explainers (permutation / kernel SHAP, occlusion) and the exact model-specific
explainers (linear SHAP, path-dependent TreeSHAP) over a grid of spectral
widths, group widths and background sizes, with a Ridge-like linear predictor
and, when scikit-learn is installed, a RandomForestRegressor. Optionally times
``shap.PermutationExplainer`` on the same per-feature cells for comparison.

The data are synthetic smooth spectra; the timings characterise the engine and
the ask/tell overhead, not a scientific result. Prediction time of the host
model is included in the session timings (it is part of the real cost).

Usage (after building libn4m; set N4M_LIB_PATH if it is not found)::

    python bench/explain/bench_explain.py --out bench/explain/results/explain
    python bench/explain/bench_explain.py --out /tmp/explain --features 256 \\
        --widths 16,32 --backgrounds 20 --compare-shap

Writes ``<out>.json`` (records) and ``<out>.md`` (table).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from n4m import explain as ex
from n4m import version as n4m_version

SESSION_METHODS = ("permutation_shap", "kernel_shap", "occlusion")
EXACT_METHODS = ("linear_shap", "tree_shap")


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _spectra(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Smooth synthetic spectra (integrated noise plus broad bands)."""
    grid = np.linspace(0.0, 1.0, p)
    base = np.cumsum(rng.normal(size=(n, p)), axis=1) / math.sqrt(p)
    bands = sum(
        rng.uniform(0.2, 1.0, size=(n, 1)) * np.exp(-((grid - c) ** 2) / (2 * 0.03**2))
        for c in (0.2, 0.45, 0.7, 0.85)
    )
    return np.ascontiguousarray(base + bands)


def _rf_flat(model) -> ex.TreeEnsemble:
    """Flatten a fitted sklearn RandomForestRegressor (bench-local converter)."""
    offsets, cols = [0], {k: [] for k in ("feature", "threshold", "left", "right", "cover", "value")}
    for est in model.estimators_:
        t = est.tree_
        offsets.append(offsets[-1] + t.node_count)
        cols["feature"].append(np.where(t.children_left < 0, -1, t.feature))
        cols["threshold"].append(t.threshold)
        cols["left"].append(t.children_left)
        cols["right"].append(t.children_right)
        cols["cover"].append(t.weighted_n_node_samples)
        cols["value"].append(t.value[:, 0, 0])
    return ex.TreeEnsemble(
        tree_offsets=offsets,
        n_outputs=1,
        tree_scale=1.0 / len(model.estimators_),
        **{k: np.concatenate(v) for k, v in cols.items()},
    )


def _estimated_model_rows(method: str, n_rows: int, n_groups: int, n_background: int,
                          n_permutations: int) -> float:
    """Rough a-priori cost guard used only to skip infeasible cells."""
    if method == "permutation_shap":
        per_row = 2 * n_permutations * n_groups
    elif method == "kernel_shap":
        per_row = min(2.0**n_groups - 2, 2 * n_groups + 2048) if n_groups < 60 else 2 * n_groups + 2048
    else:
        per_row = n_groups + 1
    return float(n_rows) * per_row * n_background


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:  # pragma: no cover - optional
        RandomForestRegressor = None
    shap = None
    if args.compare_shap:
        try:
            import shap  # type: ignore[no-redef]
        except ImportError:  # pragma: no cover - optional
            print("shap is not installed; --compare-shap ignored")
    records: list[dict[str, Any]] = []

    def record(**fields: Any) -> None:
        records.append(fields)
        wall = fields.get("wall_s")
        print(
            f"{fields['model']:>6} {fields['method']:>17} p={fields['n_features']:<5} "
            f"w={fields['width']:<3} bg={fields['n_background']:<3} "
            + (f"{wall:8.3f}s evals={fields.get('evaluations')}" if wall is not None
               else f"skipped: {fields.get('status')}")
        )

    for p in args.features:
        train = _spectra(rng, args.n_train, p)
        w = rng.normal(size=p) / math.sqrt(p)
        y = train @ w + 0.05 * rng.normal(size=args.n_train)
        models: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Any]] = {
            "ridge": (lambda rows, w=w: rows @ w + 0.25, None),
        }
        if RandomForestRegressor is not None:
            rf = RandomForestRegressor(
                n_estimators=args.rf_trees, max_depth=args.rf_depth, n_jobs=1, random_state=0
            ).fit(train, y)
            models["forest"] = (rf.predict, rf)
        X = train[: args.rows]
        for width in args.widths:
            groups, n_groups = (None, p) if width == 1 else ex.groups_fixed_width(p, width)
            for n_bg in args.backgrounds:
                background = train[-n_bg:]
                base = dict(n_features=p, width=width, n_groups=n_groups, n_background=n_bg,
                            n_rows=args.rows)
                for name, (predict, fitted) in models.items():
                    for method in args.methods:
                        cell = dict(base, model=name, method=method)
                        if method in SESSION_METHODS:
                            cost = _estimated_model_rows(method, args.rows, n_groups, n_bg,
                                                         args.n_permutations)
                            if cost > args.max_model_rows:
                                record(**cell, wall_s=None, status=f"estimated {cost:.3g} model rows")
                                continue
                            result, wall = _timed(lambda: ex.explain(
                                predict, X, background=background, feature_group=groups,
                                n_groups=None if groups is None else n_groups, method=method,
                                n_outputs=1, n_permutations=args.n_permutations,
                                n_coalitions=args.n_coalitions, max_batch_rows=args.max_batch_rows,
                                seed=0))
                        elif method == "linear_shap":
                            if name != "ridge":
                                continue
                            result, wall = _timed(lambda: ex.linear_shap(
                                X, w, reference=background.mean(axis=0), intercept=[0.25],
                                feature_group=groups,
                                n_groups=None if groups is None else n_groups))
                        elif method == "tree_shap":
                            if fitted is None:
                                continue
                            flat = _rf_flat(fitted)
                            result, wall = _timed(lambda: ex.tree_shap(
                                X, flat, feature_group=groups,
                                n_groups=None if groups is None else n_groups))
                        else:
                            raise ValueError(f"unknown method {method}")
                        record(
                            **cell,
                            wall_s=wall,
                            evaluations=result.evaluations,
                            exactness=result.exactness,
                            additivity_max_abs_error=result.additivity_max_abs_error,
                            status="ok",
                        )
                        if (shap is not None and method == "permutation_shap" and width == 1
                                and p <= args.shap_max_features):
                            masker = shap.maskers.Independent(background, max_samples=n_bg)
                            explainer = shap.PermutationExplainer(predict, masker)
                            # n_permutations antithetic orderings per row.
                            _, shap_wall = _timed(lambda: explainer(
                                X, max_evals=args.n_permutations * (2 * p + 1), silent=True))
                            record(**dict(cell, method="shap.PermutationExplainer"),
                                   wall_s=shap_wall, evaluations=None, exactness="sampled",
                                   additivity_max_abs_error=None, status="ok")
    return records


def _markdown(records: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# Explainability benchmark",
        "",
        "Synthetic smooth spectra; wall time includes host prediction. "
        f"libn4m {meta['n4m_version']}, Python {meta['python']}, {meta['platform']}.",
        "",
        "| model | method | n_features | width | n_groups | n_background | n_rows | wall (s) "
        "| evaluations | model rows/s | additivity error | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in records:
        wall, evals = r.get("wall_s"), r.get("evaluations")
        rate = f"{evals / wall:.3g}" if wall and evals else ""
        err = r.get("additivity_max_abs_error")
        lines.append(
            f"| {r['model']} | {r['method']} | {r['n_features']} | {r['width']} | {r['n_groups']} "
            f"| {r['n_background']} | {r['n_rows']} | {'' if wall is None else f'{wall:.4f}'} "
            f"| {'' if evals is None else evals} | {rate} | {'' if err is None else f'{err:.2e}'} "
            f"| {r['status']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path,
                        help="output path stem; writes <out>.json and <out>.md")
    parser.add_argument("--features", type=_ints, default=[256, 1024, 2048])
    parser.add_argument("--widths", type=_ints, default=[1, 16, 32, 64],
                        help="group widths; 1 = one group per feature")
    parser.add_argument("--backgrounds", type=_ints, default=[1, 20, 50])
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--methods", type=lambda s: s.split(","),
                        default=list(SESSION_METHODS + EXACT_METHODS))
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--n-coalitions", type=int, default=0, help="kernel budget (0 = native auto)")
    parser.add_argument("--max-batch-rows", type=int, default=4096)
    parser.add_argument("--max-model-rows", type=float, default=2e7,
                        help="skip session cells whose estimated model rows exceed this")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--rf-trees", type=int, default=50)
    parser.add_argument("--rf-depth", type=int, default=8)
    parser.add_argument("--compare-shap", action="store_true",
                        help="also time shap.PermutationExplainer on per-feature cells")
    parser.add_argument("--shap-max-features", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    unknown = set(args.methods) - set(SESSION_METHODS + EXACT_METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")

    records = run(args)
    meta = {
        "n4m_version": n4m_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "records": records}, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_markdown(records, meta), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

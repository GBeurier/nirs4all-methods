"""Python tier-1: pls4all._methods.*_fit raw entry points."""
from __future__ import annotations

import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)

from pls4all import _methods
from pls4all._context import Context
from pls4all._config import Config
from pls4all._types import Algorithm, Solver, Deflation


def _basic_cfg(nc: int):
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = nc
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    return cfg


def _make_ctx() -> Context:
    ctx = Context()
    try:
        ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
    except Exception:
        pass
    return ctx


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        y2d = y.reshape(-1, 1)
        ctx = _make_ctx()
        cfg = _basic_cfg(nc)
        try:
            if algo == "pls":
                # plain SIMPLS via sparse_simpls @ lambda=0
                res = _methods.sparse_simpls_fit(ctx, cfg, X, y2d,
                                                   sparsity_lambda=0.0)
            elif algo == "sparse_simpls":
                res = _methods.sparse_simpls_fit(ctx, cfg, X, y2d,
                                                   sparsity_lambda=0.05)
            elif algo == "cppls":
                res = _methods.cppls_fit(ctx, cfg, X, y2d, gamma=0.5)
            elif algo == "ecr":
                res = _methods.ecr_fit(ctx, cfg, X, y2d, alpha=0.5)
            elif algo == "mir_pls":
                res = _methods.mir_pls_fit(ctx, cfg, X, y2d)
            elif algo == "ridge_pls":
                res = _methods.ridge_pls_fit(ctx, cfg, X, y2d,
                                              ridge_lambda=1.0)
            elif algo == "robust_pls":
                res = _methods.robust_pls_fit(ctx, cfg, X, y2d,
                                                huber_k=1.345,
                                                max_irls_iter=20)
            elif algo == "missing_aware_nipals":
                res = _methods.missing_aware_nipals_fit(ctx, cfg, X, y2d)
            else:
                # Generic fallback: try _methods.{algo}_fit(ctx, cfg, X, y)
                fn = getattr(_methods, f"{algo}_fit", None)
                if fn is None:
                    raise RuntimeError(
                        f"python_tier1: unsupported algo '{algo}' "
                        f"(no _methods.{algo}_fit)")
                res = fn(ctx, cfg, X, y2d)
            preds = np.asarray(res.matrix("predictions"))
        finally:
            cfg.close()
        return preds

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    import pls4all
    versions = collect_versions("Python",
                                  pls4all=getattr(pls4all, "__version__", "?"),
                                  numpy=_safe_version("numpy"))
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

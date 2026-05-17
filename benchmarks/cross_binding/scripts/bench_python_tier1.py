"""Python tier-1: pls4all._methods.*_fit raw entry points."""
from __future__ import annotations

import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version,
                       dispatch_pls4all_fit)

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
            res = dispatch_pls4all_fit(algo, ctx, cfg, X, y2d, n, p, seed)
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

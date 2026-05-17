"""Python tier-1: pls4all SIMPLS via the raw _methods._fit binding."""
from _common import parse_args, time_runs, emit

import numpy as np
from pls4all import _methods
from pls4all._context import Context
from pls4all._config import Config
from pls4all._types import Algorithm, Deflation, Solver


def make_cfg(nc):
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


def main():
    X, y, nc, runs = parse_args()
    y2d = y.reshape(-1, 1)

    def do_one():
        ctx = Context()
        cfg = make_cfg(nc)
        try:
            res = _methods.sparse_simpls_fit(ctx, cfg, X, y2d,
                                              sparsity_lambda=0.0)
            _ = res.matrix("predictions")
        finally:
            cfg.close()

    rec = time_runs(do_one, runs)
    emit(rec)


if __name__ == "__main__":
    main()

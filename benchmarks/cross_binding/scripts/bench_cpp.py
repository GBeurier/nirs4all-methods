"""C++ direct baseline — uses pls4all's Python ctypes bridge to call
the SAME C ABI entry point (`_methods.<algo>_fit` via Context+Config)
that every other pls4all binding uses.

The "C++ direct" label means: no high-level Python wrapper
(no `sklearn.PLSRegression` class, no `pls()` formula) sits between
the ctypes call and the C function. The cost is essentially the C++
kernel + matrix-view marshalling.

This script is the parity reference for every cell in the matrix —
it must work for every algorithm we benchmark, otherwise downstream
backends have nothing to compare against. The actual per-algo dispatch
lives in `_common.dispatch_pls4all_fit` so cpp and python_tier1 stay
bit-identical.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version,
                       dispatch_pls4all_fit)


def _libp4a():
    lib_dir = os.environ.get(
        "PLS4ALL_LIB_DIR",
        "/home/delete/nirs4all/pls4all/build/blas-omp/cpp/src")
    return ctypes.CDLL(os.path.join(lib_dir, "libp4a.so"))


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()
    so = _libp4a()
    so.p4a_get_version_string.restype = ctypes.c_char_p
    libp4a_v = so.p4a_get_version_string().decode()

    from pls4all._context import Context  # noqa: E402
    from pls4all._config import Config  # noqa: E402
    from pls4all._types import Algorithm, Solver, Deflation  # noqa: E402

    def _basic_cfg():
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

    def _make_ctx():
        ctx = Context()
        try:
            ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
        except Exception:
            pass
        return ctx

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        y2d = y.reshape(-1, 1)
        ctx = _make_ctx()
        cfg = _basic_cfg()
        try:
            res = dispatch_pls4all_fit(algo, ctx, cfg, X, y2d, n, p, seed)
            preds = np.asarray(res.matrix("predictions"))
        finally:
            cfg.close()
        return preds

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    versions = collect_versions("C++ (via ctypes bridge)",
                                  libp4a=libp4a_v,
                                  numpy=_safe_version("numpy"))
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

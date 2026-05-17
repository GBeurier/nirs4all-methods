"""C++ direct baseline — uses pls4all's Python ctypes bridge to call
the SAME C ABI entry point (`_methods.<algo>_fit` via Context+Config)
that every other pls4all binding uses. This guarantees parity matches
across pls4all backends: same C kernel, same scaling, same NIPALS/SIMPLS
iteration. The "C++ direct" label refers to the fact that no high-level
Python wrapper (no `sklearn.PLSRegression` class, no `pls()` formula)
sits between the ctypes call and the C function — the cost is
essentially the C++ kernel + the matrix-view marshalling.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)


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

    # Defer to pls4all's Python ctx/cfg bridge — same C ABI surface as
    # all other pls4all backends, so the predictions match bit-for-bit
    # and the "cpp" cell is a meaningful parity reference.
    from pls4all import _methods  # noqa: E402
    from pls4all._context import Context  # noqa: E402
    from pls4all._config import Config  # noqa: E402
    from pls4all._types import Algorithm, Solver, Deflation  # noqa: E402

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        y2d = y.reshape(-1, 1)
        ctx = Context()
        try:
            ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
        except Exception:
            pass
        cfg = Config()
        cfg.algorithm = Algorithm.PLS_REGRESSION
        cfg.solver = Solver.SIMPLS
        cfg.deflation = Deflation.REGRESSION
        cfg.n_components = nc
        cfg.center_x = True
        cfg.scale_x = False
        cfg.center_y = True
        cfg.scale_y = False
        try:
            if algo == "pls":
                res = _methods.sparse_simpls_fit(ctx, cfg, X, y2d,
                                                   sparsity_lambda=0.0)
            elif algo == "sparse_simpls":
                res = _methods.sparse_simpls_fit(ctx, cfg, X, y2d,
                                                   sparsity_lambda=0.05)
            else:
                fit_fn = getattr(_methods, f"{algo}_fit", None)
                if fit_fn is None:
                    raise RuntimeError(f"no _methods.{algo}_fit available")
                res = fit_fn(ctx, cfg, X, y2d)
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

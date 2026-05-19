"""C++ direct baseline — calls the canonical C ABI entry point that every
other pls4all binding ultimately uses, with no high-level Python wrapper in
between.

Routes through `MethodSpec.pls4all_fn` (the same registry that pins the
parity gate's idea of "canonical pls4all call"), so this script must cover
every method declared in the registry. The orchestrator multiplexes
`PLS4ALL_LIB_DIR` across the four libp4a builds (dev-release, blas-on,
omp-on, blas-omp); this script honours that env var and runs the kernel
with whichever BLAS/OMP configuration the build ships.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   load_method)


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

    import pls4all

    method = load_method(algo)

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        params = adapted_params(method, n, p, nc)
        Y, extras = benchmark_inputs(method, X, y, params, seed)
        with pls4all.Context() as ctx, pls4all.Config() as cfg:
            try:
                ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
            except Exception:
                pass
            # Establish the same center/scale defaults the R + MATLAB
            # dispatchers use (center_*=True, scale_*=False). The
            # registry's regression functions reset these explicitly via
            # `_model_fit_predict_pls4all`, but classifier wrappers
            # (_pls_lda_pls4all, _pls_logistic_pls4all, …) only set
            # center_* — without this preamble they'd inherit the
            # `pls4all.Config()` default `scale_*=True` and produce
            # results that diverge from R/MATLAB by the std-scaling
            # factor.
            cfg.center_x = True
            cfg.scale_x = False
            cfg.center_y = True
            cfg.scale_y = False
            result = method.pls4all_fn(ctx, cfg, X, Y, **params, **extras)
            try:
                return np.asarray(result.matrix(method.prediction_key))
            finally:
                result.close()

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    versions = collect_versions("C++ (via ctypes bridge)",
                                  libp4a=libp4a_v,
                                  numpy=_safe_version("numpy"),
                                  registry_method=algo)
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

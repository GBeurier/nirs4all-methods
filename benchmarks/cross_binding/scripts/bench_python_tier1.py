"""Python tier-1 — pls4all canonical Python invocation via the registry.

Today this is materially identical to `bench_cpp.py` (both route through
`MethodSpec.pls4all_fn`); the column exists in the matrix because the
display layer historically separated "C++ direct ctypes" from "Python with
the same ctypes path" — same kernel, same parameters, useful to keep as a
sanity check that the Python-side overhead is negligible. If/when the
display columns collapse, this script can be dropped.
"""
from __future__ import annotations

import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   load_method)


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()
    method = load_method(algo)

    def fit_predict(seed: int) -> np.ndarray:
        import pls4all
        X, y = load_dataset(csv_dir, n, p, seed)
        params = adapted_params(method, n, p, nc)
        Y, extras = benchmark_inputs(method, X, y, params, seed)
        with pls4all.Context() as ctx, pls4all.Config() as cfg:
            try:
                ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
            except Exception:
                pass
            result = method.pls4all_fn(ctx, cfg, X, Y, **params, **extras)
            try:
                return np.asarray(result.matrix(method.prediction_key))
            finally:
                result.close()

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    import pls4all
    versions = collect_versions("Python",
                                  pls4all=getattr(pls4all, "__version__", "?"),
                                  numpy=_safe_version("numpy"),
                                  registry_method=algo)
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

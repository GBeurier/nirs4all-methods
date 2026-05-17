"""Canonical pls4all benchmark backend driven by parity_timing.MethodSpec."""
from __future__ import annotations

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

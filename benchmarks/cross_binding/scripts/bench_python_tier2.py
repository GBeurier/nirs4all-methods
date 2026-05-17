"""Python tier-2: pls4all.sklearn.* idiomatic estimators."""
from __future__ import annotations

import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()

    # Lazy imports so the script's import time doesn't pollute the timing.
    from pls4all.sklearn import (
        PLSRegression, SparseSimplsRegression, CPPLSRegression,
        ECRegression, MIRPLSRegression, RidgePLSRegression,
        RobustPLSRegression, MissingAwareNipalsRegression,
        PCR, PLSCanonical, OPLSRegression, NPLSRegression,
    )
    from pls4all._context import Context

    def make_estimator(seed: int):
        if algo == "pls":
            return PLSRegression(nc, solver="simpls",
                                  center_x=True, scale_x=False,
                                  center_y=True, scale_y=False)
        if algo == "sparse_simpls":
            return SparseSimplsRegression(nc, sparsity_lambda=0.05)
        if algo == "cppls":
            return CPPLSRegression(nc, gamma=0.5)
        if algo == "ecr":
            return ECRegression(nc, alpha=0.5)
        if algo == "mir_pls":
            return MIRPLSRegression(nc)
        if algo == "ridge_pls":
            return RidgePLSRegression(nc, ridge_lambda=1.0)
        if algo == "robust_pls":
            return RobustPLSRegression(nc, huber_k=1.345, max_irls_iter=20)
        if algo == "missing_aware_nipals":
            return MissingAwareNipalsRegression(nc)
        if algo == "pcr":
            return PCR(nc, center_x=True, scale_x=False,
                       center_y=True, scale_y=False)
        if algo == "pls_canonical":
            return PLSCanonical(nc, center_x=True, scale_x=False,
                                 center_y=True, scale_y=False)
        if algo == "opls":
            return OPLSRegression(nc, center_x=True, scale_x=False,
                                    center_y=True, scale_y=False)
        raise RuntimeError(f"python_tier2: unsupported algo '{algo}'")

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        est = make_estimator(seed)
        # Honour the thread cap via Context if estimator allows it.
        try:
            ctx = Context()
            ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
        except Exception:
            pass
        est.fit(X, y)
        return np.asarray(est.predict(X))

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    import pls4all
    versions = collect_versions("Python",
                                  pls4all=getattr(pls4all, "__version__", "?"),
                                  numpy=_safe_version("numpy"))
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

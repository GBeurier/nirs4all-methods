"""External Python: scikit-learn-based equivalents.

Algo dispatch:
  pls               → sklearn.cross_decomposition.PLSRegression(scale=False)
  pcr               → Pipeline(PCA + LinearRegression)
  ridge_pls         → Pipeline(PCA(nc) + Ridge(alpha=1.0))  (proxy)
  gpr_pls           → Pipeline(PCA(nc) + GaussianProcessRegressor)
For algos sklearn doesn't ship (sparse_simpls, cppls, opls, mir_pls,
n_pls, …), raise NotImplemented → orchestrator marks N/A.
"""
from __future__ import annotations

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()

    if algo == "pls":
        from sklearn.cross_decomposition import PLSRegression

        def fit_predict(seed: int) -> np.ndarray:
            X, y = load_dataset(csv_dir, n, p, seed)
            m = PLSRegression(n_components=nc, scale=False)
            m.fit(X, y)
            return np.asarray(m.predict(X))
    elif algo == "pcr":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline

        def fit_predict(seed: int) -> np.ndarray:
            X, y = load_dataset(csv_dir, n, p, seed)
            m = make_pipeline(PCA(n_components=nc),
                                LinearRegression())
            m.fit(X, y)
            return np.asarray(m.predict(X))
    elif algo == "ridge_pls":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        def fit_predict(seed: int) -> np.ndarray:
            X, y = load_dataset(csv_dir, n, p, seed)
            m = make_pipeline(PCA(n_components=nc), Ridge(alpha=1.0))
            m.fit(X, y)
            return np.asarray(m.predict(X))
    elif algo == "gpr_pls":
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from sklearn.pipeline import make_pipeline

        def fit_predict(seed: int) -> np.ndarray:
            X, y = load_dataset(csv_dir, n, p, seed)
            m = make_pipeline(
                PCA(n_components=nc),
                GaussianProcessRegressor(
                    kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4),
                    random_state=seed, normalize_y=True))
            m.fit(X, y)
            return np.asarray(m.predict(X))
    else:
        # sklearn has no native equivalent → emit a skip record so the
        # orchestrator can mark the cell N/A cleanly.
        import json
        print(json.dumps({"ok": False,
                            "reason": f"algo '{algo}' not implemented by sklearn",
                            "skipped": True,
                            "versions": collect_versions(
                                "Python",
                                sklearn=_safe_version("sklearn"),
                                numpy=_safe_version("numpy"))}))
        return

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    versions = collect_versions("Python",
                                  sklearn=_safe_version("sklearn"),
                                  numpy=_safe_version("numpy"))
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

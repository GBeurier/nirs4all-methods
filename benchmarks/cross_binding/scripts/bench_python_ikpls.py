"""External Python: ikpls (Improved Kernel PLS) — covers `pls` only.

ikpls.fast_cross_validation has a SIMPLS implementation comparable to
sklearn's PLSRegression but written in numpy + numba (PyPI: ikpls).
"""
from __future__ import annotations

import json
import sys
import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()

    if algo != "pls":
        print(json.dumps({"ok": False,
                            "reason": f"algo '{algo}' not implemented by ikpls",
                            "skipped": True,
                            "versions": collect_versions(
                                "Python",
                                ikpls=_safe_version("ikpls"),
                                numpy=_safe_version("numpy"))}))
        return

    try:
        from ikpls.numpy_ikpls import PLS as IkPLS
    except Exception as e:
        print(json.dumps({"ok": False, "reason": f"ikpls import: {e}"}))
        return

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        y2d = y.reshape(-1, 1)
        m = IkPLS(algorithm=1)  # 1 = improved kernel PLS algo #1 (SIMPLS-like)
        m.fit(X, y2d, A=nc)
        return np.asarray(m.predict(X))[nc - 1]  # ikpls returns per-component

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    versions = collect_versions("Python",
                                  ikpls=_safe_version("ikpls"),
                                  numpy=_safe_version("numpy"))
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()

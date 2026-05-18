#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate the parity fixture consumed by `run_smoke.mjs`.

Fits the same (X, Y) PLS regression that the WASM smoke fits, using
the native Python pls4all binding (which itself parity-tested against
R `pls` and `spls`). The fixture is serialized as JSON and committed
alongside the smoke so CI can compare WASM output to the native
reference without re-running Python.

Run from the repo root:

    PYTHONPATH=bindings/python/src \\
      parity/python_generator/.venv/bin/python \\
      bindings/js/test/generate_parity_fixture.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pls4all


def main() -> None:
    n, p, q = 50, 5, 1
    X = np.zeros((n, p), dtype=np.float64)
    Y = np.zeros((n, q), dtype=np.float64)
    for i in range(n):
        for j in range(p):
            X[i, j] = math.sin((i + 1) * (j + 1) * 0.3)
        Y[i, 0] = X[i, 0] + 0.5 * X[i, 1] - 0.3 * X[i, 2]

    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = 3
        cfg.center_x = True
        cfg.center_y = True
        m = pls4all.Model.fit(ctx, cfg, X, Y)
        coefs = m.get_array(ctx, pls4all.ModelArrayKind.COEFFICIENTS)
        x_mean = m.get_array(ctx, pls4all.ModelArrayKind.X_MEAN)
        y_mean = m.get_array(ctx, pls4all.ModelArrayKind.Y_MEAN)
        preds = m.predict(ctx, X)
        m.close()

    fixture = {
        "pls4all_version": pls4all.version(),
        "n": n, "p": p, "q": q, "n_components": 3,
        "coefficients": coefs.ravel().tolist(),
        "x_mean": x_mean.ravel().tolist(),
        "y_mean": y_mean.ravel().tolist(),
        "predictions": preds.ravel().tolist(),
    }
    out_path = Path(__file__).resolve().parent / "parity_fixture.json"
    out_path.write_text(json.dumps(fixture, indent=2) + "\n",
                         encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

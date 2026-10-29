#!/usr/bin/env python3
"""Cross-binding transport witness from native kernels independently tested in pytest.

No scientific golden is refreshed here; this fixture captures the same finite
named outputs for R, JS/WASM and Octave parity. NumPy/lstsq/LOF/SVD oracles live
in bindings/python/tests/test_intrinsic_diagnostics.py.
"""

import json
from pathlib import Path

import numpy as np
from n4m.roles import method_class

rng = np.random.default_rng(20261029)
ref = np.array([[1.0, 2.0, 4.0, 8.0, 9.0, 10.0]])
x = rng.normal(size=(10, 6))
cases = [
    (
        "spectral_reference_qc",
        np.r_[ref, ref * 2 + 3],
        ref,
        {"reference_mode": "supplied"},
    ),
    (
        "spectral_events_qc",
        rng.normal(0, 0.1, (3, 30)),
        None,
        {"clip_lower": -1.0, "clip_upper": 1.0},
    ),
    ("spectral_noise_proxy", 2 + rng.normal(0, 0.1, (5, 12)), None, {}),
    ("local_density", x, None, {"n_neighbors": 3}),
    ("pca_reference_qc", x[:3] * 1.1 + 0.02, x, {"n_components": 3}),
]
output = []
for name, X, target, params in cases:
    inputs = {} if target is None else {"X_target": target}
    result = method_class("diagnostics." + name)(**params).run(X, **inputs)
    output.append(
        {
            "method_id": "diagnostics." + name,
            "X": X.tolist(),
            "X_target": None if target is None else target.tolist(),
            "parameters": params,
            "expected": {k: np.asarray(v).tolist() for k, v in result.items()},
        }
    )
p = (
    Path(__file__).resolve().parents[1]
    / "parity/fixtures/intrinsic_diagnostics.v1.json"
)
p.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "provenance": "native C ABI transport capture; independent scientific tests in test_intrinsic_diagnostics.py",
            "cases": output,
        },
        indent=2,
        allow_nan=False,
    )
    + "\n"
)
copy = p.parents[2] / "bindings/r/n4m/inst/parity/intrinsic_diagnostics.v1.json"
copy.parent.mkdir(parents=True, exist_ok=True)
copy.write_bytes(p.read_bytes())
print(p)

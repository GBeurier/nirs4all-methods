# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Regenerate deterministic crossed/nested Gaussian LMM development references.

Requires statsmodels==0.14.6. Run with --check-native only when the optional
mixeff backend is built; ordinary regeneration uses the independent reference.
"""

import argparse
import json
import math
import pathlib

import numpy as np
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec

parser = argparse.ArgumentParser()
parser.add_argument("--check-native", action="store_true")
parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path(__file__).parent / "fixtures/mixed_effects_reference_v1.json",
)
args = parser.parse_args()
if args.check_native:
    from n4m.component_analysis import mixed_effects


def design(kind):
    x = []
    y = []
    groups = []
    for g in range(6):
        for h in range(4):
            for r in range(4):
                i = len(y)
                a = r - 1.5
                b = [-1, 1, 1, -1][r]
                x.append([a, 1.0, b])
                ug = 1.1 * math.sin(g + 1)
                vh = (
                    0.5 * math.cos(h + 0.2)
                    if kind == "crossed"
                    else 0.4 * math.sin((g + 1) * (h + 1))
                )
                y.append(
                    2
                    + 0.75 * a
                    - 0.2 * b
                    + ug
                    + vh
                    + 0.18 * math.sin(i * 1.13)
                    + 0.04 * math.cos(i * 0.7)
                )
                groups.append([h, g] if kind == "crossed" else [g, g * 4 + h])
    return np.array(x), np.array(y), np.array(groups, dtype=np.int64)


fixtures = []
for kind in ["crossed", "nested"]:
    x, y, groups = design(kind)
    matrices = []
    colnames = []
    for j in range(2):
        ids = sorted(set(groups[:, j]))
        matrices.append([(groups[:, j, None] == np.array(ids)[None, :]).astype(float)])
        colnames.append([[str(v) for v in ids]])
    vc = VCSpec(["term0", "term1"], colnames, matrices)
    for reml in [False, True]:
        sm = MixedLM(
            y, x, groups=np.zeros(len(y)), exog_re=np.empty((len(y), 0)), exog_vc=vc
        ).fit(reml=reml, method="lbfgs", maxiter=2000)
        # Conditional plug-in GLS covariance, matching the native contract rather
        # than statsmodels' full observed-Hessian variance-parameter coupling.
        V = np.eye(len(y)) * sm.scale
        for variance, mat in zip(sm.vcomp, matrices):
            V += variance * (mat[0] @ mat[0].T)
        cov = np.linalg.inv(x.T @ np.linalg.solve(V, x))
        if args.check_native:
            native = mixed_effects(
                x, y, groups, reml=reml, tolerance=1e-10, max_evaluations=3000
            )
            assert native["state"] == "ready"
            np.testing.assert_allclose(
                native["beta"], sm.fe_params, rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                native["variances"], [sm.scale, *sm.vcomp], rtol=2e-3, atol=3e-5
            )
            np.testing.assert_allclose(native["covariance"], cov, rtol=2e-3, atol=3e-5)
            assert abs(native["diagnostics"]["log_likelihood"] - sm.llf) < 1e-4
        fixtures.append(
            {
                "name": kind + ("_reml" if reml else "_ml"),
                "reml": reml,
                "X": x.tolist(),
                "y": y.tolist(),
                "groups": groups.tolist(),
                "expected": {
                    "beta": sm.fe_params.tolist(),
                    "variances": [sm.scale, *sm.vcomp.tolist()],
                    "covariance": cov.tolist(),
                    "log_likelihood": float(sm.llf),
                },
            }
        )
args.output.write_text(
    json.dumps(
        {
            "schema_version": "n4m.mixed-effects-reference/1",
            "reference": {
                "package": "statsmodels",
                "version": statsmodels.__version__,
                "numpy": np.__version__,
                "fit": "MixedLM VCSpec, L-BFGS, maxiter=2000",
                "covariance": "conditional plug-in GLS inverse(X.T solve(V,X)), not full observed-Hessian parameter coupling",
            },
            "cases": fixtures,
        },
        indent=2,
    )
    + "\n"
)

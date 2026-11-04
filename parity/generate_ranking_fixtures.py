#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Generate independent reference numbers; optional test-only dependencies.

uv run --with numpy==2.5.4 --with scipy==1.18.1 --with trueskill==0.4.5 \
    --with openskill==6.2.0 parity/generate_ranking_fixtures.py

No reference implementation code is copied into the numerical core.
"""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import trueskill
from openskill.models import BradleyTerryFull
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import friedmanchisquare

cases = []
for scores, mask, name in [
    ([[1, 2]], [[1, 1]], "single_win"),
    ([[1, 1]], [[1, 1]], "single_tie"),
    ([[1, 2, 0], [0, 1, 2]], [[1, 1, 0], [0, 1, 1]], "transitive_bridge"),
    ([[1, 2, 3], [2, 1, 1], [1, 3, 2]], [[1] * 3] * 3, "three_candidates_with_ties"),
]:
    k = len(scores[0])
    env = trueskill.TrueSkill(
        mu=25, sigma=25 / 3, beta=25 / 6, tau=0, draw_probability=0.1, backend="scipy"
    )
    model = BradleyTerryFull(mu=25, sigma=25 / 3, beta=25 / 6, tau=0, kappa=1e-12)
    ts = [env.create_rating() for _ in range(k)]
    os = [model.rating() for _ in range(k)]
    pairs = []
    for row, present in zip(scores, mask, strict=True):
        for i in range(k):
            for j in range(i + 1, k):
                if present[i] and present[j]:
                    ranks = (
                        [0, 0]
                        if row[i] == row[j]
                        else ([0, 1] if row[i] < row[j] else [1, 0])
                    )
                    rated = env.rate([(ts[i],), (ts[j],)], ranks=ranks)
                    ts[i], ts[j] = rated[0][0], rated[1][0]
                    rated = model.rate([[os[i]], [os[j]]], ranks=ranks)
                    os[i], os[j] = rated[0][0], rated[1][0]
                    pairs.append(
                        (
                            i,
                            j,
                            0.5 if ranks[0] == ranks[1] else float(ranks[0] < ranks[1]),
                        )
                    )

    def objective(theta, pairs=pairs):
        value = 0.5 * np.dot(theta, theta)
        gradient = theta.copy()
        for i, j, win in pairs:
            d = theta[i] - theta[j]
            value += np.logaddexp(0, d) - win * d
            residual = expit(d) - win
            gradient[i] += residual
            gradient[j] -= residual
        return value, gradient

    bt = minimize(
        objective, np.zeros(k), jac=True, method="BFGS", options={"gtol": 1e-10}
    )
    cases.append(
        {
            "name": name,
            "scores": scores,
            "present": mask,
            "trueskill": [[r.mu, r.sigma] for r in ts],
            "openskill": [[r.mu, r.sigma] for r in os],
            "bradley_terry": bt.x.tolist(),
        }
    )

friedman_scores = np.array([[1, 1, 3], [1, 2, 2], [2, 1, 3]], dtype=float)
friedman = friedmanchisquare(*friedman_scores.T)
output = {
    "provenance": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "trueskill", "openskill")
    },
    "policy": "Replay rows then i<j pairs; lower scores win; ties exact; dynamics=0; L2=1.",
    "cases": cases,
    "friedman": {
        "scores": friedman_scores.tolist(),
        "statistic": float(friedman.statistic),
        "p_value": float(friedman.pvalue),
    },
}
path = Path(__file__).parent / "fixtures" / "ranking_reference_v1.json"
path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

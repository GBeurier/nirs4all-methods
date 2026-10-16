# SPDX-License-Identifier: CECILL-2.1
"""Generate parity/fixtures/cluster_matching_null_v1.json.

Test-only reference gate for four ABI 2.20 kernels used by cluster matching,
subsampling plans and Gaussian null models: n4m_linear_sum_assignment,
n4m_cluster_jaccard_matrix, n4m_random_permutations and
n4m_random_multivariate_normal, plus HDBSCAN on simulated Gaussian data.

Requires exactly numpy==2.4.6, scipy==1.18.1, scikit-learn==1.9.1 and
hdbscan==0.8.44 (all BSD-3-Clause); none of them enters libn4m.

- Assignment: scipy.optimize.linear_sum_assignment on square, wide, tall,
  tied, constant, forbidden (+inf) and maximized Jaccard matrices; invalid
  and infeasible matrices record the scipy ValueError.
- Jaccard: intersections from sklearn.metrics.cluster.contingency_matrix on
  the assigned elements, then |A∩B| / (|A| + |B| - |A∩B|).
- Normal draws: numpy.random.default_rng(seed).multivariate_normal(mean, cov,
  size=n, method="cholesky"), including seeds above 2**63.
- Permutations: successive numpy.random.default_rng(seed).permutation(n) draws
  from one generator (the 50 subsampling plans of 40 units with seed 7, among
  others).
- HDBSCAN on data mixing magnitudes inside the rule-13 domain, against
  scikit-learn HDBSCAN(metric="precomputed") on exact math.dist distances.
- Large-n adjusted Rand indices with labelings a_i = i mod ka and
  b_i = (i // ka + shift (i mod 3)) mod kb (exact-integer scikit-learn).
- HDBSCAN on Gaussian draws: the stable-tie scikit-learn reference of
  hdbscan_v1.py (same single-linkage path), and the stock scikit-learn ARI for
  information.

Usage: python parity/generators/cluster_matching_null_v1.py [output.json]
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import math
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
import sklearn.cluster._hdbscan.hdbscan as skh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdbscan_v1 import stable_single_linkage  # noqa: E402

PINNED = {"numpy": "2.4.6", "scipy": "1.18.1", "scikit-learn": "1.9.1", "hdbscan": "0.8.44"}
SEED = 20261014


def hexd(values) -> list[str]:
    return [struct.pack(">d", float(v)).hex() for v in np.ravel(np.asarray(values, dtype=np.float64))]


def assignment_cases(rng: np.random.Generator) -> list[dict]:
    a = rng.integers(0, 4, 40)
    b = np.where(rng.uniform(size=40) < 0.8, a, rng.integers(-1, 5, 40))
    jaccard = jaccard_reference(a, b, 4, 5)[0]
    forbidden = rng.uniform(0, 10, (4, 5))
    forbidden[0, :3] = np.inf
    forbidden[2, 4] = np.inf
    matrices = [
        ("square_uniform", rng.uniform(0, 1, (6, 6)), False),
        ("wide", rng.uniform(-5, 5, (3, 7)), False),
        ("tall_transposed", rng.uniform(-5, 5, (8, 4)), False),
        ("integer_ties", rng.integers(0, 3, (7, 7)).astype(float), False),
        ("constant_identity", np.full((5, 5), 2.5), False),
        ("tall_constant", np.full((5, 3), 1.0), True),
        ("forbidden_pairs", forbidden, False),
        ("jaccard_maximize", jaccard, True),
        ("single_cell", np.array([[3.0]]), True),
        ("nan_invalid", np.array([[1.0, np.nan], [0.0, 1.0]]), False),
        ("maximize_inf_invalid", np.array([[1.0, np.inf], [0.0, 1.0]]), True),
        ("infeasible", np.array([[np.inf, np.inf], [0.0, 1.0]]), False),
    ]
    cases = []
    for name, cost, maximize in matrices:
        case = {"name": name, "rows": cost.shape[0], "cols": cost.shape[1], "maximize": int(maximize),
                "cost_hex": hexd(cost)}
        try:
            rows, cols = linear_sum_assignment(cost, maximize=maximize)
            case.update({"error": None, "row_ind": rows.tolist(), "col_ind": cols.tolist(),
                         "total_hex": hexd([cost[rows, cols].sum()])})
        except ValueError as exc:
            case["error"] = "infeasible" if "infeasible" in str(exc) else "invalid"
        cases.append(case)
    return cases


def jaccard_reference(a: np.ndarray, b: np.ndarray, ka: int, kb: int):
    both = (a >= 0) & (b >= 0)
    inter = np.zeros((ka, kb), np.int64)
    if both.any():
        table = contingency_matrix(a[both], b[both])
        rows, cols = np.unique(a[both]), np.unique(b[both])
        inter[np.ix_(rows, cols)] = table
    size_a = np.array([(a == i).sum() for i in range(ka)], np.int64)
    size_b = np.array([(b == j).sum() for j in range(kb)], np.int64)
    union = size_a[:, None] + size_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return jac, inter, size_a, size_b


def jaccard_cases(rng: np.random.Generator) -> list[dict]:
    reference = np.repeat([0, 1, 2], [12, 9, 6])
    noisy = reference.copy()
    noisy[rng.choice(27, 6, replace=False)] = -1
    shifted = np.where(reference == 2, 0, reference + 1)
    shifted[:3] = 3
    split = np.r_[np.zeros(6, int), np.ones(6, int), np.full(9, 2), np.full(6, -1)]
    pairs = [
        ("relabelled_with_noise", reference, noisy, 3, 3),
        ("merged_and_new", reference, shifted, 3, 4),
        ("split_with_noise", reference, split, 3, 3),
        ("empty_cluster_slot", np.array([0, 0, -1, 1]), np.array([-1, -1, -1, 0]), 3, 2),
    ]
    cases = []
    for name, a, b, ka, kb in pairs:
        jac, inter, size_a, size_b = jaccard_reference(a, b, ka, kb)
        cases.append({"name": name, "labels_a": a.tolist(), "labels_b": b.tolist(),
                      "n_clusters_a": ka, "n_clusters_b": kb, "jaccard_hex": hexd(jac),
                      "intersections": inter.ravel().tolist(), "sizes_a": size_a.tolist(),
                      "sizes_b": size_b.tolist()})
    return cases


def normal_cases() -> list[dict]:
    full = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
    eig = np.array([9.0, 4.0, 1.5, 0.6, 0.2])
    diag = np.diag(eig + 1e-6 * eig.mean())
    specs = [
        ("full_covariance_seed_0", 0, np.array([1.0, -2.0, 0.5]), full, 7),
        ("full_covariance_seed_above_2_63", 2**63 + 7, np.zeros(3), full, 5),
        ("diagonal_ridge_seed_max", 2**64 - 1, np.zeros(5), diag, 6),
        ("univariate", 12345, np.array([3.0]), np.array([[0.25]]), 9),
    ]
    cases = []
    for name, seed, mean, cov, n in specs:
        draws = np.random.default_rng(seed).multivariate_normal(mean, cov, size=n, method="cholesky")
        cases.append({"name": name, "seed": str(seed), "d": len(mean), "n": n, "mean_hex": hexd(mean),
                      "cov_hex": hexd(cov), "draws_hex": hexd(draws)})
    return cases


def permutation_cases() -> list[dict]:
    cases = []
    for name, seed, n, b in (("subsample_plans_40_units", 7, 40, 50), ("seed_above_2_63", 2**63 + 1, 9, 4),
                             ("single_unit", 3, 1, 2), ("no_permutation", 5, 6, 0)):
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(n).tolist() for _ in range(b)]
        cases.append({"name": name, "seed": str(seed), "n": n, "n_permutations": b, "draws": perms})
    return cases


def multiscale_hdbscan_cases() -> list[dict]:
    """HDBSCAN on data mixing magnitudes inside the rule-13 domain.

    The domain is: nonzero |x| in [2^-256, 2^256] and max/min nonzero |x| <=
    2^64 within the matrix. The reference runs scikit-learn
    HDBSCAN(metric="precomputed") on exact math.dist distances. Every case
    has distinct pairwise distances (so the MST and partition do not depend on
    ties), except the review vector, whose expected partition does not depend
    on its tie order, and the equal distances to one far outlier.
    """
    rng = np.random.default_rng(20261015)
    blobs = np.vstack([rng.normal(0.0, 1.0, (8, 2)), rng.normal(30.0, 1.0, (8, 2))]) + 5.0
    specs = [
        ("review_1d_with_2e60", np.array([[0.0], [1.0], [3.0], [10.0], [11.0], [14.0], [2.0**60]]), 2, 2),
        ("blobs_at_2e-200", blobs * 2.0**-200, 4, 3),
        ("blobs_at_2e180", blobs * 2.0**180, 4, 3),
        ("blobs_with_outlier_at_2e40", np.vstack([blobs, [[2.0**40, -(2.0**40)]]]), 4, 3),
        ("columns_at_2e-200_and_2e-250", np.c_[blobs[:, :1] * 2.0**-200, blobs[:, 1:] * 2.0**-250], 4, 3),
    ]
    cases = []
    for name, x, mcs, ms in specs:
        nonzero = np.abs(x[x != 0])
        assert nonzero.min() >= 2.0**-256 and nonzero.max() <= 2.0**256, name
        assert nonzero.max() <= nonzero.min() * 2.0**64, name
        n = len(x)
        d = np.array([[math.dist(x[i], x[j]) for j in range(n)] for i in range(n)])
        off = d[np.triu_indices(n, 1)]
        near = off[off < 2.0**39]
        assert name.startswith("review") or len(np.unique(near)) == len(near), name
        model = HDBSCAN(metric="precomputed", min_cluster_size=mcs, min_samples=ms, copy=True).fit(d)
        if name.startswith("review"):
            assert model.labels_.tolist() == [0, 0, 0, 1, 1, 1, -1], model.labels_
        cases.append({"name": name, "rows": n, "cols": x.shape[1], "input_hex": hexd(x),
                      "min_cluster_size": mcs, "min_samples": ms,
                      "labels": model.labels_.astype(np.int64).tolist(),
                      "membership_hex": hexd(model.probabilities_)})
    return cases


def ari_cases() -> list[dict]:
    """Large-n ARI values where rounded pair-count products would cancel.

    scikit-learn adjusted_rand_score works on exact Python integers; the
    labelings are given by formulas so the fixture stays small.
    """
    specs = [("independent_1024x1024", 1 << 20, 1024, 1024, 0),
             ("nearly_independent_shifted", 1 << 20, 1000, 1009, 7),
             ("coarse_vs_fine", 3_000_000, 3, 999_983, 1)]
    cases = []
    for name, n, ka, kb, shift in specs:
        i = np.arange(n, dtype=np.int64)
        a = i % ka
        b = (i // ka + shift * (i % 3)) % kb
        cases.append({"name": name, "n": n, "ka": ka, "kb": kb, "shift": shift,
                      "ari_hex": hexd([adjusted_rand_score(a, b)])})
    return cases


def gaussian_hdbscan_cases() -> list[dict]:
    eig = np.array([6.0, 2.5, 1.0, 0.4, 0.1])
    cov = np.diag(eig + 1e-6 * eig.mean())
    cases = []
    for seed in (11, 2**40 + 3, 987654321):
        n = 80
        mcs = max(5, math.ceil(n / 20))
        X = np.random.default_rng(seed).multivariate_normal(np.zeros(len(eig)), cov, size=n, method="cholesky")
        labels, _ = skh.tree_to_labels(stable_single_linkage(X, mcs), mcs, "eom", False, 0.0, None)
        stock = HDBSCAN(min_cluster_size=mcs, min_samples=mcs, copy=True).fit(X).labels_
        cases.append({"name": f"gaussian_seed_{seed}", "seed": str(seed), "n": n, "d": len(eig),
                      "cov_hex": hexd(cov), "min_cluster_size": mcs, "min_samples": mcs,
                      "labels": labels.astype(np.int64).tolist(),
                      "n_clusters": int(labels.max() + 1) if (labels >= 0).any() else 0,
                      "stock_sklearn_ari": float(adjusted_rand_score(labels, stock))})
    return cases


def main() -> None:
    versions = {name: metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise SystemExit(f"pinned reference versions required: {PINNED}, found {versions}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    rng = np.random.default_rng(SEED)
    body = {
        "format": "n4m_cluster_matching_null_v1",
        "reference": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                      "hdbscan": versions["hdbscan"], "python": sys.version.split()[0], "seed": SEED,
                      "hdbscan_tie_rule": "MST edges stably sorted by weight (as hdbscan_v1)"},
        "encoding": "ieee754_binary64_be_hex",
        "cases": ([{"kind": "assignment", **c} for c in assignment_cases(rng)] +
                  [{"kind": "jaccard", **c} for c in jaccard_cases(rng)] +
                  [{"kind": "normal", **c} for c in normal_cases()] +
                  [{"kind": "permutations", **c} for c in permutation_cases()] +
                  [{"kind": "gaussian_hdbscan", **c} for c in gaussian_hdbscan_cases()] +
                  [{"kind": "multiscale_hdbscan", **c} for c in multiscale_hdbscan_cases()] +
                  [{"kind": "ari", **c} for c in ari_cases()]),
    }
    out = (Path(sys.argv[1]) if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "fixtures" / "cluster_matching_null_v1.json")
    out.write_text(json.dumps(body, indent=1) + "\n")
    for case in body["cases"]:
        if case["kind"] == "assignment":
            print(f"assignment {case['name']}: {case.get('col_ind', case['error'])}")
        if case["kind"] == "gaussian_hdbscan":
                print(f"{case['name']}: n_clusters={case['n_clusters']} noise={sum(v < 0 for v in case['labels'])}"
                  f" stock_sklearn_ari={case['stock_sklearn_ari']:.4f}")


if __name__ == "__main__":
    main()

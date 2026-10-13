# SPDX-License-Identifier: CECILL-2.1
"""Generate parity/fixtures/hdbscan_v1.json, the HDBSCAN reference gate.

Test-only. Requires exactly scikit-learn==1.9.1, hdbscan==0.8.44 and
numpy==2.4.6 (all BSD-3-Clause); none of them enters libn4m.

Reference path, identical to scikit-learn HDBSCAN(algorithm="kd_tree") except
for one documented point: the MST edges are sorted with a STABLE sort, so equal
mutual-reachability weights keep Prim insertion order. Stock scikit-learn uses
the default ``np.argsort`` there, whose tie order is unspecified and depends on
the CPU dispatch of NumPy; the stock partition is recorded per case as an
informational ``stock_sklearn_ari``. Labels and membership come from
scikit-learn ``tree_to_labels``; GLOSH outlier scores and cluster persistence
come from hdbscan 0.8.44 (``condense_tree``/``compute_stability``/
``get_clusters``/``outlier_scores``) applied to the SAME single-linkage tree,
whose labels and probabilities must agree with scikit-learn.

Usage: python parity/generators/hdbscan_v1.py [output.json]
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
import sklearn.cluster._hdbscan.hdbscan as skh
from hdbscan._hdbscan_tree import compute_stability, condense_tree, get_clusters, outlier_scores
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

PINNED = {"scikit-learn": "1.9.1", "hdbscan": "0.8.44", "numpy": "2.4.6"}
SEED = 20261013


def hexd(values: np.ndarray) -> list[str]:
    return [struct.pack(">d", float(v)).hex() for v in np.ravel(values)]


def stable_single_linkage(X: np.ndarray, min_samples: int) -> np.ndarray:
    core = skh.NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X, min_samples)[0][:, -1]
    metric = skh.DistanceMetric.get_metric("euclidean")
    mst = skh.mst_from_data_matrix(np.ascontiguousarray(X), np.ascontiguousarray(core), metric, 1.0)
    return skh.make_single_linkage(mst[np.argsort(mst["distance"], kind="stable")])


def reference(X: np.ndarray, mcs: int, ms: int, method: str, single: bool) -> dict:
    slt = stable_single_linkage(X, ms)
    labels, membership = skh.tree_to_labels(slt, mcs, method, single, 0.0, None)
    hierarchy = np.c_[slt["left_node"], slt["right_node"], slt["value"], slt["cluster_size"]].astype(np.float64)
    tree = condense_tree(hierarchy, mcs)
    stability = compute_stability(tree)
    c_labels, c_membership, persistence = get_clusters(tree, dict(stability), method, single, False, 0.0, 0)
    if not (np.array_equal(labels, c_labels) and np.allclose(membership, c_membership, rtol=1e-14, atol=0)):
        raise SystemExit("scikit-learn and hdbscan disagree on the same single-linkage tree")
    n = X.shape[0]
    clusters = tree[tree["child_size"] > 1]
    k = int(tree["parent"].max()) - n + 1
    parent = np.full(k, -1, np.int64)
    size = np.full(k, n, np.int64)
    birth = np.zeros(k)
    for row in clusters:
        node = int(row["child"]) - n
        parent[node] = int(row["parent"]) - n
        size[node] = int(row["child_size"])
        birth[node] = row["lambda_val"]
    death = np.array([tree["lambda_val"][tree["parent"] == n + node].max() for node in range(k)])
    selected = sorted(set(np.unique(labels[labels >= 0]).tolist()))
    label_of = np.full(k, -1, np.int64)
    for label in selected:
        # A selected cluster is the lowest common ancestor of its members' parents.
        common = None
        for i in np.nonzero(labels == label)[0]:
            node, chain = int(tree["parent"][tree["child"] == i][0]) - n, []
            while node >= 0:
                chain.append(node)
                node = int(parent[node])
            common = chain if common is None else [c for c in chain if c in common]
        label_of[common[0]] = label
    stock = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=method,
                    allow_single_cluster=single, copy=True).fit(X).labels_
    return {
        "labels": labels.astype(np.int64).tolist(),
        "membership_hex": hexd(membership),
        "outlier_scores_hex": hexd(outlier_scores(tree)),
        "persistence_hex": hexd(persistence),
        "n_clusters": len(selected),
        "tree_parent": parent.tolist(),
        "tree_size": size.tolist(),
        "tree_label": label_of.tolist(),
        "tree_lambda_birth_hex": hexd(birth),
        "tree_lambda_death_hex": hexd(death),
        "tree_stability_hex": hexd(np.array([stability[n + node] for node in range(k)])),
        "stock_sklearn_ari": float(adjusted_rand_score(labels, stock)),
        "stock_sklearn_same_noise": bool(np.array_equal(labels < 0, stock < 0)),
    }


def datasets(rng: np.random.Generator) -> list[tuple[str, np.ndarray, int, int, str, bool]]:
    blobs = np.vstack([rng.normal(c, 0.4, (40, 2)) for c in ([0, 0], [6, 0], [0, 6])])
    dense = rng.normal([0, 0], 0.25, (60, 2))
    sparse = rng.normal([5, 5], 1.6, (40, 2))
    varying = np.vstack([dense, sparse, rng.uniform(-3, 9, (10, 2))])
    noise = rng.uniform(-1, 1, (100, 3))
    t = rng.uniform(0, 1, 120)
    continuum = np.c_[t, t**2] + rng.normal(0, 0.01, (120, 2))
    grid = np.round(rng.normal(0, 1.5, (70, 2)))
    ties = np.vstack([grid, np.repeat(grid[:4], 5, axis=0)])
    return [
        ("separated_blobs", blobs, 10, 10, "eom", False),
        ("varying_density_eom", varying, 8, 5, "eom", False),
        ("varying_density_leaf", varying, 8, 5, "leaf", False),
        ("pure_noise", noise, 10, 10, "eom", False),
        ("pure_noise_single_cluster", noise, 10, 10, "eom", True),
        ("continuum", continuum, 10, 5, "eom", False),
        ("duplicates_ties", ties, 5, 3, "eom", False),
        ("min_samples_gt_min_cluster_size_leaf", blobs, 5, 15, "leaf", True),
    ]


def main() -> None:
    versions = {name: metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise SystemExit(f"pinned reference versions required: {PINNED}, found {versions}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    rng = np.random.default_rng(SEED)
    cases = []
    for name, X, mcs, ms, method, single in datasets(rng):
        X = np.ascontiguousarray(X, dtype=np.float64)
        case = {
            "name": name,
            "params": {"min_cluster_size": mcs, "min_samples": ms,
                       "selection_method": 0 if method == "eom" else 1,
                       "allow_single_cluster": int(single)},
            "rows": X.shape[0], "cols": X.shape[1], "input_hex": hexd(X),
        }
        case.update(reference(X, mcs, ms, method, single))
        cases.append(case)
    body = {
        "format": "n4m_hdbscan_v1",
        "reference": {
            "scikit-learn": sklearn.__version__, "hdbscan": versions["hdbscan"], "numpy": np.__version__,
            "python": sys.version.split()[0], "seed": SEED,
            "tie_rule": "MST edges stably sorted by weight (Prim insertion order on ties)",
            "min_samples": "scikit-learn convention, includes the row itself",
        },
        "encoding": "ieee754_binary64_be_hex",
        "cases": cases,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "fixtures" / "hdbscan_v1.json"
    out.write_text(json.dumps(body, indent=1) + "\n")
    for case in cases:
        print(f"{case['name']}: n_clusters={case['n_clusters']} noise={sum(l < 0 for l in case['labels'])}"
              f" stock_sklearn_ari={case['stock_sklearn_ari']:.4f}")


if __name__ == "__main__":
    main()

# HDBSCAN for Preflight: reuse assessment and reference gate

Status: ABI 2.19.0 (unreleased), `n4m_cluster_hdbscan` and
`n4m_preflight_adjusted_rand_index` in `n4m/metrics.h`. Requested by
nirs4all-preflight lot PF-05 (clustering capability).

## Reuse assessment (2026-10-13)

Methods is C++17 with zero mandatory dependencies, so a Rust crate cannot be linked into
it. Candidates were checked for semantic fidelity to the reference (scikit-learn
`HDBSCAN` / `hdbscan`), maintenance, dependencies, WASM fit and licence.

| Candidate | Licence | Finding |
| --- | --- | --- |
| scikit-learn 1.9.1 `sklearn/cluster/_hdbscan` (Cython) | BSD-3-Clause | This is the reference itself. Prim on implicit mutual reachability, condensed tree, EOM/leaf, `allow_single_cluster` and membership are all available. It has no outlier scores. It is about 600 lines of Cython that map directly to C++. **Chosen**, as a port. |
| scikit-learn-contrib/hdbscan 0.8.44 | BSD-3-Clause | The same tree algorithms with GLOSH `outlier_scores_` and `cluster_persistence_`. **Ported only for these two outputs**. Its Boruvka/KD-tree MST is not needed at Preflight sizes. |
| HDBSCAN-CPP (rohanmohapatra) | MIT | Inactive since 2019. It copies a dense `vector<vector<double>>` distance matrix, has EOM only, no leaf mode and no `allow_single_cluster`, and uses its own membership formula. It is not reference-compatible. |
| `petal-clustering` 0.14 | Apache-2.0 | Rust, so it cannot be linked. It needs ndarray, rayon and petal-neighbors, has EOM only, and has no leaf mode, membership or `allow_single_cluster`. |
| `hdbscan` crate 0.12 | MIT/Apache-2.0 | Rust, so it cannot be linked. It has no leaf mode, membership or outlier scores. |
| R `dbscan` (`hdbscan.cpp`) | GPL-3.0 | Tied to Rcpp `List` objects throughout. Its semantics are its own (`minPts`), not scikit-learn's. |
| ELKI | AGPL-3.0 | Java. |
| cuML | Apache-2.0 | CUDA. |
| Methods building blocks | — | Methods had no MST, union-find or ARI. Its PCA, neighbour and LOF code is task-specific. |

The decision is to port the BSD-3 reference code into
`cpp/src/core/clustering/_vendored/` with attribution, following the convention of
`cpp/src/core/filters/_vendored`. See `LICENSE.scikit_learn.txt` there,
`THIRD_PARTY_LICENSES.md` and `NOTICE.md`. No change to the Methods licence is required.
The ARI is new native code, about 30 lines.

## Frozen semantics

- Metric: Euclidean, with `alpha = 1`. Not supported: `cluster_selection_epsilon`,
  `max_cluster_size` and other metrics. `min_cluster_size >= 2` and `1 <= min_samples <= n`.
- Core distance: the distance to the `min_samples`-th nearest row, **counting the row
  itself**. This is the scikit-learn convention; contrib `hdbscan` uses `min_samples - 1`.
  It is computed by brute force in O(n²p) time with O(n) memory.
- MST: Prim's algorithm on the implicit mutual-reachability graph, exactly as in
  `mst_from_data_matrix`. It starts at row 0, and on equal weights the first index wins.
- Tie rule: the MST edges are sorted with a **stable** sort, so equal weights keep Prim
  insertion order. This is the only intentional difference from scikit-learn, which uses
  the default `np.argsort`. That order is unspecified and depends on NumPy's CPU dispatch.
- Condensed tree: lambda = 1/distance (+inf at distance 0). Stability is the excess of
  mass, `sum (lambda_p - lambda_birth) * size`. EOM replaces a node by its children only
  when their summed stability is strictly larger. In leaf mode, a tree without splits
  selects nothing.
- Outputs: labels (noise -1), numbered by increasing condensed-node index. Membership is
  scikit-learn `probabilities_`, a relative membership strength and **not a probability**.
  Outlier scores are GLOSH. Persistence is contrib `cluster_persistence_`, which is 1 when
  the maximum lambda is infinite. The condensed cluster tree reports parent, size, label,
  lambda birth/death and stability. The result is deterministic for a given row order;
  callers that need order invariance must fix a canonical row order.

## Tie order matters, including in the reference

Mutual-reachability weights tie structurally, even on continuous data, because an edge
weight is often a core distance. The order in which equal edges are merged decides which
cluster a boundary point falls out of, and sometimes whether a cluster exists at all.
We ran 500 random cases (blobs, uniform, rounded, duplicated and continuum data, with
random parameters). Stock scikit-learn 1.9.1 on NumPy 2.4.6 (AVX2) gave the same
partition as the stable tie rule in 60–85 % of cases, depending on data kind. On heavily
tied data the ARI fell to 0. Stock scikit-learn also changes its result under a
permutation of the input rows: on the `pure_noise` fixture, one row order yields two
clusters and two other orders yield no cluster. A frozen, documented tie rule is
therefore necessary. Downstream reports must treat partitions as order- and
tie-sensitive, and subsample stability is the instrument for this.

## Reference gate

`parity/generators/hdbscan_v1.py` (test-only; pins scikit-learn 1.9.1, hdbscan 0.8.44 and
numpy 2.4.6) writes `parity/fixtures/hdbscan_v1.json`. Its SHA-256 at generation is
`294457336e820754b5047851760ea0d02391c1c552cd9420cc80223179297cd4`. The reference is
scikit-learn with only the stable MST edge sort changed. Contrib `hdbscan` then runs on
the **same** single-linkage tree, and the generator refuses to write the fixture if the
two projects disagree on labels or membership. The eight cases are:

- separated blobs
- varying density (EOM and leaf)
- pure noise (with and without `allow_single_cluster`)
- a continuum
- duplicates/ties
- `min_samples` > `min_cluster_size` with leaf mode and `allow_single_cluster`

Each case records the stock scikit-learn ARI for information.

`n4m_hdbscan_tests` compares partitions up to label permutation, with an identical noise
set and a bijective label map. It also compares membership, GLOSH, persistence and
condensed-tree values within 1e-12 (relative), checks the exact tree structure, and checks
that row-major, column-major and strided views give identical results. Before freezing,
a fuzz run of 500 random cases matched exactly on labels, membership, GLOSH, persistence
and ARI.

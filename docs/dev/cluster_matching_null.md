# Cluster matching and Gaussian null kernels: reuse assessment and reference gate

Status: ABI 2.20.0 (unreleased), `n4m_cluster_jaccard_matrix`,
`n4m_linear_sum_assignment`, `n4m_random_permutations` and
`n4m_random_multivariate_normal` in `n4m/metrics.h`.
Requested by nirs4all-preflight lot PF-05 (cluster appearance frequency and the
unimodal Gaussian null model). The kernels are generic; the matching threshold, the
covariance regularization and the null protocol belong to the caller.

## Reuse assessment (2026-10-13)

| Need | Candidates | Decision |
| --- | --- | --- |
| One-to-one assignment | SciPy 1.18.1 `scipy/optimize/rectangular_lsap/rectangular_lsap.cpp` (BSD-3-Clause, C++, about 250 lines, Crouse's shortest augmenting path) is the reference itself. `gatagat/lap` LAPJV (BSD-2-Clause) solves the same problem with another tie order. dlib `max_cost_assignment` (Boost) is integer-only. The Rust `pathfinding` Kuhn–Munkres (MIT/Apache-2.0) cannot be linked into C++. Methods had no assignment solver. | **Port SciPy** into `cpp/src/core/optimization/_vendored/`, keeping its control flow, tie rules and BSD-3 header (`LICENSE.scipy.txt`, `THIRD_PARTY_LICENSES.md`, `NOTICE.md`). |
| Jaccard between clusters | scikit-learn `contingency_matrix` (Python); no C/C++ library worth a dependency. | New native code (about 30 lines); scikit-learn is the test reference. |
| Subsampling plans | Methods PCG64 already reproduced NumPy `Generator.permutation` (`n4m_pcg64_engine_permutation`, used by the factor-association test). | **Expose** it as `n4m_random_permutations`: successive permutations from one seeded stream. |
| Gaussian draws | Methods PCG64 with the NumPy Ziggurat (`n4m_pcg64_engine_standard_normal_fill`, bit-identical to `default_rng(seed).standard_normal`, frozen by `test_rng_pcg64.cpp`). The Cholesky routines in `gpr_pls.cpp`, `gp.cpp` and `emsc.c` are file-local to those methods. | **Reuse** the PCG64 Ziggurat stream; a 15-line lower Cholesky lives with the kernel. No new normal sampler. |

## Frozen semantics

- Domains of validity are stated in each header: Gaussian draws factor a scaled
  covariance; Jaccard refuses `n > 2^53`; assignment is exact (see below). `n4m_cluster_robustness_tests` checks
  scaling, translation and constant-column invariance for these kernels, HDBSCAN and
  ARI.
- `n4m_cluster_jaccard_matrix`: negative labels are unassigned and belong to no cluster;
  labels must be below the declared cluster counts. `J_ij = |A_i ∩ B_j| / (|A_i| + |B_j| -
  |A_i ∩ B_j|)`, NaN when both clusters are empty. Callers restrict both labelings to the
  elements they share before the call.
- `n4m_linear_sum_assignment`: the algorithm and tie rules of
  `scipy.optimize.linear_sum_assignment`, evaluated in exact arithmetic (duals and path
  costs are floating-point expansions, `cpp/src/core/common/expansion.hpp`), so the
  result is an exact optimum. On random 2–4 × 2–6 matrices with signed costs whose
  exponents span ±60 to ±600, SciPy's double version is suboptimal in 2–9 % of cases;
  `n4m_cluster_robustness_tests` and the JS/WASM test check the exact optimum by
  enumeration with independent exact sums (2240-bit integers, BigInt). Costs are never
  centred; they are scaled by the smallest power of two that keeps the sums finite, and
  refused if that scaling is not exact. A tall
  matrix is solved on its transpose and the pairs are sorted by row; `maximize` negates
  the costs; `+inf` forbids a pair; NaN or `-inf` (after negation) is invalid; a missing
  feasible assignment is a numerical failure. Ties: in each augmenting-path search the
  columns are scanned from the last to the first, a strictly smaller reduced cost wins,
  and on equal reduced cost an unassigned column wins. A constant matrix therefore gives
  the identity assignment.
- `n4m_random_permutations`: one `PCG64(seed)` stream; row `b` of the output is the
  `b`-th `rng.permutation(n)` (Fisher–Yates from the last position over NumPy
  `random_interval`).
- `n4m_random_multivariate_normal`: `z` holds `n × d` standard normals from
  `PCG64(seed)` (NumPy `SeedSequence` expansion) in row-major order, `L` is the lower
  Cholesky factor of the lower triangle of `cov`, and `x_ij = sum_k z_ik L_jk + mean_j`.
  This is NumPy `Generator.multivariate_normal(mean, cov, size=n, method="cholesky")`;
  NumPy's default `method="svd"` uses another factor and gives other draws. A covariance
  that is not numerically positive definite fails instead of being repaired. Storage
  and cost are checked before any allocation, independently of `n`: `d·d` and `n·d`
  at most 5e7 values, `d^3/6` and `n·d·d` at most 2.5e9 operations.

## Reference gate

`parity/generators/cluster_matching_null_v1.py` (test-only; pins NumPy 2.4.6, SciPy
1.18.1, scikit-learn 1.9.1 and hdbscan 0.8.44) writes
`parity/fixtures/cluster_matching_null_v1.json`. Its SHA-256 at generation is
`ff6288f3df873b7a0589b7b25835c242db2afe6d08afeba6c74597b4eed13b27`. The cases are:

- twelve assignment matrices: square, wide, tall (transposed path), small-integer ties,
  constant (identity), tall constant maximized, forbidden `+inf` pairs, a maximized
  Jaccard matrix, a single cell, and the NaN, maximized `+inf` and infeasible errors;
- four Jaccard pairs with noise, merged and new clusters, split clusters and an empty
  cluster slot, against scikit-learn `contingency_matrix`;
- four permutation streams, including 50 permutations of 40 units, a seed above 2^63,
  a single unit and zero permutations;
- four NumPy normal draws, including a full 3×3 covariance, seeds above 2^63 and
  `2^64 - 1`, a ridge-regularized diagonal covariance and a univariate case;
- three Gaussian datasets (n = 80, d = 5, `min_cluster_size = min_samples = 5`) drawn by
  NumPy and clustered by the stable-tie scikit-learn reference of `hdbscan_v1.py`. One
  yields three clusters with 59 noise points and two yield none; the stock scikit-learn
  ARI is recorded for information (1, 1 and 0: stock scikit-learn finds clusters in the
  third dataset because of its unspecified MST tie order).

Three large-n adjusted Rand indices (up to 3 million elements, values near 0) are
checked within 1e-14 relative against exact-integer scikit-learn. Five multi-magnitude HDBSCAN cases (see `docs/dev/hdbscan.md`) use scikit-learn
`HDBSCAN(metric="precomputed")` on exact `math.dist` distances.

`n4m_cluster_matching_null_tests` checks the assignments, Jaccard values and permutations
exactly, the draws within 1e-12 relative, the HDBSCAN partitions of the Methods draws up to label
permutation, column-major cost storage, the lower-triangle-only covariance read, the
argument errors, and the review's overlapping 65536 × 65536 covariance view with zero
draws, which is refused before any allocation (natively and in WASM). The JS/WASM test replays the same fixture through
`linearSumAssignment`, `clusterJaccardMatrix`, `randomPermutations`,
`randomMultivariateNormal` and `clusterHdbscan`.

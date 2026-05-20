# Sample-partitioning splitters (Phase 11)

Phase 11 introduces the `n4m_split_*` ABI category — nine sample-partitioning
operators that produce `(train_idx[], test_idx[])` integer arrays.

| Operator                       | Public symbol prefix                            | Public type                                       |
|--------------------------------|-------------------------------------------------|---------------------------------------------------|
| KennardStone                   | `n4m_split_kennard_stone_*`                     | `n4m_split_kennard_stone_handle_t`                |
| SPXY                           | `n4m_split_spxy_*`                              | `n4m_split_spxy_handle_t`                         |
| SPXYFold                       | `n4m_split_spxy_fold_*`                         | `n4m_split_spxy_fold_handle_t`                    |
| SPXYGFold                      | `n4m_split_spxy_g_fold_*`                       | `n4m_split_spxy_g_fold_handle_t`                  |
| KMeans                         | `n4m_split_kmeans_*`                            | `n4m_split_kmeans_handle_t`                       |
| KBinsStratified                | `n4m_split_kbins_stratified_*`                  | `n4m_split_kbins_stratified_handle_t`             |
| BinnedStratifiedGroupKFold     | `n4m_split_binned_strat_group_kfold_*`          | `n4m_split_binned_strat_group_kfold_handle_t`     |
| SystematicCircular             | `n4m_split_systematic_circular_*`               | `n4m_split_systematic_circular_handle_t`          |
| SPlit (data twinning)          | `n4m_split_split_splitter_*`                    | `n4m_split_split_splitter_handle_t`               |

All operators share the result type:

```c
typedef struct n4m_split_result_t {
    int64_t* train_idx;    /* length n_train */
    int64_t  n_train;
    int64_t* test_idx;     /* length n_test  */
    int64_t  n_test;
    void*    _owner;       /* opaque allocation handle */
} n4m_split_result_t;

N4M_API void n4m_split_result_destroy(n4m_split_result_t* r);
```

Buffers are owned by the splitter side; the caller must invoke
`n4m_split_result_destroy` exactly once per filled result. Calling
`_destroy` on an empty result is safe.

## Algorithms

### KennardStone

`max-min` Euclidean distance on raw X. Reference:
`nirs4all.operators.splitters.KennardStoneSplitter` with the default metric
(Euclidean) and no PCA reduction.

1. Compute the full N × N Euclidean distance matrix.
2. Pick the two globally farthest points (`np.argmax` + `unravel_index`).
3. Iteratively add the sample maximising the minimum distance to the
   already-selected set.

### SPXY

`D = D_X / max(D_X) + D_Y / max(D_Y)`, then max-min selection. Reference:
`nirs4all.operators.splitters.SPXYSplitter`. Same algorithm as KennardStone
on the joint X-Y distance matrix.

### SPXYFold

K-fold alternating max-min assignment. Reference:
`nirs4all.operators.splitters.SPXYFold._assign_to_folds`.

1. Compute D (X-only, SPXY, or SPXY-hamming depending on `y_metric`).
2. Initialise the K folds with the K samples farthest from the dataset
   centroid (largest mean-row-distance).
3. Iterate over folds; for each fold pick the unassigned sample with the
   largest minimum distance to that fold's current members. Skip folds at
   capacity `ceil(N / K)`.

`y_metric` codes: `0` = X-only, `1` = euclidean Y, `2` = hamming Y.

### SPXYGFold

Group-aware K-fold. Aggregates samples into per-group representatives
(mean or median), runs SPXYFold on the group distance matrix, then
expands per-group fold labels back to per-sample.

`aggregation` codes: `0` = mean, `1` = median.

### KMeans

Vendored k-means++ initialisation + Lloyd iterations, PCG64-seeded.

1. `k = n_train` clusters.
2. Pick the first centroid by `floor(uniform01 * N)`.
3. Subsequent centroids: weighted-by-squared-distance sampling.
4. Lloyd iterations until convergence (or `max_iter`).
5. For each centroid, pick the sample with the minimum Euclidean distance.
6. Deduplicate (sort ascending). Train = the sorted unique set of
   per-centroid argmins. Test = complement.

### KBinsStratified

Bin Y, shuffle within each bin (Fisher-Yates with PCG64), take a fixed
fraction for the test set. Bin strategies: `0` = uniform (equal-width
bins from `y.min()` to `y.max()`), `1` = quantile (equal-frequency
bins, ranked by Y).

### BinnedStratifiedGroupKFold

Bin Y, label each group by the bin of its first sample, then round-robin
assign groups to folds within each bin stratum (optionally shuffling
groups within each bin first). Constraint: `n_groups >= n_splits`.

Round-robin may leave high-index folds empty when `n_bins * n_groups`
distributes unevenly. Callers must size `n_splits` accordingly (see the
smoke test for a canonical sizing).

### SystematicCircular

1. `ordered = argsort(y, kind='stable')`.
2. `offset = floor(uniform01 * N)` drawn from PCG64.
3. `rotated = np.roll(ordered, offset)`.
4. Step `s = N / n_train`; train = `rotated[round(s * i)]` for
   `i ∈ [0, n_train)`. Test = complement (in rotation order).

Output keeps the rotation order (NOT sorted) to mirror the Python
reference.

### SPlit (data twinning)

Bit-exact equivalent of `nirs4all.operators.splitters._twin` with the
starting index drawn from PCG64 (instead of `np.random.randint`).
`test_size` → `r = floor(1 / test_size)`. The smaller "twin" becomes
the test set.

## Parity

Integer index arrays must be byte-equal between the C engine and the
reference NumPy implementation in
`parity/python_generator/scripts/generate_phase11_fixtures.py`. The
reference is deliberately re-implemented in pure NumPy (not delegated to
sklearn / nirs4all) so the seeded RNG paths and the simplified algorithm
variants stay aligned across the two implementations.

Fixtures live under `parity/fixtures/split_*_v1.json`.

## ABI

- 31 new exports under the `n4m_split_*` prefix.
- 1 new struct: `n4m_split_result_t`.
- 1 new struct destructor: `n4m_split_result_destroy`.
- 9 new opaque handle types: `n4m_split_<op>_handle_t`.

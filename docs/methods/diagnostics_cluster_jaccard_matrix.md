# `jaccard_matrix` — n4m.cluster.jaccard_matrix

_Namespace_: **`n4m.cluster`** · _Fully-qualified_: `n4m.cluster.jaccard_matrix` · _Catalog id_: `diagnostics.cluster_jaccard_matrix`

## API surface

**C ABI (ABI 2):** [`n4m_cluster_jaccard_matrix`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L153). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Jaccard (1912), *The distribution of the flora in the alpine zone*, New Phytologist 11(2), 37–50, https://doi.org/10.1111/j.1469-8137.1912.tb05611.x; Hennig (2007), *Cluster-wise assessment of cluster stability*, Computational Statistics & Data Analysis 52(1), 258–271, https://doi.org/10.1016/j.csda.2006.11.025.

### Mathematical principle

For two labelings of the same $n$ elements, unassigned (negative) labels belong to no cluster. The kernel counts $|A_i|$, $|B_j|$ and $|A_i\cap B_j|$ and returns $J_{ij}=|A_i\cap B_j|/(|A_i|+|B_j|-|A_i\cap B_j|)$ for every cluster pair, NaN when both clusters are empty.

### Appropriate uses

Matching the clusters of a resampled or perturbed partition to a reference partition, for example to count how often each reference cluster reappears under subsampling.

### Limits and validation

The Jaccard index ignores unassigned elements and cluster shape; the caller restricts both labelings to the elements they share. Matching and thresholds are the caller's protocol (see `n4m_linear_sum_assignment`). The matrix is limited to 100 million cells.

### Implementation

The ABI-2 symbol `n4m_cluster_jaccard_matrix` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `clusterJaccardMatrix`. `parity/generators/cluster_matching_null_v1.py` writes scikit-learn 1.9.1 `contingency_matrix` references to `parity/fixtures/cluster_matching_null_v1.json`, which `cpp/tests/test_cluster_matching_null.cpp` checks exactly.

### Sources and provenance

https://doi.org/10.1111/j.1469-8137.1912.tb05611.x; https://doi.org/10.1016/j.csda.2006.11.025; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/cluster_matching_null_v1.json

## Catalog note

Cluster overlap between two partitions of the same elements: intersection counts, cluster sizes and Jaccard indices |A_i ∩ B_j| / |A_i ∪ B_j|, with negative labels unassigned. Reference gate: parity/generators/cluster_matching_null_v1.py (scikit-learn 1.9.1 contingency_matrix) -> parity/fixtures/cluster_matching_null_v1.json, cpp/tests/test_cluster_matching_null.cpp.


_See also_: [methods index](index.md).
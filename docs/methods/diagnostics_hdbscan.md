# `hdbscan` — n4m.cluster.hdbscan

_Namespace_: **`n4m.cluster`** · _Fully-qualified_: `n4m.cluster.hdbscan` · _Catalog id_: `diagnostics.hdbscan`

## API surface

**C ABI (ABI 2):** [`n4m_cluster_hdbscan`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L122). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Campello, Moulavi & Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD 2013, LNCS 7819, 160–172, https://doi.org/10.1007/978-3-642-37456-2_14; McInnes & Healy (2017), *Accelerated Hierarchical Density Based Clustering*, ICDMW 2017, 33–42, https://doi.org/10.1109/ICDMW.2017.12.

### Mathematical principle

The core distance of a row is the Euclidean distance to its `min_samples`-th nearest row, counting the row itself. The mutual-reachability distance is $d_m(a,b)=\max(\mathrm{core}(a),\mathrm{core}(b),d(a,b))$. A minimum spanning tree over $d_m$ gives a single-linkage hierarchy. It is condensed with `min_cluster_size`, and clusters are selected by excess of mass (EOM) or as leaves. The result has labels (-1 for noise), membership strengths, GLOSH outlier scores, cluster persistence and the condensed tree.

### Appropriate uses

Exploratory detection of dense sample groups and outliers in spectral or score spaces, without fixing the number of clusters in advance.

### Limits and validation

Results depend on `min_cluster_size`, `min_samples` and the metric, here Euclidean on the supplied features. Membership is a relative strength, not a probability. Cost is quadratic in rows, and workloads above 2.5e9 distance products are refused. Equal-weight MST edges keep Prim insertion order, which scikit-learn leaves unspecified.

### Implementation

The ABI-2 symbol `n4m_cluster_hdbscan` in `n4m/metrics.h` is a C++ port of scikit-learn 1.9.1 `sklearn.cluster.HDBSCAN`, with GLOSH scores and persistence from scikit-learn-contrib/hdbscan 0.8.44 (BSD-3-Clause), in `cpp/src/core/clustering/_vendored/hdbscan.cpp`. `parity/generators/hdbscan_v1.py` writes the reference fixture `parity/fixtures/hdbscan_v1.json`, which `cpp/tests/test_hdbscan.cpp` checks. JS/WASM wraps it as `clusterHdbscan`.

### Sources and provenance

https://doi.org/10.1007/978-3-642-37456-2_14; https://doi.org/10.1109/ICDMW.2017.12; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/clustering/_vendored/hdbscan.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/hdbscan_v1.json; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/dev/hdbscan.md

## Catalog note

Euclidean HDBSCAN* ported from scikit-learn 1.9.1 sklearn.cluster.HDBSCAN, with GLOSH outlier scores and cluster persistence from scikit-learn-contrib/hdbscan 0.8.44 (BSD-3-Clause, cpp/src/core/clustering/_vendored). Reference gate: parity/generators/hdbscan_v1.py -> parity/fixtures/hdbscan_v1.json, cpp/tests/test_hdbscan.cpp; reuse assessment in docs/dev/hdbscan.md.


_See also_: [methods index](index.md).
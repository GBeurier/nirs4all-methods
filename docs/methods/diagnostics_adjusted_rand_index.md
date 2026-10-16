# `adjusted_rand_index` — n4m.metrics.clustering.adjusted_rand_index

_Namespace_: **`n4m.metrics.clustering`** · _Fully-qualified_: `n4m.metrics.clustering.adjusted_rand_index` · _Catalog id_: `diagnostics.adjusted_rand_index`

## API surface

**C ABI (ABI 2):** [`n4m_preflight_adjusted_rand_index`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L147). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Hubert & Arabie (1985), *Comparing partitions*, Journal of Classification 2, 193–218, https://doi.org/10.1007/BF01908075.

### Mathematical principle

From the contingency table $n_{ij}$ of two labelings with row sums $a_i$ and column sums $b_j$, $ARI=(\sum_{ij}\binom{n_{ij}}{2}-E)/(\tfrac12[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}]-E)$, with $E=\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}/\binom{n}{2}$. It is computed from pair-confusion counts, as scikit-learn `adjusted_rand_score`.

### Appropriate uses

Measuring the agreement of two clusterings of the same rows, for example HDBSCAN partitions on subsamples when assessing cluster stability.

### Limits and validation

Every label value, including -1, is an ordinary class: callers must filter noise first. The result is NaN for fewer than two rows or for identical trivial partitions (one class or all singletons), where scikit-learn returns 1.0 by convention. ARI can be negative.

### Implementation

The ABI-2 symbol `n4m_preflight_adjusted_rand_index` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `preflightAdjustedRandIndex`. scikit-learn 1.9.1 reference values are pinned in `cpp/tests/test_hdbscan.cpp`.

### Sources and provenance

https://doi.org/10.1007/BF01908075; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_hdbscan.cpp

## Catalog note

Hubert-Arabie adjusted Rand index between two int64 label vectors, computed like scikit-learn adjusted_rand_score from pair-confusion counts; NaN (not 1.0) for n < 2 or identical trivial partitions.


_See also_: [methods index](index.md).
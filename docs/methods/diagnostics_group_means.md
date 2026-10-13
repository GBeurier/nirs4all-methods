# `group_means` — n4m.stats.group_means

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.group_means` · _Catalog id_: `diagnostics.group_means`

## API surface

**C ABI (ABI 2):** [`n4m_stats_group_means`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L382). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Arithmetic mean of replicate measurements; Welford (1962), *Note on a method for calculating corrected sums of squares and products*, Technometrics 4, 419–420, https://doi.org/10.1080/00401706.1962.10490022.

### Mathematical principle

For each group $g$ and column $j$ the kernel returns $\bar x_{gj}=\frac1{n_g}\sum_{i:g(i)=g}x_{ij}$, with equal weight per row. It updates a running mean $m\leftarrow m+(x/s_j-m)/r$ in extended precision on values divided by the column's largest magnitude $s_j$, then multiplies by $s_j$, so finite inputs never overflow.

### Appropriate uses

Aggregating homogeneous technical repeats of one physical sample (same instrument and campaign) into one row before fitting or validation, so repeats never weigh as independent samples.

### Limits and validation

Every group needs at least one row; codes must lie in $[0, n_{groups})$. Nonfinite values are refused (no mask: callers select complete rows). The mean is not a robust location estimate and does not reconcile repeats from different contexts.

### Implementation

The ABI-2 symbol `n4m_stats_group_means` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsGroupMeans`. `cpp/tests/test_group_means.cpp` checks hand-computed means, column-major strides, extreme magnitudes where a naive sum overflows, and refusals.

### Sources and provenance

https://doi.org/10.1080/00401706.1962.10490022; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_group_means.cpp

## Catalog note

Equal-weight per-group column means (scaled running means in extended precision), used to aggregate homogeneous technical repeats before fitting. Hand-computed and extreme-magnitude checks in cpp/tests/test_group_means.cpp.


_See also_: [methods index](index.md).
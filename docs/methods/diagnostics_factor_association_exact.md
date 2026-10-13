# `factor_association_exact` — n4m.metrics.statistics.factor_association_exact

_Namespace_: **`n4m.metrics.statistics`** · _Fully-qualified_: `n4m.metrics.statistics.factor_association_exact` · _Catalog id_: `diagnostics.factor_association_exact`

## API surface

**C ABI (ABI 2):** [`n4m_stats_factor_association_exact`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L265). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper; exhaustive permutation tests follow Fisher (1935), *The Design of Experiments*, Oliver & Boyd, and Good (2005), *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, Springer, https://doi.org/10.1007/b138696.

### Mathematical principle

Under uniform permutation of the target over $n$ exchangeable units, each distinct arrangement of a multiset with multiplicities $m_k$ has probability $1/N$, $N=n!/\prod_k m_k!$. The kernel enumerates the side (factor levels or target values) with the smaller $N$ and evaluates eta² or Cramér's V² on each arrangement. It returns $N$, the exact p-value $\#\{T\ge T_{obs}-\sqrt{\epsilon}\}/N$, the minimal attainable exact p-value $\#\{T\ge T_{max}-\sqrt{\epsilon}\}/N$ and $T_{max}$.

### Appropriate uses

Deciding whether a small permutation space can support a significance threshold after multiplicity correction: the minimal attainable p-value, not $1/N$, is the resolution of the test.

### Limits and validation

Exhaustive enumeration is limited to $N\le 10^7$ and to 250 million operations; larger spaces return `N4M_ERR_UNSUPPORTED`. Exchangeability of the units is assumed, as one unrestricted block.

### Implementation

The ABI-2 symbol `n4m_stats_factor_association_exact` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` with the same input contract as `n4m_preflight_factor_association`, and wrapped in JS/WASM by `statsFactorAssociationExact`. Brute-force SciPy/NumPy reference fractions are pinned in `cpp/tests/test_preflight.cpp`.

### Sources and provenance

https://doi.org/10.1007/b138696; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_preflight.cpp

## Catalog note

Exact permutation distribution of the factor-target association statistic (eta^2 or Cramer's V^2) by enumerating the distinct arrangements of the smaller side: exact p-value, minimal attainable exact p-value and maximum statistic.


_See also_: [methods index](index.md).
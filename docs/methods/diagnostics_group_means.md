# `group_means` — n4m.stats.group_means

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.group_means` · _Catalog id_: `diagnostics.group_means`

## API surface

**C ABI (ABI 2):** [`n4m_stats_group_means`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L374). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Arithmetic mean of replicate measurements, summed with the error-free TwoSum transformation of Møller (1965), *Quasi double-precision in floating point addition*, BIT 5, 37–50, https://doi.org/10.1007/BF01975722.

### Mathematical principle

For each group $g$ and column $j$ the kernel returns $\bar x_{gj}=\frac1{n_g}\sum_{i:g(i)=g}x_{ij}$, with equal weight per row. The sum is a double-double (TwoSum) accumulation divided once by $n_g$ with a fused remainder, so the mean is correct to about one rounding.

### Appropriate uses

Aggregating homogeneous technical repeats of one physical sample (same instrument and campaign) into one row before fitting or validation, so repeats never weigh as independent samples.

### Limits and validation

Declared numerical domain: finite values, every nonzero $|x|$ in $[2^{-256}, 2^{256}]$ and a spread $\max|x|/\min_{x\ne0}|x|\le 2^{64}$; outside it the kernel returns `N4M_ERR_UNSUPPORTED`. Every group needs at least one row; nonfinite values are refused (no mask). The mean is not a robust location estimate and does not reconcile repeats from different contexts.

### Implementation

The ABI-2 symbol `n4m_stats_group_means` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsGroupMeans`. `cpp/tests/test_group_means.cpp` checks exact dyadic means, a $2^{52}$ offset, scale sweeps, both sides of the domain boundaries and refusals.

### Sources and provenance

https://doi.org/10.1007/BF01975722; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_group_means.cpp

## Catalog note

Equal-weight per-group column means (double-double sums divided once), used to aggregate homogeneous technical repeats before fitting. Declared numerical domain D (nonzero |x| in [2^-256, 2^256], spread <= 2^64), refused outside. Exact dyadic, sweep and boundary checks in cpp/tests/test_group_means.cpp.


_See also_: [methods index](index.md).
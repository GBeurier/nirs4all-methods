# `correlation_test` — n4m.metrics.statistics.correlation_test

_Namespace_: **`n4m.metrics.statistics`** · _Fully-qualified_: `n4m.metrics.statistics.correlation_test` · _Catalog id_: `diagnostics.correlation_test`

## API surface

**C ABI (ABI 2):** [`n4m_stats_correlation_exact`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L288) · [`n4m_stats_correlation_test`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L263). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Pearson (1895), *Note on regression and inheritance in the case of two parents*, Proceedings of the Royal Society of London 58, 240–242, https://doi.org/10.1098/rspl.1895.0041; Spearman (1904), *The proof and measurement of association between two things*, American Journal of Psychology 15(1), 72–101, https://doi.org/10.2307/1412159; Phipson & Smyth (2010), *Permutation P-values Should Never Be Zero*, Statistical Applications in Genetics and Molecular Biology 9(1), https://doi.org/10.2202/1544-6115.1585.

### Mathematical principle

Pearson $r=\sum_i(x_i-\bar x)(y_i-\bar y)/\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}$. Spearman $\rho$ is Pearson's $r$ on average ranks (tied values share the mean of their ranks). The two-sided Monte Carlo test permutes $y$ over all units with $B$ seeded draws and reports $(1+b)/(1+B)$, where $b$ counts permuted $|r|\ge|r_{obs}|-\sqrt{\epsilon}$. The log10 bound $\min(n!/\prod m_x!,\,n!/\prod m_y!)$ limits the distinct outcomes.

### Appropriate uses

Testing a monotone or linear association between a continuous covariate (for example a measurement temperature) and a continuous target over exchangeable units, as in the Preflight metadata-target association protocol.

### Limits and validation

The test assumes exchangeable units and permutes $y$ as one unrestricted block; restricted or residual permutations need another protocol. The p-value resolution is bounded by $1/(B+1)$ and by the number of distinct outcomes. A constant vector gives NaN and runs no permutation. Work above 250 million operations returns `N4M_ERR_UNSUPPORTED`. Correlation is not causation.

### Implementation

The ABI-2 symbols `n4m_stats_correlation_test` and `n4m_stats_correlation_exact` (exhaustive enumeration of distinct arrangements: exact p-value and minimal attainable p-value) in `n4m/metrics.h` are implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsCorrelationTest` and `statsCorrelationExact`. Centering uses an exact power-of-two scaling and a shifted origin, so $r$ is translation invariant. Draws reuse the PCG64 engine seeded like `numpy.random.default_rng`. SciPy `pearsonr`/`spearmanr` coefficients and NumPy permutation counts are pinned in `cpp/tests/test_preflight.cpp`.

### Sources and provenance

https://doi.org/10.1098/rspl.1895.0041; https://doi.org/10.2307/1412159; https://doi.org/10.2202/1544-6115.1585; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_preflight.cpp

## Catalog note

Pearson or Spearman (average-rank) correlation of paired vectors with a seeded two-sided Monte Carlo permutation p-value (1+b)/(1+B), PCG64 draws equal to numpy.random.default_rng(seed).permutation(n); SciPy pearsonr/spearmanr parity; n4m_stats_correlation_exact enumerates the distinct arrangements for the exact p-value and the minimal attainable p-value.


_See also_: [methods index](index.md).
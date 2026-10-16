# `one_way_reml` — n4m.stats.variance_components.one_way_reml

_Namespace_: **`n4m.stats.variance_components`** · _Fully-qualified_: `n4m.stats.variance_components.one_way_reml` · _Catalog id_: `diagnostics.variance_components`

## API surface

**C ABI (ABI 2):** [`n4m_stats_variance_components_intervals`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L697) · [`n4m_stats_variance_components_reml`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L668). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Patterson & Thompson (1971), *Recovery of inter-block information when block sizes are unequal*, Biometrika 58(3), 545–554, https://doi.org/10.1093/biomet/58.3.545; Harville (1977), *Maximum Likelihood Approaches to Variance Component Estimation and to Related Problems*, JASA 72(358), 320–338, https://doi.org/10.1080/01621459.1977.10480998; Searle (1971), *Linear Models*, Wiley, for the exact balanced ICC interval.

### Mathematical principle

For $x_{ij}=\mu+u_i+e_{ij}$ with $u_i\sim N(0,\sigma_b^2)$ and $e_{ij}\sim N(0,\sigma_w^2)$, REML maximises the restricted likelihood. With $\gamma=\sigma_b^2/\sigma_w^2$ and $a_i=\gamma+1/r_i$, $\sigma_w^2$ profiles out as $(SSW+Q(\gamma))/(N-1)$, $Q$ the $1/a_i$-weighted sum of squares of the group means. The one-dimensional criterion is gridded on $\gamma$ and its derivative bisected; $\gamma=0$ is kept when the derivative there is non-negative. ICC(1) is $\sigma_b^2/(\sigma_b^2+\sigma_w^2)$. Intervals: $\sqrt{SSW/\chi^2}$ on $N-g$ degrees of freedom for the within SD; for balanced groups, $F_L=F/F_{1-\alpha/2}(g-1,N-g)$, $F_U=F\,F_{1-\alpha/2}(N-g,g-1)$ and $(F_*-1)/(F_*+r-1)$; otherwise percentiles of a cluster bootstrap.

### Appropriate uses

Repeatability of spectral bands from technical replicates when samples carry unequal numbers of acquisitions, including unreplicated samples.

### Limits and validation

The model assumes normal, homoscedastic within-group errors and independent groups; the chi-square interval of the within SD needs only the first two. The balanced exact interval is not truncated to [0, 1]. Bootstrap percentiles are unreliable with few resampling units; replicates without within-group degrees of freedom are counted as invalid. Boundary estimates ($\sigma_b^2=0$ or $\sigma_w^2=0$) are flagged per feature. Measured percentile coverage at 95 % nominal is 0.86-0.94 for 20 to 60 groups (`parity/generators/variance_components_coverage.py`); the intervals are exploratory. With K distinct group sizes the score has the sign of a polynomial of degree at most 4K-3, so balanced designs are unimodal; for unbalanced designs a likelihood maximum lying inside one grid cell (1/8 decade of gamma) can be missed. Supported domain: nonzero |x| in [2^-256, 2^256] with spread <= 2^32; outside it the call is refused.

### Implementation

The ABI-2 symbols `n4m_stats_variance_components_reml` and `n4m_stats_variance_components_intervals` in `n4m/metrics.h` are implemented in `cpp/src/c_api/c_api_variance_components.cpp` and `cpp/src/core/variance_components.cpp`, with chi-square and F quantiles in `cpp/src/core/common/distributions.cpp`. Bootstrap draws follow `numpy.random.default_rng(seed).integers`. `cpp/tests/test_variance_components.cpp` pins dense-REML, statsmodels 0.15.0 and SciPy 1.18.1 references. JS/WASM wraps them as `statsVarianceComponentsReml` and `statsVarianceComponentsIntervals`.

### Sources and provenance

https://doi.org/10.1093/biomet/58.3.545; https://doi.org/10.1080/01621459.1977.10480998; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/variance_components.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_variance_components.cpp

## Catalog note

One-way random-effects variance components per feature: REML for balanced and unbalanced groups with explicit boundary and convergence status, exact chi-square interval of the within-group SD, exact F interval of the balanced ICC(1,1) (Searle), and seeded cluster-bootstrap percentile intervals (NumPy-parity PCG64 draws). Dense REML score-equation, statsmodels 0.15.0 MixedLM, closed-form and SciPy quantile references are pinned in cpp/tests/test_variance_components.cpp; JS/WASM wrappers in bindings/js/src/preflightDiagnostics.ts.


_See also_: [methods index](index.md).
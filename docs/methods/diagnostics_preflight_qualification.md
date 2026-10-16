# `qualification` — n4m.metrics.preflight.qualification

_Namespace_: **`n4m.metrics.preflight`** · _Fully-qualified_: `n4m.metrics.preflight.qualification` · _Catalog id_: `diagnostics.preflight_qualification`

## API surface

**C ABI (ABI 2):** [`n4m_preflight_factor_association`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L407) · [`n4m_preflight_local_contrast`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L137) · [`n4m_preflight_nearest_neighbors`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L109) · [`n4m_preflight_noise_proxy_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L307) · [`n4m_preflight_pairs_within_tolerance`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L353) · [`n4m_preflight_pca_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L52) · [`n4m_preflight_pca_compute_resolved`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L23) · [`n4m_preflight_repeatability_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L79) · [`n4m_preflight_snv_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L329) · [`n4m_preflight_summary_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L95). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

This kernel bundle has no single canonical paper. The noise proxy smooths with the Savitzky–Golay filter (Savitzky & Golay, 1964, *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*, https://doi.org/10.1021/ac60214a047). Repeatability reports the one-way random-effects ICC(1,1) (Shrout & Fleiss, 1979, *Intraclass correlations: uses in assessing rater reliability*, https://doi.org/10.1037/0033-2909.86.2.420). Factor association uses eta² with the one-way ANOVA F statistic, or Cramér's V with the Pearson chi-square statistic.

### Mathematical principle

Eight caller-buffer kernels. PCA returns centered scores, loadings and $k$ eigenvalues (ddof=1) plus the exact full-spectrum moments $\sum\lambda$ and $\sum\lambda^2$. Repeatability returns, per feature, the pooled within-group SD and, for balanced groups of $r$ repeats, $ICC(1,1)=(MS_B-MS_W)/(MS_B+(r-1)MS_W)$. The summary returns count, min, max, mean, sample SD and type-7 quartiles. Nearest neighbors are Euclidean and can exclude rows of the same group. Local contrast compares target disagreement among neighbors with the disagreement expected over all admissible rows. The noise proxy is the RMS of $x-\mathrm{SG}(x)$ over interior features divided by the white-noise residual gain $g=\sqrt{\sum_k(\delta_k-w_k)^2}$. Tolerance pairs list row pairs whose RMS feature difference is at most a tolerance. Factor association returns eta² and F, or Cramér's V and chi-square, with an optional seeded permutation p-value $(1+b)/(1+B)$.

### Appropriate uses

Qualifying a spectral dataset before modelling: spectral rank and dominance, replicate repeatability, local target structure, a smoothing-based noise level, near-duplicate rows, and association between metadata factors and the target.

### Limits and validation

These are descriptive diagnostics, not model-validation scores. The noise proxy assumes equally spaced features and a locally polynomial signal; it is not an SNR. ICC is computed only for balanced complete groups and is NaN otherwise. The association test assumes exchangeable units and permutes targets as one unrestricted block. PCA moments and permutation tests above 250 million operations return `N4M_ERR_UNSUPPORTED`; neighbor and pair searches are exhaustive and quadratic in rows.

### Implementation

The ABI-2 symbols `n4m_preflight_{pca_compute,repeatability_compute,summary_compute,nearest_neighbors,local_contrast,noise_proxy_compute,pairs_within_tolerance,factor_association}` in `n4m/metrics.h` are implemented in `cpp/src/c_api/c_api_preflight.cpp`. They reuse the Methods PCA helper, the Savitzky–Golay engine and the PCG64 generator, seeded like `numpy.random.default_rng`. The JS/WASM binding wraps them in `bindings/js/src/preflightDiagnostics.ts`. NumPy/SciPy reference values are pinned in `cpp/tests/test_preflight.cpp`.

### Sources and provenance

https://doi.org/10.1021/ac60214a047; https://doi.org/10.1037/0033-2909.86.2.420; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_preflight.cpp

## Catalog note

Caller-buffer dataset-qualification kernels for Preflight (ABI 2.18: PCA spectral moments, repeatability/ICC(1,1), finite summary, group-excluding nearest neighbors, local target contrast; ABI 2.19: Savitzky-Golay residual noise proxy, tolerance pairs, factor-target association with seeded permutation test; ABI 2.20: SNV with per-row rounding bounds, PCA with its absolute resolution bound and per-component unresolved flags). Native contract tests in cpp/tests/test_preflight.cpp; JS/WASM wrappers in bindings/js/src/preflightDiagnostics.ts.


_See also_: [methods index](index.md).
# `regression_scores` — n4m.stats.regression_scores

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.regression_scores` · _Catalog id_: `diagnostics.regression_scores`

## API surface

**C ABI (ABI 2):** [`n4m_stats_regression_scores`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L827). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Standard regression scores, summed with the error-free TwoSum transformation of Møller (1965), *Quasi double-precision in floating point addition*, BIT 5, 37–50, https://doi.org/10.1007/BF01975722.

### Mathematical principle

With residuals $e_i=y_i-\hat y_i$, $\mathrm{RMSE}=\sqrt{\sum e_i^2/n}$, $\mathrm{MAE}=\sum|e_i|/n$ and $R^2=1-\sum e_i^2/\sum (y_i-\bar y)^2$. The mean $\bar y$ is a double-double value and every deviation is explicit; all sums are double-double, so no large nearly-equal accumulations are subtracted.

### Appropriate uses

Scoring out-of-fold predictions and reference predictors (for example a fold-train mean) on identical rows.

### Limits and validation

Declared numerical domain: finite values, every nonzero $|x|$ in $[2^{-256}, 2^{256}]$ and a spread $\max|x|/\min_{x\ne0}|x|\le 2^{32}$ per vector; outside it the kernel returns `N4M_ERR_UNSUPPORTED`. RMSE and MAE have relative errors of order $n\varepsilon$, $R^2$ an absolute error of order $n\varepsilon(1+SSE/SST)$. $R^2$ is published as a significand times a power of two, so a ratio $SSE/SST\ge 2^{999}$ (possible inside the domain) is reported scaled rather than as $-\infty$; it is NaN (undefined) when $y$ is constant.

### Implementation

The ABI-2 symbol `n4m_stats_regression_scores` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsRegressionScores`. `cpp/tests/test_group_means.cpp` checks exact dyadic scores, scale sweeps, a $2^{52}$ offset, an exact one-ulp case and the domain boundaries.

### Sources and provenance

https://doi.org/10.1007/BF01975722; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_group_means.cpp

## Catalog note

RMSE, MAE and R2 from explicit residuals and deviations around a double-double mean, summed in double-double. Declared numerical domain (nonzero |x| in [2^-256, 2^256], spread <= 2^32), refused outside; R2 NaN when y is constant. Exact dyadic, sweep, offset and boundary checks in cpp/tests/test_group_means.cpp.


_See also_: [methods index](index.md).
# `row_l2_normalize` — n4m.transform.scaling.row_l2_normalize

_Namespace_: **`n4m.transform.scaling`** · _Fully-qualified_: `n4m.transform.scaling.row_l2_normalize` · _Catalog id_: `preprocessing.scaling.row_l2_normalize`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import RowL2Normalize`

**Signature:** [`RowL2Normalize()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L3970)

**R (source-verified):** [`n4m_row_l2_normalize()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_row_l2_normalize()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper is asserted for this elementary Euclidean normalization. The implementation-specific finite-input, zero-row and portable-width-state contract is defined by the native source and analytic contract tests linked below.

### Mathematical principle

For each row x, return x/sqrt(sum_j x_j^2) when the row is nonzero; preserve a zero row unchanged. Compute s=max_j abs(x_j), then u_j=x_j/s and return u_j/sqrt(sum_j u_j^2). The scaled calculation avoids forming an overflowing or underflowing unscaled squared norm. Every row is independent of other batch rows and of the rows used for fitting.

### Appropriate uses

Normalizing finite spectra to unit Euclidean length before similarity comparisons or native role pipelines when absolute spectral magnitude should be removed.

### Limits and validation

Only L2 normalization is supported; there are no order, L1 or axis parameters. Fit and transform refuse NaN and infinity rather than imputing or propagating them. Zero rows stay zero and have no unit direction. Fit retains input width only and transform requires that width; empty dimensions follow the shared native estimator boundary. Large and small finite rows require scaled arithmetic rather than naive sum-of-squares.

### Implementation

The native preprocessing.scaling.row_l2_normalize transformer is exposed as RowL2Normalize through generated Python and JS adapters and as n4m_row_l2_normalize in R. Its kernel accepts explicit input and output strides. The shared n4m_estimator_* lifecycle and WTD1 width state supply fit, transform and N4ME replay without new C symbols. Available since project 1.3.16; ABI 2.22.0 is unchanged. Existing Normalize remains a distinct column-wise operation.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/row_l2_normalize.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_estimator.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/js/test/run_row_l2_normalize.mjs

## Catalog note

Row-wise Euclidean normalization through the shared n4m_estimator_* role API, available since project 1.3.16 with ABI 2.22.0 unchanged. No parameters or dedicated C exports. Fit records input width only; each nonzero row is divided by its own L2 norm using scaled arithmetic. Zero rows are unchanged. Fit and transform refuse NaN and infinity. Distinct from the existing column-wise Normalize; no host numerical fallback.


_See also_: [methods index](index.md).
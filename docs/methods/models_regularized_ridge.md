# `ridge` — n4m.estimators.regression.regularized.ridge

_Namespace_: **`n4m.estimators.regression.regularized`** · _Fully-qualified_: `n4m.estimators.regression.regularized.ridge` · _Catalog id_: `models.regularized.ridge`

## API surface

**C ABI (ABI 2):** [`n4m_estimators_ridge_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L155). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.estimators.regression.regularized import ridge`

**Signature:** [`ridge(X, y, *, alpha: float = 1.0, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8537)

**R (source-verified):** [`ridge_fit(X, Y, n_components = 1L, ridge_lambda = 1.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

```r
library(n4m)
result <- ridge_fit(X, Y)
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `alpha` | `float` | `1.0` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634.

### Mathematical principle

Estimate $\hat{\beta}_\lambda=(X^\top X+\lambda I)^{-1}X^\top y$. The positive L2 penalty makes the normal equations invertible under collinearity and shrinks unstable coefficients toward zero without selecting features.

### Appropriate uses

A linear calibration baseline for strongly collinear spectra or as the head of a strict-linear AOM route.

### Limits and validation

The penalty changes with feature scale, so scaling belongs inside the fitting/CV protocol. Ridge retains all variables and its selected $\lambda$ requires outer validation for an unbiased performance estimate.

### Implementation

`n4m.estimators.regression.regularized.ridge` and `RidgeRegressor`; C ABI `n4m_estimators_ridge_fit`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ridge.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/estimators/regression/regularized.py

## Catalog note

Direct closed-form Ridge regression head with primal/dual handling in the native core. This is the reusable linear head used by the moment and AOM Ridge screens.

_Timing benchmark_: `benchmarks/cross_binding/bench_direct_moment_heads_timing.py`


_See also_: [methods index](index.md).